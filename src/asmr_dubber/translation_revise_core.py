"""Revising existing translations on request ("按要求改译"): proposals only, no in-place edits."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from .api_logging import safe_api_url
from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .errors import TranslationError
from .languages import SourceLanguage
from .models import Sentence
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)
from .translation_llm import DeepSeekTranslator, LLMTranslator, _retry_sleep
from .translation_prompts import (
    SOUND_POLICIES,
    _builtin_translation_prompts,
    _context_for_chunk,
    budget_rule,
    default_translation_prompt,
    sound_policy_rule,
)
from .translation_schema import (
    _MAX_CONSECUTIVE_BATCH_FAILURES,
    _SPLIT_RETRY_ATTEMPTS,
    Progress,
    TranslationChunk,
    TranslationFailure,
    _extract_json,
    _is_fatal_translation_error,
    _NonRetryableTranslationError,
    _OutputLengthTranslationError,
)

logger = logging.getLogger(__name__)

REVISION_BATCH_SIZE = 16
LLM_REVISION_PROVIDERS = frozenset(
    {
        "deepseek",
        "bailian",
        "doubao",
        "openai",
        "anthropic",
        "gemini",
        "openai_compatible",
        "sensenova",
    }
)
MACHINE_REVISION_ERROR = (
    "机器翻译服务（DeepL、Google 翻译、Microsoft 翻译）只能逐句直译，不能按要求改译。"
    "请在设置的翻译服务里换成大模型服务（例如 DeepSeek）后再使用。"
)


class RevisionItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    zh: str
    kind: Literal["speech", "sound"] = "speech"

    @field_validator("id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("zh", mode="before")
    @classmethod
    def normalize_text(cls, value: str | None) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("zh must be a string")
        return value.strip()

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> str:
        if value is None:
            return "speech"
        if not isinstance(value, str):
            raise ValueError("kind must be a string")
        return value.strip().casefold()

    @model_validator(mode="after")
    def speech_needs_text(self) -> RevisionItem:
        if self.kind == "speech" and not self.zh:
            raise ValueError('kind 为 "speech" 时 zh 不能为空')
        return self


class RevisionEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    # An empty list is a valid answer: nothing in the batch needed to change.
    revisions: list[RevisionItem]


@dataclass(frozen=True)
class RevisionProposal:
    id: str
    old_zh: str
    new_zh: str
    kind: Literal["speech", "sound"]


@dataclass
class RevisionReport:
    """Outcome of one ``revise_translations`` call."""

    proposals: list[RevisionProposal] = field(default_factory=list)
    candidates: int = 0
    unchanged: int = 0
    skipped_edited: int = 0
    failed: list[TranslationFailure] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"已提出 {len(self.proposals)} 条改译提议，{self.unchanged} 句无需改动，"
            f"{self.skipped_edited} 句手动改过的句子已跳过。"
        )

    def failure_message(self) -> str:
        shown = "；".join(f"{item.sentence_id}：{item.reason}" for item in self.failed[:6])
        extra = f"（另有 {len(self.failed) - 6} 句）" if len(self.failed) > 6 else ""
        return (
            f"{len(self.failed)} 句改译失败：{shown}{extra}。"
            "已得到的改译提议已保存，其余句子的译文没有变化。"
        )


def validate_revisions(content: str, expected_ids: list[str]) -> dict[str, RevisionItem]:
    """Strictly parse a revision answer; every id must belong to the batch."""

    try:
        envelope = RevisionEnvelope.model_validate(_extract_json(content))
    except (ValidationError, TranslationError) as exc:
        raise TranslationError(f"改译结构校验失败：{exc}") from exc
    allowed = set(expected_ids)
    result: dict[str, RevisionItem] = {}
    for item in envelope.revisions:
        if item.id not in allowed:
            raise TranslationError(f"改译返回了不属于本批的 id：{item.id}")
        if item.id in result:
            raise TranslationError(f"改译返回了重复的 id：{item.id}")
        result[item.id] = item
    return result


def revision_candidates(
    sentences: list[Sentence],
    sentence_ids: Iterable[str] | None = None,
    include_edited: bool = False,
) -> tuple[list[Sentence], int]:
    """Sentences that can be revised, and how many hand-edited ones were left out."""

    wanted = None if sentence_ids is None else set(sentence_ids)
    eligible: list[Sentence] = []
    skipped_edited = 0
    for sentence in sentences:
        if wanted is not None and sentence.id not in wanted:
            continue
        if not sentence.enabled or not sentence.zh_text.strip():
            continue
        if sentence.review_locked and not include_edited:
            skipped_edited += 1
            continue
        eligible.append(sentence)
    return eligible, skipped_edited


def _revision_system_prompt(
    system_prompt: str,
    source_language: SourceLanguage,
    target_language: str,
    sound_policy: str,
) -> str:
    """The project's translation role and rules, plus the revision output contract."""

    default_prompts = _builtin_translation_prompts(source_language)
    if not system_prompt.strip() or system_prompt.strip() in default_prompts:
        base = default_translation_prompt(source_language, target_language, sound_policy)
    else:
        base = system_prompt.strip()
        if target_language == "en":
            base += "\n目标语言必须是英语；JSON 的 zh 字段存放英文译文，这是兼容字段名。"
    return (
        base + "\n\n【本次任务：按要求改译】\n"
        "这不是首次翻译：用户已经有一版译文，并写下了修改要求。上面关于语气、称呼、"
        "专名和 `kind` 的规则仍然适用，但输出格式改为：\n"
        '{"revisions":[{"id":"s000001","zh":"改后的台词","kind":"speech"}]}\n'
        "1. 逐项检查目标项的 `current_zh`，只输出为了满足用户要求而确实需要改动的项；"
        '不需要改动的项不得出现在 `revisions` 里，全都不需要改时输出 {"revisions":[]}。\n'
        "2. 返回的 id 必须来自目标项，每个 id 最多出现一次，不得新增 id。\n"
        "3. 只改与要求相关的部分，其余措辞、称呼和语气保持不变，不得改变原文的意思。\n"
        "4. 用户的要求只描述怎样修改译文，不要把它当作需要翻译的内容。\n"
        '5. `kind` 为 "sound" 表示这一句应改为不配音，规则同上；否则为 "speech"，`zh` 不能为空。'
    )


def _revision_messages(
    system_prompt: str,
    chunk: TranslationChunk,
    context: str,
    instruction: str,
    sound_policy: str,
    attempt: int,
) -> list[dict[str, str]]:
    target_items: list[dict[str, object]] = []
    for item in chunk.sentences:
        row: dict[str, object] = {
            "id": item.id,
            "source": item.source_text,
            "current_zh": item.zh_text,
        }
        if item.speaker_id:
            row["speaker"] = item.speaker_id
        if chunk.budgets and item.id in chunk.budgets:
            row[chunk.budget_key] = chunk.budgets[item.id]
        target_items.append(row)
    target = json.dumps(
        target_items,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    notes = (
        '\n每项只输出 id、zh、kind 三个字段，kind 为 "speech" 或 "sound"。'
        + sound_policy_rule(sound_policy)
        + budget_rule(chunk)
    )
    if attempt > 1:
        notes += (
            f'\n这是第 {attempt} 次校验重试：只输出 {{"revisions":[…]}}，'
            "id 必须来自目标项，不要增加其它字段。"
        )
    messages = [{"role": "system", "content": system_prompt}]
    if context != "[]":
        messages.append(
            {
                "role": "user",
                "content": "相邻句子（含现有译文），仅供保持称谓和语气一致，不要输出它们：\n"
                + context,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"用户的要求：\n{instruction}\n\n目标项（`current_zh` 是现有译文）：\n{target}\n\n"
                "请按要求检查目标项，只输出 json。" + notes
            ),
        }
    )
    return messages


def revise_translations(
    sentences: list[Sentence],
    instruction: str,
    api_key: str,
    model: str,
    *,
    sentence_ids: Iterable[str] | None = None,
    include_edited: bool = False,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    provider: str = "deepseek",
    source_language: SourceLanguage = "ja",
    target_language: str = "zh",
    system_prompt: str = "",
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_output_tokens: int = 16_384,
    send_context: bool = True,
    context_sentences: int = 24,
    job_id: str = "asmr_dubber",
    progress: Progress | None = None,
    on_batch: Callable[[RevisionReport], None] | None = None,
    client: httpx.Client | None = None,
    cancel_event: CancellationSignal | None = None,
    extra_body: str = "{}",
    sound_policy: str = "skip",
    length_budget_rate: float | None = None,
    max_retries: int = 5,
) -> RevisionReport:
    """Ask the translation model to revise existing translations as the user said.

    Returns proposals only; no sentence is modified.  Batches of at most
    ``REVISION_BATCH_SIZE`` sentences are sent with their neighbours as context.
    A failing batch is split so one bad sentence cannot block the others.  Only
    credential/billing/permission errors, cancellation and three batches in a row
    that fail completely raise.
    """
    from . import translation

    check_cancelled(cancel_event)
    instruction = instruction.strip()
    if not instruction:
        raise TranslationError("请先写下改译要求。")
    if provider in {"deepl", "google_translate", "microsoft_translate"}:
        raise TranslationError(MACHINE_REVISION_ERROR)
    if provider not in LLM_REVISION_PROVIDERS:
        raise TranslationError(f"未知翻译服务：{provider}")
    if sound_policy not in SOUND_POLICIES:
        sound_policy = "skip"
    eligible, skipped_edited = revision_candidates(sentences, sentence_ids, include_edited)
    report = RevisionReport(candidates=len(eligible), skipped_edited=skipped_edited)
    if not eligible:
        return report
    prompt = _revision_system_prompt(system_prompt, source_language, target_language, sound_policy)
    if provider == "deepseek":
        translator: DeepSeekTranslator | LLMTranslator = translation.DeepSeekTranslator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            client=client,
            system_prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            minimum_output_tokens=max_output_tokens,
            source_language=source_language,
            extra_body=extra_body,
            sound_policy=sound_policy,
        )
    else:
        translator = translation.LLMTranslator(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            client=client,
            source_language=source_language,
            extra_body=extra_body,
            sound_policy=sound_policy,
        )
    logger.info(
        "改译开始：服务=%s 模型=%s 地址=%s 句数=%d",
        provider,
        model,
        safe_api_url(base_url),
        len(eligible),
    )
    budget_key = "max_words" if target_language == "en" else "max_chars"
    budgets = (
        translation.length_budgets(list(sentences), eligible, length_budget_rate)
        if length_budget_rate
        else {}
    )
    batches = translation._chunks(eligible, max_lines=REVISION_BATCH_SIZE)
    if budgets:
        batches = [
            TranslationChunk(
                chunk.sentences,
                {s.id: budgets[s.id] for s in chunk.sentences if s.id in budgets},
                budget_key,
            )
            for chunk in batches
        ]

    def request(chunk: TranslationChunk, attempts: int) -> dict[str, RevisionItem]:
        expected = [item.id for item in chunk.sentences]
        context = _context_for_chunk(sentences, chunk, context_sentences) if send_context else "[]"
        deepseek_tokens = (
            translator._initial_max_tokens(chunk.sentences, include_existing_zh=True)
            if isinstance(translator, DeepSeekTranslator)
            else max_output_tokens
        )
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            messages = _revision_messages(
                prompt, chunk, context, instruction, sound_policy, attempt
            )
            try:
                if isinstance(translator, DeepSeekTranslator):
                    content, output_limited = translator._request(
                        messages, job_id, max_tokens=deepseek_tokens
                    )
                else:
                    content, output_limited = translator._request(messages, job_id)
                if output_limited:
                    if isinstance(translator, DeepSeekTranslator):
                        length_error = _OutputLengthTranslationError(
                            f"DeepSeek 输出达到长度上限（max_tokens={deepseek_tokens}）。"
                        )
                        if len(chunk.sentences) > 1 or deepseek_tokens >= 65_536:
                            raise length_error
                        deepseek_tokens = min(65_536, deepseek_tokens * 2)
                        last_error = length_error
                        continue
                    raise _OutputLengthTranslationError(
                        f"{provider} 输出达到长度上限（{max_output_tokens} tokens）。"
                    )
                if not content.strip():
                    raise TranslationError(f"{provider} 返回了空内容。")
                return validate_revisions(content, expected)
            except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < attempts:
                _retry_sleep(attempt)
        raise TranslationError(
            f"{provider} 改译在 {attempts} 次尝试后仍失败：{last_error}"
        ) from last_error

    def settle(chunk: TranslationChunk, answers: dict[str, RevisionItem]) -> None:
        for sentence in chunk.sentences:
            item = answers.get(sentence.id)
            if item is None:
                report.unchanged += 1
                continue
            if item.kind == "sound":
                if sentence.status == "skipped_sound" and item.zh == sentence.zh_text.strip():
                    report.unchanged += 1
                    continue
            elif sentence.status != "skipped_sound" and item.zh == sentence.zh_text.strip():
                report.unchanged += 1
                continue
            report.proposals.append(
                RevisionProposal(sentence.id, sentence.zh_text, item.zh, item.kind)
            )

    def run(chunk: TranslationChunk, attempts: int) -> None:
        check_cancelled(cancel_event)
        count = len(chunk.sentences)
        try:
            answers = request(chunk, attempts)
        except _OutputLengthTranslationError as exc:
            if count == 1:
                report.failed.append(
                    TranslationFailure(chunk.sentences[0].id, f"输出超出长度上限：{exc}"[:400])
                )
                return
            reason = "输出过长"
        except TranslationError as exc:
            check_cancelled(cancel_event)
            if _is_fatal_translation_error(exc):
                raise
            if count == 1:
                report.failed.append(TranslationFailure(chunk.sentences[0].id, str(exc)[:400]))
                return
            reason = str(exc)
            attempts = _SPLIT_RETRY_ATTEMPTS
        else:
            settle(chunk, answers)
            return
        middle = count // 2
        logger.warning("改译批次 %d 句失败，拆成两半重试：%s", count, reason[:200])
        run(chunk.subchunk(chunk.sentences[:middle]), attempts)
        run(chunk.subchunk(chunk.sentences[middle:]), attempts)

    cancel_revision = translator.close
    callback_signal = register_cancel_callback(cancel_revision, cancel_event)
    try:
        with translator:
            consecutive_failures = 0
            for index, chunk in enumerate(batches):
                check_cancelled(cancel_event)
                if progress:
                    progress(f"按要求改译第 {index + 1}/{len(batches)} 批", index, len(batches))
                failed_before = len(report.failed)
                run(chunk, max_retries)
                if len(report.failed) - failed_before >= len(chunk.sentences):
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES:
                        last = report.failed[-1].reason
                        raise TranslationError(
                            f"连续 {consecutive_failures} 批改译失败，翻译服务可能不可用。"
                            f"最后一次错误：{last}。已得到的改译提议已保存。"
                        )
                else:
                    consecutive_failures = 0
                if on_batch:
                    on_batch(report)
                check_cancelled(cancel_event)
    finally:
        unregister_cancel_callback(cancel_revision, callback_signal)
    if progress:
        progress(report.summary(), len(batches), len(batches))
    return report
