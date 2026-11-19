"""Unified sentence translation loop, translator factory, and public facade."""

from __future__ import annotations

import logging
import time  # noqa: F401
from collections.abc import Callable, Iterable

import httpx

from .api_logging import logged_http_client, safe_api_url  # noqa: F401
from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .errors import TranslationError
from .filtering import looks_like_sound_effect
from .languages import SourceLanguage
from .models import Sentence
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)
from .translation_checks import (  # noqa: F401
    _ANY_KANA,
    _HANZI_CHAR,
    _KANA,
    _SHARED_HANZI_ASCII_KEY,
    LENGTH_OVERRUN_RATIO,
    _allow_same_as_source,
    _chunks,
    _same_text_key,
    _used_length,
    check_translations,
    length_budgets,
)
from .translation_llm import DeepSeekTranslator, LLMTranslator
from .translation_llm_formats import (  # noqa: F401
    OPENAI_COMPATIBLE_PROVIDERS,
    _openai_provider_controls,
)
from .translation_machine import (  # noqa: F401
    MACHINE_TRANSLATION_PROVIDERS,
    DeepLTranslator,
    GoogleTranslator,
    MachineTranslationAPI,
    MicrosoftTranslator,
)
from .translation_prompts import (  # noqa: F401
    _BUDGET_RULE,
    _DEEPSEEK_STRUCTURE_PROMPT,
    _SCRIPT_RECONCILIATION_PROMPT,
    _SOUND_POLICY_RULES,
    _SOURCE_PROMPT_VALUES,
    _SYSTEM_PROMPT_TEMPLATE,
    SOUND_POLICIES,
    SYSTEM_PROMPT,
    SoundPolicy,
    _as_entry,
    _bounded_translation_memory,
    _builtin_translation_prompts,
    _clear_audio,
    _context_for_chunk,
    _deepseek_translation_request,
    _json_lines,
    _load_prompt,
    _looks_like_romaji,
    _mark_filler,
    _mark_sound,
    _mark_translated,
    _resolve_system_prompt,
    _sound_text,
    _translation_memory,
    budget_rule,
    default_translation_prompt,
    empty_translation_is_expected,
    sound_policy_rule,
)
from .translation_revise_core import (  # noqa: F401
    LLM_REVISION_PROVIDERS,
    MACHINE_REVISION_ERROR,
    REVISION_BATCH_SIZE,
    RevisionEnvelope,
    RevisionItem,
    RevisionProposal,
    RevisionReport,
    _revision_messages,
    _revision_system_prompt,
    revise_translations,
    revision_candidates,
    validate_revisions,
)
from .translation_schema import (  # noqa: F401
    _FATAL_HTTP_STATUS,
    _MAX_CONSECUTIVE_BATCH_FAILURES,
    _SPLIT_RETRY_ATTEMPTS,
    _TRAILING_COMMA,
    Progress,
    TranslationChunk,
    TranslationEntry,
    TranslationEnvelope,
    TranslationFailure,
    TranslationFlag,
    TranslationItem,
    TranslationReport,
    _extract_json,
    _is_fatal_translation_error,
    _NonRetryableTranslationError,
    _OutputLengthTranslationError,
    _strip_trailing_commas,
    validate_translation,
    validate_translation_entries,
)
from .translation_script import (  # noqa: F401
    LLM_RECONCILIATION_PROVIDERS,
    ScriptReconciliationBatch,
    _recognized_script_batches,
    _script_fragments,
    _script_reconciliation_messages,
    reconcile_script_sentences,
)
from .translation_script_spans import (  # noqa: F401
    ScriptCorrection,
    ScriptCorrectionEnvelope,
    ScriptQuote,
    ScriptSpan,
    _validated_script_mapping,
    _validated_script_spans,
    validate_script_reconciliation,
)

logger = logging.getLogger(__name__)

_GENERAL_LLM_PROVIDERS = frozenset(
    {
        "bailian",
        "doubao",
        "openai",
        "anthropic",
        "gemini",
        "openai_compatible",
        "sensenova",
    }
)


def _build_translator(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    deepl_formality: str,
    microsoft_region: str,
    client: httpx.Client | None,
    source_language: SourceLanguage,
    target_language: str,
    extra_body: str,
    sound_policy: str,
) -> DeepSeekTranslator | LLMTranslator | MachineTranslationAPI:
    if provider == "deepseek":
        return DeepSeekTranslator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            client=client,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            minimum_output_tokens=max_output_tokens,
            source_language=source_language,
            extra_body=extra_body,
            sound_policy=sound_policy,
        )
    if provider in _GENERAL_LLM_PROVIDERS:
        return LLMTranslator(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            client=client,
            source_language=source_language,
            extra_body=extra_body,
            sound_policy=sound_policy,
        )
    if provider in MACHINE_TRANSLATION_PROVIDERS:
        return MachineTranslationAPI(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            deepl_formality=deepl_formality,
            microsoft_region=microsoft_region,
            client=client,
            source_language=source_language,
            target_language=target_language,
        )
    raise TranslationError(f"未知翻译服务：{provider}")


def translate_sentences(
    sentences: list[Sentence],
    api_key: str,
    model: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    provider: str = "deepseek",
    source_language: SourceLanguage = "ja",
    system_prompt: str = "",
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_output_tokens: int = 16_384,
    deepl_formality: str = "default",
    microsoft_region: str = "",
    send_context: bool = True,
    context_sentences: int = 24,
    memory_sentences: int = 50,
    job_id: str = "asmr_dubber",
    progress: Progress | None = None,
    on_batch: Callable[[], None] | None = None,
    client: httpx.Client | None = None,
    cancel_event: CancellationSignal | None = None,
    extra_body: str = "{}",
    target_language: str = "zh",
    sound_policy: str = "skip",
    retranslate: bool = False,
    overwrite_edited: bool = False,
    length_budget_rate: float | None = None,
    sentence_ids: Iterable[str] | None = None,
    compress: bool = False,
) -> TranslationReport:
    """Translate every pending sentence; one bad sentence never stops the others.

    ``sentence_ids`` limits the call to those sentences and redoes them even when
    they have text (like ``retranslate``), e.g. the ids in ``report.flagged_ids``.
    ``compress`` shortens existing over-budget translations using ``existing_zh``
    and keeps existing text intact if compression fails.

    Failures are collected in the returned report.  Only credential, billing and
    permission errors (HTTP 401/402/403), cancellation and three consecutive failed
    batches raise, and by then every finished translation has been checkpointed.

    ``retranslate`` also selects sentences that already have text and replaces them
    one by one, keeping the old text until the new text has arrived.  Rows the user
    edited (``review_locked``) are skipped unless ``overwrite_edited`` is set.
    """

    check_cancelled(cancel_event)
    if sound_policy not in SOUND_POLICIES:
        sound_policy = "skip"
    report = TranslationReport()
    selected = None if sentence_ids is None else set(sentence_ids)
    redo = retranslate or selected is not None
    budgets: dict[str, int] = {}
    budget_key = "max_words" if target_language == "en" else "max_chars"
    if compress:
        if provider in MACHINE_TRANSLATION_PROVIDERS:
            if progress:
                progress("当前机器翻译服务不支持自动压缩译文", 1, 1)
            return report
        if source_language == target_language or not length_budget_rate:
            if progress:
                progress("所有句子已符合时长预算", 1, 1)
            return report
        pool = [
            sentence
            for sentence in sentences
            if sentence.enabled
            and sentence.zh_text.strip()
            and sentence.status in {"translated", "synthesized", "error"}
            and (selected is None or sentence.id in selected)
            and (overwrite_edited or not sentence.review_locked)
        ]
        budgets = length_budgets(list(sentences), pool, length_budget_rate) if pool else {}
        if selected is None and pool:
            too_long = {
                flag.sentence_id
                for flag in check_translations(
                    sentences,
                    budgets,
                    budget_key,
                    only_ids=[sentence.id for sentence in pool],
                    source_language=source_language,
                    target_language=target_language,
                )
                if flag.code == "too_long"
            }
            pending = [sentence for sentence in pool if sentence.id in too_long]
        else:
            pending = pool
    else:
        pending = [
            sentence
            for sentence in sentences
            if sentence.enabled
            and (selected is None or sentence.id in selected)
            and (
                not sentence.zh_text
                or sentence.status == "translation_failed"
                or (redo and (overwrite_edited or not sentence.review_locked))
            )
        ]
    if not pending:
        if progress:
            progress(
                "所有句子已符合时长预算" if compress else "所有句子已有目标语言文本",
                1,
                1,
            )
        return report
    if source_language == target_language:
        for sentence in pending:
            _mark_translated(sentence, sentence.source_text)
        report.translated = len(pending)
        if on_batch:
            on_batch()
        return report
    system_prompt = _resolve_system_prompt(
        system_prompt, source_language, target_language, sound_policy
    )
    logger.info(
        "翻译后端开始：服务=%s 模型=%s 地址=%s 句数=%d 源语言=%s",
        provider,
        model,
        safe_api_url(base_url),
        len(pending),
        source_language,
    )

    def make_chunk(items: list[Sentence], *, compress_mode: bool = False) -> TranslationChunk:
        return TranslationChunk(
            items,
            {item.id: budgets[item.id] for item in items if item.id in budgets},
            budget_key,
            compress=compress_mode,
        )

    batches = _chunks(pending)
    if compress:
        batches = [make_chunk(chunk.sentences, compress_mode=True) for chunk in batches]
    elif length_budget_rate and provider not in MACHINE_TRANSLATION_PROVIDERS:
        budgets = length_budgets(list(sentences), pending, length_budget_rate)
        batches = [make_chunk(chunk.sentences) for chunk in batches]
    translator = _build_translator(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        deepl_formality=deepl_formality,
        microsoft_region=microsoft_region,
        client=client,
        source_language=source_language,
        target_language=target_language,
        extra_body=extra_body,
        sound_policy=sound_policy,
    )

    # Sentences whose old text is about to be replaced must not leak into the
    # context or memory sent to the model; they are shown blank until redone.
    # Compression refines existing translations rather than discarding them.
    stale_ids: set[str] = (
        set() if compress else {sentence.id for sentence in pending if sentence.zh_text}
    )

    def prompt_sentences() -> list[Sentence]:
        if not stale_ids:
            return sentences
        return [
            sentence.model_copy(update={"zh_text": ""}) if sentence.id in stale_ids else sentence
            for sentence in sentences
        ]

    def context_for(chunk: TranslationChunk) -> tuple[str, str]:
        if not send_context:
            return "[]", "[]"
        visible = prompt_sentences()
        return (
            _context_for_chunk(visible, chunk, context_sentences),
            _bounded_translation_memory(visible, chunk, memory_sentences),
        )

    def request(chunk: TranslationChunk, attempts: int | None) -> dict[str, TranslationEntry]:
        context, memory = context_for(chunk)
        previous = getattr(translator, "max_retries", None)
        if attempts is not None and previous is not None:
            translator.max_retries = min(previous, attempts)  # type: ignore[union-attr]
        try:
            return translator.translate_chunk(chunk, context, memory, job_id)  # type: ignore[return-value]
        finally:
            if attempts is not None and previous is not None:
                translator.max_retries = previous  # type: ignore[union-attr]

    last_failure = ""
    translated_ids: list[str] = []

    def fail(sentence: Sentence, reason: str) -> None:
        nonlocal last_failure
        last_failure = reason
        sentence.status = "translation_failed"
        sentence.set_error(reason[:400])
        report.failed.append(TranslationFailure(sentence.id, reason[:400]))

    def settle(sentence: Sentence, entry: TranslationEntry, *, retry_allowed: bool) -> None:
        text = entry.zh.strip()
        duration = sentence.end_seconds - sentence.start_seconds
        reason = ""
        if entry.kind == "sound":
            _mark_sound(sentence, text, sound_policy, explicit=True)
            report.skipped_ids.append(sentence.id)
            stale_ids.discard(sentence.id)
            return
        if text:
            _mark_translated(sentence, text)
            report.translated += 1
            translated_ids.append(sentence.id)
            stale_ids.discard(sentence.id)
            return
        if empty_translation_is_expected(sentence):
            _mark_filler(sentence)
            report.skipped_ids.append(sentence.id)
            stale_ids.discard(sentence.id)
            return
        if retry_allowed:
            # A blank semantic line is retried alone, rather than silently
            # being classified as a sound effect.
            try:
                retried = request(make_chunk([sentence]), None)
            except TranslationError as exc:
                check_cancelled(cancel_event)
                if _is_fatal_translation_error(exc):
                    raise
                reason = f"单句重试失败：{exc}"
            else:
                settle(sentence, _as_entry(retried[sentence.id]), retry_allowed=False)
                return
        if looks_like_sound_effect(sentence.source_text, duration):
            _mark_sound(sentence, "", sound_policy)
            report.skipped_ids.append(sentence.id)
            stale_ids.discard(sentence.id)
            return
        fail(
            sentence,
            reason or "翻译服务未返回此句译文，单句重试后仍为空；保留启用，可再次翻译。",
        )

    def checkpoint() -> None:
        if on_batch:
            on_batch()

    def run(chunk: TranslationChunk, attempts: int | None) -> None:
        """Translate one chunk; on failure split it so the bad line is isolated."""

        check_cancelled(cancel_event)
        count = len(chunk.sentences)
        try:
            results = request(chunk, attempts)
        except _OutputLengthTranslationError as exc:
            if count == 1:
                fail(
                    chunk.sentences[0],
                    f"即使提高输出预算仍超出长度上限：{exc}",
                )
                return
            split_reason = "输出过长"
            retry_attempts = attempts
        except TranslationError as exc:
            check_cancelled(cancel_event)
            if _is_fatal_translation_error(exc):
                raise
            if count == 1:
                fail(chunk.sentences[0], str(exc))
                return
            split_reason = str(exc)
            retry_attempts = _SPLIT_RETRY_ATTEMPTS
        else:
            for sentence in chunk.sentences:
                settle(sentence, _as_entry(results[sentence.id]), retry_allowed=True)
            checkpoint()
            return
        middle = count // 2
        logger.warning("翻译批次 %d 句失败，拆成两半重试：%s", count, split_reason[:200])
        run(chunk.subchunk(chunk.sentences[:middle]), retry_attempts)
        run(chunk.subchunk(chunk.sentences[middle:]), retry_attempts)

    def make_compress_chunk(items: list[Sentence]) -> TranslationChunk:
        return make_chunk(items, compress_mode=True)

    def tighten_chunk(
        chunk: TranslationChunk,
        attempts: int | None,
        *,
        count_translated: bool,
        record_failure: bool,
    ) -> bool:
        """Shorten existing over-budget lines without overwriting text on error."""
        check_cancelled(cancel_event)
        count = len(chunk.sentences)
        try:
            results = request(chunk, attempts)
        except _OutputLengthTranslationError as exc:
            if count == 1:
                if record_failure:
                    report.failed.append(
                        TranslationFailure(
                            chunk.sentences[0].id, f"压缩译文超出输出上限：{exc}"[:400]
                        )
                    )
                else:
                    logger.warning(
                        "压缩过长译文超出上限（%s），保留原译文。", chunk.sentences[0].id
                    )
                return False
            next_attempts = attempts
        except TranslationError as exc:
            check_cancelled(cancel_event)
            if _is_fatal_translation_error(exc):
                raise
            if count == 1:
                if record_failure:
                    report.failed.append(TranslationFailure(chunk.sentences[0].id, str(exc)[:400]))
                else:
                    logger.warning(
                        "压缩过长译文失败（%s），保留原译文：%s", chunk.sentences[0].id, exc
                    )
                return False
            next_attempts = _SPLIT_RETRY_ATTEMPTS
        else:
            changed = False
            for sentence in chunk.sentences:
                stale_ids.discard(sentence.id)
                entry = _as_entry(results[sentence.id])
                candidate = entry.zh.strip()
                if entry.kind == "sound" or not candidate:
                    continue
                if _used_length(candidate, budget_key) > _used_length(sentence.zh_text, budget_key):
                    continue
                if candidate != sentence.zh_text:
                    _mark_translated(sentence, candidate)
                    changed = True
                if sentence.id not in translated_ids:
                    translated_ids.append(sentence.id)
                if count_translated:
                    report.translated += 1
            if changed:
                checkpoint()
            return True
        middle = count // 2
        left_ok = tighten_chunk(
            make_compress_chunk(chunk.sentences[:middle]),
            next_attempts,
            count_translated=count_translated,
            record_failure=record_failure,
        )
        right_ok = tighten_chunk(
            make_compress_chunk(chunk.sentences[middle:]),
            next_attempts,
            count_translated=count_translated,
            record_failure=record_failure,
        )
        return left_ok or right_ok

    def tighten_all(
        targets: list[Sentence], *, count_translated: bool, record_failure: bool
    ) -> None:
        tighten_batches = [make_compress_chunk(c.sentences) for c in _chunks(targets)]
        consecutive_failures = 0
        for index, chunk in enumerate(tighten_batches):
            check_cancelled(cancel_event)
            label = "压缩过长译文" if compress else "精简过长译文"
            if progress:
                progress(
                    f"{provider} · {model} {label}第 {index + 1}/{len(tighten_batches)} 批",
                    index,
                    len(tighten_batches),
                )
            try:
                ok = tighten_chunk(
                    chunk,
                    None,
                    count_translated=count_translated,
                    record_failure=record_failure,
                )
            except TranslationError as exc:
                if _is_fatal_translation_error(exc):
                    if record_failure:
                        raise
                    logger.warning("自动压缩译文遇到致命错误，保留已有译文：%s", exc)
                    return
                raise
            if ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES:
                    if record_failure:
                        report.stopped_reason = (
                            f"连续 {consecutive_failures} 批压缩失败，翻译服务可能不可用。"
                        )
                        raise TranslationError(f"{report.stopped_reason}保留已有译文。")
                    logger.warning(
                        "连续 %d 批自动压缩失败，停止压缩并保留已有译文。",
                        consecutive_failures,
                    )
                    return
            check_cancelled(cancel_event)
            if progress:
                progress(
                    f"已压缩 {index + 1}/{len(tighten_batches)} 批",
                    index + 1,
                    len(tighten_batches),
                )

    cancel_translation = translator.close
    callback_signal = register_cancel_callback(cancel_translation, cancel_event)
    try:
        with translator:
            if compress:
                tighten_all(pending, count_translated=True, record_failure=True)
            else:
                consecutive_failures = 0
                for index, chunk in enumerate(batches):
                    check_cancelled(cancel_event)
                    if progress:
                        progress(
                            f"{provider} · {model} 翻译第 {index + 1}/{len(batches)} 批",
                            index,
                            len(batches),
                        )
                    done_before = report.translated + report.skipped
                    failed_before = len(report.failed)
                    run(chunk, None)
                    if len(report.failed) != failed_before:
                        checkpoint()
                    if report.translated + report.skipped > done_before:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES:
                            report.stopped_reason = (
                                f"连续 {consecutive_failures} 批翻译失败，翻译服务可能不可用。"
                            )
                            raise TranslationError(
                                f"{report.stopped_reason}最后一次错误：{last_failure}。"
                                f"已完成的 {report.translated} 句译文已保存，"
                                "未处理的句子保持待翻译。"
                            )
                    check_cancelled(cancel_event)
                    if progress:
                        progress(f"已翻译 {index + 1}/{len(batches)} 批", index + 1, len(batches))
                if budgets and translated_ids:
                    over_ids = {
                        flag.sentence_id
                        for flag in check_translations(
                            sentences,
                            budgets,
                            budget_key,
                            only_ids=translated_ids,
                            source_language=source_language,
                            target_language=target_language,
                        )
                        if flag.code == "too_long"
                    }
                    over_sentences = [s for s in sentences if s.id in over_ids]
                    if over_sentences:
                        tighten_all(over_sentences, count_translated=False, record_failure=False)
    finally:
        unregister_cancel_callback(cancel_translation, callback_signal)
        if translated_ids:
            report.flagged = check_translations(
                sentences,
                budgets,
                budget_key,
                only_ids=translated_ids,
                source_language=source_language,
                target_language=target_language,
            )
    return report
