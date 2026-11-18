from __future__ import annotations

import html
import json
import logging
import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Literal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .api_contracts import APIContractError, merge_extra_body
from .api_logging import logged_http_client, safe_api_url
from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .errors import TranslationError
from .filtering import (
    has_speakable_text,
    is_japanese_filler_only,
    kana_to_romaji,
    looks_like_sound_effect,
    speakable_character_count,
)
from .languages import (
    MACHINE_TRANSLATION_LANGUAGE_CODES,
    SourceLanguage,
    SpeechSourceLanguage,
    source_language_label,
)
from .models import Sentence
from .storage import atomic_write_text
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)

Progress = Callable[[str, int, int], None]
logger = logging.getLogger(__name__)


SoundPolicy = Literal["skip", "romaji", "keep_source"]
SOUND_POLICIES: tuple[SoundPolicy, ...] = ("skip", "romaji", "keep_source")

# One failed request at most this many times while a batch is being split; the
# whole batch already used the translator's full retry budget.
_SPLIT_RETRY_ATTEMPTS = 2
_MAX_CONSECUTIVE_BATCH_FAILURES = 3
_FATAL_HTTP_STATUS = frozenset({401, 402, 403})


def empty_translation_is_expected(sentence: Sentence) -> bool:
    return not has_speakable_text(sentence.source_text) or is_japanese_filler_only(
        sentence.source_text
    )


class _NonRetryableTranslationError(TranslationError):
    """Authentication, billing, permission, and request-schema failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_fatal_translation_error(error: BaseException) -> bool:
    """Credentials, billing and permission errors repeat for every request."""

    return (
        isinstance(error, _NonRetryableTranslationError) and error.status_code in _FATAL_HTTP_STATUS
    )


@dataclass(frozen=True)
class TranslationEntry:
    """One validated model answer: target text and what kind of line it is."""

    zh: str
    kind: Literal["speech", "sound"] | None = None


@dataclass(frozen=True)
class TranslationFailure:
    sentence_id: str
    reason: str


@dataclass(frozen=True)
class TranslationFlag:
    """A finished translation that looks wrong and is worth redoing.

    ``code`` is one of ``kana_left``, ``too_long``, ``same_as_source`` and
    ``repeats_previous``; ``reason`` is a user-facing sentence.
    """

    sentence_id: str
    code: str
    reason: str


@dataclass
class TranslationReport:
    """What one ``translate_sentences`` call did, instead of an exception.

    ``translated`` counts sentences that received target text; ``skipped_ids`` are
    sentences deliberately left undubbed (sounds and fillers); ``failed`` are
    sentences that still have no usable translation and can be retried.
    """

    translated: int = 0
    skipped_ids: list[str] = field(default_factory=list)
    failed: list[TranslationFailure] = field(default_factory=list)
    stopped_reason: str | None = None
    # Post-translation self-check of the sentences translated in this call.
    flagged: list[TranslationFlag] = field(default_factory=list)

    @property
    def flagged_ids(self) -> list[str]:
        return list(dict.fromkeys(item.sentence_id for item in self.flagged))

    @property
    def skipped(self) -> int:
        return len(self.skipped_ids)

    @property
    def failed_ids(self) -> list[str]:
        return [item.sentence_id for item in self.failed]

    def summary(self) -> str:
        parts = [f"成功 {self.translated} 句", f"跳过 {self.skipped} 句"]
        if self.failed:
            parts.append(f"失败 {len(self.failed)} 句")
        return "，".join(parts)

    def failure_message(self) -> str:
        shown = "；".join(f"{item.sentence_id}：{item.reason}" for item in self.failed[:6])
        extra = f"（另有 {len(self.failed) - 6} 句）" if len(self.failed) > 6 else ""
        return (
            f"{len(self.failed)} 句未取得译文：{shown}{extra}。"
            "其他句子的译文已保存，修正或稍后再次点击翻译只会重试这些句子。"
        )


class _OutputLengthTranslationError(TranslationError):
    """The completion budget ended before DeepSeek finished the JSON response."""


class TranslationItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    zh: str
    # Optional so custom prompts that only know ``id`` and ``zh`` keep working.
    kind: str | None = None

    @field_validator("id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("zh")
    @classmethod
    def normalize_translation(cls, value: str) -> str:
        return value.strip()

    @field_validator("kind")
    @classmethod
    def normalize_kind(cls, value: str | None) -> str | None:
        # An unknown label is treated as missing so the fallback rules decide.
        normalized = (value or "").strip().casefold()
        return normalized if normalized in {"speech", "sound"} else None


class TranslationEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    translations: list[TranslationItem] = Field(min_length=1)


class ScriptSpan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ScriptQuote(BaseModel):
    """Model selects literal text; Python, not the model, counts characters."""

    model_config = ConfigDict(strict=True, extra="forbid")
    id: str
    quote: str = Field(min_length=1)
    skip_before: str = ""


class ScriptCorrection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    text: str
    script_ids: list[str] = Field(default_factory=list)
    script_spans: list[ScriptSpan] = Field(default_factory=list)
    script_quotes: list[ScriptQuote] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class ScriptCorrectionEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    corrections: list[ScriptCorrection] = Field(min_length=1)


@dataclass(frozen=True)
class TranslationChunk:
    sentences: list[Sentence]
    # Per-sentence length budget sent to the model ("max_chars" or "max_words").
    budgets: dict[str, int] = field(default_factory=dict)
    budget_key: str = "max_chars"
    compress: bool = False

    def subchunk(self, sentences: list[Sentence]) -> TranslationChunk:
        return TranslationChunk(
            sentences,
            {s.id: self.budgets[s.id] for s in sentences if s.id in self.budgets},
            self.budget_key,
            self.compress,
        )


_BUDGET_RULE = (
    "每项的 {key} 是按原句说话时长估算的{unit}上限，用于让配音和原声对齐。"
    "在不丢失意思的前提下尽量不超过它；需要删减时，先删重复、语气词和可以省略的修饰。"
)


def budget_rule(chunk: TranslationChunk) -> str:
    """Prompt sentence explaining the length budget, or "" when there is none."""
    if not chunk.budgets:
        return ""
    unit = "英文单词数" if chunk.budget_key == "max_words" else "字数（不含标点）"
    rule = "\n" + _BUDGET_RULE.format(key=chunk.budget_key, unit=unit)
    if chunk.compress:
        rule += (
            f"\n以上句子当前 existing_zh 超出了 {chunk.budget_key} 上限；"
            "请在不丢失核心意思、人物称谓和语气的前提下精简译文，"
            f"把 zh 压缩到 {chunk.budget_key} 以内。"
        )
    return rule


def length_budgets(
    sentences: list[Sentence], targets: list[Sentence], rate: float
) -> dict[str, int]:
    """How many characters (or words) each line can take when dubbed at ``rate`` per second.

    A dub may run into the pause before the next line, but at most one second past
    the line's own end, so a line before a long silence is not given a paragraph.
    """
    ordered = sorted(sentences, key=lambda s: (s.start_seconds, s.end_seconds, s.id))
    next_start = {current.id: later.start_seconds for current, later in pairwise(ordered)}
    budgets = {}
    for sentence in targets:
        own = sentence.end_seconds - sentence.start_seconds
        available = min(
            next_start.get(sentence.id, sentence.end_seconds + 1.0), sentence.end_seconds + 1.0
        )
        seconds = max(own, available - sentence.start_seconds)
        budgets[sentence.id] = max(2, round(seconds * rate))
    return budgets


def _load_prompt(name: str) -> str:
    return files("asmr_dubber.prompts").joinpath(name).read_text(encoding="utf-8").strip()


_SYSTEM_PROMPT_TEMPLATE = _load_prompt("translation.md")
_DEEPSEEK_STRUCTURE_PROMPT = _load_prompt("translation-structure.md")
_SCRIPT_RECONCILIATION_PROMPT = _load_prompt("script-reconciliation.md")

LLM_RECONCILIATION_PROVIDERS = frozenset(
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


def _openai_provider_controls(
    provider: str,
    model: str,
    base_url: str = "",
) -> dict[str, object]:
    """Return non-thinking controls accepted by each provider's official REST API."""

    host = base_url.casefold()
    if provider == "deepseek" or "api.deepseek.com" in host:
        return {"thinking": {"type": "disabled"}}
    if provider == "bailian" or "dashscope.aliyuncs.com" in host:
        return {"enable_thinking": False}
    if provider == "doubao" or "volces.com" in host or "volcengine.com" in host:
        if model.casefold().startswith("doubao-seed-2-0-"):
            return {"reasoning_effort": "minimal"}
        return {"thinking": {"type": "disabled"}}
    if provider == "sensenova" or "sensenova.cn" in host:
        return {"reasoning_effort": "none"}
    if (
        provider == "openai"
        and model.casefold().startswith("gpt-5")
        and "pro" not in model.casefold()
    ):
        return {"reasoning_effort": "none"}
    return {}


_SOURCE_PROMPT_VALUES: dict[SpeechSourceLanguage, tuple[str, str, str]] = {
    "ja": (
        "日语",
        "日语中的「あ」「え」「う」「ん」「ふふ」「えーと」",
        "\n   日语中的「しこしこ」「シコシコ」及其罗马字转写 “shiko shiko” 或 “shikoshiko” "
        '一律按动作或摩擦拟声处理（`kind` 为 "sound"），不得翻译成“撸”“撸撸”等会被合成语音的文字；'
        "同项还有实义台词时只翻译台词。",
    ),
    "en": ("英语", "英语中的 “uh”“um”“erm”", ""),
    "zh": ("中文", "中文中的“呃”“嗯……”", ""),
}


_SOUND_POLICY_RULES: dict[str, str] = {
    "skip": '`kind` 为 "sound" 时 `zh` 必须是空字符串。',
    "romaji": (
        '`kind` 为 "sound" 时，`zh` 写这个声音的罗马字（小写，例如「き」写 ki），'
        "不得翻成中文拟声词；无法拼写的呼吸声、笑声写空字符串。"
    ),
    "keep_source": '`kind` 为 "sound" 时，`zh` 原样抄写 `source`，不翻译。',
}


def sound_policy_rule(policy: str) -> str:
    return _SOUND_POLICY_RULES.get(policy, _SOUND_POLICY_RULES["skip"])


def default_translation_prompt(
    source_language: SourceLanguage = "ja",
    target_language: str = "zh",
    sound_policy: str = "skip",
) -> str:
    """Render the packaged prompt for one source language and sound policy."""

    prompt_language = source_language
    language, filler_examples, sound_rule = _SOURCE_PROMPT_VALUES[prompt_language]
    template = _SYSTEM_PROMPT_TEMPLATE
    if target_language == "en":
        template = template.replace("简体中文", "英语").replace("中文", "英语")
    prompt = (
        template.replace("{{SOURCE_LANGUAGE}}", language)
        .replace("{{FILLER_EXAMPLES}}", filler_examples)
        .replace("{{SOURCE_SPECIFIC_SOUND_RULE}}", sound_rule)
        .replace("{{SOUND_POLICY_RULE}}", sound_policy_rule(sound_policy))
    )
    if target_language == "en":
        prompt += "\n输出英语台词。JSON 键 zh 是兼容字段名，不表示输出中文；不得修改键名。"
    return prompt


def _builtin_translation_prompts(source_language: SourceLanguage) -> set[str]:
    return {
        default_translation_prompt(source_language, target_lang, sound_policy=policy).strip()
        for target_lang in ("zh", "en")
        for policy in SOUND_POLICIES
    }


# Backward-compatible Japanese default for callers that imported this name.
SYSTEM_PROMPT = default_translation_prompt("ja")


def _deepseek_translation_request(
    target: str,
    expected_ids: list[str],
    attempt: int,
    sound_policy: str = "skip",
) -> str:
    output_schema = json.dumps(
        {
            "translations": [
                {"id": sentence_id, "zh": "", "kind": "speech"} for sentence_id in expected_ids
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    retry_note = (
        ""
        if attempt == 1
        else (
            f"这是第 {attempt} 次校验重试。上一次输出没有通过结构校验；"
            "这次必须逐字沿用结构模板中的键名、id 和顺序。"
        )
    )
    return (
        _DEEPSEEK_STRUCTURE_PROMPT.replace("{{SOUND_POLICY_RULE}}", sound_policy_rule(sound_policy))
        .replace("{{RETRY_NOTE}}", retry_note)
        .replace("{{OUTPUT_SCHEMA}}", output_schema)
        .replace("{{TARGET_JSON}}", target)
        .strip()
    )


def _chunks(
    sentences: list[Sentence],
    max_lines: int = 16,
    max_chars: int = 4_000,
) -> list[TranslationChunk]:
    result: list[TranslationChunk] = []
    current: list[Sentence] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.source_text)
        if current and (len(current) >= max_lines or count + length > max_chars):
            result.append(TranslationChunk(current))
            current = []
            count = 0
        current.append(sentence)
        count += length
    if current:
        result.append(TranslationChunk(current))
    return result


def _json_lines(
    sentences: Iterable[Sentence],
    field: str = "source",
    budgets: dict[str, int] | None = None,
    budget_key: str = "max_chars",
) -> str:
    items = []
    for sentence in sentences:
        item: dict[str, object] = {"id": sentence.id, "source": sentence.source_text}
        if sentence.speaker_id:
            item["speaker"] = sentence.speaker_id
        if budgets and sentence.id in budgets:
            item[budget_key] = budgets[sentence.id]
        if field == "both" and sentence.zh_text:
            item["existing_zh"] = sentence.zh_text
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _translation_memory(sentences: Iterable[Sentence]) -> str:
    items = []
    for sentence in sentences:
        if not sentence.zh_text:
            continue
        entry: dict[str, object] = {
            "id": sentence.id,
            "source": sentence.source_text,
            "zh": sentence.zh_text,
        }
        if sentence.speaker_id:
            entry["speaker"] = sentence.speaker_id
        items.append(entry)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _context_for_chunk(
    sentences: list[Sentence],
    chunk: TranslationChunk,
    maximum_sentences: int,
) -> str:
    if maximum_sentences <= 0:
        return "[]"
    positions = {sentence.id: index for index, sentence in enumerate(sentences)}
    indices = [positions[item.id] for item in chunk.sentences if item.id in positions]
    if not indices:
        return "[]"
    first, last = min(indices), max(indices)
    room = max(0, maximum_sentences - (last - first + 1))
    start = max(0, first - room // 2)
    end = min(len(sentences), last + 1 + (room - (first - start)))
    if end - start < maximum_sentences:
        start = max(0, end - maximum_sentences)
    return _json_lines(sentences[start:end], field="both")


def _bounded_translation_memory(
    sentences: list[Sentence],
    chunk: TranslationChunk,
    maximum_sentences: int,
) -> str:
    if maximum_sentences <= 0:
        return "[]"
    current_ids = {sentence.id for sentence in chunk.sentences}
    confirmed = [
        sentence for sentence in sentences if sentence.zh_text and sentence.id not in current_ids
    ]
    return _translation_memory(confirmed[-maximum_sentences:])


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_trailing_commas(text: str) -> str:
    """Drop commas before ``}``/``]`` that sit outside JSON strings."""

    pieces: list[str] = []
    last = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == ",":
            match = _TRAILING_COMMA.match(text, index)
            if match:
                pieces.append(text[last:index])
                last = index + 1
    pieces.append(text[last:])
    return "".join(pieces)


def _extract_json(content: str) -> dict:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        # Models sometimes add a sentence before/after the object or leave a
        # trailing comma.  Repair only those shapes; ids are still validated later.
        start, end = value.find("{"), value.rfind("}")
        parsed = None
        if 0 <= start < end:
            candidate = value[start : end + 1]
            for text in (candidate, _strip_trailing_commas(candidate)):
                try:
                    parsed = json.loads(text)
                    break
                except json.JSONDecodeError:
                    continue
        if parsed is None:
            raise TranslationError(f"翻译模型返回的不是有效 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise TranslationError("翻译模型 JSON 顶层必须是对象。")
    return parsed


def validate_translation(content: str, expected_ids: list[str]) -> dict[str, str]:
    return {
        sentence_id: entry.zh
        for sentence_id, entry in validate_translation_entries(content, expected_ids).items()
    }


def validate_translation_entries(
    content: str, expected_ids: list[str]
) -> dict[str, TranslationEntry]:
    try:
        envelope = TranslationEnvelope.model_validate(_extract_json(content))
    except (ValidationError, TranslationError) as exc:
        raise TranslationError(f"翻译结构校验失败：{exc}") from exc
    actual_ids = [item.id for item in envelope.translations]
    # Same ids in another order are matched by id; only missing, extra or
    # duplicated ids make the answer unusable.
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        missing = [item for item in expected_ids if item not in actual_ids]
        extra = [item for item in actual_ids if item not in expected_ids]
        raise TranslationError(f"翻译返回的 id 不一致；缺少={missing[:8]}，多出={extra[:8]}")
    return {
        item.id: TranslationEntry(
            item.zh,
            "sound" if item.kind == "sound" else "speech" if item.kind == "speech" else None,
        )
        for item in envelope.translations
    }


def validate_script_reconciliation(content: str, expected_ids: list[str]) -> dict[str, str]:
    try:
        envelope = ScriptCorrectionEnvelope.model_validate(_extract_json(content))
    except (ValidationError, TranslationError) as exc:
        raise TranslationError(f"台本校对返回的结构无效：{exc}") from exc
    actual_ids = [item.id for item in envelope.corrections]
    if actual_ids != expected_ids:
        missing = [item for item in expected_ids if item not in actual_ids]
        extra = [item for item in actual_ids if item not in expected_ids]
        raise TranslationError(
            f"台本校对返回的 id/顺序不一致；缺少={missing[:8]}，多出={extra[:8]}"
        )
    return {item.id: item.text for item in envelope.corrections}


def _validated_script_mapping(
    content: str,
    expected_ids: list[str],
    available_script: list[tuple[str, str]],
    consumed_script_ids: set[str],
) -> tuple[dict[str, str], list[str]]:
    """Validate a monotonic, non-repeating mapping and return exact script text."""

    try:
        envelope = ScriptCorrectionEnvelope.model_validate(_extract_json(content))
    except (ValidationError, TranslationError) as exc:
        raise TranslationError(f"台本校对返回的结构无效：{exc}") from exc
    actual_ids = [item.id for item in envelope.corrections]
    if actual_ids != expected_ids:
        missing = [item for item in expected_ids if item not in actual_ids]
        extra = [item for item in actual_ids if item not in expected_ids]
        raise TranslationError(
            f"台本校对返回的 id/顺序不一致；缺少={missing[:8]}，多出={extra[:8]}"
        )

    script_lookup = dict(available_script)
    script_order = {script_id: index for index, (script_id, _text) in enumerate(available_script)}
    selected_ids: list[str] = []
    corrections: dict[str, str] = {}
    for item in envelope.corrections:
        item_ids = [
            str(script_id).strip() for script_id in item.script_ids if str(script_id).strip()
        ]
        if item.text and not item_ids:
            raise TranslationError(f"{item.id} 返回了文字，但没有标明对应的台本 id。")
        if len(item_ids) != len(set(item_ids)):
            raise TranslationError(f"{item.id} 重复引用了同一条台本。")
        for script_id in item_ids:
            if script_id not in script_lookup:
                raise TranslationError(f"{item.id} 引用了当前窗口之外的台本：{script_id}")
            if script_id in consumed_script_ids or script_id in selected_ids:
                raise TranslationError(f"台本 {script_id} 被重复分配。")
        selected_ids.extend(item_ids)
        parts = [script_lookup[script_id] for script_id in item_ids]
        separator = (
            "" if any(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", part) for part in parts) else " "
        )
        corrections[item.id] = separator.join(parts).strip()

    positions = [script_order[script_id] for script_id in selected_ids]
    if positions != sorted(positions):
        raise TranslationError("台本映射顺序发生倒退。")
    return corrections, selected_ids


def _validated_script_spans(
    content: str,
    recognized: list[Sentence],
    available: list[tuple[str, str]],
    consumed_ends: dict[str, int],
) -> tuple[dict[str, str], dict[str, int], list[dict[str, object]]]:
    """Allocate disjoint literal ranges; time boundaries always remain ASR-owned."""
    validate_script_reconciliation(content, [s.id for s in recognized])
    envelope = ScriptCorrectionEnvelope.model_validate(_extract_json(content))
    lookup = dict(available)
    order = {key: i for i, (key, _) in enumerate(available)}
    ends = dict(consumed_ends)
    owners: dict[str, str] = {}
    last_position = max((order[key] for key in consumed_ends if key in order), default=-1)
    result: dict[str, str] = {}
    allocations: list[dict[str, object]] = []
    for item, sentence in zip(envelope.corrections, recognized, strict=True):
        spans = item.script_spans
        skipped: list[dict[str, object]] = []
        if item.script_quotes:
            if spans or item.script_ids:
                raise TranslationError(f"{item.id} 不得混用 script_quotes 与旧范围字段。")
            quote_ends = dict(ends)
            quote_position = last_position
            spans = []
            for selection in item.script_quotes:
                text = lookup.get(selection.id)
                if text is None:
                    raise TranslationError(f"{item.id} 引用了窗口外台本 {selection.id}。")
                position = order[selection.id]
                if position < quote_position:
                    raise TranslationError(f"{item.id} 台本顺序倒退。")
                if quote_position >= 0 and position > quote_position:
                    previous_key = available[quote_position][0]
                    if quote_ends.get(previous_key, 0) < len(lookup[previous_key]):
                        raise TranslationError(f"台本 {previous_key} 尚有未分配的文字。")
                start = quote_ends.get(selection.id, 0)
                # An explicit literal skip is auditable and avoids ambiguous find().
                literal = selection.skip_before + selection.quote
                if not text.startswith(literal, start):
                    raise TranslationError(
                        f"{item.id} 的 {selection.id} 引文不是剩余台本的原文前缀；"
                        f"从 {start} 开始的原文为 {text[start:]!r}。不得编造或重复分配。"
                    )
                if selection.skip_before:
                    skipped.append(
                        {
                            "id": selection.id,
                            "start": start,
                            "end": start + len(selection.skip_before),
                            "text": selection.skip_before,
                        }
                    )
                quote_start = start + len(selection.skip_before)
                quote_end = quote_start + len(selection.quote)
                spans.append(ScriptSpan(id=selection.id, start=quote_start, end=quote_end))
                quote_ends[selection.id] = quote_end
                quote_position = position
        if spans and item.script_ids:
            raise TranslationError(
                f"{item.id} 同时返回 script_ids 和 script_spans，请只用 script_spans。"
            )
        if not spans:
            if item.script_ids and not item.text:
                # An empty correction means no reliable match. Do not consume
                # an entire line merely because the model repeated its id.
                item.script_ids = []
            # Compatibility with older responses. A single literal prefix can be
            # split without trusting model-authored text or duplicating the full line.
            spans = []
            for key in item.script_ids:
                text = lookup.get(key, "")
                start = ends.get(key, 0)
                remaining = text[start:]
                if len(item.script_ids) == 1 and item.text and remaining.startswith(item.text):
                    end = start + len(item.text)
                else:
                    start, end = 0, len(text)
                spans.append(ScriptSpan(id=key, start=start, end=max(1, end)))
        if item.text and not spans:
            raise TranslationError(
                f"{item.id} 返回文字但没有台本范围；无匹配时返回空文字和空范围。"
            )
        parts: list[str] = []
        previous_part_id: str | None = None
        for span in spans:
            text = lookup.get(span.id)
            context = f"{item.id}（{sentence.start_seconds:.2f}–{sentence.end_seconds:.2f} 秒）"
            if text is None:
                raise TranslationError(f"{context} 引用了窗口外台本 {span.id}。")
            position = order[span.id]
            if position < last_position:
                raise TranslationError(f"{context} 的台本 {span.id} 映射顺序倒退。")
            previous_end = ends.get(span.id, 0)
            if last_position >= 0 and position > last_position:
                previous_key = available[last_position][0]
                if ends.get(previous_key, 0) < len(lookup[previous_key]):
                    raise TranslationError(
                        f"台本 {previous_key} 尚有未分配的文字，请先完成该台本再使用 {span.id}。"
                    )
            if span.start < previous_end:
                owner = owners.get(span.id, "前一批识别区间")
                raise TranslationError(
                    f"台本 {span.id} 被重复分配：{owner} 与 {context}；"
                    f"已使用 [0,{previous_end})，本次 [{span.start},{span.end})。"
                    "应分配互不重叠的原文范围，不得把同一全文写进两句。"
                )
            explicit_skip = next(
                (
                    entry
                    for entry in skipped
                    if entry["id"] == span.id
                    and entry["start"] == previous_end
                    and entry["end"] == span.start
                ),
                None,
            )
            if (
                (span.start != previous_end and explicit_skip is None)
                or span.end > len(text)
                or span.end <= span.start
            ):
                raise TranslationError(
                    f"{context} 的 {span.id} 范围无效；下一个起点应为 {previous_end}，"
                    f"终点应大于起点且不超过 {len(text)}；"
                    f"实际 [{span.start},{span.end})，台本原文：{text!r}。"
                )
            if previous_part_id == span.id:
                parts[-1] += text[span.start : span.end]
            else:
                parts.append(text[span.start : span.end])
            previous_part_id = span.id
            ends[span.id] = span.end
            owners[span.id] = context
            last_position = position
            allocation: dict[str, object] = {"sentence_id": item.id, **span.model_dump()}
            if explicit_skip is not None:
                allocation["skipped_before"] = explicit_skip
            allocations.append(allocation)
        separator = "" if any(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", p) for p in parts) else " "
        result[item.id] = separator.join(parts).strip()
    return result, ends, allocations


class DeepSeekTranslator:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 900.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
        system_prompt: str = "",
        temperature: float = 0.1,
        top_p: float = 1.0,
        minimum_output_tokens: int = 16_384,
        source_language: SourceLanguage = "ja",
        extra_body: str = "{}",
        sound_policy: str = "skip",
    ) -> None:
        if not api_key.strip():
            raise TranslationError("缺少 DeepSeek API Key。请在 UI 中填写或设置 DEEPSEEK_API_KEY。")
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.sound_policy = sound_policy
        self.system_prompt = system_prompt.strip() or default_translation_prompt(
            source_language, sound_policy=sound_policy
        )
        self.temperature = temperature
        self.top_p = top_p
        self.minimum_output_tokens = minimum_output_tokens
        self.source_language = source_language
        self.extra_body = extra_body
        self.client = client or logged_http_client("DeepSeek 翻译 API", timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> DeepSeekTranslator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initial_max_tokens(
        self, sentences: Iterable[Sentence], *, include_existing_zh: bool = False
    ) -> int:
        char_count = sum(
            len(item.source_text) + (len(item.zh_text) if include_existing_zh else 0)
            for item in sentences
        )
        return min(
            65_536,
            max(self.minimum_output_tokens, char_count * 4 + 4_096),
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        job_id: str,
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, bool]:
        tokens = self.minimum_output_tokens if max_tokens is None else max_tokens
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": tokens,
            # Translation is a constrained text-to-text mapping.  Thinking
            # consumes the same completion budget as the final JSON and can
            # end the request before any usable content is returned.
            "thinking": {"type": "disabled"},
            "user_id": job_id,
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        try:
            payload = merge_extra_body(
                payload,
                self.extra_body,
                label="翻译附加请求参数",
                reserved={"model", "messages", "stream"},
            )
        except APIContractError as exc:
            raise _NonRetryableTranslationError(str(exc)) from exc
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code in {401, 402, 403, 422}:
            raise _NonRetryableTranslationError(
                f"DeepSeek API 拒绝请求（HTTP {response.status_code}）：{response.text[:500]}",
                response.status_code,
            )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            return "", True
        content = choice.get("message", {}).get("content") or ""
        return content, False

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, TranslationEntry]:
        expected_ids = [sentence.id for sentence in chunk.sentences]
        target = _json_lines(
            chunk.sentences,
            field="both" if chunk.compress else "source",
            budgets=chunk.budgets,
            budget_key=chunk.budget_key,
        )
        max_tokens = self._initial_max_tokens(chunk.sentences, include_existing_zh=chunk.compress)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            messages = [{"role": "system", "content": self.system_prompt}]
            if full_context != "[]":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"完整{source_language_label(self.source_language)}转写，"
                            "仅供保持人物称谓和上下文一致：\n" + full_context
                        ),
                    }
                )
            if translation_memory != "[]":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "以下是已经确认的译法。后续称谓、专名和语气必须与其一致：\n"
                            + translation_memory
                        ),
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": _deepseek_translation_request(
                        target, expected_ids, attempt, self.sound_policy
                    )
                    + budget_rule(chunk),
                }
            )
            try:
                content, output_limited = self._request(messages, job_id, max_tokens=max_tokens)
                if output_limited:
                    length_error = _OutputLengthTranslationError(
                        f"DeepSeek 输出达到长度上限（max_tokens={max_tokens}）。"
                    )
                    # A multi-line batch is more reliably recovered by splitting
                    # it in the caller.  A single pathological line gets two
                    # larger budgets before failing clearly.
                    if len(chunk.sentences) > 1 or max_tokens >= 65_536:
                        raise length_error
                    max_tokens = min(65_536, max_tokens * 2)
                    last_error = length_error
                    continue
                if not content.strip():
                    raise TranslationError("DeepSeek 偶发返回了空内容。")
                return validate_translation_entries(content, expected_ids)
            except _NonRetryableTranslationError:
                raise
            except _OutputLengthTranslationError:
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                delay = min(20.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.75))
                time.sleep(delay)
        raise TranslationError(
            f"DeepSeek 翻译在 {self.max_retries} 次尝试后仍失败：{last_error}"
        ) from last_error


class LLMTranslator:
    """Provider-specific REST adapters sharing the strict translation contract."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        system_prompt: str = "",
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        timeout_seconds: float = 900.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        extra_body: str = "{}",
        sound_policy: str = "skip",
    ) -> None:
        if not api_key.strip() and provider != "openai_compatible":
            raise TranslationError(f"{provider} 缺少 API Key。")
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.sound_policy = sound_policy
        self.system_prompt = system_prompt.strip() or default_translation_prompt(
            source_language, sound_policy=sound_policy
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.source_language = source_language
        self.extra_body = extra_body
        self.client = client or logged_http_client(f"{provider} 翻译 API", timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> LLMTranslator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _messages(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        attempt: int,
    ) -> list[dict[str, str]]:
        target = _json_lines(
            chunk.sentences,
            field="both" if chunk.compress else "source",
            budgets=chunk.budgets,
            budget_key=chunk.budget_key,
        )
        # Every item carries ``kind``; the sound rule matches the configured policy.
        correction = (
            '\n每项只输出 id、zh、kind 三个字段，kind 为 "speech" 或 "sound"。'
            + sound_policy_rule(self.sound_policy)
            + budget_rule(chunk)
        )
        if attempt > 1:
            correction += (
                f"\n这是第 {attempt} 次校验重试：务必返回全部 id 和原顺序；"
                '没有实义的项 kind 写 "sound"，不要省略。'
            )
        messages = [{"role": "system", "content": self.system_prompt}]
        if full_context != "[]":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"完整{source_language_label(self.source_language)}转写，"
                        "仅供保持人物称谓和上下文一致：\n" + full_context
                    ),
                }
            )
        if translation_memory != "[]":
            messages.append(
                {
                    "role": "user",
                    "content": "以下是已经确认的译法，请保持称谓和专名一致：\n"
                    + translation_memory,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": "请翻译以下目标项并只输出 json：\n" + target + correction,
            }
        )
        return messages

    def _request(self, messages: list[dict[str, str]], job_id: str) -> tuple[str, bool]:
        if self.provider in {
            "deepseek",
            "bailian",
            "doubao",
            "openai",
            "openai_compatible",
            "sensenova",
        }:
            token_field = (
                "max_completion_tokens"
                if self.provider in {"openai", "sensenova"}
                else "max_tokens"
            )
            payload: dict[str, object] = {
                "model": self.model,
                "messages": messages,
                token_field: self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "response_format": {"type": "json_object"},
                "stream": False,
                "user": job_id,
            }
            payload.update(_openai_provider_controls(self.provider, self.model, self.base_url))
            if self.provider == "sensenova":
                # SenseNova's compatibility gateway uses a strict request schema.
                payload.pop("user", None)
            try:
                payload = merge_extra_body(
                    payload,
                    self.extra_body,
                    label="翻译附加请求参数",
                    reserved={"model", "messages", "stream"},
                )
            except APIContractError as exc:
                raise TranslationError(str(exc)) from exc
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            # Some local OpenAI-compatible servers implement chat but not JSON mode.
            if self.provider in {"openai_compatible", "sensenova"} and response.status_code == 400:
                payload.pop("response_format", None)
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            self._raise_for_status(response)
            data = response.json()
            choice = data["choices"][0]
            return choice.get("message", {}).get("content") or "", choice.get(
                "finish_reason"
            ) == "length"

        if self.provider == "anthropic":
            user_messages = [item for item in messages if item["role"] != "system"]
            payload = {
                "model": self.model,
                "system": self.system_prompt,
                "messages": user_messages,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            response = self.client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            self._raise_for_status(response)
            data = response.json()
            content = "".join(
                part.get("text", "")
                for part in data.get("content", [])
                if part.get("type") == "text"
            )
            return content, data.get("stop_reason") == "max_tokens"

        if self.provider == "gemini":
            user_text = "\n\n".join(
                item["content"] for item in messages if item["role"] != "system"
            )
            generation_config: dict[str, object] = {
                "responseMimeType": "application/json",
                "maxOutputTokens": self.max_output_tokens,
            }
            if self.model.startswith("gemini-3.1-pro"):
                generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
            elif self.model.startswith(("gemini-3.6", "gemini-3.5")):
                generation_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
            if not self.model.startswith(("gemini-3.6", "gemini-3.5-flash-lite")):
                generation_config.update(
                    temperature=self.temperature,
                    topP=self.top_p,
                )
            payload = {
                "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generationConfig": generation_config,
            }
            response = self.client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            self._raise_for_status(response)
            data = response.json()
            candidate = data["candidates"][0]
            content = "".join(
                part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
            )
            return content, candidate.get("finishReason") in {"MAX_TOKENS", "LENGTH"}

        raise TranslationError(f"不支持的大模型翻译服务：{self.provider}")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code in {400, 401, 402, 403, 404, 422}:
            raise _NonRetryableTranslationError(
                f"{self.provider} API 拒绝请求（HTTP {response.status_code}）："
                f"{response.text[:500]}",
                response.status_code,
            )
        response.raise_for_status()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, TranslationEntry]:
        expected_ids = [sentence.id for sentence in chunk.sentences]
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                content, output_limited = self._request(
                    self._messages(chunk, full_context, translation_memory, attempt),
                    job_id,
                )
                if output_limited:
                    raise _OutputLengthTranslationError(
                        f"{self.provider} 输出达到长度上限（{self.max_output_tokens} tokens）。"
                    )
                if not content.strip():
                    raise TranslationError(f"{self.provider} 返回了空内容。")
                return validate_translation_entries(content, expected_ids)
            except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(20.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.75)))
        raise TranslationError(
            f"{self.provider} 翻译在 {self.max_retries} 次尝试后仍失败：{last_error}"
        ) from last_error


@dataclass(frozen=True)
class ScriptReconciliationBatch:
    recognized: list[Sentence]
    script: list[tuple[str, str]]


def _script_fragments(script_lines: list[str]) -> list[str]:
    """Split long script rows at sentence boundaries before assigning them once."""

    fragments: list[str] = []
    for raw in script_lines:
        line = str(raw or "").strip()
        if not line:
            continue
        sentences = [
            item.strip() for item in re.split(r"(?<=[。！？!?])\s*|(?<=…)\s*", line) if item.strip()
        ]
        for sentence in sentences or [line]:
            if len(sentence) <= 80:
                fragments.append(sentence)
                continue
            clauses = [
                item.strip()
                for item in re.split(r"(?<=[，、,；;：:])\s*", sentence)
                if item.strip()
            ]
            fragments.extend(clauses or [sentence])
    return fragments


def _recognized_script_batches(
    recognized: list[Sentence],
    *,
    maximum_recognized: int = 8,
) -> list[list[Sentence]]:
    if not recognized:
        raise TranslationError("没有可供台本校对的识别结果。")
    return [
        recognized[start : start + maximum_recognized]
        for start in range(0, len(recognized), maximum_recognized)
    ]


def _script_reconciliation_messages(
    batch: ScriptReconciliationBatch,
    *,
    source_language: SourceLanguage,
    target: Literal["source", "zh"],
    attempt: int,
    last_error: str = "",
    consumed_ends: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    output_label = source_language_label(source_language) if target == "source" else "中文"
    retry_note = (
        "" if attempt == 1 else f"这是第 {attempt} 次校验重试。请修正以下具体错误：{last_error}"
    )
    recognized_payload = []
    for sentence in batch.recognized:
        item: dict[str, object] = {
            "id": sentence.id,
            "start": round(sentence.start_seconds, 3),
            "end": round(sentence.end_seconds, 3),
            "recognized": sentence.source_text,
        }
        if target == "zh":
            item["translation"] = sentence.zh_text
        recognized_payload.append(item)
    script_payload = [
        {
            "id": line_id,
            "text": text,
            "length": len(text),
            "available_start": (consumed_ends or {}).get(line_id, 0),
        }
        for line_id, text in batch.script
    ]
    recognized_json = json.dumps(recognized_payload, ensure_ascii=False, separators=(",", ":"))
    script_json = json.dumps(script_payload, ensure_ascii=False, separators=(",", ":"))
    system = (
        _SCRIPT_RECONCILIATION_PROMPT.replace("{{OUTPUT_LABEL}}", output_label)
        .replace("{{RETRY_NOTE}}", retry_note)
        .replace("{{RECOGNIZED_JSON}}", recognized_json)
        .replace("{{SCRIPT_JSON}}", script_json)
        .strip()
    )
    user = (
        "请只根据下面两组数据完成校对。输出结构必须是："
        + json.dumps(
            {
                "corrections": [
                    {"id": sentence.id, "script_quotes": [], "text": ""}
                    for sentence in batch.recognized
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n请根据系统消息中的数据完成校对。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def reconcile_script_sentences(
    recognized: list[Sentence],
    script_lines: list[str],
    *,
    source_language: SourceLanguage,
    target: Literal["source", "zh"],
    provider: str,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_output_tokens: int = 16_384,
    job_id: str = "asmr_dubber_script_review",
    progress: Progress | None = None,
    on_batch: Callable[[], None] | None = None,
    client: httpx.Client | None = None,
    cancel_event: CancellationSignal | None = None,
    extra_body: str = "{}",
    diagnostic_path: Path | None = None,
    script_source: str = "",
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Use an LLM to map an untimed script onto ASR timing without changing time ranges."""

    if provider not in LLM_RECONCILIATION_PROVIDERS:
        raise TranslationError(
            "台本校对需要大模型翻译服务；请选择 DeepSeek、百炼、豆包、商汤、OpenAI、Claude、"
            "Gemini 或本地/自定义 OpenAI-compatible。"
        )
    fragments = _script_fragments(script_lines)
    if not fragments:
        raise TranslationError("台本中没有可供校对的文字。")
    recognized_batches = _recognized_script_batches(recognized)
    script_items = [(f"p{index:06d}", text) for index, text in enumerate(fragments, start=1)]
    script_lookup = dict(script_items)
    script_position = {script_id: index for index, (script_id, _text) in enumerate(script_items)}
    script_cursor = 0
    consumed_script_ids: set[str] = set()
    consumed_ends: dict[str, int] = {}
    origins = []
    for line_number, line in enumerate(script_lines, start=1):
        origins.extend([line_number] * len(_script_fragments([line])))
    translator = LLMTranslator(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt="",
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        client=client,
        source_language=source_language,
        extra_body=extra_body,
    )
    cancel_translation = translator.close
    callback_signal = register_cancel_callback(cancel_translation, cancel_event)
    corrections: dict[str, str] = {}
    report: list[dict[str, object]] = []
    try:
        with translator:
            for index, recognized_batch in enumerate(recognized_batches):
                check_cancelled(cancel_event)
                remaining_batches = len(recognized_batches) - index
                remaining_script = max(1, len(script_items) - script_cursor)
                expected_span = (remaining_script + remaining_batches - 1) // remaining_batches
                window_size = min(48, max(12, expected_span + 8))
                available_script = script_items[script_cursor : script_cursor + window_size]
                batch = ScriptReconciliationBatch(recognized_batch, available_script)
                expected_ids = [sentence.id for sentence in batch.recognized]
                last_error: Exception | None = None
                result: dict[str, str] | None = None
                selected_script_ids: list[str] = []
                new_ends: dict[str, int] = {}
                allocations: list[dict[str, object]] = []
                content = ""
                review_ids: list[str] = []
                for attempt in range(1, 4):
                    check_cancelled(cancel_event)
                    messages = _script_reconciliation_messages(
                        batch,
                        source_language=source_language,
                        target=target,
                        attempt=attempt,
                        last_error=str(last_error or ""),
                        consumed_ends=consumed_ends,
                    )
                    # Anthropic and Gemini use this field as their system instruction;
                    # OpenAI-compatible providers also receive the explicit system message.
                    translator.system_prompt = messages[0]["content"]
                    try:
                        content, output_limited = translator._request(messages, job_id)
                        if output_limited:
                            raise _OutputLengthTranslationError(
                                f"台本校对第 {index + 1} 批输出达到长度上限。"
                            )
                        if not content.strip():
                            raise TranslationError("台本校对模型返回了空内容。")
                        result, new_ends, allocations = _validated_script_spans(
                            content,
                            batch.recognized,
                            available_script,
                            consumed_ends,
                        )
                        selected_script_ids = list(dict.fromkeys(str(a["id"]) for a in allocations))
                        break
                    except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                        raise
                    except TranslationError as exc:
                        last_error = exc
                    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                        last_error = exc
                    if attempt < 3:
                        time.sleep(min(8.0, attempt * 1.5 + random.uniform(0.0, 0.5)))
                if result is None:
                    # Transport/auth failures must not be reported as a successful review.
                    if not isinstance(last_error, TranslationError):
                        raise TranslationError(f"台本校对请求失败：{last_error}") from last_error
                    ids_in_error = re.findall(r"\bp\d{6}\b", str(last_error))
                    locations = [
                        f"{key}=导入后第 {origins[script_position[key]]} 行"
                        for key in dict.fromkeys(ids_in_error)
                        if key in script_position
                    ]
                    diagnostic = {
                        "source": script_source,
                        "batch": index + 1,
                        "error": str(last_error),
                        "recognized": [s.model_dump() for s in batch.recognized],
                        "script": [
                            {"id": key, "parsed_line": origins[script_position[key]], "text": text}
                            for key, text in batch.script
                        ],
                        "response": content,
                        "completed_batches": report,
                    }
                    if diagnostic_path is not None:
                        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_text(
                            diagnostic_path, json.dumps(diagnostic, ensure_ascii=False, indent=2)
                        )
                    logger.warning("台本校对降级为逐句重试；位置：%s；详见项目诊断", locations)
                    result, new_ends, allocations = {}, dict(consumed_ends), []
                    for sentence in batch.recognized:
                        check_cancelled(cancel_event)
                        messages = _script_reconciliation_messages(
                            ScriptReconciliationBatch([sentence], available_script),
                            source_language=source_language,
                            target=target,
                            attempt=2,
                            last_error=str(last_error),
                            consumed_ends=new_ends,
                        )
                        translator.system_prompt = messages[0]["content"]
                        single_content, limited = translator._request(messages, job_id)
                        if limited:
                            raise _OutputLengthTranslationError("台本逐句校对达到输出长度上限。")
                        try:
                            single, next_ends, assigned = _validated_script_spans(
                                single_content,
                                [sentence],
                                available_script,
                                new_ends,
                            )
                            if not single.get(sentence.id):
                                raise TranslationError("逐句重试没有可靠匹配")
                        except TranslationError:
                            review_ids.append(sentence.id)
                            result[sentence.id] = (
                                sentence.source_text if target == "source" else sentence.zh_text
                            )
                            logger.warning("%s 待人工校对，保留原文；详见项目诊断", sentence.id)
                        else:
                            result.update(single)
                            new_ends = next_ends
                            allocations.extend(assigned)
                    selected_script_ids = list(dict.fromkeys(str(a["id"]) for a in allocations))
                corrections.update(result)
                consumed_ends = new_ends
                consumed_script_ids.update(
                    key for key in selected_script_ids if new_ends[key] == len(script_lookup[key])
                )
                if selected_script_ids:
                    last_key = selected_script_ids[-1]
                    script_cursor = script_position[last_key] + (last_key in consumed_script_ids)
                report.append(
                    {
                        "recognized_ids": expected_ids,
                        "recognized": [
                            {
                                "id": s.id,
                                "start": s.start_seconds,
                                "end": s.end_seconds,
                                "original": s.source_text,
                            }
                            for s in batch.recognized
                        ],
                        "available_script_ids": [line_id for line_id, _ in batch.script],
                        "consumed_script_ids": selected_script_ids,
                        "corrections": result,
                        "allocations": allocations,
                        "script_source": script_source,
                        "needs_review_ids": review_ids,
                    }
                )
                if on_batch:
                    on_batch()
                if progress:
                    progress(
                        f"台本校对第 {index + 1}/{len(recognized_batches)} 批完成"
                        + (f"；{len(review_ids)} 句待人工校对，已保留原文" if review_ids else ""),
                        index + 1,
                        len(recognized_batches),
                    )
                check_cancelled(cancel_event)
    finally:
        unregister_cancel_callback(cancel_translation, callback_signal)
    unused = [
        script_id for script_id, _text in script_items if script_id not in consumed_script_ids
    ]
    if not any(corrections.values()) and not any(entry.get("needs_review_ids") for entry in report):
        if diagnostic_path is not None:
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                diagnostic_path,
                json.dumps(
                    {"error": "未匹配到任何台本文字", "batches": report, "script": script_items},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        raise TranslationError(
            "台本没有可靠匹配到任何识别区间；未应用空结果，请核对音轨、台本和语言。"
            + (f"详细报告：{diagnostic_path}" if diagnostic_path else "")
        )
    if unused:
        report.append(
            {
                "unused_script_ids": unused,
                "remaining_script": [
                    {
                        "id": key,
                        "start": consumed_ends.get(key, 0),
                        "text": text[consumed_ends.get(key, 0) :],
                    }
                    for key, text in script_items
                    if key in unused
                ],
                "reason": "这些台本片段没有可靠匹配到 ASR 时间区间，需要人工检查",
            }
        )
    return corrections, report


class MachineTranslationAPI:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        deepl_formality: str,
        microsoft_region: str,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        target_language: str = "zh",
    ) -> None:
        if not api_key.strip():
            raise TranslationError(f"{provider} 缺少 API Key。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.deepl_formality = deepl_formality
        self.microsoft_region = microsoft_region.strip()
        self.source_language: SourceLanguage = source_language
        self.target_language = target_language
        self.client = client or logged_http_client(f"{provider} 翻译 API", timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> MachineTranslationAPI:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        _full_context: str,
        _translation_memory: str,
        _job_id: str,
    ) -> dict[str, str]:
        texts = [sentence.source_text for sentence in chunk.sentences]
        language_codes = MACHINE_TRANSLATION_LANGUAGE_CODES[self.provider]
        source_code = language_codes[self.source_language]
        try:
            if self.provider == "deepl":
                payload: dict[str, object] = {
                    "text": texts,
                    "source_lang": source_code,
                    "target_lang": "EN-US" if self.target_language == "en" else "ZH-HANS",
                    "model_type": self.model,
                }
                if self.deepl_formality != "default":
                    payload["formality"] = self.deepl_formality
                response = self.client.post(
                    f"{self.base_url}/v2/translate",
                    headers={
                        "Authorization": f"DeepL-Auth-Key {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                translated = [item["text"] for item in response.json()["translations"]]
            elif self.provider == "google_translate":
                response = self.client.post(
                    self.base_url,
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json={
                        "q": texts,
                        "source": source_code,
                        "target": "en" if self.target_language == "en" else "zh-CN",
                        "format": "text",
                        "model": self.model,
                    },
                )
                response.raise_for_status()
                translated = [
                    html.unescape(item["translatedText"])
                    for item in response.json()["data"]["translations"]
                ]
            elif self.provider == "microsoft_translate":
                headers = {
                    "Ocp-Apim-Subscription-Key": self.api_key,
                    "Content-Type": "application/json; charset=UTF-8",
                }
                if self.microsoft_region:
                    headers["Ocp-Apim-Subscription-Region"] = self.microsoft_region
                response = self.client.post(
                    f"{self.base_url}/translate",
                    params={
                        "api-version": "3.0",
                        "from": source_code,
                        "to": "en" if self.target_language == "en" else "zh-Hans",
                    },
                    headers=headers,
                    json=[{"Text": text} for text in texts],
                )
                response.raise_for_status()
                translated = [item["translations"][0]["text"] for item in response.json()]
            else:
                raise TranslationError(f"不支持的机器翻译服务：{self.provider}")
        except httpx.HTTPStatusError as exc:
            response = exc.response
            message = (
                f"{self.provider} API 请求失败（HTTP {response.status_code}）："
                f"{response.text[:500]}"
            )
            if response.status_code in _FATAL_HTTP_STATUS:
                raise _NonRetryableTranslationError(message, response.status_code) from exc
            raise TranslationError(message) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise TranslationError(f"{self.provider} 返回格式无效：{exc}") from exc
        if len(translated) != len(chunk.sentences) or any(not item.strip() for item in translated):
            raise TranslationError(f"{self.provider} 返回的译文数量或内容无效。")
        return {
            sentence.id: text.strip()
            for sentence, text in zip(chunk.sentences, translated, strict=True)
        }


def _as_entry(value: TranslationEntry | str) -> TranslationEntry:
    """Machine translators and test doubles may still return plain strings."""

    return value if isinstance(value, TranslationEntry) else TranslationEntry(str(value))


def _clear_audio(sentence: Sentence) -> None:
    sentence.tts_file = None
    sentence.tts_cache_key = None
    sentence.tts_duration_seconds = None
    sentence.tts_candidates = []
    sentence.tts_warning = None
    sentence.tts_attempt = 0


def _mark_translated(sentence: Sentence, text: str) -> None:
    if sentence.zh_text != text:
        _clear_audio(sentence)
    if sentence.status == "skipped_sound" and not sentence.chinese_audio_enabled:
        # The row was a sound under an earlier run and is a real line now.
        sentence.chinese_audio_enabled = True
    sentence.zh_text = text
    sentence.status = "translated"
    sentence.error = None


def _mark_filler(sentence: Sentence) -> None:
    sentence.zh_text = ""
    sentence.enabled = False
    _clear_audio(sentence)
    sentence.status = "skipped_filler"
    sentence.error = None


def _looks_like_romaji(text: str) -> bool:
    return bool(text) and all(ord(character) < 128 or character in "…’‘—–" for character in text)


def _sound_text(sentence: Sentence, model_text: str, policy: str) -> str:
    """Subtitle text for a sound row under the romaji and keep_source policies."""

    if policy == "keep_source":
        return sentence.source_text
    if _looks_like_romaji(model_text):
        return model_text
    converted = kana_to_romaji(sentence.source_text)
    # Kanji cannot be spelled locally; the original is the honest subtitle then.
    return sentence.source_text if re.search(r"[㐀-鿿]", converted) else converted


def _mark_sound(sentence: Sentence, model_text: str, policy: str) -> None:
    """Apply the sound policy: skip it, or keep a subtitle that is never voiced."""

    _clear_audio(sentence)
    text = "" if policy == "skip" else _sound_text(sentence, model_text, policy)
    sentence.zh_text = text
    sentence.status = "skipped_sound"
    sentence.error = None
    if text:
        # Visible in subtitles, but the Chinese TTS must not read "ki" aloud.
        sentence.chinese_audio_enabled = False
    else:
        sentence.enabled = False


_KANA = re.compile(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\uff66-\uff9d]")
_ANY_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\uff66-\uff9f]")
_SHARED_HANZI_ASCII_KEY = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff0-9a-z]+$")
_HANZI_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# A line over its length budget by more than this share is flagged.
LENGTH_OVERRUN_RATIO = 1.2


def _same_text_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def _allow_same_as_source(
    source_text: str,
    target_text: str,
    key: str,
    *,
    source_language: SourceLanguage,
    target_language: str,
) -> bool:
    normalized_source = unicodedata.normalize("NFKC", source_text)
    normalized_target = unicodedata.normalize("NFKC", target_text)
    if _ANY_KANA.search(normalized_source) or _ANY_KANA.search(normalized_target):
        return False
    if not _SHARED_HANZI_ASCII_KEY.fullmatch(key):
        return False
    if _HANZI_CHAR.search(key) and (target_language != "zh" or source_language == "zh"):
        return False
    words = re.findall(r"[A-Za-z]+", normalized_target)
    if not words:
        return True
    if len(words) > 1:
        return False
    word = words[0]
    if word.casefold() == "ok" or (word.isupper() and len(word) <= 6):
        return True
    return source_language == "ja" and len(word) <= 4


def _used_length(text: str, budget_key: str) -> int:
    return len(text.split()) if budget_key == "max_words" else speakable_character_count(text)


def check_translations(
    sentences: list[Sentence],
    budgets: dict[str, int] | None = None,
    budget_key: str = "max_chars",
    only_ids: Iterable[str] | None = None,
    *,
    source_language: SourceLanguage = "ja",
    target_language: str | None = None,
) -> list[TranslationFlag]:
    """Self-check finished translations; nothing is written to the sentences.

    Flags leftover kana, a length over ``budgets`` by more than 20%, text equal to
    the source, and text equal to the previous line's translation (unless the
    source lines are equal too, which is ordinary repetition).  The check can be
    run again on any project, so the result never needs to be stored.
    """

    wanted = None if only_ids is None else set(only_ids)
    resolved_target = (
        target_language
        if target_language is not None
        else ("en" if budget_key == "max_words" else "zh")
    )
    ordered = sorted(
        (s for s in sentences if s.enabled and s.zh_text),
        key=lambda s: (s.start_seconds, s.end_seconds, s.id),
    )
    flags: list[TranslationFlag] = []
    previous: Sentence | None = None
    for sentence in ordered:
        before, previous = previous, sentence
        if (wanted is not None and sentence.id not in wanted) or sentence.status != "translated":
            continue
        text = sentence.zh_text
        if _KANA.search(text):
            flags.append(
                TranslationFlag(sentence.id, "kana_left", "译文里还有日文假名，可能漏译或没译完。")
            )
        budget = (budgets or {}).get(sentence.id)
        if budget:
            used = _used_length(text, budget_key)
            if used > budget * LENGTH_OVERRUN_RATIO:
                flags.append(
                    TranslationFlag(
                        sentence.id,
                        "too_long",
                        f"译文 {used} 超出时长预算 {budget} 的 20% 以上，配音可能赶不上。",
                    )
                )
        key = _same_text_key(text)
        if (
            key
            and key == _same_text_key(sentence.source_text)
            and has_speakable_text(text)
            and not _allow_same_as_source(
                sentence.source_text,
                text,
                key,
                source_language=source_language,
                target_language=resolved_target,
            )
        ):
            flags.append(TranslationFlag(sentence.id, "same_as_source", "译文与原文完全相同。"))
        if (
            before is not None
            and key
            and key == _same_text_key(before.zh_text)
            and _same_text_key(sentence.source_text) != _same_text_key(before.source_text)
        ):
            flags.append(
                TranslationFlag(
                    sentence.id, "repeats_previous", "译文与上一句完全重复，原文并不相同。"
                )
            )
    return flags


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
    mt_providers = {"deepl", "google_translate", "microsoft_translate"}
    report = TranslationReport()
    selected = None if sentence_ids is None else set(sentence_ids)
    redo = retranslate or selected is not None
    budgets: dict[str, int] = {}
    budget_key = "max_words" if target_language == "en" else "max_chars"
    if compress:
        if source_language == target_language or not length_budget_rate or provider in mt_providers:
            if progress:
                progress("所有句子已符合时长预算", 1, 1)
            return report
        pool = [
            sentence
            for sentence in sentences
            if sentence.enabled
            and sentence.zh_text.strip()
            and sentence.status == "translated"
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
    default_prompts = _builtin_translation_prompts(source_language)
    if not system_prompt.strip() or system_prompt.strip() in default_prompts:
        system_prompt = default_translation_prompt(source_language, target_language, sound_policy)
    elif target_language == "en":
        system_prompt += (
            "\n本次覆盖目标语言为英语，输入原文语言为"
            + source_language_label(source_language)
            + "；忽略此前输出中文的要求。"
        )
        system_prompt += "\n目标语言必须是英语；JSON 的 zh 字段存放英文译文，这是兼容字段名。"
    if not system_prompt.strip():
        system_prompt = default_translation_prompt(source_language, target_language, sound_policy)
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
    elif length_budget_rate and provider not in mt_providers:
        budgets = length_budgets(list(sentences), pending, length_budget_rate)
        batches = [make_chunk(chunk.sentences) for chunk in batches]
    if provider == "deepseek":
        translator: DeepSeekTranslator | LLMTranslator | MachineTranslationAPI = DeepSeekTranslator(
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
    elif provider in {
        "bailian",
        "doubao",
        "openai",
        "anthropic",
        "gemini",
        "openai_compatible",
        "sensenova",
    }:
        translator = LLMTranslator(
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
    elif provider in {"deepl", "google_translate", "microsoft_translate"}:
        translator = MachineTranslationAPI(
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
    else:
        raise TranslationError(f"未知翻译服务：{provider}")

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
        sentence.error = reason[:400]
        report.failed.append(TranslationFailure(sentence.id, reason[:400]))

    def settle(sentence: Sentence, entry: TranslationEntry, *, retry_allowed: bool) -> None:
        text = entry.zh.strip()
        duration = sentence.end_seconds - sentence.start_seconds
        reason = ""
        if entry.kind == "sound":
            _mark_sound(sentence, text, sound_policy)
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
    ) -> None:
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
                return
            next_attempts = attempts
        except TranslationError as exc:
            check_cancelled(cancel_event)
            if _is_fatal_translation_error(exc):
                if record_failure:
                    raise
                logger.warning("自动压缩译文遇到致命错误，保留已有译文：%s", exc)
                return
            if count == 1:
                if record_failure:
                    report.failed.append(TranslationFailure(chunk.sentences[0].id, str(exc)[:400]))
                else:
                    logger.warning(
                        "压缩过长译文失败（%s），保留原译文：%s", chunk.sentences[0].id, exc
                    )
                return
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
            return
        middle = count // 2
        tighten_chunk(
            make_compress_chunk(chunk.sentences[:middle]),
            next_attempts,
            count_translated=count_translated,
            record_failure=record_failure,
        )
        tighten_chunk(
            make_compress_chunk(chunk.sentences[middle:]),
            next_attempts,
            count_translated=count_translated,
            record_failure=record_failure,
        )

    def tighten_all(
        targets: list[Sentence], *, count_translated: bool, record_failure: bool
    ) -> None:
        tighten_batches = [make_compress_chunk(c.sentences) for c in _chunks(targets)]
        for index, chunk in enumerate(tighten_batches):
            check_cancelled(cancel_event)
            label = "压缩过长译文" if compress else "精简过长译文"
            if progress:
                progress(
                    f"{provider} · {model} {label}第 {index + 1}/{len(tighten_batches)} 批",
                    index,
                    len(tighten_batches),
                )
            tighten_chunk(
                chunk,
                None,
                count_translated=count_translated,
                record_failure=record_failure,
            )
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
    report.flagged = check_translations(
        sentences,
        budgets,
        budget_key,
        only_ids=translated_ids,
        source_language=source_language,
        target_language=target_language,
    )
    return report


# ---------------------------------------------------------------------------
# Revising existing translations on request ("按要求改译")
#
# Nothing below writes to a sentence.  The result is a list of proposals that the
# caller stores and the user accepts or rejects one by one.
# ---------------------------------------------------------------------------

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

    @field_validator("zh")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

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
        translator: DeepSeekTranslator | LLMTranslator = DeepSeekTranslator(
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
        translator = LLMTranslator(
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
        length_budgets(list(sentences), eligible, length_budget_rate) if length_budget_rate else {}
    )
    batches = _chunks(eligible, max_lines=REVISION_BATCH_SIZE)
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
                time.sleep(min(20.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.75)))
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
