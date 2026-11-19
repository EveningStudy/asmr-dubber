"""Core schemas, reports, error types, and JSON envelope validation for translation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .api_contracts import APIContractError
from .errors import TranslationError
from .models import Sentence

Progress = Callable[[str, int, int], None]

# One failed request at most this many times while a batch is being split; the
# whole batch already used the translator's full retry budget.
_SPLIT_RETRY_ATTEMPTS = 2
_MAX_CONSECUTIVE_BATCH_FAILURES = 3
_FATAL_HTTP_STATUS = frozenset({401, 402, 403, 404, 422})


class _NonRetryableTranslationError(TranslationError):
    """Authentication, billing, permission, and request-schema failures."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        fatal: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.fatal = (status_code in _FATAL_HTTP_STATUS) if fatal is None else fatal


class _OutputLengthTranslationError(TranslationError):
    """The completion budget ended before the model finished the JSON response."""


def _is_fatal_translation_error(error: BaseException) -> bool:
    """Credentials, billing, misconfiguration, contract and safety-block errors stop immediately."""

    if isinstance(error, APIContractError):
        return True
    if isinstance(error, _NonRetryableTranslationError):
        return error.fatal or isinstance(error.__cause__, APIContractError)
    return False


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

    @field_validator("zh", mode="before")
    @classmethod
    def normalize_translation(cls, value: str | None) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("zh must be a string or null")
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


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def _extract_json(content: str) -> dict:
    value = _THINK_BLOCK.sub("", content).strip()
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
