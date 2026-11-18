"""Pydantic schemas and monotonic literal span allocation for script reconciliation."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import TranslationError
from .models import Sentence
from .translation_schema import _extract_json


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
