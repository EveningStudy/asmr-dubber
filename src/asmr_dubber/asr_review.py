from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from .errors import AsmrDubberError
from .forced_alignment import align_sentences_with_qwen
from .languages import SourceLanguage, source_language_label
from .models import ProjectSettings, Sentence
from .task_control import CancellationSignal, check_cancelled
from .translation import LLMTranslator
from .user_settings import PROVIDER_PRESETS, resolve_api_key

Progress = Callable[[str, int, int], None]
_LLM_PROVIDERS = {
    "deepseek",
    "bailian",
    "doubao",
    "openai",
    "anthropic",
    "gemini",
    "openai_compatible",
    "sensenova",
}


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    text: str
    start: float
    end: float


@dataclass
class ReviewWindow:
    id: str
    start: float
    end: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewCandidate:
    text: str
    evidence_ids: tuple[str, ...]
    sources: tuple[str, ...]
    families: tuple[str, ...]


@dataclass(frozen=True)
class ReviewDecision:
    text: str
    evidence_ids: tuple[str, ...]
    confidence: float
    decision: str
    reason: str
    selected_candidate: int | None


@dataclass(frozen=True)
class FlattenedTranscript:
    raw_text: str
    normalized_text: str
    raw_positions: tuple[int, ...]
    token_starts: tuple[float, ...]
    token_ends: tuple[float, ...]
    sentence_ranges: tuple[tuple[int, int], ...]


_MIN_LLM_CONFIDENCE = 0.65
_FUZZY_CONSENSUS_THRESHOLD = 0.84
_ALIGNMENT_BLOCK_SECONDS = 60.0


_SELECTION_CONTRACT = """硬性输出规则（优先级高于其它提示）：
1. 你只能为每个目标窗口选择程序给出的候选序号，不得自由生成、改写、拼接或翻译文字。
2. selected_candidate 使用每个窗口中从 1 开始的 candidate 编号；0 仅表示确定没有实义语音。
3. 每个 target_window_id 恰好输出一项，顺序必须一致。
4. 只输出严格 JSON：
{"results":[{"window_id":"w000001","selected_candidate":1,"confidence":0.8}]}
"""


def _alignment_characters(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    ]


def _flatten_transcript(sentences: list[Sentence]) -> FlattenedTranscript:
    raw_parts: list[str] = []
    normalized: list[str] = []
    raw_positions: list[int] = []
    token_starts: list[float] = []
    token_ends: list[float] = []
    sentence_ranges: list[tuple[int, int]] = []
    raw_offset = 0
    for sentence_index, sentence in enumerate(sentences):
        if sentence_index:
            raw_parts.append("\n")
            raw_offset += 1
        text = sentence.source_text.strip()
        raw_parts.append(text)
        characters: list[tuple[str, int]] = []
        for raw_index, raw_character in enumerate(text):
            characters.extend(
                (character, raw_offset + raw_index)
                for character in _alignment_characters(raw_character)
            )
        token_start = len(normalized)
        count = len(characters)
        duration = max(0.0, sentence.end_seconds - sentence.start_seconds)
        for character_index, (character, raw_position) in enumerate(characters):
            normalized.append(character)
            raw_positions.append(raw_position)
            token_starts.append(sentence.start_seconds + duration * character_index / max(1, count))
            token_ends.append(
                sentence.start_seconds + duration * (character_index + 1) / max(1, count)
            )
        sentence_ranges.append((token_start, len(normalized)))
        raw_offset += len(text)
    return FlattenedTranscript(
        raw_text="".join(raw_parts),
        normalized_text="".join(normalized),
        raw_positions=tuple(raw_positions),
        token_starts=tuple(token_starts),
        token_ends=tuple(token_ends),
        sentence_ranges=tuple(sentence_ranges),
    )


def _project_boundary(
    opcodes: Sequence[tuple[str, int, int, int, int]],
    primary_index: int,
    primary_length: int,
    secondary_length: int,
) -> int:
    if primary_index <= 0:
        return 0
    if primary_index >= primary_length:
        return secondary_length
    for _tag, primary_start, primary_end, secondary_start, secondary_end in opcodes:
        if primary_start <= primary_index < primary_end:
            primary_span = primary_end - primary_start
            if primary_span <= 0:
                return secondary_start
            ratio = (primary_index - primary_start) / primary_span
            return round(secondary_start + ratio * (secondary_end - secondary_start))
    return secondary_length


def _raw_fragment(transcript: FlattenedTranscript, start: int, end: int) -> str:
    start = max(0, min(start, len(transcript.raw_positions)))
    end = max(start, min(end, len(transcript.raw_positions)))
    if start == end:
        return ""
    raw_start = transcript.raw_positions[start]
    raw_end = (
        transcript.raw_positions[end]
        if end < len(transcript.raw_positions)
        else len(transcript.raw_text)
    )
    while raw_end <= raw_start and end < len(transcript.raw_positions):
        end += 1
        raw_end = (
            transcript.raw_positions[end]
            if end < len(transcript.raw_positions)
            else len(transcript.raw_text)
        )
    return " ".join(transcript.raw_text[raw_start:raw_end].split()).strip()


def _temporal_token_range(
    transcript: FlattenedTranscript,
    start: float,
    end: float,
    max_drift_seconds: float,
) -> tuple[int, int]:
    matching = [
        index
        for index, (token_start, token_end) in enumerate(
            zip(transcript.token_starts, transcript.token_ends, strict=True)
        )
        if token_end >= start - max_drift_seconds and token_start <= end + max_drift_seconds
    ]
    if not matching:
        return (0, 0)
    return matching[0], matching[-1] + 1


def _project_source_to_primary_windows(
    primary_sentences: list[Sentence],
    secondary_sentences: list[Sentence],
    max_drift_seconds: float,
) -> list[tuple[str, float, float] | None]:
    projected: list[tuple[str, float, float] | None] = []
    block_start = 0
    while block_start < len(primary_sentences):
        block_end = block_start + 1
        first_start = primary_sentences[block_start].start_seconds
        while (
            block_end < len(primary_sentences)
            and primary_sentences[block_end].end_seconds - first_start <= _ALIGNMENT_BLOCK_SECONDS
        ):
            block_end += 1
        primary_block_sentences = primary_sentences[block_start:block_end]
        time_start = primary_block_sentences[0].start_seconds - max_drift_seconds
        time_end = primary_block_sentences[-1].end_seconds + max_drift_seconds
        secondary_block_sentences = [
            sentence
            for sentence in secondary_sentences
            if sentence.end_seconds >= time_start and sentence.start_seconds <= time_end
        ]
        primary = _flatten_transcript(primary_block_sentences)
        secondary = _flatten_transcript(secondary_block_sentences)
        if not primary.normalized_text or not secondary.normalized_text:
            projected.extend(None for _ in primary_block_sentences)
            block_start = block_end
            continue
        opcodes = SequenceMatcher(
            None,
            primary.normalized_text,
            secondary.normalized_text,
            autojunk=False,
        ).get_opcodes()
        for sentence, (primary_start, primary_end) in zip(
            primary_block_sentences, primary.sentence_ranges, strict=True
        ):
            secondary_start = _project_boundary(
                opcodes,
                primary_start,
                len(primary.normalized_text),
                len(secondary.normalized_text),
            )
            secondary_end = _project_boundary(
                opcodes,
                primary_end,
                len(primary.normalized_text),
                len(secondary.normalized_text),
            )
            if secondary_end <= secondary_start:
                secondary_start, secondary_end = _temporal_token_range(
                    secondary,
                    sentence.start_seconds,
                    sentence.end_seconds,
                    max_drift_seconds,
                )
            text = _raw_fragment(secondary, secondary_start, secondary_end)
            if not text:
                projected.append(None)
                continue
            projected_start = secondary.token_starts[secondary_start]
            projected_end = secondary.token_ends[secondary_end - 1]
            distance = max(
                0.0,
                sentence.start_seconds - projected_end,
                projected_start - sentence.end_seconds,
            )
            if distance > max_drift_seconds:
                temporal_start, temporal_end = _temporal_token_range(
                    secondary,
                    sentence.start_seconds,
                    sentence.end_seconds,
                    max_drift_seconds,
                )
                temporal_text = _raw_fragment(secondary, temporal_start, temporal_end)
                if not temporal_text:
                    projected.append(None)
                    continue
                secondary_start, secondary_end, text = temporal_start, temporal_end, temporal_text
                projected_start = secondary.token_starts[secondary_start]
                projected_end = secondary.token_ends[secondary_end - 1]
            projected.append((text, projected_start, projected_end))
        block_start = block_end
    return projected


def _join_source_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}{right}"


def _stabilize_primary_sentences(sentences: list[Sentence]) -> list[Sentence]:
    """Merge timestamp glitches that are too short to be useful review windows."""

    stable = [sentence.model_copy(deep=True) for sentence in sentences]
    while len(stable) > 1:
        micro_index = next(
            (
                index
                for index, sentence in enumerate(stable)
                if sentence.end_seconds - sentence.start_seconds < 0.2
            ),
            None,
        )
        if micro_index is None:
            break
        if micro_index == 0:
            merge_left = False
        elif micro_index == len(stable) - 1:
            merge_left = True
        else:
            previous_gap = max(
                0.0, stable[micro_index].start_seconds - stable[micro_index - 1].end_seconds
            )
            next_gap = max(
                0.0, stable[micro_index + 1].start_seconds - stable[micro_index].end_seconds
            )
            merge_left = previous_gap < next_gap
        if merge_left:
            target = stable[micro_index - 1]
            source = stable[micro_index]
            target.end_seconds = max(target.end_seconds, source.end_seconds)
            target.source_text = _join_source_text(target.source_text, source.source_text)
            stable.pop(micro_index)
        else:
            source = stable[micro_index]
            target = stable[micro_index + 1]
            target.start_seconds = min(source.start_seconds, target.start_seconds)
            target.source_text = _join_source_text(source.source_text, target.source_text)
            stable.pop(micro_index)
    return stable


def _build_windows(
    transcriptions: list[tuple[str, list[Sentence]]],
    max_drift_seconds: float,
) -> list[ReviewWindow]:
    if not transcriptions or not transcriptions[0][1]:
        return []
    primary_label, primary_sentences = transcriptions[0]
    primary = _stabilize_primary_sentences(primary_sentences)
    windows = [
        ReviewWindow(id="", start=item.start_seconds, end=item.end_seconds) for item in primary
    ]
    for source_index, (label, sentences) in enumerate(transcriptions):
        if source_index == 0:
            projected = [
                (sentence.source_text, sentence.start_seconds, sentence.end_seconds)
                for sentence in primary
            ]
        else:
            projected = _project_source_to_primary_windows(
                primary,
                sentences,
                max_drift_seconds,
            )
        for window, candidate in zip(windows, projected, strict=True):
            if candidate is None:
                continue
            text, start, end = candidate
            window.evidence.append(
                Evidence(
                    id="",
                    source=label if source_index else primary_label,
                    text=text,
                    start=start,
                    end=end,
                )
            )
    for window_index, window in enumerate(windows, start=1):
        window.id = f"w{window_index:06d}"
        window.evidence = [
            Evidence(
                id=f"{window.id}-c{evidence_index:02d}",
                source=evidence.source,
                text=evidence.text,
                start=evidence.start,
                end=evidence.end,
            )
            for evidence_index, evidence in enumerate(window.evidence, start=1)
        ]
    return windows


def _window_payload(
    window: ReviewWindow,
    text_priority_source: str = "",
    timestamp_priority_source: str = "",
) -> dict[str, Any]:
    return {
        "window_id": window.id,
        "approximate_time": [round(window.start, 3), round(window.end, 3)],
        "candidates": [
            {
                "id": item.id,
                "source": item.source,
                "family": _source_family(item.source),
                "text": item.text,
                "time": [round(item.start, 3), round(item.end, 3)],
                "text_priority": item.source == text_priority_source,
                "timestamp_priority": item.source == timestamp_priority_source,
            }
            for item in window.evidence
        ],
    }


def _normalize_candidate_text(text: str) -> str:
    """Normalize harmless presentation differences before deterministic voting."""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _source_family(source: str) -> str:
    backend = source.partition("|")[0].casefold().strip()
    if "parakeet" in backend:
        return "parakeet"
    if "whisper" in backend:
        return "whisper"
    return backend or source.casefold().strip()


def _candidate_similarity(left: str, right: str) -> float:
    left_normalized = _normalize_candidate_text(left)
    right_normalized = _normalize_candidate_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    shorter = min(len(left_normalized), len(right_normalized))
    longer = max(len(left_normalized), len(right_normalized))
    if shorter <= 3 or shorter / longer < 0.65:
        return 0.0
    return SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
        autojunk=False,
    ).ratio()


def _candidate_quality(text: str) -> float:
    normalized = _normalize_candidate_text(text)
    if not normalized:
        return -10.0
    score = 0.0
    if "�" in text:
        score -= 4.0
    if len(normalized) >= 12:
        trigrams = [normalized[index : index + 3] for index in range(len(normalized) - 2)]
        unique_ratio = len(set(trigrams)) / max(1, len(trigrams))
        if unique_ratio < 0.7:
            score -= (0.7 - unique_ratio) * 5.0
    return score


def _candidate_warning(text: str) -> str:
    normalized = _normalize_candidate_text(text)
    warnings: list[str] = []
    if "�" in text:
        warnings.append("包含解码异常字符")
    if len(normalized) >= 12:
        trigrams = [normalized[index : index + 3] for index in range(len(normalized) - 2)]
        unique_ratio = len(set(trigrams)) / max(1, len(trigrams))
        if unique_ratio < 0.7:
            warnings.append("存在异常重复")
    return "；".join(warnings)


def _window_candidates(window: ReviewWindow) -> list[ReviewCandidate]:
    evidence = [item for item in window.evidence if _normalize_candidate_text(item.text)]
    parents = list(range(len(evidence)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(evidence)):
        for right in range(left + 1, len(evidence)):
            if (
                _candidate_similarity(evidence[left].text, evidence[right].text)
                >= _FUZZY_CONSENSUS_THRESHOLD
            ):
                union(left, right)
    grouped: dict[int, list[Evidence]] = {}
    for index, item in enumerate(evidence):
        grouped.setdefault(root(index), []).append(item)

    candidates: list[ReviewCandidate] = []
    for evidence_group in grouped.values():
        representative = max(
            evidence_group,
            key=lambda item: (
                sum(_candidate_similarity(item.text, other.text) for other in evidence_group),
                _candidate_quality(item.text),
                len(_normalize_candidate_text(item.text)),
                -window.evidence.index(item),
            ),
        )
        sources = tuple(dict.fromkeys(item.source for item in evidence_group))
        candidates.append(
            ReviewCandidate(
                text=representative.text.strip(),
                evidence_ids=tuple(item.id for item in evidence_group),
                sources=sources,
                families=tuple(dict.fromkeys(_source_family(source) for source in sources)),
            )
        )
    return candidates


def _candidate_payload(
    window: ReviewWindow,
    text_priority_source: str,
) -> dict[str, Any]:
    return {
        "window_id": window.id,
        "approximate_time": [round(window.start, 3), round(window.end, 3)],
        "candidates": [
            {
                "candidate": index,
                "text": candidate.text,
                "sources": list(candidate.sources),
                "families": list(candidate.families),
                "text_priority": text_priority_source in candidate.sources,
                "quality_warning": _candidate_warning(candidate.text),
            }
            for index, candidate in enumerate(_window_candidates(window), start=1)
        ],
    }


def _fallback_candidate_index(window: ReviewWindow, text_priority_source: str) -> int:
    candidates = _window_candidates(window)
    primary_source = window.evidence[0].source if window.evidence else ""
    ranked = sorted(
        enumerate(candidates, start=1),
        key=lambda item: (
            len(item[1].families),
            _candidate_quality(item[1].text),
            text_priority_source in item[1].sources,
            primary_source in item[1].sources,
            -item[0],
        ),
        reverse=True,
    )
    return ranked[0][0] if ranked else 0


def _decision_for_candidate(
    window: ReviewWindow,
    selected_candidate: int,
    *,
    confidence: float,
    decision: str,
    reason: str,
) -> ReviewDecision:
    candidates = _window_candidates(window)
    if selected_candidate == 0:
        return ReviewDecision("", (), confidence, decision, reason, 0)
    candidate = candidates[selected_candidate - 1]
    return ReviewDecision(
        candidate.text,
        candidate.evidence_ids,
        confidence,
        decision,
        reason,
        selected_candidate,
    )


def _deterministic_decision(
    window: ReviewWindow,
    text_priority_source: str,
) -> ReviewDecision | None:
    candidates = _window_candidates(window)
    if not candidates:
        return ReviewDecision("", (), 1.0, "non_speech", "没有非空识别候选", 0)
    if len(candidates) == 1:
        if _candidate_quality(candidates[0].text) < -1.0:
            return None
        families = len(candidates[0].families)
        return _decision_for_candidate(
            window,
            1,
            confidence=0.96 if families >= 2 else 0.72,
            decision="consensus" if families >= 2 else "single_family",
            reason=(
                f"{families} 个独立模型家族文字一致"
                if families >= 2
                else "只有一个模型家族提供有效候选"
            ),
        )
    votes = [
        len(candidate.families) if _candidate_quality(candidate.text) >= -1.0 else 0
        for candidate in candidates
    ]
    best_votes = max(votes)
    winners = [
        index for index, votes_count in enumerate(votes, start=1) if votes_count == best_votes
    ]
    if best_votes >= 2 and len(winners) == 1:
        return _decision_for_candidate(
            window,
            winners[0],
            confidence=min(0.99, 0.75 + best_votes * 0.08),
            decision="consensus",
            reason=f"{best_votes} 个独立模型家族文字一致",
        )
    return None


def _extract_object(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AsmrDubberError(f"ASR（语音识别）校对模型返回的不是有效 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise AsmrDubberError("ASR（语音识别）校对 JSON 顶层必须是对象。")
    return payload


def _request_json(
    settings: ProjectSettings,
    messages: list[dict[str, str]],
    job_id: str,
) -> str:
    provider = settings.translation_provider
    if provider not in _LLM_PROVIDERS:
        raise AsmrDubberError(
            "多 ASR（语音识别）校对需要大模型；当前翻译供应商不是 LLM。"
            "请改用 DeepSeek、百炼、豆包、商汤、OpenAI、Claude、Gemini 或本地兼容接口。"
        )
    key = resolve_api_key(provider)
    preset = PROVIDER_PRESETS[provider]
    base_url = (settings.translation_base_url or str(preset["base_url"])).rstrip("/")
    if provider == "deepseek":
        if not key:
            raise AsmrDubberError("多 ASR（语音识别）校对缺少 DeepSeek API Key。")
        with httpx.Client(timeout=settings.asr_timeout_seconds) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": settings.translation_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "max_tokens": min(65_536, settings.translation_max_output_tokens),
                    "thinking": {"type": "disabled"},
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                    "user_id": job_id,
                },
            )
            if response.status_code >= 400:
                raise AsmrDubberError(
                    "DeepSeek ASR（语音识别）校对失败"
                    f"（HTTP {response.status_code}）：{response.text[:800]}"
                )
            data = response.json()
            return str(data["choices"][0]["message"]["content"])

    adapter = LLMTranslator(
        provider=provider,
        api_key=key,
        model=settings.translation_model,
        base_url=base_url,
        system_prompt=f"{settings.asr_review_prompt.strip()}\n\n{_SELECTION_CONTRACT}",
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=settings.translation_max_output_tokens,
        timeout_seconds=settings.asr_timeout_seconds,
        extra_body=settings.translation_extra_body,
    )
    try:
        content, limited = adapter._request(messages, job_id)
    finally:
        adapter.close()
    if limited:
        raise AsmrDubberError("ASR（语音识别）校对模型输出达到长度上限。")
    return content


def _validate_results(
    payload: Mapping[str, Any],
    targets: list[ReviewWindow],
    text_priority_source: str,
) -> dict[str, ReviewDecision]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise AsmrDubberError("ASR（语音识别）校对 JSON 缺少 results 数组。")
    expected = [window.id for window in targets]
    actual = [str(item.get("window_id", "")) for item in results if isinstance(item, Mapping)]
    if actual != expected:
        raise AsmrDubberError("ASR（语音识别）校对返回的 window_id 数量或顺序不一致。")
    validated: dict[str, ReviewDecision] = {}
    for window, item in zip(targets, results, strict=True):
        assert isinstance(item, Mapping)
        try:
            selected_candidate = int(item.get("selected_candidate", -1))
        except (TypeError, ValueError) as exc:
            raise AsmrDubberError(f"{window.id} 的候选序号无效。") from exc
        if not 0 <= selected_candidate <= len(_window_candidates(window)):
            raise AsmrDubberError(f"{window.id} 选择了不存在的候选序号。")
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        if confidence < _MIN_LLM_CONFIDENCE:
            fallback = _fallback_candidate_index(window, text_priority_source)
            validated[window.id] = _decision_for_candidate(
                window,
                fallback,
                confidence=confidence,
                decision="low_confidence_fallback",
                reason=(
                    f"大模型置信度 {confidence:.2f} 低于 {_MIN_LLM_CONFIDENCE:.2f}，"
                    "已回退到程序评分最高的现有候选"
                ),
            )
        else:
            validated[window.id] = _decision_for_candidate(
                window,
                selected_candidate,
                confidence=confidence,
                decision="llm_choice",
                reason="大模型从现有候选中选择",
            )
    return validated


def _review_chunk(
    windows: list[ReviewWindow],
    targets: list[ReviewWindow],
    settings: ProjectSettings,
    job_id: str,
    source_language: SourceLanguage,
) -> dict[str, ReviewDecision]:
    target_ids = [window.id for window in targets]
    messages = [
        {
            "role": "system",
            "content": f"{settings.asr_review_prompt.strip()}\n\n{_SELECTION_CONTRACT}",
        },
        {
            "role": "user",
            "content": f"当前音频的源语言是：{source_language_label(source_language)}。",
        },
        {
            "role": "user",
            "content": (
                "作品、人物、场景及专有词背景（可能为空，只作为消歧信息，不是台词证据）：\n"
                + (settings.asr_review_background or "未提供")
            ),
        },
        {
            "role": "user",
            "content": (
                "以下包含目标窗口和少量相邻上下文。只输出 target_window_ids 中的项目：\n"
                f"文字优先来源：{settings.asr_review_text_priority_model or '未指定'}\n"
                f"时间戳优先来源：{settings.asr_review_timestamp_priority_model or '未指定'}\n"
                + json.dumps(
                    {
                        "target_window_ids": target_ids,
                        "windows": [
                            _candidate_payload(window, settings.asr_review_text_priority_model)
                            for window in windows
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        },
    ]
    content = _request_json(settings, messages, job_id)
    return _validate_results(
        _extract_object(content),
        targets,
        settings.asr_review_text_priority_model,
    )


def _evidence_range(
    window: ReviewWindow,
    selected_ids: list[str],
    timestamp_priority_source: str = "",
) -> tuple[float, float]:
    preferred = [item for item in window.evidence if item.source == timestamp_priority_source]
    if selected_ids and preferred:
        return (
            min(item.start for item in preferred),
            max(item.end for item in preferred),
        )
    return window.start, window.end


def review_transcriptions(
    transcriptions: list[tuple[str, list[Sentence]]],
    settings: ProjectSettings,
    report_path: Path,
    analysis_audio: Path | None = None,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SourceLanguage = "ja",
) -> list[Sentence]:
    """Resolve several timed ASR hypotheses while keeping timestamps evidence-bound."""
    check_cancelled(cancel_event)
    windows = _build_windows(list(transcriptions), settings.asr_review_max_drift_seconds)
    if not windows:
        raise AsmrDubberError("多 ASR（语音识别）校对没有可比较的候选窗口。")
    reviewed: dict[str, ReviewDecision] = {}
    ambiguous: list[ReviewWindow] = []
    position_by_id = {window.id: index for index, window in enumerate(windows)}
    for window in windows:
        decision = _deterministic_decision(window, settings.asr_review_text_priority_model)
        if decision is None:
            ambiguous.append(window)
        else:
            reviewed[window.id] = decision

    chunk_size = 8
    total = math.ceil(len(ambiguous) / chunk_size) if ambiguous else 0
    for chunk_index, start in enumerate(range(0, len(ambiguous), chunk_size), start=1):
        check_cancelled(cancel_event)
        targets = ambiguous[start : start + chunk_size]
        target_positions = [position_by_id[window.id] for window in targets]
        context = windows[
            max(0, min(target_positions) - 2) : min(len(windows), max(target_positions) + 3)
        ]
        if progress:
            progress(f"大模型复核争议句：{chunk_index}/{total}", chunk_index - 1, total)
        try:
            reviewed.update(
                _review_chunk(
                    context,
                    targets,
                    settings,
                    f"asr-review-{chunk_index}-{int(time.time())}",
                    source_language,
                )
            )
        except (AsmrDubberError, httpx.HTTPError, KeyError, IndexError, ValueError) as batch_error:
            check_cancelled(cancel_event)
            for window in targets:
                position = position_by_id[window.id]
                try:
                    reviewed.update(
                        _review_chunk(
                            windows[max(0, position - 2) : position + 3],
                            [window],
                            settings,
                            f"asr-review-{window.id}-{int(time.time())}",
                            source_language,
                        )
                    )
                except (AsmrDubberError, httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    check_cancelled(cancel_event)
                    selected = _fallback_candidate_index(
                        window, settings.asr_review_text_priority_model
                    )
                    reviewed[window.id] = _decision_for_candidate(
                        window,
                        selected,
                        confidence=0.35,
                        decision="fallback",
                        reason=f"大模型复核失败，已回退主模型：{exc or batch_error}",
                    )
        check_cancelled(cancel_event)

    sentences: list[Sentence] = []
    sentence_by_window: dict[str, Sentence] = {}
    report_results: list[dict[str, Any]] = []
    for window in windows:
        decision = reviewed[window.id]
        text = decision.text
        selected_ids = list(decision.evidence_ids)
        confidence = decision.confidence
        start, end = _evidence_range(
            window,
            selected_ids,
            settings.asr_review_timestamp_priority_model,
        )
        if text and end > start:
            needs_review = decision.decision in {
                "single_family",
                "fallback",
                "low_confidence_fallback",
            }
            sentence = Sentence(
                id=f"s{len(sentences) + 1:06d}",
                start_seconds=max(0.0, start),
                end_seconds=end,
                source_text=text,
                status="review_uncertain" if needs_review else "pending",
                error=(
                    f"多模型校对未形成可靠共识：{decision.reason}。请试听并核对原文。"
                    if needs_review
                    else None
                ),
            )
            sentences.append(sentence)
            sentence_by_window[window.id] = sentence
        report_results.append(
            {
                "window_id": window.id,
                "source": text,
                "evidence_ids": selected_ids,
                "confidence": confidence,
                "decision": decision.decision,
                "reason": decision.reason,
                "selected_candidate": decision.selected_candidate,
                "needs_review": decision.decision
                in {"single_family", "fallback", "low_confidence_fallback"},
                "computed_time": [start, end],
                "text_priority_model": settings.asr_review_text_priority_model,
                "timestamp_priority_model": settings.asr_review_timestamp_priority_model,
                "candidates": _window_payload(
                    window,
                    settings.asr_review_text_priority_model,
                    settings.asr_review_timestamp_priority_model,
                )["candidates"],
            }
        )
    if not sentences:
        raise AsmrDubberError(
            f"多 ASR（语音识别）校对没有保留任何可信{source_language_label(source_language)}句子。"
        )
    sentences.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    for index, sentence in enumerate(sentences, start=1):
        sentence.id = f"s{index:06d}"
    alignment_report: list[dict[str, Any]] = []
    if settings.asr_review_timestamp_priority_model.startswith("qwen_forced_aligner|"):
        if analysis_audio is None:
            raise AsmrDubberError("Qwen3 ForcedAligner 需要 ASR（语音识别）分析音频。")
        cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        alignment_report = align_sentences_with_qwen(
            analysis_audio,
            sentences,
            settings,
            progress=progress,
            source_language=source_language,
            **cancel_kwargs,
        )
    for item in report_results:
        sentence = sentence_by_window.get(str(item["window_id"]))
        if sentence is not None:
            item["sentence_id"] = sentence.id
            item["computed_time"] = [sentence.start_seconds, sentence.end_seconds]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"results": report_results, "timestamp_alignment": alignment_report},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if progress:
        progress(
            f"多 ASR（语音识别）校对完成：保留 {len(sentences)} 句，"
            f"其中 {len(ambiguous)} 句需要大模型复核",
            total,
            total,
        )
    return sentences
