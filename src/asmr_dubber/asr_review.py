from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ReviewDecision:
    text: str
    evidence_ids: tuple[str, ...]
    confidence: float
    decision: str
    reason: str
    selected_candidate: int | None


_SELECTION_CONTRACT = """硬性输出规则（优先级高于其它提示）：
1. 你只能为每个目标窗口选择程序给出的候选序号，不得自由生成、改写、拼接或翻译文字。
2. selected_candidate 使用每个窗口中从 1 开始的 candidate 编号；0 仅表示确定没有实义语音。
3. 每个 target_window_id 恰好输出一项，顺序必须一致。
4. 只输出严格 JSON：
{"results":[{"window_id":"w000001","selected_candidate":1,"confidence":0.8}]}
"""


def _overlap(left: Sentence, right: ReviewWindow) -> float:
    return max(0.0, min(left.end_seconds, right.end) - max(left.start_seconds, right.start))


def _build_windows(
    transcriptions: list[tuple[str, list[Sentence]]],
    max_drift_seconds: float,
) -> list[ReviewWindow]:
    if not transcriptions or not transcriptions[0][1]:
        return []
    primary_label, primary = transcriptions[0]
    windows = [
        ReviewWindow(id="", start=item.start_seconds, end=item.end_seconds) for item in primary
    ]
    for source_index, (label, sentences) in enumerate(transcriptions):
        for sentence_index, sentence in enumerate(sentences):
            if source_index == 0:
                window = windows[sentence_index]
            else:
                midpoint = (sentence.start_seconds + sentence.end_seconds) / 2
                ranked = sorted(
                    (
                        (
                            _overlap(sentence, window),
                            -abs(midpoint - (window.start + window.end) / 2),
                            index,
                        )
                        for index, window in enumerate(windows)
                    ),
                    reverse=True,
                )
                best_overlap, negative_distance, best_index = ranked[0]
                if best_overlap <= 0 and -negative_distance > max_drift_seconds:
                    continue
                window = windows[best_index]
            window.evidence.append(
                Evidence(
                    id="",
                    source=label if source_index else primary_label,
                    text=sentence.source_text,
                    start=sentence.start_seconds,
                    end=sentence.end_seconds,
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


def _window_candidates(window: ReviewWindow) -> list[ReviewCandidate]:
    grouped: dict[str, list[Evidence]] = {}
    for evidence in window.evidence:
        normalized = _normalize_candidate_text(evidence.text)
        if normalized:
            grouped.setdefault(normalized, []).append(evidence)
    return [
        ReviewCandidate(
            text=evidence_group[0].text.strip(),
            evidence_ids=tuple(item.id for item in evidence_group),
            sources=tuple(dict.fromkeys(item.source for item in evidence_group)),
        )
        for evidence_group in grouped.values()
    ]


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
                "text_priority": text_priority_source in candidate.sources,
            }
            for index, candidate in enumerate(_window_candidates(window), start=1)
        ],
    }


def _fallback_candidate_index(window: ReviewWindow, text_priority_source: str) -> int:
    candidates = _window_candidates(window)
    for index, candidate in enumerate(candidates, start=1):
        if text_priority_source and text_priority_source in candidate.sources:
            return index
    primary_source = window.evidence[0].source if window.evidence else ""
    for index, candidate in enumerate(candidates, start=1):
        if primary_source in candidate.sources:
            return index
    return 1 if candidates else 0


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
        sources = len(candidates[0].sources)
        return _decision_for_candidate(
            window,
            1,
            confidence=0.98 if sources >= 2 else 0.75,
            decision="consensus" if sources >= 2 else "single_candidate",
            reason=(f"{sources} 个识别模型文字一致" if sources >= 2 else "只有一个非空候选"),
        )
    votes = [len(candidate.sources) for candidate in candidates]
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
            reason=f"{best_votes} 个识别模型文字一致",
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
        validated[window.id] = _decision_for_candidate(
            window,
            selected_candidate,
            confidence=min(1.0, max(0.0, confidence)),
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
    return _validate_results(_extract_object(content), targets)


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
            sentence = Sentence(
                id=f"s{len(sentences) + 1:06d}",
                start_seconds=max(0.0, start),
                end_seconds=end,
                source_text=text,
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
