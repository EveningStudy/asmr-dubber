from __future__ import annotations

import json
import math
import random
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .errors import AsmrDubberError
from .models import ProjectSettings, Sentence
from .translation import LLMTranslator
from .user_settings import PROVIDER_PRESETS, resolve_api_key

Progress = Callable[[str, int, int], None]
_LLM_PROVIDERS = {"deepseek", "openai", "anthropic", "gemini", "openai_compatible"}


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
    pending: list[tuple[str, Sentence]] = []
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
                    pending.append((label, sentence))
                    continue
                window = windows[best_index]
            window.evidence.append(
                Evidence(
                    id="",
                    source=label if source_index else primary_label,
                    text=sentence.ja_text,
                    start=sentence.start_seconds,
                    end=sentence.end_seconds,
                )
            )

    # Keep content seen only by a secondary recognizer. Nearby secondary-only
    # segments are grouped, but never merged into a distant primary sentence.
    for label, sentence in sorted(pending, key=lambda item: item[1].start_seconds):
        midpoint = (sentence.start_seconds + sentence.end_seconds) / 2
        candidates = [
            window
            for window in windows
            if not any(e.source == primary_label for e in window.evidence)
            and abs(midpoint - (window.start + window.end) / 2) <= max_drift_seconds
        ]
        if candidates:
            window = min(
                candidates,
                key=lambda item: abs(midpoint - (item.start + item.end) / 2),
            )
            window.start = min(window.start, sentence.start_seconds)
            window.end = max(window.end, sentence.end_seconds)
        else:
            window = ReviewWindow(id="", start=sentence.start_seconds, end=sentence.end_seconds)
            windows.append(window)
        window.evidence.append(
            Evidence(
                id="",
                source=label,
                text=sentence.ja_text,
                start=sentence.start_seconds,
                end=sentence.end_seconds,
            )
        )

    windows.sort(key=lambda item: (item.start, item.end))
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


def _window_payload(window: ReviewWindow) -> dict[str, Any]:
    return {
        "window_id": window.id,
        "approximate_time": [round(window.start, 3), round(window.end, 3)],
        "candidates": [
            {
                "id": item.id,
                "source": item.source,
                "text": item.text,
                "time": [round(item.start, 3), round(item.end, 3)],
            }
            for item in window.evidence
        ],
    }


def _extract_object(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AsmrDubberError(f"ASR 校对模型返回的不是有效 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise AsmrDubberError("ASR 校对 JSON 顶层必须是对象。")
    return payload


def _request_json(
    settings: ProjectSettings,
    messages: list[dict[str, str]],
    job_id: str,
) -> str:
    provider = settings.translation_provider
    if provider not in _LLM_PROVIDERS:
        raise AsmrDubberError(
            "多 ASR 校对需要大模型；当前翻译供应商不是 LLM。"
            "请改用 DeepSeek/OpenAI/Claude/Gemini/本地兼容接口。"
        )
    key = resolve_api_key(provider)
    preset = PROVIDER_PRESETS[provider]
    base_url = (settings.translation_base_url or str(preset["base_url"])).rstrip("/")
    if provider == "deepseek":
        if not key:
            raise AsmrDubberError("多 ASR 校对缺少 DeepSeek API Key。")
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
                    f"DeepSeek ASR 校对失败（HTTP {response.status_code}）：{response.text[:800]}"
                )
            data = response.json()
            return str(data["choices"][0]["message"]["content"])

    adapter = LLMTranslator(
        provider=provider,
        api_key=key,
        model=settings.translation_model,
        base_url=base_url,
        system_prompt=settings.asr_review_prompt,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=settings.translation_max_output_tokens,
        timeout_seconds=settings.asr_timeout_seconds,
    )
    try:
        content, limited = adapter._request(messages, job_id)  # noqa: SLF001
    finally:
        adapter.close()
    if limited:
        raise AsmrDubberError("ASR 校对模型输出达到长度上限。")
    return content


def _validate_results(
    payload: Mapping[str, Any],
    targets: list[ReviewWindow],
) -> dict[str, tuple[str, list[str], float]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise AsmrDubberError("ASR 校对 JSON 缺少 results 数组。")
    expected = [window.id for window in targets]
    actual = [str(item.get("window_id", "")) for item in results if isinstance(item, Mapping)]
    if actual != expected:
        raise AsmrDubberError("ASR 校对返回的 window_id 数量或顺序不一致。")
    validated: dict[str, tuple[str, list[str], float]] = {}
    for window, item in zip(targets, results, strict=True):
        assert isinstance(item, Mapping)
        text = str(item.get("ja", "") or "").strip()
        ids = item.get("evidence_ids") or []
        if not isinstance(ids, list):
            raise AsmrDubberError(f"{window.id} 的 evidence_ids 不是数组。")
        allowed = {evidence.id for evidence in window.evidence}
        selected = [str(value) for value in ids]
        if any(value not in allowed for value in selected):
            raise AsmrDubberError(f"{window.id} 引用了不存在的 ASR 证据。")
        if text and not selected:
            raise AsmrDubberError(f"{window.id} 生成了文字但没有引用证据。")
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        validated[window.id] = (text, selected, min(1.0, max(0.0, confidence)))
    return validated


def _review_chunk(
    windows: list[ReviewWindow],
    targets: list[ReviewWindow],
    settings: ProjectSettings,
    job_id: str,
) -> dict[str, tuple[str, list[str], float]]:
    target_ids = [window.id for window in targets]
    messages = [
        {"role": "system", "content": settings.asr_review_prompt},
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
                + json.dumps(
                    {
                        "target_window_ids": target_ids,
                        "windows": [_window_payload(window) for window in windows],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        },
    ]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            content = _request_json(settings, messages, job_id)
            return _validate_results(_extract_object(content), targets)
        except (AsmrDubberError, httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"第 {attempt + 1} 次严格修正：只返回 {target_ids}，保持顺序，"
                            "每个非空 ja 必须引用本窗口已有 evidence_ids。"
                        ),
                    }
                )
                time.sleep(min(4.0, 2 ** (attempt - 1) + random.random()))
    raise AsmrDubberError(f"多 ASR 校对在 3 次尝试后失败：{last_error}") from last_error


def _evidence_range(window: ReviewWindow, selected_ids: list[str]) -> tuple[float, float]:
    selected = [item for item in window.evidence if item.id in set(selected_ids)]
    by_source: dict[str, list[Evidence]] = {}
    for item in selected:
        by_source.setdefault(item.source, []).append(item)
    starts = [min(item.start for item in items) for items in by_source.values()]
    ends = [max(item.end for item in items) for items in by_source.values()]
    if not starts or not ends:
        return window.start, window.end
    return statistics.median(starts), statistics.median(ends)


def review_transcriptions(
    transcriptions: list[tuple[str, list[Sentence]]],
    settings: ProjectSettings,
    report_path: Path,
    progress: Progress | None = None,
) -> list[Sentence]:
    """Resolve several timed ASR hypotheses while keeping timestamps evidence-bound."""
    windows = _build_windows(transcriptions, settings.asr_review_max_drift_seconds)
    if not windows:
        raise AsmrDubberError("多 ASR 校对没有可比较的候选窗口。")
    reviewed: dict[str, tuple[str, list[str], float]] = {}
    chunk_size = 24
    total = math.ceil(len(windows) / chunk_size)
    for chunk_index, start in enumerate(range(0, len(windows), chunk_size), start=1):
        targets = windows[start : start + chunk_size]
        context = windows[max(0, start - 2) : min(len(windows), start + chunk_size + 2)]
        if progress:
            progress(f"大模型交叉校对：{chunk_index}/{total}", chunk_index - 1, total)
        reviewed.update(
            _review_chunk(
                context,
                targets,
                settings,
                f"asr-review-{chunk_index}-{int(time.time())}",
            )
        )

    sentences: list[Sentence] = []
    report_results: list[dict[str, Any]] = []
    for window in windows:
        text, selected_ids, confidence = reviewed[window.id]
        start, end = _evidence_range(window, selected_ids)
        if text and end > start:
            sentences.append(
                Sentence(
                    id=f"s{len(sentences) + 1:06d}",
                    start_seconds=max(0.0, start),
                    end_seconds=end,
                    ja_text=text,
                )
            )
        report_results.append(
            {
                "window_id": window.id,
                "ja": text,
                "evidence_ids": selected_ids,
                "confidence": confidence,
                "computed_time": [start, end],
                "candidates": _window_payload(window)["candidates"],
            }
        )
    if not sentences:
        raise AsmrDubberError("多 ASR 校对没有保留任何可信日语句子。")
    sentences.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    for index, sentence in enumerate(sentences, start=1):
        sentence.id = f"s{index:06d}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"results": report_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if progress:
        progress(f"多 ASR 校对完成：保留 {len(sentences)} 句", total, total)
    return sentences
