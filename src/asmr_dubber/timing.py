from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .constants import MAX_CHINESE_AUTO_SPEED
from .models import Sentence

DubbingTimingMode = Literal["fit_window", "sequential", "fit_then_shift"]


@dataclass(frozen=True)
class DubbingTiming:
    sentence_id: str
    start_seconds: float
    original_duration_seconds: float
    speed_factor: float
    effective_duration_seconds: float
    next_start_seconds: float | None
    remaining_overlap_seconds: float


def dubbing_start_seconds(sentence: Sentence, offset_ms: int = 0) -> float:
    """Return the shifted dubbing start without allowing audio before zero."""

    return max(0.0, sentence.start_seconds + offset_ms / 1000.0)


def _normalize_obstacles(
    obstacles: Iterable[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    if obstacles is None:
        return []
    cleaned = sorted(
        (max(0.0, float(lo)), float(hi))
        for lo, hi in obstacles
        if math.isfinite(lo) and math.isfinite(hi) and hi > max(0.0, float(lo)) + 1e-4
    )
    merged: list[list[float]] = []
    for lo, hi in cleaned:
        if merged and lo <= merged[-1][1] + 0.01:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _obstacle_adjusted_start(
    start: float,
    source_start: float,
    obstacles: list[tuple[float, float]],
) -> float:
    """Prevent a negative dubbing offset from backing a line onto a kept kiss/breath."""
    for lo, hi in obstacles:
        if lo < source_start - 1e-3 and lo < start < hi:
            start = hi
    return start


def _next_boundary_seconds(
    start: float,
    source_start: float,
    next_start: float | None,
    obstacles: list[tuple[float, float]],
) -> float | None:
    """Earliest hard ceiling: either the next dubbed line or a kept original sound."""
    ceiling = next_start
    for lo, _hi in obstacles:
        if lo >= max(start + 0.02, source_start + 0.01):
            if ceiling is None or lo < ceiling:
                ceiling = lo
            break
    return ceiling


def plan_dubbing_timing(
    sentences: Iterable[Sentence],
    *,
    offset_ms: int = 0,
    max_auto_speed: float = 1.2,
    min_auto_speed: float = 1.0,
    durations: Mapping[str, float] | None = None,
    mode: DubbingTimingMode = "fit_window",
    obstacles: Iterable[tuple[float, float]] | None = None,
    end_seconds: float | None = None,
) -> list[DubbingTiming]:
    """Plan starts and minimal tempo changes for available Chinese clips.

    Only enabled sentences with Chinese text and a known positive TTS duration
    participate. In replacement mode, ``obstacles`` supplies preserved original-voice
    intervals (e.g. kisses, breaths, kept sound lines) so a dub never talks over them,
    and ``end_seconds`` (the end of the recording) lets the last line speed up to stay
    inside it instead of running past the end at its natural pace.
    """

    if not math.isfinite(max_auto_speed) or not 1.0 <= max_auto_speed <= MAX_CHINESE_AUTO_SPEED:
        raise ValueError(f"max_auto_speed must be between 1.0 and {MAX_CHINESE_AUTO_SPEED}")
    if not math.isfinite(min_auto_speed) or not 0.75 <= min_auto_speed <= 1.0:
        raise ValueError("min_auto_speed must be between 0.75 and 1.0")
    if mode not in {"fit_window", "sequential", "fit_then_shift"}:
        raise ValueError(f"unsupported dubbing timing mode: {mode}")

    kept_obstacles = _normalize_obstacles(obstacles)
    candidates: list[tuple[Sentence, float]] = []
    for sentence in sentences:
        if not sentence.enabled or not sentence.zh_text:
            continue
        duration = (
            durations.get(sentence.id) if durations is not None else sentence.tts_duration_seconds
        )
        if duration is None:
            continue
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0.0:
            continue
        candidates.append((sentence, duration))

    candidates.sort(key=lambda item: (item[0].start_seconds, item[0].end_seconds, item[0].id))
    requested_starts = [
        _obstacle_adjusted_start(
            dubbing_start_seconds(sentence, offset_ms),
            sentence.start_seconds,
            kept_obstacles,
        )
        for sentence, _ in candidates
    ]
    if mode == "fit_then_shift":
        return _plan_fit_then_shift(
            candidates,
            requested_starts,
            max_auto_speed,
            min_auto_speed,
            kept_obstacles,
            end_seconds,
        )
    if mode == "sequential":
        starts: list[float] = []
        previous_end = 0.0
        for requested, (sentence, duration) in zip(requested_starts, candidates, strict=True):
            start = _obstacle_adjusted_start(
                max(requested, previous_end),
                sentence.start_seconds,
                kept_obstacles,
            )
            starts.append(start)
            previous_end = start + duration
    else:
        starts = requested_starts
    planned: list[DubbingTiming] = []
    for index, ((sentence, duration), start) in enumerate(zip(candidates, starts, strict=True)):
        raw_next = starts[index + 1] if index + 1 < len(starts) else None
        boundary = _next_boundary_seconds(start, sentence.start_seconds, raw_next, kept_obstacles)
        speed = 1.0
        ceiling = _speed_ceiling(boundary, start, end_seconds)
        if mode == "fit_window" and ceiling is not None:
            available = ceiling - start
            required = duration / available if available > 0.0 else math.inf
            speed = min(max(required, min_auto_speed), max_auto_speed)
        effective_duration = duration / speed
        remaining_overlap = (
            max(0.0, start + effective_duration - boundary) if boundary is not None else 0.0
        )
        planned.append(
            DubbingTiming(
                sentence_id=sentence.id,
                start_seconds=start,
                original_duration_seconds=duration,
                speed_factor=speed,
                effective_duration_seconds=effective_duration,
                next_start_seconds=boundary if kept_obstacles else raw_next,
                remaining_overlap_seconds=remaining_overlap,
            )
        )
    return planned


def _speed_ceiling(boundary: float | None, start: float, end_seconds: float | None) -> float | None:
    """The point a line should finish by: the next boundary, else the end of the recording."""
    if boundary is not None:
        return boundary
    return end_seconds if end_seconds is not None and end_seconds > start else None


def _plan_fit_then_shift(
    candidates: list[tuple[Sentence, float]],
    requested_starts: list[float],
    max_auto_speed: float,
    min_auto_speed: float,
    obstacles: list[tuple[float, float]],
    end_seconds: float | None = None,
) -> list[DubbingTiming]:
    """Speed a line up to fit before the next line or kept sound; only then delay."""
    planned: list[DubbingTiming] = []
    previous_end = 0.0
    for index, ((sentence, duration), requested) in enumerate(
        zip(candidates, requested_starts, strict=True)
    ):
        start = _obstacle_adjusted_start(
            max(requested, previous_end),
            sentence.start_seconds,
            obstacles,
        )
        next_requested = requested_starts[index + 1] if index + 1 < len(candidates) else None
        boundary = _next_boundary_seconds(
            start,
            sentence.start_seconds,
            next_requested,
            obstacles,
        )
        speed = 1.0
        ceiling = _speed_ceiling(boundary, start, end_seconds)
        if ceiling is not None:
            available = ceiling - start
            required = duration / available if available > 0.0 else math.inf
            speed = min(max(required, min_auto_speed), max_auto_speed)
        elif min_auto_speed < 1.0:
            own_window = max(0.05, sentence.end_seconds - start)
            speed = min(1.0, max(min_auto_speed, duration / own_window))
        effective = duration / speed
        previous_end = start + effective
        planned.append(
            DubbingTiming(
                sentence_id=sentence.id,
                start_seconds=start,
                original_duration_seconds=duration,
                speed_factor=speed,
                effective_duration_seconds=effective,
                next_start_seconds=boundary if obstacles else next_requested,
                remaining_overlap_seconds=max(0.0, previous_end - boundary)
                if boundary is not None and obstacles and boundary != next_requested
                else 0.0,
            )
        )
    return planned
