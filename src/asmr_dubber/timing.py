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


def plan_dubbing_timing(
    sentences: Iterable[Sentence],
    *,
    offset_ms: int = 0,
    max_auto_speed: float = 1.2,
    durations: Mapping[str, float] | None = None,
    mode: DubbingTimingMode = "fit_window",
) -> list[DubbingTiming]:
    """Plan starts and minimal tempo changes for available Chinese clips.

    Only enabled sentences with Chinese text and a known positive TTS duration
    participate. This means a disabled, untranslated, or not-yet-synthesized
    row never shortens the preceding clip's available window.
    """

    if not math.isfinite(max_auto_speed) or not 1.0 <= max_auto_speed <= MAX_CHINESE_AUTO_SPEED:
        raise ValueError(f"max_auto_speed must be between 1.0 and {MAX_CHINESE_AUTO_SPEED}")
    if mode not in {"fit_window", "sequential", "fit_then_shift"}:
        raise ValueError(f"unsupported dubbing timing mode: {mode}")

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
    requested_starts = [dubbing_start_seconds(sentence, offset_ms) for sentence, _ in candidates]
    if mode == "fit_then_shift":
        return _plan_fit_then_shift(candidates, requested_starts, max_auto_speed)
    if mode == "sequential":
        starts: list[float] = []
        previous_end = 0.0
        for requested, (_sentence, duration) in zip(requested_starts, candidates, strict=True):
            start = max(requested, previous_end)
            starts.append(start)
            previous_end = start + duration
    else:
        starts = requested_starts
    planned: list[DubbingTiming] = []
    for index, ((sentence, duration), start) in enumerate(zip(candidates, starts, strict=True)):
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        speed = 1.0
        if mode == "fit_window" and next_start is not None:
            available = next_start - start
            required = duration / available if available > 0.0 else math.inf
            speed = min(max(required, 1.0), max_auto_speed)
        effective_duration = duration / speed
        remaining_overlap = (
            max(0.0, start + effective_duration - next_start) if next_start is not None else 0.0
        )
        planned.append(
            DubbingTiming(
                sentence_id=sentence.id,
                start_seconds=start,
                original_duration_seconds=duration,
                speed_factor=speed,
                effective_duration_seconds=effective_duration,
                next_start_seconds=next_start,
                remaining_overlap_seconds=remaining_overlap,
            )
        )
    return planned


def _plan_fit_then_shift(
    candidates: list[tuple[Sentence, float]],
    requested_starts: list[float],
    max_auto_speed: float,
) -> list[DubbingTiming]:
    """Speed a line up to fit before the next line's own start; only then delay it.

    ``fit_window`` keeps every line on its source start and lets long lines overlap
    the next one; ``sequential`` never overlaps but lets one long line push every
    later line back, so the delay accumulates over a track.  Here each line first
    uses the allowed speed-up to end before the next line's requested start.  If it
    still cannot, the next line waits for it, and that next line in turn speeds up
    to win the time back, so a single long line does not delay the rest.
    """
    planned: list[DubbingTiming] = []
    previous_end = 0.0
    for index, ((sentence, duration), requested) in enumerate(
        zip(candidates, requested_starts, strict=True)
    ):
        start = max(requested, previous_end)
        next_requested = requested_starts[index + 1] if index + 1 < len(candidates) else None
        speed = 1.0
        if next_requested is not None:
            available = next_requested - start
            required = duration / available if available > 0.0 else math.inf
            speed = min(max(required, 1.0), max_auto_speed)
        effective = duration / speed
        previous_end = start + effective
        planned.append(
            DubbingTiming(
                sentence_id=sentence.id,
                start_seconds=start,
                original_duration_seconds=duration,
                speed_factor=speed,
                effective_duration_seconds=effective,
                next_start_seconds=next_requested,
                remaining_overlap_seconds=0.0,
            )
        )
    return planned
