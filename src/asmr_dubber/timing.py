from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .models import Sentence


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
) -> list[DubbingTiming]:
    """Plan starts and minimal tempo changes for available Chinese clips.

    Only enabled sentences with Chinese text and a known positive TTS duration
    participate. This means a disabled, untranslated, or not-yet-synthesized
    row never shortens the preceding clip's available window.
    """

    if not math.isfinite(max_auto_speed) or not 1.0 <= max_auto_speed <= 2.0:
        raise ValueError("max_auto_speed must be between 1.0 and 2.0")

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
    starts = [dubbing_start_seconds(sentence, offset_ms) for sentence, _ in candidates]
    planned: list[DubbingTiming] = []
    for index, ((sentence, duration), start) in enumerate(zip(candidates, starts, strict=True)):
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        speed = 1.0
        if next_start is not None:
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
