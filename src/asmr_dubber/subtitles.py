from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from .errors import ProjectError
from .models import Sentence
from .storage import atomic_write_text
from .timing import DubbingTiming, DubbingTimingMode, plan_dubbing_timing

SubtitleLanguage = Literal["bilingual", "zh", "source"]
SubtitleTimeline = Literal["source", "dubbing"]


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _lrc_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    minutes, remainder = divmod(centiseconds, 6_000)
    secs, fraction = divmod(remainder, 100)
    return f"[{minutes:02d}:{secs:02d}.{fraction:02d}]"


def _clean_line(value: str) -> str:
    return " ".join(str(value).replace("\ufeff", "").split())


def _wrap_line(value: str, width: int) -> list[str]:
    cleaned = _clean_line(value)
    if not cleaned:
        return []
    return textwrap.wrap(
        cleaned,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [cleaned]


def _subtitle_lines(
    sentence: Sentence,
    language: SubtitleLanguage,
    maximum_chars: int,
) -> list[str]:
    source = _wrap_line(sentence.source_text, maximum_chars)
    chinese = _wrap_line(sentence.zh_text, maximum_chars)
    if language in {"bilingual", "zh"} and not chinese:
        raise ProjectError(f"句子 {sentence.id} 没有中文，无法生成所选字幕。")
    if language == "source":
        if not source:
            raise ProjectError(f"句子 {sentence.id} 没有原文，请先识别或导入原文字幕。")
        return source
    if language == "zh":
        return chinese
    return [*source, *chinese]


def _subtitle_range(
    sentence: Sentence,
    *,
    timeline: SubtitleTimeline,
    dubbing_timing: DubbingTiming | None,
    text: str,
    minimum_duration: float,
    maximum_cps: float,
) -> tuple[float, float]:
    if timeline == "dubbing" and dubbing_timing is not None:
        start = dubbing_timing.start_seconds
        end = start + dubbing_timing.effective_duration_seconds
    else:
        start = sentence.start_seconds
        end = sentence.end_seconds
    visible_characters = sum(not character.isspace() for character in text)
    readable_duration = visible_characters / maximum_cps if maximum_cps > 0 else 0.0
    end = max(end, start + minimum_duration, start + readable_duration)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ProjectError(f"句子 {sentence.id} 的字幕时间范围无效。")
    return start, end


def write_subtitle_files(
    sentences: Iterable[Sentence],
    output_dir: Path,
    language: SubtitleLanguage,
    *,
    timeline: SubtitleTimeline = "source",
    maximum_chars: int = 22,
    minimum_duration: float = 1.0,
    maximum_cps: float = 18.0,
    chinese_dubbing_offset_ms: int = 0,
    chinese_max_auto_speed: float = 1.2,
    chinese_dubbing_timing_mode: DubbingTimingMode = "fit_window",
    media_duration_seconds: float | None = None,
    warnings: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write readable, atomic UTF-8 SRT and LRC subtitle files."""

    if language not in {"bilingual", "zh", "source"}:
        raise ProjectError(f"未知字幕语言：{language}")
    if timeline not in {"source", "dubbing"}:
        raise ProjectError(f"未知字幕时间轴：{timeline}")
    included = [sentence for sentence in sentences if sentence.enabled]
    if not included:
        raise ProjectError("没有可生成字幕的已启用句子。")
    timing_by_id = {
        timing.sentence_id: timing
        for timing in plan_dubbing_timing(
            included,
            offset_ms=chinese_dubbing_offset_ms,
            max_auto_speed=chinese_max_auto_speed,
            mode=chinese_dubbing_timing_mode,
        )
    }

    if media_duration_seconds is not None and (
        not math.isfinite(media_duration_seconds) or media_duration_seconds <= 0
    ):
        raise ProjectError("字幕对应媒体的时长无效。")
    included.sort(
        key=lambda sentence: (
            timing_by_id[sentence.id].start_seconds
            if timeline == "dubbing" and sentence.id in timing_by_id
            else sentence.start_seconds,
            sentence.id,
        )
    )
    # Validate the complete export before replacing either existing subtitle file.
    entries = []
    for sentence in included:
        lines = _subtitle_lines(sentence, language, maximum_chars)
        timing = timing_by_id.get(sentence.id) if timeline == "dubbing" else None
        start = timing.start_seconds if timing else sentence.start_seconds
        original_end = start + timing.effective_duration_seconds if timing else sentence.end_seconds
        entries.append((sentence, lines, start, original_end))
    srt_blocks: list[str] = []
    lrc_lines: list[str] = []
    for index, (sentence, lines, _start, original_end) in enumerate(entries, start=1):
        start, end = _subtitle_range(
            sentence,
            timeline=timeline,
            dubbing_timing=timing_by_id.get(sentence.id),
            text="".join(lines),
            minimum_duration=minimum_duration,
            maximum_cps=maximum_cps,
        )
        requested_end = end
        if index < len(entries):
            next_start = entries[index][2]
            # Preserve genuine overlap; only prevent readability padding from creating it.
            end = min(end, max(original_end, next_start))
        if media_duration_seconds is not None:
            end = min(end, media_duration_seconds)
        if end <= start:
            raise ProjectError(f"句子 {sentence.id} 的字幕起点超出媒体范围。")
        if warnings is not None and end < requested_end:
            warnings.append(f"句子 {sentence.id} 的字幕可读时长受下一句或媒体末尾限制。")
        srt_blocks.append(
            f"{index}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n" + "\n".join(lines)
        )
        timestamp = _lrc_timestamp(start)
        lrc_lines.extend(f"{timestamp}{line}" for line in lines)

    srt = output_dir / f"subtitles_{language}.srt"
    lrc = output_dir / f"subtitles_{language}.lrc"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(srt, "\n\n".join(srt_blocks) + "\n")
    atomic_write_text(lrc, "\n".join(lrc_lines) + "\n")
    return srt, lrc
