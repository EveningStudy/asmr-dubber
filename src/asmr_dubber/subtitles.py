from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from .errors import ProjectError
from .models import Sentence
from .storage import atomic_write_text
from .timing import DubbingTiming, DubbingTimingMode, plan_dubbing_timing

SubtitleLanguage = Literal["bilingual", "zh", "source"]
SubtitleTimeline = Literal["source", "dubbing"]

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_NUMBER = re.compile(
    r"[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-−]?\d+|[×xX]10(?:\^[+\-−]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+))?"
)


class _NumberAwareWrapper(textwrap.TextWrapper):
    """Keep a number together when it can fit on the next line."""

    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        space_left = width - cur_len
        # Python 3.12's default appends an empty chunk when a line is full,
        # which can prevent the wrapper from dropping trailing whitespace.
        if space_left <= 0 and cur_line:
            return
        chunk = reversed_chunks[-1]
        for number in _NUMBER.finditer(chunk):
            if number.start() < space_left < number.end() and len(number.group()) <= width:
                if number.start():
                    cur_line.append(chunk[: number.start()])
                    reversed_chunks[-1] = chunk[number.start() :]
                    return
                if cur_line:
                    return
                break
        # A number wider than a line still needs the normal hard-break fallback.
        super()._handle_long_word(reversed_chunks, cur_line, cur_len, width)


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
    wrapper = _NumberAwareWrapper if _CJK.search(cleaned) else textwrap.TextWrapper
    return wrapper(
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ).wrap(cleaned) or [cleaned]


def _escape_srt_text(value: str) -> str:
    """Keep literal angle brackets visible in every player.

    SRT has no escape syntax: players and FFmpeg's burn-in treat ``<...>`` as a
    tag (``<breath>`` disappears, ``< b >`` turns bold) and show ``&lt;`` as is.
    Full-width brackets read the same in CJK subtitles and are never markup.
    """
    return value.replace("<", "＜").replace(">", "＞")


def _visible_char_count(value: str) -> int:
    return sum(not character.isspace() for character in value)


def _subtitle_lines(
    sentence: Sentence,
    language: SubtitleLanguage,
    maximum_chars: int,
    *,
    target_language: str = "zh",
    allow_source_fallback: bool = False,
) -> tuple[list[str], list[str], int, bool]:
    source_raw = _clean_line(sentence.source_text)
    target_raw = _clean_line(sentence.zh_text)
    target_width = max(42, maximum_chars * 2) if target_language == "en" else maximum_chars
    source = _wrap_line(sentence.source_text, maximum_chars)
    chinese = _wrap_line(sentence.zh_text, target_width)
    if language in {"bilingual", "zh"} and not chinese:
        if allow_source_fallback:
            srt_lines = [_escape_srt_text(line) for line in source]
            lrc_items = [source_raw] if source_raw else []
            return srt_lines, lrc_items, _visible_char_count(source_raw), True
        raise ProjectError(f"句子 {sentence.id} 没有中文，无法生成所选字幕。")
    if language == "source":
        if not source:
            raise ProjectError(f"句子 {sentence.id} 没有原文，请先识别或导入原文字幕。")
        srt_lines = [_escape_srt_text(line) for line in source]
        return srt_lines, [source_raw], _visible_char_count(source_raw), False
    if language == "zh":
        srt_lines = [_escape_srt_text(line) for line in chinese]
        return srt_lines, [target_raw], _visible_char_count(target_raw), False
    srt_lines = [_escape_srt_text(line) for line in (*source, *chinese)]
    lrc_items = [part for part in (source_raw, target_raw) if part]
    cps_chars = max(_visible_char_count(source_raw), _visible_char_count(target_raw))
    return srt_lines, lrc_items, cps_chars, False


def _subtitle_range(
    sentence: Sentence,
    *,
    timeline: SubtitleTimeline,
    dubbing_timing: DubbingTiming | None,
    visible_characters: int,
    minimum_duration: float,
    maximum_cps: float,
) -> tuple[float, float]:
    if timeline == "dubbing" and dubbing_timing is not None:
        start = dubbing_timing.start_seconds
        end = start + dubbing_timing.effective_duration_seconds
    else:
        start = sentence.start_seconds
        end = sentence.end_seconds
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
    target_language: str = "zh",
    chinese_dubbing_offset_ms: int = 0,
    chinese_max_auto_speed: float = 1.2,
    chinese_dubbing_timing_mode: DubbingTimingMode = "fit_window",
    dubbing_durations: Mapping[str, float] | None = None,
    media_duration_seconds: float | None = None,
    warnings: list[str] | None = None,
    dubbing_obstacles: Iterable[tuple[float, float]] | None = None,
    dubbing_end_seconds: float | None = None,
) -> tuple[Path, Path]:
    """Write readable, atomic UTF-8 SRT and LRC subtitle files."""

    if language not in {"bilingual", "zh", "source"}:
        raise ProjectError(f"未知字幕语言：{language}")
    if timeline not in {"source", "dubbing"}:
        raise ProjectError(f"未知字幕时间轴：{timeline}")
    included = [sentence for sentence in sentences if sentence.enabled]
    if not included:
        raise ProjectError("没有可生成字幕的已启用句子。")
    has_translation = any(bool(_clean_line(sentence.zh_text)) for sentence in included)
    timing_by_id = {
        timing.sentence_id: timing
        for timing in plan_dubbing_timing(
            included,
            offset_ms=chinese_dubbing_offset_ms,
            max_auto_speed=chinese_max_auto_speed,
            durations=dubbing_durations,
            mode=chinese_dubbing_timing_mode,
            obstacles=dubbing_obstacles,
            end_seconds=dubbing_end_seconds,
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
    collected_warnings: list[str] = []
    for sentence in included:
        lines, lrc_items, cps_chars, used_fallback = _subtitle_lines(
            sentence,
            language,
            maximum_chars,
            target_language=target_language,
            allow_source_fallback=has_translation,
        )
        if not lines:
            collected_warnings.append(f"句子 {sentence.id} 缺少译文与原文，已跳过字幕。")
            continue
        if used_fallback:
            collected_warnings.append(f"句子 {sentence.id} 缺少译文，字幕已回退为原文。")
        timing = timing_by_id.get(sentence.id) if timeline == "dubbing" else None
        start = timing.start_seconds if timing else sentence.start_seconds
        original_end = start + timing.effective_duration_seconds if timing else sentence.end_seconds
        entries.append((sentence, lines, lrc_items, cps_chars, start, original_end))
    if not entries:
        raise ProjectError("没有可生成字幕的已启用句子。")
    effective_media_duration = media_duration_seconds
    if media_duration_seconds is not None and timeline == "dubbing":
        max_planned_end = max(orig_end for _s, _lines, _lrc, _cps, _st, orig_end in entries)
        effective_media_duration = max(media_duration_seconds, max_planned_end)
    srt_blocks: list[str] = []
    lrc_lines: list[str] = []
    for index, (sentence, lines, lrc_items, cps_chars, _start, original_end) in enumerate(
        entries, start=1
    ):
        start, end = _subtitle_range(
            sentence,
            timeline=timeline,
            dubbing_timing=timing_by_id.get(sentence.id),
            visible_characters=cps_chars,
            minimum_duration=minimum_duration,
            maximum_cps=maximum_cps,
        )
        requested_end = end
        if index < len(entries):
            next_start = entries[index][4]
            # Preserve genuine overlap; only prevent readability padding from creating it.
            end = min(end, max(original_end, next_start))
        if effective_media_duration is not None:
            end = min(end, effective_media_duration)
        if end <= start:
            raise ProjectError(f"句子 {sentence.id} 的字幕起点超出媒体范围。")
        if end < requested_end:
            collected_warnings.append(f"句子 {sentence.id} 的字幕可读时长受下一句或媒体末尾限制。")
        srt_blocks.append(
            f"{index}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n" + "\n".join(lines)
        )
        timestamp = _lrc_timestamp(start)
        lrc_lines.extend(f"{timestamp}{line}" for line in lrc_items)

    if warnings is not None:
        warnings.extend(collected_warnings)

    srt = output_dir / f"subtitles_{language}.srt"
    lrc = output_dir / f"subtitles_{language}.lrc"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(srt, "\n\n".join(srt_blocks) + "\n")
    atomic_write_text(lrc, "\n".join(lrc_lines) + "\n")
    return srt, lrc
