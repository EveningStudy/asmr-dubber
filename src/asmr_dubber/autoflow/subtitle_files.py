"""SRT/LRC text shifting and merging."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .errors import VideoPreparerError
from .fileio import atomic_write_text


def shift_srt_text(text: str, offset_ms: int) -> str:
    timestamp = re.compile(r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})")

    def replace(match: re.Match[str]) -> str:
        hours, minutes, seconds, millis = map(int, match.groups())
        total = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis + offset_ms
        total = max(0, total)
        out_hours, remainder = divmod(total, 3_600_000)
        out_minutes, remainder = divmod(remainder, 60_000)
        out_seconds, out_millis = divmod(remainder, 1000)
        return f"{out_hours:02d}:{out_minutes:02d}:{out_seconds:02d},{out_millis:03d}"

    lines = [timestamp.sub(replace, line) if "-->" in line else line for line in text.splitlines()]
    return "\n".join(lines) + "\n"


def shift_lrc_text(text: str, offset_ms: int) -> str:
    timestamp = re.compile(r"\[(\d+):(\d{2})(?:[.](\d{2,3}))?\]")

    def replace(match: re.Match[str]) -> str:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction_text = match.group(3) or "00"
        fraction_ms = int(fraction_text.ljust(3, "0")[:3])
        total = (minutes * 60 + seconds) * 1000 + fraction_ms + offset_ms
        total = max(0, total)
        out_minutes, remainder = divmod(total, 60_000)
        out_seconds, out_millis = divmod(remainder, 1000)
        return f"[{out_minutes:02d}:{out_seconds:02d}.{out_millis // 10:02d}]"

    return timestamp.sub(replace, text).rstrip("\n") + "\n"


def _srt_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for raw in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [line.rstrip("\r") for line in raw.splitlines()]
        if any("-->" in line for line in lines):
            blocks.append(lines)
    return blocks


def _srt_timestamp_ms(value: str) -> int:
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if match is None:
        raise ValueError(value)
    hours, minutes, seconds, millis = map(int, match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _format_srt_timestamp(milliseconds: int) -> str:
    total = max(0, int(milliseconds))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def combine_srt_files(
    entries: Sequence[tuple[Path, int] | tuple[Path, int, int]],
    destination: Path,
    *,
    final_offset_ms: int = 0,
) -> Path:
    """Merge SRT files, shifting and re-numbering cues in source order."""

    output: list[str] = []
    sequence = 1
    timing_pattern = re.compile(
        r"^(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
        r"(\d{2,}:\d{2}:\d{2}[,.]\d{3})(.*)$"
    )
    for entry in entries:
        source, offset_ms = entry[0], entry[1]
        clip_end_ms = entry[2] if len(entry) == 3 else None
        if not source.is_file():
            raise VideoPreparerError(f"找不到字幕文件：{source}")
        text = source.read_text(encoding="utf-8-sig")
        for block in _srt_blocks(text):
            timing_index = next(
                (index for index, line in enumerate(block) if "-->" in line),
                None,
            )
            if timing_index is None:
                continue
            shifted = shift_srt_text(
                block[timing_index] + "\n",
                int(offset_ms) + int(final_offset_ms),
            ).strip("\r\n")
            timing_match = timing_pattern.match(shifted)
            if timing_match is not None and clip_end_ms is not None:
                cue_start = _srt_timestamp_ms(timing_match.group(1))
                cue_end = _srt_timestamp_ms(timing_match.group(2))
                absolute_limit = int(clip_end_ms) + int(final_offset_ms)
                if cue_start >= absolute_limit:
                    continue
                cue_end = min(cue_end, absolute_limit)
                if cue_end <= cue_start:
                    continue
                shifted = (
                    f"{_format_srt_timestamp(cue_start)} --> "
                    f"{_format_srt_timestamp(cue_end)}{timing_match.group(3)}"
                )
            payload = [str(sequence), shifted, *block[timing_index + 1 :]]
            output.extend(payload)
            output.append("")
            sequence += 1
    if not output:
        raise VideoPreparerError("字幕文件中没有可合并的 SRT 时间轴。")
    atomic_write_text(destination, "\n".join(output).rstrip() + "\n")
    return destination


def combine_lrc_files(
    entries: Sequence[tuple[Path, int]],
    destination: Path,
    *,
    final_offset_ms: int = 0,
) -> Path:
    """Merge LRC files while retaining metadata from the first file only."""

    timestamp_line = re.compile(r"\[\d+:\d{2}(?:[.]\d{1,3})?\]")
    headers: list[str] = []
    timed_lines: list[str] = []
    for entry_index, (source, offset_ms) in enumerate(entries):
        if not source.is_file():
            raise VideoPreparerError(f"找不到字幕文件：{source}")
        text = source.read_text(encoding="utf-8-sig")
        shifted = shift_lrc_text(text, int(offset_ms) + int(final_offset_ms))
        for line in shifted.splitlines():
            if timestamp_line.search(line):
                timed_lines.append(line)
            elif entry_index == 0 and line.strip():
                headers.append(line)
    if not timed_lines:
        raise VideoPreparerError("字幕文件中没有可合并的 LRC 时间轴。")
    atomic_write_text(destination, "\n".join((*headers, *timed_lines)) + "\n")
    return destination
