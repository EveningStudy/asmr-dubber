from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from .errors import ProjectError
from .models import Sentence

_TIMESTAMP = r"(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
_ARROW_LINE = re.compile(rf"^\s*(?P<start>{_TIMESTAMP})\s*-->\s*(?P<end>{_TIMESTAMP})(?:\s+.*)?$")
_LRC_TAG = re.compile(r"\[(?P<time>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?)\]")
_ASS_OVERRIDE = re.compile(r"\{[^{}]*\}")
_HTML_TAG = re.compile(r"<[^>]*>")

TranscriptLanguage = Literal["ja", "zh"]


@dataclass(frozen=True)
class TranscriptImport:
    sentences: list[Sentence]
    source_format: str
    timed: bool
    source_text: str
    language: TranscriptLanguage


def _decode_script(path: Path) -> str:
    if not path.is_file():
        raise ProjectError(f"台本/字幕文件不存在：{path}")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ProjectError("台本/字幕文件超过 20 MB；请确认没有误选音频或视频文件。")
    payload = path.read_bytes()
    encodings = ["utf-8-sig", "utf-16", "cp932", "shift_jis", "gb18030"]
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")) or payload.count(b"\x00") > len(payload) // 8:
        encodings = ["utf-16", "utf-8-sig", "cp932", "shift_jis", "gb18030"]
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ProjectError("无法识别台本/字幕编码；请另存为 UTF-8 后重试。")


def _seconds(value: str) -> float:
    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    try:
        if len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            raise ValueError
        result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError as exc:
        raise ProjectError(f"无法解析字幕时间：{value}") from exc
    if not math.isfinite(result) or result < 0:
        raise ProjectError(f"字幕时间无效：{value}")
    return result


def _clean_text(value: str) -> str:
    text = value.replace("\\N", "\n").replace("\\n", "\n")
    text = _ASS_OVERRIDE.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = html.unescape(text).replace("\ufeff", "")
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _arrow_cues(text: str) -> list[tuple[float, float, str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[tuple[float, float, str]] = []
    index = 0
    while index < len(lines):
        match = _ARROW_LINE.match(lines[index])
        if match is None:
            index += 1
            continue
        start = _seconds(match.group("start"))
        end = _seconds(match.group("end"))
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip():
            if _ARROW_LINE.match(lines[index]):
                break
            content.append(lines[index])
            index += 1
        cleaned = _clean_text("\n".join(content))
        if cleaned:
            cues.append((start, end, cleaned))
    return cues


def _ass_cues(text: str) -> list[tuple[float, float, str]]:
    in_events = False
    fields: list[str] = []
    cues: list[tuple[float, float, str]] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.casefold() == "[events]"
            continue
        if not in_events:
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip().casefold() == "format":
            fields = [item.strip().casefold() for item in value.split(",")]
            continue
        if key.strip().casefold() != "dialogue":
            continue
        active_fields = fields or [
            "layer",
            "start",
            "end",
            "style",
            "name",
            "marginl",
            "marginr",
            "marginv",
            "effect",
            "text",
        ]
        values = [item.strip() for item in value.split(",", len(active_fields) - 1)]
        if len(values) != len(active_fields):
            continue
        row = dict(zip(active_fields, values, strict=True))
        cleaned = _clean_text(row.get("text", ""))
        if cleaned and row.get("start") and row.get("end"):
            cues.append((_seconds(row["start"]), _seconds(row["end"]), cleaned))
    return cues


def _lrc_cues(text: str, duration_seconds: float) -> list[tuple[float, float, str]]:
    starts: list[tuple[float, str]] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        matches = list(_LRC_TAG.finditer(line))
        if not matches:
            continue
        cleaned = _clean_text(_LRC_TAG.sub("", line))
        if not cleaned:
            continue
        starts.extend((_seconds(match.group("time")), cleaned) for match in matches)
    starts.sort(key=lambda item: item[0])
    if not starts:
        return []
    gaps = [right[0] - left[0] for left, right in pairwise(starts) if right[0] > left[0]]
    typical_gap = sorted(gaps)[len(gaps) // 2] if gaps else 5.0
    last_duration = min(15.0, max(1.0, typical_gap))
    cues: list[tuple[float, float, str]] = []
    for index, (start, content) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else start + last_duration
        cues.append((start, min(duration_seconds, end), content))
    return cues


def _plain_lines(text: str) -> list[str]:
    lines = [_clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) == 1:
        split = re.split(r"(?<=[。！？!?])\s*", lines[0])
        lines = [item.strip() for item in split if item.strip()]
    if not lines:
        raise ProjectError("台本中没有可导入的非空文字。")
    if len(lines) > 10_000:
        raise ProjectError("台本超过 10,000 行；请先按作品或音轨拆分。")
    return lines


def _sentence(
    *,
    sentence_id: str,
    start_seconds: float,
    end_seconds: float,
    text: str,
    language: TranscriptLanguage,
) -> Sentence:
    if language == "zh":
        return Sentence(
            id=sentence_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            ja_text="",
            zh_text=text,
            status="translated",
        )
    return Sentence(
        id=sentence_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        ja_text=text,
    )


def _estimated_sentences(
    lines: list[str],
    duration_seconds: float,
    language: TranscriptLanguage,
) -> list[Sentence]:
    weights = [max(1, sum(not character.isspace() for character in line)) for line in lines]
    total = sum(weights)
    elapsed_weight = 0
    sentences: list[Sentence] = []
    for index, (line, weight) in enumerate(zip(lines, weights, strict=True), start=1):
        start = duration_seconds * elapsed_weight / total
        elapsed_weight += weight
        end = duration_seconds * elapsed_weight / total
        if index == len(lines):
            end = duration_seconds
        sentences.append(
            _sentence(
                sentence_id=f"s{index:06d}",
                start_seconds=start,
                end_seconds=max(start + 0.001, end),
                text=line,
                language=language,
            )
        )
    return sentences


def _timed_sentences(
    cues: list[tuple[float, float, str]],
    duration_seconds: float,
    language: TranscriptLanguage,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    for start, end, content in sorted(cues, key=lambda item: (item[0], item[1])):
        start = min(duration_seconds, max(0.0, start))
        end = min(duration_seconds, max(0.0, end))
        if end <= start or not content:
            continue
        sentences.append(
            _sentence(
                sentence_id=f"s{len(sentences) + 1:06d}",
                start_seconds=start,
                end_seconds=end,
                text=content,
                language=language,
            )
        )
    if not sentences:
        raise ProjectError("字幕中没有位于当前音频时长内的有效台词。")
    if len(sentences) > 10_000:
        raise ProjectError("字幕超过 10,000 条；请先按作品或音轨拆分。")
    return sentences


def parse_transcript(
    *,
    duration_seconds: float,
    path: str | Path | None = None,
    pasted_text: str = "",
    language: TranscriptLanguage = "ja",
) -> TranscriptImport:
    if language not in {"ja", "zh"}:
        raise ProjectError("台本语言必须是日语或中文。")
    source_path = Path(path).expanduser().resolve() if path else None
    file_text = _decode_script(source_path) if source_path is not None else ""
    text = pasted_text.strip() or file_text
    if not text.strip():
        raise ProjectError("请选择台本/字幕文件，或粘贴台本文字。")
    suffix = source_path.suffix.casefold() if source_path is not None else ""

    if suffix in {".ass", ".ssa"} or "[events]" in text.casefold():
        cues = _ass_cues(text)
        source_format = "ASS/SSA"
    elif suffix == ".lrc" or _LRC_TAG.search(text):
        cues = _lrc_cues(text, duration_seconds)
        source_format = "LRC"
    elif suffix in {".srt", ".vtt"} or "-->" in text:
        cues = _arrow_cues(text)
        source_format = "VTT" if suffix == ".vtt" or text.lstrip().startswith("WEBVTT") else "SRT"
    else:
        cues = []
        source_format = "纯文本"

    if cues:
        return TranscriptImport(
            sentences=_timed_sentences(cues, duration_seconds, language),
            source_format=source_format,
            timed=True,
            source_text=text,
            language=language,
        )
    if source_format != "纯文本":
        raise ProjectError(f"没有从 {source_format} 文件中解析到有效时间轴。")
    lines = _plain_lines(text)
    return TranscriptImport(
        sentences=_estimated_sentences(lines, duration_seconds, language),
        source_format=source_format,
        timed=False,
        source_text=text,
        language=language,
    )
