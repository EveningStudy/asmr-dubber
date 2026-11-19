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
_SRT_VTT_TIMESTAMP_ARROW = re.compile(
    r"^\s*(?:\d{1,3}:)?\d{2}:\d{2}[.,]\d+\s*-->\s*(?:\d{1,3}:)?\d{2}:\d{2}[.,]\d+(?:\s|$)",
    re.MULTILINE,
)
_LRC_TAG = re.compile(r"\[(?P<time>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?)\]")
_ASS_OVERRIDE = re.compile(r"\{[^{}]*\}")
_HTML_TAG = re.compile(r"<[^>]*>")

TranscriptLanguage = Literal["ja", "en", "zh"]
_Cue = tuple[float, float, str, str | None]


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
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")) or (
        len(payload) >= 2 and payload.count(b"\x00") > len(payload) // 8
    ):
        encodings = ["utf-16", "utf-8-sig", "cp932", "shift_jis", "gb18030"]
    else:
        encodings = ["utf-8-sig", "cp932", "shift_jis", "gb18030"]
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


def _lrc_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 3 and len(parts[2]) == 2 and parts[2].isdigit():
        return _seconds(f"{parts[0]}:{parts[1]}.{parts[2]}")
    return _seconds(value)


def _clean_text(value: str) -> str:
    text = value.replace("\\N", "\n").replace("\\n", "\n")
    text = _ASS_OVERRIDE.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = html.unescape(text).replace("\ufeff", "")
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _clean_plain_line(value: str) -> str:
    text = value.replace("\ufeff", "")
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _arrow_cues(text: str) -> list[_Cue]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[_Cue] = []
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
            cues.append((start, end, cleaned, None))
    return cues


def _ass_cues(text: str) -> list[_Cue]:
    in_events = False
    fields: list[str] = []
    cues: list[_Cue] = []
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
        speaker_id = (row.get("name") or "").strip() or None
        if cleaned and row.get("start") and row.get("end"):
            cues.append((_seconds(row["start"]), _seconds(row["end"]), cleaned, speaker_id))
    return cues


def _lrc_cues(text: str, duration_seconds: float) -> list[_Cue]:
    events: list[tuple[float, str]] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        matches = list(_LRC_TAG.finditer(line))
        if not matches:
            continue
        cleaned = _clean_text(_LRC_TAG.sub("", line))
        events.extend((_lrc_seconds(match.group("time")), cleaned) for match in matches)
    if not events:
        return []
    events.sort(key=lambda item: item[0])
    grouped: dict[float, list[str]] = {}
    for timestamp, content in events:
        bucket = grouped.setdefault(timestamp, [])
        if content:
            bucket.append(content)
    timeline = [(timestamp, " ".join(parts)) for timestamp, parts in grouped.items()]
    gaps = [right[0] - left[0] for left, right in pairwise(timeline) if right[0] > left[0]]
    typical_gap = sorted(gaps)[len(gaps) // 2] if gaps else 5.0
    last_duration = min(15.0, max(1.0, typical_gap))
    cues: list[_Cue] = []
    for index, (start, content) in enumerate(timeline):
        if not content:
            continue
        end = timeline[index + 1][0] if index + 1 < len(timeline) else start + last_duration
        cues.append((start, min(duration_seconds, end), content, None))
    return cues


def _is_lrc_text(text: str) -> bool:
    non_empty = [line for line in text.splitlines() if line.strip()]
    if len(non_empty) < 2:
        return False
    lrc_lines = sum(1 for line in non_empty if _LRC_TAG.search(line))
    return lrc_lines >= 2 and lrc_lines * 2 >= len(non_empty)


def _plain_lines(text: str) -> list[str]:
    lines = [_clean_plain_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) == 1:
        # English scripts commonly arrive as one paragraph with ASCII full
        # stops; keep the existing Japanese/Chinese punctuation too.
        split = re.split(r"(?<=[。！？!?\.])\s*", lines[0])
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
    role: Literal["source", "target"] | None = None,
    speaker_id: str | None = None,
) -> Sentence:
    use_target = (role == "target") or (role is None and language == "zh")
    if use_target:
        return Sentence(
            id=sentence_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            source_text="",
            target_text=text,
            status="translated",
            speaker_id=speaker_id,
        )
    return Sentence(
        id=sentence_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        source_text=text,
        speaker_id=speaker_id,
    )


def _estimated_sentences(
    lines: list[str],
    duration_seconds: float,
    language: TranscriptLanguage,
    *,
    role: Literal["source", "target"] | None = None,
    speakers: list[str | None] | None = None,
) -> list[Sentence]:
    weights = [max(1, sum(not character.isspace() for character in line)) for line in lines]
    total = sum(weights)
    elapsed_weight = 0
    sentences: list[Sentence] = []
    speaker_list = speakers or [None] * len(lines)
    for index, (line, weight, speaker_id) in enumerate(
        zip(lines, weights, speaker_list, strict=True), start=1
    ):
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
                role=role,
                speaker_id=speaker_id,
            )
        )
    return sentences


def _timed_sentences(
    cues: list[_Cue],
    duration_seconds: float,
    language: TranscriptLanguage,
    *,
    role: Literal["source", "target"] | None = None,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    for start, end, content, speaker_id in sorted(cues, key=lambda item: (item[0], item[1])):
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
                role=role,
                speaker_id=speaker_id,
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
    role: Literal["source", "target"] | None = None,
    use_embedded_timing: bool = True,
) -> TranscriptImport:
    if language not in {"ja", "en", "zh"}:
        raise ProjectError("台本语言必须是日语、英语或中文。")
    source_path = Path(path).expanduser().resolve() if path else None
    file_text = _decode_script(source_path) if source_path is not None else ""
    text = pasted_text.strip() or file_text
    if not text.strip():
        raise ProjectError("请选择台本/字幕文件，或粘贴台本文字。")
    suffix = source_path.suffix.casefold() if source_path is not None else ""

    if suffix in {".ass", ".ssa"} or "[events]" in text.casefold():
        cues = _ass_cues(text)
        source_format = "ASS/SSA"
    elif suffix == ".lrc" or _is_lrc_text(text):
        cues = _lrc_cues(text, duration_seconds)
        source_format = "LRC"
    elif suffix in {".srt", ".vtt"} or _SRT_VTT_TIMESTAMP_ARROW.search(text):
        cues = _arrow_cues(text)
        source_format = "VTT" if suffix == ".vtt" or text.lstrip().startswith("WEBVTT") else "SRT"
    else:
        cues = []
        source_format = "纯文本"

    if cues and use_embedded_timing:
        return TranscriptImport(
            sentences=_timed_sentences(cues, duration_seconds, language, role=role),
            source_format=source_format,
            timed=True,
            source_text=text,
            language=language,
        )
    if cues:
        filtered = [(content, speaker_id) for _start, _end, content, speaker_id in cues if content]
        lines = [content for content, _speaker in filtered]
        speakers = [speaker for _content, speaker in filtered]
        return TranscriptImport(
            sentences=_estimated_sentences(
                lines, duration_seconds, language, role=role, speakers=speakers
            ),
            source_format=f"{source_format}（仅文字）",
            timed=False,
            source_text=text,
            language=language,
        )
    if source_format != "纯文本":
        raise ProjectError(f"没有从 {source_format} 文件中解析到有效时间轴。")
    lines = _plain_lines(text)
    return TranscriptImport(
        sentences=_estimated_sentences(lines, duration_seconds, language, role=role),
        source_format=source_format,
        timed=False,
        source_text=text,
        language=language,
    )
