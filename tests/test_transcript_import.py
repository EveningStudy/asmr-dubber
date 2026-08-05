import json
from pathlib import Path

import pytest

import asmr_dubber.pipeline as pipeline
from asmr_dubber.audio import sha256_file
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import AudioInfo, DubProject
from asmr_dubber.transcript_import import parse_transcript


def test_imports_srt_timeline_without_asr(tmp_path: Path) -> None:
    source = tmp_path / "script.srt"
    source.write_text(
        "1\n00:00:01,250 --> 00:00:03,500\nこんにちは。\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\n大丈夫？\n",
        encoding="utf-8",
    )

    parsed = parse_transcript(duration_seconds=10, path=source)

    assert parsed.timed is True
    assert parsed.source_format == "SRT"
    assert [item.ja_text for item in parsed.sentences] == ["こんにちは。", "大丈夫？"]
    assert parsed.sentences[0].start_seconds == pytest.approx(1.25)
    assert parsed.sentences[1].end_seconds == pytest.approx(7.0)


def test_imports_ass_and_removes_style_overrides(tmp_path: Path) -> None:
    source = tmp_path / "script.ass"
    source.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:02.00,0:00:04.20,Default,,0,0,0,,{\\i1}囁き\\Nです。\n",
        encoding="utf-8",
    )

    parsed = parse_transcript(duration_seconds=8, path=source)

    assert parsed.timed is True
    assert parsed.sentences[0].ja_text == "囁き です。"
    assert parsed.sentences[0].end_seconds == pytest.approx(4.2)


def test_imports_lrc_and_derives_end_times() -> None:
    parsed = parse_transcript(
        duration_seconds=20,
        pasted_text="[00:01.00]一行目\n[00:04.50]二行目\n",
    )

    assert parsed.source_format == "LRC"
    assert parsed.sentences[0].end_seconds == pytest.approx(4.5)
    assert parsed.sentences[1].end_seconds == pytest.approx(8.0)


def test_plain_script_uses_each_line_and_estimates_by_text_length() -> None:
    parsed = parse_transcript(
        duration_seconds=12,
        pasted_text="短い。\nこれは少し長い台詞です。",
    )

    assert parsed.timed is False
    assert len(parsed.sentences) == 2
    assert parsed.sentences[0].end_seconds < 6
    assert parsed.sentences[-1].end_seconds == pytest.approx(12)


def test_english_single_paragraph_splits_on_ascii_full_stops() -> None:
    parsed = parse_transcript(
        duration_seconds=12,
        pasted_text="Hello there. Please make yourself comfortable.",
        language="en",
    )

    assert [item.source_text for item in parsed.sentences] == [
        "Hello there.",
        "Please make yourself comfortable.",
    ]


def test_reads_shift_jis_dlsite_script(tmp_path: Path) -> None:
    source = tmp_path / "台本.txt"
    source.write_bytes("一行目です。\n二行目です。".encode("cp932"))

    parsed = parse_transcript(duration_seconds=10, path=source)

    assert [item.ja_text for item in parsed.sentences] == ["一行目です。", "二行目です。"]


def test_imports_chinese_srt_as_ready_to_synthesize_text(tmp_path: Path) -> None:
    source = tmp_path / "translated.srt"
    source.write_text(
        "1\n00:00:01,250 --> 00:00:03,500\n你好。\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\n没问题吗？\n",
        encoding="utf-8",
    )

    parsed = parse_transcript(duration_seconds=10, path=source, language="zh")

    assert parsed.language == "zh"
    assert parsed.timed is True
    assert [item.ja_text for item in parsed.sentences] == ["", ""]
    assert [item.zh_text for item in parsed.sentences] == ["你好。", "没问题吗？"]
    assert all(item.status == "translated" for item in parsed.sentences)
    assert parsed.sentences[0].start_seconds == pytest.approx(1.25)
    assert parsed.sentences[1].end_seconds == pytest.approx(7.0)


def test_chinese_plain_script_estimates_timeline_by_text_length() -> None:
    parsed = parse_transcript(
        duration_seconds=12,
        pasted_text="短句。\n这是一句稍微长一些的中文配音台词。",
        language="zh",
    )

    assert parsed.language == "zh"
    assert parsed.timed is False
    assert [item.ja_text for item in parsed.sentences] == ["", ""]
    assert [item.zh_text for item in parsed.sentences] == [
        "短句。",
        "这是一句稍微长一些的中文配音台词。",
    ]
    assert parsed.sentences[0].end_seconds < 6
    assert parsed.sentences[-1].end_seconds == pytest.approx(12)


def _import_project(tmp_path: Path) -> DubProject:
    source = tmp_path / "source.wav"
    source.write_bytes(b"test source")
    return DubProject(
        source=AudioInfo(
            path=source.name,
            sha256=sha256_file(source),
            duration_seconds=10,
            sample_rate=16_000,
            channels=1,
        )
    )


def test_project_import_marks_chinese_script_as_direct_tts_input(tmp_path: Path) -> None:
    project = _import_project(tmp_path)

    result = pipeline.import_project_transcript(
        project,
        tmp_path,
        pasted_text="第一句。\n第二句。",
        script_language="zh",
    )

    assert result["language"] == "zh"
    assert project.asr_language == "中文 (imported 纯文本)"
    assert [item.ja_text for item in project.sentences] == ["", ""]
    assert [item.zh_text for item in project.sentences] == ["第一句。", "第二句。"]
    report = json.loads(
        (tmp_path / "imports" / "latest-transcript.json").read_text(encoding="utf-8")
    )
    assert report["language"] == "zh"
    assert report["plain_timing"] == "estimate"


def test_project_import_marks_english_script_and_uses_existing_faster_whisper(
    tmp_path: Path,
) -> None:
    project = _import_project(tmp_path)

    result = pipeline.import_project_transcript(
        project,
        tmp_path,
        pasted_text="Good evening.\nAre you comfortable?",
        script_language="en",
    )

    assert result["language"] == "en"
    assert project.source_language == "en"
    assert project.settings.asr_backend == "faster_whisper"
    assert project.settings.asr_model == "large-v2"
    assert [item.source_text for item in project.sentences] == [
        "Good evening.",
        "Are you comfortable?",
    ]
    assert project.asr_language == "英语 (imported 纯文本)"


def test_project_import_rejects_qwen_alignment_for_chinese_script(tmp_path: Path) -> None:
    project = _import_project(tmp_path)

    with pytest.raises(ProjectError, match="中文纯台本不能使用 Qwen3"):
        pipeline.import_project_transcript(
            project,
            tmp_path,
            pasted_text="这是中文配音台词。",
            plain_timing="qwen",
            script_language="zh",
        )
