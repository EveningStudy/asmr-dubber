from pathlib import Path

import pytest

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


def test_reads_shift_jis_dlsite_script(tmp_path: Path) -> None:
    source = tmp_path / "台本.txt"
    source.write_bytes("一行目です。\n二行目です。".encode("cp932"))

    parsed = parse_transcript(duration_seconds=10, path=source)

    assert [item.ja_text for item in parsed.sentences] == ["一行目です。", "二行目です。"]
