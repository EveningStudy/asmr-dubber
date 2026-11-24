import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import probe_audio, sha256_file
from asmr_dubber.environment import ffmpeg_executable
from asmr_dubber.errors import OperationCancelledError, ProjectError
from asmr_dubber.models import AudioInfo, DubProject, Sentence, load_project, save_project
from asmr_dubber.pipeline import generate_subtitles
from asmr_dubber.subtitles import _wrap_line, write_subtitle_files


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("我们测得阿伏伽德罗常数约为6.02214076×10²³每摩尔", "6.02214076×10²³"),
        ("配音响度测量结果约为负的-28.5dB需要留出动态空间", "-28.5"),
        ("配音响度测量结果约为负的−18.5dB需要留出动态空间", "−18.5"),
        ("画面在第十二秒之后等待0.35秒再出现字幕", "0.35"),
        ("科学记数法示例的测量结果为-1.602e-19库仑", "-1.602e-19"),
        ("光速约为每秒299792458米可以在真空传播", "299792458"),
    ],
)
def test_chinese_wrap_keeps_numbers_that_fit_together(text: str, number: str) -> None:
    lines = _wrap_line(text, 22)
    assert "".join(lines) == text
    assert all(len(line) <= 22 for line in lines)
    assert any(number in line for line in lines)


def test_oversized_number_does_not_prevent_protecting_the_next_number() -> None:
    text = "甲123456789乙−2.5秒"
    lines = _wrap_line(text, 5)
    assert "".join(lines) == text
    assert all(len(line) <= 5 for line in lines)
    assert any("−2.5" in line for line in lines)


def test_wrap_number_at_boundaries_and_oversized_fallback() -> None:
    assert _wrap_line("甲乙丙丁12.34秒", 5) == ["甲乙丙丁", "12.34", "秒"]
    assert _wrap_line("甲乙丙丁 12.34秒", 5) == ["甲乙丙丁", "12.34", "秒"]
    assert _wrap_line("甲乙丙丁−18.5秒", 5) == ["甲乙丙丁", "−18.5", "秒"]
    assert _wrap_line("甲乙丙丁-18.5秒", 5) == ["甲乙丙丁", "-18.5", "秒"]
    text = "甲123456789乙"
    lines = _wrap_line(text, 4)
    assert lines == ["甲123", "4567", "89乙"]
    assert "".join(lines) == text


def test_english_wrap_keeps_existing_behavior() -> None:
    text = "A very long English sentence with 12.3456789 and words."
    expected = textwrap.wrap(
        text, width=8, break_long_words=True, break_on_hyphens=False, replace_whitespace=False
    )
    assert _wrap_line(text, 8) == expected


def test_number_aware_wrap_preserves_srt_timing_and_lrc(tmp_path: Path) -> None:
    text = "我们测得阿伏伽德罗常数约为6.02214076×10²³每摩尔"
    sentence = Sentence(id="s1", start_seconds=1.25, end_seconds=10, source_text=text)
    srt, lrc = write_subtitle_files([sentence], tmp_path, "source", maximum_chars=22)
    lines = srt.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["1", "00:00:01,250 --> 00:00:10,000"]
    assert "".join(lines[2:]) == text
    assert any("6.02214076×10²³" in line for line in lines[2:])
    assert lrc.read_text(encoding="utf-8").splitlines() == [f"[00:01.25]{text}"]


@pytest.mark.parametrize("width", [40, 60, 80, 120, 500])
def test_custom_line_width_persists_and_controls_srt_lrc(tmp_path, width):
    from asmr_dubber.models import ProjectSettings

    settings = ProjectSettings(subtitle_max_chars_per_line=width)
    restored = ProjectSettings.model_validate_json(settings.model_dump_json())
    text = "字" * (width + 5)
    srt, lrc = write_subtitle_files(
        [
            Sentence(id="s1", start_seconds=0, end_seconds=60, source_text=text),
        ],
        tmp_path,
        "source",
        maximum_chars=restored.subtitle_max_chars_per_line,
    )
    assert srt.read_text(encoding="utf-8").splitlines()[2:] == ["字" * width, "字" * 5]
    # LRC keeps each language on one unwrapped line so players do not drop wrapped tails.
    assert [line.split("]", 1)[1] for line in lrc.read_text(encoding="utf-8").splitlines()] == [
        text,
    ]


def _sentences() -> list[Sentence]:
    return [
        Sentence(
            id="s000001",
            start_seconds=1.25,
            end_seconds=2.75,
            ja_text="さあ始めましょう。",
            zh_text="让我们开始吧。",
        ),
        Sentence(
            id="s000002",
            start_seconds=3.0,
            end_seconds=4.0,
            ja_text="ああ。",
            zh_text="啊。",
            enabled=False,
        ),
    ]


def test_subtitle_writer_creates_selected_srt_and_lrc_and_omits_disabled_rows(
    tmp_path: Path,
) -> None:
    srt, lrc = write_subtitle_files(_sentences(), tmp_path, "bilingual")

    srt_text = srt.read_text(encoding="utf-8")
    lrc_text = lrc.read_text(encoding="utf-8")
    assert "00:00:01,250 --> 00:00:02,750" in srt_text
    assert "さあ始めましょう。\n让我们开始吧。" in srt_text
    assert "ああ。" not in srt_text
    assert "[00:01.25]さあ始めましょう。" in lrc_text
    assert "[00:01.25]让我们开始吧。" in lrc_text


def test_chinese_subtitles_require_translation(tmp_path: Path) -> None:
    sentence = Sentence(
        id="s000001",
        start_seconds=0,
        end_seconds=1,
        ja_text="テスト。",
    )
    with pytest.raises(ProjectError, match="没有中文"):
        write_subtitle_files([sentence], tmp_path, "zh")


def test_dubbing_timeline_uses_offset_and_effective_accelerated_duration(
    tmp_path: Path,
) -> None:
    sentences = [
        Sentence(
            id="s000001",
            start_seconds=1.0,
            end_seconds=2.0,
            ja_text="一。",
            zh_text="一。",
            tts_duration_seconds=3.0,
        ),
        Sentence(
            id="s000002",
            start_seconds=3.0,
            end_seconds=4.0,
            ja_text="二。",
            zh_text="二。",
            tts_duration_seconds=1.0,
        ),
    ]

    srt, _lrc = write_subtitle_files(
        sentences,
        tmp_path,
        "zh",
        timeline="dubbing",
        minimum_duration=0.2,
        maximum_cps=40.0,
        chinese_dubbing_offset_ms=500,
        chinese_max_auto_speed=1.5,
    )

    content = srt.read_text(encoding="utf-8")
    assert "00:00:01,500 --> 00:00:03,500" in content
    assert "00:00:03,500 --> 00:00:04,500" in content


def test_audio_project_generates_external_subtitles_without_touching_audio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(80_000, dtype=np.float32), 16_000, subtype="FLOAT")
    source_hash = sha256_file(source)
    info = probe_audio(source, sha256=source_hash)
    project = DubProject(source=info, sentences=_sentences())
    save_project(project, tmp_path)

    srt, lrc, video = generate_subtitles(project, tmp_path, language="zh")
    loaded, _ = load_project(tmp_path)

    assert video is None
    assert srt.is_file() and lrc.is_file()
    assert loaded.subtitle_language == "zh"
    assert loaded.subtitle_srt_file == "subtitles/subtitles_zh.srt"
    assert loaded.subtitle_lrc_file == "subtitles/subtitles_zh.lrc"
    assert loaded.subtitle_video_file is None
    assert sha256_file(source) == source_hash


def test_legacy_audio_info_defaults_to_audio_media_type() -> None:
    info = AudioInfo.model_validate(
        {
            "path": "source.wav",
            "sha256": "0" * 64,
            "duration_seconds": 1,
            "sample_rate": 16_000,
            "channels": 1,
        }
    )
    assert info.media_type == "audio"


def test_video_project_generates_subtitled_video_with_existing_mixed_audio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    subprocess.run(
        [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=10:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=16000:duration=5",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )
    mixed = tmp_path / "output" / "mixed.wav"
    mixed.parent.mkdir()
    sf.write(mixed, np.zeros(80_000, dtype=np.float32), 16_000, subtype="PCM_24")
    project = DubProject(
        source=probe_audio(source),
        sentences=_sentences(),
        output_file="output/mixed.wav",
    )
    save_project(project, tmp_path)

    _srt, _lrc, video = generate_subtitles(project, tmp_path, language="bilingual")
    loaded, _ = load_project(tmp_path)

    assert video is not None and video.is_file()
    assert probe_audio(video).media_type == "video"
    assert loaded.subtitle_video_file is not None
    assert "__mixed" in loaded.subtitle_video_file


def test_cancelled_video_subtitles_preserve_previous_project_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    old_srt = tmp_path / "subtitles" / "old.srt"
    old_lrc = tmp_path / "subtitles" / "old.lrc"
    old_video = tmp_path / "output" / "old.mp4"
    old_srt.parent.mkdir()
    old_video.parent.mkdir()
    old_srt.write_text("old srt", encoding="utf-8")
    old_lrc.write_text("old lrc", encoding="utf-8")
    old_video.write_bytes(b"old video")
    project = DubProject(
        source=AudioInfo(
            path="source.mp4",
            sha256=sha256_file(source),
            duration_seconds=5,
            sample_rate=16_000,
            channels=2,
            media_type="video",
        ),
        sentences=_sentences(),
        subtitle_language="bilingual",
        subtitle_srt_file="subtitles/old.srt",
        subtitle_lrc_file="subtitles/old.lrc",
        subtitle_video_file="output/old.mp4",
    )
    save_project(project, tmp_path)

    def cancel_render(*_args, **_kwargs):
        raise OperationCancelledError("cancelled")

    monkeypatch.setattr(
        "asmr_dubber.pipeline.render_subtitled_video",
        cancel_render,
    )

    with pytest.raises(OperationCancelledError):
        generate_subtitles(project, tmp_path, language="ja")

    loaded, _ = load_project(tmp_path)
    assert loaded.subtitle_language == "bilingual"
    assert loaded.subtitle_srt_file == "subtitles/old.srt"
    assert loaded.subtitle_lrc_file == "subtitles/old.lrc"
    assert loaded.subtitle_video_file == "output/old.mp4"
    assert old_video.read_bytes() == b"old video"


def test_source_subtitles_render_english_text_without_japanese_assumptions(tmp_path: Path) -> None:
    sentence = Sentence(
        id="s000001",
        start_seconds=0.0,
        end_seconds=2.0,
        source_text="Please make yourself comfortable.",
        zh_text="请放松一点。",
    )

    source_srt, _ = write_subtitle_files([sentence], tmp_path / "source", "source")
    bilingual_srt, _ = write_subtitle_files([sentence], tmp_path / "bilingual", "bilingual")

    source_text = source_srt.read_text(encoding="utf-8")
    bilingual_text = bilingual_srt.read_text(encoding="utf-8")
    assert "Please make yourself\ncomfortable." in source_text
    assert "请放松一点。" not in source_text
    assert "Please make yourself\ncomfortable." in bilingual_text
    assert "请放松一点。" in bilingual_text
