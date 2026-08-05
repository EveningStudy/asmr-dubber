import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import probe_audio, sha256_file
from asmr_dubber.environment import ffmpeg_executable
from asmr_dubber.errors import OperationCancelledError, ProjectError
from asmr_dubber.models import AudioInfo, DubProject, Sentence, load_project, save_project
from asmr_dubber.pipeline import generate_subtitles
from asmr_dubber.subtitles import write_subtitle_files


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
