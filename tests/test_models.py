from pathlib import Path

import pytest

from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.pipeline import output_filename


def audio_info() -> AudioInfo:
    return AudioInfo(
        path="source.wav",
        sha256="a" * 64,
        duration_seconds=10.0,
        sample_rate=48_000,
        channels=2,
        channel_layout="stereo",
        codec="pcm_f32le",
    )


def test_timing_uses_global_and_per_sentence_override() -> None:
    sentence = Sentence(id="s000001", start_seconds=2.0, end_seconds=5.0, ja_text="始めましょう。")
    assert sentence.effective_overlap_seconds(1.0, 20.0) == pytest.approx(0.6)
    assert sentence.chinese_start_seconds(1.0, 20.0) == pytest.approx(4.4)
    sentence.overlap_seconds = 0.25
    assert sentence.chinese_start_seconds(1.0, 20.0) == pytest.approx(4.75)
    sentence.overlap_seconds = -0.5
    assert sentence.chinese_start_seconds(1.0, 20.0) == 5.5


def test_short_sentence_chinese_never_starts_before_its_japanese() -> None:
    sentence = Sentence(id="s000001", start_seconds=10.0, end_seconds=10.2, ja_text="はい。")
    assert sentence.effective_overlap_seconds(1.0, 20.0) == pytest.approx(0.04)
    assert sentence.chinese_start_seconds(1.0, 20.0) == pytest.approx(10.16)
    assert sentence.chinese_start_seconds(1.0, 20.0) > sentence.start_seconds


def test_long_sentence_uses_configured_maximum_overlap() -> None:
    sentence = Sentence(id="s000001", start_seconds=2.0, end_seconds=12.0, ja_text="長い文章。")
    assert sentence.effective_overlap_seconds(1.0, 20.0) == 1.0
    assert sentence.chinese_start_seconds(1.0, 20.0) == 11.0


def test_overlap_percentage_is_configurable() -> None:
    sentence = Sentence(id="s000001", start_seconds=2.0, end_seconds=5.0, ja_text="始めましょう。")
    assert sentence.effective_overlap_seconds(2.0, 50.0) == pytest.approx(1.5)
    assert sentence.chinese_start_seconds(2.0, 50.0) == pytest.approx(3.5)


def test_project_round_trip(tmp_path: Path) -> None:
    project = DubProject(
        source=audio_info(),
        settings=ProjectSettings(global_overlap_seconds=1.25),
        sentences=[Sentence(id="s000001", start_seconds=1.0, end_seconds=2.0, ja_text="はい。")],
    )
    manifest = save_project(project, tmp_path)
    loaded, directory = load_project(manifest)
    assert directory == tmp_path.resolve()
    assert loaded.settings.global_overlap_seconds == 1.25
    assert loaded.sentences[0].ja_text == "はい。"


def test_safe_default_asr_batch_size() -> None:
    settings = ProjectSettings()
    assert settings.asr_batch_size == 1
    assert settings.tts_inference_timesteps == 30
    assert settings.tts_clone_mode == "stable_reference"
    assert settings.match_source_loudness is True
    assert settings.chinese_min_active_rms_dbfs == -42.0
    assert settings.chinese_target_active_rms_dbfs == -30.0
    assert settings.chinese_relative_loudness_db == -4.0
    assert settings.global_overlap_seconds == 5.0
    assert settings.global_overlap_percentage == 50.0
    assert settings.tts_index_speaker_source == "project_reference"
    assert settings.tts_index_emotion_source == "sentence_reference"


def test_output_filename_identifies_tts_configuration(tmp_path: Path) -> None:
    project = DubProject(
        source=audio_info(),
        settings=ProjectSettings(
            tts_backend="indextts2",
            tts_model="IndexTTS2",
            tts_clone_mode="stable_reference",
        ),
    )

    filename = output_filename(project, tmp_path / "voice_sample_20260723T120000Z")

    assert filename == (
        "voice_sample__ja-zh__indextts2-IndexTTS2-project_reference-sentence_reference.wav"
    )


def test_legacy_index_text_emotion_setting_is_migrated() -> None:
    settings = ProjectSettings.model_validate({"tts_index_use_emo_text": True})

    assert settings.tts_index_emotion_source == "text"


def test_sentence_id_cannot_escape_generated_audio_directory() -> None:
    with pytest.raises(ValueError):
        Sentence(
            id="../../outside",
            start_seconds=0,
            end_seconds=1,
            ja_text="テスト",
        )
