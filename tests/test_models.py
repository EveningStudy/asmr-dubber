import json
from pathlib import Path

import pytest

from asmr_dubber.errors import ProjectConflictError
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


def test_project_round_trip(tmp_path: Path) -> None:
    project = DubProject(
        source=audio_info(),
        settings=ProjectSettings(chinese_dubbing_offset_ms=-250, chinese_max_auto_speed=1.3),
        sentences=[Sentence(id="s000001", start_seconds=1.0, end_seconds=2.0, ja_text="はい。")],
    )
    manifest = save_project(project, tmp_path)
    loaded, directory = load_project(manifest)
    assert directory == tmp_path.resolve()
    assert loaded.settings.chinese_dubbing_offset_ms == -250
    assert loaded.settings.chinese_max_auto_speed == 1.3
    assert loaded.sentences[0].ja_text == "はい。"


def test_concurrent_project_save_detects_stale_revision(tmp_path: Path) -> None:
    manifest = save_project(DubProject(source=audio_info()), tmp_path)
    first, _ = load_project(manifest)
    stale, _ = load_project(manifest)

    first.settings.chinese_dubbing_offset_ms = -100
    save_project(first, tmp_path)
    stale.settings.chinese_dubbing_offset_ms = 200

    with pytest.raises(ProjectConflictError, match="另一个窗口或命令修改"):
        save_project(stale, tmp_path)


def test_schema_one_project_migrates_removed_backends_and_creates_backup(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "app_version": "0.3.4",
        "source": audio_info().model_dump(),
        "settings": {
            "asr_backend": "qwen3_asr",
            "asr_model": "old-asr",
            "tts_backend": "voxcpm2",
            "tts_model": "old-tts",
            "asr_vad_mode": "asmr",
            "asr_review_models": ["qwen3_asr|old-asr"],
        },
    }
    manifest = tmp_path / "project.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    project, directory = load_project(manifest)
    assert project.schema_version == 2
    assert project.settings.asr_backend == "parakeet_nemo"
    assert project.settings.tts_backend == "indextts2"
    assert project.settings.asr_review_enabled is False
    assert project.migration_warnings

    save_project(project, directory)
    backups = list((tmp_path / "backups").glob("project-schema-v1-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["schema_version"] == 1


def test_safe_default_asr_batch_size() -> None:
    settings = ProjectSettings()
    assert settings.asr_batch_size == 1
    assert settings.tts_request_concurrency == 2
    assert settings.tts_clone_mode == "stable_reference"
    assert settings.match_source_loudness is True
    assert settings.chinese_min_active_rms_dbfs == -42.0
    assert settings.chinese_target_active_rms_dbfs == -30.0
    assert settings.chinese_relative_loudness_db == -4.0
    assert settings.chinese_dubbing_offset_ms == 0
    assert settings.chinese_max_auto_speed == 1.2
    assert settings.tts_index_speaker_source == "project_reference"
    assert settings.tts_index_emotion_source == "sentence_reference"
    assert settings.mix_peak_protection is True


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


def test_legacy_vad_boolean_is_migrated_to_backend_mode() -> None:
    settings = ProjectSettings.model_validate({"asr_vad_filter": True})

    assert settings.asr_vad_mode == "backend"
    assert settings.asr_vad_filter is True


def test_legacy_automatic_parakeet_chunking_is_migrated() -> None:
    settings = ProjectSettings.model_validate({"asr_chunk_seconds": 0})

    assert settings.asr_chunk_seconds == 120.0


def test_asmr_vad_and_standalone_alignment_are_independent_stages() -> None:
    settings = ProjectSettings(
        asr_backend="parakeet_nemo",
        asr_vad_mode="asmr",
        asr_forced_alignment_enabled=True,
    )

    assert settings.asr_vad_mode == "asmr"
    assert settings.asr_vad_filter is False
    assert settings.asr_forced_alignment_enabled is True


def test_new_asr_defaults_use_only_supported_review_models_and_translation_context() -> None:
    settings = ProjectSettings()

    assert settings.asr_vad_mode == "off"
    assert settings.asr_review_text_priority_model.startswith("parakeet_nemo|")
    assert settings.asr_review_timestamp_priority_model.startswith("parakeet_nemo|")
    assert {item.partition("|")[0] for item in settings.asr_review_models} <= {
        "parakeet_nemo",
        "kotoba_whisper",
        "faster_whisper",
    }
    assert settings.asr_forced_alignment_enabled is False
    assert settings.translation_send_context is True


def test_sentence_id_cannot_escape_generated_audio_directory() -> None:
    with pytest.raises(ValueError):
        Sentence(
            id="../../outside",
            start_seconds=0,
            end_seconds=1,
            ja_text="テスト",
        )


def test_legacy_timing_fields_are_ignored_when_loading_current_models() -> None:
    settings = ProjectSettings.model_validate(
        {
            "global_overlap_seconds": 5.0,
            "global_overlap_percentage": 50.0,
        }
    )
    sentence = Sentence.model_validate(
        {
            "id": "s000001",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "ja_text": "はい。",
            "overlap_seconds": 0.5,
        }
    )

    assert settings.chinese_dubbing_offset_ms == 0
    assert settings.chinese_max_auto_speed == 1.2
    assert "overlap_seconds" not in sentence.model_dump()
