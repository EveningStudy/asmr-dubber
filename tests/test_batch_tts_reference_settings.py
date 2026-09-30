import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.autoflow import engine
from asmr_dubber.autoflow.output_policy import configure_project, policy_for_settings
from asmr_dubber.autoflow.ui_services import config_from_settings
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.ui_services import select_autoflow_project_reference
from asmr_dubber.user_settings import UserSettings, load_user_settings, save_user_settings


def make_project(tmp_path, **settings):
    project = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="a" * 64, duration_seconds=10, sample_rate=24000, channels=1
        ),
        settings=ProjectSettings(**settings),
        sentences=[Sentence(id="s000001", start_seconds=0, end_seconds=3, source_text="original")],
    )
    save_project(project, tmp_path)
    return tmp_path / "project.json"


def test_index25_survives_real_batch_cli_creation(tmp_path, monkeypatch):
    home = tmp_path / "custom-home"
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(home))
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(home / "config"))
    settings = UserSettings(tts_backend="indextts2_5", tts_model="IndexTTS-2.5")
    save_user_settings(settings)
    assert load_user_settings().tts_backend == "indextts2_5"
    paths = engine.find_tool_paths(config_from_settings(settings))
    assert paths.asmr_home == home
    assert os.environ["ASMR_DUBBER_HOME"] == str(home)
    media = tmp_path / "source.wav"
    sf.write(media, np.zeros(24000), 24000)
    manifest = engine.create_asmr_project(paths, media)
    created, _ = load_project(manifest)
    assert created.settings.tts_backend == "indextts2_5"


def test_queue_snapshots_backend_and_external_reference(tmp_path):
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference")
    settings = ProjectSettings(
        tts_backend="indextts2_5",
        tts_index_speaker_source="external",
        tts_external_reference_audio=str(reference),
    )
    policy = policy_for_settings(settings, "dubbing", "bilingual")
    manifest = make_project(tmp_path / "project")
    configure_project(manifest, policy)
    loaded, _ = load_project(manifest)
    assert loaded.settings.tts_backend == "indextts2_5"
    assert engine.project_has_external_reference(manifest)
    reference.unlink()
    with pytest.raises(engine.VideoPreparerError, match="文件不存在"):
        engine.project_has_external_reference(manifest)


def test_reference_edits_are_saved_before_confirmation(tmp_path, monkeypatch):
    manifest = make_project(tmp_path, tts_backend="indextts2_5")
    monkeypatch.setattr("asmr_dubber.ui_services.reference_preview", lambda *args: None)
    select_autoflow_project_reference(str(manifest), "s000001", 1.25, 5.75, "edited")
    loaded, _ = load_project(manifest)
    assert loaded.sentences[0].start_seconds == 1.25
    assert loaded.sentences[0].end_seconds == 5.75
    assert loaded.sentences[0].source_text == "edited"
    assert loaded.settings.tts_index_speaker_source == "project_reference"
    before = manifest.read_bytes()
    with pytest.raises(Exception, match="参考范围"):
        select_autoflow_project_reference(str(manifest), "s000001", 8, 2, "bad")
    assert manifest.read_bytes() == before


def test_index_external_reference_is_the_shared_anchor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from asmr_dubber import voice_reference

    manifest = make_project(
        tmp_path / "project", tts_backend="indextts2_5", tts_index_speaker_source="external"
    )
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"chosen-external")
    monkeypatch.setattr(
        voice_reference,
        "prepare_voice_reference",
        lambda *args: pytest.fail("wrong reference selector"),
    )
    monkeypatch.setattr(
        voice_reference,
        "prepare_index_speaker_reference",
        lambda *args: SimpleNamespace(path=audio, text="", language="ja"),
    )
    result = engine.extract_shared_reference(manifest, tmp_path / "shared.wav")
    assert Path(result["audio"]).read_bytes() == b"chosen-external"
