"""Regression coverage for queue defaults, cover previews and unattended references."""

from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from asmr_dubber.api import API
from asmr_dubber.autoflow import engine
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, load_project, save_project
from asmr_dubber.services import batch_queue, settings
from asmr_dubber.services.application import Application
from asmr_dubber.services.batch import Batch
from asmr_dubber.services.media import MediaStore


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "home/config"))
    return Application(tmp_path / "tasks.json")


def work(tmp_path):
    tmp_path.mkdir(parents=True)
    sf.write(tmp_path / "01.wav", np.zeros(24000), 24000)
    Image.new("RGB", (120, 80), "red").save(tmp_path / "cover.jpg")
    Image.new("RGB", (100, 80), "blue").save(tmp_path / "other.png")
    return tmp_path


def test_cover_scan_change_edit_and_path_boundary(app, tmp_path):
    api = API(app)
    folder = work(tmp_path / "work")
    scan = api.call("batch/scan", {"folder": str(folder)})
    assert scan["preview"].startswith("/media/")
    preview = api.call("batch/background", {"folder": str(folder), "background": "other.png"})
    resolved, _ = app.media.resolve(preview["preview"].split("/")[-1])
    assert resolved == folder / "other.png"
    assert api.call("batch/background", {"folder": str(folder), "background": "black"}) == {
        "preview": None
    }
    outside = tmp_path / "outside.png"
    Image.new("RGB", (10, 10)).save(outside)
    with pytest.raises(Exception, match="作品文件夹"):
        api.call("batch/background", {"folder": str(folder), "background": "../outside.png"})
    queue = api.call(
        "batch/save",
        {
            "folder": str(folder),
            "edition": scan["selected_edition"],
            "sources": scan["source_payloads"],
            "mode": "video_normal",
            "layout": "merged",
            "background": "other.png",
        },
    )
    assert queue[0]["edition"]["output_policy"]["variants"] == ["bilingual", "replace"]
    assert queue[0]["edition"]["output_policy"]["settings"]["separation_enabled"]
    edited = api.call("batch/edit", {"identifier": queue[0]["plan_id"]})
    assert edited["preview"] == preview["preview"]
    settings.update({"autoflow_default_content": "dubbing"})
    assert settings.current().autoflow_default_content == "dubbing"


def test_translation_ready_uses_selected_provider_and_environment(app, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert not settings.key_statuses()["translation_available"]["deepseek"]
    settings.key("translation", "deepseek", "test-key-not-real")
    status = settings.key_statuses()
    assert status["translation_available"]["deepseek"]
    assert "test-key-not-real" not in str(status)
    settings.key("translation", "deepseek", clear=True)
    assert not settings.key_statuses()["translation_available"]["deepseek"]
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key-not-real")
    assert settings.key_statuses()["translation_available"]["deepseek"]


def test_live_queue_reference_applies_at_synthesis_after_snapshot_and_survives_restart(
    app, tmp_path, monkeypatch
):
    plans = [{"plan_id": name} for name in ("done", "waiting-1", "waiting-2")]
    app.batch._save(plans)
    audio = tmp_path / "external.wav"
    sf.write(audio, np.zeros(24000), 24000)
    manifests = []
    for name in ("waiting-1", "waiting-2"):
        project = DubProject(
            source=AudioInfo(
                path="source.wav",
                sha256="a" * 64,
                duration_seconds=1,
                sample_rate=24000,
                channels=1,
            ),
            settings=ProjectSettings(tts_backend="indextts2_5"),
        )
        directory = tmp_path / name
        save_project(project, directory)
        manifests.append(directory / "project.json")
    seen = []

    class Synthesized(Exception):
        pass

    def synthesize(paths, command, manifest, *args):
        assert command == "synthesize"
        project, _ = load_project(manifest)
        assert project.settings.tts_index_speaker_source == "external"
        assert project.settings.tts_reference_source == "external"
        assert project.settings.tts_external_reference_text == "chosen words"
        assert project.settings.tts_external_reference_language == "ja"
        assert Path(project.settings.tts_external_reference_audio).is_file()
        seen.append(manifest)
        raise Synthesized()

    def run(queue_snapshot, *, cancel_event, reference_event_callback):
        assert queue_snapshot == plans
        reference_event_callback({"kind": "plan_finished", "plan_id": "done"})
        # The user chooses an external reference while this queue is already running.
        chosen = app.batch.reference(str(audio), "chosen words", "ja")
        assert chosen["plans"] == ["waiting-1", "waiting-2"]
        audio.unlink()  # Upload lifetime must not affect unattended later works.
        for name, manifest in zip(("waiting-1", "waiting-2"), manifests, strict=True):
            state = {
                "status": "awaiting_reference",
                "project_json": str(manifest),
                "plan_id": name,
                "reference_wait_seconds": 0,
            }
            with pytest.raises(Synthesized):
                engine.execute_task(
                    None,
                    manifest.parent,
                    manifest.parent / "state.json",
                    state,
                    [],
                    reference_event_callback=reference_event_callback,
                )
        return 0, []

    monkeypatch.setattr(engine, "run_asmr_cli", synthesize)
    monkeypatch.setattr(batch_queue, "run_queue", run)
    result = app.batch.run({}, lambda *args: None, Event(), lambda event: None)
    assert result["exit_code"] == 0
    assert len(seen) == 2
    restarted = Batch(MediaStore())
    chosen = restarted.reference()
    assert Path(chosen["audio"]).is_file()
    assert chosen["plans"] == ["waiting-1", "waiting-2"]
    restarted._save([{"plan_id": "edited"}])
    assert restarted.reference()["plans"] == ["edited"]
    assert not restarted.reference(clear=True).get("audio")


def test_quality_defaults_keep_source_loudness():
    defaults = settings.Settings()
    assert defaults.autoflow_default_content == "both"
    assert defaults.chinese_max_auto_speed == 1.4
    assert defaults.replacement_normalize_chinese_loudness
    assert defaults.replacement_match_source_loudness
    assert defaults.replacement_chinese_relative_loudness_db == 0
    assert defaults.replacement_chinese_gain_db == 0
