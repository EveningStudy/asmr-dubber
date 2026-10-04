import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.api import API
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import ProjectSettings, load_project
from asmr_dubber.services import models, parameters, settings
from asmr_dubber.services.application import Application
from asmr_dubber.services.tasks import ACTIVE, Tasks


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return Application(tmp_path / "tasks.json")


def completed(tasks, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        task = tasks.get(identifier)
        if task["status"] not in ACTIVE:
            return task
        time.sleep(0.01)
    pytest.fail("Task failed to finish")


def create(application, tmp_path):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000 * 3), 16000)
    request = {"kind": "create", "source": str(audio), "source_language": "en"}
    task = application.start(request)
    result = completed(application.tasks, task["id"])
    assert result["status"] == "completed", result
    return result["result"]


def test_catalog_has_every_setting_once_and_real_constraints():
    catalog = parameters.catalog()
    by_key = {item["key"]: item for item in catalog}
    assert len(by_key) == len(catalog) == len(settings.Settings.model_fields)
    actual = settings.Settings().model_dump()
    for key, value in actual.items():
        assert by_key[key]["default"] == value
        assert by_key[key]["label"]
        assert by_key[key]["group"]
        assert by_key[key]["scope"] == (
            "project" if key in ProjectSettings.model_fields else "global"
        )
    assert by_key["asr_batch_size"]["maximum"] == 32
    assert by_key["asr_review_models"]["type"] == "array"
    assert by_key["tts_index25_emotion_vector"]["items"]["maximum"] == 1


def test_global_defaults_do_not_change_existing_project(application, tmp_path):
    project = create(application, tmp_path)
    original = project["settings"]["tts_speed"]
    application.update_settings({"tts_speed": 1.2})
    assert load_project(project["manifest"])[0].settings.tts_speed == original
    assert settings.current().tts_speed == 1.2


def test_project_settings_leave_defaults_and_unrelated_parameters_intact(application, tmp_path):
    project = create(application, tmp_path)
    defaults = settings.current().tts_speed
    application.update_settings({"tts_speed": 1.3}, project["manifest"], project["revision"])
    active = load_project(project["manifest"])[0]
    assert active.settings.tts_speed == 1.3
    assert active.settings.asr_model == project["settings"]["asr_model"]
    assert settings.current().tts_speed == defaults
    with pytest.raises(ProjectError):
        application.update_settings({"tts_speed": 1.4}, project["manifest"], project["revision"])
    assert load_project(project["manifest"])[0].settings.tts_speed == 1.3


@pytest.mark.parametrize(
    "changes",
    [{"bogus": 1}, {"tts_speed": 100}, {"asr_batch_size": 0}, {"tts_api_extra_body": "[]"}],
)
def test_settings_reject_invalid_changes(application, changes):
    before = settings.current().model_dump()
    with pytest.raises((ValueError, ProjectError)):
        application.update_settings(changes)
    assert settings.current().model_dump() == before


def test_table_preserves_sentence_controls_and_rejects_stale_save(application, tmp_path):
    project = create(application, tmp_path)
    rows = [["s1", True, 0, 1, "hello", "你好", "off", -3, True, 2]]
    saved = application.save_table(project["manifest"], rows, project["revision"])
    assert saved["rows"] == rows
    with pytest.raises(ProjectError):
        application.save_table(project["manifest"], rows, project["revision"])
    assert load_project(project["manifest"])[0].sentences[0].original_audio_enabled is False


def test_cancel_resume_and_restart_use_persisted_request(tmp_path):
    began = threading.Event()
    runs = []

    def execute(request, report, token, reference):
        runs.append(request)
        report("progress", 1, 4)
        reference({"kind": "ready", "project_json": "example"})
        began.set()
        if len(runs) == 1:
            token.wait(2)
        from asmr_dubber.task_control import check_cancelled

        check_cancelled(token)
        return {"ok": True}

    tasks = Tasks(execute, tmp_path / "tasks.json")
    task = tasks.start({"kind": "example", "value": 3}, "project")
    assert began.wait(3)
    assert tasks.get(task["id"])["reference"]["kind"] == "ready"
    with pytest.raises(ValueError):
        tasks.start({"kind": "example"}, "project")
    tasks.cancel(task["id"])
    assert completed(tasks, task["id"])["status"] == "cancelled"
    restored = Tasks(execute, tmp_path / "tasks.json")
    restored.resume(task["id"])
    result = completed(restored, task["id"])
    assert result["status"] == "completed"
    assert result["result"] == {"ok": True}
    assert runs == [{"kind": "example", "value": 3}] * 2
    assert result["reference"] is None


def test_restart_marks_unfinished_tasks_interrupted(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"t": {"id": "t", "status": "running"}}))
    tasks = Tasks(lambda *args: None, path)
    assert tasks.get("t")["status"] == "interrupted"
    assert json.loads(path.read_text())["t"]["status"] == "interrupted"


def test_task_failure_is_saved_and_secrets_are_rejected(tmp_path):
    def execute(*args):
        raise ValueError("failed")

    tasks = Tasks(execute, tmp_path / "tasks.json")
    with pytest.raises(ValueError):
        tasks.start({"kind": "example", "api_key": "secret"}, "p")
    task = tasks.start({"kind": "example"}, "p")
    assert completed(tasks, task["id"])["error"] == "failed"


def test_upload_is_owned_and_truncated_upload_is_removed(application):
    result = application.media.upload(io.BytesIO(b"audio"), 5, "../audio.wav")
    path = Path(result["path"])
    assert path.name == "audio.wav"
    assert path.read_bytes() == b"audio"
    assert application.media.resolve(result["url"].split("/")[-1])[0] == path
    with pytest.raises(ValueError):
        application.media.upload(io.BytesIO(b"a"), 3, "bad.wav")
    assert not list(path.parent.parent.rglob("bad.wav"))
    with pytest.raises(FileNotFoundError):
        application.media.resolve("../secrets.json")


def test_batch_queue_scan_save_edit_reorder_and_subtitles(application, tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    for name in ("01.wav", "02.wav"):
        sf.write(folder / name, np.zeros(16000), 16000)
    (folder / "01.srt").write_text("1\n00:00:00,000 --> 00:00:00,900\nhello\n", encoding="utf-8")
    scan = application.batch.scan(str(folder))
    assert len(scan["source_payloads"]) == 2
    queue = application.batch.save(
        str(folder), scan["selected_edition"], scan["source_payloads"], "audio", "merged"
    )
    assert len(queue) == 1
    identifier = queue[0]["plan_id"]
    edited = application.batch.edit(identifier)
    assert edited["mode"] == "audio"
    assert edited["source_payloads"] == scan["source_payloads"]
    assert application.batch.reorder([identifier]) == queue
    with pytest.raises(ProjectError):
        application.batch.reorder([])
    assert application.batch.remove(identifier) == []


def test_native_api_validates_before_calling_service(application, monkeypatch):
    api = API(application)
    called = []
    api.routes["projects/table"] = (api.routes["projects/table"][0], lambda **kw: called.append(kw))
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        api.call("projects/table", {"project": "p", "revision": "1", "rows": []})
    assert called == []
    with pytest.raises(ValidationError):
        api.call("tasks/start", {"kind": "analyze"})
    assert "parameters" in api.call("bootstrap", {})
    with pytest.raises(KeyError):
        api.call("unrecognized", {})


def test_model_catalog_uses_registry_numbers(application, monkeypatch):
    monkeypatch.setattr(
        models.runtime,
        "backend_status",
        lambda *a, **k: models.runtime.BackendStatus("ready", "ready"),
    )
    result = models.catalog()
    from asmr_dubber.model_registry import ASR_BACKENDS

    item = next(item for item in result["items"] if item["id"] == "parakeet_nemo")
    assert item["disk_gb"] == ASR_BACKENDS["parakeet_nemo"].disk_gb
    assert item["vram_gb"] == ASR_BACKENDS["parakeet_nemo"].recommended_vram_gb


def test_model_delete_refuses_external_path(application, monkeypatch, tmp_path):
    external = tmp_path / "outside-models"
    external.mkdir()
    (external / "weights.bin").write_bytes(b"keep")
    monkeypatch.setattr(models, "cached_model_path", lambda *_: external)
    with pytest.raises(ValueError):
        models.remove("faster_whisper")
    assert (external / "weights.bin").read_bytes() == b"keep"
