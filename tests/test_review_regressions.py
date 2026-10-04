import io
import json
import os
import time

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.services import models, project_records
from asmr_dubber.services.application import Application
from asmr_dubber.services.media import STALE_UPLOAD_SECONDS, MediaStore, discard_upload
from asmr_dubber.services.tasks import ACTIVE, HISTORY, Tasks


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def finished(tasks, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        task = tasks.get(identifier)
        if task["status"] not in ACTIVE:
            return task
        time.sleep(0.01)
    pytest.fail("Task failed to finish")


def test_finished_task_history_is_bounded(tmp_path):
    tasks = Tasks(lambda request, report, token, reference: {"ok": True}, tmp_path / "tasks.json")
    for index in range(HISTORY + 5):
        finished(tasks, tasks.start({"kind": "health"}, f"resource-{index}")["id"])
    assert len(tasks.list()) <= HISTORY + 1
    assert len(Tasks(lambda *arguments: None, tmp_path / "tasks.json").list()) <= HISTORY


def test_project_task_result_does_not_persist_project_snapshot(home, tmp_path):
    application = Application(tmp_path / "tasks.json")
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000), 16000)
    task = application.start({"kind": "create", "source": str(audio), "source_language": "en"})
    result = finished(application.tasks, task["id"])
    assert result["status"] == "completed", result
    assert set(result["result"]) == {"manifest"}


def test_consumed_upload_is_removed_and_other_paths_are_untouched(home, tmp_path):
    store = MediaStore()
    upload = store.upload(io.BytesIO(b"data"), 4, "clip.wav")
    outside = tmp_path / "keep.wav"
    outside.write_bytes(b"data")
    discard_upload(outside)
    discard_upload(upload["path"])
    assert outside.is_file()
    assert not list((home / "temp" / "uploads").iterdir())


def test_stale_uploads_are_removed_on_startup(home):
    upload = MediaStore().upload(io.BytesIO(b"data"), 4, "clip.wav")
    recent = MediaStore().upload(io.BytesIO(b"data"), 4, "new.wav")
    directory = os.path.dirname(upload["path"])
    old = time.time() - STALE_UPLOAD_SECONDS - 60
    os.utime(directory, (old, old))
    MediaStore()
    assert not os.path.exists(directory)
    assert os.path.isfile(recent["path"])


def test_media_registration_is_stable_for_the_same_file(home, tmp_path):
    store = MediaStore()
    file = tmp_path / "a.wav"
    file.write_bytes(b"data")
    assert store.register(file) == store.register(str(file))


def test_unreadable_project_does_not_hide_the_other_projects(home, tmp_path, monkeypatch):
    application = Application(tmp_path / "tasks.json")
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000), 16000)
    task = application.start({"kind": "create", "source": str(audio), "source_language": "en"})
    manifest = finished(application.tasks, task["id"])["result"]["manifest"]
    broken = tmp_path / "broken" / "project.json"
    broken.parent.mkdir()
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        project_records,
        "recent_projects",
        lambda root=None: [("broken", str(broken)), ("good", manifest)],
    )
    listed = application.projects.list()
    assert listed[0]["manifest"] == str(broken) and listed[0]["error"]
    assert listed[1]["manifest"] == manifest and "error" not in listed[1]


def test_separation_model_needs_every_recorded_file_and_removal_keeps_shared_ones(home):
    root = models.model_directory()
    root.mkdir(parents=True)
    (root / "htdemucs.yaml").write_text("models: ['a']", encoding="utf-8")

    def state(identifier):
        items = models.catalog()["items"]
        return next(item["state"] for item in items if item["id"] == identifier)

    assert state("separation_demucs") == "missing"
    records = {"htdemucs.yaml": "x", "a.th": "x", "download_checks.json": "x"}
    (root / "htdemucs.yaml.integrity.json").write_text(json.dumps(records), encoding="utf-8")
    (root / "download_checks.json").write_text("{}", encoding="utf-8")
    assert state("separation_demucs") == "missing"
    (root / "a.th").write_bytes(b"weights")
    assert state("separation_demucs") == "ready"

    other = {"htdemucs_ft.yaml": "x", "download_checks.json": "x"}
    (root / "htdemucs_ft.yaml").write_text("models: ['b']", encoding="utf-8")
    (root / "htdemucs_ft.yaml.integrity.json").write_text(json.dumps(other), encoding="utf-8")
    models.remove("separation_demucs")
    assert not (root / "a.th").exists() and not (root / "htdemucs.yaml").exists()
    assert not (root / "htdemucs.yaml.integrity.json").exists()
    assert (root / "download_checks.json").is_file()
    assert state("separation_demucs_ft") == "ready"


def test_optional_model_skips_backend_install_when_runtime_modules_exist(home, monkeypatch):
    installed = []
    monkeypatch.setattr(
        models.runtime, "install_backend", lambda *a, **k: installed.append(a) or "ok"
    )
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        "asmr_dubber.model_pack_download.prepare_remote_model_pack", lambda *a, **k: "pack.zip"
    )
    monkeypatch.setattr("asmr_dubber.model_packs.import_model_pack", lambda *a, **k: None)

    class Token:
        def is_set(self):
            return False

    models._download({"model": "asmr_vad"}, lambda *a, **k: None, Token())
    assert installed == []
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: None)
    models._download({"model": "asmr_vad"}, lambda *a, **k: None, Token())
    assert [call[0] for call in installed] == ["kotoba_whisper"]


@pytest.fixture
def server(home, tmp_path):
    import threading

    from asmr_dubber.http_server import Server

    server = Server(("127.0.0.1", 0), Application(tmp_path / "tasks.json"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(3)


def test_remote_host_names_need_authentication_and_loopback_keeps_its_allow_list(server):
    import httpx

    backend, url = server
    foreign = {"Host": f"192.168.1.5:{backend.server_port}"}
    with httpx.Client(trust_env=False) as client:
        assert client.get(url + "/", headers=foreign).status_code == 403
        backend.auth = ("asmr", "secret")
        assert client.get(url + "/", headers=foreign).status_code == 401
        assert client.get(url + "/", headers=foreign, auth=("asmr", "wrong")).status_code == 401
        assert client.get(url + "/", headers=foreign, auth=backend.auth).status_code == 200
        cross_origin = {**foreign, "Origin": "http://evil.example", "X-ASMR-Token": backend.token}
        response = client.post(
            url + "/api/settings/update",
            json={"changes": {"tts_speed": 1.4}},
            headers=cross_origin,
            auth=backend.auth,
        )
        assert response.status_code == 403
