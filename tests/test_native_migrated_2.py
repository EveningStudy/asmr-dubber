import time
from importlib.resources import files
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import sha256_file
from asmr_dubber.http_server import remote_auth as _remote_auth
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.services import model_status, parameters
from asmr_dubber.services.application import Application
from asmr_dubber.services.model_status import offline_model_pack_markdown

FRONTEND = files("asmr_dubber").joinpath("frontend")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return Application(tmp_path / "tasks.json")


def wait_task(app, identifier):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = app.tasks.get(identifier)
        if task["status"] not in {"queued", "running", "cancelling"}:
            return task
        time.sleep(0.01)
    pytest.fail("Task did not finish")


def _project(tmp_path: Path) -> tuple[DubProject, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 10, dtype=np.float32), 16_000, subtype="FLOAT")
    project = DubProject(
        source=AudioInfo(
            path=source.name,
            sha256=sha256_file(source),
            duration_seconds=10.0,
            sample_rate=16_000,
            channels=1,
        ),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.0,
                end_seconds=0.5,
                ja_text="あ、",
                zh_text="啊，",
            ),
            Sentence(
                id="s000002",
                start_seconds=1.0,
                end_seconds=7.0,
                ja_text="これは十分に長くて明瞭な参考文章です。",
                zh_text="这是一句足够长而清晰的参考句。",
            ),
        ],
    )
    save_project(project, tmp_path)
    return load_project(tmp_path / "project.json")[0], tmp_path / "project.json"


def test_tts_detail_visibility_tracks_active_reference_mode(app):
    by_key = {item["key"]: item for item in parameters.catalog()}

    def shown(
        backend, speaker="project_reference", emotion="sentence_reference", source="external"
    ):
        values = {
            "tts_backend": backend,
            "tts_reference_source": source,
            "tts_index_speaker_source": speaker,
            "tts_index_emotion_source": emotion,
        }
        return [
            parameters.visible(by_key[key], values)
            for key in [
                "tts_external_reference_audio",
                "tts_external_reference_text",
                "tts_external_reference_language",
                "tts_index_external_emotion_audio",
                "tts_index_emo_text",
            ]
        ]

    assert shown("gpt_sovits") == [True, True, True, False, False]
    assert shown("cosyvoice") == [True, False, False, False, False]
    assert shown("indextts2", "external", "external", "project_sentence") == [
        True,
        False,
        False,
        True,
        False,
    ]
    assert shown("indextts2", "project_reference", "text", "project_sentence") == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_new_tts_backend_controls_only_show_relevant_options(app):
    by_key = {item["key"]: item for item in parameters.catalog()}

    def shown(key, backend):
        return parameters.visible(by_key[key], {"tts_backend": backend})

    assert shown("tts_device", "indextts2") and not shown("tts_device", "edge_tts")
    assert parameters.choices()["tts"]["edge_tts"]["default_model"] == "edge-tts"
    assert parameters.choices()["tts"]["edge_tts"]["default_voice"] == "zh-CN-XiaoxiaoNeural"
    assert shown("tts_voice", "edge_tts") and shown("tts_volume", "edge_tts")
    assert not shown("tts_pitch", "edge_tts") and not shown("tts_emotion", "edge_tts")
    assert shown("tts_style_prompt", "mimo_tts") and not shown("tts_volume", "mimo_tts")
    assert shown("tts_volume", "minimax") and shown("tts_pitch", "minimax")
    assert parameters.choices()["tts"]["minimax"]["default_voice"] == "female-shaonv"


def test_download_controller_pauses_only_active_download(app):
    def execute(request, report, token, reference):
        token.wait(2)
        from asmr_dubber.task_control import check_cancelled

        check_cancelled(token)
        return {}

    app.tasks.execute = execute
    task = app.start({"kind": "download", "model": "kotoba_whisper"})
    app.tasks.cancel(task["id"])
    assert wait_task(app, task["id"])["status"] == "cancelled"
    assert app.tasks.cancel(task["id"])["status"] == "cancelled"


def test_backend_install_log_stream_yields_output_and_heartbeats(app):
    release = Event()

    def execute(request, report, token, reference):
        report("下载第一部分", 1, 2)
        assert release.wait(2)
        report("下载第二部分", 2, 2)
        return {"message": "安装完成"}

    app.tasks.execute = execute
    task = app.start({"kind": "download", "model": "test_backend"})
    deadline = time.monotonic() + 2
    while not app.tasks.get(task["id"])["logs"] and time.monotonic() < deadline:
        time.sleep(0.01)
    initial = app.tasks.get(task["id"])
    assert initial["status"] == "running" and initial["logs"] == ["下载第一部分"]
    assert initial["current"] == 1 and initial["total"] == 2
    release.set()
    result = wait_task(app, task["id"])
    assert result["logs"] == ["下载第一部分", "下载第二部分"]
    assert result["result"]["message"] == "安装完成"
    assert result["status"] == "completed"


def test_non_loopback_ui_always_requires_authentication(monkeypatch) -> None:
    monkeypatch.delenv("ASMR_DUBBER_UI_PASSWORD", raising=False)
    assert _remote_auth("127.0.0.1") is None
    username, password = _remote_auth("0.0.0.0")
    assert username == "asmr"
    assert len(password) >= 20


def test_launch_exposes_only_ui_stage_directory(app, tmp_path):
    owned = tmp_path / "owned.wav"
    owned.write_bytes(b"media")
    unknown = tmp_path / "private.txt"
    unknown.write_text("private")
    url = app.media.register(owned)
    assert app.media.resolve(url.split("/")[-1])[0] == owned
    with pytest.raises(FileNotFoundError):
        app.media.resolve(str(unknown))
    with pytest.raises(FileNotFoundError):
        app.media.resolve("../private.txt")


def test_offline_model_pack_status_names_inbox(monkeypatch, tmp_path):
    inbox = tmp_path / "model-packs"
    monkeypatch.setattr(model_status, "model_pack_directory", lambda: inbox)
    text = offline_model_pack_markdown()
    assert str(inbox) in text and "未发现 ZIP" in text
