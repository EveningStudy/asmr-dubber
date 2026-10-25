import hashlib
import json
import threading

import httpx
import pytest
from pydantic import ValidationError

from asmr_dubber import moss
from asmr_dubber.errors import AsmrDubberError, InstallPausedError
from asmr_dubber.models import ProjectSettings, Sentence, settings_for_source_language
from asmr_dubber.moss_worker import parse_transcript
from asmr_dubber.services.parameters import catalog, visible
from asmr_dubber.task_control import CancellationToken


@pytest.fixture
def download_environment(tmp_path, monkeypatch):
    data = b"test-model-data"
    monkeypatch.setattr(moss, "model_directory", lambda: tmp_path)
    monkeypatch.setattr(
        moss,
        "manifest",
        lambda: {
            "revision": "test",
            "source_revision": "source",
            "files": {
                "model.safetensors": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            },
        },
    )
    monkeypatch.setattr(
        moss,
        "download_urls",
        lambda name: [
            "https://modelscope.cn/" + name,
            "https://huggingface.co/" + name,
        ],
    )
    real_client = httpx.Client

    def route(handler):
        monkeypatch.setattr(
            moss.httpx,
            "Client",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            ),
        )

    return tmp_path, data, route


def test_mirror_first_and_no_network_when_already_verified(download_environment):
    root, data, route = download_environment
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200, content=data)

    route(handler)
    assert moss.download_model(log=lambda _: None) == root
    moss.download_model(log=lambda _: None)
    assert calls == ["modelscope.cn"]


def test_fallback_and_resume_when_mirror_unavailable(download_environment):
    root, data, route = download_environment
    (root / "model.safetensors.partial").write_bytes(data[:4])
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "modelscope.cn":
            return httpx.Response(503)
        assert request.headers["range"] == "bytes=4-"
        return httpx.Response(
            206,
            content=data[4:],
            headers={
                "Content-Range": f"bytes 4-{len(data) - 1}/{len(data)}",
            },
        )

    route(handler)
    moss.download_model(log=lambda _: None)
    assert calls == ["modelscope.cn", "modelscope.cn", "huggingface.co"]
    assert (root / "model.safetensors").read_bytes() == data
    assert not (root / "model.safetensors.partial").exists()


def test_corrupt_mirror_falls_back_and_never_publishes_bad_weights(download_environment):
    root, data, route = download_environment
    route(
        lambda request: httpx.Response(
            200, content=(b"x" * len(data) if request.url.host == "modelscope.cn" else data)
        )
    )
    moss.download_model(log=lambda _: None)
    assert (root / "model.safetensors").read_bytes() == data


def test_failed_sources_do_not_publish(download_environment):
    root, _, route = download_environment
    route(lambda request: httpx.Response(200, content=b"bad"))
    with pytest.raises(AsmrDubberError, match="SHA-256"):
        moss.download_model(log=lambda _: None)
    assert not (root / "model.safetensors").exists()


def test_cancelled_download_preserves_partial(download_environment):
    root, _, route = download_environment
    partial = root / "model.safetensors.partial"
    partial.write_bytes(b"test")
    route(lambda request: pytest.fail("cancelled downloads must not request a URL"))
    with pytest.raises(InstallPausedError):
        moss.download_model(cancelled=lambda: True)
    assert partial.read_bytes() == b"test"


def test_parser_keeps_speaker_turns_and_rejects_incomplete_or_bad_timestamps():
    rows = parse_transcript("[0.1][S01]Hello[1.2][1.0][S02]Hi![1.8]", 2)
    assert [row["speaker"] for row in rows] == ["S01", "S02"]
    assert rows[1]["start"] == 1  # Simultaneous speakers may overlap.
    for text in ("[0][S01]unfinished", "[2][S01]bad[1]", "intro[0][S01]bad[1]", "[3][S01]late[4]"):
        with pytest.raises(ValueError):
            parse_transcript(text, 2)


def test_moss_settings_survive_languages_and_are_all_in_the_ui():
    settings = ProjectSettings(asr_backend="moss_transcribe_diarize", asr_model=moss.MODEL_ID)
    for language in ("ja", "en", "zh"):
        assert settings_for_source_language(settings, language).asr_backend == settings.asr_backend
    parameters = [item for item in catalog() if item["key"].startswith("asr_moss_")]
    assert len(parameters) == 17
    values = settings.model_dump()
    values["asr_moss_do_sample"] = True
    assert all(visible(item, values) for item in parameters)
    values["asr_backend"] = "faster_whisper"
    assert not any(visible(item, values) for item in parameters)
    row = Sentence(
        id="s1", start_seconds=0, end_seconds=1, source_text="Hi", speaker_id="c0001:S01"
    )
    assert Sentence.model_validate_json(row.model_dump_json()).speaker_id == row.speaker_id
    for params in ("[]", '{"max_new_tokens":1}', '{"return_dict_in_generate":true}'):
        with pytest.raises(ValidationError):
            ProjectSettings(asr_moss_generation_params=params)


def test_adapter_offsets_speakers_and_registers_cancellation(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    from asmr_dubber import asr

    audio = tmp_path / "audio.wav"
    sf.write(audio, np.ones(32000, dtype=np.float32) * 0.1, 16000)
    monkeypatch.setattr(moss, "runtime_python", lambda: tmp_path / "python")
    (tmp_path / "python").touch()
    monkeypatch.setattr(moss, "complete_model", lambda **kwargs: True)
    monkeypatch.setattr(asr, "portable_home", lambda: tmp_path)
    token = CancellationToken()

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            result = {"segments": [{"text": "Hi", "start": 0.1, "end": 1, "speaker": "c0001:S01"}]}
            from pathlib import Path

            Path(command[-1]).write_text(json.dumps(result))

        def poll(self):
            return 0

    monkeypatch.setattr(asr.subprocess, "Popen", Process)
    monkeypatch.setattr(asr, "terminate_process_tree", lambda process: None)
    settings = ProjectSettings(asr_backend="moss_transcribe_diarize", asr_model=moss.MODEL_ID)
    rows, language = asr.transcribe_source(audio, settings, "en", cancel_event=token)
    assert language == "en" and rows[0].speaker_id == "c0001:S01"
    assert rows[0].end_seconds == 1
    assert not token._processes


def test_adapter_already_cancelled_never_launches(tmp_path, monkeypatch):
    from asmr_dubber import asr
    from asmr_dubber.errors import OperationCancelledError

    token = threading.Event()
    token.set()
    monkeypatch.setattr(asr.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launched"))
    with pytest.raises(OperationCancelledError):
        asr.transcribe_source(tmp_path / "missing.wav", ProjectSettings(), cancel_event=token)


def test_moss_removal_and_concrete_model_status(tmp_path, monkeypatch):
    from asmr_dubber.model_registry import ASR_BACKENDS
    from asmr_dubber.runtime_manager import backend_model_status
    from asmr_dubber.services import models
    from asmr_dubber.user_settings import UserSettings

    root = tmp_path / "models" / "moss" / "revision"
    root.mkdir(parents=True)
    (root / "weights").write_bytes(b"model")
    runtime = tmp_path / "runtimes" / "moss"
    runtime.mkdir(parents=True)
    python = runtime / "python"
    python.touch()
    (runtime / "ready.json").write_text("{}")
    monkeypatch.setattr(moss, "model_directory", lambda: root)
    monkeypatch.setattr(moss, "runtime_root", lambda: runtime)
    monkeypatch.setattr(moss, "runtime_python", lambda: python)
    monkeypatch.setattr(moss, "complete_model", lambda **kwargs: root.is_dir())
    monkeypatch.setattr(models, "portable_home", lambda: tmp_path)
    monkeypatch.setattr(models, "current", UserSettings)
    settings = UserSettings(asr_backend="moss_transcribe_diarize", asr_model=moss.MODEL_ID)
    spec = ASR_BACKENDS[settings.asr_backend]
    assert backend_model_status(spec, settings.asr_model).state == "ready"
    models.remove(settings.asr_backend)
    assert not root.exists()
    assert python.exists()
    assert backend_model_status(spec, settings.asr_model).state == "partial"
    assert ProjectSettings(asr_moss_prompt="").asr_moss_prompt == moss.DEFAULT_PROMPT
