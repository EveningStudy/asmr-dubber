from __future__ import annotations

import asyncio
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dubber import edge_tts_client as client
from asmr_dubber.errors import OperationCancelledError, ProjectError, SynthesisError
from asmr_dubber.model_registry import TTS_BACKENDS
from asmr_dubber.runtime_manager import backend_status
from asmr_dubber.services.project_audio import preview_edge_tts_voice
from asmr_dubber.task_control import CancellationToken, cancellation_scope


@pytest.mark.parametrize(
    ("installed", "compatible"),
    [
        ("7.2.8", False),
        ("7.3.0", False),
        ("7.3.1", True),
        ("7.4.0", True),
        ("8.0.0", False),
        (None, False),
    ],
)
def test_client_version_gate(monkeypatch, installed, compatible) -> None:
    monkeypatch.setattr(client, "version", lambda _name: installed)
    assert (client.compatibility_problem() is None) == compatible


def test_missing_or_old_client_is_not_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: True)
    monkeypatch.setattr(client, "version", lambda _name: "7.2.8")
    assert backend_status(TTS_BACKENDS["edge_tts"]).state == "broken"
    with pytest.raises(SynthesisError, match=r"7\.3\.1"):
        client.save_audio("测试", "zh-CN-XiaoxiaoNeural", tmp_path / "test.mp3")

    def missing(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(client, "version", missing)
    assert "未安装" in str(client.compatibility_problem())


def _fake_client(monkeypatch, save, **exceptions) -> None:
    class Communicate:
        def __init__(self, *_args, **_kwargs):
            pass

    Communicate.save = save
    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        SimpleNamespace(Communicate=Communicate, exceptions=SimpleNamespace(**exceptions)),
    )
    monkeypatch.setattr(client, "version", lambda _name: "7.3.1")


def test_authentication_failure_is_specific_and_not_retried(monkeypatch, tmp_path: Path) -> None:
    class Rejected(OSError):
        status = 403

    calls = []

    async def save(_self, path):
        calls.append(path)
        Path(path).write_bytes(b"partial")
        raise Rejected("Forbidden")

    _fake_client(monkeypatch, save)
    output = tmp_path / "temporary.mp3"
    cached = tmp_path / "successful.wav"
    cached.write_bytes(b"retained")
    with pytest.raises(SynthesisError, match=r"鉴权.*403"):
        client.save_audio("测试", "zh-CN-XiaoxiaoNeural", output)
    assert len(calls) == 1
    assert not output.exists()
    assert cached.read_bytes() == b"retained"


@pytest.mark.parametrize("error_name", ["NoAudioReceived", "RateLimited"])
def test_preview_retries_service_failures_and_caches_only_success(
    monkeypatch, tmp_path: Path, error_name: str
) -> None:
    error = type(error_name, (OSError,), {"status": 429 if error_name == "RateLimited" else None})
    calls = []
    delays = []

    async def save(_self, path):
        calls.append(path)
        if len(calls) < 3:
            Path(path).write_bytes(b"partial")
            raise error("temporary")
        assert not Path(path).exists()
        Path(path).write_bytes(b"success")

    async def delay(seconds):
        delays.append(seconds)

    _fake_client(monkeypatch, save, **{error_name: error})
    monkeypatch.setattr(client, "_retry_delay", delay)
    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: tmp_path)
    first = Path(preview_edge_tts_voice("zh-CN-XiaoxiaoNeural"))
    assert Path(preview_edge_tts_voice("zh-CN-XiaoxiaoNeural")) == first
    assert first.read_bytes() == b"success"
    assert len(calls) == 3
    assert delays == [1, 2]
    assert not list(first.parent.glob(".*.tmp.mp3"))


def test_silent_connection_has_outer_deadline(monkeypatch, tmp_path: Path) -> None:
    stopped = []

    async def save(_self, path):
        Path(path).write_bytes(b"partial")
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    _fake_client(monkeypatch, save)
    output = tmp_path / "temporary.mp3"
    started = time.monotonic()
    with pytest.raises(SynthesisError, match="超时"):
        client.save_audio("测试", "zh-CN-XiaoxiaoNeural", output, timeout_seconds=0.1)
    assert time.monotonic() - started < 2
    assert stopped == [True]
    assert not output.exists()


@pytest.mark.parametrize("signal_type", [CancellationToken, threading.Event])
@pytest.mark.parametrize("during_retry", [False, True])
def test_cancellation_stops_requests_and_backoff(
    monkeypatch, tmp_path: Path, signal_type, during_retry: bool
) -> None:
    signal = signal_type()
    calls = []
    stopped = []

    async def save(_self, path):
        calls.append(path)
        Path(path).write_bytes(b"partial")
        if during_retry:
            signal.set()
            raise OSError("retry needed")
        asyncio.get_running_loop().call_later(0.05, signal.set)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    _fake_client(monkeypatch, save)
    output = tmp_path / "temporary.mp3"
    started = time.monotonic()
    with cancellation_scope(signal), pytest.raises(OperationCancelledError):
        client.save_audio("测试", "zh-CN-XiaoxiaoNeural", output, timeout_seconds=10)
    assert time.monotonic() - started < 2
    assert len(calls) == 1
    assert bool(stopped) != during_retry
    assert not output.exists()


def test_failed_preview_is_not_cached(monkeypatch, tmp_path: Path) -> None:
    async def save(_self, path):
        Path(path).write_bytes(b"partial")
        raise ValueError("invalid response")

    _fake_client(monkeypatch, save)
    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: tmp_path)
    with pytest.raises(ProjectError, match="invalid response"):
        preview_edge_tts_voice("en-US-AriaNeural")
    assert not list(tmp_path.rglob("*.mp3"))
