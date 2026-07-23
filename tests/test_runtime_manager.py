from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES
from asmr_dubber.errors import EnvironmentError as AppEnvironmentError
from asmr_dubber.model_registry import ASR_BACKENDS, TTS_BACKENDS
from asmr_dubber.platforms import PlatformInfo
from asmr_dubber.runtime_manager import (
    HardwareProfile,
    backend_status,
    compatibility_note,
    download_backend_models,
    hardware_markdown,
    install_backend,
    recommended_stack_markdown,
)
from asmr_dubber.user_settings import UserSettings


def _hardware(*, cuda: bool) -> HardwareProfile:
    return HardwareProfile(
        platform_id="windows",
        platform_label="Windows",
        architecture="AMD64",
        cpu="Test CPU",
        memory_gb=32,
        gpu="Test GPU" if cuda else None,
        vram_gb=12 if cuda else None,
        driver="999.1" if cuda else None,
        cuda_available=cuda,
    )


def test_hardware_markdown_is_user_facing() -> None:
    profile = HardwareProfile(
        platform_id="windows",
        platform_label="Windows",
        architecture="AMD64",
        cpu="Test CPU",
        memory_gb=32,
        gpu="Test GPU",
        vram_gb=12,
        driver="999.1",
        cuda_available=True,
    )
    text = hardware_markdown(profile)
    assert "Windows" in text
    assert "Test GPU" in text
    assert "12.0 GB" in text
    assert "`cuda`" in text
    assert "质量优先" in recommended_stack_markdown(profile)


def test_compatibility_reports_insufficient_vram() -> None:
    profile = HardwareProfile(
        platform_id="linux",
        platform_label="Linux",
        architecture="x86_64",
        cpu="CPU",
        memory_gb=16,
        gpu="Small GPU",
        vram_gb=2,
        driver="1",
        cuda_available=True,
    )
    assert "显存不足" in compatibility_note(TTS_BACKENDS["voxcpm2"], profile)


def test_http_backend_is_external_service(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Linux", "x86_64", False, True, False),
    )
    status = backend_status(ASR_BACKENDS["openai_compatible_asr"])
    assert status.state == "external"


def test_indextts_status_accepts_windows_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    root = tmp_path / "index-tts"
    model_dir = root / "checkpoints"
    model_dir.mkdir(parents=True)
    for relative in INDEXTTS_REQUIRED_FILES:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for relative in INDEXTTS_REQUIRED_DIRS:
        (model_dir / relative).mkdir(parents=True, exist_ok=True)
    executable = root / ".venv" / "Scripts" / "indextts2.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    settings = UserSettings(tts_model_path=str(model_dir))

    status = backend_status(TTS_BACKENDS["indextts2"], settings=settings)
    assert status.state == "ready"
    assert Path(status.detail).samefile(executable)


def test_install_backend_uses_current_interpreter(monkeypatch, tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.touch()
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, stdout="Installed faster-whisper", stderr="")

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager.subprocess.run", fake_run)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda *_args, **_kwargs: "models ready",
    )

    result = install_backend("faster_whisper")
    assert "安装完成" in result
    assert calls[0][0] == str(uv)
    assert "asr-faster-whisper" in calls[0][-1]


def test_windows_local_backend_uses_runtime_installer(monkeypatch, tmp_path: Path) -> None:
    uv = tmp_path / "uv.exe"
    uv.touch()
    calls: list[str] = []

    def fake_install(backend_id: str, **_kwargs):
        calls.append(backend_id)
        return CompletedProcess(["pwsh"], 0, stdout="runtime ready", stderr="")

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager._install_windows_backend", fake_install)
    monkeypatch.setattr("asmr_dubber.runtime_manager.detect_hardware", lambda: _hardware(cuda=True))
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda *_args, **_kwargs: "models ready",
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )

    result = install_backend("voxcpm2")

    assert calls == ["voxcpm2"]
    assert "runtime ready" in result


def test_download_backend_models_verifies_each_snapshot(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, revision: str, **_kwargs):
        calls.append((repo_id, revision))
        return str(tmp_path / repo_id)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path", lambda model_id: tmp_path / model_id
    )

    result = download_backend_models("qwen3_asr")

    assert len(calls) == 2
    assert "Qwen3-ASR-1.7B" in result


def test_indextts_uses_isolated_runtime_installer(monkeypatch, tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.touch()
    calls: list[dict[str, object]] = []

    def fake_install(**kwargs):
        calls.append(kwargs)
        return CompletedProcess(["installer"], 0, stdout="IndexTTS ready", stderr="")

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager._install_indextts_runtime", fake_install)
    monkeypatch.setattr("asmr_dubber.runtime_manager.detect_hardware", lambda: _hardware(cuda=True))

    result = install_backend("indextts2")

    assert calls
    assert "IndexTTS ready" in result


def test_install_cuda_only_backend_is_blocked_without_nvidia(monkeypatch, tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.touch()
    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.detect_hardware", lambda: _hardware(cuda=False)
    )

    with pytest.raises(AppEnvironmentError, match="需要 NVIDIA CUDA GPU"):
        install_backend("voxcpm2")
