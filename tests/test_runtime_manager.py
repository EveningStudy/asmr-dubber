from __future__ import annotations

import os
import sys
from pathlib import Path
from subprocess import CompletedProcess
from threading import Event

import pytest

from asmr_dubber.constants import (
    DEFAULT_ALIGNER_MODEL,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
)
from asmr_dubber.errors import EnvironmentError as AppEnvironmentError
from asmr_dubber.model_registry import ASR_BACKENDS, TTS_BACKENDS
from asmr_dubber.platforms import PlatformInfo
from asmr_dubber.runtime_manager import (
    BackendStatus,
    HardwareProfile,
    _run_streaming_process,
    available_asr_review_choices,
    available_timestamp_review_choices,
    backend_catalog_rows,
    backend_model_status,
    backend_status,
    compatibility_note,
    download_backend_models,
    hardware_markdown,
    install_backend,
    ordered_backends,
    recommended_stack_markdown,
)
from asmr_dubber.user_settings import UserSettings


@pytest.fixture(autouse=True)
def _disable_remote_model_pack_download(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.prepare_remote_model_pack",
        lambda *_args, **_kwargs: None,
    )
    # A developer checkout may legitimately contain multi-gigabyte offline
    # packs in the real inbox. Unit tests must never import those into their
    # temporary homes unless a test explicitly replaces this stub.
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda *_args, **_kwargs: 0,
    )


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
        compute_capability="8.6",
    )
    text = hardware_markdown(profile)
    assert "Windows" in text
    assert "Test GPU" in text
    assert "12.0 GB" in text
    assert "计算能力 8.6" in text
    assert "`cuda`" in text
    assert "质量优先" in recommended_stack_markdown(profile)


def test_legacy_nvidia_architecture_is_not_recommended_for_full_cuda_stack() -> None:
    profile = HardwareProfile(
        platform_id="windows",
        platform_label="Windows",
        architecture="AMD64",
        cpu="Test CPU",
        memory_gb=32,
        gpu="Legacy GPU",
        vram_gb=24,
        driver="999.1",
        cuda_available=True,
        compute_capability="6.1",
    )

    assert profile.full_cuda_stack_supported is False
    assert "Turing" in hardware_markdown(profile)
    assert "旧架构 NVIDIA" in recommended_stack_markdown(profile)


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
    assert "显存不足" in compatibility_note(TTS_BACKENDS["indextts2"], profile)


def test_http_backend_is_external_service(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Linux", "x86_64", False, True, False),
    )
    status = backend_status(TTS_BACKENDS["gpt_sovits"])
    assert status.state == "external"


def test_edge_tts_status_reflects_bundled_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Linux", "x86_64", False, True, False),
    )
    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: True)

    ready = backend_status(TTS_BACKENDS["edge_tts"])

    assert ready.state == "ready"
    assert ready.label == "可用"
    assert "无需 API Key" in ready.detail

    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: False)
    missing = backend_status(TTS_BACKENDS["edge_tts"])
    assert missing.state == "missing"
    assert "Setup" in missing.detail


def test_windows_kotoba_reports_missing_cuda_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setattr("asmr_dubber.runtime_manager.detect_hardware", lambda: _hardware(cuda=True))
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cuda_summary",
        lambda: {"available": False},
    )
    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: True)

    status = backend_status(ASR_BACKENDS["kotoba_whisper"])

    assert status.state == "partial"
    assert "CUDA" in status.label


def test_parakeet_model_status_is_reported_per_concrete_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setattr("asmr_dubber.runtime_manager.portable_home", lambda: tmp_path)
    executable = tmp_path / "runtimes" / "crispasr" / "bin" / "crispasr.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model = tmp_path / "models" / "parakeet" / "parakeet-ctc-1.1b-ja-f16.gguf"
    model.parent.mkdir(parents=True)
    model.touch()
    backend = ASR_BACKENDS["parakeet_nemo"]

    first = backend_model_status(backend, backend.models[0])
    second = backend_model_status(backend, backend.models[1])
    choices = available_asr_review_choices()

    assert first.state == "ready"
    assert second.state == "missing"
    values = [value for _, value in choices]
    assert f"parakeet_nemo|{backend.models[0]}" in values
    assert f"parakeet_nemo|{backend.models[1]}" not in values


def test_timestamp_choices_show_qwen_aligner_only_when_locally_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path",
        lambda model_id: tmp_path if model_id == DEFAULT_ALIGNER_MODEL else None,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.importlib.util.find_spec",
        lambda module: object() if module == "qwen_asr" else None,
    )

    values = [value for _, value in available_timestamp_review_choices()]

    assert f"qwen_forced_aligner|{DEFAULT_ALIGNER_MODEL}" in values


def test_backend_catalog_can_be_split_by_kind(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.detect_hardware",
        lambda: _hardware(cuda=True),
    )

    asr_rows = backend_catalog_rows(UserSettings(), "asr")
    tts_rows = backend_catalog_rows(UserSettings(), "tts")

    assert len(asr_rows) == len(ASR_BACKENDS)
    assert len(tts_rows) == len(TTS_BACKENDS)
    assert all(len(row) == 6 for row in [*asr_rows, *tts_rows])


def test_backend_order_places_ready_recommended_first_and_generic_last(monkeypatch) -> None:
    states = {
        "parakeet_nemo": BackendStatus("ready", "可用"),
        "kotoba_whisper": BackendStatus("missing", "未安装"),
        "faster_whisper": BackendStatus("ready", "可用"),
        "generic_asr_api": BackendStatus("external", "外部服务"),
    }
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.backend_status",
        lambda backend, settings=None: states[backend.id],
    )

    ordered = [backend.id for backend in ordered_backends(ASR_BACKENDS, UserSettings())]

    assert ordered[:2] == ["parakeet_nemo", "faster_whisper"]
    assert ordered[-1] == "generic_asr_api"


def test_tts_backend_order_keeps_generic_api_last(monkeypatch) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.backend_status",
        lambda backend, settings=None: BackendStatus(
            "ready" if backend.id in {"indextts2", "edge_tts"} else "external",
            "可用",
        ),
    )

    ordered = [backend.id for backend in ordered_backends(TTS_BACKENDS, UserSettings())]

    assert ordered[:2] == ["indextts2", "edge_tts"]
    assert ordered[-1] == "generic_tts_api"


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
    monkeypatch.setattr("asmr_dubber.runtime_manager._run_streaming_process", fake_run)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Linux", "x86_64", False, True, False),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda *_args, **_kwargs: "models ready",
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda *_args, **_kwargs: 0,
    )

    result = install_backend("faster_whisper", force=True)
    assert "安装完成" in result
    assert calls[0][0] == str(uv)
    assert any("asr-faster-whisper" in argument for argument in calls[0])


def test_windows_kotoba_uses_runtime_installer(monkeypatch, tmp_path: Path) -> None:
    uv = tmp_path / "uv.exe"
    uv.touch()
    calls: list[str] = []

    def fake_install(backend_id: str, **_kwargs):
        calls.append(backend_id)
        return CompletedProcess(["pwsh"], 0, stdout="CUDA runtime ready", stderr="")

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager._install_windows_backend", fake_install)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda *_args, **_kwargs: "models ready",
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda *_args, **_kwargs: 0,
    )

    result = install_backend("kotoba_whisper", force=True)

    assert calls == ["kotoba_whisper"]
    assert "CUDA runtime ready" in result


def test_windows_advanced_backend_installs_dependencies_before_model_pack(
    monkeypatch,
    tmp_path: Path,
) -> None:
    uv = tmp_path / "uv.exe"
    uv.touch()
    order: list[str] = []

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager._install_windows_backend",
        lambda backend_id, **_kwargs: (
            order.append(f"dependencies:{backend_id}")
            or CompletedProcess(["pwsh"], 0, stdout="advanced ready", stderr="")
        ),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda backend_id, **_kwargs: order.append(f"model:{backend_id}") or 1,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda *_args, **_kwargs: "models ready",
    )

    result = install_backend("kotoba_whisper", force=True)

    assert order == ["dependencies:kotoba_whisper", "model:kotoba_whisper"]
    assert "安装完成" in result


def test_install_backend_imports_matching_pack_before_network(monkeypatch) -> None:
    statuses = iter(
        [
            type("Status", (), {"state": "missing"})(),
            type("Status", (), {"state": "ready"})(),
        ]
    )
    imported: list[str] = []

    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.backend_status",
        lambda *_args, **_kwargs: next(statuses),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda backend_id, **_kwargs: imported.append(backend_id) or 1,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager._uv_executable",
        lambda: pytest.fail("a complete local model pack must be used before network install"),
    )

    result = install_backend("parakeet_nemo")

    assert imported == ["parakeet_nemo"]
    assert "本地模型包" in result


def test_download_backend_models_verifies_each_snapshot(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    downloaded: set[str] = set()

    def fake_download(*, repo_id: str, revision: str, **_kwargs):
        calls.append((repo_id, revision))
        downloaded.add(repo_id)
        return str(tmp_path / repo_id)

    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.snapshot_download_with_fallback",
        fake_download,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path",
        lambda model_id: tmp_path / model_id if model_id in downloaded else None,
    )

    result = download_backend_models("faster_whisper")

    assert len(calls) == 1
    assert "faster-whisper-large-v2" in result


def test_download_backend_models_reuses_imported_pack_without_network(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = "kotoba-tech/kotoba-whisper-v2.2"
    local = tmp_path / "models" / "kotoba-whisper-v2.2"
    logs: list[str] = []
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path",
        lambda model_id: local if model_id == model else None,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.snapshot_download_with_fallback",
        lambda **_kwargs: pytest.fail("an imported offline model must not access the network"),
    )

    result = download_backend_models("kotoba_whisper", log_callback=logs.append)

    assert str(local) in result
    assert any("复用已导入或已缓存的模型" in message for message in logs)


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
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda *_args, **_kwargs: 0,
    )

    result = install_backend("indextts2", force=True)

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
        install_backend("indextts2", force=True)


def test_streaming_installer_forwards_stdout_and_stderr(tmp_path: Path) -> None:
    logs: list[str] = []
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys; "
            "print('first line', flush=True); "
            "print('second line', file=sys.stderr, flush=True)"
        ),
    ]

    completed = _run_streaming_process(
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_callback=logs.append,
    )

    assert completed.returncode == 0
    assert logs == ["first line", "second line"]
    assert "first line" in completed.stdout
    assert "second line" in completed.stdout


def test_streaming_installer_can_be_paused(tmp_path: Path) -> None:
    cancel_event = Event()
    logs: list[str] = []
    command = [
        sys.executable,
        "-u",
        "-c",
        "import time; print('started', flush=True); time.sleep(30)",
    ]

    def pause_after_start(message: str) -> None:
        logs.append(message)
        if message == "started":
            cancel_event.set()

    with pytest.raises(AppEnvironmentError, match="下载已暂停"):
        _run_streaming_process(
            command,
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=10,
            log_callback=pause_after_start,
            cancel_event=cancel_event,
        )

    assert logs == ["started"]


def test_ready_parakeet_returns_without_running_installer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setattr("asmr_dubber.runtime_manager.portable_home", lambda: tmp_path)
    executable = tmp_path / "runtimes" / "crispasr" / "bin" / "crispasr.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model_root = tmp_path / "models" / "parakeet"
    model_root.mkdir(parents=True)
    (model_root / "parakeet-ctc-1.1b-ja-f16.gguf").touch()
    (model_root / "parakeet-tdt-0.6b-ja.gguf").touch()
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager._uv_executable",
        lambda: pytest.fail("ready fast-path must not invoke the installer"),
    )
    logs: list[str] = []

    result = install_backend("parakeet_nemo", log_callback=logs.append)

    assert "已经可用" in result
    assert logs == [result]


def test_force_reinstalls_ready_parakeet(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.current_platform",
        lambda: PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setattr("asmr_dubber.runtime_manager.portable_home", lambda: tmp_path)
    executable = tmp_path / "runtimes" / "crispasr" / "bin" / "crispasr.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model_root = tmp_path / "models" / "parakeet"
    model_root.mkdir(parents=True)
    (model_root / "parakeet-ctc-1.1b-ja-f16.gguf").touch()
    (model_root / "parakeet-tdt-0.6b-ja.gguf").touch()
    uv = tmp_path / "uv.exe"
    uv.touch()
    calls: list[dict[str, object]] = []

    def fake_install(**kwargs):
        calls.append(kwargs)
        return CompletedProcess(["installer"], 0, stdout="Parakeet ready", stderr="")

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager._install_parakeet_runtime", fake_install)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda *_args, **_kwargs: 0,
    )

    result = install_backend("parakeet_nemo", force=True)

    assert calls
    assert "安装完成" in result
