from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT
from .errors import AsmrDubberError, EnvironmentError, InstallPausedError
from .model_registry import ASR_BACKENDS, TTS_BACKENDS
from .runtime_specs import _WINDOWS_SCRIPT_BACKENDS, InstallLogCallback


def _uv_executable() -> Path | None:
    from . import runtime_manager

    home = runtime_manager.portable_home()
    candidates = [
        home / "bootstrap" / "windows" / "uv" / "uv.exe",
        home / "bootstrap" / "linux" / "uv" / "uv",
        Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv"),
    ]
    discovered = shutil.which("uv")
    if discovered:
        candidates.insert(0, Path(discovered))
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _powershell_executable() -> Path | None:
    candidates: list[Path] = []
    for name in ("pwsh.exe", "pwsh", "powershell.exe", "powershell"):
        discovered = shutil.which(name)
        if discovered:
            candidates.append(Path(discovered))
    program_files = os.getenv("PROGRAMFILES", "").strip()
    system_root = os.getenv("SYSTEMROOT", "").strip()
    if program_files:
        candidates.append(Path(program_files) / "PowerShell" / "7" / "pwsh.exe")
    if system_root:
        candidates.append(
            Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _install_windows_backend(
    backend_id: str,
    *,
    timeout_seconds: float,
    env: dict[str, str],
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    from . import runtime_manager

    powershell = runtime_manager._powershell_executable()
    script = PROJECT_ROOT / "scripts" / "windows" / "install-backend.ps1"
    if powershell is None:
        raise EnvironmentError("找不到 PowerShell；Windows 自带 5.1 或 PowerShell 7 均可。")
    if not script.is_file():
        raise EnvironmentError("Windows 后端安装脚本缺失；请重新下载完整项目。")
    return runtime_manager._run_streaming_process(
        [
            str(powershell),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Backend",
            backend_id,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        timeout_seconds=timeout_seconds,
        log_callback=log_callback,
        cancel_event=cancel_event,
    )


def _install_indextts_runtime(
    backend_id: str,
    *,
    timeout_seconds: float,
    env: dict[str, str],
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    from . import runtime_manager

    info = runtime_manager.current_platform()
    if info.is_windows:
        executable = runtime_manager._powershell_executable()
        script_name = (
            "install-indextts25.ps1" if backend_id == "indextts2_5" else "install-indextts2.ps1"
        )
        script = PROJECT_ROOT / "scripts" / "windows" / script_name
        if executable is None:
            raise EnvironmentError("找不到 PowerShell；Windows 自带 5.1 或 PowerShell 7 均可。")
        command = [
            str(executable),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ]
    else:
        executable = Path(shutil.which("bash") or "")
        script_name = (
            "install-indextts25.sh" if backend_id == "indextts2_5" else "install-indextts2.sh"
        )
        script = PROJECT_ROOT / "scripts" / "linux" / script_name
        if not executable.is_file():
            raise EnvironmentError("找不到 bash，无法安装 IndexTTS2 独立运行时。")
        command = [str(executable), str(script)]
    if not script.is_file():
        raise EnvironmentError("IndexTTS2 安装脚本缺失；请重新下载完整项目。")
    return runtime_manager._run_streaming_process(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        timeout_seconds=max(timeout_seconds, 14_400),
        log_callback=log_callback,
        cancel_event=cancel_event,
    )


def _install_parakeet_runtime(
    *,
    timeout_seconds: float,
    env: dict[str, str],
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    from . import runtime_manager

    info = runtime_manager.current_platform()
    if info.is_windows:
        executable = runtime_manager._powershell_executable()
        script = PROJECT_ROOT / "scripts" / "windows" / "install-parakeet.ps1"
        if executable is None:
            raise EnvironmentError("找不到 PowerShell；Windows 自带 5.1 或 PowerShell 7 均可。")
        command = [
            str(executable),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Variant",
            "Auto",
        ]
    else:
        executable = Path(shutil.which("bash") or "")
        script = PROJECT_ROOT / "scripts" / "linux" / "install-parakeet.sh"
        if not executable.is_file():
            raise EnvironmentError("找不到 bash，无法安装 CrispASR。")
        command = [str(executable), str(script)]
    if not script.is_file():
        raise EnvironmentError("Parakeet 安装脚本缺失；请重新下载完整项目。")
    return runtime_manager._run_streaming_process(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        timeout_seconds=max(timeout_seconds, 14_400),
        log_callback=log_callback,
        cancel_event=cancel_event,
    )


def _install_backend_unlocked(
    backend_id: str,
    *,
    progress: Any | None = None,
    timeout_seconds: float = 3600,
    log_callback: InstallLogCallback | None = None,
    force: bool = False,
    cancel_event: threading.Event | None = None,
) -> str:
    """Install a verified Python backend into the current application environment."""
    from . import runtime_manager

    registry = ASR_BACKENDS if backend_id in ASR_BACKENDS else TTS_BACKENDS
    spec = registry.get(backend_id)
    if spec is None:
        raise EnvironmentError(f"未知后端：{backend_id}")
    if spec.installer is None:
        raise EnvironmentError(
            "该后端暂不支持应用内自动安装；请按照后端说明启动外部服务或独立运行时。"
        )
    if not (PROJECT_ROOT / "scripts").is_dir() or not (PROJECT_ROOT / "mirrors.json").is_file():
        raise EnvironmentError(
            "当前为 wheel/API 安装，不包含便携安装器资源。"
            "本地模型自动安装请使用完整源码或便携发行包；现有 CLI/API 功能不受影响。"
        )
    if backend_id == "moss_transcribe_diarize":
        from . import moss

        uv = runtime_manager._uv_executable()
        if uv is None:
            raise EnvironmentError("找不到 uv，请运行启动器准备安装环境。")

        def run_moss(
            command: list[str], *, env: dict[str, str]
        ) -> subprocess.CompletedProcess[str]:
            return runtime_manager._run_streaming_process(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                timeout_seconds=timeout_seconds,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )

        return moss.install_runtime(
            uv,
            run_moss,
            env=os.environ.copy(),
            log=log_callback or (lambda message: None),
            cancelled=cancel_event.is_set if cancel_event else lambda: False,
        )
    extra = spec.python_extra
    if spec.installer == "python-extra" and not extra:
        raise EnvironmentError(f"{spec.label} 的安装声明缺少 Python extra。")
    saved_settings: Any | None = None
    try:
        from .user_settings import load_user_settings

        saved_settings = load_user_settings()
        pypi_index = saved_settings.pypi_index_url.strip()
        huggingface_endpoint = saved_settings.huggingface_endpoint.strip()
    except (OSError, ValueError, AsmrDubberError):
        pypi_index = ""
        huggingface_endpoint = ""
    if not force:
        status = runtime_manager.backend_status(spec, settings=saved_settings)
        if status.state == "ready":
            result = f"{spec.label} 已经可用，无需重复安装。"
            if log_callback is not None:
                log_callback(result)
            if progress:
                progress(1, desc=result)
            return result
    if cancel_event is not None and cancel_event.is_set():
        raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
    defer_model_pack = (
        runtime_manager.current_platform().is_windows and backend_id in _WINDOWS_SCRIPT_BACKENDS
    )
    imported_packs = 0
    if not defer_model_pack:
        imported_packs = runtime_manager.import_backend_model_packs(
            backend_id,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    if cancel_event is not None and cancel_event.is_set():
        raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
    if not force and imported_packs:
        status = runtime_manager.backend_status(spec, settings=saved_settings)
        if status.state == "ready":
            result = f"{spec.label} 已从本地模型包导入并可直接使用。"
            if log_callback is not None:
                log_callback(result)
            if progress:
                progress(1, desc=result)
            return result
    uv = runtime_manager._uv_executable()
    if uv is None:
        raise EnvironmentError("找不到项目安装器 uv；请重新运行对应平台的 setup 脚本。")
    if spec.devices == ("cuda",):
        profile = runtime_manager.detect_hardware()
        if not profile.cuda_available:
            raise EnvironmentError(
                f"{spec.label} 需要 NVIDIA CUDA GPU；当前未检测到 NVIDIA GPU。"
                "请选择 CPU/外部服务后端，或在带 NVIDIA GPU 的机器上安装。"
            )
    start_message = f"正在安装 {spec.label}，请勿关闭页面。"
    if log_callback is not None:
        log_callback(start_message)
    if progress:
        progress(0, desc=start_message)
    env = os.environ.copy()
    env["ASMR_DUBBER_INSTALL_LOCK_HELD"] = "1"
    env.setdefault("UV_LINK_MODE", "copy" if os.name == "nt" else "clone")
    if imported_packs:
        env["ASMR_DUBBER_MODEL_PACKS_PREPARED"] = "1"
    if pypi_index:
        env["ASMR_DUBBER_PYPI_MIRROR"] = pypi_index
    if huggingface_endpoint:
        env["ASMR_DUBBER_HF_ENDPOINT"] = huggingface_endpoint
    try:
        if backend_id == "parakeet_nemo":
            completed = runtime_manager._install_parakeet_runtime(
                timeout_seconds=timeout_seconds,
                env=env,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        elif spec.installer == "isolated":
            completed = runtime_manager._install_indextts_runtime(
                backend_id,
                timeout_seconds=timeout_seconds,
                env=env,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        elif (
            runtime_manager.current_platform().is_windows and backend_id in _WINDOWS_SCRIPT_BACKENDS
        ):
            completed = runtime_manager._install_windows_backend(
                backend_id,
                timeout_seconds=timeout_seconds,
                env=env,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        else:
            attempts: list[str] = []
            for index_url in runtime_manager.download_candidates(
                "pypi_indexes", preferred=pypi_index
            ):
                if log_callback is not None:
                    log_callback(f"使用 Python 软件源：{index_url}")
                command = [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--editable",
                    f"{PROJECT_ROOT}[{extra}]",
                    "--default-index",
                    index_url,
                ]
                completed = runtime_manager._run_streaming_process(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    timeout_seconds=timeout_seconds,
                    log_callback=log_callback,
                    cancel_event=cancel_event,
                )
                if completed.returncode == 0:
                    break
                attempts.append(f"{index_url}: {(completed.stderr or completed.stdout)[-800:]}")
            else:
                raise EnvironmentError(
                    f"安装 {spec.label} 失败，所有 PyPI 源均不可用：\n" + "\n".join(attempts)
                )
    except subprocess.TimeoutExpired as exc:
        raise EnvironmentError(f"安装 {spec.label} 超时。") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise EnvironmentError(f"安装 {spec.label} 失败：\n{detail}")
    if defer_model_pack:
        if cancel_event is not None and cancel_event.is_set():
            raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
        imported_packs = runtime_manager.import_backend_model_packs(
            backend_id,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    if spec.installer == "isolated":
        if progress:
            progress(1, desc=f"{spec.label} 独立环境与模型安装完成")
        tail = (completed.stdout or completed.stderr).strip().splitlines()[-12:]
        result = f"{spec.label} 安装完成。\n" + "\n".join(tail) + "\n请重启 ASMR Dubber。"
        if log_callback is not None:
            log_callback(f"{spec.label} 独立环境与模型安装完成。")
        return result
    if progress:
        progress(0.5, desc=f"{spec.label} 运行环境安装完成，正在检查模型")
    if log_callback is not None:
        log_callback(f"{spec.label} 运行环境安装完成，正在检查模型。")
    if cancel_event is not None:
        model_detail = runtime_manager._download_backend_models_subprocess(
            backend_id,
            timeout_seconds=timeout_seconds,
            env=env,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    else:
        model_detail = runtime_manager.download_backend_models(
            backend_id,
            progress=progress,
            log_callback=log_callback,
        )
    tail = (completed.stdout or completed.stderr).strip().splitlines()[-8:]
    detail = "\n".join(tail)
    restart = "\n请重启 ASMR Dubber，使新安装的运行时生效。" if os.name == "nt" else ""
    result = f"{spec.label} 安装完成。\n{detail}\n{model_detail}{restart}".strip()
    if progress:
        progress(1, desc=f"{spec.label} 安装完成")
    if log_callback is not None:
        log_callback(f"{spec.label} 安装完成。")
    return result


def install_backend(
    backend_id: str,
    *,
    progress: Any | None = None,
    timeout_seconds: float = 3600,
    log_callback: InstallLogCallback | None = None,
    force: bool = False,
    cancel_event: threading.Event | None = None,
) -> str:
    """Install one backend while excluding every inference/install process."""
    from . import runtime_manager

    lock_path = runtime_manager.portable_home() / ".runtime-install.lock"
    with runtime_manager.exclusive_file_lock(lock_path, timeout_seconds=30.0):
        return runtime_manager._install_backend_unlocked(
            backend_id,
            progress=progress,
            timeout_seconds=timeout_seconds,
            log_callback=log_callback,
            force=force,
            cancel_event=cancel_event,
        )
