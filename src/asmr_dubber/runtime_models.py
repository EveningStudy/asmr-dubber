from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from .constants import OPTIONAL_ASR_MODEL_REVISIONS, PROJECT_ROOT
from .errors import AsmrDubberError, EnvironmentError, InstallPausedError
from .model_pack_download import ModelPackDownloadError, ModelPackDownloadPaused
from .model_packs import ModelPackError
from .runtime_specs import (
    _MODEL_DOWNLOAD_IGNORE,
    _MODELSCOPE_PACK_FOR_MODEL,
    InstallLogCallback,
)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = shutil.which("taskkill.exe") or shutil.which("taskkill")
        if not taskkill:
            candidate = (
                Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"
            )
            taskkill = str(candidate) if candidate.is_file() else None
        if taskkill:
            with suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    [taskkill, "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            with suppress(OSError, ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()


def _run_streaming_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an installer while forwarding merged stdout/stderr line by line."""
    from . import runtime_manager

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    output: list[str] = []
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, ""):
                messages.put(line)
        finally:
            process.stdout.close()
            messages.put(None)

    reader = threading.Thread(
        target=read_output,
        name="asmr-dubber-installer-output",
        daemon=True,
    )
    reader.start()
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    reader_finished = False
    try:
        while not reader_finished:
            if cancel_event is not None and cancel_event.is_set():
                runtime_manager._terminate_process_tree(process)
                raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds, output="".join(output))
            try:
                line = messages.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if line is None:
                reader_finished = True
                continue
            output.append(line)
            rendered = line.rstrip("\r\n")
            if rendered and log_callback is not None:
                log_callback(rendered)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds, output="".join(output))
        return_code = process.wait(timeout=remaining)
    except BaseException:
        runtime_manager._terminate_process_tree(process)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        reader.join(timeout=1)
    return subprocess.CompletedProcess(
        command,
        return_code,
        stdout="".join(output),
        stderr="",
    )


def import_backend_model_packs(
    backend_id: str,
    *,
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    from . import runtime_manager

    pack_ids = runtime_manager.backend_model_pack_ids(backend_id)
    if not pack_ids:
        return 0
    for pack_id in sorted(pack_ids):
        try:
            runtime_manager.prepare_remote_model_pack(
                pack_id,
                log=log_callback,
                cancelled=cancel_event.is_set if cancel_event is not None else None,
            )
        except ModelPackDownloadPaused as exc:
            raise InstallPausedError(str(exc)) from exc
        except (ModelPackDownloadError, ModelPackError, OSError, ValueError) as exc:
            if log_callback is not None:
                if runtime_manager.external_downloads_allowed():
                    log_callback(f"ModelScope 模型包不可用；已允许海外源，将尝试备用源：{exc}")
                else:
                    log_callback(f"ModelScope 模型包不可用；断点已保留且不会切换海外源：{exc}")
    try:
        results = runtime_manager.import_discovered_model_packs(
            pack_ids=pack_ids,
            log=log_callback,
            progress=(
                (lambda message, current, total: log_callback(f"[{current}/{total}] {message}"))
                if log_callback is not None
                else None
            ),
        )
    except (ModelPackError, OSError) as exc:
        raise EnvironmentError(f"本地模型包导入失败：{exc}") from exc
    if results and log_callback is not None:
        log_callback(f"已优先处理 {len(results)} 个本地模型包。")
    return len(results)


def download_backend_models(
    backend_id: str,
    *,
    progress: Any | None = None,
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Download and verify pinned built-in model snapshots."""
    from . import runtime_manager

    model_ids = runtime_manager._BACKEND_MODEL_REPOS.get(backend_id, ())
    if not model_ids:
        return "该后端在首次使用时由其官方运行库管理模型，无需预下载。"
    if any(runtime_manager.cached_model_path(model_id) is None for model_id in model_ids):
        runtime_manager.import_backend_model_packs(
            backend_id,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    try:
        from .user_settings import load_user_settings

        saved_endpoint = load_user_settings().huggingface_endpoint.strip()
    except (OSError, ValueError, AsmrDubberError):
        saved_endpoint = ""
    endpoint = (
        os.getenv("ASMR_DUBBER_HF_ENDPOINT", "").strip()
        or os.getenv("HF_ENDPOINT", "").strip()
        or saved_endpoint
        or None
    )
    resolved: list[str] = []
    for index, model_id in enumerate(model_ids, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
        revision = OPTIONAL_ASR_MODEL_REVISIONS.get(model_id)
        if revision is None:
            raise EnvironmentError(f"模型 {model_id} 没有经过验证的固定版本。")
        path = runtime_manager.cached_model_path(model_id)
        if path is None:
            message = f"正在下载并校验模型 {index}/{len(model_ids)}：{model_id}"
            if log_callback is not None:
                log_callback(message)
            if progress:
                progress(
                    (index - 1) / len(model_ids),
                    desc=message,
                )
            # snapshot_download keeps real files (not symlinks) in the snapshot
            # directory when the filesystem cannot symlink, as on Windows.
            download_options: dict[str, Any] = {}
            if model_id in _MODEL_DOWNLOAD_IGNORE:
                download_options["ignore_patterns"] = list(_MODEL_DOWNLOAD_IGNORE[model_id])
            try:
                runtime_manager.snapshot_download_with_fallback(
                    repo_id=model_id,
                    revision=revision,
                    preferred_endpoint=endpoint,
                    max_workers=4,
                    **download_options,
                )
            except Exception as exc:
                mirror = f"（当前端点：{endpoint}）" if endpoint else ""
                hint = ""
                if (
                    model_id not in _MODELSCOPE_PACK_FOR_MODEL
                    and not runtime_manager.external_downloads_allowed()
                ):
                    hint = "\n该模型暂无国内镜像，需要在“设置 → 下载来源”选择“海外源”后重试。"
                raise EnvironmentError(f"下载模型 {model_id} 失败{mirror}：{exc}{hint}") from exc
            path = runtime_manager.cached_model_path(model_id)
            if path is None:
                raise EnvironmentError(f"模型 {model_id} 下载后未通过完整性检查。")
        elif log_callback is not None:
            log_callback(f"复用已导入或已缓存的模型 {index}/{len(model_ids)}：{model_id}")
        resolved.append(f"{model_id} → {path}")
        if log_callback is not None:
            log_callback(f"模型 {model_id} 已通过完整性检查。")
        if cancel_event is not None and cancel_event.is_set():
            raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
    if progress:
        progress(1, desc="模型下载与完整性检查完成")
    if log_callback is not None:
        log_callback("模型下载与完整性检查完成。")
    return "模型准备完成：\n" + "\n".join(resolved)


def _download_backend_models_subprocess(
    backend_id: str,
    *,
    timeout_seconds: float,
    env: dict[str, str],
    log_callback: InstallLogCallback | None,
    cancel_event: threading.Event,
) -> str:
    from . import runtime_manager

    completed = runtime_manager._run_streaming_process(
        [
            sys.executable,
            "-m",
            "asmr_dubber.cli",
            "download-backend-models",
            backend_id,
        ],
        cwd=PROJECT_ROOT,
        env={**env, "PYTHONUNBUFFERED": "1"},
        timeout_seconds=max(timeout_seconds, 14_400),
        log_callback=log_callback,
        cancel_event=cancel_event,
    )
    if completed.returncode != 0:
        raise EnvironmentError(
            f"模型下载失败（退出码 {completed.returncode}）：\n{completed.stdout[-4000:]}"
        )
    return completed.stdout.strip()
