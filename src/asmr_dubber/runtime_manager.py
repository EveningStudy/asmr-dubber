from __future__ import annotations

import importlib.util
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .constants import (
    ASMR_VAD_MODEL,
    DEFAULT_ALIGNER_MODEL,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    OPTIONAL_ASR_MODEL_REVISIONS,
    PROJECT_ROOT,
)
from .environment import cached_model_path, cuda_summary
from .errors import AsmrDubberError, EnvironmentError, InstallPausedError
from .mirrors import (
    download_candidates,
    external_downloads_allowed,
    snapshot_download_with_fallback,
)
from .model_pack_download import (
    ModelPackDownloadError,
    ModelPackDownloadPaused,
    prepare_remote_model_pack,
)
from .model_packs import ModelPackError, import_discovered_model_packs
from .model_registry import ASR_BACKENDS, TTS_BACKENDS, ModelBackend
from .platforms import current_platform, portable_home, runtime_executable_candidates
from .storage import exclusive_file_lock


@dataclass(frozen=True)
class HardwareProfile:
    platform_id: str
    platform_label: str
    architecture: str
    cpu: str
    memory_gb: float | None
    gpu: str | None
    vram_gb: float | None
    driver: str | None
    cuda_available: bool
    compute_capability: str | None = None

    @property
    def recommended_device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"

    @property
    def full_cuda_stack_supported(self) -> bool:
        if not self.cuda_available:
            return False
        if self.compute_capability is None:
            return True
        try:
            return float(self.compute_capability) >= 7.5
        except ValueError:
            return True


@dataclass(frozen=True)
class BackendStatus:
    state: str
    label: str
    detail: str = ""


InstallLogCallback = Callable[[str], None]


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
                _terminate_process_tree(process)
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
        _terminate_process_tree(process)
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


_IMPORTS: dict[str, tuple[str, ...]] = {
    "edge_tts": ("edge_tts",),
    "faster_whisper": ("faster_whisper",),
    "kotoba_whisper": ("transformers",),
}

_BACKEND_MODEL_REPOS: dict[str, tuple[str, ...]] = {
    "kotoba_whisper": ("kotoba-tech/kotoba-whisper-v2.2",),
    "faster_whisper": ("Systran/faster-whisper-large-v2",),
}

_PARAKEET_MODEL_FILES = {
    "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo": (
        "parakeet-ctc-1.1b-ja-f16.gguf"
    ),
    "nvidia/parakeet-tdt_ctc-0.6b-ja": "parakeet-tdt-0.6b-ja.gguf",
}

_FASTER_WHISPER_REPOS = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
}

ASR_REVIEW_MODEL_OPTIONS = (
    (
        "Parakeet CTC 1.1B 日语 GAL（推荐）",
        "parakeet_nemo",
        "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo",
    ),
    (
        "Parakeet TDT/CTC 0.6B 日语（官方）",
        "parakeet_nemo",
        "nvidia/parakeet-tdt_ctc-0.6b-ja",
    ),
    (
        "Kotoba-Whisper v2.2（最新推荐）",
        "kotoba_whisper",
        "kotoba-tech/kotoba-whisper-v2.2",
    ),
    (
        "Kotoba-Whisper v2.1（旧版标点模型）",
        "kotoba_whisper",
        "kotoba-tech/kotoba-whisper-v2.1",
    ),
    (
        "Kotoba-Whisper v2.0 Faster",
        "faster_whisper",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
    ),
    ("Faster-Whisper large-v2", "faster_whisper", "large-v2"),
    ("Faster-Whisper large-v3", "faster_whisper", "large-v3"),
)


def _windows_memory_gb() -> float | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total_physical / 1024**3
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _linux_memory_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia_gpu() -> tuple[str | None, float | None, str | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32/nvidia-smi.exe"
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        return None, None, None, None

    def query(fields: str) -> list[str]:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or "nvidia-smi 查询失败")
        return [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]

    try:
        name, memory_mib, driver, capability = query("name,memory.total,driver_version,compute_cap")
        return name, float(memory_mib) / 1024, driver, capability
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        # Older nvidia-smi versions may not expose compute_cap. Keep hardware
        # detection useful and simply omit the architecture warning.
        try:
            name, memory_mib, driver = query("name,memory.total,driver_version")
            return name, float(memory_mib) / 1024, driver, None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None, None, None


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Detect hardware once per process.

    The settings page asks for the same profile several times while it is being
    built.  Caching avoids repeatedly launching nvidia-smi.  Explicit refreshes
    use ``refresh_hardware`` below.
    """
    info = current_platform()
    gpu, vram, driver, capability = _nvidia_gpu()
    memory = _windows_memory_gb() if info.is_windows else _linux_memory_gb()
    cpu = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if not cpu:
        import platform

        cpu = platform.processor() or "未知 CPU"
    return HardwareProfile(
        platform_id=info.id,
        platform_label=info.label,
        architecture=info.architecture,
        cpu=cpu,
        memory_gb=memory,
        gpu=gpu,
        vram_gb=vram,
        driver=driver,
        cuda_available=gpu is not None,
        compute_capability=capability,
    )


def refresh_hardware() -> HardwareProfile:
    detect_hardware.cache_clear()
    return detect_hardware()


def hardware_markdown(profile: HardwareProfile | None = None) -> str:
    profile = profile or detect_hardware()
    memory = f"{profile.memory_gb:.1f} GB" if profile.memory_gb is not None else "未检测"
    if profile.gpu:
        vram = f"{profile.vram_gb:.1f} GB" if profile.vram_gb is not None else "未知显存"
        capability = (
            f" · 计算能力 {profile.compute_capability}" if profile.compute_capability else ""
        )
        gpu = f"{profile.gpu} · {vram} · 驱动 {profile.driver or '未知'}{capability}"
    else:
        gpu = "未检测到 NVIDIA GPU；可选择 CPU、云端或 HTTP 后端"
    warning = (
        "  \n**兼容提醒：** 当前 Windows 完整 GPU 环境需要 Turing（计算能力 7.5）或更新架构。"
        if profile.cuda_available and not profile.full_cuda_stack_supported
        else ""
    )
    return (
        f"**系统：** {profile.platform_label} {profile.architecture}  \n"
        f"**内存：** {memory}  \n"
        f"**GPU：** {gpu}  \n"
        f"**推荐设备：** `{profile.recommended_device}`{warning}"
    )


def recommended_stack_markdown(profile: HardwareProfile | None = None) -> str:
    profile = profile or detect_hardware()
    vram = profile.vram_gb or 0.0
    if profile.cuda_available and not profile.full_cuda_stack_supported:
        title = "旧架构 NVIDIA（部分组件改用 CPU）"
        detail = (
            "当前完整 GPU 依赖要求 Turing 或更新架构。Parakeet/Faster-Whisper 可按实际状态选择，"
            "Kotoba 和 Qwen 对齐建议改用 CPU，TTS 可使用外部 API。"
        )
    elif profile.cuda_available and vram >= 11.5:
        title = "质量优先（推荐）"
        detail = (
            "日语识别首选 Parakeet CTC 1.1B GAL；疑难音频可启用"
            " Parakeet + Kotoba + Faster-Whisper 多模型校对。翻译用 DeepSeek，"
            "配音首选 IndexTTS2。"
        )
    elif profile.cuda_available and vram >= 8:
        title = "显存均衡"
        detail = (
            "Parakeet CTC 1.1B GAL、Kotoba-Whisper v2.2 或 Faster-Whisper large-v2；"
            "生成 TTS 前关闭其他显存占用程序。IndexTTS2 建议启用 FP16。"
        )
    elif profile.cuda_available and vram >= 6:
        title = "低显存 NVIDIA"
        detail = (
            "Parakeet CTC 1.1B GAL、Kotoba-Whisper v2.0-faster 或 Faster-Whisper large-v2"
            "（float16 或 int8_float16）/ DeepSeek / IndexTTS2 FP16。"
        )
    else:
        title = "CPU / 无 NVIDIA"
        detail = (
            "Parakeet CTC 1.1B GAL（CPU F16）或 Faster-Whisper（cpu + int8）/ "
            "DeepSeek / 外部 TTS（语音合成）API。"
            "质量优先的本地克隆模型通常需要 NVIDIA GPU。"
        )
    return f"### 本机建议：{title}\n{detail}"


def _imports_available(backend_id: str) -> bool:
    modules = _IMPORTS.get(backend_id, ())
    return bool(modules) and all(importlib.util.find_spec(module) is not None for module in modules)


def backend_status(
    backend: ModelBackend,
    *,
    settings: Any | None = None,
) -> BackendStatus:
    platform_id = current_platform().id
    if platform_id not in backend.platforms:
        return BackendStatus("incompatible", "当前系统不支持")
    if backend.id == "edge_tts":
        if not _imports_available(backend.id):
            return BackendStatus("missing", "运行依赖缺失", "请重新运行 Setup 修复基础依赖")
        return BackendStatus("ready", "可用", "无需 API Key；需要联网")
    if backend.runtime == "http":
        return BackendStatus("external", "外部服务", "保存服务地址后请先用短音频试运行")
    if backend.id == "parakeet_nemo":
        executable_name = "crispasr.exe" if current_platform().is_windows else "crispasr"
        executable = portable_home() / "runtimes" / "crispasr" / "bin" / executable_name
        models = portable_home() / "models" / "parakeet"
        required = (
            models / "parakeet-tdt-0.6b-ja.gguf",
            models / "parakeet-ctc-1.1b-ja-f16.gguf",
        )
        if not executable.is_file():
            return BackendStatus("missing", "未安装", "缺少 CrispASR 运行时")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return BackendStatus("broken", "模型不完整", "缺少 " + "、".join(missing))
        return BackendStatus("ready", "可用", str(executable))
    if backend.id == "indextts2":
        configured = str(getattr(settings, "tts_model_path", "") or "").strip()
        if not configured:
            return BackendStatus("missing", "未安装")
        model_dir = Path(configured).expanduser().resolve()
        executable = next(
            (
                path
                for path in runtime_executable_candidates(model_dir.parent, "indextts2")
                if path.is_file()
            ),
            None,
        )
        if executable is None:
            return BackendStatus("missing", "未安装", "缺少独立运行环境")
        missing = sorted(
            [name for name in INDEXTTS_REQUIRED_FILES if not (model_dir / name).is_file()]
            + [name + "/" for name in INDEXTTS_REQUIRED_DIRS if not (model_dir / name).is_dir()]
        )
        if missing:
            detail = "、".join(missing[:4])
            if len(missing) > 4:
                detail += f" 等 {len(missing)} 项"
            return BackendStatus("broken", "模型不完整", f"缺少 {detail}")
        return BackendStatus("ready", "可用", str(executable))
    if backend.runtime == "command":
        executable = str(getattr(settings, "tts_executable", "") or "").strip()
        resolved = shutil.which(executable) if executable else None
        if not resolved and executable and Path(executable).expanduser().is_file():
            resolved = str(Path(executable).expanduser().resolve())
        return (
            BackendStatus("ready", "可用", resolved)
            if resolved
            else BackendStatus("missing", "未安装", "未找到命令行程序")
        )
    if not _imports_available(backend.id):
        return BackendStatus("missing", "未安装", "缺少 Python 运行环境")
    if (
        backend.id == "kotoba_whisper"
        and current_platform().is_windows
        and detect_hardware().cuda_available
        and not cuda_summary()["available"]
    ):
        return BackendStatus(
            "partial",
            "CPU 可用，CUDA 待修复",
            "点击安装/修复以安装 CUDA PyTorch",
        )
    optional_models = _BACKEND_MODEL_REPOS.get(backend.id, ())
    if optional_models and backend.runtime != "builtin":
        missing_models = [
            model_id for model_id in optional_models if cached_model_path(model_id) is None
        ]
        if missing_models:
            return BackendStatus(
                "partial",
                "运行库可用，推荐模型待下载",
                "点击安装/修复：" + "、".join(missing_models),
            )
    return BackendStatus("ready", "可用")


def _local_huggingface_snapshot(model_id: str) -> Path | None:
    """Return a complete local snapshot without making a network request."""
    pinned = cached_model_path(model_id)
    if pinned is not None:
        return pinned
    revision = OPTIONAL_ASR_MODEL_REVISIONS.get(model_id)
    if revision is None:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=True,
            )
        ).resolve()
    except (ImportError, OSError, ValueError):
        return None
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        return None
    weight_patterns = (
        "*.safetensors",
        "model.bin",
        "pytorch_model*.bin",
        "*.pth",
    )
    if not any(next(snapshot.glob(pattern), None) for pattern in weight_patterns):
        return None
    return snapshot


def backend_model_status(
    backend: ModelBackend,
    model_id: str,
    *,
    settings: Any | None = None,
) -> BackendStatus:
    """Report whether one concrete model is locally usable without loading it."""
    if backend.id == "parakeet_nemo":
        executable_name = "crispasr.exe" if current_platform().is_windows else "crispasr"
        executable = portable_home() / "runtimes" / "crispasr" / "bin" / executable_name
        filename = _PARAKEET_MODEL_FILES.get(model_id)
        if not executable.is_file():
            return BackendStatus("missing", "运行库未安装")
        if filename is None:
            return BackendStatus("unknown", "自定义模型，未验证")
        path = portable_home() / "models" / "parakeet" / filename
        return (
            BackendStatus("ready", "可用", str(path))
            if path.is_file()
            else BackendStatus("missing", "模型未下载", str(path))
        )

    overall = backend_status(backend, settings=settings)
    if overall.state in {"incompatible", "missing", "broken"}:
        return overall
    if overall.state == "external":
        return BackendStatus("external", "需连接外部服务确认")
    if backend.id == "indextts2":
        return overall

    repository: str | None = None
    if backend.id == "faster_whisper":
        repository = _FASTER_WHISPER_REPOS.get(model_id, model_id if "/" in model_id else None)
    elif backend.id == "kotoba_whisper":
        repository = model_id if "/" in model_id else None
    if repository is not None:
        path = _local_huggingface_snapshot(repository)
        return (
            BackendStatus("ready", "可用", str(path))
            if path is not None
            else BackendStatus("missing", "模型未下载")
        )

    if overall.state == "ready":
        return BackendStatus("managed", "运行库可用，模型由后端管理")
    return overall


def available_backend_models_markdown(
    kind: Literal["asr", "tts"],
    settings: Any | None = None,
) -> str:
    registry = ASR_BACKENDS if kind == "asr" else TTS_BACKENDS
    title = "ASR（语音识别）" if kind == "asr" else "TTS（语音合成）"
    lines = [f"**本机 {title} 可用状态（只读检测）**"]
    for backend in registry.values():
        statuses = [
            (model_id, backend_model_status(backend, model_id, settings=settings))
            for model_id in backend.models
        ]
        ready = [model_id for model_id, status in statuses if status.state == "ready"]
        managed = [model_id for model_id, status in statuses if status.state == "managed"]
        external = [model_id for model_id, status in statuses if status.state == "external"]
        if ready:
            rendered = "、".join(f"`{model_id}`" for model_id in ready)
            lines.append(f"- **[可用] {backend.label}**：{rendered}")
        elif external:
            rendered = "、".join(f"`{model_id}`" for model_id in external)
            lines.append(f"- **[外部] {backend.label}**：需连接服务确认（{rendered}）")
        elif managed:
            rendered = "、".join(f"`{model_id}`" for model_id in managed)
            lines.append(
                f"- **[待确认] {backend.label}**：运行库可用，模型由后端管理（{rendered}）"
            )
        else:
            overall = backend_status(backend, settings=settings)
            lines.append(f"- **[不可用] {backend.label}**：{overall.label}")
    return "\n".join(lines)


def available_asr_review_choices(settings: Any | None = None) -> list[tuple[str, str]]:
    """Return only locally verified ASR models suitable for cross-review."""
    choices: list[tuple[str, str]] = []
    for label, backend_id, model_id in ASR_REVIEW_MODEL_OPTIONS:
        status = backend_model_status(
            ASR_BACKENDS[backend_id],
            model_id,
            settings=settings,
        )
        if status.state == "ready":
            choices.append((label, f"{backend_id}|{model_id}"))
    return choices


def asmr_vad_status() -> BackendStatus:
    """Report whether the standalone ASMR VAD can run entirely offline."""

    if cached_model_path(ASMR_VAD_MODEL) is None:
        return BackendStatus("missing", "模型未下载")
    missing = [
        module
        for module in ("onnxruntime", "transformers")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return BackendStatus("broken", "运行依赖未安装", "、".join(missing))
    return BackendStatus("ready", "可用", ASMR_VAD_MODEL)


def forced_aligner_status() -> BackendStatus:
    """Report whether the pinned standalone Qwen timestamp aligner is usable."""

    if cached_model_path(DEFAULT_ALIGNER_MODEL) is None:
        return BackendStatus("missing", "模型未下载")
    if importlib.util.find_spec("qwen_asr") is None:
        return BackendStatus("broken", "运行依赖未安装", "qwen-asr")
    return BackendStatus("ready", "可用", DEFAULT_ALIGNER_MODEL)


def available_timestamp_review_choices(settings: Any | None = None) -> list[tuple[str, str]]:
    """Return locally usable time-boundary sources, including the standalone aligner."""

    choices = available_asr_review_choices(settings)
    if forced_aligner_status().state == "ready":
        choices.insert(
            0,
            (
                "Qwen3 ForcedAligner 0.6B（阿里 · 独立时间戳对齐）",
                f"qwen_forced_aligner|{DEFAULT_ALIGNER_MODEL}",
            ),
        )
    return choices


def compatibility_note(backend: ModelBackend, profile: HardwareProfile) -> str:
    if profile.platform_id not in backend.platforms:
        return "不兼容"
    if backend.devices == ("cuda",) and not profile.cuda_available:
        return "需要 NVIDIA GPU"
    if (
        profile.vram_gb is not None
        and backend.minimum_vram_gb is not None
        and profile.vram_gb < backend.minimum_vram_gb
    ):
        return f"显存不足（最低 {backend.minimum_vram_gb:g} GB）"
    if "cuda" in backend.devices and profile.cuda_available:
        if backend.recommended_vram_gb:
            return f"适合 CUDA（建议 {backend.recommended_vram_gb:g} GB）"
        return "适合 CUDA"
    if "cpu" in backend.devices:
        return "可使用 CPU"
    return "需检查设备"


def backend_catalog_rows(
    settings: Any | None = None,
    kind: Literal["asr", "tts"] | None = None,
) -> list[list[str]]:
    profile = detect_hardware()
    rows: list[list[str]] = []
    registries = (
        (("ASR（语音识别）", ASR_BACKENDS),)
        if kind == "asr"
        else (("TTS（语音合成）", TTS_BACKENDS),)
        if kind == "tts"
        else (("ASR（语音识别）", ASR_BACKENDS), ("TTS（语音合成）", TTS_BACKENDS))
    )
    for kind_label, registry in registries:
        for backend in registry.values():
            status = backend_status(backend, settings=settings)
            row = [
                backend.label,
                backend.support_label,
                compatibility_note(backend, profile),
                status.label,
                status.detail,
                f"约 {backend.disk_gb:g} GB" if backend.disk_gb else "外部/未知",
            ]
            rows.append([kind_label, *row] if kind is None else row)
    return rows


def support_report_json(settings: Any | None = None) -> str:
    """Machine-readable report for issue templates and remote troubleshooting."""
    profile = detect_hardware()
    payload = {
        "platform": profile.__dict__,
        "backends": {
            backend.id: backend_status(backend, settings=settings).__dict__
            for backend in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values())
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _uv_executable() -> Path | None:
    home = portable_home()
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
    powershell = _powershell_executable()
    script = PROJECT_ROOT / "scripts" / "windows" / "install-backend.ps1"
    if powershell is None:
        raise EnvironmentError("找不到 PowerShell；Windows 自带 5.1 或 PowerShell 7 均可。")
    if not script.is_file():
        raise EnvironmentError("Windows 后端安装脚本缺失；请重新下载完整项目。")
    return _run_streaming_process(
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


def installable_backend_ids() -> tuple[str, ...]:
    return tuple(
        backend.id
        for backend in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values())
        if backend.installer is not None
    )


def backend_model_pack_ids(backend_id: str) -> set[str]:
    if backend_id == "parakeet_nemo":
        pack = "parakeet-ja-windows" if current_platform().is_windows else "parakeet-ja-linux"
        return {pack}
    return {
        "indextts2": {"indextts2-checkpoints"},
        "kotoba_whisper": {"kotoba-whisper-v2.2"},
        "faster_whisper": {"faster-whisper-large-v2"},
    }.get(backend_id, set())


def import_backend_model_packs(
    backend_id: str,
    *,
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    pack_ids = backend_model_pack_ids(backend_id)
    if not pack_ids:
        return 0
    for pack_id in sorted(pack_ids):
        try:
            prepare_remote_model_pack(
                pack_id,
                log=log_callback,
                cancelled=cancel_event.is_set if cancel_event is not None else None,
            )
        except ModelPackDownloadPaused as exc:
            raise InstallPausedError(str(exc)) from exc
        except (ModelPackDownloadError, ModelPackError, OSError, ValueError) as exc:
            if log_callback is not None:
                if external_downloads_allowed():
                    log_callback(f"ModelScope 模型包不可用；已允许海外源，将尝试备用源：{exc}")
                else:
                    log_callback(f"ModelScope 模型包不可用；断点已保留且不会切换海外源：{exc}")
    try:
        results = import_discovered_model_packs(
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


def _install_indextts_runtime(
    *,
    timeout_seconds: float,
    env: dict[str, str],
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    info = current_platform()
    if info.is_windows:
        executable = _powershell_executable()
        script = PROJECT_ROOT / "scripts" / "windows" / "install-indextts2.ps1"
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
        script = PROJECT_ROOT / "scripts" / "linux" / "install-indextts2.sh"
        if not executable.is_file():
            raise EnvironmentError("找不到 bash，无法安装 IndexTTS2 独立运行时。")
        command = [str(executable), str(script)]
    if not script.is_file():
        raise EnvironmentError("IndexTTS2 安装脚本缺失；请重新下载完整项目。")
    return _run_streaming_process(
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
    info = current_platform()
    if info.is_windows:
        executable = _powershell_executable()
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
    return _run_streaming_process(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        timeout_seconds=max(timeout_seconds, 14_400),
        log_callback=log_callback,
        cancel_event=cancel_event,
    )


def download_backend_models(
    backend_id: str,
    *,
    progress: Any | None = None,
    log_callback: InstallLogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Download and verify pinned built-in model snapshots."""
    model_ids = _BACKEND_MODEL_REPOS.get(backend_id, ())
    if not model_ids:
        return "该后端在首次使用时由其官方运行库管理模型，无需预下载。"
    if any(cached_model_path(model_id) is None for model_id in model_ids):
        import_backend_model_packs(
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
        path = cached_model_path(model_id)
        if path is None:
            message = f"正在下载并校验模型 {index}/{len(model_ids)}：{model_id}"
            if log_callback is not None:
                log_callback(message)
            if progress:
                progress(
                    (index - 1) / len(model_ids),
                    desc=message,
                )
            try:
                snapshot_download_with_fallback(
                    repo_id=model_id,
                    revision=revision,
                    preferred_endpoint=endpoint,
                    max_workers=4,
                )
            except Exception as exc:
                mirror = f"（当前端点：{endpoint}）" if endpoint else ""
                raise EnvironmentError(f"下载模型 {model_id} 失败{mirror}：{exc}") from exc
            path = cached_model_path(model_id)
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
    completed = _run_streaming_process(
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
    registry = ASR_BACKENDS if backend_id in ASR_BACKENDS else TTS_BACKENDS
    spec = registry.get(backend_id)
    if spec is None:
        raise EnvironmentError(f"未知后端：{backend_id}")
    if spec.installer is None:
        raise EnvironmentError(
            "该后端暂不支持应用内自动安装；请按照后端说明启动外部服务或独立运行时。"
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
        status = backend_status(spec, settings=saved_settings)
        if status.state == "ready":
            result = f"{spec.label} 已经可用，无需重复安装。"
            if log_callback is not None:
                log_callback(result)
            if progress:
                progress(1, desc=result)
            return result
    if cancel_event is not None and cancel_event.is_set():
        raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
    defer_model_pack = current_platform().is_windows and backend_id in {
        "faster_whisper",
        "kotoba_whisper",
    }
    imported_packs = 0
    if not defer_model_pack:
        imported_packs = import_backend_model_packs(
            backend_id,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    if cancel_event is not None and cancel_event.is_set():
        raise InstallPausedError("下载已暂停；再次点击安装/修复可继续。")
    if not force and imported_packs:
        status = backend_status(spec, settings=saved_settings)
        if status.state == "ready":
            result = f"{spec.label} 已从本地模型包导入并可直接使用。"
            if log_callback is not None:
                log_callback(result)
            if progress:
                progress(1, desc=result)
            return result
    uv = _uv_executable()
    if uv is None:
        raise EnvironmentError("找不到项目安装器 uv；请重新运行对应平台的 setup 脚本。")
    if spec.devices == ("cuda",):
        profile = detect_hardware()
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
    env.setdefault("UV_LINK_MODE", "copy" if os.name == "nt" else "clone")
    if imported_packs:
        env["ASMR_DUBBER_MODEL_PACKS_PREPARED"] = "1"
    if pypi_index:
        env["ASMR_DUBBER_PYPI_MIRROR"] = pypi_index
    if huggingface_endpoint:
        env["ASMR_DUBBER_HF_ENDPOINT"] = huggingface_endpoint
    try:
        if backend_id == "parakeet_nemo":
            completed = _install_parakeet_runtime(
                timeout_seconds=timeout_seconds,
                env=env,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        elif spec.installer == "isolated":
            completed = _install_indextts_runtime(
                timeout_seconds=timeout_seconds,
                env=env,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        elif current_platform().is_windows and backend_id in {
            "faster_whisper",
            "kotoba_whisper",
        }:
            completed = _install_windows_backend(
                backend_id,
                timeout_seconds=timeout_seconds,
                env=env,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
        else:
            attempts: list[str] = []
            for index_url in download_candidates("pypi_indexes", preferred=pypi_index):
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
                completed = _run_streaming_process(
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
        imported_packs = import_backend_model_packs(
            backend_id,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    if spec.installer == "isolated":
        if progress:
            progress(1, desc="IndexTTS2 独立环境与模型安装完成")
        tail = (completed.stdout or completed.stderr).strip().splitlines()[-12:]
        result = "IndexTTS2 安装完成。\n" + "\n".join(tail) + "\n请重启 ASMR Dubber。"
        if log_callback is not None:
            log_callback("IndexTTS2 独立环境与模型安装完成。")
        return result
    if progress:
        progress(0.5, desc=f"{spec.label} 运行环境安装完成，正在检查模型")
    if log_callback is not None:
        log_callback(f"{spec.label} 运行环境安装完成，正在检查模型。")
    if cancel_event is not None:
        model_detail = _download_backend_models_subprocess(
            backend_id,
            timeout_seconds=timeout_seconds,
            env=env,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )
    else:
        model_detail = download_backend_models(
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

    lock_path = portable_home() / ".runtime-install.lock"
    with exclusive_file_lock(lock_path, timeout_seconds=30.0):
        return _install_backend_unlocked(
            backend_id,
            progress=progress,
            timeout_seconds=timeout_seconds,
            log_callback=log_callback,
            force=force,
            cancel_event=cancel_event,
        )
