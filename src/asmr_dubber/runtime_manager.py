from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ASR_MODEL,
    DEFAULT_TTS_MODEL,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    MODEL_REQUIRED_FILES,
    MODEL_REVISIONS,
    OPTIONAL_ASR_MODEL_REVISIONS,
    PROJECT_ROOT,
)
from .environment import cached_model_path
from .errors import AsmrDubberError, EnvironmentError
from .mirrors import mirror_candidates, snapshot_download_with_fallback
from .model_registry import ASR_BACKENDS, TTS_BACKENDS, ModelBackend
from .platforms import current_platform, portable_home, runtime_executable_candidates


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

    @property
    def recommended_device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"


@dataclass(frozen=True)
class BackendStatus:
    state: str
    label: str
    detail: str = ""


_IMPORTS: dict[str, tuple[str, ...]] = {
    "qwen3_asr": ("qwen_asr",),
    "faster_whisper": ("faster_whisper",),
    "openai_whisper": ("whisper",),
    "whisperx": ("whisperx",),
    "funasr": ("funasr",),
    "kotoba_whisper": ("transformers",),
    "voxcpm2": ("voxcpm",),
    "qwen3_tts": ("qwen_tts",),
    "xtts_v2": ("TTS",),
}

_BACKEND_MODEL_REPOS: dict[str, tuple[str, ...]] = {
    "qwen3_asr": (DEFAULT_ASR_MODEL, DEFAULT_ALIGNER_MODEL),
    "kotoba_whisper": ("kotoba-tech/kotoba-whisper-v2.2",),
    "faster_whisper": (
        "Systran/faster-whisper-large-v2",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
    ),
    "voxcpm2": (DEFAULT_TTS_MODEL,),
}


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


def _nvidia_gpu() -> tuple[str | None, float | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32/nvidia-smi.exe"
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        return None, None, None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first = completed.stdout.strip().splitlines()[0]
        name, memory_mib, driver = (part.strip() for part in first.split(",", 2))
        return name, float(memory_mib) / 1024, driver
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None, None


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Detect hardware once per process.

    The settings page asks for the same profile several times while it is being
    built.  Caching avoids repeatedly launching nvidia-smi.  Explicit refreshes
    use ``refresh_hardware`` below.
    """
    info = current_platform()
    gpu, vram, driver = _nvidia_gpu()
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
    )


def refresh_hardware() -> HardwareProfile:
    detect_hardware.cache_clear()
    return detect_hardware()


def hardware_markdown(profile: HardwareProfile | None = None) -> str:
    profile = profile or detect_hardware()
    memory = f"{profile.memory_gb:.1f} GB" if profile.memory_gb is not None else "未检测"
    if profile.gpu:
        vram = f"{profile.vram_gb:.1f} GB" if profile.vram_gb is not None else "未知显存"
        gpu = f"{profile.gpu} · {vram} · 驱动 {profile.driver or '未知'}"
    else:
        gpu = "未检测到 NVIDIA GPU；可选择 CPU、云端或 HTTP 后端"
    return (
        f"**系统：** {profile.platform_label} {profile.architecture}  \n"
        f"**内存：** {memory}  \n"
        f"**GPU：** {gpu}  \n"
        f"**推荐设备：** `{profile.recommended_device}`"
    )


def recommended_stack_markdown(profile: HardwareProfile | None = None) -> str:
    profile = profile or detect_hardware()
    vram = profile.vram_gb or 0.0
    if profile.cuda_available and vram >= 11.5:
        title = "质量优先（推荐）"
        detail = (
            "日语识别首选 Parakeet CTC 1.1B GAL；疑难音频可启用"
            " Parakeet + Kotoba + Qwen 多模型校对。翻译用 DeepSeek，配音首选 IndexTTS2。"
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
            "Parakeet CTC 1.1B GAL（CPU F16）、Faster-Whisper（cpu + int8）或云端 ASR / "
            "DeepSeek / 外部 HTTP TTS。"
            "质量优先的本地克隆模型通常需要 NVIDIA GPU。"
        )
    return f"### 本机建议：{title}\n{detail}"


def _imports_available(backend_id: str) -> bool:
    modules = _IMPORTS.get(backend_id, ())
    return bool(modules) and all(importlib.util.find_spec(module) is not None for module in modules)


def _builtin_models_complete(backend_id: str) -> bool:
    if backend_id == "qwen3_asr":
        model_ids: Iterable[str] = tuple(MODEL_REVISIONS)[:2]
    elif backend_id == "voxcpm2":
        model_ids = tuple(MODEL_REVISIONS)[2:]
    else:
        return True
    return all(
        model_id in MODEL_REQUIRED_FILES and cached_model_path(model_id) for model_id in model_ids
    )


def backend_status(
    backend: ModelBackend,
    *,
    settings: Any | None = None,
) -> BackendStatus:
    platform_id = current_platform().id
    if platform_id not in backend.platforms:
        return BackendStatus("incompatible", "当前系统不支持")
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
            return BackendStatus("missing", "未安装", "缺少 CrispASR 便携运行时")
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
    if backend.runtime == "builtin" and not _builtin_models_complete(backend.id):
        return BackendStatus("broken", "模型未下载完整")
    return BackendStatus("ready", "可用")


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


def backend_catalog_rows(settings: Any | None = None) -> list[list[str]]:
    profile = detect_hardware()
    rows: list[list[str]] = []
    for kind, registry in (("ASR", ASR_BACKENDS), ("TTS", TTS_BACKENDS)):
        for backend in registry.values():
            status = backend_status(backend, settings=settings)
            rows.append(
                [
                    kind,
                    backend.label,
                    backend.support_label,
                    compatibility_note(backend, profile),
                    status.label,
                    status.detail,
                    f"约 {backend.disk_gb:g} GB" if backend.disk_gb else "外部/未知",
                ]
            )
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
) -> subprocess.CompletedProcess[str]:
    powershell = _powershell_executable()
    script = PROJECT_ROOT / "scripts" / "windows" / "install-backend.ps1"
    if powershell is None:
        raise EnvironmentError("找不到 PowerShell；请安装 PowerShell 7 后重试。")
    if not script.is_file():
        raise EnvironmentError("Windows 后端安装脚本缺失；请重新下载完整项目。")
    return subprocess.run(
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
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        env=env,
    )


def installable_backend_ids() -> tuple[str, ...]:
    return tuple(
        backend.id
        for backend in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values())
        if backend.installer is not None
    )


def _install_indextts_runtime(
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    info = current_platform()
    if info.is_windows:
        executable = _powershell_executable()
        script = PROJECT_ROOT / "scripts" / "windows" / "install-indextts2.ps1"
        if executable is None:
            raise EnvironmentError("找不到 PowerShell；请安装 PowerShell 7 后重试。")
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
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(timeout_seconds, 14_400),
        check=False,
        env=env,
    )


def _install_parakeet_runtime(
    *,
    timeout_seconds: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    info = current_platform()
    if info.is_windows:
        executable = _powershell_executable()
        script = PROJECT_ROOT / "scripts" / "windows" / "install-parakeet.ps1"
        if executable is None:
            raise EnvironmentError("找不到 PowerShell；请安装 PowerShell 7 后重试。")
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
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(timeout_seconds, 14_400),
        check=False,
        env=env,
    )


def download_backend_models(
    backend_id: str,
    *,
    progress: Any | None = None,
) -> str:
    """Download and verify pinned built-in model snapshots."""
    model_ids = _BACKEND_MODEL_REPOS.get(backend_id, ())
    if not model_ids:
        return "该后端在首次使用时由其官方运行库管理模型，无需预下载。"
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
        if progress:
            progress(
                (index - 1) / len(model_ids),
                desc=f"正在下载并校验模型 {index}/{len(model_ids)}：{model_id}",
            )
        revision = MODEL_REVISIONS.get(model_id) or OPTIONAL_ASR_MODEL_REVISIONS.get(model_id)
        if revision is None:
            raise EnvironmentError(f"模型 {model_id} 没有经过验证的固定版本。")
        try:
            snapshot_download_with_fallback(
                repo_id=model_id,
                revision=revision,
                preferred_endpoint=endpoint,
                max_workers=4,
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider/network failures
            mirror = f"（当前端点：{endpoint}）" if endpoint else ""
            raise EnvironmentError(f"下载模型 {model_id} 失败{mirror}：{exc}") from exc
        path = cached_model_path(model_id)
        if path is None:
            raise EnvironmentError(f"模型 {model_id} 下载后未通过完整性检查。")
        resolved.append(f"{model_id} → {path}")
    if progress:
        progress(1, desc="模型下载与完整性检查完成")
    return "模型准备完成：\n" + "\n".join(resolved)


def install_backend(
    backend_id: str,
    *,
    progress: Any | None = None,
    timeout_seconds: float = 3600,
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
    if progress:
        progress(0, desc=f"正在安装 {spec.label}，请勿关闭页面")
    env = os.environ.copy()
    env.setdefault("UV_LINK_MODE", "copy" if os.name == "nt" else "clone")
    try:
        from .user_settings import load_user_settings

        saved_settings = load_user_settings()
        pypi_index = saved_settings.pypi_index_url.strip()
        huggingface_endpoint = saved_settings.huggingface_endpoint.strip()
    except (OSError, ValueError, AsmrDubberError):
        pypi_index = ""
        huggingface_endpoint = ""
    if pypi_index:
        env["ASMR_DUBBER_PYPI_MIRROR"] = pypi_index
    if huggingface_endpoint:
        env["ASMR_DUBBER_HF_ENDPOINT"] = huggingface_endpoint
    try:
        if backend_id == "parakeet_nemo":
            completed = _install_parakeet_runtime(
                timeout_seconds=timeout_seconds,
                env=env,
            )
        elif spec.installer == "isolated":
            completed = _install_indextts_runtime(
                timeout_seconds=timeout_seconds,
                env=env,
            )
        elif current_platform().is_windows and backend_id in {"qwen3_asr", "voxcpm2"}:
            completed = _install_windows_backend(
                backend_id,
                timeout_seconds=timeout_seconds,
                env=env,
            )
        else:
            attempts: list[str] = []
            for index_url in mirror_candidates("pypi_indexes", preferred=pypi_index):
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
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    check=False,
                    env=env,
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
    if spec.installer == "isolated":
        if progress:
            progress(1, desc="IndexTTS2 独立环境与模型安装完成")
        tail = (completed.stdout or completed.stderr).strip().splitlines()[-12:]
        return "IndexTTS2 安装完成。\n" + "\n".join(tail) + "\n请重启 ASMR Dubber。"
    if progress:
        progress(0.5, desc=f"{spec.label} 运行环境安装完成，正在检查模型")
    model_detail = download_backend_models(backend_id, progress=progress)
    tail = (completed.stdout or completed.stderr).strip().splitlines()[-8:]
    detail = "\n".join(tail)
    restart = "\n请重启 ASMR Dubber，使新安装的运行时生效。" if os.name == "nt" else ""
    return f"{spec.label} 安装完成。\n{detail}\n{model_detail}{restart}".strip()
