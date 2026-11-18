from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from .constants import (
    ASMR_VAD_MODEL,
    DEFAULT_ALIGNER_MODEL,
)
from .model_registry import ASR_BACKENDS, TTS_BACKENDS, ModelBackend


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

_IMPORTS: dict[str, tuple[str, ...]] = {
    "edge_tts": ("edge_tts",),
    "faster_whisper": ("faster_whisper",),
    "kotoba_whisper": ("transformers",),
    "qwen3_asr": ("transformers", "qwen_asr", "onnxruntime"),
    "anime_whisper": ("transformers", "qwen_asr", "onnxruntime"),
}

_BACKEND_MODEL_REPOS: dict[str, tuple[str, ...]] = {
    "kotoba_whisper": ("kotoba-tech/kotoba-whisper-v2.2",),
    "faster_whisper": ("Systran/faster-whisper-large-v2",),
    # Both new recognizers cut Japanese audio with the ASMR VAD and timestamp
    # it with the Qwen3 aligner, so the install is only complete with them.
    "anime_whisper": ("litagin/anime-whisper", DEFAULT_ALIGNER_MODEL, ASMR_VAD_MODEL),
    "qwen3_asr": (
        "Qwen/Qwen3-ASR-1.7B",
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
        DEFAULT_ALIGNER_MODEL,
        ASMR_VAD_MODEL,
    ),
}

# Backends whose Windows install goes through scripts/windows/install-backend.ps1
# (CUDA PyTorch + the advanced dependency pack) instead of a plain uv install.
_WINDOWS_SCRIPT_BACKENDS = frozenset(
    {"faster_whisper", "kotoba_whisper", "anime_whisper", "qwen3_asr"}
)
_TORCH_BACKENDS = frozenset({"kotoba_whisper", "anime_whisper", "qwen3_asr"})

# Repository files that inference never reads. The community Qwen3 fine-tune
# ships ~8 GB of optimizer state next to the 4 GB checkpoint.
_MODEL_DOWNLOAD_IGNORE: dict[str, tuple[str, ...]] = {
    "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame": (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state*",
        "trainer_state.json",
        "training_args.bin",
        "checkpoint-*",
        "*.pt",
        "*.pth",
        "*.bin",
    ),
    "litagin/anime-whisper": ("figs/*",),
}

# Models that have a ModelScope offline pack; everything else is Hugging Face only.
_MODELSCOPE_PACK_FOR_MODEL = {
    ASMR_VAD_MODEL: "whisper-vad-asmr-onnx",
    DEFAULT_ALIGNER_MODEL: "qwen3-forced-aligner",
    "kotoba-tech/kotoba-whisper-v2.2": "kotoba-whisper-v2.2",
    "Systran/faster-whisper-large-v2": "faster-whisper-large-v2",
    "litagin/anime-whisper": "anime-whisper",
    "Qwen/Qwen3-ASR-1.7B": "qwen3-asr-1.7b",
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
    from pathlib import Path

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia_gpu() -> tuple[str | None, float | None, str | None, str | None]:
    from pathlib import Path

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
    from . import runtime_manager

    info = runtime_manager.current_platform()
    gpu, vram, driver, capability = runtime_manager._nvidia_gpu()
    memory = (
        runtime_manager._windows_memory_gb()
        if info.is_windows
        else runtime_manager._linux_memory_gb()
    )
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
    from . import runtime_manager

    cache_clear = getattr(runtime_manager.detect_hardware, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    return runtime_manager.detect_hardware()


def hardware_markdown(profile: HardwareProfile | None = None) -> str:
    from . import runtime_manager

    profile = profile or runtime_manager.detect_hardware()
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
    from . import runtime_manager

    profile = profile or runtime_manager.detect_hardware()
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


def installable_backend_ids() -> tuple[str, ...]:
    return tuple(
        backend.id
        for backend in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values())
        if backend.installer is not None
    )


def backend_model_pack_ids(backend_id: str) -> set[str]:
    from . import runtime_manager

    if backend_id == "parakeet_nemo":
        pack = (
            "parakeet-ja-windows"
            if runtime_manager.current_platform().is_windows
            else "parakeet-ja-linux"
        )
        return {pack}
    return {
        "indextts2_5": {"indextts2_5-checkpoints"},
        "indextts2": {"indextts2-checkpoints"},
        "kotoba_whisper": {"kotoba-whisper-v2.2"},
        "faster_whisper": {"faster-whisper-large-v2"},
        # anime-whisper and the base Qwen3-ASR 1.7B packs are published on
        # ModelScope alongside the VAD and aligner packs; the community Qwen3
        # fine-tune pack is picked up only when dropped into model-packs/.
        "anime_whisper": {"anime-whisper", "whisper-vad-asmr-onnx", "qwen3-forced-aligner"},
        "qwen3_asr": {
            "qwen3-asr-1.7b",
            "qwen3-asr-1.7b-ja-anime-galgame",
            "whisper-vad-asmr-onnx",
            "qwen3-forced-aligner",
        },
    }.get(backend_id, set())
