from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Literal

from .constants import (
    ASMR_VAD_MODEL,
    DEFAULT_ALIGNER_MODEL,
    FALLBACK_ASR_BACKEND,
    FALLBACK_ASR_MODEL,
    INDEXTTS25_REQUIRED_DIRS,
    INDEXTTS25_REQUIRED_FILES,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    OPTIONAL_ASR_MODEL_REVISIONS,
    RECOMMENDED_ASR_BACKEND,
    RECOMMENDED_ASR_MODEL,
    RECOMMENDED_TTS_BACKEND,
)
from .constants import (
    PROJECT_ROOT as PROJECT_ROOT,
)
from .environment import cached_model_path as cached_model_path
from .environment import cuda_summary as cuda_summary
from .errors import (
    AsmrDubberError as AsmrDubberError,
)
from .errors import (
    EnvironmentError as EnvironmentError,
)
from .errors import (
    InstallPausedError as InstallPausedError,
)
from .mirrors import (
    download_candidates as download_candidates,
)
from .mirrors import (
    external_downloads_allowed as external_downloads_allowed,
)
from .mirrors import (
    snapshot_download_with_fallback as snapshot_download_with_fallback,
)
from .model_pack_download import (
    ModelPackDownloadError as ModelPackDownloadError,
)
from .model_pack_download import (
    ModelPackDownloadPaused as ModelPackDownloadPaused,
)
from .model_pack_download import (
    prepare_remote_model_pack as prepare_remote_model_pack,
)
from .model_packs import (
    ModelPackError as ModelPackError,
)
from .model_packs import (
    import_discovered_model_packs as import_discovered_model_packs,
)
from .model_registry import ASR_BACKENDS, TTS_BACKENDS, ModelBackend
from .platforms import (
    current_platform as current_platform,
)
from .platforms import (
    portable_home as portable_home,
)
from .platforms import (
    runtime_executable_candidates as runtime_executable_candidates,
)
from .runtime_installers import (
    _install_backend_unlocked as _install_backend_unlocked,
)
from .runtime_installers import (
    _install_indextts_runtime as _install_indextts_runtime,
)
from .runtime_installers import (
    _install_parakeet_runtime as _install_parakeet_runtime,
)
from .runtime_installers import (
    _install_windows_backend as _install_windows_backend,
)
from .runtime_installers import (
    _powershell_executable as _powershell_executable,
)
from .runtime_installers import (
    _uv_executable as _uv_executable,
)
from .runtime_installers import (
    install_backend as install_backend,
)
from .runtime_models import (
    _download_backend_models_subprocess as _download_backend_models_subprocess,
)
from .runtime_models import (
    _run_streaming_process as _run_streaming_process,
)
from .runtime_models import (
    _terminate_process_tree as _terminate_process_tree,
)
from .runtime_models import (
    download_backend_models as download_backend_models,
)
from .runtime_models import (
    import_backend_model_packs as import_backend_model_packs,
)
from .runtime_specs import (
    _BACKEND_MODEL_REPOS as _BACKEND_MODEL_REPOS,
)
from .runtime_specs import (
    _FASTER_WHISPER_REPOS as _FASTER_WHISPER_REPOS,
)
from .runtime_specs import (
    _IMPORTS as _IMPORTS,
)
from .runtime_specs import (
    _MODEL_DOWNLOAD_IGNORE as _MODEL_DOWNLOAD_IGNORE,
)
from .runtime_specs import (
    _MODELSCOPE_PACK_FOR_MODEL as _MODELSCOPE_PACK_FOR_MODEL,
)
from .runtime_specs import (
    _PARAKEET_MODEL_FILES as _PARAKEET_MODEL_FILES,
)
from .runtime_specs import (
    _TORCH_BACKENDS as _TORCH_BACKENDS,
)
from .runtime_specs import (
    _WINDOWS_SCRIPT_BACKENDS as _WINDOWS_SCRIPT_BACKENDS,
)
from .runtime_specs import (
    ASR_REVIEW_MODEL_OPTIONS as ASR_REVIEW_MODEL_OPTIONS,
)
from .runtime_specs import (
    BackendStatus as BackendStatus,
)
from .runtime_specs import (
    HardwareProfile as HardwareProfile,
)
from .runtime_specs import (
    InstallLogCallback as InstallLogCallback,
)
from .runtime_specs import (
    _linux_memory_gb as _linux_memory_gb,
)
from .runtime_specs import (
    _nvidia_gpu as _nvidia_gpu,
)
from .runtime_specs import (
    _windows_memory_gb as _windows_memory_gb,
)
from .runtime_specs import (
    backend_model_pack_ids as backend_model_pack_ids,
)
from .runtime_specs import (
    compatibility_note as compatibility_note,
)
from .runtime_specs import (
    detect_hardware as detect_hardware,
)
from .runtime_specs import (
    hardware_markdown as hardware_markdown,
)
from .runtime_specs import (
    installable_backend_ids as installable_backend_ids,
)
from .runtime_specs import (
    recommended_stack_markdown as recommended_stack_markdown,
)
from .runtime_specs import (
    refresh_hardware as refresh_hardware,
)
from .storage import exclusive_file_lock as exclusive_file_lock


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
        from .edge_tts_client import compatibility_problem

        if problem := compatibility_problem():
            return BackendStatus("broken", "客户端需更新", problem)
        return BackendStatus("ready", "可用", "无需 API Key；需要联网")
    if backend.runtime == "http":
        return BackendStatus("external", "外部服务", "保存服务地址后请先用短音频试运行")
    if backend.id == "moss_transcribe_diarize":
        from . import moss

        if (
            not moss.runtime_python().is_file()
            or not (moss.runtime_root() / "ready.json").is_file()
        ):
            return BackendStatus("missing", "未安装", "缺少 MOSS 独立运行环境")
        if not moss.complete_model():
            return BackendStatus("partial", "模型不完整", "点击下载以校验并恢复固定快照")
        return BackendStatus("ready", "可用", str(moss.runtime_python()))
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
    if backend.id in {"indextts2_5", "indextts2"}:
        is_25 = backend.id == "indextts2_5"
        configured = str(
            getattr(
                settings,
                "tts_index25_model_path" if is_25 else "tts_model_path",
                "",
            )
            or ""
        ).strip()
        if not configured:
            return BackendStatus("missing", "未安装")
        model_dir = Path(configured).expanduser().resolve()
        executable = next(
            (
                path
                for path in runtime_executable_candidates(
                    model_dir.parent,
                    "python" if is_25 else "indextts2",
                )
                if path.is_file()
            ),
            None,
        )
        if executable is None:
            return BackendStatus("missing", "未安装", "缺少独立运行环境")
        required_files = INDEXTTS25_REQUIRED_FILES if is_25 else INDEXTTS_REQUIRED_FILES
        required_dirs = INDEXTTS25_REQUIRED_DIRS if is_25 else INDEXTTS_REQUIRED_DIRS
        missing = sorted(
            [name for name in required_files if not (model_dir / name).is_file()]
            + [name + "/" for name in required_dirs if not (model_dir / name).is_dir()]
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
        backend.id in _TORCH_BACKENDS
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
    if backend.id in {"indextts2_5", "indextts2", "moss_transcribe_diarize"}:
        return overall

    repository: str | None = None
    if backend.id == "faster_whisper":
        repository = _FASTER_WHISPER_REPOS.get(model_id, model_id if "/" in model_id else None)
    elif backend.id in {"kotoba_whisper", "qwen3_asr", "anime_whisper"}:
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
    for backend in ordered_backends(registry, settings=settings):
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


def ordered_backends(
    registry: dict[str, ModelBackend],
    settings: Any | None = None,
) -> list[ModelBackend]:
    """Order UI choices by local readiness while keeping generic APIs last."""
    original_order = {backend_id: index for index, backend_id in enumerate(registry)}

    def key(backend: ModelBackend) -> tuple[int, int, int, int, int]:
        if backend.id.startswith("generic_"):
            return (4, 1, 1, 1, original_order[backend.id])
        status = backend_status(backend, settings=settings)
        if backend.runtime == "http":
            group = 2
        elif status.state == "ready":
            group = 0
        else:
            group = 1
        # The product default leads its group; other tested defaults follow.
        default = 0 if backend.id in {RECOMMENDED_ASR_BACKEND, RECOMMENDED_TTS_BACKEND} else 1
        recommended = 0 if backend.tested_default else 1
        support = {"verified": 0, "supported": 1, "experimental": 2, "community": 3}[
            backend.support_level
        ]
        return (group, default, recommended, support, original_order[backend.id])

    return sorted(registry.values(), key=key)


def asr_default_fallback(backend: str, model: str) -> tuple[str, str] | None:
    """Return the Parakeet choice that replaces the recommended ASR until it is installed.

    ``None`` means keep ``backend``/``model``: it is not the recommended default
    or that model is ready. Callers apply this only to a default nobody chose
    (no saved ``asr_backend``), never to an explicit selection.
    """
    if (backend, model) != (RECOMMENDED_ASR_BACKEND, RECOMMENDED_ASR_MODEL):
        return None
    if backend_model_status(ASR_BACKENDS[backend], model).state == "ready":
        return None
    return FALLBACK_ASR_BACKEND, FALLBACK_ASR_MODEL


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
        for backend in ordered_backends(registry, settings=settings):
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
