from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import (
    MODEL_REQUIRED_FILES,
    MODEL_REVISIONS,
    OPTIONAL_ASR_MODEL_REVISIONS,
)
from .errors import EnvironmentError
from .mirrors import hf_hub_download_with_fallback
from .platforms import current_platform


def is_wsl() -> bool:
    return current_platform().is_wsl


def ffmpeg_executable() -> str:
    override = os.getenv("ASMR_DUBBER_FFMPEG")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise EnvironmentError(f"ASMR_DUBBER_FFMPEG 不是可执行文件：{candidate}")

    system = shutil.which("ffmpeg")
    if system:
        return system

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as exc:
        raise EnvironmentError(
            "找不到 FFmpeg；请重新运行对应平台的安装脚本，或设置 ASMR_DUBBER_FFMPEG。"
        ) from exc


def cuda_summary() -> dict[str, str | bool | int | None]:
    try:
        import torch

        available = torch.cuda.is_available()
        return {
            "available": available,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if available else None,
            "capability": (
                ".".join(str(part) for part in torch.cuda.get_device_capability(0))
                if available
                else None
            ),
            "memory_bytes": (torch.cuda.get_device_properties(0).total_memory if available else 0),
        }
    except ImportError:
        return {
            "available": False,
            "torch": None,
            "cuda_runtime": None,
            "device": None,
            "capability": None,
            "memory_bytes": 0,
        }


def require_cuda() -> None:
    summary = cuda_summary()
    if not summary["available"]:
        raise EnvironmentError(
            "未检测到 PyTorch CUDA。当前 ASR/TTS 后端需要 NVIDIA GPU；"
            "请重新运行对应平台的 Recommended 安装，或切换 CPU/外部服务后端。"
        )


def cached_model_path(model_id: str) -> Path | None:
    """Return the pinned, complete local snapshot for a known model."""
    revision = MODEL_REVISIONS.get(model_id) or OPTIONAL_ASR_MODEL_REVISIONS.get(model_id)
    if revision is None:
        return None
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except (ImportError, OSError, ValueError):
        return None
    candidate = Path(path).resolve()
    if not candidate.is_dir():
        return None
    required_files = MODEL_REQUIRED_FILES.get(model_id)
    if required_files is None:
        # snapshot_download verifies the Hub manifest.  This small structural
        # check prevents an empty or metadata-only optional snapshot from
        # being accepted as an offline model.
        if not (candidate / "config.json").is_file():
            return None
        weight_patterns = ("*.safetensors", "model.bin", "pytorch_model*.bin")
        if not any(next(candidate.glob(pattern), None) for pattern in weight_patterns):
            return None
        return candidate
    for relative, expected_size in required_files.items():
        file = candidate / relative
        try:
            if not file.is_file() or file.stat().st_size != expected_size:
                return None
        except OSError:
            return None
    return candidate


def resolve_model_source(model_id: str) -> str:
    """Prefer a pinned local model; retain custom/user model ids as fallback."""
    cached = cached_model_path(model_id)
    return str(cached) if cached is not None else model_id


_UNSAFE_TRANSFORMERS_CONFIG_FIELDS = frozenset(
    {"_attn_implementation_internal", "_experts_implementation_internal"}
)


def _reject_unsafe_transformers_fields(value: Any, *, source: str) -> None:
    if isinstance(value, Mapping):
        unsafe = _UNSAFE_TRANSFORMERS_CONFIG_FIELDS.intersection(value)
        if unsafe:
            fields = "、".join(sorted(unsafe))
            raise EnvironmentError(f"模型配置包含不安全字段（{fields}）：{source}")
        for nested in value.values():
            _reject_unsafe_transformers_fields(nested, source=source)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_unsafe_transformers_fields(nested, source=source)


def resolve_transformers_model_source(model_id: str) -> tuple[str, str | None]:
    """Resolve a Transformers model without trusting mutable remote configuration.

    Known Hub models are pinned to reviewed revisions. Custom models must be
    downloaded by the user and selected through a local directory.
    """
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        config_path = candidate / "config.json"
        revision = None
        source = str(candidate.resolve())
    else:
        revision = MODEL_REVISIONS.get(model_id) or OPTIONAL_ASR_MODEL_REVISIONS.get(model_id)
        if revision is None:
            raise EnvironmentError(
                "自定义 Transformers 模型必须使用本地目录；远程仓库只允许应用内置的固定快照。"
            )
        cached = cached_model_path(model_id)
        if cached is not None:
            config_path = cached / "config.json"
            revision = None
            source = str(cached)
        else:
            try:
                config_path = Path(
                    hf_hub_download_with_fallback(
                        repo_id=model_id,
                        filename="config.json",
                        revision=revision,
                    )
                )
            except (ImportError, OSError, ValueError) as exc:
                raise EnvironmentError(f"无法取得固定模型配置：{model_id}@{revision}") from exc
            source = model_id

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EnvironmentError(f"无法验证模型配置：{config_path}") from exc
    _reject_unsafe_transformers_fields(payload, source=str(config_path))
    return source, revision


def ffmpeg_version() -> str:
    executable = ffmpeg_executable()
    completed = subprocess.run(
        [executable, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()[0]
