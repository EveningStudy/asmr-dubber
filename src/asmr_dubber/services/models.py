"""Catalog numbers and install state come from the existing runtime registry."""

import os
import shutil
from dataclasses import asdict
from pathlib import Path

from .. import runtime_manager as runtime
from ..constants import ASMR_VAD_MODEL, DEFAULT_ALIGNER_MODEL
from ..environment import cached_model_path
from ..model_packs import import_discovered_model_packs
from ..model_registry import ASR_BACKENDS, TTS_BACKENDS
from ..platforms import portable_home
from ..separation import local_install_status, model_directory, prepare_local_model
from .settings import current


def directory_bytes(path):
    if not path.exists():
        return 0
    return sum(
        file.stat().st_size for file in path.rglob("*") if file.is_file() and not file.is_symlink()
    )


def catalog():
    settings = current()
    hardware = asdict(runtime.detect_hardware())
    home = portable_home()
    home.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(home)
    items = []
    for spec in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values()):
        if spec.installer is None:
            continue
        status = runtime.backend_status(spec, settings=settings)
        items.append(
            {
                "id": spec.id,
                "name": spec.label,
                "group": "识别" if spec.kind == "asr" else "配音",
                "description": spec.help,
                "disk_gb": spec.disk_gb,
                "vram_gb": spec.recommended_vram_gb or spec.minimum_vram_gb,
                "state": status.state,
                "detail": status.detail,
                "models": list(spec.models),
                "recommended": spec.id in {"parakeet_nemo", "indextts2"},
            }
        )
    for identifier, name, status in (
        ("asmr_vad", "ASMR 语音检测", runtime.asmr_vad_status()),
        ("qwen_aligner", "Qwen3 时间对齐", runtime.forced_aligner_status()),
    ):
        items.append(
            {
                "id": identifier,
                "name": name,
                "group": "可选",
                "description": status.detail,
                "state": status.state,
                "disk_gb": None,
                "vram_gb": None,
            }
        )
    for identifier, name, model in (
        ("separation_roformer", "人声分离 · Mel-Band RoFormer", "vocals_mel_band_roformer.ckpt"),
        ("separation_demucs", "人声分离 · Demucs", "htdemucs.yaml"),
        ("separation_demucs_ft", "人声分离 · Demucs FT", "htdemucs_ft.yaml"),
    ):
        installed = (model_directory() / model).is_file()
        items.append(
            {
                "id": identifier,
                "name": name,
                "model": model,
                "group": "可选",
                "description": local_install_status(model),
                "state": "ready" if installed else "missing",
                "disk_gb": None,
                "vram_gb": None,
            }
        )
    return {
        "hardware": hardware,
        "free_bytes": disk.free,
        "used_bytes": directory_bytes(home / "models"),
        "items": items,
    }


def download(request, report, token):
    name = "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS"
    previous = os.environ.get(name)
    os.environ[name] = "1" if current().download_source == "original" else "0"
    try:
        return _download(request, report, token)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _download(request, report, token):
    identifier = request["model"]
    if identifier.startswith("separation_"):
        models = {
            "separation_roformer": "vocals_mel_band_roformer.ckpt",
            "separation_demucs": "htdemucs.yaml",
            "separation_demucs_ft": "htdemucs_ft.yaml",
        }
        return {
            "message": prepare_local_model(
                models[identifier], install=True, source=current().download_source, log=report
            )
        }
    if identifier in {"asmr_vad", "qwen_aligner"}:
        # The advanced runtime installer already supplies both optional dependencies.
        runtime.install_backend(
            "kotoba_whisper", progress=report, log_callback=report, cancel_event=token
        )
        from ..model_pack_download import prepare_remote_model_pack
        from ..model_packs import import_model_pack

        pack = "whisper-vad-asmr-onnx" if identifier == "asmr_vad" else "qwen3-forced-aligner"
        archive = prepare_remote_model_pack(pack, log=report, cancelled=token.is_set)
        if archive:
            import_model_pack(archive, log=report, progress=report)
            return {"message": "模型已安装。"}
        from ..mirrors import snapshot_download_with_fallback

        model = ASMR_VAD_MODEL if identifier == "asmr_vad" else DEFAULT_ALIGNER_MODEL
        result = snapshot_download_with_fallback(repo_id=model)
        return {"message": str(result)}
    return {
        "message": runtime.install_backend(
            identifier, progress=report, log_callback=report, cancel_event=token
        )
    }


def import_packs(request, report, token):
    results = import_discovered_model_packs(log=report, progress=report)
    return {"message": f"已处理 {len(results)} 个模型包。"}


def remove(identifier):
    settings = current()
    candidates = []
    if identifier in ASR_BACKENDS:
        for model in ASR_BACKENDS[identifier].models:
            path = cached_model_path(model)
            if path:
                candidates.append(path)
    elif identifier in {"indextts2", "indextts2_5"}:
        candidates.append(
            Path(
                settings.tts_model_path
                if identifier == "indextts2"
                else settings.tts_index25_model_path
            )
        )
    elif identifier in {"asmr_vad", "qwen_aligner"}:
        path = cached_model_path(
            ASMR_VAD_MODEL if identifier == "asmr_vad" else DEFAULT_ALIGNER_MODEL
        )
        if path:
            candidates.append(path)
    else:
        models = {
            "separation_roformer": "vocals_mel_band_roformer.ckpt",
            "separation_demucs": "htdemucs.yaml",
            "separation_demucs_ft": "htdemucs_ft.yaml",
        }
        if identifier not in models:
            raise ValueError("Unknown model")
        candidates.append(model_directory() / models[identifier])
    roots = [(portable_home() / name).resolve() for name in ("models", "cache", "runtimes")]
    for candidate in candidates:
        resolved = candidate.resolve()
        if candidate.is_symlink() or not any(
            resolved.is_relative_to(root) and resolved != root for root in roots
        ):
            raise ValueError("Model path is outside a managed directory")
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink(missing_ok=True)
    return {"removed": identifier}
