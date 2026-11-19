"""Catalog numbers and install state come from the existing runtime registry."""

import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from .. import runtime_manager as runtime
from ..constants import (
    ASMR_VAD_MODEL,
    DEFAULT_ALIGNER_MODEL,
    OPTIONAL_ASR_MODEL_REVISIONS,
    PROJECT_ROOT,
)
from ..environment import cached_model_path
from ..errors import AsmrDubberError, EnvironmentError, InstallPausedError
from ..model_packs import import_discovered_model_packs
from ..model_registry import ASR_BACKENDS, TTS_BACKENDS
from ..platforms import portable_home
from ..separation import (
    local_install_status,
    local_model_name,
    model_directory,
    prepare_local_model,
)
from . import model_status, onboarding
from .settings import current

SEPARATION_MODELS = {
    "separation_roformer": "vocals_mel_band_roformer.ckpt",
    "separation_demucs": "htdemucs.yaml",
    "separation_demucs_ft": "htdemucs_ft.yaml",
}
# Optional models: repository, model pack, required runtime modules, and Python extra.
OPTIONAL_MODELS = {
    "asmr_vad": (
        ASMR_VAD_MODEL,
        "whisper-vad-asmr-onnx",
        ("onnxruntime", "transformers"),
        "asr-asmr-vad",
    ),
    "qwen_aligner": (
        DEFAULT_ALIGNER_MODEL,
        "qwen3-forced-aligner",
        ("qwen_asr",),
        "asr-forced-aligner",
    ),
}


def separation_files(model):
    """Files recorded for one separation model; None until its download was verified."""
    manifest = model_directory() / f"{model}.integrity.json"
    try:
        records = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(records, dict) or model not in records:
        return None
    return {local_model_name(name) for name in records}


def directory_bytes(path):
    if not path.exists():
        return 0
    return sum(
        file.stat().st_size for file in path.rglob("*") if file.is_file() and not file.is_symlink()
    )


def model_storage(settings):
    home = portable_home()
    cache_roots = [home / "cache" / name / "hub" for name in ("huggingface", "modelscope")]
    paths = [home / "models", model_directory(), *cache_roots]
    paths.extend(
        Path(value) for value in (settings.tts_model_path, settings.tts_index25_model_path) if value
    )
    roots = []
    for path in sorted({path.resolve() for path in paths}, key=lambda path: len(path.parts)):
        if not any(path.is_relative_to(root) for root in roots):
            roots.append(path)
    return {
        "total_bytes": sum(directory_bytes(path) for path in roots),
        "cache_bytes": sum(directory_bytes(path) for path in cache_roots),
    }


def recommended_models():
    choice = onboarding.recommendation()
    return [entry["id"] for entry in (choice["asr"], choice["tts"]) if entry["id"]]


def catalog():
    settings = current()
    hardware = asdict(runtime.detect_hardware())
    recommended_ids = recommended_models()
    recommended_set = set(recommended_ids)
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
                "installation": (
                    model_status.indextts_installation_status(settings.tts_model_path)
                    if spec.id == "indextts2"
                    else model_status.indextts25_installation_status(
                        settings.tts_index25_model_path
                    )
                    if spec.id == "indextts2_5"
                    else status.detail
                ),
                "models": list(spec.models),
                "recommended": spec.id in recommended_set,
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
    for identifier, name in (
        ("separation_roformer", "人声分离 · Mel-Band RoFormer"),
        ("separation_demucs", "人声分离 · Demucs"),
        ("separation_demucs_ft", "人声分离 · Demucs FT"),
    ):
        model = SEPARATION_MODELS[identifier]
        recorded = separation_files(model)
        installed = recorded is not None and all(
            (model_directory() / file).is_file() for file in recorded
        )
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
        "used_bytes": model_storage(settings)["total_bytes"],
        "items": items,
        "recommended": recommended_ids,
        "packs": model_status.offline_model_pack_markdown(),
    }


def download(request, report, token):
    name = "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS"
    previous = os.environ.get(name)
    os.environ[name] = "1" if current().download_source == "original" else "0"
    current_progress, total_progress = 0, 0

    def progress(message="", current=None, total=None, **kwargs):
        nonlocal current_progress, total_progress
        if "desc" in kwargs:
            current, total = message if isinstance(message, tuple) else (message, 1)
            message = kwargs["desc"]
        percent = re.search(r"(?:下载|download).*?(\d+(?:\.\d+)?)%", str(message), re.I)
        curl = re.match(r"\s*(\d+)\s+[\d.]+[kMGT]?\s+\d+\s+[\d.]+[kMGT]?\s", str(message))
        match = percent or curl
        if match:
            current, total = float(match.group(1)), 100
        if current is not None and total is not None:
            current_progress, total_progress = current, total
        report(message, current_progress, total_progress)

    try:
        return _download(request, progress, token)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _install_optional_extra(extra, report, token):
    uv = runtime._uv_executable()
    if uv is None:
        raise EnvironmentError("找不到项目安装器 uv；请重新运行对应平台的 setup 脚本。")
    pypi_index = current().pypi_index_url.strip()
    env = os.environ.copy()
    env.setdefault("UV_LINK_MODE", "copy" if os.name == "nt" else "clone")
    if pypi_index:
        env["ASMR_DUBBER_PYPI_MIRROR"] = pypi_index
    lock_path = runtime.portable_home() / ".runtime-install.lock"
    with runtime.exclusive_file_lock(lock_path, timeout_seconds=30.0):
        attempts = []
        for index_url in runtime.download_candidates("pypi_indexes", preferred=pypi_index):
            report(f"使用 Python 软件源：{index_url}")
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
            completed = runtime._run_streaming_process(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                timeout_seconds=3600,
                log_callback=report,
                cancel_event=token,
            )
            if completed.returncode == 0:
                return
            attempts.append(f"{index_url}: {(completed.stderr or completed.stdout)[-800:]}")
        raise EnvironmentError(
            "安装可选运行依赖失败，所有 PyPI 源均不可用：\n" + "\n".join(attempts)
        )


def _download(request, report, token):
    identifier = request["model"]
    if identifier == "recommended":
        results = [
            runtime.install_backend(
                backend, progress=report, log_callback=report, cancel_event=token
            )
            for backend in recommended_models()
        ]
        return {"message": "\n".join(results)}
    if identifier.startswith("separation_"):
        return {
            "message": prepare_local_model(
                SEPARATION_MODELS[identifier],
                install=True,
                source=current().download_source,
                log=report,
            )
        }
    if identifier in OPTIONAL_MODELS:
        model, pack, modules, extra = OPTIONAL_MODELS[identifier]
        if any(importlib.util.find_spec(module) is None for module in modules):
            report(f"正在安装可选模型运行依赖：{extra}")
            _install_optional_extra(extra, report, token)
        from ..model_pack_download import prepare_remote_model_pack
        from ..model_packs import import_model_pack

        archive = prepare_remote_model_pack(pack, log=report, cancelled=token.is_set)
        if archive:
            import_model_pack(archive, log=report, progress=report)
            return {"message": "模型已安装。"}
        from ..mirrors import snapshot_download_with_fallback

        if token.is_set():
            raise InstallPausedError("下载已暂停；再次点击下载可继续。")
        report(f"正在下载：{model}")
        result = snapshot_download_with_fallback(
            repo_id=model,
            revision=OPTIONAL_ASR_MODEL_REVISIONS.get(model),
        )
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
    home = portable_home()
    candidates = []
    if identifier == "moss_transcribe_diarize":
        from .. import moss

        candidates.append(moss.model_directory())
    elif identifier == "parakeet_nemo":
        parakeet_dir = home / "models" / "parakeet"
        if parakeet_dir.exists():
            candidates.append(parakeet_dir)
    elif identifier == "faster_whisper":
        seen = set()
        for model in ASR_BACKENDS[identifier].models:
            repo_id = runtime._FASTER_WHISPER_REPOS.get(model, model)
            path = cached_model_path(repo_id)
            if path and path not in seen:
                seen.add(path)
                candidates.append(path)
        models_dir = home / "models"
        if models_dir.is_dir():
            for path in sorted(models_dir.glob("faster-whisper-*")):
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)
    elif identifier in ASR_BACKENDS:
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
    elif identifier in OPTIONAL_MODELS:
        path = cached_model_path(OPTIONAL_MODELS[identifier][0])
        if path:
            candidates.append(path)
    elif identifier in SEPARATION_MODELS:
        model = SEPARATION_MODELS[identifier]
        # Weights shared with another installed separation model stay on disk.
        shared = set().union(
            *(
                separation_files(other) or ()
                for other in SEPARATION_MODELS.values()
                if other != model
            )
        )
        files = (separation_files(model) or {model}) - shared
        candidates.extend(model_directory() / file for file in files)
        candidates.append(model_directory() / f"{model}.integrity.json")
    else:
        raise ValueError("未知的模型。")
    roots = [(home / name).resolve() for name in ("models", "cache", "runtimes")]
    for candidate in candidates:
        resolved = candidate.resolve()
        if candidate.is_symlink() or not any(
            resolved.is_relative_to(root) and resolved != root for root in roots
        ):
            raise ValueError("模型路径不在受管理的文件夹内。")
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink(missing_ok=True)
    if any(candidate.exists() for candidate in candidates):
        raise AsmrDubberError("模型文件未能完全删除，请关闭占用程序后重试。")
    return {"removed": identifier}
