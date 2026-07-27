"""Validate a prepared Windows portable profile without loading model weights."""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from asmr_dubber.environment import ffmpeg_version
from asmr_dubber.model_registry import ASR_BACKENDS, TTS_BACKENDS
from asmr_dubber.platforms import portable_home
from asmr_dubber.runtime_manager import (
    asmr_vad_status,
    backend_status,
    forced_aligner_status,
)
from asmr_dubber.user_settings import UserSettings

PROFILE_ALIASES = {
    "核心": "core",
    "core": "core",
    "推荐": "recommended",
    "recommended": "recommended",
    "进阶": "advanced",
    "advanced": "advanced",
}


def _status_payload(status: Any) -> dict[str, str]:
    return {"state": status.state, "label": status.label, "detail": status.detail}


def _require_ready(label: str, status: Any, result: dict[str, Any]) -> None:
    result["checks"][label] = _status_payload(status)
    if status.state != "ready":
        result["errors"].append(f"{label}: {status.label} {status.detail}".strip())


def validate(profile: str) -> dict[str, Any]:
    home = portable_home().resolve()
    result: dict[str, Any] = {
        "profile": profile,
        "portable_home": str(home),
        "checks": {},
        "errors": [],
    }

    for module_name in ("asmr_dubber", "av", "gradio", "soundfile"):
        try:
            module = importlib.import_module(module_name)
            result["checks"][f"import:{module_name}"] = getattr(module, "__version__", "ok")
        except Exception as exc:
            result["errors"].append(f"import {module_name}: {type(exc).__name__}: {exc}")

    try:
        result["checks"]["ffmpeg"] = ffmpeg_version()
    except Exception as exc:
        result["errors"].append(f"FFmpeg: {type(exc).__name__}: {exc}")

    required_base = (
        home / "venv" / "Scripts" / "python.exe",
        home / "runtimes" / "python" / "cpython-3.12.13-windows-x86_64-none" / "python.exe",
        home / "bootstrap" / "windows" / "uv" / "uv.exe",
    )
    for path in required_base:
        exists = path.is_file()
        result["checks"][f"file:{path.relative_to(home)}"] = exists
        if not exists:
            result["errors"].append(f"missing file: {path}")

    settings = UserSettings()
    if profile in {"recommended", "advanced"}:
        _require_ready(
            "asr:parakeet_nemo",
            backend_status(ASR_BACKENDS["parakeet_nemo"], settings=settings),
            result,
        )
        _require_ready(
            "tts:indextts2",
            backend_status(TTS_BACKENDS["indextts2"], settings=settings),
            result,
        )

    if profile == "advanced":
        for backend_id in ("kotoba_whisper", "faster_whisper"):
            _require_ready(
                f"asr:{backend_id}",
                backend_status(ASR_BACKENDS[backend_id], settings=settings),
                result,
            )
        _require_ready("analysis:asmr_vad", asmr_vad_status(), result)
        _require_ready("analysis:qwen_forced_aligner", forced_aligner_status(), result)
        for module_name in (
            "torch",
            "torchaudio",
            "transformers",
            "faster_whisper",
            "qwen_asr",
            "onnxruntime",
        ):
            try:
                module = importlib.import_module(module_name)
                result["checks"][f"import:{module_name}"] = getattr(module, "__version__", "ok")
            except Exception as exc:
                result["errors"].append(f"import {module_name}: {type(exc).__name__}: {exc}")

    for forbidden in (
        home / "config" / "secrets.json",
        home / "config" / "settings.json",
    ):
        if forbidden.exists():
            result["errors"].append(f"portable package contains private state: {forbidden}")

    result["ok"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=tuple(PROFILE_ALIASES))
    args = parser.parse_args()
    profile = PROFILE_ALIASES[args.profile]
    result = validate(profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
