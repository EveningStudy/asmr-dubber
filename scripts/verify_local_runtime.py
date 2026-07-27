"""Small release/runtime smoke check with no model downloads or inference."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from typing import Any

import asmr_dubber


def main() -> int:
    ffmpeg = os.environ.get("ASMR_DUBBER_FFMPEG")
    result: dict[str, Any] = {
        "asmr_dubber": asmr_dubber.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": ffmpeg,
        "imports": {},
    }
    failed = False
    for name in ("torch", "torchaudio", "transformers", "faster_whisper"):
        try:
            module = importlib.import_module(name)
            result["imports"][name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            failed = True
            result["imports"][name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        pass
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
