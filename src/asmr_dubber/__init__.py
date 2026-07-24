"""ASMR Dubber cross-platform Japanese-to-Chinese dubbing application."""

from __future__ import annotations

import os
from pathlib import Path


def _use_portable_temp() -> None:
    """Keep Python/model temporary files inside the application directory."""
    home = os.getenv("ASMR_DUBBER_HOME", "").strip()
    if not home:
        return
    temporary = Path(home).expanduser() / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TMP", "TEMP"):
        os.environ[variable] = str(temporary)


_use_portable_temp()

from .platforms import configure_windows_dll_directories  # noqa: E402

configure_windows_dll_directories()

__version__ = "0.2.1"
