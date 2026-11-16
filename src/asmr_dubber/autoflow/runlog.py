"""AutoFlow log file writing."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from . import constants, stages
from .constants import LOG_MAX_BYTES

# While a queue works on one plan, everything logged also goes to that work's own
# log file so a single failing work can be diagnosed on its own.
_WORK_LOG: Path | None = None


def append_log(text: str) -> None:
    """Append console output to a UTF-8 log without exposing API keys."""

    if not text:
        return
    if _WORK_LOG is not None:
        stages.append_work_log(_WORK_LOG, text)
    try:
        constants.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if constants.LOG_FILE.is_file() and constants.LOG_FILE.stat().st_size >= LOG_MAX_BYTES:
            rotated = constants.LOG_FILE.with_suffix(".log.1")
            rotated.unlink(missing_ok=True)
            os.replace(constants.LOG_FILE, rotated)
        with constants.LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except OSError:
        # Logging must never stop a media task.
        pass


def log_event(message: str) -> None:
    append_log(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}\n")
