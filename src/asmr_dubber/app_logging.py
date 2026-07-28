from __future__ import annotations

import logging
import re
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .platforms import portable_home

_CONFIGURE_LOCK = threading.Lock()
_CONFIGURED_PATH: Path | None = None
_SENSITIVE_PATTERNS = (
    (re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)"), r"\1[已隐藏]"),
    (
        re.compile(
            r"(?i)(\b(?:api[_ -]?key|authorization|password|secret|token)\b\s*[:=]\s*)"
            r"([^\s,;]+)"
        ),
        r"\1[已隐藏]",
    ),
    (re.compile(r"\bms-[0-9a-fA-F-]{20,}\b"), "[已隐藏]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[已隐藏]"),
)


def redact_sensitive(value: Any) -> str:
    text = str(value)
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return redact_sensitive(rendered)


def application_log_path() -> Path:
    return portable_home() / "logs" / "asmr-dubber.log"


def configure_logging() -> Path:
    """Configure one bounded, portable application log for CLI and WebUI."""

    global _CONFIGURED_PATH
    path = application_log_path().resolve()
    with _CONFIGURE_LOCK:
        if path == _CONFIGURED_PATH:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,
            backupCount=4,
            encoding="utf-8",
        )
        handler.setFormatter(
            _RedactingFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s"
            )
        )
        handler._asmr_dubber_handler = True  # type: ignore[attr-defined]
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        for existing in tuple(root.handlers):
            if getattr(existing, "_asmr_dubber_handler", False):
                root.removeHandler(existing)
                existing.close()
        root.addHandler(handler)
        logging.captureWarnings(True)
        _CONFIGURED_PATH = path
        logging.getLogger(__name__).info("日志已启动：%s", path)
    return path


def recent_log_text(max_characters: int = 30_000) -> str:
    path = configure_logging()
    if not path.is_file():
        return "日志文件尚未产生内容。"
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - max_characters * 4))
            payload = handle.read()
        return payload.decode("utf-8", errors="replace")[-max_characters:]
    except OSError as exc:
        return f"无法读取日志：{exc}"
