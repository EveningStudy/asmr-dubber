from __future__ import annotations

import json
import logging
import os
import re
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .platforms import portable_home

_CONFIGURE_LOCK = threading.Lock()
_CONFIGURED_PATH: Path | None = None
_SECRETS_CACHE_LOCK = threading.Lock()
_SECRETS_CACHE: tuple[Path, int, int, tuple[str, ...]] | None = None
_EXTRA_SECRET_ENV_VARS = ("REPLICATE_API_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_SENSITIVE_PATTERNS = (
    (re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)"), r"\1[已隐藏]"),
    (
        re.compile(
            r"(?i)(\b(?:api[_ -]?key|apikey|x-goog-api-key|authorization|password|secret|token)"
            r"\b\s*[:=]\s*[\"']?)([^\s,;&\"']+)"
        ),
        r"\1[已隐藏]",
    ),
    (re.compile(r"(?i)([?&](?:api[_ -]?key|apikey|key|token)=)([^&\s]+)"), r"\1[已隐藏]"),
    (re.compile(r"\bms-[0-9a-fA-F-]{20,}\b"), "[已隐藏]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[已隐藏]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "[已隐藏]"),
    (re.compile(r"\br8_[0-9A-Za-z]{20,}\b"), "[已隐藏]"),
    (re.compile(r"(://)([^:@/\s]+):([^@/\s]+)(@)"), r"\1[已隐藏]:[已隐藏]\4"),
)


def _saved_secret_values() -> tuple[str, ...]:
    global _SECRETS_CACHE
    try:
        from .user_settings import config_dir

        path = config_dir() / "secrets.json"
        if not path.is_file():
            return ()
        stat = path.stat()
        with _SECRETS_CACHE_LOCK:
            if (
                _SECRETS_CACHE is not None
                and _SECRETS_CACHE[0] == path
                and _SECRETS_CACHE[1] == stat.st_mtime_ns
                and _SECRETS_CACHE[2] == stat.st_size
            ):
                return _SECRETS_CACHE[3]
        raw = json.loads(path.read_text(encoding="utf-8"))
        values: list[str] = []
        if isinstance(raw, dict):
            for section in ("api_keys", "service_keys"):
                mapping = raw.get(section)
                if isinstance(mapping, dict):
                    values.extend(str(v).strip() for v in mapping.values() if str(v).strip())
        deduped = tuple(sorted({v for v in values if len(v) >= 6}, key=len, reverse=True))
        with _SECRETS_CACHE_LOCK:
            _SECRETS_CACHE = (path, stat.st_mtime_ns, stat.st_size, deduped)
        return deduped
    except Exception:
        return ()


def _known_secret_values() -> list[str]:
    values = list(_saved_secret_values())
    try:
        from .user_settings import PROVIDER_PRESETS

        env_names = [str(p.get("env") or "") for p in PROVIDER_PRESETS.values()]
    except Exception:
        env_names = []
    for name in (*env_names, *_EXTRA_SECRET_ENV_VARS):
        if name:
            val = os.environ.get(name, "").strip()
            if len(val) >= 6:
                values.append(val)
    return sorted(set(values), key=len, reverse=True)


def redact_sensitive(value: Any) -> str:
    text = str(value)
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    for secret in _known_secret_values():
        if secret in text:
            text = text.replace(secret, "[已隐藏]")
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
        return redact_sensitive(payload.decode("utf-8", errors="replace")[-max_characters:])
    except OSError as exc:
        return f"无法读取日志：{exc}"
