"""Small atomic file helpers."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from .errors import VideoPreparerError


def clean_user_path(value: str) -> Path:
    text = value.strip().strip('"').strip("'")
    if not text:
        raise VideoPreparerError("没有输入作品文件夹路径。")
    return Path(text).expanduser().resolve()


_RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def safe_filename_component(value: str, *, fallback: str = "未命名", limit: int = 80) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" ._")
    text = re.sub(r"\s+", " ", text)
    if not text:
        text = fallback
    text = text[:limit].rstrip(" ._") or fallback
    if text.partition(".")[0].upper() in _RESERVED_WINDOWS_NAMES:
        text = f"_{text}"[:limit].rstrip(" ._") or fallback
    return text


def file_stat_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def partial_output_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.stem}.partial.{uuid.uuid4().hex}{destination.suffix}"
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_output_path(destination)
    try:
        with source.open("rb") as reader, partial.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def atomic_write_text(destination: Path, text: str, *, encoding: str = "utf-8-sig") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def bracketed(value: str) -> str:
    text = value.strip()
    if text.startswith("【") and text.endswith("】"):
        return text
    return f"【{text}】"
