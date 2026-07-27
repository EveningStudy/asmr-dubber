from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .storage import atomic_write_text, exclusive_file_lock

_MAX_EVENTS = 200


def _safe_detail(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return str(value)


def _append_event(project_dir: Path, event: Mapping[str, Any]) -> None:
    destination = project_dir / "performance.json"
    with exclusive_file_lock(project_dir / ".performance.lock"):
        events: list[dict[str, Any]] = []
        if destination.is_file():
            try:
                loaded = json.loads(destination.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    events = [item for item in loaded if isinstance(item, dict)]
            except (OSError, json.JSONDecodeError):
                events = []
        events.append(dict(event))
        events = events[-_MAX_EVENTS:]
        payload = json.dumps(events, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(destination, payload)


@contextmanager
def measure_stage(
    project_dir: Path,
    stage: str,
    **initial_details: Any,
) -> Iterator[dict[str, Any]]:
    """Record one project stage without exposing prompts, text, keys, or paths."""
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    details = dict(initial_details)
    status = "completed"
    error_type: str | None = None
    try:
        yield details
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        event: dict[str, Any] = {
            "stage": stage,
            "status": status,
            "started_at": started_at,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "details": {
                key: _safe_detail(value)
                for key, value in details.items()
                if not key.lower().endswith(("key", "token", "prompt", "text", "path"))
            },
        }
        if error_type:
            event["error_type"] = error_type
        _append_event(project_dir.resolve(), event)
