"""AutoFlow log file writing and context-scoped output/stage callbacks."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from ..app_logging import redact_sensitive
from . import constants, stages
from .constants import LOG_MAX_BYTES

MessageCallback = Callable[[str], None]
StageCallback = Callable[[str], None]
InputReader = Callable[[str], str]


class _UnsetType:
    pass


_UNSET = _UnsetType()

# Read-only legacy fallback kept for compatibility; active runs bind ContextVars
# via ``bind_autoflow_callbacks`` so concurrent tasks remain isolated.
_WORK_LOG: Path | None = None

_ON_MESSAGE_VAR: ContextVar[MessageCallback | None] = ContextVar(
    "autoflow_on_message", default=None
)
_ON_ERROR_VAR: ContextVar[MessageCallback | None] = ContextVar("autoflow_on_error", default=None)
_ON_STAGE_VAR: ContextVar[StageCallback | None] = ContextVar("autoflow_on_stage", default=None)
_WORK_LOG_VAR: ContextVar[Path | None] = ContextVar("autoflow_work_log", default=None)
_INPUT_READER_VAR: ContextVar[InputReader | None] = ContextVar(
    "autoflow_input_reader", default=None
)


def _bind_var[T](stack: ExitStack, var: ContextVar[T], value: T) -> None:
    token = var.set(value)
    stack.callback(var.reset, token)


def current_work_log() -> Path | None:
    bound = _WORK_LOG_VAR.get()
    return bound if bound is not None else _WORK_LOG


def resolve_input_reader(input_reader: InputReader | None = None) -> InputReader | None:
    return input_reader if input_reader is not None else _INPUT_READER_VAR.get()


@contextmanager
def bind_autoflow_callbacks(
    *,
    on_message: MessageCallback | None | _UnsetType = _UNSET,
    on_error: MessageCallback | None | _UnsetType = _UNSET,
    on_stage: StageCallback | None | _UnsetType = _UNSET,
    work_log: Path | None | _UnsetType = _UNSET,
    input_reader: InputReader | None | _UnsetType = _UNSET,
) -> Iterator[None]:
    """Bind context-local callbacks and per-work log path for the current execution."""

    with ExitStack() as stack:
        if not isinstance(on_message, _UnsetType):
            _bind_var(stack, _ON_MESSAGE_VAR, on_message)
        if not isinstance(on_error, _UnsetType):
            _bind_var(stack, _ON_ERROR_VAR, on_error)
        if not isinstance(on_stage, _UnsetType):
            _bind_var(stack, _ON_STAGE_VAR, on_stage)
        if not isinstance(work_log, _UnsetType):
            _bind_var(stack, _WORK_LOG_VAR, work_log)
        if not isinstance(input_reader, _UnsetType):
            _bind_var(stack, _INPUT_READER_VAR, input_reader)
        yield


def append_log(text: str) -> None:
    """Append console output to a UTF-8 log without exposing API keys."""

    if not text:
        return
    text = redact_sensitive(text)
    work_log = current_work_log()
    if work_log is not None:
        stages.append_work_log(work_log, text)
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


def emit(message: str = "", *, error: bool = False) -> None:
    """Write ``message`` to AutoFlow logs and invoke the context-bound message callback."""

    redacted = redact_sensitive(message) if message else ""
    append_log(redacted)
    callback = (_ON_ERROR_VAR.get() or _ON_MESSAGE_VAR.get()) if error else _ON_MESSAGE_VAR.get()
    if callback is not None:
        callback(redacted)


def emit_stage(stage: str) -> None:
    """Notify the context-bound stage callback without ever breaking the engine."""

    if not stage:
        return
    callback = _ON_STAGE_VAR.get()
    if callback is not None:
        with suppress(Exception):
            callback(stage)
