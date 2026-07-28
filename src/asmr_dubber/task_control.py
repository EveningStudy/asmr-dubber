from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

from .errors import OperationCancelledError


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """Stop a child process and its descendants without touching unrelated work."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class CancellationToken(threading.Event):
    """An Event that also owns the subprocesses started by one UI operation."""

    def __init__(self) -> None:
        super().__init__()
        self._process_lock = threading.Lock()
        self._processes: set[subprocess.Popen[Any]] = set()
        self._callbacks: set[Callable[[], None]] = set()

    def register_process(self, process: subprocess.Popen[Any]) -> None:
        with self._process_lock:
            if self.is_set():
                should_stop = True
            else:
                self._processes.add(process)
                should_stop = False
        if should_stop:
            terminate_process_tree(process)
            raise OperationCancelledError("操作已取消；已经完成并保存的内容会保留。")

    def unregister_process(self, process: subprocess.Popen[Any]) -> None:
        with self._process_lock:
            self._processes.discard(process)

    def register_callback(self, callback: Callable[[], None]) -> None:
        with self._process_lock:
            if self.is_set():
                should_cancel = True
            else:
                self._callbacks.add(callback)
                should_cancel = False
        if should_cancel:
            with suppress(Exception):
                callback()
            raise OperationCancelledError("操作已取消；已经完成并保存的内容会保留。")

    def unregister_callback(self, callback: Callable[[], None]) -> None:
        with self._process_lock:
            self._callbacks.discard(callback)

    def set(self) -> None:
        super().set()
        with self._process_lock:
            processes = tuple(self._processes)
            callbacks = tuple(self._callbacks)
        for process in processes:
            terminate_process_tree(process)
        for callback in callbacks:
            # Cancellation must continue even if one optional resource has
            # already closed itself on another thread.
            with suppress(Exception):
                callback()


CancellationSignal = threading.Event | CancellationToken
_CURRENT_CANCELLATION: ContextVar[CancellationSignal | None] = ContextVar(
    "asmr_dubber_cancellation",
    default=None,
)


@contextmanager
def cancellation_scope(signal: CancellationSignal | None) -> Iterator[None]:
    marker = _CURRENT_CANCELLATION.set(signal)
    try:
        yield
    finally:
        _CURRENT_CANCELLATION.reset(marker)


def current_cancellation() -> CancellationSignal | None:
    return _CURRENT_CANCELLATION.get()


def check_cancelled(
    signal: CancellationSignal | None = None,
    message: str = "操作已取消；已经完成并保存的内容会保留。",
) -> None:
    active = signal if signal is not None else current_cancellation()
    if active is not None and active.is_set():
        raise OperationCancelledError(message)


def register_process(
    process: subprocess.Popen[Any],
    signal: CancellationSignal | None = None,
) -> CancellationSignal | None:
    active = signal if signal is not None else current_cancellation()
    if isinstance(active, CancellationToken):
        active.register_process(process)
    elif active is not None and active.is_set():
        terminate_process_tree(process)
        check_cancelled(active)
    return active


def unregister_process(
    process: subprocess.Popen[Any],
    signal: CancellationSignal | None = None,
) -> None:
    active = signal if signal is not None else current_cancellation()
    if isinstance(active, CancellationToken):
        active.unregister_process(process)


def register_cancel_callback(
    callback: Callable[[], None],
    signal: CancellationSignal | None = None,
) -> CancellationSignal | None:
    active = signal if signal is not None else current_cancellation()
    if isinstance(active, CancellationToken):
        active.register_callback(callback)
    elif active is not None and active.is_set():
        with suppress(Exception):
            callback()
        check_cancelled(active)
    return active


def unregister_cancel_callback(
    callback: Callable[[], None],
    signal: CancellationSignal | None = None,
) -> None:
    active = signal if signal is not None else current_cancellation()
    if isinstance(active, CancellationToken):
        active.unregister_callback(callback)
