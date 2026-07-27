from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import ProjectError

_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_STATE = threading.local()


def _process_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(resolved, threading.RLock())


@contextmanager
def exclusive_file_lock(path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Hold a cross-process lock represented by ``path``.

    The additional re-entrant process lock makes nested saves deterministic on
    Windows, where byte-range locks are process scoped rather than thread scoped.
    """

    lock_path = path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = _process_lock(lock_path)
    if not process_lock.acquire(timeout=timeout_seconds):
        raise ProjectError(f"等待文件锁超时：{lock_path}")
    held: dict[Path, int] = getattr(_THREAD_STATE, "held", {})
    if not hasattr(_THREAD_STATE, "held"):
        _THREAD_STATE.held = held
    if held.get(lock_path, 0):
        held[lock_path] += 1
        try:
            yield
        finally:
            held[lock_path] -= 1
            process_lock.release()
        return
    held[lock_path] = 1
    handle = None
    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise ProjectError(f"项目正在被另一个进程使用：{lock_path.parent}") from exc
                time.sleep(0.05)
        yield
    finally:
        if handle is not None:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()
        held.pop(lock_path, None)
        process_lock.release()


def atomic_write_text(
    destination: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> None:
    """Durably replace a text file without sharing a fixed temporary name."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None and os.name != "nt":
            temporary.chmod(mode)
        os.replace(temporary, destination)
        if mode is not None and os.name != "nt":
            destination.chmod(mode)
        if os.name != "nt":
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
