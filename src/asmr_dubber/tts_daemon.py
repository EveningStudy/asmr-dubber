"""Resident --daemon worker pool for IndexTTS2 / IndexTTS-2.5 during batch queues (W17)."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .errors import SynthesisError
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_process,
    terminate_process_tree,
    unregister_process,
)

logger = logging.getLogger(__name__)
Progress = Callable[[str, int, int], None]


class ResidentWorkerExited(SynthesisError):
    """The resident TTS worker ended mid-batch without finishing (crash, out of memory)."""

    def __init__(self, label: str, return_code: int) -> None:
        super().__init__(f"{label} 常驻进程意外退出（退出码 {return_code}）。")
        self.return_code = return_code


class _ResidentWorker:
    def __init__(
        self,
        key: tuple[object, ...],
        label: str,
        command: list[str],
        cwd: Path | None,
        env: dict[str, str],
    ) -> None:
        self.key = key
        self.label = label
        self.command = command
        self.cwd = cwd
        self.env = env
        self.process: subprocess.Popen[str] | None = None
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.ready = False

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.ready

    def close(self) -> None:
        proc = self.process
        self.process = None
        self.ready = False
        if proc is None:
            return
        if proc.poll() is None and proc.stdin is not None:
            with suppress(OSError, ValueError):
                proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                proc.stdin.flush()
            with suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1.5)
        if proc.poll() is None:
            terminate_process_tree(proc)
        if self.reader is not None:
            self.reader.join(timeout=1.0)

    def ensure_started(
        self,
        timeout_seconds: float,
        progress: Progress | None,
        total_items: int,
        cancel_event: CancellationSignal | None,
    ) -> None:
        if self.is_alive():
            return
        self.close()
        self.messages = queue.Queue()
        try:
            proc = subprocess.Popen(
                self.command,
                start_new_session=os.name != "nt",
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=self.env,
            )
        except OSError as exc:
            raise SynthesisError(f"无法启动 {self.label} 常驻进程：{exc}") from exc
        self.process = proc

        def _read_loop() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    self.messages.put(line)
            finally:
                self.messages.put(None)

        self.reader = threading.Thread(
            target=_read_loop, name=f"{self.label}-daemon-reader", daemon=True
        )
        self.reader.start()

        register_process(proc, cancel_event)
        tail: deque[str] = deque(maxlen=80)
        started = time.monotonic()
        last_hb = started
        deadline = started + max(timeout_seconds, 120.0)
        try:
            while True:
                check_cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    self.close()
                    raise SynthesisError(f"{self.label} 常驻模型加载超时。")
                try:
                    line = self.messages.get(timeout=0.25)
                except queue.Empty:
                    if proc.poll() is not None and not self.reader.is_alive():
                        detail = "".join(tail)[-2000:]
                        self.close()
                        raise SynthesisError(
                            f"{self.label} 常驻进程启动失败（退出码 {proc.returncode}）：{detail}"
                        ) from None
                    now = time.monotonic()
                    if progress and now - last_hb >= 2.0:
                        progress(
                            f"{self.label} 正在初始化常驻模型，已等待 {int(now - started)} 秒",
                            0,
                            total_items,
                        )
                        last_hb = now
                    continue
                if line is None:
                    detail = "".join(tail)[-2000:]
                    self.close()
                    raise SynthesisError(f"{self.label} 常驻进程意外退出：{detail}")
                tail.append(line)
                if line.startswith("READY:"):
                    self.ready = True
                    return
        except BaseException:
            self.close()
            raise
        finally:
            unregister_process(proc, cancel_event)

    def run_batch(
        self,
        manifest: Path,
        *,
        total_items: int,
        timeout_seconds: float,
        progress: Progress | None,
        cancel_event: CancellationSignal | None,
        on_line: Callable[[str], bool],
        output_tail: deque[str],
    ) -> int:
        self.ensure_started(timeout_seconds, progress, total_items, cancel_event)
        proc = self.process
        assert proc is not None
        stdin = proc.stdin
        assert stdin is not None
        register_process(proc, cancel_event)
        generated_count = 0
        started = time.monotonic()
        last_hb = started
        deadline = started + timeout_seconds
        batch_finished = False
        try:
            try:
                stdin.write(
                    json.dumps({"action": "run", "batch_file": str(manifest)}, ensure_ascii=False)
                    + "\n"
                )
                stdin.flush()
            except OSError as exc:
                self.close()
                raise SynthesisError(f"{self.label} 常驻进程通信中断：{exc}") from exc

            while True:
                check_cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    self.close()
                    raise SynthesisError(
                        f"{self.label} 批量生成超过 {timeout_seconds:g} 秒/句，已停止常驻进程。"
                    )
                try:
                    line = self.messages.get(timeout=0.25)
                except queue.Empty:
                    if proc.poll() is not None and (
                        self.reader is None or not self.reader.is_alive()
                    ):
                        self.close()
                        raise ResidentWorkerExited(
                            self.label, proc.returncode if proc.returncode is not None else 1
                        ) from None
                    now = time.monotonic()
                    if progress and now - last_hb >= 2.0:
                        progress(
                            f"{self.label} 常驻模型生成中，已等待 {int(now - started)} 秒"
                            f"（完成 {generated_count}/{total_items} 句）",
                            generated_count,
                            total_items,
                        )
                        last_hb = now
                    continue
                if line is None:
                    rc = proc.poll()
                    self.close()
                    raise ResidentWorkerExited(self.label, rc if rc is not None else 1)
                stripped = line.strip()
                if stripped.startswith("{") and '"batch_done"' in stripped:
                    try:
                        done = json.loads(stripped)
                        if done.get("event") == "batch_done":
                            batch_finished = True
                            return int(done.get("exit_code", 0))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                output_tail.append(line)
                if on_line(line):
                    generated_count = min(total_items, generated_count + 1)
                    deadline = time.monotonic() + timeout_seconds
                    if progress:
                        progress(
                            f"{self.label} 已生成 {generated_count}/{total_items} 句",
                            generated_count,
                            total_items,
                        )
        finally:
            if not batch_finished:
                self.close()
            unregister_process(proc, cancel_event)


_DAEMON_POOL: ContextVar[dict[tuple[object, ...], _ResidentWorker] | None] = ContextVar(
    "indextts_daemon_pool", default=None
)


def daemon_pool_active() -> bool:
    return _DAEMON_POOL.get() is not None


def evict_tts_daemons() -> None:
    pool = _DAEMON_POOL.get()
    if not pool:
        return
    for worker in list(pool.values()):
        worker.close()
    pool.clear()


@contextmanager
def tts_daemon_session() -> Iterator[None]:
    """Keep IndexTTS2 / IndexTTS-2.5 worker processes warm across batch tracks."""
    existing = _DAEMON_POOL.get()
    if existing is not None:
        yield
        return
    pool: dict[tuple[object, ...], _ResidentWorker] = {}
    token = _DAEMON_POOL.set(pool)
    try:
        yield
    finally:
        for worker in list(pool.values()):
            worker.close()
        pool.clear()
        _DAEMON_POOL.reset(token)


@contextmanager
def batch_worker_session() -> Iterator[None]:
    """Unified batch queue session keeping both ASR models and TTS daemons resident."""
    from .asr import recognition_session

    with recognition_session(), tts_daemon_session():
        yield


def run_resident_indextts_batch(
    *,
    worker_id: str,
    label: str,
    daemon_command: list[str],
    cwd: Path | None,
    env: dict[str, str],
    manifest: Path,
    total_items: int,
    timeout_seconds: float,
    progress: Progress | None,
    cancel_event: CancellationSignal | None,
    on_line: Callable[[str], bool],
    output_tail: deque[str],
) -> int:
    pool = _DAEMON_POOL.get()
    if pool is None:
        raise RuntimeError("tts_daemon_session is not active")
    from .asr import evict_recognition_session_if_low_vram

    evict_recognition_session_if_low_vram(min_free_gb=5.0)
    key = (
        worker_id,
        tuple(daemon_command),
        str(cwd or ""),
        env.get("PYTHONPATH", ""),
        env.get("HF_HUB_CACHE", ""),
    )
    # Keep at most one TTS model resident in VRAM; switch cleanly if backend changed.
    for other_key in list(pool):
        if other_key != key:
            pool.pop(other_key).close()
    worker = pool.get(key)
    if worker is None:
        worker = _ResidentWorker(key, label, daemon_command, cwd, env)
        pool[key] = worker
    return worker.run_batch(
        manifest,
        total_items=total_items,
        timeout_seconds=timeout_seconds,
        progress=progress,
        cancel_event=cancel_event,
        on_line=on_line,
        output_tail=output_tail,
    )


def run_resident_with_one_restart(
    *,
    manifest: Path,
    entries: list[tuple[str, str]],
    committed: set[str],
    label: str,
    progress: Progress | None,
    **batch: Any,
) -> int:
    """Run a resident batch; if the worker dies part-way, restart it once for the rest.

    ``entries`` pairs each sentence id with its manifest line, and ``committed`` is
    filled by the caller's ``on_line`` as lines arrive, so a restart only resubmits
    the lines the dead worker never delivered.
    """
    for attempt in (1, 2):
        remaining = [line for sentence_id, line in entries if sentence_id not in committed]
        if attempt == 2:
            if not remaining:
                return 0
            manifest.write_text("\n".join(remaining) + "\n", encoding="utf-8")
            logger.warning("%s 常驻进程意外退出，重新启动补完剩下的 %d 句", label, len(remaining))
            if progress:
                progress(
                    f"{label} 常驻进程意外退出，重新启动后补完剩下的 {len(remaining)} 句",
                    len(entries) - len(remaining),
                    len(entries),
                )
        try:
            return run_resident_indextts_batch(
                label=label,
                manifest=manifest,
                total_items=len(remaining),
                progress=progress,
                **batch,
            )
        except ResidentWorkerExited as exc:
            if attempt == 2:
                return exc.return_code
    return 1


_PROGRESS_BAR = re.compile(r"\d+%\||\d+(?:\.\d+)?\s*(?:it/s|s/it)\]")


def readable_output_tail(lines: Iterable[str], limit: int = 3000) -> str:
    """The worker's last output for an error message, without progress-bar redraws."""
    kept = [
        part.rstrip()
        for line in lines
        for part in line.replace("\r", "\n").split("\n")
        if part.strip() and not _PROGRESS_BAR.search(part)
    ]
    return "\n".join(kept)[-limit:]
