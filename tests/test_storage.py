from pathlib import Path
from threading import Event, Thread

import pytest

from asmr_dubber.errors import ProjectError
from asmr_dubber.storage import atomic_write_text, exclusive_file_lock


def test_file_lock_is_reentrant_in_one_thread(tmp_path: Path) -> None:
    lock = tmp_path / "project.lock"

    with (
        exclusive_file_lock(lock, timeout_seconds=0.2),
        exclusive_file_lock(lock, timeout_seconds=0.2),
    ):
        assert lock.is_file()


def test_file_lock_serializes_threads_and_times_out(tmp_path: Path) -> None:
    lock = tmp_path / "project.lock"
    acquired = Event()
    release = Event()

    def holder() -> None:
        with exclusive_file_lock(lock):
            acquired.set()
            assert release.wait(timeout=2)

    thread = Thread(target=holder)
    thread.start()
    assert acquired.wait(timeout=2)
    try:
        with (
            pytest.raises(ProjectError, match="等待文件锁超时"),
            exclusive_file_lock(lock, timeout_seconds=0.05),
        ):
            pass
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_atomic_write_replaces_complete_content_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "settings.json"
    atomic_write_text(destination, "first\n")
    atomic_write_text(destination, "second\n")

    assert destination.read_text(encoding="utf-8") == "second\n"
    assert not list(tmp_path.glob(".settings.json.*.tmp"))
