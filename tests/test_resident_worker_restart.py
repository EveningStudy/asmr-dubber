"""A resident TTS worker that dies mid-batch is restarted once for the lines it missed."""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path

from asmr_dubber.tts_daemon import (
    readable_output_tail,
    run_resident_with_one_restart,
    tts_daemon_session,
)

# Dies after the first line of its first batch when the crash marker file is present.
DAEMON = """import json, pathlib, sys
crash = pathlib.Path(sys.argv[1])
print("READY: mock", flush=True)
for raw in sys.stdin:
    msg = json.loads(raw)
    if msg.get("action") == "quit":
        break
    batch = pathlib.Path(msg["batch_file"])
    seen = pathlib.Path(sys.argv[2])
    ids = [json.loads(l)["id"] for l in batch.read_text("utf-8").splitlines() if l.strip()]
    with seen.open("a", encoding="utf-8") as log:
        log.write(json.dumps(ids) + "\\n")
    for line in batch.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        pathlib.Path(task["output"]).write_bytes(b"RIFF")
        print(f"Generated: {task['id']}", flush=True)
        if crash.exists() and crash.read_text() != "never-again":
            if crash.read_text() == "once":
                crash.write_text("never-again")
            sys.exit(3)
    print(json.dumps({"event": "batch_done", "exit_code": 0}), flush=True)
"""


def _run(tmp_path: Path, crash_mode: str | None) -> tuple[int, set[str], list[list[str]]]:
    script = tmp_path / "daemon.py"
    script.write_text(DAEMON, encoding="utf-8")
    crash = tmp_path / "crash.txt"
    if crash_mode:
        crash.write_text(crash_mode, encoding="utf-8")
    seen = tmp_path / "seen.jsonl"
    manifest = tmp_path / "batch.jsonl"
    entries = []
    for index in range(1, 5):
        line = json.dumps({"id": f"s{index}", "output": str(tmp_path / f"s{index}.wav")})
        entries.append((f"s{index}", line))
    manifest.write_text("\n".join(line for _, line in entries) + "\n", encoding="utf-8")
    committed: set[str] = set()

    def on_line(line: str) -> bool:
        if not line.startswith("Generated:"):
            return False
        committed.add(line.partition(":")[2].strip())
        return True

    with tts_daemon_session():
        code = run_resident_with_one_restart(
            manifest=manifest,
            entries=entries,
            committed=committed,
            label="Mock",
            progress=None,
            worker_id="mock",
            daemon_command=[sys.executable, str(script), str(crash), str(seen)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout_seconds=30.0,
            cancel_event=None,
            on_line=on_line,
            output_tail=deque(maxlen=50),
        )
    batches = [json.loads(line) for line in seen.read_text("utf-8").splitlines()]
    return code, committed, batches


def test_a_worker_crash_is_retried_once_with_only_the_missing_lines(tmp_path: Path) -> None:
    code, committed, batches = _run(tmp_path, "once")
    assert code == 0
    assert committed == {"s1", "s2", "s3", "s4"}
    assert batches == [["s1", "s2", "s3", "s4"], ["s2", "s3", "s4"]]


def test_a_worker_that_keeps_crashing_gives_up_after_one_restart(tmp_path: Path) -> None:
    code, committed, batches = _run(tmp_path, "always")
    assert code != 0  # the exact code depends on how fast the process is reaped
    assert committed == {"s1", "s2"}
    assert len(batches) == 2


def test_a_healthy_worker_runs_once(tmp_path: Path) -> None:
    code, committed, batches = _run(tmp_path, None)
    assert code == 0 and committed == {"s1", "s2", "s3", "s4"} and len(batches) == 1


def test_error_detail_drops_progress_bars() -> None:
    tail = [
        ">> starting inference...\n",
        "\r  0%|          | 0/25 [00:00<?, ?it/s]"
        "\r 20%|##        | 5/25 [00:00<00:00, 46.36it/s]\n",
        "RuntimeError: CUDA out of memory\n",
    ]
    expected = ">> starting inference...\nRuntimeError: CUDA out of memory"
    assert readable_output_tail(tail) == expected
