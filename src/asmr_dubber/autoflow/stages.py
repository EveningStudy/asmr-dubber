"""Stage results, the failure budget and per-work logs for the batch engine.

The batch engine runs translate, synthesize and mix as child processes.  This
module keeps the logic that turns their result files into outcomes: what failed
and why, whether a few lost sentences still deliver the work, and where each
work's own log lives.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import ProjectError
from ..stage_result import (
    DEFAULT_BUDGET_COUNT,
    DEFAULT_BUDGET_PERCENT,
    STAGE_LABELS,
    describe_partial,
    judge_partial,
    over_budget_message,
    read_result,
)

RESULT_STAGES = frozenset(STAGE_LABELS)
WORK_LOG_MAX_BYTES = 4 * 1024 * 1024
MAX_MESSAGE = 1500


def budget_of(state: dict[str, Any]) -> tuple[int, float]:
    value = state.get("failure_budget")
    if isinstance(value, dict):
        try:
            return int(value["count"]), float(value["percent"])
        except (KeyError, TypeError, ValueError):
            pass
    return DEFAULT_BUDGET_COUNT, DEFAULT_BUDGET_PERCENT


def run_with_result(
    stage: str,
    command: list[str],
    runner: Callable[[list[str], list[str]], None],
    results_dir: Path,
    *,
    failure: Callable[[str, str, dict[str, Any] | None], Exception],
) -> dict[str, Any] | None:
    """Run ``command`` with ``--result-json``; raise ``failure(...)`` with the real reason."""

    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{stage}-{uuid.uuid4().hex}.json"
    tail: list[str] = []
    try:
        try:
            runner([*command, "--result-json", str(result_path)], tail)
        except ProjectError as exc:
            result = read_result(result_path)
            detail = str((result or {}).get("error") or "").strip()
            if not detail:
                detail = tail[-1][-300:] if tail else str(exc)
            label = STAGE_LABELS[stage]
            raise failure(stage, f"{label}失败：{detail}"[:MAX_MESSAGE], result) from exc
        return read_result(result_path)
    finally:
        result_path.unlink(missing_ok=True)


def accept_partial(state: dict[str, Any], stage: str, result: dict[str, Any] | None) -> bool:
    """Record a within-budget failure in ``state["partial"]``; False when over budget."""

    count, percent = budget_of(state)
    record = judge_partial(stage, result, count=count, percent=percent)
    if record is None:
        return False
    state["partial"] = [
        item for item in state.get("partial") or [] if item.get("stage") != stage
    ] + [record]
    return True


def failure_text(
    state: dict[str, Any], stage: str, message: str, result: dict[str, Any] | None
) -> str:
    if not result:
        return message
    count, percent = budget_of(state)
    detail = over_budget_message(stage, result, count=count, percent=percent)
    return f"{STAGE_LABELS[stage]}失败：{detail}"[:MAX_MESSAGE] if detail else message


def has_missing_dubs(state: dict[str, Any]) -> bool:
    return any(item.get("stage") == "synthesize" for item in state.get("partial") or [])


def mix_arguments(state: dict[str, Any]) -> tuple[str, ...]:
    """Extra ``mix`` options: lost dubs stay undubbed instead of stopping the mix."""

    return ("--allow-missing",) if has_missing_dubs(state) else ()


def collect_partial(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the partial records of a plan's jobs, labelling each with its track."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        job = str(state.get("job_id") or "")
        if job in seen:
            continue
        seen.add(job)
        folder = Path(str(state.get("output_folder") or state.get("source_folder") or "")).name
        track = folder if len(states) > 1 else ""
        merged.extend({**item, "job": job, "track": track} for item in state.get("partial") or [])
    return merged


def summarize(records: list[dict[str, Any]]) -> str:
    return "；".join(describe_partial(item) for item in records)


def work_log_path(root: Path, plan_id: str, work: str) -> Path:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", work).strip(" ._")[:40] or "work"
    key = re.sub(r"[^A-Za-z0-9_-]+", "", str(plan_id))[:12] or "plan"
    return root / "logs" / f"{name}-{key}.log"


def append_work_log(path: Path, text: str) -> None:
    """Append to one work's log; the file never grows beyond ``WORK_LOG_MAX_BYTES``."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size >= WORK_LOG_MAX_BYTES:
            os.replace(path, path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        # A work log must never stop a media task.
        pass
