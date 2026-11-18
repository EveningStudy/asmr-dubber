"""Machine-readable outcome of one pipeline stage run from the command line.

The batch engine runs ``translate``, ``synthesize`` and ``mix`` as child
processes.  A bare exit code cannot say *why* a stage failed or *which sentences*
were left out, so the CLI writes this small JSON file when ``--result-json`` is
given, and the batch engine reads it back.  The module is dependency-light on
purpose: it is imported by every stage process.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESULT_SCHEMA = 1
STAGE_LABELS = {"translate": "翻译", "synthesize": "配音", "mix": "混音"}
# Text the pipeline puts in its error when it finished the stage but left some
# sentences without a result.  Other errors (bad key, service down) never match,
# so a stage that failed as a whole is never mistaken for a few lost sentences.
PARTIAL_MARKERS = {"translate": "句未取得译文：", "synthesize": "部分句子的逐句克隆失败"}
DEFAULT_BUDGET_COUNT = 3
DEFAULT_BUDGET_PERCENT = 2.0
MAX_LISTED = 60


def failure_budget(
    total: int,
    count: int = DEFAULT_BUDGET_COUNT,
    percent: float = DEFAULT_BUDGET_PERCENT,
) -> int:
    """Largest number of lost sentences a work may have and still be delivered."""

    pct = max(0.0, float(percent))
    raw = max(int(count), math.ceil(max(0, total) * pct / 100))
    if total <= 0:
        return max(0, raw)
    cap = max((total - 1) // 2, math.floor(total * pct / 100))
    return max(0, min(raw, cap))


def collect_failures(stage: str, project: Any) -> tuple[int, list[dict[str, str]]]:
    """Return ``(sentences the stage had to handle, sentences it failed on)``."""

    from .filtering import has_speakable_text

    total = 0
    failed: list[dict[str, str]] = []
    for sentence in project.sentences:
        if not sentence.enabled:
            continue
        if stage == "translate":
            total += 1
            if sentence.status == "translation_failed":
                failed.append({"id": sentence.id, "reason": str(sentence.error or "")[:300]})
        elif (
            stage == "synthesize"
            and sentence.chinese_audio_enabled
            and sentence.zh_text
            and has_speakable_text(sentence.zh_text)
        ):
            total += 1
            if sentence.status == "error":
                failed.append({"id": sentence.id, "reason": str(sentence.error or "")[:300]})
    return total, failed


def write_result(
    path: Path | None,
    stage: str,
    project_path: Path | str,
    error: BaseException | None = None,
) -> None:
    """Write the stage result; never raises, because reporting must not hide the real error."""

    if path is None:
        return
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "stage": stage,
        "ok": error is None,
        "error": "",
        "error_type": "",
        "total": 0,
        "failed": [],
    }
    try:
        if error is not None:
            from .app_logging import redact_sensitive

            payload["error"] = redact_sensitive(error)[:4000]
            payload["error_type"] = type(error).__name__
        if stage in PARTIAL_MARKERS:
            from .models import load_project

            project, _ = load_project(project_path)
            total, failed = collect_failures(stage, project)
            payload["total"] = total
            payload["failed"] = failed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # The exit code still tells the batch engine the stage failed.
        pass


def read_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _prior_failures(
    stage: str, prior: Sequence[Mapping[str, Any]] | None
) -> tuple[set[str], int, list[int]]:
    other = [
        item for item in (prior or []) if isinstance(item, Mapping) and item.get("stage") != stage
    ]
    ids = {
        str(entry.get("id") or "")
        for item in other
        for entry in (item.get("items") or [])
        if isinstance(entry, Mapping) and entry.get("id")
    }
    unlisted = sum(
        max(0, int(item.get("count") or 0) - len(item.get("items") or [])) for item in other
    )
    totals = [int(item.get("total") or 0) for item in other]
    return ids, unlisted, totals


def judge_partial(
    stage: str,
    result: Mapping[str, Any] | None,
    *,
    count: int = DEFAULT_BUDGET_COUNT,
    percent: float = DEFAULT_BUDGET_PERCENT,
    prior: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Decide whether a failed stage still delivers the work.

    Returns a record for ``state["partial"]`` when only a few sentences were lost
    (within ``max(count, percent%)`` and at least one sentence succeeded), or
    ``None`` when the stage must count as failed.
    """

    marker = PARTIAL_MARKERS.get(stage)
    if not marker or not result or result.get("ok"):
        return None
    if marker not in str(result.get("error") or ""):
        return None
    failed = [item for item in result.get("failed") or [] if isinstance(item, dict)]
    total = int(result.get("total") or 0)
    if not failed or len(failed) >= max(total, 1):
        return None
    if len(failed) > failure_budget(total, count, percent):
        return None
    prior_ids, unlisted, prior_totals = _prior_failures(stage, prior)
    if prior_ids or unlisted:
        current_ids = {str(item.get("id") or "") for item in failed if item.get("id")}
        cumulative = len(prior_ids | current_ids) + unlisted
        track_total = max([total, *prior_totals])
        if cumulative >= max(track_total, 1) or cumulative > failure_budget(
            track_total, count, percent
        ):
            return None
    return {
        "stage": stage,
        "count": len(failed),
        "total": total,
        "items": [
            {"id": str(item.get("id") or ""), "reason": str(item.get("reason") or "")[:300]}
            for item in failed[:MAX_LISTED]
        ],
    }


def over_budget_message(
    stage: str,
    result: Mapping[str, Any],
    *,
    count: int = DEFAULT_BUDGET_COUNT,
    percent: float = DEFAULT_BUDGET_PERCENT,
    prior: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Explain a stage failure; says so when sentences were lost beyond the budget."""

    message = str(result.get("error") or "").strip()
    failed = [item for item in result.get("failed") or [] if isinstance(item, dict)]
    marker = PARTIAL_MARKERS.get(stage)
    if marker and marker in message and failed:
        total = int(result.get("total") or 0)
        allowed = failure_budget(total, count, percent)
        listed = "；".join(f"{item.get('id')}：{item.get('reason')}" for item in failed[:5])
        more = f"（另有 {len(failed) - 5} 句）" if len(failed) > 5 else ""
        prior_ids, unlisted, prior_totals = _prior_failures(stage, prior)
        if (prior_ids or unlisted) and len(failed) <= allowed:
            current_ids = {str(item.get("id") or "") for item in failed if item.get("id")}
            cumulative = len(prior_ids | current_ids) + unlisted
            track_total = max([total, *prior_totals])
            track_allowed = failure_budget(track_total, count, percent)
            return (
                f"累计 {cumulative} 句失败（本阶段 {len(failed)} 句），"
                f"超过批量允许的 {track_allowed} 句（共 {track_total} 句）。"
                f"失败句：{listed}{more}"
            )
        return (
            f"{len(failed)} 句失败，超过批量允许的 {allowed} 句（共 {total} 句）。"
            f"失败句：{listed}{more}"
        )
    return message


def describe_partial(record: Mapping[str, Any]) -> str:
    """One Chinese sentence for logs and the plain-text summary."""

    label = {"translate": "未翻译", "synthesize": "未配音"}.get(str(record.get("stage")), "有问题")
    ids = "、".join(str(item.get("id")) for item in (record.get("items") or [])[:8])
    more = "…" if int(record.get("count") or 0) > 8 else ""
    return f"{record.get('count')} 句{label}（共 {record.get('total')} 句）：{ids}{more}"
