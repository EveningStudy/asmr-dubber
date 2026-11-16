"""Persisting and restoring failed tasks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import (
    ScanResult,
    scan_work,
)
from .constants import (
    FAILED_TASKS_FILE,
    FAILED_TASKS_SCHEMA,
    MODE_AUDIO,
    MODE_VIDEO_NORMAL,
    TRANSCRIPT_MODES,
)
from .domain import (
    TRANSCRIPT_MODE_DIRECT,
    AudioSource,
    QueuedSmartWork,
    SmartPlanDefaults,
    SmartPlanDraft,
    SmartTaskPlan,
)
from .errors import VideoPreparerError
from .fileio import atomic_write_text
from .modes import normalize_layout, normalize_mode
from .transcript_choices import transcript_key


def serialize_audio_source(source: AudioSource) -> dict[str, Any]:
    return {
        "order": source.order,
        "path": str(source.path),
        "title_ja": source.title_ja,
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "relative_path": source.relative_path,
        "category": source.category,
        "transcript_path": str(source.transcript_path) if source.transcript_path else None,
        "transcript_language": source.transcript_language,
        "transcript_timed": source.transcript_timed,
        "transcript_mode": source.transcript_mode,
        "source_language": source.source_language,
    }


def deserialize_audio_source(payload: Any) -> AudioSource:
    if not isinstance(payload, dict):
        raise VideoPreparerError("失败任务记录中的音轨信息无效。")
    transcript_text = str(payload.get("transcript_path") or "").strip()
    return AudioSource(
        order=int(payload.get("order") or 0),
        path=Path(str(payload.get("path") or "")).expanduser().resolve(),
        title_ja=str(payload.get("title_ja") or ""),
        size=int(payload.get("size") or 0),
        mtime_ns=int(payload.get("mtime_ns") or 0),
        relative_path=str(payload.get("relative_path") or ""),
        category=str(payload.get("category") or "main"),
        transcript_path=Path(transcript_text).expanduser().resolve() if transcript_text else None,
        transcript_language=str(payload.get("transcript_language") or "").strip() or None,
        transcript_timed=bool(payload.get("transcript_timed", False)),
        transcript_mode=(
            str(payload.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT).strip()
            if str(payload.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT).strip()
            in TRANSCRIPT_MODES
            else TRANSCRIPT_MODE_DIRECT
        ),
        source_language=str(payload.get("source_language") or "ja"),
    )


def failed_task_payload(
    plan: SmartTaskPlan,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "plan_id": plan.plan_id,
        "retry_of": plan.retry_of,
        "folder": str(plan.folder),
        "output_root": str(plan.output_root),
        "edition_label": plan.edition_label,
        "edition": plan.edition,
        "mode": plan.mode,
        "layout": plan.layout,
        "background": str(plan.background) if plan.background else None,
        "embed_subtitles": plan.embed_subtitles,
        "translate_work_title": plan.translate_work_title,
        "translate_track_titles": plan.translate_track_titles,
        "subtitles_only": plan.subtitles_only,
        "source_subtitles_only": plan.source_subtitles_only,
        "subtitle_language": plan.subtitle_language,
        "subtitle_naming": plan.subtitle_naming,
        "subtitle_custom_name": plan.subtitle_custom_name,
        "sources": [serialize_audio_source(item) for item in plan.sources],
    }
    if error:
        payload["error"] = error
    payload["failed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return payload


def read_failed_task_records() -> list[dict[str, Any]]:
    if not FAILED_TASKS_FILE.is_file():
        return []
    try:
        payload = json.loads(FAILED_TASKS_FILE.read_text(encoding="utf-8"))
        if payload.get("schema") != FAILED_TASKS_SCHEMA:
            return []
        records = payload.get("tasks") or []
        return [item for item in records if isinstance(item, dict) and item.get("plan_id")]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def write_failed_task_records(records: Sequence[dict[str, Any]]) -> None:
    if not records:
        FAILED_TASKS_FILE.unlink(missing_ok=True)
        return
    atomic_write_text(
        FAILED_TASKS_FILE,
        json.dumps(
            {
                "schema": FAILED_TASKS_SCHEMA,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tasks": list(records),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_failed_task(plan: SmartTaskPlan, *, error: str | None = None) -> None:
    records = [
        item for item in read_failed_task_records() if str(item.get("plan_id")) != plan.plan_id
    ]
    records.append(failed_task_payload(plan, error=error))
    write_failed_task_records(records)


def remove_failed_task(plan_id: str | None) -> None:
    if not plan_id:
        return
    records = [
        item for item in read_failed_task_records() if str(item.get("plan_id")) != str(plan_id)
    ]
    write_failed_task_records(records)


def load_failed_task_plans() -> list[SmartTaskPlan]:
    plans: list[SmartTaskPlan] = []
    for record in read_failed_task_records():
        try:
            sources = [deserialize_audio_source(item) for item in record.get("sources") or []]
            if not sources:
                continue
            plans.append(
                SmartTaskPlan(
                    folder=Path(str(record.get("folder") or "")).expanduser().resolve(),
                    output_root=Path(str(record.get("output_root") or "")).expanduser().resolve(),
                    edition_label=str(record.get("edition_label") or "失败任务"),
                    sources=tuple(sources),
                    edition=dict(record.get("edition") or {}),
                    mode=normalize_mode(record.get("mode")),
                    layout=normalize_layout(record.get("layout")),
                    background=(
                        Path(str(record.get("background"))).expanduser().resolve()
                        if record.get("background")
                        else None
                    ),
                    embed_subtitles=bool(record.get("embed_subtitles", True)),
                    plan_id=str(record.get("plan_id")),
                    rebuild=False,
                    force=False,
                    retry_of=str(record.get("retry_of") or "").strip() or None,
                    translate_work_title=bool(record.get("translate_work_title", True)),
                    translate_track_titles=bool(record.get("translate_track_titles", True)),
                    subtitles_only=bool(record.get("subtitles_only", False)),
                    source_subtitles_only=bool(record.get("source_subtitles_only", False)),
                    subtitle_language=str(record.get("subtitle_language", "source")),
                    subtitle_naming=str(record.get("subtitle_naming", "standard")),
                    subtitle_custom_name=str(record.get("subtitle_custom_name", "字幕")),
                )
            )
        except (TypeError, ValueError, OSError, VideoPreparerError):
            continue
    return plans


def defaults_for_failed_plan(plan: SmartTaskPlan) -> SmartPlanDefaults:
    video_mode = plan.mode if plan.mode != MODE_AUDIO else MODE_VIDEO_NORMAL
    edition = plan.edition
    background_relative: str | None = None
    if plan.background is not None:
        try:
            background_relative = (
                plan.background.resolve().relative_to(plan.folder.resolve()).as_posix()
            )
        except ValueError:
            # A valid plan normally keeps its background inside the work folder,
            # but retaining the basename is safer than losing the remembered
            # choice if an older record used an external path.
            background_relative = plan.background.name
    return SmartPlanDefaults(
        mode=plan.mode,
        video_mode=video_mode,
        layout=plan.layout,
        include_bonus=bool(edition.get("included_optional")),
        background_choice="black" if plan.background is None else "relative",
        background_relative=background_relative,
        embed_subtitles=plan.embed_subtitles,
        edition_extension=str(edition.get("extension") or "").strip() or None,
        edition_language=str(edition.get("language") or "").strip() or None,
        edition_mix_variant=str(edition.get("mix_variant") or "").strip() or None,
        edition_orientation=str(edition.get("orientation") or "").strip() or None,
    )


def relative_background_for_plan(plan: SmartTaskPlan) -> str | None:
    """Return a failed plan's background relative to its work folder."""

    if plan.background is None:
        return None
    try:
        return plan.background.resolve().relative_to(plan.folder.resolve()).as_posix()
    except ValueError:
        return plan.background.name


def work_from_failed_plan(plan: SmartTaskPlan) -> QueuedSmartWork:
    if plan.folder.is_dir():
        try:
            scan = scan_work(
                plan.folder,
                excluded_directories=("AutoFlow输出", plan.output_root.name),
            )
        except Exception:
            scan = ScanResult(plan.folder, (), (), (), (), len(plan.sources))
    else:
        scan = ScanResult(plan.folder, (), (), (), (), len(plan.sources))
    choices = {
        transcript_key(source.transcript_path): str(source.transcript_language or "zh")
        for source in plan.sources
        if source.transcript_path is not None
    }
    draft = SmartPlanDraft(
        folder=plan.folder,
        output_root=plan.output_root,
        scan=scan,
        edition_label=plan.edition_label,
        sources=list(plan.sources),
        edition=dict(plan.edition),
        include_bonus=bool(plan.edition.get("included_optional")),
        mode=plan.mode,
        video_mode=plan.mode if plan.mode != MODE_AUDIO else MODE_VIDEO_NORMAL,
        layout=plan.layout,
        background=plan.background,
        background_choice="black" if plan.background is None else "relative",
        background_relative=relative_background_for_plan(plan),
        embed_subtitles=plan.embed_subtitles,
        transcript_choices=choices,
    )
    return QueuedSmartWork(
        draft=draft,
        plan=plan,
        defaults=defaults_for_failed_plan(plan),
    )
