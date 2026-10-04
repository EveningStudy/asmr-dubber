from __future__ import annotations

from pathlib import Path
from typing import Any

from ..autoflow import engine
from ..errors import ProjectError
from ..platforms import open_directory
from ..task_control import CancellationSignal, check_cancelled
from .batch_configuration import config_from_settings
from .batch_plan import deserialize_plan

MODE_LABELS = {
    engine.MODE_AUDIO: "纯音频",
    engine.MODE_VIDEO_NORMAL: "普通静态视频",
    engine.MODE_VIDEO_HARMONIZED: "和谐静态视频",
}

LAYOUT_LABELS = {
    engine.LAYOUT_MERGED: "合并成一部",
    engine.LAYOUT_SEPARATE: "每条音轨分别处理并输出（不合并）",
    engine.LAYOUT_BOTH: "分轨输出 + 合并版",
}


def add_plan_to_queue(queue_payload: Any, plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    plan = deserialize_plan(plan_payload)
    for current_payload in queue:
        current = deserialize_plan(current_payload)
        if current.folder == plan.folder:
            raise ProjectError("这个作品已经在队列里。")
        if current.output_root == plan.output_root:
            raise ProjectError(f"队列中已有任务使用输出目录：{plan.output_root}")
    queue.append(plan_payload)
    return queue


def replace_plan_in_queue(
    queue_payload: Any,
    original_plan_id: Any,
    plan_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    selected = str(original_plan_id or "")
    index = next(
        (index for index, item in enumerate(queue) if str(item.get("plan_id") or "") == selected),
        None,
    )
    if index is None:
        raise ProjectError("要修改的队列任务已经不存在。")
    plan = deserialize_plan(plan_payload)
    for current_index, current_payload in enumerate(queue):
        if current_index == index:
            continue
        current = deserialize_plan(current_payload)
        if current.folder == plan.folder:
            raise ProjectError("队列中已有这个作品。")
        if current.output_root == plan.output_root:
            raise ProjectError(f"队列中已有任务使用输出目录：{plan.output_root}")
    queue[index] = plan_payload
    return queue


def remove_plan_from_queue(queue_payload: Any, plan_id: Any) -> list[dict[str, Any]]:
    selected = str(plan_id or "")
    return [
        dict(item)
        for item in (queue_payload or [])
        if isinstance(item, dict) and str(item.get("plan_id") or "") != selected
    ]


def reorder_queue_for_ui(queue_payload: Any, ordered_ids: Any) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    requested = [str(item) for item in (ordered_ids or [])]
    by_id = {str(item.get("plan_id") or ""): item for item in queue}
    if len(requested) != len(queue) or set(requested) != set(by_id):
        raise ProjectError("队列已经变化，请重试。")
    return [by_id[item] for item in requested]


def toggle_plan_rebuild(queue_payload: Any, plan_id: Any) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    selected = str(plan_id or "")
    found = False
    for item in queue:
        if str(item.get("plan_id") or "") != selected:
            continue
        enabled = not bool(item.get("rebuild", False))
        item["rebuild"] = enabled
        item["force"] = enabled
        found = True
        break
    if not found:
        raise ProjectError("要重新处理的队列任务已经不存在。")
    return queue


def _plan_content_label(plan: engine.SmartTaskPlan) -> str:
    content = plan.edition.get("output_policy", {}).get("content", "")
    return {
        "dubbing": "双语成品",
        "replacement": "替换配音",
        "both": "双语 + 替换配音",
        "subtitles": "原声 + 字幕",
    }.get(content, "原声 + 字幕" if plan.subtitles_only else "配音")


def queue_rows(queue_payload: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for index, payload in enumerate(queue_payload or [], start=1):
        plan = deserialize_plan(payload)
        rows.append(
            [
                index,
                plan.folder.name,
                len(plan.sources),
                "仅字幕文件 · "
                + {"source": "原文", "zh": "译文", "bilingual": "双语"}[plan.subtitle_language]
                if plan.source_subtitles_only
                else f"{_plan_content_label(plan)} · {MODE_LABELS[plan.mode]}",
                LAYOUT_LABELS[plan.layout],
                str(plan.output_root),
            ]
        )
    return rows


def queue_items_for_ui(
    queue_payload: Any,
    *,
    runtime: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    queue = list(queue_payload or [])
    total = len(queue)
    runtime_by_plan = runtime or {}
    for index, payload in enumerate(queue, start=1):
        plan = deserialize_plan(payload)
        task_runtime = runtime_by_plan.get(plan.plan_id, {})
        items.append(
            {
                "id": plan.plan_id,
                "position": index,
                "work": plan.folder.name,
                "tracks": len(plan.sources),
                "mode": "仅字幕文件 · "
                + {"source": "原文", "zh": "译文", "bilingual": "双语"}[plan.subtitle_language]
                if plan.source_subtitles_only
                else f"{_plan_content_label(plan)} · {MODE_LABELS[plan.mode]}",
                "layout": LAYOUT_LABELS[plan.layout],
                "output": plan.output_root.as_posix(),
                "titles": (
                    "翻译作品名和音轨标题"
                    if plan.translate_work_title and plan.translate_track_titles
                    else "只翻译作品名"
                    if plan.translate_work_title
                    else "只翻译音轨标题"
                    if plan.translate_track_titles
                    else "保留原标题"
                ),
                "rebuild": plan.rebuild,
                "rebuild_label": "取消重新处理" if plan.rebuild else "重新处理",
                "can_move_up": index > 1,
                "can_move_down": index < total,
                "reference_ready": bool(task_runtime.get("reference_ready")),
                "reference_request_id": str(task_runtime.get("request_id") or ""),
                "reference_status": str(task_runtime.get("status") or ""),
            }
        )
    return items


def queue_choices(queue_payload: Any) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for index, payload in enumerate(queue_payload or [], start=1):
        plan = deserialize_plan(payload)
        choices.append((f"{index}. {plan.folder.name}", plan.plan_id))
    return choices


def run_queue(
    queue_payload: Any,
    *,
    cancel_event: CancellationSignal | None = None,
    reference_event_callback: engine.ReferenceEventCallback | None = None,
) -> tuple[int, list[str]]:
    plans = [deserialize_plan(payload) for payload in (queue_payload or [])]
    if not plans:
        raise ProjectError("队列还是空的，请先扫描作品并加入队列。")
    check_cancelled(cancel_event)
    config = config_from_settings()
    paths = engine.find_tool_paths(config)
    engine.validate_asmr_version()
    result = engine.execute_smart_queue(
        paths,
        config,
        plans,
        reference_event_callback=reference_event_callback,
    )
    check_cancelled(cancel_event)
    return result, [str(plan.output_root) for plan in plans]


def open_output_directory(path_value: Any) -> str:
    path = Path(str(path_value or "")).expanduser().resolve()
    opened = open_directory(path)
    return f"已打开输出目录：{opened}"


def subtitle_output_rows(output_directories: Any) -> list[list[str]]:
    """List the external subtitle files published by the completed queue."""

    rows: list[list[str]] = []
    seen: set[Path] = set()
    for value in output_directories or []:
        root = Path(str(value or "")).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(
            (
                candidate.resolve()
                for candidate in root.rglob("*")
                if candidate.is_file() and candidate.suffix.casefold() in {".srt", ".lrc"}
            ),
            key=lambda item: item.as_posix().casefold(),
        ):
            if path in seen:
                continue
            seen.add(path)
            rows.append([root.name, path.suffix.removeprefix(".").upper(), str(path)])
    return rows


def recent_log_text(max_characters: int = 30_000) -> str:
    path = engine.LOG_FILE
    if not path.is_file():
        return "还没有自动处理日志。"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"无法读取自动处理日志：{exc}"
    return text[-max_characters:]
