"""Interactive queue collection in the terminal."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from . import failed_tasks
from .app_settings import load_smart_plan_defaults, save_smart_plan_defaults
from .configure import configure_smart_plan_draft
from .constants import LAST_OPTIONS_FILE, MODE_AUDIO
from .domain import (
    AppConfig,
    QueuedSmartWork,
    SmartTaskPlan,
)
from .drafts import prepare_smart_plan_draft
from .errors import VideoPreparerError
from .failed_tasks import load_failed_task_plans, read_failed_task_records, work_from_failed_plan
from .fileio import clean_user_path
from .modes import layout_label, mode_label, video_branch_label
from .prompts import ask_yes_no


def print_smart_work_queue(
    queue: Sequence[QueuedSmartWork],
    config: AppConfig,
) -> None:
    print("\n" + "=" * 68)
    print(f"作品串（已加入 {len(queue)} 个作品）")
    print("=" * 68)
    for index, work in enumerate(queue, start=1):
        plan = work.plan
        transcript_count = len(work.draft.scan.transcripts)
        transcript_label = (
            f"台本/字幕 {transcript_count} 份，默认中文" if transcript_count else "没有台本/字幕"
        )
        if plan.background is None:
            background = "黑色背景"
        else:
            try:
                background = plan.background.resolve().relative_to(plan.folder.resolve()).as_posix()
            except ValueError:
                background = plan.background.name
        subtitle = "内嵌字幕" if plan.embed_subtitles else "外部字幕"
        content = "仅生成字幕" if plan.subtitles_only else "中文配音和字幕"
        detail = (
            f"{content} · {layout_label(plan.layout)} · 背景：{background} · "
            f"{subtitle} · {transcript_label}"
        )
        print(
            f"  {index}. {plan.folder.name}\n"
            f"     {plan.edition_label} · {mode_label(plan.mode)} · "
            f"{video_branch_label(plan.mode, config) if plan.mode != MODE_AUDIO else '音频'}\n"
            f"     {detail}"
        )
    print("\n直接粘贴下一个作品路径：添加")
    print("输入作品编号：返回修改；输入 D+编号：删除；直接按 Enter：开始处理")


def _queue_index(value: str, maximum: int) -> int | None:
    raw = value.strip()
    if raw.casefold().startswith("d") or raw.casefold().startswith("e"):
        raw = raw[1:].strip()
    if not raw.isdigit():
        return None
    index = int(raw)
    return index if 1 <= index <= maximum else None


def _prefixed_queue_index(value: str, prefix: str, maximum: int) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}\s*(\d+)", value.strip(), re.IGNORECASE)
    if match is None:
        return None
    index = int(match.group(1))
    return index if 1 <= index <= maximum else None


def collect_interactive_smart_plans(
    config: AppConfig,
    args: argparse.Namespace,
) -> list[SmartTaskPlan]:
    """Configure a visible, editable queue before the first task starts."""

    queue: list[QueuedSmartWork] = []
    configured_folders: set[Path] = set()
    defaults = load_smart_plan_defaults(config)
    failed_available = load_failed_task_plans()
    if LAST_OPTIONS_FILE.is_file():
        print("已载入上次使用的默认选项；每个作品都可以单独修改。")

    while True:
        if queue:
            print_smart_work_queue(queue, config)
        if failed_available:
            print("\n上次失败、可重新开始的任务：")
            failed_records = {str(item.get("plan_id")): item for item in read_failed_task_records()}
            for index, plan in enumerate(failed_available, start=1):
                reason = str(failed_records.get(plan.plan_id, {}).get("error") or "").strip()
                suffix = f" · {reason}" if reason else ""
                print(f"  F{index}. {plan.folder} · {mode_label(plan.mode)}{suffix}")
        if queue:
            prompt = (
                "请粘贴下一个作品路径；输入编号修改、D+编号删除、"
                "F+编号重试、X+编号丢弃失败记录，或直接按 Enter 开始："
            )
        elif failed_available:
            prompt = "请粘贴新作品路径；输入 F+编号加入重试、X+编号丢弃失败记录："
        else:
            prompt = "请粘贴第一个作品文件夹路径："
        raw = input(prompt).strip()

        if not raw:
            if queue:
                break
            print("还没有添加作品，请先输入一个文件夹路径。")
            continue

        failed_index = _prefixed_queue_index(raw, "f", len(failed_available))
        if failed_index is not None:
            failed_plan = failed_available.pop(failed_index - 1)
            if failed_plan.folder.resolve() in configured_folders:
                print("这个作品已经在作品串中。")
                failed_available.insert(failed_index - 1, failed_plan)
                continue
            work = work_from_failed_plan(failed_plan)
            work.plan = replace(work.plan, retry_of=failed_plan.plan_id)
            queue.append(work)
            configured_folders.add(work.plan.folder.resolve())
            defaults = work.defaults
            continue

        discard_index = _prefixed_queue_index(raw, "x", len(failed_available))
        if discard_index is not None:
            failed_plan = failed_available.pop(discard_index - 1)
            if ask_yes_no(
                f"确定丢弃失败记录“{failed_plan.folder.name}”？输入 Y 丢弃，输入 N 保留："
            ):
                failed_tasks.remove_failed_task(failed_plan.plan_id)
                print("已丢弃失败记录。")
            else:
                failed_available.insert(discard_index - 1, failed_plan)
            continue

        if queue:
            edit_index = _queue_index(raw, len(queue))
            command = raw.casefold()
            if command.startswith(("d", "e")) or raw.isdigit():
                if edit_index is None:
                    print("作品编号无效，请重新输入。")
                    continue
                if command.startswith("d"):
                    work_name = queue[edit_index - 1].plan.folder.name
                    if not ask_yes_no(
                        f"确定从作品串删除第 {edit_index} 项“{work_name}”？"
                        "输入 Y 删除，输入 N 保留："
                    ):
                        continue
                    queue.pop(edit_index - 1)
                    configured_folders = {work.plan.folder.resolve() for work in queue}
                    if queue:
                        defaults = queue[-1].defaults
                        save_smart_plan_defaults(defaults)
                    print("已从作品串删除。")
                    continue

                work = queue[edit_index - 1]
                old_plan_id = work.plan.plan_id
                plan, edited_defaults, start_now = configure_smart_plan_draft(
                    work.draft,
                    config,
                    index=edit_index,
                    rebuild=args.rebuild,
                    force=args.force,
                )
                failed_ids = {item.plan_id for item in load_failed_task_plans()}
                if old_plan_id in failed_ids:
                    retry_of = work.plan.retry_of or old_plan_id
                else:
                    retry_of = work.plan.retry_of
                work.plan = replace(plan, retry_of=retry_of)
                work.defaults = edited_defaults
                if edit_index == len(queue):
                    defaults = edited_defaults
                    save_smart_plan_defaults(defaults)
                if start_now:
                    break
                continue

        try:
            folder = clean_user_path(raw)
        except VideoPreparerError as exc:
            print(f"路径无效：{exc}")
            continue
        resolved = folder.resolve()
        if resolved in configured_folders:
            print("这个作品已经在作品串中，请换一个文件夹。")
            continue
        try:
            draft = prepare_smart_plan_draft(config, folder, defaults, args)
        except VideoPreparerError as exc:
            print(f"无法配置这个作品：{exc}")
            continue
        plan, work_defaults, start_now = configure_smart_plan_draft(
            draft,
            config,
            index=len(queue) + 1,
            rebuild=args.rebuild,
            force=args.force,
        )
        if any(existing.plan.output_root == plan.output_root for existing in queue):
            print(f"这个输出目录已经被队列中的其他作品使用：{plan.output_root}")
            continue
        queue.append(QueuedSmartWork(draft=draft, plan=plan, defaults=work_defaults))
        configured_folders.add(resolved)
        defaults = work_defaults
        save_smart_plan_defaults(defaults)
        print(f"\n已加入作品串：{plan.folder.name}")
        if start_now:
            break
    return [work.plan for work in queue]
