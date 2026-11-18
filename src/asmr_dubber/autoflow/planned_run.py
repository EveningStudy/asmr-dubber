"""Running one planned job and resuming a single work folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import (
    output_files,
    work_state,
)
from .constants import (
    MODE_AUDIO,
    TIMESTAMP_SCHEMA,
    ReferenceEventCallback,
)
from .domain import (
    AppConfig,
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .modes import mode_label, normalize_mode, parse_embed_subtitles_argument
from .prompts import ask_embed_subtitles, ask_mode, confirm_overwrite
from .source_scan import discover_audio, discover_background
from .task_run import execute_task
from .work_state import (
    create_initial_state,
    create_planned_state,
    fingerprint,
    legacy_chinese_timeline_needs_reimport,
    load_state,
    missing_resume_artifacts,
    planned_workspace_path,
    reset_legacy_chinese_timeline_state,
    safe_reset_workspace,
    save_state,
    state_path,
    status_at_least,
    workspace_path,
)


def execute_planned_job(
    paths: ToolPaths,
    config: AppConfig,
    *,
    source_folder: Path,
    output_folder: Path,
    mode: str,
    sources: list[AudioSource],
    background: Path | None,
    plan_id: str,
    job_id: str,
    folder_translation: str,
    title_translations: dict[str, str],
    embed_subtitles: bool,
    subtitles_only: bool,
    rebuild: bool,
    shared_reference: dict[str, str] | None = None,
    capture_reference_path: Path | None = None,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> dict[str, Any]:
    output_folder.mkdir(parents=True, exist_ok=True)
    state_file = work_state.planned_state_path(source_folder, plan_id, job_id)
    state = None if rebuild else load_state(state_file)
    current_fingerprint = fingerprint(sources, background)

    if state is not None and not rebuild:
        if bool(state.get("original_hard_subtitles", False)) != config.original_hard_subtitles:
            raise VideoPreparerError("原声硬字幕设置已改变；请明确选择重做，保留现有结果。")
        if state.get("timestamp_footer_position", "after") != config.timestamp_footer_position:
            state["timestamp_footer_position"] = config.timestamp_footer_position
            if status_at_least(state, "completed"):
                output_files.write_timestamp_document(state, output_folder)
            save_state(state_file, state)
        if state.get("fingerprint") != current_fingerprint:
            raise VideoPreparerError(
                f"{output_folder.name} 的源音频、字幕或背景发生变化；请使用 --rebuild 重做。"
            )
        if normalize_mode(mode) != MODE_AUDIO and bool(state.get("embed_subtitles", True)) != bool(
            embed_subtitles
        ):
            raise VideoPreparerError(
                f"{output_folder.name} 的字幕内嵌设置发生变化；请使用 --rebuild 重做。"
            )
        if bool(state.get("subtitles_only", False)) != bool(subtitles_only):
            raise VideoPreparerError(
                f"{output_folder.name} 的处理内容发生变化；请使用 --rebuild 重做。"
            )
        if status_at_least(state, "completed") and state.get("partial"):
            project_json = Path(str(state.get("project_json") or ""))
            if project_json.is_file():
                partial_stages = {
                    str(item.get("stage") or "") for item in state.get("partial") or []
                }
                if "translate" in partial_stages:
                    state["status"] = "analyzed"
                elif "synthesize" in partial_stages:
                    state["status"] = "awaiting_reference"
                else:
                    state["status"] = "synthesized"
                state.pop("partial", None)
                save_state(state_file, state)
            else:
                print(f"\n{output_folder.name} 之前有部分失败句子且工作区已清理，将从头重做。")
                rebuild = True
        if not rebuild:
            missing = missing_resume_artifacts(state, output_folder)
            if missing:
                timestamp_file = output_folder / "时间戳.txt"
                if missing == [timestamp_file] and status_at_least(state, "outputs_ready"):
                    print(f"\n{output_folder.name} 只缺时间戳文档，正在直接补写。")
                    state["status"] = "outputs_ready"
                    save_state(state_file, state)
                else:
                    print(f"\n{output_folder.name} 的旧任务缺少文件，将从头重做。")
                    rebuild = True
            elif legacy_chinese_timeline_needs_reimport(state):
                print(f"\n{output_folder.name} 将按已有中文字幕重建时间轴，不会重跑 ASR 或翻译。")
                reset_legacy_chinese_timeline_state(state)
                save_state(state_file, state)
            elif status_at_least(state, "completed"):
                print(f"\n跳过已经完成的任务：{output_folder.name}")
                return state
            else:
                print(
                    f"\n继续未完成任务：{output_folder.name} "
                    f"（阶段：{state.get('status') or '尚未开始'}）"
                )

    if state is None or rebuild:
        workspace = planned_workspace_path(source_folder, plan_id, job_id)
        safe_reset_workspace(workspace)
        state = create_planned_state(
            source_folder,
            output_folder,
            mode,
            sources,
            background,
            config,
            plan_id=plan_id,
            job_id=job_id,
            folder_translation=folder_translation,
            title_translations=title_translations,
            embed_subtitles=embed_subtitles,
            subtitles_only=subtitles_only,
        )
        save_state(state_file, state)
    elif not status_at_least(state, "synthesized"):
        # Waiting is a current run preference rather than an immutable media
        # setting. This also lets older resumable jobs adopt the WebUI default.
        state["reference_wait_seconds"] = max(0, int(config.reference_wait_seconds))
        save_state(state_file, state)

    state["failure_budget"] = {
        "count": int(getattr(config, "failure_budget_count", 3)),
        "percent": float(getattr(config, "failure_budget_percent", 2.0)),
    }
    execute_task(
        paths,
        output_folder,
        state_file,
        state,
        sources,
        shared_reference=shared_reference,
        capture_reference_path=capture_reference_path,
        reference_event_callback=reference_event_callback,
    )
    return load_state(state_file) or state


def prepare_or_resume(
    paths: ToolPaths,
    config: AppConfig,
    folder: Path,
    mode_argument: str | None,
    embed_subtitles_argument: str | None,
    *,
    rebuild: bool,
    force: bool,
) -> None:
    if not folder.is_dir():
        raise VideoPreparerError(f"文件夹不存在：{folder}")
    sources = discover_audio(folder)
    state_file = state_path(folder)
    state = None if rebuild else load_state(state_file)
    requested_mode = normalize_mode(mode_argument) if mode_argument else None

    if state is not None and not rebuild:
        stored_embed_subtitles = bool(state.get("embed_subtitles", True))
        state_mode = normalize_mode(state["mode"])
        if requested_mode is not None and requested_mode != state_mode:
            raise VideoPreparerError("现有任务模式与 --mode 不一致；如需更换，请使用 --rebuild。")
        if embed_subtitles_argument is not None:
            requested_embed = parse_embed_subtitles_argument(embed_subtitles_argument)
            if state_mode != MODE_AUDIO and requested_embed != stored_embed_subtitles:
                raise VideoPreparerError(
                    "现有任务的字幕内嵌设置与 --embed-subtitles 不一致；"
                    "如需更换，请使用 --rebuild。"
                )
        background = discover_background(folder) if state_mode != MODE_AUDIO else None
        current_fingerprint = fingerprint(sources, background)
    else:
        background = None
        current_fingerprint = {}

    if state is not None and not rebuild:
        if state.get("fingerprint") != current_fingerprint:
            print("\n检测到小音频或视频背景在上次任务后发生变化。")
            answer = input("输入 R 按当前文件从头重做；直接按 Enter 退出：").strip().casefold()
            if answer != "r":
                return
            rebuild = True
        if not rebuild and legacy_chinese_timeline_needs_reimport(state):
            print("\n已有中文字幕将直接重建时间轴，不会重跑 ASR 或翻译。")
            reset_legacy_chinese_timeline_state(state)
            save_state(state_file, state)
        missing = missing_resume_artifacts(state, folder) if not rebuild else []
        if not rebuild and missing:
            print("\n旧任务记录依赖的文件已经不存在：")
            for path in missing:
                print(f"  {path}")
            answer = input("输入 R 从头重做；直接按 Enter 退出：").strip().casefold()
            if answer != "r":
                return
            rebuild = True
        if not rebuild and status_at_least(state, "completed"):
            if int(state.get("timestamp_schema") or 0) < TIMESTAMP_SCHEMA:
                print("\n正在补充文件夹名称的中英文信息；不会重跑识别、配音或混音。")
                state["status"] = "outputs_ready"
                save_state(state_file, state)
            else:
                print("\n这个文件夹的任务已经完成。")
                answer = input("输入 R 从头重做；直接按 Enter 退出：").strip().casefold()
                if answer != "r":
                    return
                rebuild = True
        elif not rebuild:
            print(f"\n发现未完成任务，当前阶段：{state.get('status') or '尚未开始'}")
            answer = input("输入 1 继续；输入 2 从头开始：").strip()
            if answer == "2":
                rebuild = True
            elif answer != "1":
                return

    if state is None or rebuild:
        mode = requested_mode or ask_mode(config)
        embed_subtitles = ask_embed_subtitles(mode, embed_subtitles_argument)
        background = discover_background(folder) if mode != MODE_AUDIO else None
        current_fingerprint = fingerprint(sources, background)
        confirm_overwrite(folder, mode, force=force)
        workspace = workspace_path(folder)
        safe_reset_workspace(workspace)
        state = create_initial_state(
            folder,
            mode,
            sources,
            background,
            config,
            embed_subtitles=embed_subtitles,
        )
        save_state(state_file, state)

    print("\n将按以下顺序处理：")
    for item in sources:
        print(f"  {item.order:>4}  {item.path.name}")
    if normalize_mode(state["mode"]) != MODE_AUDIO:
        print(f"背景：{background.name if background else '未找到 null 图片，使用黑色背景'}")
    print(f"模式：{mode_label(state['mode'])}")
    if normalize_mode(state["mode"]) != MODE_AUDIO:
        print(
            "视频字幕："
            + (
                "内嵌，同时保留 SRT/LRC"
                if state.get("embed_subtitles", True)
                else "不内嵌，仅保留 SRT/LRC"
            )
        )
    execute_task(paths, folder, state_file, state, sources)
