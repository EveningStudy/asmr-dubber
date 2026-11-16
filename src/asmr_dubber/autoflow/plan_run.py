"""Running a prepared smart plan."""

from __future__ import annotations

import json
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import OperationCancelledError
from . import failed_tasks, output_files, plan_files, runlog, stages, title_translation
from .configure import prepare_smart_plan, print_smart_plan_summary
from .constants import LAYOUT_BOTH, LAYOUT_MERGED, MODE_AUDIO, ReferenceEventCallback
from .domain import (
    AppConfig,
    AudioSource,
    SmartTaskPlan,
    ToolPaths,
)
from .errors import VideoPreparerError
from .fileio import atomic_write_text
from .merged_output_files import _timeline_from_states, build_merged_outputs, write_smart_summary
from .plan_files import (
    cleanup_completed_workspace,
    output_mapping_complete,
    prepare_smart_output_root,
    smart_job_descriptors,
    write_plan_manifest,
)
from .plan_layouts import run_merged_job, run_track_jobs
from .runlog import append_log


def _announce_plan(plan: SmartTaskPlan) -> None:
    sources = plan.sources
    print(f"\n已选择：{plan.edition_label}")
    print(f"扫描到 {len(sources)} 条音轨；输出目录：{plan.output_root}")
    if plan.mode != MODE_AUDIO:
        if plan.background is not None:
            print(f"背景图片：{plan.background.name}")
        else:
            print("背景图片：黑色背景")
        if plan.subtitles_only:
            print(f"原声视频字幕：{'生成' if plan.embed_subtitles else '不生成'}")
        else:
            print(
                f"视频字幕：{'内嵌' if plan.embed_subtitles else '不内嵌（外部 SRT/LRC 仍保留）'}"
            )


def _load_job_metadata(
    plan: SmartTaskPlan, descriptors: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[Any]]:
    """Carry previous job results into the new descriptors and persist the plan metadata."""
    metadata = plan_files.load_plan_metadata(plan.plan_id)
    previous_jobs = list(metadata.get("jobs") or [])
    previous_by_id = {
        str(item.get("job_id")): item
        for item in previous_jobs
        if isinstance(item, dict) and item.get("job_id")
    }
    for descriptor in descriptors:
        previous = previous_by_id.get(str(descriptor["job_id"]))
        if previous is None:
            continue
        for key in ("status", "outputs", "project_json"):
            if key in previous:
                descriptor[key] = previous[key]
    metadata.update(
        {
            "source_folder": str(plan.folder),
            "output_folder": str(plan.output_root),
            "mode": plan.mode,
            "layout": plan.layout,
            "edition": plan.edition,
            "background": str(plan.background) if plan.background else None,
            "embed_subtitles": plan.embed_subtitles,
            "subtitles_only": plan.subtitles_only,
            "jobs": descriptors,
        }
    )
    plan_files.save_plan_metadata(plan.plan_id, metadata)
    return metadata, previous_jobs


def _merged_outputs_reusable(
    config: AppConfig, plan: SmartTaskPlan, previous_merged: dict[str, Any]
) -> bool:
    return (
        not plan.rebuild
        and str(previous_merged.get("status") or "") == "completed"
        and output_mapping_complete(
            previous_merged.get("outputs"),
            mode=plan.mode,
            subtitles_only=plan.subtitles_only,
            embed_subtitles=plan.embed_subtitles,
            original_hard_subtitles=config.original_hard_subtitles,
        )
    )


def _finish_merged_from_tracks(
    paths: ToolPaths,
    config: AppConfig,
    plan: SmartTaskPlan,
    descriptors: list[dict[str, Any]],
    previous_jobs: list[Any],
    states: list[dict[str, Any]],
    folder_translation: str,
    title_translations: dict[str, str],
) -> None:
    """For the "both" layout, build (or reuse) the merged outputs from the track results."""
    merged_descriptor = next(item for item in descriptors if item["kind"] == "merged")
    previous_merged = next(
        (item for item in previous_jobs if item.get("kind") == "merged"),
        {},
    )
    if _merged_outputs_reusable(config, plan, previous_merged):
        print("\n合并成品已经完整，跳过重复生成。")
        merged_descriptor["outputs"] = dict(previous_merged["outputs"])
        merged_descriptor["status"] = "completed"
        output_files.write_timestamp_document(
            {
                "source_folder": str(plan.folder),
                "mode": plan.mode,
                "folder_name_translation": folder_translation,
                "title_translations": title_translations,
                "timeline": _timeline_from_states(states),
                "timestamp_footer": config.timestamp_footer,
                "timestamp_footer_position": config.timestamp_footer_position,
                "harmonized_delay_seconds": config.harmonized_delay_seconds,
            },
            Path(str(merged_descriptor["output"])),
        )
    else:
        print("\n分轨项目已完成，正在从分轨结果生成合并成品（不会再次 ASR/TTS）……")
        merged_outputs = build_merged_outputs(
            paths,
            config,
            source_folder=plan.folder,
            output_folder=Path(str(merged_descriptor["output"])),
            mode=plan.mode,
            sources=list(plan.sources),
            states=states,
            background=plan.background,
            folder_translation=folder_translation,
            title_translations=title_translations,
            embed_subtitles=plan.embed_subtitles,
            subtitles_only=plan.subtitles_only,
        )
        merged_descriptor["outputs"] = merged_outputs
        merged_descriptor["status"] = "completed"


def _record_job_results(
    descriptors: list[dict[str, Any]],
    layout: str,
    states_by_index: dict[int, dict[str, Any]],
    states: list[dict[str, Any]],
) -> None:
    for descriptor in descriptors:
        state: dict[str, Any] | None = None
        if descriptor["kind"] == "track":
            state = states_by_index.get(int(descriptor["index"]))
        elif layout == LAYOUT_MERGED:
            state = states[0]
        if state is not None:
            descriptor["status"] = str(state.get("status") or "")
            descriptor["outputs"] = dict(state.get("outputs") or {})
            descriptor["project_json"] = str(state.get("project_json") or "")


def _summary_states(
    layout: str, sources: list[AudioSource], states: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if layout != LAYOUT_MERGED:
        return states
    # The merged job's timeline already contains the complete order.
    summary_states = [{"timeline": [{"duration_samples": 0}]} for _source in sources]
    # Use the actual merged timeline to derive per-track durations where
    # possible; this avoids re-opening every source solely for a summary.
    merged_timeline = list(states[0].get("timeline") or [])
    for index, _source in enumerate(sources):
        if index < len(merged_timeline):
            summary_states[index] = {
                "timeline": [
                    {"duration_samples": merged_timeline[index].get("duration_samples", 0)}
                ]
            }
    return summary_states


def _write_plan_summary_and_manifest(
    config: AppConfig,
    plan: SmartTaskPlan,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    descriptors: list[dict[str, Any]],
    states: list[dict[str, Any]],
    folder_translation: str,
    title_translations: dict[str, str],
) -> list[dict[str, Any]]:
    """Write the summary and manifest; returns the partial-failure records."""
    output_root = plan.output_root
    sources = list(plan.sources)
    write_smart_summary(
        output_root,
        source_folder=plan.folder,
        mode=plan.mode,
        layout=plan.layout,
        sources=sources,
        states=_summary_states(plan.layout, sources, states),
        descriptors=descriptors,
        folder_translation=folder_translation,
        title_translations=title_translations,
        footer=config.timestamp_footer,
        footer_position=config.timestamp_footer_position,
        harmonized_delay_seconds=config.harmonized_delay_seconds,
    )
    partial = stages.collect_partial(list(states))
    manifest["jobs"] = descriptors
    if partial:
        manifest["partial"] = partial
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_text(
        output_root / "处理清单.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["jobs"] = descriptors
    metadata["completed_at"] = manifest["completed_at"]
    plan_files.save_plan_metadata(plan.plan_id, metadata)
    return partial


def execute_prepared_smart_plan(
    paths: ToolPaths,
    config: AppConfig,
    plan: SmartTaskPlan,
    *,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> list[dict[str, Any]]:
    """Execute a previously configured smart task without asking plan questions.

    Returns the sentences that were left untranslated or undubbed within the failure
    budget; an empty list means the work is complete.
    """

    config = replace(config, output_policy=dict(plan.edition.get("output_policy") or {}))
    if plan.source_subtitles_only:
        from .source_subtitles import execute_source_subtitles

        execute_source_subtitles(paths, config, plan)
        return []

    folder = plan.folder
    output_root = plan.output_root
    sources = list(plan.sources)
    layout = plan.layout
    runlog.log_event(
        f"开始智能任务 plan={plan.plan_id} source={folder} mode={plan.mode} layout={layout} "
        f"tracks={len(sources)} embed_subtitles={plan.embed_subtitles} "
        f"subtitles_only={plan.subtitles_only}"
    )
    prepare_smart_output_root(
        output_root, plan_id=plan.plan_id, force=plan.force, rebuild=plan.rebuild
    )

    _announce_plan(plan)

    folder_translation, title_translations = title_translation.translated_plan_titles(
        plan.plan_id,
        folder,
        list(sources),
        paths,
        translate_work_title=plan.translate_work_title,
        translate_track_titles=plan.translate_track_titles,
    )
    descriptors = smart_job_descriptors(output_root, layout, sources)
    manifest = write_plan_manifest(
        output_root / "处理清单.json",
        source_folder=folder,
        output_folder=output_root,
        mode=plan.mode,
        layout=layout,
        edition=plan.edition,
        sources=list(sources),
        background=plan.background,
        embed_subtitles=plan.embed_subtitles,
        subtitles_only=plan.subtitles_only,
        plan_id=plan.plan_id,
        jobs=descriptors,
    )
    metadata, previous_jobs = _load_job_metadata(plan, descriptors)

    if layout == LAYOUT_MERGED:
        states_by_index, states = run_merged_job(
            paths,
            config,
            plan,
            descriptors,
            folder_translation,
            title_translations,
            reference_event_callback,
        )
    else:
        states_by_index, states = run_track_jobs(
            paths,
            config,
            plan,
            descriptors,
            metadata,
            folder_translation,
            title_translations,
            reference_event_callback,
        )

    if layout == LAYOUT_BOTH:
        _finish_merged_from_tracks(
            paths,
            config,
            plan,
            descriptors,
            previous_jobs,
            states,
            folder_translation,
            title_translations,
        )

    _record_job_results(descriptors, layout, states_by_index, states)
    partial = _write_plan_summary_and_manifest(
        config,
        plan,
        manifest,
        metadata,
        descriptors,
        states,
        folder_translation,
        title_translations,
    )
    for state in {str(item.get("job_id")): item for item in states}.values():
        cleanup_completed_workspace(state)
    runlog.log_event(f"智能任务完成 plan={plan.plan_id} output={output_root}")
    print("\n智能任务完成。源作品文件没有被修改。")
    print(f"所有成品位于：{output_root}")
    if partial:
        print(f"注意：{stages.summarize(partial)}。成品已生成，这些句子保留原声/只有原文字幕。")
    return partial


def execute_smart_plan(
    paths: ToolPaths,
    config: AppConfig,
    folder: Path,
    *,
    mode_argument: str | None,
    layout_argument: str | None,
    edition_argument: str | None,
    include_bonus: bool,
    output_root_argument: str | None,
    background_argument: str | None,
    embed_subtitles_argument: str | None,
    rebuild: bool,
    force: bool,
) -> None:
    """Compatibility wrapper for a single recursive DLsite task."""

    plan = prepare_smart_plan(
        config,
        folder,
        mode_argument=mode_argument,
        layout_argument=layout_argument,
        edition_argument=edition_argument,
        include_bonus=include_bonus,
        output_root_argument=output_root_argument,
        background_argument=background_argument,
        embed_subtitles_argument=embed_subtitles_argument,
        rebuild=rebuild,
        force=force,
    )
    print_smart_plan_summary(plan)
    try:
        execute_prepared_smart_plan(paths, config, plan)
    except (KeyboardInterrupt, OperationCancelledError):
        raise
    except VideoPreparerError as exc:
        failed_tasks.save_failed_task(plan, error=str(exc))
        runlog.log_event(f"单作品任务失败 source={plan.folder} error={exc}")
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        failed_tasks.save_failed_task(plan, error=message)
        runlog.log_event(f"单作品任务未预期失败 source={plan.folder} error={message}")
        append_log(traceback.format_exc())
        raise
    else:
        failed_tasks.remove_failed_task(plan.plan_id)
        failed_tasks.remove_failed_task(plan.retry_of)
