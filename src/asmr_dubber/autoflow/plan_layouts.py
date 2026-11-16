"""Executing the jobs of a smart plan for the merged and per-track output layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import plan_files, work_state
from .constants import ReferenceEventCallback
from .domain import AppConfig, SmartTaskPlan, ToolPaths
from .plan_files import _single_track_source
from .planned_run import execute_planned_job
from .processes import audio_duration_samples
from .project_cli import extract_shared_reference
from .work_state import save_state


def run_merged_job(
    paths: ToolPaths,
    config: AppConfig,
    plan: SmartTaskPlan,
    descriptors: list[dict[str, Any]],
    folder_translation: str,
    title_translations: dict[str, str],
    reference_event_callback: ReferenceEventCallback | None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Run the single job that processes all tracks as one merged work."""
    sources = list(plan.sources)
    descriptor = next(item for item in descriptors if item["kind"] == "merged")
    merged_folder = Path(str(descriptor["output"]))
    state = execute_planned_job(
        paths,
        config,
        source_folder=plan.folder,
        output_folder=merged_folder,
        mode=plan.mode,
        sources=list(sources),
        background=plan.background,
        plan_id=plan.plan_id,
        job_id="merged",
        folder_translation=folder_translation,
        title_translations=title_translations,
        embed_subtitles=plan.embed_subtitles,
        subtitles_only=plan.subtitles_only,
        rebuild=plan.rebuild,
        reference_event_callback=reference_event_callback,
    )
    states_by_index = {index: state for index in range(1, len(sources) + 1)}
    return states_by_index, [state]


def _usable_shared_reference(metadata: dict[str, Any]) -> dict[str, str] | None:
    shared_reference = metadata.get("shared_reference")
    if (
        not isinstance(shared_reference, dict)
        or not Path(str(shared_reference.get("audio") or "")).is_file()
    ):
        return None
    return shared_reference


def _processing_order(paths: ToolPaths, plan: SmartTaskPlan) -> list[int]:
    sources = list(plan.sources)
    # Let the longest source establish the shared voice anchor. Processing
    # order is otherwise restored in the original track order for output.
    if plan.subtitles_only:
        return list(range(len(sources)))
    durations = [audio_duration_samples(paths, item.path) for item in sources]
    anchor_index = max(range(len(sources)), key=lambda index: (durations[index], -index))
    order = [
        anchor_index,
        *[index for index in range(len(sources)) if index != anchor_index],
    ]
    print(f"\n分轨模式先处理最长音轨（第 {anchor_index + 1} 条）以建立共用参考。")
    return order


def _learn_shared_reference(
    plan: SmartTaskPlan,
    metadata: dict[str, Any],
    descriptor: dict[str, Any],
    state: dict[str, Any],
    shared_path: Path,
    capture: Path | None,
) -> dict[str, str] | None:
    """After a track ran, pick up (or extract) the voice reference to share with the rest."""
    shared_reference: dict[str, str] | None = None
    candidate = state.get("shared_reference")
    if isinstance(candidate, dict) and Path(str(candidate.get("audio") or "")).is_file():
        shared_reference = candidate
    else:
        project_json = Path(str(state.get("project_json") or ""))
        if project_json.is_file() and capture is not None:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_reference = extract_shared_reference(project_json, shared_path)
            state["shared_reference"] = shared_reference
            save_state(
                work_state.planned_state_path(plan.folder, plan.plan_id, str(descriptor["job_id"])),
                state,
            )
    if shared_reference is not None:
        metadata["shared_reference"] = shared_reference
        plan_files.save_plan_metadata(plan.plan_id, metadata)
    return shared_reference


def run_track_jobs(
    paths: ToolPaths,
    config: AppConfig,
    plan: SmartTaskPlan,
    descriptors: list[dict[str, Any]],
    metadata: dict[str, Any],
    folder_translation: str,
    title_translations: dict[str, str],
    reference_event_callback: ReferenceEventCallback | None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Run one job per track, longest first, sharing one voice reference between them."""
    sources = list(plan.sources)
    states_by_index: dict[int, dict[str, Any]] = {}
    shared_reference = _usable_shared_reference(metadata)
    shared_path = plan.output_root / ".autoflow" / "shared-reference.wav"
    for zero_index in _processing_order(paths, plan):
        index = zero_index + 1
        descriptor = next(item for item in descriptors if item.get("index") == index)
        track_source = _single_track_source(sources[zero_index])
        capture = shared_path if shared_reference is None else None
        state = execute_planned_job(
            paths,
            config,
            source_folder=plan.folder,
            output_folder=Path(str(descriptor["output"])),
            mode=plan.mode,
            sources=[track_source],
            background=plan.background,
            plan_id=plan.plan_id,
            job_id=str(descriptor["job_id"]),
            folder_translation=folder_translation,
            title_translations=title_translations,
            embed_subtitles=plan.embed_subtitles,
            subtitles_only=plan.subtitles_only,
            rebuild=plan.rebuild,
            shared_reference=shared_reference,
            capture_reference_path=capture,
            reference_event_callback=reference_event_callback,
        )
        states_by_index[index] = state
        if not plan.subtitles_only and shared_reference is None:
            shared_reference = _learn_shared_reference(
                plan, metadata, descriptor, state, shared_path, capture
            )
    return states_by_index, [states_by_index[index] for index in range(1, len(sources) + 1)]
