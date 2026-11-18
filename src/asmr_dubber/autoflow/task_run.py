"""Running a single work: the stage sequence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from . import (
    project_cli,
    runlog,
    stages,
)
from .constants import (
    ReferenceEventCallback,
)
from .domain import (
    REFERENCE_SELECTION_TIMEOUT_SECONDS,
    TRANSCRIPT_MODE_DIRECT,
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .project_cli import (
    apply_shared_reference,
    ensure_autoflow_project_outputs,
    extract_shared_reference,
    project_asset,
    project_source_language,
    project_target_language,
    read_project,
    run_budgeted_stage,
    source_language_for_sources,
)
from .task_media import prepare_media_phase
from .task_outputs import _finish_stage, _outputs_stage, _report_completion
from .transcripts import (
    import_available_source_transcript,
    overlay_timed_chinese_transcripts,
    reconcile_timeline_scripts,
)
from .work_state import (
    legacy_chinese_timeline_needs_reimport,
    reset_legacy_chinese_timeline_state,
    save_state,
    status_at_least,
)


def _create_project_stage(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
) -> None:
    runlog.emit("\n[3/5] 创建 ASMR Dubber 项目")
    from ..project_origin import batch_project_name

    project_json = project_cli.create_asmr_project(
        paths,
        Path(state["dubbing_input"]),
        source_language=source_language_for_sources(sources),
        project_name=batch_project_name(
            Path(state.get("source_folder", folder)).name,
            sources[0].path.stem if len(sources) == 1 else "合并版",
            state.get("plan_id", ""),
        ),
    )
    state["project_json"] = str(project_json)
    ensure_autoflow_project_outputs(project_json)
    from .output_policy import configure_project

    configure_project(
        project_json,
        state.get("output_policy", {}),
        subtitles_only=bool(state.get("subtitles_only")),
    )
    state["status"] = "project_created"
    save_state(state_file, state)


def _record_origin_and_check_project(
    folder: Path, state: dict[str, Any], sources: list[AudioSource]
) -> Path:
    project_json = Path(state["project_json"])
    from ..project_origin import record_batch_origin

    if project_json.is_file():
        record_batch_origin(
            project_json,
            source_folder=state.get("source_folder", folder),
            plan_id=state.get("plan_id", ""),
            job_id=state.get("job_id", ""),
            track=sources[0].path.stem if len(sources) == 1 else "合并版",
        )
    if not project_json.is_file() and not status_at_least(state, "outputs_ready"):
        raise VideoPreparerError(f"ASMR Dubber 项目已经不存在：{project_json}")
    return project_json


def _reject_unsafe_partial_fallback(state: dict[str, Any]) -> None:
    old_import = state.get("transcript_import") or {}
    current_timeline = list(state.get("timeline") or [])
    if (
        old_import.get("kind") in {"partial", "zh_overlay_partial"}
        and current_timeline
        and all(
            item.get("transcript")
            and item.get("transcript_timed")
            and item.get("transcript_language") in {"zh", "ja", "en"}
            and item.get("transcript_mode", TRANSCRIPT_MODE_DIRECT) == TRANSCRIPT_MODE_DIRECT
            for item in current_timeline
        )
    ):
        raise VideoPreparerError(
            "此任务由旧版完整字幕误回退流程创建，不能继续复用 ASR/翻译结果。"
            "请重新扫描，确认每轨字幕语言后，显式选择重做；旧项目与成品尚未修改。"
        )


def _announce_transcript_import(transcript_result: dict[str, Any]) -> None:
    kind = str(transcript_result.get("kind") or "")
    if kind == "direct":
        language = str(transcript_result.get("language") or "ja")
        runlog.emit(f"\n  已处理{language.upper()}台本/字幕。")
        reconciled = bool(transcript_result.get("script_reconciled"))
        if reconciled:
            runlog.emit("  已使用 ASR 时间轴校对台本文字；只会翻译没有匹配到中文的句子。")
        else:
            runlog.emit("  已沿用文件时间轴，不再运行不必要的 ASR（语音识别）。")
    elif kind in {"partial", "zh_overlay_partial"}:
        runlog.emit(
            f"\n  找到 {transcript_result.get('count', 0)} 份台本/字幕，"
            "但不足以覆盖全部合并音轨；本次仍运行 ASR。"
        )
    elif kind == "reconcile_after_asr":
        runlog.emit(
            f"\n  找到 {transcript_result.get('count', 0)} 份需要重新定时的台本；"
            "本次先运行 ASR，再按音轨校对文字。"
        )


def _analyze_stage(
    paths: ToolPaths, state_file: Path, state: dict[str, Any], project_json: Path
) -> None:
    transcript_result = import_available_source_transcript(
        paths,
        project_json,
        list(state.get("timeline") or []),
    )
    if transcript_result:
        state["transcript_import"] = transcript_result
        _announce_transcript_import(transcript_result)
        if str(transcript_result.get("kind") or "") == "direct":
            state["status"] = "analyzed"
            save_state(state_file, state)

    if not status_at_least(state, "analyzed"):
        runlog.emit("\n  运行 ASR（语音识别）……")
        project_cli.run_asmr_cli(paths, "analyze", str(project_json))
        state["status"] = "analyzed"
        save_state(state_file, state)


def _transcript_followups(state_file: Path, state: dict[str, Any], project_json: Path) -> None:
    transcript_result = state.get("transcript_import") or {}
    if str(transcript_result.get("kind") or "") == "reconcile_after_asr" and not state.get(
        "script_reconciliation_done"
    ):
        results = reconcile_timeline_scripts(
            project_json,
            list(state.get("timeline") or []),
        )
        state["script_reconciliation_done"] = True
        state["script_reconciliation"] = results
        runlog.emit(f"  已使用 ASR 时间轴校对 {len(results)} 份台本。")
        warning_ids = [
            str(sentence_id)
            for result in results
            for sentence_id in cast(list[object], result.get("timing_warning_ids", []))
        ]
        if warning_ids:
            preview = "、".join(warning_ids[:12])
            suffix = f"，另有 {len(warning_ids) - 12} 句" if len(warning_ids) > 12 else ""
            runlog.emit(
                f"  注意：{preview}{suffix} 的中文台本相对 ASR 时间窗过长；"
                "配音达到最大自动加速后仍可能重叠。"
            )
        save_state(state_file, state)
    if str(transcript_result.get("kind") or "") == "zh_overlay_partial" and not state.get(
        "chinese_transcript_overlay_done"
    ):
        applied = overlay_timed_chinese_transcripts(
            project_json,
            list(state.get("timeline") or []),
        )
        state["chinese_transcript_overlay_done"] = True
        state["chinese_transcript_overlay_sentences"] = applied
        runlog.emit(f"  已把已有中文字幕匹配到 {applied} 条日文识别结果。")
        save_state(state_file, state)


def _translate_stage(
    paths: ToolPaths, state_file: Path, state: dict[str, Any], project_json: Path
) -> None:
    if project_source_language(project_json) != project_target_language(project_json) and (
        state.get("output_policy", {}).get("subtitle") != "source"
        or not state.get("subtitles_only")
    ):
        runlog.emit("\n  准备目标语言文本……")
        run_budgeted_stage(paths, state, "translate", project_json)
    state["status"] = "awaiting_reference"
    save_state(state_file, state)


def _resolve_reference(
    paths: ToolPaths,
    folder: Path,
    state: dict[str, Any],
    project_json: Path,
    shared_reference: dict[str, str] | None,
    reference_event_callback: ReferenceEventCallback | None,
) -> bool:
    """Make sure a voice reference exists; False means the user did not provide one."""
    if reference_event_callback is not None:
        reference_event_callback(
            {
                "kind": "prepare",
                "plan_id": str(state.get("plan_id") or ""),
                "project_json": str(project_json.resolve()),
            }
        )
    if project_cli.project_has_external_reference(project_json):
        runlog.emit("\n  使用当前项目已配置的外部音色参考。")
    elif shared_reference is not None:
        apply_shared_reference(project_json, shared_reference)
        runlog.emit("\n  已复用本作品的统一音色参考，不需要再次选择。")
    else:
        reference_id = project_cli.project_reference_id(project_json)
        if reference_id:
            runlog.emit(f"\n复用已保存的统一音色参考：{reference_id}")
        elif project_cli.project_has_external_reference(project_json):
            runlog.emit("\n复用当前项目已经保存的外部参考音频。")
        elif not project_cli.wait_for_reference(
            paths,
            project_json,
            timeout_seconds=int(
                state.get("reference_wait_seconds", REFERENCE_SELECTION_TIMEOUT_SECONDS)
            ),
            event_callback=reference_event_callback,
            event_context={
                "plan_id": str(state.get("plan_id") or ""),
                "job_id": str(state.get("job_id") or ""),
                "work": Path(str(state.get("source_folder") or folder)).name,
                "output_folder": str(folder.resolve()),
            },
            launch_ui=reference_event_callback is None,
        ):
            return False
    return True


def _synthesize_stage(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    project_json: Path,
    shared_reference: dict[str, str] | None,
    capture_reference_path: Path | None,
    reference_event_callback: ReferenceEventCallback | None,
) -> bool:
    """Run reference selection and TTS; False means the work stops waiting for a reference."""
    if not _resolve_reference(
        paths, folder, state, project_json, shared_reference, reference_event_callback
    ):
        return False
    if capture_reference_path is not None and not state.get("shared_reference"):
        state["shared_reference"] = extract_shared_reference(
            project_json,
            capture_reference_path,
        )
        save_state(state_file, state)
    runlog.emit("\n[5/5] TTS（语音合成）、混音与字幕")
    run_budgeted_stage(paths, state, "synthesize", project_json)
    state["status"] = "synthesized"
    save_state(state_file, state)
    return True


def _mix_stage(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    project_json: Path,
) -> None:
    if state.get("output_policy", {}).get("variants"):
        from .output_policy import render_variants

        render_variants(paths, project_json, folder, state)
    else:
        run_budgeted_stage(paths, state, "mix", project_json, *stages.mix_arguments(state))
    project = read_project(project_json)
    mixed_audio = project_asset(project_json, project.get("output_file"))
    if mixed_audio is not None:
        state["project_mixed_audio"] = str(mixed_audio)
    state["status"] = "mixed"
    save_state(state_file, state)


def _subtitles_stage(
    paths: ToolPaths, state_file: Path, state: dict[str, Any], project_json: Path
) -> None:
    if not status_at_least(state, "subtitles_ready") and state.get("variant_outputs"):
        state["status"] = "subtitles_ready"
        save_state(state_file, state)
    if not status_at_least(state, "subtitles_ready"):
        project_cli.run_asmr_cli(
            paths,
            "subtitles",
            str(project_json),
            "--language",
            state.get("output_policy", {}).get("subtitle", "bilingual"),
        )
        state["status"] = "subtitles_ready"
        save_state(state_file, state)


def execute_task(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
    *,
    shared_reference: dict[str, str] | None = None,
    capture_reference_path: Path | None = None,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> None:
    if not status_at_least(state, "media_ready"):
        prepare_media_phase(paths, folder, state, sources)
        save_state(state_file, state)

    if not status_at_least(state, "project_created"):
        _create_project_stage(paths, folder, state_file, state, sources)

    project_json = _record_origin_and_check_project(folder, state, sources)
    _reject_unsafe_partial_fallback(state)

    if legacy_chinese_timeline_needs_reimport(state):
        reset_legacy_chinese_timeline_state(state)
        save_state(state_file, state)

    if not status_at_least(state, "analyzed"):
        _analyze_stage(paths, state_file, state, project_json)

    _transcript_followups(state_file, state, project_json)

    if not status_at_least(state, "awaiting_reference"):
        _translate_stage(paths, state_file, state, project_json)

    subtitles_only = bool(state.get("subtitles_only", False))
    if not subtitles_only:
        if not status_at_least(state, "synthesized") and not _synthesize_stage(
            paths,
            folder,
            state_file,
            state,
            project_json,
            shared_reference,
            capture_reference_path,
            reference_event_callback,
        ):
            return
        if not status_at_least(state, "mixed"):
            _mix_stage(paths, folder, state_file, state, project_json)

    _subtitles_stage(paths, state_file, state, project_json)

    if not status_at_least(state, "outputs_ready"):
        _outputs_stage(paths, folder, state_file, state, sources, project_json, subtitles_only)

    if not status_at_least(state, "completed"):
        _finish_stage(paths, folder, state_file, state)

    _report_completion(folder, state, project_json, subtitles_only)
