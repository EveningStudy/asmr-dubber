"""Compatibility facade for AutoFlow.

The implementation lives in the sibling modules of this package; this module only
re-exports the historical ``engine`` names so existing imports keep working.

Names that tests (and callers) replace at runtime are not bound here: attribute
access is forwarded to the module that defines them, so a replacement made on that
module is seen by every caller."""

# ruff: noqa: F401

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import stages
from .app_settings import (
    initial_smart_plan_defaults,
    load_app_config,
    load_smart_plan_defaults,
    save_smart_plan_defaults,
)
from .catalog import Edition, ScanResult, TrackCandidate, natural_key, scan_work
from .cli_main import (
    build_parser,
    main,
    validate_interactive_queue_arguments,
)
from .configure import (
    configure_smart_plan_draft,
    prepare_smart_plan,
    print_smart_plan_summary,
)
from .constants import (
    AUDIO_EXTENSIONS,
    DEFAULT_HARMONIZED_DELAY_MINUTES,
    DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB,
    DEFAULT_TIMESTAMP_FOOTER,
    FAILED_TASKS_FILE,
    FAILED_TASKS_SCHEMA,
    IMAGE_EXTENSIONS,
    KEYFRAME_INTERVAL_SECONDS,
    LAST_OPTIONS_FILE,
    LAST_OPTIONS_SCHEMA,
    LAYOUT_ALIASES,
    LAYOUT_BOTH,
    LAYOUT_MERGED,
    LAYOUT_SEPARATE,
    LOG_MAX_BYTES,
    MODE_ALIASES,
    MODE_AUDIO,
    MODE_VIDEO_HARMONIZED,
    MODE_VIDEO_NORMAL,
    NUMBERED_NAME,
    PERIODIC_KEYFRAME_OPTIONS,
    SAMPLE_RATE,
    SETTINGS_FILE,
    STATUS_ORDER,
    TIMESTAMP_SCHEMA,
    TOOL_ROOT,
    TRANSCRIPT_MODE_ASR_RECONCILE,
    TRANSCRIPT_MODES,
    VIDEO_FPS,
    VIDEO_SIZE,
    ReferenceEventCallback,
)
from .domain import (
    REFERENCE_SELECTION_TIMEOUT_SECONDS,
    TRANSCRIPT_MODE_DIRECT,
    AppConfig,
    AudioSource,
    QueuedSmartWork,
    SmartPlanDefaults,
    SmartPlanDraft,
    SmartTaskPlan,
    ToolPaths,
)
from .drafts import (
    prepare_smart_plan_draft,
)
from .errors import (
    StageFailedError,
    VideoPreparerError,
)
from .failed_tasks import (
    defaults_for_failed_plan,
    deserialize_audio_source,
    failed_task_payload,
    load_failed_task_plans,
    read_failed_task_records,
    relative_background_for_plan,
    serialize_audio_source,
    work_from_failed_plan,
    write_failed_task_records,
)
from .fileio import (
    atomic_copy,
    atomic_write_text,
    bracketed,
    clean_user_path,
    file_stat_payload,
    partial_output_path,
    safe_filename_component,
)
from .interactive_queue import (
    _prefixed_queue_index,
    _queue_index,
    collect_interactive_smart_plans,
    print_smart_work_queue,
)
from .media_audio import (
    concat_audio_files,
)
from .media_video import (
    background_input,
    remux_video_without_subtitles,
    render_delayed_existing_video,
)
from .merged_output_files import (
    _build_merged_outputs,
    _state_master_audio,
    _state_mixed_audio,
    _subtitle_entries_from_states,
    _timeline_from_states,
    build_merged_outputs,
    write_smart_summary,
)
from .modes import (
    layout_label,
    media_type_label,
    mode_label,
    normalize_layout,
    normalize_mode,
    parse_embed_subtitles_argument,
    video_branch_label,
)
from .output_files import (
    copy_subtitles,
    format_timestamp_from_samples,
)
from .plan_files import (
    _single_track_source,
    _smart_track_folder_name,
    cleanup_completed_workspace,
    clear_smart_outputs,
    output_mapping_complete,
    plan_identity,
    plan_metadata_path,
    prepare_smart_output_root,
    smart_job_descriptors,
    write_plan_manifest,
)
from .plan_run import (
    execute_prepared_smart_plan,
    execute_smart_plan,
)
from .planned_run import (
    execute_planned_job,
    prepare_or_resume,
)
from .processes import (
    audio_duration_samples,
    media_duration_seconds,
    media_stream_types,
    run_process_captured,
    terminate_process_tree,
    video_keyframe_times,
)
from .project_cli import (
    apply_shared_reference,
    configured_projects_root,
    ensure_autoflow_project_outputs,
    extract_shared_reference,
    known_project_manifests,
    launch_asmr_ui,
    project_asset,
    project_source_language,
    project_target_language,
    read_project,
    run_asmr_cli_captured,
    run_budgeted_stage,
    source_language_for_sources,
)
from .prompts import (
    ask_embed_subtitles,
    ask_media_type_for_draft,
    ask_mode,
    ask_output_layout,
    ask_video_branch_for_draft,
    ask_yes_no,
    configure_draft_transcripts,
    confirm_overwrite,
    parse_yes_no,
    print_header,
    print_smart_plan_defaults_panel,
)
from .runlog import (
    append_log,
)
from .self_test import (
    self_test,
)
from .source_scan import (
    _all_scan_tracks,
    _background_from_argument,
    _edition_rank,
    _parse_track_selection,
    _sorted_editions,
    background_draft_label,
    background_selection_details,
    category_label,
    default_edition_for_work,
    discover_audio,
    discover_background,
    resolve_inherited_background,
    smart_background,
    source_from_candidate,
)
from .subtitle_files import (
    _format_srt_timestamp,
    _srt_blocks,
    _srt_timestamp_ms,
    combine_lrc_files,
    combine_srt_files,
    shift_lrc_text,
    shift_srt_text,
)
from .task_media import (
    prepare_media_phase,
)
from .task_run import (
    execute_task,
)
from .title_translation import (
    TITLE_TRANSLATION_PROMPT,
    identifier_only_name,
)
from .track_prompts import (
    ask_tracks_for_draft,
    choose_tracks,
    tracks_for_edition,
    update_draft_bonus,
    update_draft_edition,
)
from .transcript_choices import (
    TRANSCRIPT_LANGUAGE_LABELS,
    default_transcript_choices,
    default_transcript_language,
    effective_sources_for_draft,
    transcript_choice_label,
    transcript_key,
    transcript_rows_for_draft,
)
from .transcripts import (
    _load_transcript_sentences,
    _qwen_script_alignment_available,
    _replace_project_with_timed_transcripts,
    import_available_source_transcript,
    overlay_timed_chinese_transcripts,
    reconcile_timeline_scripts,
)
from .work_state import (
    create_initial_state,
    create_planned_state,
    expected_output_paths,
    fingerprint,
    legacy_chinese_timeline_needs_reimport,
    load_state,
    missing_resume_artifacts,
    planned_job_key,
    planned_workspace_path,
    reset_legacy_chinese_timeline_state,
    safe_reset_workspace,
    save_state,
    state_path,
    status_at_least,
    task_key,
    workspace_path,
)

if TYPE_CHECKING:
    from .app_settings import (
        find_tool_paths,
        validate_asmr_version,
    )
    from .constants import (
        LOG_FILE,
        STATE_ROOT,
        VIDEO_FILTER_SIZE,
        WORK_ROOT,
    )
    from .failed_tasks import (
        remove_failed_task,
        save_failed_task,
    )
    from .media_audio import (
        normalize_and_concat,
    )
    from .media_video import (
        remux_video_with_subtitle,
        render_static_bilingual_video,
        render_static_video,
    )
    from .output_files import (
        copy_final_outputs,
        copy_subtitle_only_outputs,
        write_timestamp_document,
    )
    from .plan_files import (
        load_plan_metadata,
        save_plan_metadata,
    )
    from .processes import (
        run_ffmpeg,
        run_process,
    )
    from .project_cli import (
        create_asmr_project,
        project_has_external_reference,
        project_reference_id,
        run_asmr_cli,
        wait_for_reference,
    )
    from .runlog import (
        _WORK_LOG,
        log_event,
    )
    from .smart_queue import (
        execute_smart_queue,
    )
    from .title_translation import (
        translate_titles,
        translated_plan_titles,
    )
    from .work_state import (
        planned_state_path,
    )

_FORWARDED: dict[str, str] = {
    "LOG_FILE": "constants",
    "STATE_ROOT": "constants",
    "VIDEO_FILTER_SIZE": "constants",
    "WORK_ROOT": "constants",
    "_WORK_LOG": "runlog",
    "log_event": "runlog",
    "find_tool_paths": "app_settings",
    "validate_asmr_version": "app_settings",
    "run_ffmpeg": "processes",
    "run_process": "processes",
    "normalize_and_concat": "media_audio",
    "remux_video_with_subtitle": "media_video",
    "render_static_bilingual_video": "media_video",
    "render_static_video": "media_video",
    "planned_state_path": "work_state",
    "load_plan_metadata": "plan_files",
    "save_plan_metadata": "plan_files",
    "create_asmr_project": "project_cli",
    "project_has_external_reference": "project_cli",
    "project_reference_id": "project_cli",
    "run_asmr_cli": "project_cli",
    "wait_for_reference": "project_cli",
    "copy_final_outputs": "output_files",
    "copy_subtitle_only_outputs": "output_files",
    "write_timestamp_document": "output_files",
    "translate_titles": "title_translation",
    "translated_plan_titles": "title_translation",
    "remove_failed_task": "failed_tasks",
    "save_failed_task": "failed_tasks",
    "execute_smart_queue": "smart_queue",
}


def __getattr__(name: str) -> Any:
    module_name = _FORWARDED.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__package__}.{module_name}"), name)


if __name__ == "__main__":
    raise SystemExit(main())
