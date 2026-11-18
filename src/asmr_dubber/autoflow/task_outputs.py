"""Output, title and completion stages of a single work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import (
    media_audio,
    media_video,
    output_files,
    runlog,
    title_translation,
)
from .constants import (
    MODE_AUDIO,
    MODE_VIDEO_HARMONIZED,
    TIMESTAMP_SCHEMA,
)
from .domain import (
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .modes import normalize_mode
from .work_state import (
    save_state,
)


def _copy_subtitle_only_stage(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
    project_json: Path,
) -> None:
    master_audio = Path(str(state.get("master_audio") or ""))
    if (
        normalize_mode(state["mode"]) != MODE_AUDIO
        and bool(state.get("embed_subtitles", True))
        and not master_audio.is_file()
    ):
        master_audio, _timeline = media_audio.normalize_and_concat(
            paths,
            sources,
            Path(str(state["workspace"])),
        )
        state["master_audio"] = str(master_audio)
        save_state(state_file, state)
    state["outputs"] = output_files.copy_subtitle_only_outputs(
        paths,
        project_json,
        folder,
        str(state["mode"]),
        original_media=Path(str(state.get("original_media") or "")),
        master_audio=master_audio,
        background=(Path(str(state["background"])) if state.get("background") else None),
        harmonized_delay_seconds=int(state["harmonized_delay_seconds"]),
        harmonized_volume_db=float(state["harmonized_volume_db"]),
        embed_subtitles=bool(state.get("embed_subtitles", True))
        and not bool(state.get("original_hard_subtitles", False)),
    )


def _copy_final_stage(
    paths: ToolPaths, folder: Path, state: dict[str, Any], project_json: Path
) -> None:
    if state.get("variant_outputs"):
        state["outputs"] = dict(state["variant_outputs"])
        original_media = state.get("original_media") or state.get("original_video")
        if original_media:
            state["outputs"].setdefault("original", str(original_media))
    else:
        state["outputs"] = output_files.copy_final_outputs(
            paths,
            project_json,
            folder,
            str(state["mode"]),
            harmonized_delay_seconds=int(state["harmonized_delay_seconds"]),
            harmonized_volume_db=float(state["harmonized_volume_db"]),
            embed_subtitles=bool(state.get("embed_subtitles", True)),
        )


def _render_original_subtitle_video(
    paths: ToolPaths,
    folder: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
    subtitles_only: bool,
) -> None:
    master_audio = Path(str(state.get("master_audio") or ""))
    if not master_audio.is_file():
        master_audio, _timeline = media_audio.normalize_and_concat(
            paths, sources, Path(str(state["workspace"]))
        )
        state["master_audio"] = str(master_audio)
    destination = folder / "原声字幕版.mp4"
    media_video.render_static_bilingual_video(
        paths,
        master_audio,
        Path(str(state["background"])) if state.get("background") else None,
        Path(state["outputs"]["srt"]),
        destination,
        lead_seconds=int(state["harmonized_delay_seconds"])
        if normalize_mode(state["mode"]) == MODE_VIDEO_HARMONIZED
        else 0,
        volume_db=float(state["harmonized_volume_db"])
        if normalize_mode(state["mode"]) == MODE_VIDEO_HARMONIZED
        else 0.0,
        require_hard_subtitles=True,
    )
    state["outputs"]["original_subtitle_video"] = str(destination)
    if subtitles_only:
        state["outputs"]["video"] = str(destination)


def _outputs_stage(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
    project_json: Path,
    subtitles_only: bool,
) -> None:
    if subtitles_only:
        _copy_subtitle_only_stage(paths, folder, state_file, state, sources, project_json)
    else:
        _copy_final_stage(paths, folder, state, project_json)
    if (
        bool(state.get("original_hard_subtitles", False))
        and normalize_mode(state["mode"]) != MODE_AUDIO
    ):
        _render_original_subtitle_video(paths, folder, state, sources, subtitles_only)
    state["status"] = "outputs_ready"
    save_state(state_file, state)


def _finish_stage(paths: ToolPaths, folder: Path, state_file: Path, state: dict[str, Any]) -> None:
    translations = state.get("title_translations") or {}
    missing_titles = [
        str(item.get("relative_path") or item["filename"])
        for item in state.get("timeline") or []
        if not str(translations.get(str(item.get("relative_path") or item["filename"]), "")).strip()
    ]
    if missing_titles or not str(state.get("folder_name_translation") or "").strip():
        runlog.emit("\n  使用 ASMR Dubber 当前翻译服务处理作品名和曲目标题……")
        try:
            state["title_translations"] = title_translation.translate_titles(state, paths)
        except VideoPreparerError:
            save_state(state_file, state)
            raise
    save_state(state_file, state)
    timestamp_file = output_files.write_timestamp_document(state, folder)
    state.setdefault("outputs", {})["timestamps"] = str(timestamp_file)
    state["timestamp_schema"] = TIMESTAMP_SCHEMA
    state["status"] = "completed"
    save_state(state_file, state)


def _report_completion(
    folder: Path, state: dict[str, Any], project_json: Path, subtitles_only: bool
) -> None:
    runlog.emit("\n全部完成。文件已放回：")
    runlog.emit(f"  {folder}")
    if subtitles_only:
        if normalize_mode(state["mode"]) != MODE_AUDIO and state["outputs"].get("video"):
            runlog.emit("  - 原声字幕版.mp4")
    elif normalize_mode(state["mode"]) == MODE_AUDIO:
        runlog.emit("  - 原声.flac")
        runlog.emit(f"  - {Path(state['outputs']['audio']).name}")
    else:
        runlog.emit("  - 原声.mp4")
        runlog.emit("  - 双语版.mp4")
    runlog.emit("  - 双语版.srt")
    runlog.emit("  - 双语版.lrc")
    runlog.emit("  - 时间戳.txt")
    if project_json.is_file():
        runlog.emit(f"\nASMR Dubber 工作项目保留在：\n  {project_json.parent}")
