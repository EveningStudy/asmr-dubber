"""Merged multi-track outputs and the smart summary."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import constants, media_audio, media_video, output_files
from .constants import LAYOUT_BOTH, LAYOUT_SEPARATE, MODE_AUDIO, MODE_VIDEO_HARMONIZED, SAMPLE_RATE
from .domain import (
    AppConfig,
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .fileio import atomic_copy, atomic_write_text, bracketed
from .media_audio import concat_audio_files
from .modes import layout_label, mode_label, normalize_mode
from .output_files import format_timestamp_from_samples
from .plan_files import _single_track_source
from .project_cli import project_asset, read_project
from .subtitle_files import combine_lrc_files, combine_srt_files
from .work_state import safe_reset_workspace, task_key


def _state_master_audio(
    paths: ToolPaths,
    state: dict[str, Any],
    source: AudioSource,
    workspace: Path,
) -> Path:
    candidate = Path(str(state.get("master_audio") or "")).expanduser()
    if candidate.is_file():
        return candidate
    # A user may have removed .work after a successful run. Recreate only the
    # normalized single-track master needed for a later merge; ASR/TTS remain
    # untouched.
    rebuilt_workspace = workspace / f"rebuild-{source.order:04d}"
    rebuilt_workspace.mkdir(parents=True, exist_ok=True)
    rebuilt, _ = media_audio.normalize_and_concat(
        paths, [_single_track_source(source)], rebuilt_workspace
    )
    return rebuilt


def _state_mixed_audio(
    state: dict[str, Any],
    project_json: Path | None = None,
) -> Path:
    candidate = Path(str(state.get("project_mixed_audio") or "")).expanduser()
    if candidate.is_file():
        return candidate
    if project_json is not None and project_json.is_file():
        project = read_project(project_json)
        candidate = project_asset(project_json, project.get("output_file"))
        if candidate is not None:
            return candidate
    outputs = state.get("outputs") or {}
    candidate = Path(str(outputs.get("audio") or "")).expanduser()
    if candidate.is_file():
        return candidate
    raise VideoPreparerError("ASMR Dubber 已完成，但找不到可合并的双语音频。")


def _timeline_from_states(states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    offset = 0
    for state in states:
        for item in state.get("timeline") or []:
            copied = dict(item)
            copied["start_samples"] = int(item.get("start_samples") or 0) + offset
            relative = str(item.get("relative_path") or item.get("filename") or "")
            copied["relative_path"] = relative
            merged.append(copied)
        offset += sum(
            int(item.get("duration_samples") or 0) for item in state.get("timeline") or []
        )
    return merged


def _subtitle_entries_from_states(
    states: Sequence[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int]], int]:
    srt_entries: list[tuple[Path, int, int]] = []
    lrc_entries: list[tuple[Path, int]] = []
    offset_samples = 0
    delay_samples = (
        int(states[0].get("harmonized_delay_seconds") or 0) * SAMPLE_RATE
        if normalize_mode(mode) == MODE_VIDEO_HARMONIZED
        else 0
    )
    for state in states:
        outputs = state.get("outputs") or {}
        srt = Path(str(outputs.get("srt") or "")).expanduser()
        lrc = Path(str(outputs.get("lrc") or "")).expanduser()
        # Per-track harmonious outputs already contain their own delay. Remove
        # it here; the merged product receives exactly one delay at the end.
        per_track_delay = (
            delay_samples if normalize_mode(state["mode"]) == MODE_VIDEO_HARMONIZED else 0
        )
        shift = offset_samples - per_track_delay
        duration_samples = sum(
            int(item.get("duration_samples") or 0) for item in state.get("timeline") or []
        )
        srt_entries.append(
            (
                srt,
                shift * 1000 // SAMPLE_RATE,
                (offset_samples + duration_samples) * 1000 // SAMPLE_RATE,
            )
        )
        lrc_entries.append((lrc, shift * 1000 // SAMPLE_RATE))
        offset_samples += duration_samples
    return srt_entries, lrc_entries, delay_samples * 1000 // SAMPLE_RATE


def build_merged_outputs(paths: ToolPaths, config: AppConfig, **kwargs: Any) -> dict[str, str]:
    variants = config.output_policy.get("variants", [])
    if not variants or kwargs.get("subtitles_only"):
        return _build_merged_outputs(paths, config, **kwargs)
    result: dict[str, str] = {}
    for variant in variants:
        selected = dict(kwargs)
        selected["output_folder"] = kwargs["output_folder"] / (
            "双语版" if variant == "bilingual" else "替换配音版"
        )
        selected["output_stem"] = "双语版" if variant == "bilingual" else "配音版"
        selected["states"] = [
            {
                **state,
                "project_mixed_audio": state["mix_variants"][variant]["audio"],
                "outputs": state["mix_variants"][variant]["outputs"],
            }
            for state in kwargs["states"]
        ]
        outputs = _build_merged_outputs(paths, replace(config, output_policy={}), **selected)
        if not result:
            result.update(outputs)
        result.update({f"{variant}_{key}": value for key, value in outputs.items()})
    return result


def _build_merged_outputs(
    paths: ToolPaths,
    config: AppConfig,
    *,
    source_folder: Path,
    output_folder: Path,
    mode: str,
    sources: Sequence[AudioSource],
    states: Sequence[dict[str, Any]],
    background: Path | None,
    folder_translation: str,
    title_translations: dict[str, str],
    embed_subtitles: bool,
    subtitles_only: bool = False,
    output_stem: str = "双语版",
) -> dict[str, str]:
    """Build the merged product from completed per-track projects."""

    mode = normalize_mode(mode)
    output_folder.mkdir(parents=True, exist_ok=True)
    work = constants.WORK_ROOT / f"merge-{task_key(output_folder)}"
    safe_reset_workspace(work)
    expected = [
        sum(int(item.get("duration_samples") or 0) for item in state.get("timeline") or [])
        for state in states
    ]
    masters = [
        _state_master_audio(paths, state, source, work)
        for source, state in zip(sources, states, strict=True)
    ]
    original_master, _ = concat_audio_files(
        paths,
        masters,
        work,
        expected_samples=expected,
        name="original",
    )
    mixed_master: Path | None = None
    if not subtitles_only:
        mixed_files = [
            _state_mixed_audio(state, Path(str(state.get("project_json") or "")))
            for state in states
        ]
        mixed_master, _ = concat_audio_files(
            paths,
            mixed_files,
            work,
            expected_samples=expected,
            name="mixed",
        )

    timeline = _timeline_from_states(states)
    merged_state: dict[str, Any] = {
        "source_folder": str(source_folder),
        "mode": mode,
        "harmonized_delay_seconds": int(config.harmonized_delay_seconds),
        "timestamp_footer": config.timestamp_footer,
        "timestamp_footer_position": config.timestamp_footer_position,
        "original_hard_subtitles": config.original_hard_subtitles,
        "timeline": timeline,
        "title_translations": dict(title_translations),
        "folder_name_original": source_folder.name,
        "folder_name_translation": folder_translation,
    }
    srt_entries, lrc_entries, final_offset_ms = _subtitle_entries_from_states(
        states,
        mode=mode,
    )
    srt = combine_srt_files(
        srt_entries,
        output_folder / f"{output_stem}.srt",
        final_offset_ms=final_offset_ms,
    )
    lrc = combine_lrc_files(
        lrc_entries,
        output_folder / f"{output_stem}.lrc",
        final_offset_ms=final_offset_ms,
    )

    if mode == MODE_AUDIO:
        original_destination = output_folder / "原声.flac"
        atomic_copy(original_master, original_destination)
        outputs = {
            "original": str(original_destination),
            "srt": str(srt),
            "lrc": str(lrc),
        }
        if not subtitles_only:
            if mixed_master is None:
                raise VideoPreparerError("找不到可合并的双语音频。")
            mixed_destination = output_folder / f"{output_stem}.flac"
            atomic_copy(mixed_master, mixed_destination)
            outputs["audio"] = str(mixed_destination)
    else:
        original_destination = output_folder / "原声.mp4"
        lead = int(config.harmonized_delay_seconds) if mode == MODE_VIDEO_HARMONIZED else 0
        volume = config.harmonized_volume_db if mode == MODE_VIDEO_HARMONIZED else 0.0
        media_video.render_static_video(
            paths,
            original_master,
            background,
            original_destination,
            lead_seconds=lead,
            volume_db=volume,
        )
        outputs = {
            "original": str(original_destination),
            "srt": str(srt),
            "lrc": str(lrc),
        }
        if (subtitles_only and embed_subtitles) or config.original_hard_subtitles:
            subtitle_destination = output_folder / "原声字幕版.mp4"
            media_video.render_static_bilingual_video(
                paths,
                original_master,
                background,
                srt,
                subtitle_destination,
                require_hard_subtitles=config.original_hard_subtitles,
                lead_seconds=lead,
                volume_db=volume,
            )
            outputs["original_subtitle_video"] = str(subtitle_destination)
            if subtitles_only:
                outputs["video"] = str(subtitle_destination)
        if not subtitles_only:
            if mixed_master is None:
                raise VideoPreparerError("找不到可合并的双语音频。")
            bilingual_destination = output_folder / f"{output_stem}.mp4"
            if embed_subtitles:
                media_video.render_static_bilingual_video(
                    paths,
                    mixed_master,
                    background,
                    srt,
                    bilingual_destination,
                    lead_seconds=lead,
                    volume_db=volume,
                )
            else:
                media_video.render_static_video(
                    paths,
                    mixed_master,
                    background,
                    bilingual_destination,
                    lead_seconds=lead,
                    volume_db=volume,
                )
            outputs["video"] = str(bilingual_destination)
    timestamp = output_files.write_timestamp_document(merged_state, output_folder)
    outputs["timestamps"] = str(timestamp)
    shutil.rmtree(work, ignore_errors=True)
    return outputs


def write_smart_summary(
    output_root: Path,
    *,
    source_folder: Path,
    mode: str,
    layout: str,
    sources: Sequence[AudioSource],
    states: Sequence[dict[str, Any]],
    descriptors: Sequence[dict[str, Any]],
    folder_translation: str,
    title_translations: dict[str, str],
    footer: str,
    harmonized_delay_seconds: int,
    footer_position: str = "after",
) -> tuple[Path, Path]:
    """Write a human-readable track index and a timeline reference."""

    output_root.mkdir(parents=True, exist_ok=True)
    descriptor_by_id = {str(item["job_id"]): item for item in descriptors}
    lines = [
        f"中文名称：{folder_translation}",
        f"原始名称：{source_folder.name}",
        f"模式：{mode_label(mode)}",
        f"成品组织：{layout_label(layout)}",
        "",
        "曲目与输出：",
    ]
    timeline_lines = [
        f"中文名称：{folder_translation}",
        f"原始名称：{source_folder.name}",
        "",
    ]
    cumulative = 0
    timeline_offset = (
        harmonized_delay_seconds * SAMPLE_RATE
        if normalize_mode(mode) == MODE_VIDEO_HARMONIZED
        else 0
    )
    for index, (source, state) in enumerate(zip(sources, states, strict=True), start=1):
        key = source.relative_path or source.path.name
        translated = str(title_translations.get(key) or "未翻译").strip()
        descriptor = descriptor_by_id.get(f"track-{index:04d}")
        track_output = ""
        if descriptor is not None:
            try:
                track_output = Path(str(descriptor["output"])).relative_to(output_root).as_posix()
            except ValueError:
                track_output = str(descriptor["output"])
        duration = sum(
            int(item.get("duration_samples") or 0) for item in state.get("timeline") or []
        )
        lines.extend(
            (
                f"{index:03d}. {source.title_ja}",
                f"     中文：{translated}",
                f"     原文件：{source.relative_path or source.path.name}",
                f"     输出：{track_output or '仅合并版'}",
                "",
            )
        )
        timeline_lines.append(
            f"{format_timestamp_from_samples(cumulative + timeline_offset)} {bracketed(translated)}"
        )
        timeline_lines.extend((bracketed(source.title_ja), ""))
        cumulative += duration

    if layout in {LAYOUT_SEPARATE, LAYOUT_BOTH}:
        lines.extend(
            (
                "说明：分轨文件各自从 00:00 开始；上面的时间轴按原曲目顺序计算，",
                "仅用于查找，不代表把分轨文件直接无缝播放时的播放器时间。",
                "",
            )
        )
    if footer.strip():
        if footer_position == "before":
            lines[5:5] = [footer.strip(), ""]
            timeline_lines[3:3] = [footer.strip(), ""]
        else:
            lines.append(footer.strip())
            timeline_lines.append(footer.strip())
    index_file = output_root / "曲目清单.txt"
    timeline_file = output_root / "总时间戳.txt"
    atomic_write_text(index_file, "\n".join(lines).rstrip() + "\n")
    atomic_write_text(timeline_file, "\n".join(timeline_lines).rstrip() + "\n")
    return index_file, timeline_file
