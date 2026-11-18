"""Final output, subtitle and timestamp file production."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import media_video, runlog
from .constants import DEFAULT_TIMESTAMP_FOOTER, MODE_AUDIO, MODE_VIDEO_HARMONIZED, SAMPLE_RATE
from .domain import (
    ToolPaths,
)
from .errors import VideoPreparerError
from .fileio import atomic_copy, atomic_write_text, bracketed
from .media_video import remux_video_without_subtitles, render_delayed_existing_video
from .modes import normalize_mode
from .project_cli import project_asset, read_project
from .subtitle_files import shift_lrc_text, shift_srt_text


def copy_subtitles(
    project_json: Path,
    folder: Path,
    mode: str,
    *,
    harmonized_delay_seconds: int,
) -> tuple[Path, Path]:
    project = read_project(project_json)
    srt_source = project_asset(project_json, project.get("subtitle_srt_file"))
    lrc_source = project_asset(project_json, project.get("subtitle_lrc_file"))
    if srt_source is None or lrc_source is None:
        raise VideoPreparerError("ASMR Dubber 没有生成完整的 SRT/LRC 字幕。")

    srt_destination = folder / "双语版.srt"
    lrc_destination = folder / "双语版.lrc"
    if normalize_mode(mode) == MODE_VIDEO_HARMONIZED:
        offset_ms = harmonized_delay_seconds * 1000
        srt_text = srt_source.read_text(encoding="utf-8-sig")
        lrc_text = lrc_source.read_text(encoding="utf-8-sig")
        atomic_write_text(srt_destination, shift_srt_text(srt_text, offset_ms))
        atomic_write_text(lrc_destination, shift_lrc_text(lrc_text, offset_ms))
    else:
        atomic_copy(srt_source, srt_destination)
        atomic_copy(lrc_source, lrc_destination)
    return srt_destination, lrc_destination


def copy_final_outputs(
    paths: ToolPaths,
    project_json: Path,
    folder: Path,
    mode: str,
    *,
    harmonized_delay_seconds: int,
    harmonized_volume_db: float,
    embed_subtitles: bool,
    output_stem: str = "双语版",
) -> dict[str, str]:
    project = read_project(project_json)
    mode = normalize_mode(mode)
    if mode == MODE_AUDIO:
        audio_source = project_asset(project_json, project.get("output_file"))
        if audio_source is None:
            raise VideoPreparerError("ASMR Dubber 没有生成可用的双语音频。")
        srt, lrc = copy_subtitles(
            project_json,
            folder,
            mode,
            harmonized_delay_seconds=harmonized_delay_seconds,
        )
        audio_suffix = audio_source.suffix if audio_source.suffix else ".wav"
        audio_destination = folder / f"{output_stem}{audio_suffix}"
        runlog.emit("  正在把双语音频送回原文件夹……")
        atomic_copy(audio_source, audio_destination)
        return {
            "audio": str(audio_destination),
            "srt": str(srt),
            "lrc": str(lrc),
        }

    mixed_audio_source: Path | None = None
    if mode == MODE_VIDEO_HARMONIZED:
        # Do not freeze the first frame of a hard-subtitled video for 20 minutes.
        # Delay the clean mixed video, then attach the already shifted subtitle.
        video_source = project_asset(project_json, project.get("output_video_file"))
        if video_source is None and embed_subtitles:
            video_source = project_asset(project_json, project.get("subtitle_video_file"))
        # Use ASMR Dubber's lossless mixed WAV as the final audio source. The
        # delayed harmonious version is then encoded to AAC only once.
        mixed_audio_source = project_asset(project_json, project.get("output_file"))
    elif embed_subtitles:
        video_source = project_asset(project_json, project.get("subtitle_video_file"))
        if video_source is None:
            video_source = project_asset(project_json, project.get("output_video_file"))
    else:
        video_source = project_asset(project_json, project.get("output_video_file"))
    if video_source is None:
        if not embed_subtitles:
            raise VideoPreparerError("ASMR Dubber 没有生成可用的无字幕双语视频。")
        raise VideoPreparerError("ASMR Dubber 没有生成可用的双语视频。")

    srt, lrc = copy_subtitles(
        project_json,
        folder,
        mode,
        harmonized_delay_seconds=harmonized_delay_seconds,
    )
    video_destination = folder / f"{output_stem}.mp4"
    runlog.emit("  正在把双语视频送回原文件夹……")
    if mode == MODE_VIDEO_HARMONIZED:
        render_delayed_existing_video(
            paths,
            video_source,
            video_destination,
            lead_seconds=harmonized_delay_seconds,
            subtitle_file=srt if embed_subtitles else None,
            volume_db=harmonized_volume_db,
            audio_source=mixed_audio_source,
        )
    else:
        if not embed_subtitles:
            remux_video_without_subtitles(paths, video_source, video_destination)
        elif video_source.suffix.casefold() == ".mp4":
            atomic_copy(video_source, video_destination)
        else:
            media_video.remux_video_with_subtitle(paths, video_source, srt, video_destination)
    return {
        "video": str(video_destination),
        "srt": str(srt),
        "lrc": str(lrc),
    }


def copy_subtitle_only_outputs(
    paths: ToolPaths,
    project_json: Path,
    folder: Path,
    mode: str,
    *,
    original_media: Path,
    master_audio: Path,
    background: Path | None,
    harmonized_delay_seconds: int,
    harmonized_volume_db: float,
    embed_subtitles: bool,
) -> dict[str, str]:
    """Publish subtitles without creating a synthesized or mixed track."""

    if not original_media.is_file():
        raise VideoPreparerError("找不到已经生成的原声成品。")
    srt, lrc = copy_subtitles(
        project_json,
        folder,
        mode,
        harmonized_delay_seconds=harmonized_delay_seconds,
    )
    outputs = {
        "original": str(original_media),
        "srt": str(srt),
        "lrc": str(lrc),
    }
    normalized_mode = normalize_mode(mode)
    if normalized_mode != MODE_AUDIO and embed_subtitles:
        destination = folder / "原声字幕版.mp4"
        lead = harmonized_delay_seconds if normalized_mode == MODE_VIDEO_HARMONIZED else 0
        volume = harmonized_volume_db if normalized_mode == MODE_VIDEO_HARMONIZED else 0.0
        runlog.emit("  正在生成带字幕的原声视频……")
        media_video.render_static_bilingual_video(
            paths,
            master_audio,
            background,
            srt,
            destination,
            lead_seconds=lead,
            volume_db=volume,
        )
        outputs["video"] = str(destination)
    return outputs


def format_timestamp_from_samples(samples: int) -> str:
    # 向下取整，保证显示时间不会晚于音频真正开始的时刻。
    seconds = max(0, samples // SAMPLE_RATE)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_timestamp_document(state: dict[str, Any], folder: Path) -> Path:
    translations = state.get("title_translations") or {}
    folder_name_original = str(
        state.get("folder_name_original") or Path(state["source_folder"]).name
    ).strip()
    folder_name_translation = str(state.get("folder_name_translation") or "").strip()
    if not folder_name_translation:
        raise VideoPreparerError("缺少文件夹名称的中文翻译。")
    offset_samples = (
        int(state.get("harmonized_delay_seconds") or 0) * SAMPLE_RATE
        if normalize_mode(state["mode"]) == MODE_VIDEO_HARMONIZED
        else 0
    )
    lines: list[str] = [
        f"中文名称：{folder_name_translation}",
        f"原始名称：{folder_name_original}",
        "",
    ]
    for item in state["timeline"]:
        filename = str(item.get("relative_path") or item["filename"])
        chinese = str(translations.get(filename, "")).strip()
        if not chinese:
            raise VideoPreparerError(f"缺少中文标题：{filename}")
        start = int(item["start_samples"]) + offset_samples
        lines.append(f"{format_timestamp_from_samples(start)} {bracketed(chinese)}")
        lines.append(bracketed(str(item["title_ja"])))
        lines.append("")
    stored_footer = state.get("timestamp_footer")
    footer = DEFAULT_TIMESTAMP_FOOTER if stored_footer is None else str(stored_footer).strip()
    if footer:
        if state.get("timestamp_footer_position", "after") == "before":
            lines[3:3] = [footer, ""]
        else:
            lines.append(footer)
    destination = folder / "时间戳.txt"
    atomic_write_text(destination, "\n".join(lines).rstrip() + "\n")
    return destination
