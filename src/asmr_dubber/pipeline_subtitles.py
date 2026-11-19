"""Subtitle generation, replacement dubbing duration inspection, and transcript export."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .audio import replacement_obstacles
from .lifecycle import project_operation
from .models import DubProject
from .subtitles import SubtitleLanguage
from .task_control import CancellationSignal

Progress = Callable[[str, int, int], None]

_SPEECH_DURATION_CACHE: dict[tuple[str, int, int, int], float] = {}
_MAX_SPEECH_DURATION_CACHE = 4096


def _cached_speech_duration(
    audio_path: Path, speech_bounds_fn: Callable[[Path], tuple[float, float]]
) -> float:
    stat = audio_path.stat()
    key = (str(audio_path), stat.st_size, stat.st_mtime_ns, id(speech_bounds_fn))
    cached = _SPEECH_DURATION_CACHE.get(key)
    if cached is not None:
        return cached
    lo, hi = speech_bounds_fn(audio_path)
    duration = hi - lo
    if len(_SPEECH_DURATION_CACHE) >= _MAX_SPEECH_DURATION_CACHE:
        _SPEECH_DURATION_CACHE.pop(next(iter(_SPEECH_DURATION_CACHE)))
    _SPEECH_DURATION_CACHE[key] = duration
    return duration


def _replacement_dubbing_durations(project: DubProject, project_dir: Path) -> dict[str, float]:
    from . import pipeline

    durations: dict[str, float] = {}
    for sentence in project.sentences:
        if (
            not sentence.enabled
            or not sentence.chinese_audio_enabled
            or not pipeline.has_speakable_text(sentence.zh_text)
        ):
            continue
        if not sentence.tts_file or sentence.tts_cache_key != pipeline.tts_cache_key(
            project, sentence
        ):
            continue
        try:
            audio_path = pipeline.resolve_project_path(
                project_dir,
                sentence.tts_file,
                f"句子 {sentence.id} 的中文音频",
            )
            if audio_path.is_file():
                durations[sentence.id] = _cached_speech_duration(audio_path, pipeline.speech_bounds)
                continue
        except Exception:
            pass
    return durations


@project_operation
def generate_subtitles(
    project: DubProject,
    project_dir: Path,
    language: SubtitleLanguage | Literal["ja"] = "bilingual",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    output_variant: str | None = None,
) -> tuple[Path, Path, Path | None]:
    """Create external subtitles and, for video projects, a subtitled video."""
    from . import pipeline
    from .stem_cache import variant_directory

    pipeline.require_supported_platform()
    pipeline.check_cancelled(cancel_event)
    normalized_language: SubtitleLanguage = "source" if language == "ja" else language
    output_dir = variant_directory(project_dir, output_variant)
    source = pipeline.verify_source(project_dir, project.source)

    if progress:
        progress("生成 SRT 与 LRC 字幕", 0, 2 if project.source.media_type == "video" else 1)
    media_duration = project.source.duration_seconds
    if project.settings.subtitle_timeline == "dubbing":
        from .audio import probe_audio

        audio_ref = project.output_file or project.chinese_stem_file
        if audio_ref:
            mixed = pipeline.resolve_project_path(project_dir, audio_ref, "完成音频")
            if mixed.is_file():
                media_duration = probe_audio(mixed).duration_seconds
    replacing = (
        output_variant == "replace"
        if output_variant in {"bilingual", "replace"}
        else (
            project.settings.separation_enabled
            and project.settings.separation_mix_mode == "replace"
        )
    )
    dubbing_durations = (
        pipeline._replacement_dubbing_durations(project, project_dir)
        if replacing and project.settings.subtitle_timeline == "dubbing"
        else None
    )
    subtitle_warnings: list[str] = []
    srt, lrc = pipeline.write_subtitle_files(
        project.sentences,
        project_dir / "subtitles" / output_variant if output_variant else project_dir / "subtitles",
        normalized_language,
        timeline=project.settings.subtitle_timeline,
        maximum_chars=project.settings.subtitle_max_chars_per_line,
        minimum_duration=project.settings.subtitle_min_duration_seconds,
        maximum_cps=project.settings.subtitle_max_cps,
        target_language=project.settings.tts_target_language,
        chinese_dubbing_offset_ms=(
            project.settings.replacement_dubbing_offset_ms
            if replacing
            else project.settings.chinese_dubbing_offset_ms
        ),
        chinese_max_auto_speed=(
            project.settings.replacement_max_auto_speed
            if replacing
            else project.settings.chinese_max_auto_speed
        ),
        chinese_dubbing_timing_mode=(
            project.settings.replacement_timing_mode
            if replacing
            else project.settings.chinese_dubbing_timing_mode
        ),
        dubbing_durations=dubbing_durations,
        media_duration_seconds=media_duration,
        warnings=subtitle_warnings,
        # Same plan as the fully dubbed mix: kept original sounds and the recording's end.
        dubbing_obstacles=(
            replacement_obstacles(project.sentences, dubbing_durations)
            if dubbing_durations is not None
            else None
        ),
        dubbing_end_seconds=(
            project.source.duration_seconds if dubbing_durations is not None else None
        ),
    )
    warning_summary = ""
    if subtitle_warnings:
        fallback_count = sum("缺少译文" in item for item in subtitle_warnings)
        duration_count = len(subtitle_warnings) - fallback_count
        parts = []
        if fallback_count:
            parts.append(f"{fallback_count} 句缺少译文已回退原文")
        if duration_count:
            parts.append(f"{duration_count} 句字幕可读时长受下一句或媒体末尾限制")
        warning_summary = "；".join(parts)
        if progress and warning_summary:
            progress(warning_summary, 1, 2)
    pipeline.check_cancelled(cancel_event)

    video_output: Path | None = None
    if project.source.media_type == "video":
        mixed_audio: Path | None = None
        if project.output_file:
            candidate = pipeline.resolve_project_path(project_dir, project.output_file, "完成音频")
            if candidate.is_file():
                mixed_audio = candidate
        if progress:
            progress("生成带字幕视频", 1, 2)
        video_output = pipeline.render_subtitled_video(
            source,
            srt,
            output_dir
            / pipeline.subtitle_video_filename(
                project_dir,
                normalized_language,
                mixed=mixed_audio is not None,
            ),
            replacement_audio=mixed_audio,
            subtitle_language=normalized_language,
            source_language=project.source_language,
            target_language=project.settings.tts_target_language,
        )
        pipeline.check_cancelled(cancel_event)
    # Commit metadata only after every requested artifact succeeds. Cancellation
    # or FFmpeg failure must leave the previous downloadable outputs visible.
    project.subtitle_language = normalized_language
    project.subtitle_srt_file = srt.relative_to(project_dir).as_posix()
    project.subtitle_lrc_file = lrc.relative_to(project_dir).as_posix()
    project.subtitle_video_file = (
        video_output.relative_to(project_dir).as_posix() if video_output is not None else None
    )
    pipeline.save_project(project, project_dir)
    if progress:
        final_message = f"字幕生成完成（{warning_summary}）" if warning_summary else "字幕生成完成"
        progress(final_message, 1, 1)
    return srt, lrc, video_output


def export_transcript(project: DubProject, project_dir: Path) -> None:
    from . import pipeline

    exports = project_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    replacing = (
        project.settings.separation_enabled and project.settings.separation_mix_mode == "replace"
    )
    offset_ms = (
        project.settings.replacement_dubbing_offset_ms
        if replacing
        else project.settings.chinese_dubbing_offset_ms
    )
    max_auto_speed = (
        project.settings.replacement_max_auto_speed
        if replacing
        else project.settings.chinese_max_auto_speed
    )
    timing_mode = (
        project.settings.replacement_timing_mode
        if replacing
        else project.settings.chinese_dubbing_timing_mode
    )
    durations = pipeline._replacement_dubbing_durations(project, project_dir) if replacing else None
    timing_by_id = {
        timing.sentence_id: timing
        for timing in pipeline.plan_dubbing_timing(
            project.sentences,
            offset_ms=offset_ms,
            max_auto_speed=max_auto_speed,
            durations=durations,
            mode=timing_mode,
            obstacles=(
                replacement_obstacles(project.sentences, durations)
                if durations is not None
                else None
            ),
            end_seconds=project.source.duration_seconds if durations is not None else None,
        )
    }
    rows = [
        {
            "id": sentence.id,
            "speaker_id": sentence.speaker_id,
            "enabled": sentence.enabled,
            "source_language": project.source_language,
            "source_start_seconds": sentence.start_seconds,
            "source_end_seconds": sentence.end_seconds,
            "zh_start_seconds": (
                timing_by_id[sentence.id].start_seconds
                if sentence.id in timing_by_id
                else pipeline.dubbing_start_seconds(
                    sentence,
                    offset_ms,
                )
            ),
            "tts_duration_seconds": sentence.tts_duration_seconds,
            "auto_speed_factor": (
                timing_by_id[sentence.id].speed_factor if sentence.id in timing_by_id else 1.0
            ),
            "zh_effective_duration_seconds": (
                timing_by_id[sentence.id].effective_duration_seconds
                if sentence.id in timing_by_id
                else sentence.tts_duration_seconds
            ),
            "remaining_overlap_seconds": (
                timing_by_id[sentence.id].remaining_overlap_seconds
                if sentence.id in timing_by_id
                else 0.0
            ),
            "source": sentence.source_text,
            "zh": sentence.zh_text,
            "status": sentence.status,
            "error": sentence.error,
        }
        for sentence in project.sentences
    ]
    (exports / "transcript.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (exports / "transcript.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
