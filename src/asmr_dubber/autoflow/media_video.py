"""Static video rendering, remuxing and subtitle burning."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from . import constants, processes, runlog
from .constants import SAMPLE_RATE, VIDEO_FPS, VIDEO_SIZE
from .domain import (
    ToolPaths,
)
from .errors import VideoPreparerError
from .fileio import partial_output_path
from .processes import audio_duration_samples, media_duration_seconds


def background_input(background: Path | None, duration_seconds: str) -> list[str]:
    if background is not None:
        return ["-i", str(background)]
    return [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={VIDEO_SIZE}:r=1:d={duration_seconds}",
    ]


def render_static_video(
    paths: ToolPaths,
    audio_source: Path,
    background: Path | None,
    destination: Path,
    *,
    lead_seconds: int = 0,
    volume_db: float = 0.0,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_output_path(destination)
    total_samples = audio_duration_samples(paths, audio_source) + lead_seconds * SAMPLE_RATE
    duration_text = f"{total_samples / SAMPLE_RATE:.6f}"
    audio_filters = [
        f"aresample={SAMPLE_RATE}:async=0",
        f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
    ]
    if volume_db:
        audio_filters.append(f"volume={volume_db:g}dB")
    if lead_seconds:
        audio_filters.append(f"adelay={lead_seconds * 1000}:all=1")
    audio_filters.append("asetpts=N/SR/TB")

    # Scale the source picture once, then loop that prepared 1080p frame in
    # memory. This avoids decoding an 8K JPEG again for every video frame.
    visual_filters = (
        f"scale={constants.VIDEO_FILTER_SIZE}:force_original_aspect_ratio=decrease:"
        f"flags=lanczos,pad={constants.VIDEO_FILTER_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,format=yuv420p,"
    )
    if background is None:
        # The lavfi color source already has the exact target duration.  The
        # single-frame loop used for still images truncates this generated
        # stream on Linux FFmpeg builds.
        visual_filters += f"fps={VIDEO_FPS},setpts=PTS-STARTPTS,trim=duration={duration_text}"
    else:
        visual_filters += (
            f"loop=loop=-1:size=1:start=0,setpts=N/{VIDEO_FPS}/TB,trim=duration={duration_text}"
        )
    video_filter = f"[0:v:0]{visual_filters}[v];[1:a:0]{','.join(audio_filters)}[a]"
    try:
        processes.run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-stats",
                "-stats_period",
                "5",
                *background_input(background, duration_text),
                "-i",
                str(audio_source),
                "-filter_complex",
                video_filter,
                "-map",
                "[v]",
                "-map",
                "[a]",
                *paths.video_encoder_options,
                "-r",
                str(VIDEO_FPS),
                "-fps_mode",
                "cfr",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-t",
                duration_text,
                "-shortest",
                "-movflags",
                "+faststart",
                "-metadata",
                f"title={destination.stem}",
                str(partial),
            ],
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def render_delayed_existing_video(
    paths: ToolPaths,
    source: Path,
    destination: Path,
    *,
    lead_seconds: int,
    subtitle_file: Path | None,
    volume_db: float = 0.0,
    audio_source: Path | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_output_path(destination)
    target_duration = (
        max(
            media_duration_seconds(paths, source),
            media_duration_seconds(paths, audio_source) if audio_source is not None else 0.0,
        )
        + lead_seconds
    )
    audio_filters = [
        f"aresample={SAMPLE_RATE}:async=0",
        f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
    ]
    if volume_db:
        audio_filters.append(f"volume={volume_db:g}dB")
    audio_filters.extend((f"adelay={lead_seconds * 1000}:all=1", "asetpts=N/SR/TB"))
    audio_input_index = 1 if audio_source is not None else 0
    filter_complex = (
        f"[0:v:0]tpad=start_mode=clone:start_duration={lead_seconds},"
        f"scale={constants.VIDEO_FILTER_SIZE}:force_original_aspect_ratio=decrease,"
        f"pad={constants.VIDEO_FILTER_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={VIDEO_FPS},format=yuv420p,setpts=PTS-STARTPTS[v];"
        f"[{audio_input_index}:a:0]{','.join(audio_filters)}[a]"
    )
    input_arguments = ["-i", str(source)]
    if audio_source is not None:
        input_arguments.extend(("-i", str(audio_source)))
    subtitle_input_index: int | None = None
    if subtitle_file is not None:
        subtitle_input_index = 2 if audio_source is not None else 1
        input_arguments.extend(("-f", "srt", "-i", str(subtitle_file)))
    subtitle_arguments = (
        [
            "-map",
            f"{subtitle_input_index}:s:0",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=und",
        ]
        if subtitle_input_index is not None
        else ["-sn"]
    )
    try:
        processes.run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-stats",
                "-stats_period",
                "5",
                *input_arguments,
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[a]",
                *subtitle_arguments,
                *paths.video_encoder_options,
                "-r",
                str(VIDEO_FPS),
                "-fps_mode",
                "cfr",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-t",
                f"{target_duration:.6f}",
                "-movflags",
                "+faststart",
                "-metadata",
                "title=双语版",
                str(partial),
            ],
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def remux_video_with_subtitle(
    paths: ToolPaths,
    source: Path,
    subtitle_file: Path,
    destination: Path,
) -> None:
    """Convert an MKV fallback to MP4 while keeping a selectable subtitle track."""
    attempts = (
        ["-c:v", "copy"],
        [*paths.video_encoder_options, "-pix_fmt", "yuv420p"],
    )
    failures: list[str] = []
    for video_options in attempts:
        partial = partial_output_path(destination)
        try:
            processes.run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "warning",
                    "-i",
                    str(source),
                    "-f",
                    "srt",
                    "-i",
                    str(subtitle_file),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-map",
                    "1:s:0",
                    *video_options,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-c:s",
                    "mov_text",
                    "-metadata:s:s:0",
                    "language=und",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
            )
            os.replace(partial, destination)
            return
        except VideoPreparerError as exc:
            failures.append(str(exc))
        finally:
            partial.unlink(missing_ok=True)
    raise VideoPreparerError("无法把 ASMR Dubber 的字幕视频转换成 MP4：" + "；".join(failures))


def remux_video_without_subtitles(
    paths: ToolPaths,
    source: Path,
    destination: Path,
) -> None:
    """Create an MP4 with video and audio only, stripping every subtitle stream."""

    attempts = (
        ["-c:v", "copy"],
        [*paths.video_encoder_options, "-pix_fmt", "yuv420p"],
    )
    failures: list[str] = []
    for video_options in attempts:
        partial = partial_output_path(destination)
        try:
            processes.run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "warning",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-sn",
                    *video_options,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
            )
            os.replace(partial, destination)
            return
        except VideoPreparerError as exc:
            failures.append(str(exc))
        finally:
            partial.unlink(missing_ok=True)
    raise VideoPreparerError("无法生成不含字幕的 MP4：" + "；".join(failures))


def render_static_bilingual_video(
    paths: ToolPaths,
    audio_source: Path,
    background: Path | None,
    subtitle_file: Path,
    destination: Path,
    *,
    lead_seconds: int = 0,
    volume_db: float = 0.0,
    require_hard_subtitles: bool = False,
) -> None:
    """Render a lightweight static video, preferring visible hard subtitles."""

    constants.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    clean_video = constants.WORK_ROOT / f"clean-{uuid.uuid4().hex[:12]}.mp4"
    render_dir: Path | None = None
    try:
        render_static_video(
            paths,
            audio_source,
            background,
            clean_video,
            lead_seconds=lead_seconds,
            volume_db=volume_db,
        )
        render_dir = Path(tempfile.mkdtemp(prefix="subtitle-render-", dir=constants.WORK_ROOT))
        local_subtitle = render_dir / "subtitle.srt"
        shutil.copy2(subtitle_file, local_subtitle)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = partial_output_path(destination)
        try:
            processes.run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "warning",
                    "-i",
                    str(clean_video),
                    "-vf",
                    "subtitles=filename=subtitle.srt:charenc=UTF-8",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    *paths.video_encoder_options,
                    "-r",
                    str(VIDEO_FPS),
                    "-fps_mode",
                    "cfr",
                    "-c:a",
                    "copy",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
                cwd=render_dir,
            )
            os.replace(partial, destination)
        except VideoPreparerError:
            partial.unlink(missing_ok=True)
            if require_hard_subtitles:
                raise
            runlog.emit("  当前 FFmpeg 无法烧录字幕，改为写入可选择的内嵌字幕轨。")
            remux_video_with_subtitle(paths, clean_video, subtitle_file, destination)
        finally:
            partial.unlink(missing_ok=True)
    finally:
        clean_video.unlink(missing_ok=True)
        if render_dir is not None:
            shutil.rmtree(render_dir, ignore_errors=True)
