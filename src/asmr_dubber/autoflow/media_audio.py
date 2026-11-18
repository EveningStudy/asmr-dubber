"""Audio normalisation and concatenation with ffmpeg."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import processes, runlog
from .constants import SAMPLE_RATE
from .domain import (
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .processes import audio_duration_samples


def normalize_and_concat(
    paths: ToolPaths,
    sources: list[AudioSource],
    workspace: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    segments_dir = workspace / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict[str, Any]] = []
    cumulative_samples = 0

    runlog.emit("\n[1/5] 统一音频规格并计算实际时间轴")
    for index, source in enumerate(sources, start=1):
        output = segments_dir / f"seg_{index:06d}.flac"
        runlog.emit(f"  [{index}/{len(sources)}] {source.path.name}")
        processes.run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-i",
                str(source.path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                f"aresample={SAMPLE_RATE}:async=0",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-sample_fmt",
                "s16",
                "-c:a",
                "flac",
                "-compression_level",
                "0",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(output),
            ],
        )
        samples = audio_duration_samples(paths, output)
        timeline.append(
            {
                "order": source.order,
                "source": str(source.path),
                "filename": source.path.name,
                "relative_path": source.relative_path or source.path.name,
                "title_ja": source.title_ja,
                "category": source.category,
                "transcript": str(source.transcript_path) if source.transcript_path else None,
                "transcript_language": source.transcript_language,
                "transcript_timed": source.transcript_timed,
                "transcript_mode": source.transcript_mode,
                "source_language": source.source_language,
                "normalized": str(output),
                "start_samples": cumulative_samples,
                "duration_samples": samples,
            }
        )
        cumulative_samples += samples

    concat_file = workspace / "concat.ffconcat"
    concat_lines = ["ffconcat version 1.0"]
    concat_lines.extend(
        f"file 'segments/seg_{index:06d}.flac'" for index in range(1, len(sources) + 1)
    )
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="ascii")

    master = workspace / "master.flac"
    runlog.emit("  正在拼接无损母带……")
    processes.run_ffmpeg(
        paths,
        [
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            concat_file.name,
            "-map",
            "0:a:0",
            "-af",
            "asetpts=N/SR/TB",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            str(master),
        ],
        cwd=workspace,
    )
    master_samples = audio_duration_samples(paths, master)
    if abs(master_samples - cumulative_samples) > 1:
        raise VideoPreparerError("拼接后的采样数与各小音频之和不一致，已停止，避免生成错误时间轴。")
    return master, timeline


def concat_audio_files(
    paths: ToolPaths,
    files: Sequence[Path],
    workspace: Path,
    *,
    expected_samples: Sequence[int] | None = None,
    name: str = "combined",
) -> tuple[Path, list[int]]:
    """Normalize and concatenate already-produced audio files.

    ASMR Dubber normally preserves the source duration, but a backend can
    differ by a few samples.  When the source timeline is available we pad or
    trim each part to that exact length before concatenation, so the resulting
    subtitle offsets remain deterministic.
    """

    if not files:
        raise VideoPreparerError("没有可合并的音频文件。")
    if expected_samples is not None and len(expected_samples) != len(files):
        raise VideoPreparerError("合并音频的时长清单与文件数量不一致。")
    workspace.mkdir(parents=True, exist_ok=True)
    segments = workspace / f"{name}-segments"
    segments.mkdir(parents=True, exist_ok=True)
    actual: list[int] = []
    for index, source in enumerate(files, start=1):
        source = source.resolve()
        if not source.is_file():
            raise VideoPreparerError(f"找不到要合并的音频：{source}")
        destination = segments / f"seg_{index:06d}.flac"
        filters = [
            f"aresample={SAMPLE_RATE}:async=0",
            f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
        ]
        target = int(expected_samples[index - 1]) if expected_samples is not None else 0
        if target > 0:
            filters.extend((f"apad=whole_len={target}", f"atrim=end_sample={target}"))
        filters.append("asetpts=N/SR/TB")
        processes.run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                ",".join(filters),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-sample_fmt",
                "s16",
                "-c:a",
                "flac",
                "-compression_level",
                "0",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(destination),
            ],
        )
        measured = audio_duration_samples(paths, destination)
        if target > 0 and abs(measured - target) > 1:
            raise VideoPreparerError(
                f"无法把第 {index} 段音频规整到原始时长（{measured}/{target} 采样）。"
            )
        actual.append(measured)

    concat_file = workspace / f"{name}.ffconcat"
    concat_file.write_text(
        "ffconcat version 1.0\n"
        + "\n".join(
            f"file '{segments.name}/seg_{index:06d}.flac'" for index in range(1, len(files) + 1)
        )
        + "\n",
        encoding="ascii",
    )
    destination = workspace / f"{name}.flac"
    processes.run_ffmpeg(
        paths,
        [
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            concat_file.name,
            "-map",
            "0:a:0",
            "-af",
            "asetpts=N/SR/TB",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            str(destination),
        ],
        cwd=workspace,
    )
    measured_total = audio_duration_samples(paths, destination)
    expected_total = sum(actual)
    if abs(measured_total - expected_total) > 1:
        raise VideoPreparerError(
            f"合并音频采样数异常：得到 {measured_total}，应为 {expected_total}。"
        )
    return destination, actual
