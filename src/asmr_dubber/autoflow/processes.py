"""Subprocess, ffmpeg and ffprobe helpers."""

from __future__ import annotations

import os
import subprocess
from fractions import Fraction
from pathlib import Path

from ..errors import OperationCancelledError
from ..task_control import (
    check_cancelled,
    register_process,
    unregister_process,
)
from ..task_control import (
    terminate_process_tree as _terminate_process_tree,
)
from . import runlog
from .constants import SAMPLE_RATE
from .domain import (
    ToolPaths,
)
from .errors import VideoPreparerError


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop the current FFmpeg/ASMR CLI command and its child processes."""

    _terminate_process_tree(process)


def run_process(
    arguments: list[str], *, cwd: Path | None = None, tail: list[str] | None = None
) -> None:
    """Run a command, streaming its output to the log.

    ``tail`` receives the last few output lines so a failure can say what the
    command actually printed instead of only its exit code.
    """

    check_cancelled()
    if tail is None:
        tail = []
    runlog.log_event("运行命令：" + " ".join(str(item) for item in arguments))
    try:
        process = subprocess.Popen(
            arguments,
            start_new_session=os.name != "nt",
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法启动命令：{arguments[0]}: {exc}") from exc
    signal = register_process(process)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            runlog.emit(line.rstrip("\r\n"))
            if line.strip():
                tail.append(line.rstrip())
                del tail[:-12]
            check_cancelled(signal)
        result_code = process.wait()
        check_cancelled(signal)
    except (KeyboardInterrupt, OperationCancelledError):
        terminate_process_tree(process)
        raise
    finally:
        unregister_process(process, signal)
    if result_code != 0:
        raise VideoPreparerError(
            f"命令执行失败（退出码 {result_code}）：{Path(arguments[0]).name}"
            + (f"；最后输出：{tail[-1][-240:]}" if tail else "")
        )


def run_process_captured(arguments: list[str], *, cwd: Path | None = None) -> str:
    check_cancelled()
    environment = os.environ.copy()
    environment.update({"COLUMNS": "10000", "NO_COLOR": "1", "TERM": "dumb"})
    try:
        runlog.log_event("运行命令：" + " ".join(str(item) for item in arguments))
        process = subprocess.Popen(
            arguments,
            start_new_session=os.name != "nt",
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法启动命令：{arguments[0]}: {exc}") from exc
    signal = register_process(process)
    try:
        stdout, stderr = process.communicate()
        check_cancelled(signal)
    except OperationCancelledError:
        terminate_process_tree(process)
        raise
    finally:
        unregister_process(process, signal)
    if stdout:
        runlog.emit(stdout.rstrip("\r\n"))
    if stderr:
        runlog.emit(stderr.rstrip("\r\n"), error=True)
    if process.returncode != 0:
        raise VideoPreparerError(
            f"命令执行失败（退出码 {process.returncode}）：{Path(arguments[0]).name}"
        )
    return "\n".join(part for part in (stdout, stderr) if part)


_DEFAULT_RUN_PROCESS = run_process
_DEFAULT_RUN_PROCESS_CAPTURED = run_process_captured


def uses_default_process_runners() -> bool:
    return (
        run_process is _DEFAULT_RUN_PROCESS
        and run_process_captured is _DEFAULT_RUN_PROCESS_CAPTURED
    )


def run_ffmpeg(paths: ToolPaths, arguments: list[str], *, cwd: Path | None = None) -> None:
    command = [
        str(paths.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-y",
        *arguments,
    ]
    run_process(command, cwd=cwd)


def audio_duration_samples(paths: ToolPaths, media: Path) -> int:
    del paths
    try:
        import av

        with av.open(str(media)) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise VideoPreparerError(f"文件没有可用音轨：{media}")
            if stream.duration is not None and stream.time_base is not None:
                duration = Fraction(stream.duration) * Fraction(stream.time_base)
            elif container.duration is not None:
                duration = Fraction(container.duration, av.time_base)
            else:
                raise VideoPreparerError(f"无法取得准确音频时长：{media}")
            samples = round(float(duration) * SAMPLE_RATE)
            if samples <= 0:
                raise VideoPreparerError(f"音频长度无效：{media}")
            return samples
    except VideoPreparerError:
        raise
    except Exception as exc:
        raise VideoPreparerError(f"无法读取音频长度 {media.name}：{exc}") from exc


def media_duration_seconds(paths: ToolPaths, media: Path) -> float:
    del paths
    try:
        import av

        with av.open(str(media)) as container:
            durations: list[float] = []
            if container.duration is not None:
                durations.append(float(Fraction(container.duration, av.time_base)))
            durations.extend(
                float(Fraction(stream.duration) * Fraction(stream.time_base))
                for stream in container.streams
                if stream.duration is not None and stream.time_base is not None
            )
            if durations:
                return max(durations)
    except Exception as exc:
        raise VideoPreparerError(f"无法取得媒体时长 {media.name}：{exc}") from exc
    raise VideoPreparerError(f"无法取得媒体时长：{media}")


def video_keyframe_times(paths: ToolPaths, media: Path) -> list[float]:
    del paths
    try:
        import av

        with av.open(str(media)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return []
            return [
                float(Fraction(packet.pts) * Fraction(packet.time_base))
                for packet in container.demux(stream)
                if packet.is_keyframe and packet.pts is not None and packet.time_base is not None
            ]
    except Exception as exc:
        raise VideoPreparerError(f"无法检查视频关键帧 {media.name}：{exc}") from exc


def media_stream_types(paths: ToolPaths, media: Path) -> list[str]:
    del paths
    try:
        import av

        with av.open(str(media)) as container:
            return [str(stream.type) for stream in container.streams]
    except Exception as exc:
        raise VideoPreparerError(f"无法检查媒体流 {media.name}：{exc}") from exc
