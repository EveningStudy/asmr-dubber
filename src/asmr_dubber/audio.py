from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .environment import ffmpeg_executable
from .errors import AsmrDubberError, OperationCancelledError, ProjectError
from .models import AudioInfo, Sentence
from .task_control import (
    check_cancelled,
    register_process,
    terminate_process_tree,
    unregister_process,
)
from .timing import plan_dubbing_timing

Progress = Callable[[str, int, int], None]
_SOURCE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}
_SOURCE_DIGEST_LOCK = threading.Lock()
_COPY_BUFFER_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class StemEvent:
    sentence_id: str
    start_seconds: float
    audio_path: Path
    source_start_seconds: float = 0.0
    source_end_seconds: float = 0.0
    speed_factor: float = 1.0
    effective_duration_seconds: float | None = None
    remaining_overlap_seconds: float = 0.0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path: str | Path, sha256: str | None = None) -> AudioInfo:
    import av

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ProjectError(f"找不到音频：{source}")
    try:
        with av.open(str(source)) as container:
            if not container.streams.audio:
                raise ProjectError(f"文件中没有音轨：{source}")
            stream = container.streams.audio[0]
            video_stream = next(
                (
                    candidate
                    for candidate in container.streams.video
                    if not bool(
                        getattr(candidate, "disposition", 0)
                        & getattr(
                            type(getattr(candidate, "disposition", 0)),
                            "attached_pic",
                            0,
                        )
                    )
                ),
                None,
            )
            context = stream.codec_context
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                duration = 0.0
            if duration <= 0:
                # Some containers omit duration. Decode timestamps without changing the file.
                last_end = 0.0
                for frame in container.decode(stream):
                    if frame.time is not None:
                        last_end = max(last_end, frame.time + (frame.samples / frame.sample_rate))
                duration = last_end
            sample_rate = int(context.sample_rate or stream.rate or 0)
            channels = int(context.channels or 0)
            layout = context.layout.name if context.layout else None
            codec = context.name
            video_context = video_stream.codec_context if video_stream is not None else None
            video_rate = getattr(video_stream, "average_rate", None)
            try:
                frame_rate = float(video_rate) if video_rate is not None else None
            except (TypeError, ValueError, ZeroDivisionError):
                frame_rate = None
    except ProjectError:
        raise
    except Exception as exc:  # PyAV moved its public exception classes across releases.
        raise ProjectError(f"无法读取音频 {source}: {exc}") from exc
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise ProjectError(f"音频参数无效：{source}")
    return AudioInfo(
        path=source.name,
        sha256=sha256 or sha256_file(source),
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        channel_layout=layout,
        codec=codec,
        media_type="video" if video_stream is not None else "audio",
        video_codec=video_context.name if video_context is not None else None,
        video_width=(
            int(video_context.width)
            if video_context is not None and video_context.width > 0
            else None
        ),
        video_height=(
            int(video_context.height)
            if video_context is not None and video_context.height > 0
            else None
        ),
        video_frame_rate=frame_rate if frame_rate and frame_rate > 0 else None,
    )


def _cache_source_digest(path: Path, digest: str) -> None:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _SOURCE_DIGEST_LOCK:
        if len(_SOURCE_DIGEST_CACHE) >= 64:
            _SOURCE_DIGEST_CACHE.pop(next(iter(_SOURCE_DIGEST_CACHE)))
        _SOURCE_DIGEST_CACHE[key] = digest


def _verified_source_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _SOURCE_DIGEST_LOCK:
        cached = _SOURCE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(path, chunk_size=_COPY_BUFFER_SIZE)
    _cache_source_digest(path, digest)
    return digest


def copy_source_verbatim(
    source: str | Path,
    project_dir: Path,
    progress: Progress | None = None,
) -> tuple[Path, AudioInfo]:
    original = Path(source).expanduser().resolve()
    if not original.is_file():
        raise ProjectError(f"找不到输入文件：{original}")
    project_dir.mkdir(parents=True, exist_ok=True)
    suffix = original.suffix.lower() or ".audio"
    destination = project_dir / f"source{suffix}"
    source_hash: str
    if destination.exists():
        source_hash = sha256_file(original, chunk_size=_COPY_BUFFER_SIZE)
        if sha256_file(destination) != source_hash:
            raise ProjectError(f"项目中已存在不同的源文件：{destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        digest = hashlib.sha256()
        copied = 0
        total = original.stat().st_size
        try:
            with original.open("rb") as source_handle, temporary.open("xb") as target_handle:
                while block := source_handle.read(_COPY_BUFFER_SIZE):
                    target_handle.write(block)
                    digest.update(block)
                    copied += len(block)
                    if progress:
                        progress("建立项目：保存源文件", copied, total)
            shutil.copystat(original, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        source_hash = digest.hexdigest()
    _cache_source_digest(destination.resolve(), source_hash)
    info = probe_audio(destination, sha256=source_hash)
    info.path = destination.name
    return destination, info


def resolve_project_path(project_dir: Path, stored_path: str, label: str = "项目文件") -> Path:
    """Resolve a manifest path without allowing absolute paths or project escape."""
    root = project_dir.resolve()
    relative = Path(stored_path)
    if relative.is_absolute():
        raise ProjectError(f"{label}路径必须位于项目目录内。")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProjectError(f"{label}路径超出项目目录。") from exc
    return candidate


def project_file_exists(project_dir: Path, stored_path: str | None, label: str) -> bool:
    """Return whether a manifest-owned file exists, rejecting unsafe paths."""
    if not stored_path:
        return False
    return resolve_project_path(project_dir, stored_path, label).is_file()


def verify_source(project_dir: Path, info: AudioInfo) -> Path:
    root = project_dir.resolve()
    source = resolve_project_path(project_dir, info.path, "项目源文件")
    if source.parent != root:
        raise ProjectError("项目源文件路径超出项目目录。")
    if not source.is_file():
        raise ProjectError(f"项目源文件丢失：{source}")
    actual = _verified_source_digest(source)
    if actual != info.sha256:
        raise ProjectError("项目源文件已发生变化。为防止时间轴错位，已停止处理；请新建项目。")
    return source


def _run_ffmpeg(arguments: list[str], *, cwd: Path | None = None) -> None:
    command = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", *arguments]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    signal = register_process(process)
    try:
        while True:
            check_cancelled(signal)
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
    except OperationCancelledError:
        if process.poll() is None:
            terminate_process_tree(process)
        # Drain and close redirected pipes before callers remove FFmpeg's
        # temporary output. Windows can otherwise briefly retain the file
        # handle and replace cancellation with a misleading WinError 32.
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process)
            process.communicate()
        raise
    finally:
        unregister_process(process, signal)
    check_cancelled(signal)
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "unknown ffmpeg error"
        raise AsmrDubberError(f"ffmpeg 失败：{detail}")


def extract_reference(
    source: Path,
    destination: Path,
    start_seconds: float,
    end_seconds: float,
    padding_seconds: float = 0.0,
) -> Path:
    start = max(0.0, start_seconds - padding_seconds)
    end = max(start + 0.05, end_seconds + padding_seconds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
    )
    return destination


def make_analysis_copy(source: Path, destination: Path) -> Path:
    """Create the model-only 16 kHz mono copy; the source itself is never rewritten."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            cached = sf.info(destination)
            if cached.samplerate == 16_000 and cached.channels == 1 and cached.subtype == "FLOAT":
                return destination
        except (OSError, RuntimeError):
            pass
    temporary = destination.with_name(f".{destination.name}.tmp.wav")
    try:
        _run_ffmpeg(
            [
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_f32le",
                str(temporary),
            ]
        )
        converted = sf.info(temporary)
        if converted.samplerate != 16_000 or converted.channels != 1 or converted.frames <= 0:
            raise AsmrDubberError("ASR 分析副本参数无效。")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def sentence_events(
    project_dir: Path,
    sentences: Iterable[Sentence],
    chinese_dubbing_offset_ms: int = 0,
    chinese_max_auto_speed: float = 1.2,
) -> list[StemEvent]:
    available: list[Sentence] = []
    audio_paths: dict[str, Path] = {}
    durations: dict[str, float] = {}
    sentence_by_id: dict[str, Sentence] = {}
    for sentence in sentences:
        if not sentence.enabled or not sentence.zh_text or not sentence.tts_file:
            continue
        audio_path = resolve_project_path(
            project_dir,
            sentence.tts_file,
            f"句子 {sentence.id} 的中文音频",
        )
        if not audio_path.is_file():
            raise ProjectError(f"句子 {sentence.id} 的中文音频不存在：{audio_path}")
        try:
            clip_info = sf.info(audio_path)
        except (OSError, RuntimeError) as exc:
            raise ProjectError(f"无法读取句子 {sentence.id} 的中文音频：{audio_path}") from exc
        if clip_info.samplerate <= 0 or clip_info.frames <= 0:
            raise ProjectError(f"中文句子 {sentence.id} 的音频参数无效。")
        available.append(sentence)
        sentence_by_id[sentence.id] = sentence
        audio_paths[sentence.id] = audio_path
        durations[sentence.id] = clip_info.frames / clip_info.samplerate

    timings = plan_dubbing_timing(
        available,
        offset_ms=chinese_dubbing_offset_ms,
        max_auto_speed=chinese_max_auto_speed,
        durations=durations,
    )
    return [
        StemEvent(
            sentence_id=timing.sentence_id,
            start_seconds=timing.start_seconds,
            audio_path=audio_paths[timing.sentence_id],
            source_start_seconds=sentence_by_id[timing.sentence_id].start_seconds,
            source_end_seconds=sentence_by_id[timing.sentence_id].end_seconds,
            speed_factor=timing.speed_factor,
            effective_duration_seconds=timing.effective_duration_seconds,
            remaining_overlap_seconds=timing.remaining_overlap_seconds,
        )
        for timing in timings
    ]


def _tempo_adjusted_audio(source: Path, destination: Path, speed_factor: float) -> Path:
    if not math.isfinite(speed_factor) or not 1.0 <= speed_factor <= 2.0:
        raise ProjectError("中文配音自动加速倍速必须在 1.0 到 2.0 之间。")
    if speed_factor <= 1.0 + 1e-9:
        return source
    _run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-filter:a",
            f"atempo={speed_factor:.8f}",
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
    )
    return destination


def _read_resampled_mono(path: Path, target_rate: int) -> np.ndarray:
    data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1, dtype=np.float32)
    if source_rate != target_rate:
        mono = soxr.resample(mono, source_rate, target_rate, quality="VHQ")
    return np.asarray(mono, dtype=np.float32)


def active_rms_dbfs(waveform: np.ndarray, sample_rate: int, frame_ms: float = 50.0) -> float:
    """Return a short-utterance-safe gated RMS level in dBFS."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return float("-inf")
    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    padding = (-samples.size) % frame_size
    if padding:
        samples = np.pad(samples, (0, padding))
    frames = samples.reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float64))
    maximum = float(np.max(frame_rms))
    if maximum <= 1e-8:
        return float("-inf")
    # A relative 20 dB gate ignores leading/trailing silence without assuming
    # that quiet ASMR speech must meet a broadcast loudness threshold.
    active = frame_rms[frame_rms >= maximum * 0.1]
    rms = float(np.sqrt(np.mean(active * active, dtype=np.float64)))
    return 20.0 * math.log10(max(rms, 1e-12))


def prepare_chinese_clip(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_active_rms_dbfs: float,
    max_loudness_boost_db: float,
    peak_dbfs: float,
    fade_ms: float,
) -> np.ndarray:
    """Apply one transparent gain, short edge fades, and a peak ceiling."""
    clip = np.asarray(waveform, dtype=np.float32).reshape(-1).copy()
    level = active_rms_dbfs(clip, sample_rate)
    if math.isfinite(level):
        gain_db = min(target_active_rms_dbfs - level, max_loudness_boost_db)
        clip *= np.float32(10.0 ** (gain_db / 20.0))

    fade_frames = min(round(sample_rate * fade_ms / 1000.0), clip.size // 2)
    if fade_frames > 0:
        fade_in = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_frames, endpoint=False, dtype=np.float32)
        clip[:fade_frames] *= fade_in
        clip[-fade_frames:] *= fade_out

    peak = float(np.max(np.abs(clip), initial=0.0))
    ceiling = 10.0 ** (peak_dbfs / 20.0)
    if peak > ceiling:
        clip *= np.float32(ceiling / peak)
    return clip


def _sum_with_peak_ceiling(
    current: np.ndarray,
    addition: np.ndarray,
    peak_dbfs: float | None,
    sample_rate: int,
) -> np.ndarray:
    combined = np.asarray(current, dtype=np.float32) + np.asarray(addition, dtype=np.float32)
    if peak_dbfs is None:
        return combined
    ceiling = np.float32(10.0 ** (peak_dbfs / 20.0))
    frame_peak = np.max(np.abs(combined), axis=1)
    target = np.ones_like(frame_peak, dtype=np.float32)
    over = frame_peak > ceiling
    target[over] = ceiling / frame_peak[over]
    # A limiter needs an immediate attack to guarantee the ceiling, followed
    # by a smooth 50 ms release instead of independent per-sample gain jumps.
    release = math.exp(-1.0 / max(1.0, sample_rate * 0.050))
    envelope = np.empty_like(target)
    gain = np.float32(1.0)
    for index, requested in enumerate(target):
        if requested < gain:
            gain = requested
        else:
            gain = np.float32(release * gain + (1.0 - release) * requested)
        envelope[index] = gain
    return combined * envelope[:, None]


def _route_mono_to_channels(
    waveform: np.ndarray,
    channels: int,
    channel_layout: str | None,
    routing: str,
) -> np.ndarray:
    if channels == 1:
        return waveform[:, None]
    if channels == 2 or routing == "all":
        return np.broadcast_to(waveform[:, None], (len(waveform), channels))
    values = np.zeros((len(waveform), channels), dtype=np.float32)
    # FFmpeg/soundfile use center as the third channel for conventional
    # 3.0/5.1/7.1 layouts. Unknown multichannel layouts still avoid spraying a
    # mono voice into every surround channel.
    center = 2 if channels >= 3 else 0
    values[:, center] = waveform
    return values


def build_chinese_stem(
    destination: Path,
    events: list[StemEvent],
    source_info: AudioInfo,
    chinese_gain_db: float,
    normalize_loudness: bool = True,
    source_reference_path: Path | None = None,
    match_source_loudness: bool = True,
    relative_loudness_db: float = 0.0,
    minimum_active_rms_dbfs: float = -42.0,
    target_active_rms_dbfs: float = -30.0,
    max_loudness_boost_db: float = 12.0,
    line_peak_dbfs: float = -9.0,
    stem_peak_dbfs: float | None = -3.0,
    fade_ms: float = 8.0,
    channel_routing: str = "auto",
    progress: Progress | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = source_info.sample_rate
    channels = source_info.channels
    gain = 10.0 ** (chinese_gain_db / 20.0)

    prepared: list[tuple[StemEvent, int]] = []
    total_frames = math.ceil(source_info.duration_seconds * rate)
    for event in events:
        clip_info = sf.info(event.audio_path)
        if clip_info.samplerate <= 0 or clip_info.frames <= 0:
            raise ProjectError(f"中文句子 {event.sentence_id} 的音频参数无效。")
        resampled_frames = math.ceil(clip_info.frames * rate / clip_info.samplerate)
        start_frame = round(event.start_seconds * rate)
        prepared.append((event, start_frame))
        total_frames = max(total_frames, start_frame + resampled_frames)

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.wav")
    source_reference: sf.SoundFile | None = None
    try:
        block_frames = max(rate * 10, 1)
        initialization_blocks = max(1, math.ceil(total_frames / block_frames))
        total_work = initialization_blocks + len(prepared)
        with sf.SoundFile(
            temporary,
            mode="w",
            samplerate=rate,
            channels=channels,
            format="RF64",
            subtype="FLOAT",
        ) as output:
            zeros = np.zeros((block_frames, channels), dtype=np.float32)
            for block_index, offset in enumerate(
                range(0, total_frames, block_frames),
                start=1,
            ):
                check_cancelled()
                output.write(zeros[: min(block_frames, total_frames - offset)])
                if progress:
                    progress("初始化中文中间轨", block_index, total_work)

        if normalize_loudness and match_source_loudness and source_reference_path is not None:
            source_reference = sf.SoundFile(source_reference_path)

        with sf.SoundFile(temporary, mode="r+") as stem:
            for index, (event, start_frame) in enumerate(prepared, start=1):
                check_cancelled()
                # Only one synthesized line is resident at a time, so multi-hour
                # projects do not accumulate every Chinese waveform in RAM.
                tempo_file = destination.with_name(
                    f".{destination.stem}.{uuid.uuid4().hex}.tempo.wav"
                )
                try:
                    clip_path = _tempo_adjusted_audio(
                        event.audio_path,
                        tempo_file,
                        event.speed_factor,
                    )
                    clip = _read_resampled_mono(clip_path, rate)
                finally:
                    tempo_file.unlink(missing_ok=True)
                if normalize_loudness:
                    target_level = target_active_rms_dbfs
                    if source_reference is not None:
                        source_rate = int(source_reference.samplerate)
                        window_start = max(
                            0,
                            min(
                                source_reference.frames,
                                round(event.source_start_seconds * source_rate),
                            ),
                        )
                        window_end = max(
                            window_start,
                            min(
                                source_reference.frames,
                                round(event.source_end_seconds * source_rate),
                            ),
                        )
                        source_reference.seek(window_start)
                        source_window = source_reference.read(
                            window_end - window_start,
                            dtype="float32",
                            always_2d=True,
                        )
                        source_level = active_rms_dbfs(
                            np.asarray(
                                np.mean(source_window, axis=1, dtype=np.float32),
                                dtype=np.float32,
                            ),
                            source_rate,
                        )
                        if math.isfinite(source_level):
                            target_level = source_level + relative_loudness_db
                        else:
                            target_level = minimum_active_rms_dbfs
                        target_level = min(
                            target_active_rms_dbfs,
                            max(minimum_active_rms_dbfs, target_level),
                        )
                    clip = prepare_chinese_clip(
                        clip,
                        rate,
                        target_active_rms_dbfs=target_level,
                        max_loudness_boost_db=max_loudness_boost_db,
                        peak_dbfs=line_peak_dbfs,
                        fade_ms=fade_ms,
                    )
                clip *= gain
                clip_offset = 0
                if start_frame < 0:
                    clip_offset = -start_frame
                    start_frame = 0
                usable = clip[clip_offset:]
                if len(usable):
                    end_frame = start_frame + len(usable)
                    if end_frame > total_frames:
                        raise ProjectError(
                            f"中文句子 {event.sentence_id} 重采样长度超出预估；已停止以避免截断。"
                        )
                    stem.seek(start_frame)
                    current = stem.read(len(usable), dtype="float32", always_2d=True)
                    values = _route_mono_to_channels(
                        usable,
                        channels,
                        source_info.channel_layout,
                        channel_routing,
                    )
                    stem.seek(start_frame)
                    stem.write(
                        _sum_with_peak_ceiling(
                            current,
                            values,
                            stem_peak_dbfs,
                            rate,
                        )
                    )
                if progress:
                    speed_note = (
                        f"（自动加速 {event.speed_factor:.2f}×）"
                        if event.speed_factor > 1.0 + 1e-9
                        else ""
                    )
                    progress(
                        f"放置中文句子 {event.sentence_id}{speed_note}",
                        initialization_blocks + index,
                        total_work,
                    )

        temporary.replace(destination)
    finally:
        if source_reference is not None:
            source_reference.close()
        temporary.unlink(missing_ok=True)
    return destination


def mix_original_and_stem(
    source: Path,
    stem: Path,
    destination: Path,
    source_info: AudioInfo,
    *,
    output_codec: str = "pcm_s24le",
    peak_protection: bool = True,
    peak_limit_dbfs: float = -1.0,
) -> Path:
    """Add the Chinese stem with optional transparent final peak protection.

    The default 24-bit PCM WAV provides high-resolution browser playback.
    Internal stems remain float32.
    """
    if output_codec not in {"pcm_s24le", "pcm_f32le"}:
        raise ValueError(f"unsupported mix output codec: {output_codec}")
    if not -12.0 <= peak_limit_dbfs < 0.0:
        raise ValueError("peak_limit_dbfs must be between -12 and 0 dBFS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mix_filter = "[0:a:0][1:a:0]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0"
    if peak_protection:
        limit = 10.0 ** (peak_limit_dbfs / 20.0)
        mix_filter += f",alimiter=limit={limit:.8f}:attack=5:release=50:level=false:latency=true"
    mix_filter += "[out]"
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        _run_ffmpeg(
            [
                "-y",
                "-i",
                str(source),
                "-i",
                str(stem),
                "-filter_complex",
                mix_filter,
                "-map",
                "[out]",
                "-map_metadata",
                "0",
                "-vn",
                "-ar",
                str(source_info.sample_rate),
                "-ac",
                str(source_info.channels),
                "-c:a",
                output_codec,
                "-rf64",
                "auto",
                str(temporary),
            ]
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def mux_mixed_video(
    source_video: Path,
    mixed_audio: Path,
    destination: Path,
) -> Path:
    """Copy the original video stream and attach the completed mixed audio.

    MP4/AAC is attempted first for broad playback support. If the source video
    codec cannot be copied into MP4, Matroska/FLAC keeps both streams without a
    video re-encode or a lossy audio encode.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        (
            destination.with_suffix(".mp4"),
            ["-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart"],
        ),
        (destination.with_suffix(".mkv"), ["-c:a", "flac"]),
    ]
    failures: list[str] = []
    for output, audio_arguments in candidates:
        temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}")
        try:
            _run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(source_video),
                    "-i",
                    str(mixed_audio),
                    "-map",
                    "0:v?",
                    "-map",
                    "1:a:0",
                    "-map",
                    "0:a?",
                    "-map",
                    "0:s?",
                    "-map",
                    "0:t?",
                    "-map_metadata",
                    "0",
                    "-map_chapters",
                    "0",
                    "-c",
                    "copy",
                    *audio_arguments,
                    "-disposition:a",
                    "0",
                    "-disposition:a:0",
                    "default",
                    str(temporary),
                ]
            )
            temporary.replace(output)
            return output
        except OperationCancelledError:
            raise
        except AsmrDubberError as exc:
            failures.append(str(exc))
        finally:
            temporary.unlink(missing_ok=True)
    raise AsmrDubberError("无法把混音音轨封装回视频：" + "；".join(failures[-2:]))


def render_subtitled_video(
    source_video: Path,
    subtitle_file: Path,
    destination: Path,
    *,
    replacement_audio: Path | None = None,
    subtitle_language: str = "bilingual",
) -> Path:
    """Create a video with subtitles, preferring a burned-in MP4.

    The original source is never modified. When hard-subtitle support is not
    present in the bundled FFmpeg, the function falls back to an embedded,
    selectable subtitle stream while still copying the original video stream.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_dir = destination.parent / f".subtitle-render-{uuid.uuid4().hex}"
    render_dir.mkdir(parents=True)
    local_subtitle = render_dir / "subtitle.srt"
    shutil.copyfile(subtitle_file, local_subtitle)
    failures: list[str] = []
    input_arguments = ["-i", str(source_video)]
    audio_maps = ["-map", "0:a?"]
    if replacement_audio is not None:
        input_arguments.extend(["-i", str(replacement_audio)])
        audio_maps = ["-map", "1:a:0", "-map", "0:a?"]

    video_encoders = [
        ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"],
        ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p"],
    ]
    audio_variants = (
        [["-c:a", "aac", "-b:a", "320k"]]
        if replacement_audio is not None
        else [["-c:a", "copy"], ["-c:a", "aac", "-b:a", "320k"]]
    )
    try:
        output = destination.with_suffix(".mp4")
        for video_arguments in video_encoders:
            for audio_arguments in audio_variants:
                temporary = output.with_name(
                    f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
                )
                try:
                    _run_ffmpeg(
                        [
                            "-y",
                            *input_arguments,
                            "-vf",
                            "subtitles=filename=subtitle.srt:charenc=UTF-8",
                            "-map",
                            "0:v:0",
                            *audio_maps,
                            "-map_metadata",
                            "0",
                            "-map_chapters",
                            "0",
                            *video_arguments,
                            *audio_arguments,
                            "-movflags",
                            "+faststart",
                            str(temporary),
                        ],
                        cwd=render_dir,
                    )
                    temporary.replace(output)
                    return output
                except OperationCancelledError:
                    raise
                except AsmrDubberError as exc:
                    failures.append(str(exc))
                finally:
                    temporary.unlink(missing_ok=True)

        # A selectable subtitle track is a reliable fallback for FFmpeg builds
        # without libass or a usable video encoder.
        subtitle_input_index = 2 if replacement_audio is not None else 1
        soft_candidates = [
            (
                destination.with_suffix(".mp4"),
                ["-c:s", "mov_text"],
                [["-c:a", "aac", "-b:a", "320k"]]
                if replacement_audio is not None
                else [["-c:a", "copy"], ["-c:a", "aac", "-b:a", "320k"]],
            ),
            (
                destination.with_suffix(".mkv"),
                ["-c:s", "srt"],
                [["-c:a", "flac"]] if replacement_audio is not None else [["-c:a", "copy"]],
            ),
        ]
        for output, subtitle_arguments, fallback_audio_variants in soft_candidates:
            for audio_arguments in fallback_audio_variants:
                temporary = output.with_name(
                    f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}"
                )
                try:
                    _run_ffmpeg(
                        [
                            "-y",
                            *input_arguments,
                            "-f",
                            "srt",
                            "-i",
                            str(local_subtitle),
                            "-map",
                            "0:v:0",
                            *audio_maps,
                            "-map",
                            f"{subtitle_input_index}:s:0",
                            "-map",
                            "0:s?",
                            "-map_metadata",
                            "0",
                            "-map_chapters",
                            "0",
                            "-c:v",
                            "copy",
                            *audio_arguments,
                            *subtitle_arguments,
                            "-metadata:s:s:0",
                            "language="
                            + {"zh": "zho", "ja": "jpn", "bilingual": "und"}.get(
                                subtitle_language, "und"
                            ),
                            str(temporary),
                        ],
                        cwd=render_dir,
                    )
                    temporary.replace(output)
                    return output
                except OperationCancelledError:
                    raise
                except AsmrDubberError as exc:
                    failures.append(str(exc))
                finally:
                    temporary.unlink(missing_ok=True)
    finally:
        shutil.rmtree(render_dir, ignore_errors=True)
    raise AsmrDubberError("生成带字幕视频失败：" + "；".join(failures[-3:]))
