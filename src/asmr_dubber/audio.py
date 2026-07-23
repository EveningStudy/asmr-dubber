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
from .errors import AsmrDubberError, ProjectError
from .models import AudioInfo, Sentence

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
        raise ProjectError(f"找不到输入音频：{original}")
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
                        progress("建立项目：保存源音频", copied, total)
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
    source = resolve_project_path(project_dir, info.path, "项目源音频")
    if source.parent != root:
        raise ProjectError("项目源音频路径超出项目目录。")
    if not source.is_file():
        raise ProjectError(f"项目源音频丢失：{source}")
    actual = _verified_source_digest(source)
    if actual != info.sha256:
        raise ProjectError("项目源音频已发生变化。为防止时间轴错位，已停止处理；请新建项目。")
    return source


def _run_ffmpeg(arguments: list[str]) -> None:
    command = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", *arguments]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
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
    global_overlap_seconds: float,
    global_overlap_percentage: float = 50.0,
) -> list[StemEvent]:
    events: list[StemEvent] = []
    for sentence in sentences:
        if not sentence.enabled or not sentence.tts_file:
            continue
        audio_path = resolve_project_path(
            project_dir,
            sentence.tts_file,
            f"句子 {sentence.id} 的中文音频",
        )
        if not audio_path.is_file():
            raise ProjectError(f"句子 {sentence.id} 的中文音频不存在：{audio_path}")
        events.append(
            StemEvent(
                sentence_id=sentence.id,
                start_seconds=sentence.chinese_start_seconds(
                    global_overlap_seconds,
                    global_overlap_percentage,
                ),
                audio_path=audio_path,
                source_start_seconds=sentence.start_seconds,
                source_end_seconds=sentence.end_seconds,
            )
        )
    return sorted(events, key=lambda event: (event.start_seconds, event.sentence_id))


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
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
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

    fade_frames = min(int(round(sample_rate * fade_ms / 1000.0)), clip.size // 2)
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
) -> np.ndarray:
    combined = np.asarray(current, dtype=np.float32) + np.asarray(addition, dtype=np.float32)
    if peak_dbfs is None:
        return combined
    ceiling = np.float32(10.0 ** (peak_dbfs / 20.0))
    frame_peak = np.max(np.abs(combined), axis=1)
    scale = np.ones_like(frame_peak, dtype=np.float32)
    over = frame_peak > ceiling
    scale[over] = ceiling / frame_peak[over]
    return combined * scale[:, None]


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
    progress: Progress | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = source_info.sample_rate
    channels = source_info.channels
    gain = 10.0 ** (chinese_gain_db / 20.0)

    prepared: list[tuple[StemEvent, int]] = []
    total_frames = int(math.ceil(source_info.duration_seconds * rate))
    for event in events:
        clip_info = sf.info(event.audio_path)
        if clip_info.samplerate <= 0 or clip_info.frames <= 0:
            raise ProjectError(f"中文句子 {event.sentence_id} 的音频参数无效。")
        resampled_frames = math.ceil(clip_info.frames * rate / clip_info.samplerate)
        start_frame = int(round(event.start_seconds * rate))
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
                output.write(zeros[: min(block_frames, total_frames - offset)])
                if progress:
                    progress("初始化中文中间轨", block_index, total_work)

        if normalize_loudness and match_source_loudness and source_reference_path is not None:
            source_reference = sf.SoundFile(source_reference_path)

        with sf.SoundFile(temporary, mode="r+") as stem:
            for index, (event, start_frame) in enumerate(prepared, start=1):
                # Only one synthesized line is resident at a time, so multi-hour
                # projects do not accumulate every Chinese waveform in RAM.
                clip = _read_resampled_mono(event.audio_path, rate)
                if normalize_loudness:
                    target_level = target_active_rms_dbfs
                    if source_reference is not None:
                        source_rate = int(source_reference.samplerate)
                        window_start = max(
                            0,
                            min(
                                source_reference.frames,
                                int(round(event.source_start_seconds * source_rate)),
                            ),
                        )
                        window_end = max(
                            window_start,
                            min(
                                source_reference.frames,
                                int(round(event.source_end_seconds * source_rate)),
                            ),
                        )
                        source_reference.seek(window_start)
                        source_window = source_reference.read(
                            window_end - window_start,
                            dtype="float32",
                            always_2d=True,
                        )
                        source_level = active_rms_dbfs(
                            np.mean(source_window, axis=1, dtype=np.float32),
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
                    values = np.broadcast_to(usable[:, None], (len(usable), channels))
                    stem.seek(start_frame)
                    stem.write(
                        _sum_with_peak_ceiling(
                            current,
                            values,
                            stem_peak_dbfs,
                        )
                    )
                if progress:
                    progress(
                        f"放置中文句子 {event.sentence_id}",
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
) -> Path:
    """Add the Chinese stem without gain, normalization, limiting, or timeline edits.

    The default 24-bit PCM WAV provides high-resolution browser playback.
    Internal stems remain float32.
    """
    if output_codec not in {"pcm_s24le", "pcm_f32le"}:
        raise ValueError(f"unsupported mix output codec: {output_codec}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-i",
            str(stem),
            "-filter_complex",
            "[0:a:0][1:a:0]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[out]",
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
            str(destination),
        ]
    )
    return destination
