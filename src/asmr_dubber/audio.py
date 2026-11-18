from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_dsp import (
    StemEvent as StemEvent,
)
from .audio_dsp import (
    _fold_by_energy as _fold_by_energy,
)
from .audio_dsp import (
    _limiter_envelope as _limiter_envelope,
)
from .audio_dsp import (
    _read_adjusted_mono as _read_adjusted_mono,
)
from .audio_dsp import (
    _read_resampled_mono as _read_resampled_mono,
)
from .audio_dsp import (
    _route_mono_to_channels as _route_mono_to_channels,
)
from .audio_dsp import (
    _sum_with_peak_ceiling as _sum_with_peak_ceiling,
)
from .audio_dsp import (
    _tempo_adjusted_audio as _tempo_adjusted_audio,
)
from .audio_dsp import (
    _VoiceMatcher as _VoiceMatcher,
)
from .audio_dsp import (
    active_rms_dbfs as active_rms_dbfs,
)
from .audio_dsp import (
    dominant_channel_mono as dominant_channel_mono,
)
from .audio_dsp import (
    extract_waveform_peaks as extract_waveform_peaks,
)
from .audio_dsp import (
    prepare_chinese_clip as prepare_chinese_clip,
)
from .audio_dsp import (
    speech_bounds as speech_bounds,
)
from .audio_ffmpeg import (
    _FOLD_FRAME_SECONDS as _FOLD_FRAME_SECONDS,
)
from .audio_ffmpeg import (
    ASR_ANALYSIS_NAME as ASR_ANALYSIS_NAME,
)
from .audio_ffmpeg import (
    _extract_fullband_reference as _extract_fullband_reference,
)
from .audio_ffmpeg import (
    extract_reference as extract_reference,
)
from .audio_ffmpeg import (
    make_analysis_copy as make_analysis_copy,
)
from .audio_ffmpeg import (
    make_asr_analysis_copy as make_asr_analysis_copy,
)
from .audio_ffmpeg import (
    mix_original_and_stem as mix_original_and_stem,
)
from .audio_ffmpeg import (
    mux_mixed_video as mux_mixed_video,
)
from .audio_ffmpeg import (
    render_subtitled_video as render_subtitled_video,
)
from .environment import ffmpeg_executable
from .errors import AsmrDubberError, OperationCancelledError, ProjectError
from .filtering import has_speakable_text
from .models import AudioInfo, ProjectSettings, Sentence
from .storage import require_disk_space
from .task_control import (
    check_cancelled,
    register_process,
    terminate_process_tree,
    unregister_process,
)
from .timing import DubbingTimingMode, plan_dubbing_timing

Progress = Callable[[str, int, int], None]
_SOURCE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}
_SOURCE_DIGEST_LOCK = threading.Lock()
_COPY_BUFFER_SIZE = 8 * 1024 * 1024


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
        # Text tags are optional for every operation below.  Some MP3 files in
        # the wild contain truncated or incorrectly declared ID3 text (often
        # Japanese metadata).  PyAV decodes stream metadata eagerly and its
        # default ``strict`` policy would otherwise reject an audio stream that
        # FFmpeg can decode perfectly well.
        with av.open(str(source), metadata_errors="replace") as container:
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
        total = original.stat().st_size
        linked = False
        try:
            os.link(original, destination)
            linked = True
        except OSError:
            linked = False
        if linked:
            digest = hashlib.sha256()
            copied = 0
            with destination.open("rb") as source_handle:
                while block := source_handle.read(_COPY_BUFFER_SIZE):
                    digest.update(block)
                    copied += len(block)
                    if progress:
                        progress("建立项目：保存源文件", copied, total)
            if total == 0 and progress:
                progress("建立项目：保存源文件", 0, 0)
            source_hash = digest.hexdigest()
        else:
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            require_disk_space(destination.parent, total)
            digest = hashlib.sha256()
            copied = 0
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
        start_new_session=os.name != "nt",
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


def _default_replacement_obstacles(
    sentences: list[Sentence],
    dubbed_ids: set[str],
) -> list[tuple[float, float]]:
    obstacles: list[tuple[float, float]] = []
    for s in sentences:
        if s.id in dubbed_ids or s.original_audio_enabled is False:
            continue
        duration = s.end_seconds - s.start_seconds
        if duration < 0.20:
            continue
        if s.status == "skipped_filler" and s.original_audio_enabled is not True and duration < 0.5:
            continue
        obstacles.append((s.start_seconds, s.end_seconds))
    return obstacles


def sentence_events(
    project_dir: Path,
    sentences: Iterable[Sentence],
    chinese_dubbing_offset_ms: int = 0,
    chinese_max_auto_speed: float = 1.2,
    chinese_dubbing_timing_mode: DubbingTimingMode = "fit_window",
    *,
    trim_silence: bool = False,
    obstacles: Iterable[tuple[float, float]] | None = None,
) -> list[StemEvent]:
    all_sentences = list(sentences)
    available: list[Sentence] = []
    audio_paths: dict[str, Path] = {}
    durations: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    sentence_by_id: dict[str, Sentence] = {}
    for sentence in all_sentences:
        if (
            not sentence.enabled
            or not has_speakable_text(sentence.zh_text)
            or not sentence.tts_file
        ):
            continue
        audio_path = resolve_project_path(
            project_dir,
            sentence.tts_file,
            f"句子 {sentence.id} 的中文音频",
        )
        if not sentence.chinese_audio_enabled:
            if not trim_silence:
                available.append(sentence)
                sentence_by_id[sentence.id] = sentence
                audio_paths[sentence.id] = audio_path
                durations[sentence.id] = (
                    sentence.tts_duration_seconds or sentence.end_seconds - sentence.start_seconds
                )
            continue
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
        if trim_silence:
            bounds[sentence.id] = speech_bounds(audio_path)
            durations[sentence.id] = bounds[sentence.id][1] - bounds[sentence.id][0]

    effective_obstacles = (
        list(obstacles)
        if obstacles is not None
        else (
            _default_replacement_obstacles(
                all_sentences,
                {s.id for s in available if s.chinese_audio_enabled},
            )
            if trim_silence
            else None
        )
    )
    timings = plan_dubbing_timing(
        available,
        offset_ms=chinese_dubbing_offset_ms,
        max_auto_speed=chinese_max_auto_speed,
        durations=durations,
        mode=chinese_dubbing_timing_mode,
        obstacles=effective_obstacles,
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
            gain_db=sentence_by_id[timing.sentence_id].chinese_audio_gain_db,
            trim_start_seconds=bounds.get(timing.sentence_id, (0.0, None))[0],
            trim_end_seconds=bounds.get(timing.sentence_id, (0.0, None))[1],
        )
        for timing in timings
        if sentence_by_id[timing.sentence_id].chinese_audio_enabled
    ]


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
    spatial_reference_path: Path | None = None,
    spatial_settings: ProjectSettings | None = None,
    voice_match_path: str | None = None,
    voice_match_strength: float = 0.0,
    bandwidth_extension: bool = False,
    whisper_match: float = 0.0,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = source_info.sample_rate
    voice_matcher = (
        _VoiceMatcher(Path(voice_match_path), events, rate)
        if voice_match_path
        and (voice_match_strength > 0 or bandwidth_extension or whisper_match > 0)
        else None
    )
    channels = source_info.channels
    gain = 10.0 ** (chinese_gain_db / 20.0)

    prepared: list[tuple[StemEvent, int]] = []
    total_frames = math.ceil(source_info.duration_seconds * rate)
    for event in events:
        clip_info = sf.info(event.audio_path)
        if clip_info.samplerate <= 0 or clip_info.frames <= 0:
            raise ProjectError(f"中文句子 {event.sentence_id} 的音频参数无效。")
        resampled_frames = (
            math.ceil(
                clip_info.frames / max(event.speed_factor, 0.70) * rate / clip_info.samplerate
            )
            + 64
        )
        start_frame = round(event.start_seconds * rate)
        prepared.append((event, start_frame))
        total_frames = max(total_frames, start_frame + resampled_frames)

    require_disk_space(destination.parent, total_frames * channels * 8)
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
                clip = _read_adjusted_mono(
                    event.audio_path,
                    rate,
                    event.speed_factor,
                    event.trim_start_seconds,
                    event.trim_end_seconds,
                )
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
                        # Total binaural power sqrt(L^2 + R^2) matches FFmpeg's -ac 1 diotic
                        # reference while avoiding single-ear -3 dB loss and phase cancellation.
                        ch_scale = min(2, source_window.shape[1])
                        equal_power_env = np.sqrt(
                            np.mean(source_window.astype(np.float64) ** 2, axis=1) * ch_scale
                        ).astype(np.float32)
                        source_level = active_rms_dbfs(equal_power_env, source_rate)
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
                if voice_matcher is not None:
                    clip = voice_matcher.apply(
                        clip, event, voice_match_strength, bandwidth_extension, whisper_match
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
                    if spatial_reference_path is not None:
                        from .experimental_mix import spatialize_clip

                        if channels != 2:
                            raise ProjectError("RTF 仅支持双声道原音频。")
                        if spatial_settings is None:
                            raise ProjectError("RTF 参数未提供。")
                        values = spatialize_clip(
                            clip, rate, event, spatial_reference_path, spatial_settings
                        )[clip_offset:]
                    else:
                        values = _route_mono_to_channels(
                            usable,
                            channels,
                            source_info.channel_layout,
                            channel_routing,
                        )
                    values = values * np.float32(10 ** (event.gain_db / 20))
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
        if voice_matcher is not None:
            voice_matcher.close()
        temporary.unlink(missing_ok=True)
    return destination
