from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import soundfile as sf
import soxr

from .audio_dsp import dominant_channel_mono
from .errors import AsmrDubberError, OperationCancelledError
from .models import AudioInfo
from .task_control import check_cancelled

ASR_ANALYSIS_NAME = "asr_16k_dominant.wav"
_FOLD_FRAME_SECONDS = 0.05


def extract_reference(
    source: Path,
    destination: Path,
    start_seconds: float,
    end_seconds: float,
    padding_seconds: float = 0.0,
    *,
    fullband: bool = False,
) -> Path:
    from . import audio

    start = max(0.0, start_seconds - padding_seconds)
    end = max(start + 0.05, end_seconds + padding_seconds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fullband:
        return audio._extract_fullband_reference(source, destination, start, end)
    audio._run_ffmpeg(
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


def _extract_fullband_reference(source: Path, destination: Path, start: float, end: float) -> Path:
    """Reference at the source bandwidth (capped at 48 kHz), folded by dominant channel."""
    from . import audio

    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.full.wav")
    try:
        audio._run_ffmpeg(
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
                "-c:a",
                "pcm_f32le",
                str(temporary),
            ]
        )
        samples, rate = sf.read(temporary, dtype="float32", always_2d=True)
    finally:
        temporary.unlink(missing_ok=True)
    mono = dominant_channel_mono(samples[:, :2], rate)
    if rate > 48_000:
        mono = soxr.resample(mono, rate, 48_000, quality="VHQ")
        rate = 48_000
    sf.write(destination, mono, rate, subtype="FLOAT", format="WAV")
    return destination


def make_asr_analysis_copy(source: Path, destination: Path) -> Path:
    """16 kHz mono copy for speech recognition that follows the louder ear.

    Averaging the two channels of a binaural whisper halves the voice (-6 dB) and
    mixes in whatever the other ear hears.  Each 50 ms is instead weighted by the
    channels' own energy, so a whisper at one ear arrives at full level, while
    a centred voice still averages to the usual mono.
    """
    from . import audio

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            cached = sf.info(destination)
            if cached.samplerate == 16_000 and cached.channels == 1 and cached.subtype == "FLOAT":
                return destination
        except (OSError, RuntimeError):
            pass
    stereo = destination.with_name(f".{destination.name}.stereo.tmp.wav")
    temporary = destination.with_name(f".{destination.name}.tmp.wav")
    try:
        audio._run_ffmpeg(
            [
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ar",
                "16000",
                "-c:a",
                "pcm_f32le",
                "-rf64",
                "auto",
                str(stereo),
            ]
        )
        with (
            sf.SoundFile(stereo) as handle,
            sf.SoundFile(
                temporary, "w", samplerate=16_000, channels=1, format="WAV", subtype="FLOAT"
            ) as out,
        ):
            frame = round(16_000 * _FOLD_FRAME_SECONDS)
            for block in handle.blocks(blocksize=frame * 200, dtype="float32", always_2d=True):
                check_cancelled()
                out.write(audio._fold_by_energy(block, frame))
        converted = sf.info(temporary)
        if converted.samplerate != 16_000 or converted.channels != 1 or converted.frames <= 0:
            raise AsmrDubberError("ASR 分析副本参数无效。")
        temporary.replace(destination)
    finally:
        stereo.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return destination


def make_analysis_copy(source: Path, destination: Path) -> Path:
    """Create the model-only 16 kHz mono copy; the source itself is never rewritten."""
    from . import audio

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
        audio._run_ffmpeg(
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
    from . import audio

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
        audio._run_ffmpeg(
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
    from . import audio

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
            audio._run_ffmpeg(
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
    source_language: str = "ja",
    target_language: str = "zh",
) -> Path:
    """Create a video with subtitles, preferring a burned-in MP4.

    The original source is never modified. When hard-subtitle support is not
    present in the bundled FFmpeg, the function falls back to an embedded,
    selectable subtitle stream while still copying the original video stream.
    """
    from . import audio

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
                    audio._run_ffmpeg(
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
                    audio._run_ffmpeg(
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
                            + (
                                {"ja": "jpn", "en": "eng", "zh": "zho"}.get(source_language, "und")
                                if subtitle_language == "source"
                                else {
                                    "zh": "eng" if target_language == "en" else "zho",
                                    "bilingual": "und",
                                }.get(subtitle_language, "und")
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
