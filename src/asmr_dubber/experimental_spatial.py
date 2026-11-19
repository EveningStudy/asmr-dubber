"""Spatial RTF reference preparation, clip spatialization, and spectral helpers."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .errors import ProjectError
from .models import ProjectSettings
from .spatial_cues import rms
from .storage import atomic_write_text
from .task_control import check_cancelled


def _short_reference_fallback(mono, reference, settings, sentence_id):
    """Preserve the utterance without inventing phase from insufficient evidence."""
    from . import experimental_mix

    ratio = 1.0
    if reference.size and rms(reference) >= 1e-7:
        levels = np.sqrt(np.mean(reference.astype(np.float64) ** 2, axis=0))
        ratio = float(np.clip(levels[1] / max(levels[0], 1e-8), 1 / 16, 16))
    strength = float(settings.spatial_rtf_strength)
    ratio = ratio**strength
    gains = np.array([1.0, ratio]) * np.sqrt(2.0 / (1.0 + ratio**2))
    result = mono[:, None] * gains
    peak = float(np.max(np.abs(result), initial=0))
    ceiling = 10 ** (settings.chinese_line_peak_dbfs / 20)
    if peak > ceiling:
        result *= ceiling / peak
    experimental_mix.logger.warning(
        "RTF 降级：%s 参考过短或无有效信号；保留中文、时长和位置，"
        "仅使用有限左右电平差（无信号时居中），不估计相位。",
        sentence_id,
    )
    return result.astype(np.float32)


def stereo_reference(source: Path, directory: Path) -> Path:
    from . import experimental_mix
    from .audio import probe_audio

    destination = directory / "analysis/spatial-reference.wav"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        target_rate = 48000
        try:
            src_rate = probe_audio(source).sample_rate
            if src_rate > 0:
                target_rate = min(src_rate, 48000)
        except Exception:
            target_rate = 48000
        temporary = destination.with_name(f".{uuid.uuid4().hex}.wav")
        try:
            experimental_mix._run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    str(target_rate),
                    "-c:a",
                    "pcm_f32le",
                    "-rf64",
                    "auto",
                    str(temporary),
                ]
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def vocal_weighted_reference(source: Path, vocals: Path, directory: Path) -> Path:
    """Spatial reference that keeps the mix's true ear-to-ear phase but only where the voice is.

    Separation models do not promise to preserve binaural cues, so the separated
    voice is used only as a weight: a real time-frequency mask
    ``voice / (voice + rest)`` per ear is applied to the original mix.  The phase
    at each ear stays exactly that of the recording, while effects, music, hiss and
    whatever else is happening at the *other* ear stop steering the estimate.
    Binaural ASMR routinely has the voice at one ear and a different sound at the
    other, which a mask shared by both ears cannot tell apart.
    """
    from . import experimental_mix
    from .hashing import cached_sha256_file
    from .spatial_rtf import istft, stft

    base = experimental_mix.stereo_reference(source, directory)
    key = f"vocal-weighted-v2|{cached_sha256_file(base)}|{cached_sha256_file(vocals)}"
    destination = directory / "analysis/spatial-reference-vocal.wav"
    receipt = destination.with_suffix(".key")
    if destination.is_file() and receipt.is_file() and receipt.read_text("utf-8") == key:
        return destination
    token = uuid.uuid4().hex
    temporary = destination.with_name(f".{token}.wav")
    voice_file = destination.with_name(f".{token}.vocals.wav")
    size, hop = 2048, 512
    try:
        with sf.SoundFile(base) as mix:
            rate, frames = mix.samplerate, mix.frames
            experimental_mix._resample_file(vocals, voice_file, rate, frames)
            with (
                sf.SoundFile(voice_file) as voice,
                sf.SoundFile(
                    temporary, "w", samplerate=rate, channels=2, format="RF64", subtype="FLOAT"
                ) as out,
            ):
                block, pad = rate * 20, rate // 2
                for start in range(0, frames, block):
                    check_cancelled()
                    lo, hi = max(0, start - pad), min(frames, start + block + pad)
                    mix.seek(lo)
                    voice.seek(lo)
                    original = mix.read(hi - lo, dtype="float32", always_2d=True)
                    estimate = voice.read(hi - lo, dtype="float32", always_2d=True)
                    if estimate.shape[1] == 1:
                        estimate = np.repeat(estimate, 2, axis=1)
                    spectrum = stft(original.astype(np.float64), size, hop)
                    voiced = stft(estimate.astype(np.float64), size, hop)
                    voice_power = np.abs(voiced) ** 2
                    rest_power = np.abs(spectrum - voiced) ** 2
                    mask = voice_power / np.maximum(voice_power + rest_power, 1e-14)
                    weighted = istft(spectrum * mask, size, hop, len(original))
                    keep = slice(start - lo, min(frames, start + block) - lo)
                    out.write(weighted[keep].astype(np.float32))
        temporary.replace(destination)
        atomic_write_text(receipt, key)
    finally:
        temporary.unlink(missing_ok=True)
        voice_file.unlink(missing_ok=True)
    return destination


def spatialize_clip(
    mono: np.ndarray, rate: int, event, reference: Path, settings: ProjectSettings
) -> np.ndarray:
    """Second-version RTF in bounded overlapping blocks, after tempo/loudness processing."""
    from . import experimental_mix

    if not np.isfinite(mono).all():
        raise ProjectError(f"{event.sentence_id} 中文音频包含非有限采样。")
    if not len(mono) or rms(mono) < 1e-8:
        return np.column_stack((mono, mono))
    with sf.SoundFile(reference) as handle:
        sr = handle.samplerate
        if handle.channels != 2:
            raise ProjectError("RTF 参考必须是双声道。")
        lo = min(handle.frames, max(0, round(event.source_start_seconds * sr)))
        hi = max(lo, min(handle.frames, round(event.source_end_seconds * sr)))
        if hi - lo < sr // 10:
            handle.seek(lo)
            ref = handle.read(hi - lo, dtype="float32", always_2d=True)
            if not np.isfinite(ref).all():
                raise ProjectError(f"{event.sentence_id} 原声参考包含非有限采样。")
            return experimental_mix._short_reference_fallback(
                mono, ref, settings, event.sentence_id
            )
        chunk = max(1, round(settings.spatial_rtf_block_seconds * rate))
        pad = min(round(rate * 0.2), chunk // 4)
        result = np.zeros((len(mono), 2), dtype=np.float32)
        weights = np.zeros(len(mono), dtype=np.float32)
        single_chunk = len(mono) <= chunk
        handle.seek(lo)
        sentence_ref_rms = max(
            rms(handle.read(hi - lo, dtype="float32", always_2d=True)),
            1e-7,
        )
        for start in range(0, len(mono), chunk):
            check_cancelled()
            end = min(len(mono), start + chunk)
            a, b = max(0, start - pad), min(len(mono), end + pad)
            ref_start = lo + round((hi - lo) * a / len(mono))
            ref_end = lo + round((hi - lo) * b / len(mono))
            handle.seek(ref_start)
            ref = handle.read(max(1, ref_end - ref_start), dtype="float32", always_2d=True)
            if not np.isfinite(ref).all():
                raise ProjectError(f"{event.sentence_id} 原声参考包含非有限采样。")
            if sr != rate:
                ref = soxr.resample(ref, sr, rate)
            try:
                if len(ref) < rate // 10 or rms(ref) < 1e-7:
                    audio = experimental_mix._short_reference_fallback(
                        mono[a:b],
                        ref,
                        settings,
                        event.sentence_id,
                    )
                elif rms(mono[a:b]) < 1e-8:
                    audio = np.column_stack((mono[a:b], mono[a:b]))
                else:
                    audio, _ = experimental_mix.render_rtf(
                        mono[a:b],
                        ref,
                        rate,
                        strength=settings.spatial_rtf_strength,
                        level_strength=settings.spatial_rtf_level_strength,
                        color_strength=settings.spatial_rtf_color_strength,
                        fft_size=settings.spatial_rtf_fft_size,
                        hop_divisor=settings.spatial_rtf_hop_divisor,
                    )
            except ValueError as exc:
                if "参考片段电平过低" in str(exc):
                    audio = experimental_mix._short_reference_fallback(
                        mono[a:b],
                        ref,
                        settings,
                        event.sentence_id,
                    )
                else:
                    raise ProjectError(f"{event.sentence_id} RTF 失败：{exc}") from exc
            # Preserve inter-block approaching/receding distance envelope across multi-chunk lines.
            chunk_scale = rms(mono[a:b]) / max(rms(audio), 1e-8)
            if not single_chunk and rms(ref) >= 1e-7:
                rel_distance = float(
                    np.clip(
                        (rms(ref) / sentence_ref_rms) ** settings.spatial_rtf_level_strength,
                        0.25,
                        2.5,
                    )
                )
                chunk_scale *= rel_distance
            audio *= chunk_scale
            weight = np.ones(b - a, dtype=np.float32)
            if a < start:
                weight[: start - a] = np.linspace(0, 1, start - a)
            if b > end:
                weight[end - a :] = np.linspace(1, 0, b - end)
            result[a:b] += audio * weight[:, None]
            weights[a:b] += weight
        result /= np.maximum(weights[:, None], 1e-8)
        if not single_chunk and rms(result) > 1e-8:
            result *= rms(mono) / rms(result)
    peak = float(np.max(np.abs(result)))
    ceiling = 10 ** (settings.chinese_line_peak_dbfs / 20)
    if peak > ceiling:
        result *= ceiling / peak
    return result


def _decode_float(source: Path, destination: Path, rate: int, channels: int) -> Path:
    from . import experimental_mix

    experimental_mix._run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            "-rf64",
            "auto",
            str(destination),
        ]
    )
    return destination


def _resample_file(source: Path, destination: Path, rate: int, frames: int) -> Path:
    """Stream-resample to ``rate`` and pad/trim to exactly ``frames``."""
    with sf.SoundFile(source) as handle:
        written = 0
        with sf.SoundFile(
            destination,
            "w",
            samplerate=rate,
            channels=handle.channels,
            format="RF64",
            subtype="FLOAT",
        ) as out:
            stream = (
                None
                if handle.samplerate == rate
                else soxr.ResampleStream(
                    handle.samplerate, rate, handle.channels, dtype="float32", quality="VHQ"
                )
            )
            block = handle.samplerate * 10
            position = 0
            while position < handle.frames:
                check_cancelled()
                data = handle.read(block, dtype="float32", always_2d=True)
                position += len(data)
                if stream is not None:
                    data = stream.resample_chunk(data, last=position >= handle.frames)
                data = data[: max(0, frames - written)]
                out.write(data)
                written += len(data)
            if written < frames:
                out.write(np.zeros((frames - written, handle.channels), dtype=np.float32))
    return destination


def _suppress_vocal_residual(
    subtracted: np.ndarray,
    vocal_estimate: np.ndarray,
    removal: np.ndarray,
    rate: int,
    strength: float = 0.5,
) -> np.ndarray:
    """Soft Wiener residual suppression in vocal-dominated time-frequency bins."""
    from .spatial_rtf import istft, stft

    if strength <= 0.0:
        return subtracted
    active_idx = np.flatnonzero(removal > 1e-4)
    if not len(active_idx):
        return subtracted
    lo = max(0, int(active_idx[0]) - 1024)
    hi = min(len(subtracted), int(active_idx[-1]) + 1025)
    if hi - lo < 512:
        return subtracted
    size, hop = 2048, 512
    sub_slice = subtracted[lo:hi].astype(np.float64)
    voc_slice = vocal_estimate[lo:hi].astype(np.float64)
    channels = sub_slice.shape[1]
    if channels == 1:
        sub_2ch = np.repeat(sub_slice, 2, axis=1)
        voc_2ch = np.repeat(voc_slice, 2, axis=1)
    else:
        sub_2ch = sub_slice[:, :2]
        voc_2ch = voc_slice[:, :2]
    r_spec = stft(sub_2ch, size, hop)
    v_spec = stft(voc_2ch, size, hop)
    v_pow = np.abs(v_spec) ** 2
    r_pow = np.abs(r_spec) ** 2
    vocal_share = v_pow / np.maximum(v_pow + r_pow, 1e-16)

    hz = np.fft.rfftfreq(size, 1.0 / rate)
    # Mirror vocal dominance from 14-21.5 kHz into >21.8 kHz where 44.1 kHz separator had 0 output.
    upper_ref = (hz >= 14_000.0) & (hz <= 21_500.0)
    ultrasonic = hz > 21_800.0
    if np.any(upper_ref) and np.any(ultrasonic):
        high_share = np.mean(vocal_share[:, upper_ref, :], axis=1, keepdims=True)
        vocal_share[:, ultrasonic, :] = np.maximum(vocal_share[:, ultrasonic, :], high_share)

    # Attenuate up to 6 dB (gain >= 0.5 at strength=0.5) where separator confirms vocal dominance.
    max_atten = min(0.75, float(strength))
    suppression = 1.0 - max_atten * np.clip((vocal_share - 0.62) / 0.32, 0.0, 1.0)
    filtered = istft(r_spec * suppression, size, hop, hi - lo)
    if channels == 1:
        filtered = filtered[:, :1]
    out = subtracted.copy()
    gate = removal[lo:hi, None].astype(np.float32)
    out[lo:hi, : filtered.shape[1]] = (
        subtracted[lo:hi, : filtered.shape[1]] * (1.0 - gate) + filtered.astype(np.float32) * gate
    )
    return out
