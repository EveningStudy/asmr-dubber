from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .constants import MAX_CHINESE_AUTO_SPEED
from .errors import ProjectError


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
    gain_db: float = 0.0
    # Part of the synthesized file that is actually placed (seconds, before tempo).
    trim_start_seconds: float = 0.0
    trim_end_seconds: float | None = None


def downmix_to_stereo(audio: np.ndarray) -> np.ndarray:
    """Downmix >2 channel surround audio to stereo while preserving Center (ITU-R BS.775)."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data[:, None]
    channels = data.shape[1]
    if channels <= 2:
        return data
    inv_sqrt2 = math.sqrt(0.5)
    left = data[:, 0].astype(np.float64)
    right = data[:, 1].astype(np.float64)
    if channels == 3:
        center = data[:, 2].astype(np.float64) * inv_sqrt2
        left += center
        right += center
    elif channels == 4:
        left += data[:, 2].astype(np.float64) * inv_sqrt2
        right += data[:, 3].astype(np.float64) * inv_sqrt2
    elif channels == 5:
        center = data[:, 2].astype(np.float64) * inv_sqrt2
        left += center + data[:, 3].astype(np.float64) * inv_sqrt2
        right += center + data[:, 4].astype(np.float64) * inv_sqrt2
    else:
        # Standard 5.1 / 7.1 WAV layout: FL(0), FR(1), FC(2), LFE(3), Ls(4), Rs(5), [Lb(6), Rb(7)].
        center = data[:, 2].astype(np.float64) * inv_sqrt2
        left += center + data[:, 4].astype(np.float64) * inv_sqrt2
        right += center + data[:, 5].astype(np.float64) * inv_sqrt2
        if channels >= 8:
            left += data[:, 6].astype(np.float64) * inv_sqrt2
            right += data[:, 7].astype(np.float64) * inv_sqrt2
    return np.column_stack((left, right)).astype(np.float32)


def dominant_channel_mono(audio: np.ndarray, rate: int) -> np.ndarray:
    """Fold a binaural or surround clip to mono without comb loss or dropping Center."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1 or data.shape[1] == 1:
        return data.reshape(-1)
    if data.shape[1] > 2:
        data = downmix_to_stereo(data)
    left, right = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    power_left, power_right = float(np.mean(left**2)), float(np.mean(right**2))
    if min(power_left, power_right) <= 1e-14:
        return (left if power_left >= power_right else right).astype(np.float32)
    ratio_db = 10.0 * math.log10(power_left / power_right)
    if abs(ratio_db) >= 3.0:
        return (left if ratio_db > 0 else right).astype(np.float32)
    limit = max(1, round(rate * 0.001))
    size = 1 << (2 * len(left) - 1).bit_length()
    spectrum = np.fft.rfft(left, size) * np.conj(np.fft.rfft(right, size))
    correlation = np.fft.irfft(spectrum / np.maximum(np.abs(spectrum), 1e-12), size)
    lags = np.concatenate((np.arange(0, limit + 1), np.arange(-limit, 0)))
    lag = int(lags[np.argmax(correlation[lags])])
    # ``lag`` > 0 means the left channel is late; delay the right one to match.
    shifted = np.zeros_like(right)
    if lag > 0:
        shifted[lag:] = right[:-lag]
    elif lag < 0:
        shifted[:lag] = right[-lag:]
    else:
        shifted = right
    return ((left + shifted) * 0.5).astype(np.float32)


def _fold_by_energy(block: np.ndarray, frame: int, rate: int = 16_000) -> np.ndarray:
    if block.shape[1] == 1:
        return block[:, 0]
    if block.shape[1] > 2:
        block = downmix_to_stereo(block)
    left, right = block[:, 0].astype(np.float64), block[:, 1].astype(np.float64)
    count = max(1, int(np.ceil(len(block) / frame)))
    padded = count * frame - len(block)
    frames_l = np.pad(left, (0, padded)).reshape(count, frame)
    frames_r = np.pad(right, (0, padded)).reshape(count, frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    vocal_band = (freqs >= 300.0) & (freqs <= 4000.0)
    if not np.any(vocal_band):
        vocal_band = slice(None)
    spec_l = np.fft.rfft(frames_l, axis=1)[:, vocal_band]
    spec_r = np.fft.rfft(frames_r, axis=1)[:, vocal_band]
    amp_left = np.sqrt(np.mean(np.abs(spec_l) ** 2, axis=1))
    amp_right = np.sqrt(np.mean(np.abs(spec_r) ** 2, axis=1))
    weight = np.clip((amp_left + 1e-12) / (amp_left + amp_right + 2e-12), 0.2, 0.8)
    if count >= 3:
        weight = np.convolve(np.pad(weight, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    positions = (np.arange(len(block)) + 0.5) / frame - 0.5
    weight_left = np.interp(positions, np.arange(count), weight)
    return (left * weight_left + right * (1.0 - weight_left)).astype(np.float32)


def speech_bounds(path: Path) -> tuple[float, float]:
    """Start and end (seconds) of the audible part of a synthesized line.

    Zero-shot TTS often pads a line with up to a second of silence.  The bounds
    keep everything within 40 dB of the line's loud frames, plus a short margin,
    so soft whispered onsets and breaths survive.
    """
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / rate
    hop = max(1, round(rate * 0.010))
    count = len(mono) // hop
    if count < 4:
        return 0.0, duration
    blocks = mono[: count * hop].reshape(count, hop).astype(np.float64)
    frames = np.sqrt(np.mean(blocks**2, axis=1))
    reference = float(np.percentile(frames, 95))
    if reference <= 1e-7:
        return 0.0, duration
    active = np.flatnonzero(frames >= reference * 0.01)
    start = max(0.0, active[0] * hop / rate - 0.015)
    end = min(duration, (active[-1] + 1) * hop / rate + 0.040)
    if end - start < 0.05:
        return 0.0, duration
    return start, end


def _read_adjusted_mono(
    source: Path,
    target_rate: int,
    speed_factor: float,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float | None = None,
) -> np.ndarray:
    """Trim, change tempo with FFmpeg ``atempo``, and resample in memory.

    ``atempo`` won the blind comparison against the in-tree phase-locked vocoder
    (``time_stretch.py``) on whispered lines at 1.1x and 1.2x, so it stays the method.
    """
    if not math.isfinite(speed_factor) or not 0.75 <= speed_factor <= MAX_CHINESE_AUTO_SPEED:
        raise ProjectError(f"中文配音自动变速倍速必须在 0.75 到 {MAX_CHINESE_AUTO_SPEED:g} 之间。")
    data, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = np.asarray(np.mean(data, axis=1, dtype=np.float32), dtype=np.float32)
    if trim_start_seconds > 0.0 or trim_end_seconds is not None:
        lo = max(0, min(len(mono), round(trim_start_seconds * source_rate)))
        hi = (
            len(mono)
            if trim_end_seconds is None
            else max(lo + 1, min(len(mono), round(trim_end_seconds * source_rate)))
        )
        mono = mono[lo:hi]
    if abs(speed_factor - 1.0) > 1e-4 and len(mono) > 0:
        mono = _atempo(mono, source_rate, speed_factor, source)
    if source_rate != target_rate and len(mono) > 0:
        mono = soxr.resample(mono, source_rate, target_rate, quality="VHQ")
    return np.asarray(mono, dtype=np.float32)


def _atempo(mono: np.ndarray, rate: int, speed_factor: float, near: Path) -> np.ndarray:
    """Pitch-preserving tempo change; one ``atempo`` stage per factor-of-two step."""
    import uuid

    from .audio import _run_ffmpeg

    stages = max(1, math.ceil(abs(math.log2(speed_factor))))
    stage_factor = speed_factor ** (1.0 / stages)
    token = uuid.uuid4().hex
    source = near.with_name(f".{near.stem}.{token}.tempo-in.wav")
    destination = near.with_name(f".{near.stem}.{token}.tempo-out.wav")
    try:
        sf.write(source, mono, rate, subtype="FLOAT")
        _run_ffmpeg(
            [
                "-y",
                "-i",
                str(source),
                "-filter:a",
                ",".join(f"atempo={stage_factor:.8f}" for _ in range(stages)),
                "-c:a",
                "pcm_f32le",
                str(destination),
            ]
        )
        stretched, _ = sf.read(destination, dtype="float32", always_2d=True)
    finally:
        source.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
    return np.asarray(stretched[:, 0], dtype=np.float32)


def _tempo_adjusted_audio(
    source: Path,
    destination: Path,
    speed_factor: float,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float | None = None,
) -> Path:
    if not math.isfinite(speed_factor) or not 0.75 <= speed_factor <= MAX_CHINESE_AUTO_SPEED:
        raise ProjectError(f"中文配音自动变速倍速必须在 0.75 到 {MAX_CHINESE_AUTO_SPEED:g} 之间。")
    trimmed = trim_start_seconds > 0.0 or trim_end_seconds is not None
    if abs(speed_factor - 1.0) <= 1e-4 and not trimmed:
        return source
    info = sf.info(source)
    mono = _read_adjusted_mono(
        source,
        info.samplerate,
        speed_factor,
        trim_start_seconds,
        trim_end_seconds,
    )
    sf.write(destination, mono, info.samplerate, subtype="FLOAT")
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


def apply_edge_fade(waveform: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Apply short linear fade-in/out ramps to avoid boundary clicks."""
    clip = np.asarray(waveform, dtype=np.float32).reshape(-1).copy()
    fade_frames = min(round(sample_rate * fade_ms / 1000.0), clip.size // 2)
    if fade_frames > 0:
        fade_in = np.linspace(0.0, 1.0, fade_frames, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_frames, endpoint=False, dtype=np.float32)
        clip[:fade_frames] *= fade_in
        clip[-fade_frames:] *= fade_out
    return clip


def apply_peak_ceiling(waveform: np.ndarray, peak_dbfs: float) -> np.ndarray:
    """Scale down a clip if its sample peak exceeds ``peak_dbfs``."""
    clip = np.asarray(waveform, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(clip), initial=0.0))
    ceiling = 10.0 ** (peak_dbfs / 20.0)
    if peak > ceiling:
        return (clip * np.float32(ceiling / peak)).astype(np.float32)
    return clip


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

    clip = apply_edge_fade(clip, sample_rate, fade_ms)
    return apply_peak_ceiling(clip, peak_dbfs)


def _limiter_envelope(target: np.ndarray, release: float) -> np.ndarray:
    """Vectorised limiter gain: g[i] = min(t[i], release*g[i-1] + (1-release)*t[i]).

    Instant attack, exponential release, g[-1] = 1.  Frames needing no
    reduction are closed-form (1 - (1 - g) * release**k); the frames that do
    need reduction are composed with a parallel prefix scan over the affine-min
    maps x -> min(A*x + B, C), so no Python per-sample loop remains.
    """
    envelope = np.ones(len(target), dtype=np.float64)
    over = np.flatnonzero(target < 1.0)
    if not over.size:
        return envelope
    a = float(release)
    b = 1.0 - a
    request = target[over].astype(np.float64)
    gap = (np.diff(over, prepend=0) - 1).astype(np.float64)
    gap[0] = over[0]
    decay = np.power(a, gap)
    scale = a * decay
    offset = a * (1.0 - decay) + b * request
    cap = request.copy()
    shift = 1
    while shift < len(over):
        # Compose element i-shift (first) with element i (second).
        new_scale = scale[:-shift] * scale[shift:]
        new_offset = scale[shift:] * offset[:-shift] + offset[shift:]
        new_cap = np.minimum(scale[shift:] * cap[:-shift] + offset[shift:], cap[shift:])
        scale = np.concatenate([scale[:shift], new_scale])
        offset = np.concatenate([offset[:shift], new_offset])
        cap = np.concatenate([cap[:shift], new_cap])
        shift *= 2
    at_over = np.minimum(scale + offset, cap)
    envelope[over] = at_over
    rest = np.flatnonzero(target >= 1.0)
    previous = np.searchsorted(over, rest, side="right") - 1
    distance = rest - over[np.maximum(previous, 0)]
    # Beyond ~30 time constants the release has recovered to within 1e-13.
    reach = 30.0 / max(-math.log(a), 1e-12) if a < 1.0 else float("inf")
    near = (previous >= 0) & (distance <= reach)
    rest, previous, distance = rest[near], previous[near], distance[near]
    envelope[rest] = 1.0 - (1.0 - at_over[previous]) * np.power(a, distance)
    return envelope


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
    envelope = _limiter_envelope(target, release).astype(np.float32)
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


class _VoiceMatcher:
    """Per-line timbre matching against the separated original voice."""

    def __init__(self, vocals: Path, events: list[StemEvent], rate: int) -> None:
        from .voice_match import long_term_spectrum

        self.rate = rate
        self.handle = sf.SoundFile(vocals)
        pieces = [self._window(e) for e in events]
        pieces = [p for p in pieces if len(p) >= rate // 10]
        # The whole performance gives a stable target; one short line alone would
        # drag the dub towards that line's particular vowels and consonants.
        self.project = long_term_spectrum(np.concatenate(pieces)) if pieces else None

    def _window(self, event: StemEvent) -> np.ndarray:
        source_rate = int(self.handle.samplerate)
        start = max(0, min(self.handle.frames, round(event.source_start_seconds * source_rate)))
        end = max(start, min(self.handle.frames, round(event.source_end_seconds * source_rate)))
        self.handle.seek(start)
        audio = self.handle.read(end - start, dtype="float32", always_2d=True)
        if not len(audio):
            return np.zeros(0, dtype=np.float32)
        mono = dominant_channel_mono(audio, source_rate)
        if source_rate != self.rate:
            mono = soxr.resample(mono, source_rate, self.rate, quality="HQ")
        return np.asarray(mono, dtype=np.float32)

    def apply(
        self,
        clip: np.ndarray,
        event: StemEvent,
        strength: float,
        extend: bool,
        whisper: float = 0.0,
    ) -> np.ndarray:
        from .voice_match import (
            long_term_spectrum,
            process_voice_match_fused,
            voiced_ratio,
        )

        if self.project is None or len(clip) < self.rate // 20:
            return clip
        level = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        if level <= 1e-8:
            return clip
        window = self._window(event)
        target = self.project
        sentence_spec: np.ndarray | None = None
        if len(window) >= self.rate // 4:
            own = long_term_spectrum(window)
            sentence_spec = own
            if len(window) >= self.rate // 2:
                weight = min(1.0, len(window) / (3.0 * self.rate)) * 0.5
                scale = np.sum(self.project) / max(float(np.sum(own)), 1e-30)
                target = self.project ** (1 - weight) * (own * scale) ** weight
        whisper_amount = 0.0
        if whisper > 0 and len(window) >= self.rate // 4:
            original, dub = voiced_ratio(window, self.rate), voiced_ratio(clip, self.rate)
            whisper_amount = whisper * float(np.clip((dub - original) / max(dub, 1e-3), 0.0, 1.0))
        return process_voice_match_fused(
            clip,
            self.rate,
            target,
            sentence_spectrum=sentence_spec,
            match_strength=strength,
            extend=extend,
            whisper_amount=whisper_amount,
        )

    def close(self) -> None:
        self.handle.close()


def extract_waveform_peaks(
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    bins: int = 160,
) -> list[float]:
    """Read bounded waveform amplitude peaks for ``[start_seconds, end_seconds]``."""
    import subprocess

    from .environment import ffmpeg_executable

    n_bins = max(16, min(512, int(bins)))
    t0 = max(0.0, float(start_seconds))
    t1 = max(t0 + 0.05, float(end_seconds))
    samples: np.ndarray | None = None
    try:
        with sf.SoundFile(audio_path) as handle:
            rate = handle.samplerate
            first = min(handle.frames, round(t0 * rate))
            count = max(1, min(handle.frames - first, round((t1 - t0) * rate)))
            if count > 0:
                handle.seek(first)
                block = handle.read(count, dtype="float32", always_2d=True)
                samples = np.max(np.abs(block), axis=1) if block.size else np.zeros(0, np.float32)
    except (OSError, RuntimeError):
        samples = None
    if samples is None:
        proc = subprocess.run(
            [
                ffmpeg_executable(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t0:.3f}",
                "-t",
                f"{t1 - t0:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "f32le",
                "-",
            ],
            capture_output=True,
            check=False,
        )
        raw = np.frombuffer(proc.stdout, dtype=np.float32) if proc.returncode == 0 else np.zeros(0)
        samples = np.abs(raw.astype(np.float32, copy=False))
    if samples.size == 0:
        return [0.0] * n_bins
    edges = np.linspace(0, samples.size, n_bins + 1, dtype=int)
    peaks = [
        float(np.max(samples[edges[i] : max(edges[i] + 1, edges[i + 1])]))
        if edges[i] < samples.size
        else 0.0
        for i in range(n_bins)
    ]
    ceiling = max(max(peaks), 1e-4)
    return [round(min(1.0, val / ceiling), 4) for val in peaks]
