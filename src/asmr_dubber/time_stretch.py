"""MIT-licensed high-fidelity speech & whisper time-stretcher.

Uses an exact dual-frame phase-locked vocoder (Laroche & Dolson 1999) combined
with spectral-flux transient phase reset and high-frequency whisper de-flanging.
Unlike FFmpeg ``atempo`` (20 ms SOLA), sibilants and plosives keep sharp onsets
without double-attacks or comb-filtering, and whispered breath avoids metallic
ringing across ``[0.75x, 2.5x]`` speed ratios.
"""

from __future__ import annotations

import math

import numpy as np


def _fft_params(sample_rate: int) -> tuple[int, int]:
    if sample_rate >= 32_000:
        return 2048, 256
    if sample_rate >= 20_000:
        return 1024, 128
    return 512, 64


def _nearest_peak_indices(magnitudes: np.ndarray) -> np.ndarray:
    """Map every FFT bin in each frame to its dominant local spectral peak."""
    frames, bins = magnitudes.shape
    is_peak = np.zeros((frames, bins), dtype=bool)
    if bins >= 3:
        is_peak[:, 1:-1] = (magnitudes[:, 1:-1] >= magnitudes[:, :-2]) & (
            magnitudes[:, 1:-1] >= magnitudes[:, 2:]
        )
    is_peak[:, 0] = magnitudes[:, 0] >= magnitudes[:, min(1, bins - 1)]
    is_peak[:, -1] = magnitudes[:, -1] >= magnitudes[:, max(0, bins - 2)]

    grid = np.broadcast_to(np.arange(bins, dtype=np.int32), (frames, bins))
    left = np.where(is_peak, grid, 0)
    np.maximum.accumulate(left, axis=1, out=left)

    right = np.where(is_peak, grid, bins - 1)
    right = np.minimum.accumulate(right[:, ::-1], axis=1)[:, ::-1]

    choose_right = (right - grid) < (grid - left)
    return np.where(choose_right, right, left)


def _stretch_mono(
    mono: np.ndarray,
    sample_rate: int,
    speed_factor: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    source = np.asarray(mono, dtype=np.float64).reshape(-1)
    if not len(source):
        return np.zeros(0, dtype=np.float32)
    if abs(speed_factor - 1.0) <= 1e-4:
        return source.astype(np.float32)
    target_len = max(1, round(len(source) / speed_factor))
    if len(source) < 64 or target_len < 16:
        positions = np.linspace(0, max(0, len(source) - 1), target_len)
        return np.interp(positions, np.arange(len(source)), source).astype(np.float32)

    n_fft, hop_s = _fft_params(sample_rate)
    pad = n_fft
    padded = np.pad(source, (pad, pad + n_fft), mode="constant")
    window = np.sqrt(np.hanning(n_fft))

    frame_count = max(1, math.ceil((target_len + n_fft) / hop_s))
    synth_indices = np.arange(frame_count, dtype=np.float64)
    # Center of each synthesis frame mapped back onto the padded input timeline.
    curr_starts = np.clip(
        np.round(pad - n_fft // 2 + synth_indices * hop_s * speed_factor).astype(np.int64),
        hop_s,
        len(padded) - n_fft,
    )
    prev_starts = curr_starts - hop_s

    offsets = np.arange(n_fft, dtype=np.int64)[None, :]
    curr_frames = padded[curr_starts[:, None] + offsets] * window[None, :]
    prev_frames = padded[prev_starts[:, None] + offsets] * window[None, :]

    spec_curr = np.fft.rfft(curr_frames, n=n_fft, axis=1)
    spec_prev = np.fft.rfft(prev_frames, n=n_fft, axis=1)
    mag = np.abs(spec_curr)
    ana_phase = np.angle(spec_curr)
    # Exact phase advance over one synthesis hop H_s without bin-center unwrapping drift.
    delta_phase = np.angle(spec_curr * np.conj(spec_prev))

    peak_idx = _nearest_peak_indices(mag)
    hz = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Detect broadband transient attacks (plosives / crisp sibilant onsets) for phase reset.
    flux = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        diff = np.maximum(0.0, mag[1:] - mag[:-1])
        high_weight = np.clip(hz / 2500.0, 0.4, 2.0)[None, :]
        flux[1:] = np.sum(diff * high_weight, axis=1)
    baseline = np.maximum(np.percentile(flux, 65) * 2.4, 1e-9)
    is_transient = flux > baseline

    # Estimate per-frame whisper/noiseiness in 1-5 kHz (low spectral crest -> breath/whisper).
    mid_band = (hz >= 800.0) & (hz <= min(5000.0, sample_rate * 0.45))
    if np.any(mid_band):
        band_mag = mag[:, mid_band] + 1e-12
        geo_mean = np.exp(np.mean(np.log(band_mag), axis=1))
        arith_mean = np.mean(band_mag, axis=1)
        flatness = np.clip(geo_mean / np.maximum(arith_mean, 1e-12), 0.0, 1.0)
    else:
        flatness = np.zeros(frame_count, dtype=np.float64)
    whisper_mask = (hz >= 4200.0)[None, :]
    rng = np.random.default_rng(seed)
    jitter = (rng.random(mag.shape) - 0.5) * (0.55 * np.pi)

    synth_phase = np.empty_like(ana_phase)
    synth_phase[0] = ana_phase[0]
    row_idx = np.arange(mag.shape[1], dtype=np.int32)
    for m in range(1, frame_count):
        if is_transient[m]:
            synth_phase[m] = ana_phase[m]
            continue
        peaks = peak_idx[m]
        # Advance peak phases and lock surrounding bins to preserve vertical harmonic coherence.
        advanced_peaks = synth_phase[m - 1, peaks] + delta_phase[m, peaks]
        locked = advanced_peaks + (ana_phase[m, row_idx] - ana_phase[m, peaks])
        if flatness[m] > 0.35:
            breath_amount = min(1.0, (flatness[m] - 0.35) / 0.35)
            locked = np.where(whisper_mask[0], locked + breath_amount * jitter[m], locked)
        synth_phase[m] = locked

    synth_spec = mag * np.exp(1j * synth_phase)
    synth_spec[:, 0] = np.real(synth_spec[:, 0])
    if n_fft % 2 == 0:
        synth_spec[:, -1] = np.real(synth_spec[:, -1])

    frames_out = np.fft.irfft(synth_spec, n=n_fft, axis=1) * window[None, :]
    total_samples = hop_s * (frame_count - 1) + n_fft
    output = np.zeros(total_samples, dtype=np.float64)
    weights = np.zeros(total_samples, dtype=np.float64)
    win_sq = window**2
    for m in range(frame_count):
        start = m * hop_s
        output[start : start + n_fft] += frames_out[m]
        weights[start : start + n_fft] += win_sq
    output /= np.maximum(weights, 1e-9)

    start_idx = n_fft // 2
    trimmed = output[start_idx : start_idx + target_len]
    if len(trimmed) < target_len:
        trimmed = np.pad(trimmed, (0, target_len - len(trimmed)))

    # Preserve overall RMS energy accurately across stretch ratios.
    in_rms = float(np.sqrt(np.mean(source**2)))
    out_rms = float(np.sqrt(np.mean(trimmed**2)))
    if in_rms > 1e-8 and out_rms > 1e-8:
        trimmed *= in_rms / out_rms
    peak_in = float(np.max(np.abs(source)))
    peak_out = float(np.max(np.abs(trimmed)))
    if peak_out > max(peak_in * 1.05, 0.999) and peak_out > 1e-8:
        trimmed *= max(peak_in, 0.98) / peak_out
    return trimmed.astype(np.float32)


def stretch_speech(
    audio: np.ndarray,
    sample_rate: int,
    speed_factor: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Time-stretch mono ``(N,)`` or multichannel ``(N, C)`` speech without pitch change."""
    if not math.isfinite(speed_factor) or speed_factor <= 0.0:
        raise ValueError(f"speed_factor must be positive and finite, got {speed_factor}")
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return _stretch_mono(array, sample_rate, speed_factor, seed=seed)
    if array.ndim == 2:
        if array.shape[0] == 0:
            return array.copy()
        channels = [
            _stretch_mono(array[:, ch], sample_rate, speed_factor, seed=seed + ch)
            for ch in range(array.shape[1])
        ]
        min_len = min(len(ch) for ch in channels)
        return np.column_stack([ch[:min_len] for ch in channels]).astype(np.float32)
    raise ValueError("audio must be 1D mono or 2D (frames, channels)")
