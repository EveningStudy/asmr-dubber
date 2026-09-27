"""Experimental complex relative-transfer baseline (not a pretrained neural model).

Single-source, locally stationary spatial covariance -> dominant eigenvector.
Retain frequency-dependent interaural phase, unlike a scalar-delay renderer.
No noise-only covariance is available: this is NOT covariance whitening or DP-RTF.
Only differential filtering is identifiable; common room response is not recovered.
"""

from __future__ import annotations

import numpy as np

from .spatial_cues import analyze, rms, smooth


def stft(audio: np.ndarray, size: int, hop: int) -> np.ndarray:
    padded = np.pad(audio, ((size // 2, size), (0, 0)))
    starts = range(0, len(padded) - size + 1, hop)
    window = np.sqrt(np.hanning(size))
    return np.stack([np.fft.rfft(padded[i : i + size] * window[:, None], axis=0) for i in starts])


def istft(spectra: np.ndarray, size: int, hop: int, length: int) -> np.ndarray:
    window = np.sqrt(np.hanning(size))
    output = np.zeros((hop * (len(spectra) - 1) + size, 2))
    weights = np.zeros(len(output))
    for i, frame in enumerate(np.fft.irfft(spectra, n=size, axis=1)):
        start = i * hop
        output[start : start + size] += frame * window[:, None]
        weights[start : start + size] += window**2
    output /= np.maximum(weights[:, None], 1e-12)
    return output[size // 2 : size // 2 + length]


def filters(reference: np.ndarray, rate: int, size: int = 4096, hop: int = 512):
    cues = analyze(reference, rate)
    spec = stft(reference, size, hop)
    times = np.minimum(np.arange(len(spec)) * hop / rate, cues.times[-1])
    hz = np.fft.rfftfreq(size, 1 / rate)
    itd = np.interp(times, cues.times, cues.itd_seconds)
    ll, rr = np.abs(spec[:, :, 0]) ** 2, np.abs(spec[:, :, 1]) ** 2
    # Derotate the estimated bulk ITD before covariance averaging: otherwise
    # motion cancels cross spectra at high frequencies and destroys coherence.
    rotation = np.exp(2j * np.pi * itd[:, None] * hz)
    cross = spec[:, :, 1] * spec[:, :, 0].conj() * rotation
    time_width = max(3, round(0.16 * rate / hop))
    ll, rr, cross = (smooth(x, time_width) for x in (ll, rr, cross))
    # Small *linear-frequency* smoothing preserves substantially finer notches
    # than the previous 28-band representation; not a phoneme EQ transplant.
    ll, rr, cross = (smooth(x.T, 3).T for x in (ll, rr, cross))
    power = ll + rr
    floor = np.maximum(np.max(power, axis=1, keepdims=True) * 1e-5, 1e-14)
    coherence = np.clip(np.abs(cross) ** 2 / np.maximum(ll * rr, floor**2), 0, 1)
    # Closed-form principal eigenvector ratio v_R/v_L of a Hermitian 2x2 matrix.
    discriminant = np.sqrt((ll - rr) ** 2 + 4 * np.abs(cross) ** 2)
    eigenvalue = (ll + rr + discriminant) / 2
    magnitude = np.sqrt(np.maximum(eigenvalue - ll, floor) / np.maximum(eigenvalue - rr, floor))
    log_ratio = np.clip(20 * np.log10(magnitude), -24, 24)
    prior_ild = np.stack([np.interp(times, cues.times, col) for col in cues.ild_db.T], axis=1)
    prior_ratio = np.stack([-np.interp(hz, cues.band_hz, row) for row in prior_ild])
    reliability = np.clip((coherence - 0.35) / 0.45, 0, 1)
    reliability *= np.clip(power / (floor * 20), 0, 1)
    reliability *= ((hz >= 100) & (hz <= min(18000, rate * 0.44)))[None, :]
    # Temporal/frequency smoothing of confidence prevents abrupt fallback edges.
    reliability = smooth(smooth(reliability, 5).T, 5).T
    log_ratio = prior_ratio * (1 - reliability) + log_ratio * reliability
    # Remove bulk delay first so residual phase is normally near zero; unwrap
    # both axes to prevent polarity jumps when splitting the phase across ears.
    residual = np.unwrap(np.unwrap(np.angle(cross), axis=1), axis=0)
    residual -= 2 * np.pi * np.round(np.median(residual[:, :5], axis=1)[:, None] / (2 * np.pi))
    residual = np.clip(residual, -np.pi, np.pi) * reliability
    residual = smooth(residual, 5)
    total_phase = residual - 2 * np.pi * itd[:, None] * hz
    ratio = 10 ** (log_ratio / 20)
    left_gain = np.sqrt(2 / (1 + ratio**2))
    right_gain = ratio * left_gain
    response = np.stack(
        [left_gain * np.exp(-0.5j * total_phase), right_gain * np.exp(0.5j * total_phase)], axis=-1
    )
    # Real waveform boundary bins cannot carry arbitrary complex phase.
    response[:, 0, :] = np.abs(response[:, 0, :])
    response[:, -1, :] = np.abs(response[:, -1, :])
    return (
        response,
        cues,
        {
            "method": "time-varying principal-eigenvector complex RTF (experimental DSP)",
            "fft_size": size,
            "hop_samples": hop,
            "mean_rtf_confidence": float(np.mean(reliability)),
            "reliable_bin_fraction": float(np.mean(reliability > 0.5)),
            "residual_phase_rms_radians": rms(residual),
            "fallback": "v1 smoothed ILD and bulk ITD for low-confidence bins",
            "common_room_response_recovered": False,
        },
    )


def render_rtf(
    mono: np.ndarray,
    reference: np.ndarray,
    rate: int,
    *,
    level_strength: float = 0.7,
    color_strength: float = 0.35,
    fft_size: int = 0,
    hop_divisor: int = 8,
):
    if mono.ndim != 1 or not len(mono) or not np.isfinite(mono).all() or rms(mono) < 1e-8:
        raise ValueError("中文语音为空、静音或包含非有限采样。")
    size = fft_size or (4096 if rate >= 32000 else 2048)
    hop = size // hop_divisor
    response, cues, report = filters(reference, rate, size, hop)
    spec = stft(mono[:, None], size, hop)[:, :, 0]
    # Pipeline normally supplies equal-length phrases; support different lengths
    # explicitly with trajectory interpolation, not rounded nearest-frame jumps.
    positions = np.linspace(0, len(response) - 1, len(spec))
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(response) - 1)
    fraction = (positions - lower)[:, None, None]
    response = response[lower] * (1 - fraction) + response[upper] * fraction
    ref_times = np.linspace(0, cues.times[-1], len(spec))
    base_level = float(np.median(cues.level_db))
    gain = np.clip(10 ** (base_level / 20) / max(rms(mono), 1e-8), 0.01, 30)
    level = np.interp(ref_times, cues.times, cues.level_db)
    envelope = 10 ** (np.clip(level - base_level, -18, 12) * level_strength / 20)
    hz = np.fft.rfftfreq(size, 1 / rate)
    colors = np.stack([np.interp(ref_times, cues.times, col) for col in cues.color_db.T], axis=1)
    coloration = np.stack(
        [10 ** (np.interp(hz, cues.band_hz, row) * color_strength / 20) for row in colors]
    )
    spectrum = (
        spec[:, :, None] * response * coloration[:, :, None] * (envelope * gain)[:, None, None]
    )
    output = istft(spectrum, size, hop, len(mono))
    fade = min(round(rate * 0.008), len(output) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade)[:, None]
        output[:fade] *= ramp
        output[-fade:] *= ramp[::-1]
    if not np.isfinite(output).all():
        raise ValueError("RTF 渲染出现非有限采样，未写出音频。")
    return output.astype(np.float32), report
