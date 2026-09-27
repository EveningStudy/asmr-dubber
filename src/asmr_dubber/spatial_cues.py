"""Reference-guided binaural cue transfer, not a neural renderer or 3-D tracker.

Assumes a single dominant speaker in two coherent microphone channels.
Positive ITD means the right ear receives the sound later (source toward left).
Broadband/time-smoothed features deliberately avoid copying Japanese phonemes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width) | 1)
    if width == 1:
        return x.copy()
    pad = width // 2
    padded = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    acc = np.concatenate([np.zeros_like(padded[:1]), np.cumsum(padded, axis=0)], axis=0)
    return (acc[width:] - acc[:-width]) / width


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return np.zeros_like(values)
    positions = np.arange(len(values))
    if values.ndim == 1:
        return np.interp(positions, positions[valid], values[valid])
    return np.column_stack(
        [np.interp(positions, positions[valid], column[valid]) for column in values.T]
    )


@dataclass
class Cues:
    times: np.ndarray
    band_hz: np.ndarray
    ild_db: np.ndarray
    itd_seconds: np.ndarray
    level_db: np.ndarray
    color_db: np.ndarray
    confidence: np.ndarray

    def summary(self) -> dict:
        return {
            "method": "reference-guided spectral ILD + GCC-PHAT ITD + relative level/color",
            "itd_convention": "positive = right ear delayed",
            "itd_ms_range": (
                self.itd_seconds[[np.argmin(self.itd_seconds), np.argmax(self.itd_seconds)]] * 1000
            ).tolist(),
            "median_ild_db_range": [
                float(x) for x in np.percentile(np.median(self.ild_db, axis=1), [5, 95])
            ],
            "reliable_itd_fraction": float(np.mean(self.confidence >= 0.15)),
            "distance": "relative loudness/color proxy only; no metric distance or room recovery",
        }


def analyze(reference: np.ndarray, rate: int) -> Cues:
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("空间分析必须输入双声道音频。")
    if len(reference) < rate // 10 or not np.isfinite(reference).all():
        raise ValueError("参考音频过短或包含非有限采样。")
    if rms(reference) < 1e-7:
        raise ValueError("参考音频没有可用信号。")
    hop = round(rate * 0.02)
    size = 1 << int(np.ceil(np.log2(rate * 0.085)))
    centers = np.arange(0, len(reference), hop)
    padded = np.pad(reference, ((size // 2, size // 2), (0, 0)))
    window = np.hanning(size)
    hz = np.fft.rfftfreq(size, 1 / rate)
    bands = np.geomspace(100, min(18000, rate * 0.44), 28)
    masks = [(hz >= f / 1.22) & (hz <= f * 1.22) for f in bands]
    low_band = (hz >= 180) & (hz <= min(6500, rate * 0.4))
    maxlag = round(rate * 0.0009)
    ild, delays, levels, spectra, confidence = [], [], [], [], []
    for start in centers:
        frame = padded[start : start + size] * window[:, None]
        spectrum = np.fft.rfft(frame, axis=0)
        power = np.abs(spectrum) ** 2
        # Regularize bins that contain very little speech energy.
        floor = max(float(power.max()) * 1e-7, 1e-14)
        band_power = np.array([power[mask].mean(axis=0) for mask in masks]) + floor
        ild.append(np.clip(10 * np.log10(band_power[:, 0] / band_power[:, 1]), -24, 24))
        spectra.append(10 * np.log10(band_power.mean(axis=1)))
        levels.append(20 * np.log10(max(rms(frame) / np.sqrt(np.mean(window**2)), 1e-9)))
        # R * conj(L) yields positive lag when R is delayed.
        cross = spectrum[:, 1] * np.conj(spectrum[:, 0])
        cross = np.where(low_band, cross / np.maximum(np.abs(cross), floor), 0)
        corr = np.fft.irfft(cross, n=size)
        search = np.concatenate([corr[-maxlag:], corr[: maxlag + 1]])
        peak = int(np.argmax(search))
        frac = 0.0
        if 0 < peak < len(search) - 1:
            a, b, c = search[peak - 1 : peak + 2]
            denominator = a - 2 * b + c
            if abs(denominator) > 1e-12:
                frac = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
        delay = (peak - maxlag + frac) / rate
        lag = round(delay * rate)
        left, right = frame[:, 0], frame[:, 1]
        if lag >= 0:
            left, right = left[: size - lag], right[lag:]
        else:
            left, right = left[-lag:], right[: size + lag]
        coherence = float(
            np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)
        )
        delays.append(delay)
        confidence.append(max(0, coherence))
    levels = np.array(levels)
    active = levels > max(-85, float(np.percentile(levels, 90)) - 30)
    if not active.any():
        raise ValueError("参考片段电平过低，无法可靠分析空间线索；请先提高输入音量。")
    reliable = active & (np.array(confidence) >= 0.15)
    delay = _fill(np.array(delays), reliable)
    # Median rejection of isolated correlation-peak jumps, then smooth motion.
    delay = np.median(
        np.lib.stride_tricks.sliding_window_view(np.pad(delay, (2, 2), mode="edge"), 5),
        axis=1,
    )
    ild = smooth(_fill(np.array(ild), active), 9)
    level = smooth(_fill(levels, active), 19)
    spec = smooth(_fill(np.array(spectra), active), 19)
    # Remove broadband level before computing gentle, relative coloration.
    spec -= np.mean(spec, axis=1, keepdims=True)
    color = np.clip(spec - np.median(spec[active], axis=0), -6, 6)
    return Cues(centers / rate, bands, ild, smooth(delay, 7), level, color, np.array(confidence))
