"""The vectorised limiter and STFT must match the original loop implementations."""

import math
import time

import numpy as np
import pytest

from asmr_dubber import audio
from asmr_dubber.spatial_rtf import istft, stft


def _original_sum_with_peak_ceiling(current, addition, peak_dbfs, sample_rate):
    combined = np.asarray(current, dtype=np.float32) + np.asarray(addition, dtype=np.float32)
    if peak_dbfs is None:
        return combined
    ceiling = np.float32(10.0 ** (peak_dbfs / 20.0))
    frame_peak = np.max(np.abs(combined), axis=1)
    target = np.ones_like(frame_peak, dtype=np.float32)
    over = frame_peak > ceiling
    target[over] = ceiling / frame_peak[over]
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


def _float64_reference_envelope(target, release):
    gain = 1.0
    out = np.empty(len(target))
    for index, requested in enumerate(target.astype(np.float64)):
        gain = requested if requested < gain else release * gain + (1 - release) * requested
        out[index] = gain
    return out


def _signals(seed, frames, loud_probability):
    rng = np.random.default_rng(seed)
    loud = np.where(rng.random(frames) < loud_probability, 3.0, 0.2)[:, None]
    current = (rng.standard_normal((frames, 2)) * loud).astype(np.float32)
    addition = (rng.standard_normal((frames, 2)) * 0.3).astype(np.float32)
    return current, addition


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("probability", [0.0, 0.001, 0.05, 0.6])
def test_limiter_matches_original_loop(seed, probability):
    current, addition = _signals(seed, 30000, probability)
    expected = _original_sum_with_peak_ceiling(current, addition, -1.0, 48000)
    actual = audio._sum_with_peak_ceiling(current, addition, -1.0, 48000)
    assert actual.shape == expected.shape and actual.dtype == np.float32
    # The original accumulates float32 rounding in its recursion (up to ~4e-6
    # against exact arithmetic); the vectorised version is exact in float64.
    assert np.max(np.abs(actual - expected)) < 1e-5
    limit = 10 ** (-1.0 / 20)
    assert np.max(np.abs(actual)) <= limit * (1 + 1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_limiter_envelope_is_exact_against_float64_loop(seed):
    rng = np.random.default_rng(seed)
    target = np.where(rng.random(20000) < 0.02, rng.uniform(0.1, 0.99, 20000), 1.0)
    release = math.exp(-1.0 / 2400)
    assert (
        np.max(
            np.abs(
                audio._limiter_envelope(target, release)
                - _float64_reference_envelope(target, release)
            )
        )
        < 1e-9
    )


def test_limiter_without_ceiling_and_empty_input():
    current, addition = _signals(1, 100, 0.5)
    assert np.array_equal(
        audio._sum_with_peak_ceiling(current, addition, None, 48000), current + addition
    )
    empty = np.zeros((0, 2), dtype=np.float32)
    assert audio._sum_with_peak_ceiling(empty, empty, -1.0, 48000).shape == (0, 2)


def test_limiter_is_much_faster_than_the_loop():
    current, addition = _signals(7, 400000, 0.02)
    start = time.perf_counter()
    _original_sum_with_peak_ceiling(current, addition, -1.0, 48000)
    slow = time.perf_counter() - start
    start = time.perf_counter()
    audio._sum_with_peak_ceiling(current, addition, -1.0, 48000)
    fast = time.perf_counter() - start
    assert fast < slow


def _original_stft(audio_in, size, hop):
    padded = np.pad(audio_in, ((size // 2, size), (0, 0)))
    starts = range(0, len(padded) - size + 1, hop)
    window = np.sqrt(np.hanning(size))
    return np.stack([np.fft.rfft(padded[i : i + size] * window[:, None], axis=0) for i in starts])


def _original_istft(spectra, size, hop, length):
    window = np.sqrt(np.hanning(size))
    output = np.zeros((hop * (len(spectra) - 1) + size, 2))
    weights = np.zeros(len(output))
    for i, frame in enumerate(np.fft.irfft(spectra, n=size, axis=1)):
        start = i * hop
        output[start : start + size] += frame * window[:, None]
        weights[start : start + size] += window**2
    output /= np.maximum(weights[:, None], 1e-12)
    return output[size // 2 : size // 2 + length]


@pytest.mark.parametrize(
    ("frames", "channels", "size", "hop"),
    [
        (20000, 2, 1024, 128),
        (5000, 1, 512, 64),
        (777, 2, 256, 64),
        (20000, 2, 1024, 300),
        (3, 2, 256, 64),
    ],
)
def test_stft_and_istft_match_original(frames, channels, size, hop):
    signal = np.random.default_rng(frames).standard_normal((frames, channels))
    spectra = stft(signal, size, hop)
    reference = _original_stft(signal, size, hop)
    assert spectra.shape == reference.shape
    assert np.max(np.abs(spectra - reference)) < 1e-9
    if channels == 2:
        rebuilt = istft(spectra * 0.7, size, hop, frames)
        assert np.max(np.abs(rebuilt - _original_istft(reference * 0.7, size, hop, frames))) < 1e-9
    else:
        mono = istft(spectra, size, hop, frames)
        assert mono.shape == (frames, 2)


def test_stft_is_faster_than_the_comprehension():
    signal = np.random.default_rng(3).standard_normal((300000, 2))
    start = time.perf_counter()
    _original_stft(signal, 4096, 512)
    slow = time.perf_counter() - start
    start = time.perf_counter()
    stft(signal, 4096, 512)
    fast = time.perf_counter() - start
    print(f"stft speedup {slow / fast:.2f}x")
    assert fast < slow * 1.5
