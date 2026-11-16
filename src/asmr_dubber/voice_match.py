"""Make a synthesized line sit in the recording like the voice it replaces.

Three independent steps, all driven by the separated original voice:

* ``match_spectrum``   - long-term spectrum (timbre, proximity) towards the original
* ``extend_bandwidth`` - rebuild the air band that a 22 kHz TTS model cannot produce
* ``noise_floor``      - the recording's own room noise, to refill replaced windows

None of this copies phonemes: only 1/3-octave long-term averages are transferred.
"""

from __future__ import annotations

import numpy as np

_SIZE = 2048
_HOP = 512


def _frames(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    padded = np.pad(signal, (_SIZE // 2, _SIZE))
    count = 1 + (len(padded) - _SIZE) // _HOP
    index = np.arange(_SIZE)[None, :] + _HOP * np.arange(count)[:, None]
    return np.fft.rfft(padded[index] * np.sqrt(np.hanning(_SIZE)), axis=1)


def _overlap_add(spectra: np.ndarray, length: int) -> np.ndarray:
    window = np.sqrt(np.hanning(_SIZE))
    frames = np.fft.irfft(spectra, n=_SIZE, axis=1) * window
    output = np.zeros(_HOP * (len(frames) - 1) + _SIZE)
    weight = np.zeros_like(output)
    for i, frame in enumerate(frames):
        output[i * _HOP : i * _HOP + _SIZE] += frame
        weight[i * _HOP : i * _HOP + _SIZE] += window**2
    output /= np.maximum(weight, 1e-9)
    return output[_SIZE // 2 : _SIZE // 2 + length]


def long_term_spectrum(signal: np.ndarray) -> np.ndarray:
    """Mean power per FFT bin over the active (non-silent) frames."""
    power = np.abs(_frames(signal)) ** 2
    level = power.sum(axis=1)
    if not len(level) or level.max() <= 0:
        return np.full(_SIZE // 2 + 1, 1e-20)
    active = level >= level.max() * 1e-3
    return power[active].mean(axis=0) + 1e-20


def _third_octave_smooth(power: np.ndarray, rate: int) -> np.ndarray:
    """Smooth a power spectrum over a sliding 1/3-octave window."""
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    lower = np.searchsorted(hz, hz / 2 ** (1 / 6), side="left")
    upper = np.maximum(np.searchsorted(hz, hz * 2 ** (1 / 6), side="right"), lower + 1)
    return (cumulative[upper] - cumulative[lower]) / (upper - lower)


def bandwidth_hz(spectrum: np.ndarray, rate: int, drop_db: float = 40.0) -> float:
    """Highest frequency still within ``drop_db`` of the 1-4 kHz speech band."""
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    smooth = _third_octave_smooth(spectrum, rate)
    reference = smooth[(hz >= 1000) & (hz <= 4000)].mean()
    present = np.flatnonzero(smooth >= reference * 10 ** (-drop_db / 10))
    return float(hz[present[-1]]) if len(present) else 0.0


def match_spectrum(
    dub: np.ndarray,
    rate: int,
    target_spectrum: np.ndarray,
    *,
    strength: float = 0.8,
    max_db: float = 9.0,
) -> np.ndarray:
    """Equalize ``dub`` so its long-term spectrum follows ``target_spectrum``.

    The correction is 1/3-octave smoothed, level-neutral over 200 Hz-4 kHz, limited
    to ``max_db`` and never tries to boost a band the dub does not contain.
    """
    dub = np.asarray(dub, dtype=np.float32)
    if not len(dub) or strength <= 0:
        return dub
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    own = _third_octave_smooth(long_term_spectrum(dub), rate)
    target = _third_octave_smooth(target_spectrum, rate)
    difference = 10 * np.log10(target / own)
    core = (hz >= 200) & (hz <= 4000)
    difference -= np.mean(difference[core])
    limit = min(bandwidth_hz(long_term_spectrum(dub), rate), bandwidth_hz(target_spectrum, rate))
    difference[(hz < 80) | (hz > limit)] = 0
    gain = 10 ** (np.clip(difference * strength, -max_db, max_db) / 20)
    return _overlap_add(_frames(dub) * gain, len(dub)).astype(np.float32)


def extend_bandwidth(
    dub: np.ndarray,
    rate: int,
    target_spectrum: np.ndarray,
    *,
    strength: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Rebuild the band above the TTS model's cutoff from the band just below it.

    Each missing bin takes the frame-by-frame magnitude of a bin one "fold" lower
    (so breaths and sibilants keep their timing), a random phase (that band of a
    voice is noise-like), and a level that continues the original voice's own
    high-frequency slope.  Nothing is added when the dub already reaches as high
    as the original.
    """
    dub = np.asarray(dub, dtype=np.float32)
    if not len(dub) or strength <= 0:
        return dub
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    own_raw = long_term_spectrum(dub)
    cutoff = bandwidth_hz(own_raw, rate, 25.0)
    ceiling = min(bandwidth_hz(target_spectrum, rate), rate * 0.46)
    if cutoff < 3000 or ceiling < cutoff * 1.15:
        return dub
    spectra = _frames(dub)
    own = _third_octave_smooth(own_raw, rate)
    target = _third_octave_smooth(target_spectrum, rate)
    edge = int(np.searchsorted(hz, cutoff * 0.92))
    width = edge - int(np.searchsorted(hz, cutoff * 0.55))
    top = int(np.searchsorted(hz, ceiling))
    if width < 8 or top <= edge:
        return dub
    # Anchor the target slope to what the dub really has just below its cutoff.
    anchor = slice(edge - width, edge - width // 3)
    offset = np.mean(own[anchor]) / np.mean(target[anchor])
    random = np.random.default_rng(seed)
    result = spectra.copy()
    bins = np.arange(edge, top)
    source_bins = edge - width + (bins - edge) % width
    magnitude = np.abs(spectra[:, source_bins])
    wanted = np.sqrt(target[bins] * offset)
    have = np.sqrt(np.mean(magnitude**2, axis=0)) + 1e-12
    phase = np.exp(2j * np.pi * random.random(magnitude.shape))
    fade = np.clip((bins - edge) / max(1, width // 8), 0, 1)
    patch = magnitude * (wanted / have) * phase * strength
    result[:, bins] = spectra[:, bins] * (1 - fade) + patch * fade
    output = _overlap_add(result, len(dub))
    # Random phases in overlapping frames partly cancel; measure and correct once.
    solid = bins[fade >= 1]
    if len(solid):
        achieved = long_term_spectrum(output)[solid].mean()
        correction = float(np.sqrt(np.mean(wanted[fade >= 1] ** 2) * strength**2 / achieved))
        result[:, bins] = spectra[:, bins] * (1 - fade) + patch * fade * min(correction, 4.0)
        output = _overlap_add(result, len(dub))
    return output.astype(np.float32)


def voiced_ratio(signal: np.ndarray, rate: int) -> float:
    """Share of active 40 ms frames with a clear pitch.  A whisper scores near zero."""
    mono = np.asarray(signal, dtype=np.float64).reshape(-1)
    step = max(1, round(rate / 16000))
    if step > 1:  # crude decimation is enough for a 70-400 Hz pitch search
        usable = len(mono) // step * step
        mono = mono[:usable].reshape(-1, step).mean(axis=1)
    reduced = rate / step
    size, hop = round(reduced * 0.040), round(reduced * 0.010)
    if len(mono) < size:
        return 0.0
    count = 1 + (len(mono) - size) // hop
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    frames = mono[index]
    energy = np.sqrt(np.mean(frames**2, axis=1))
    active = energy > max(float(energy.max()) * 0.1, 1e-6)
    if not active.any():
        return 0.0
    frames = frames[active] - frames[active].mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames * np.hanning(size), 4 * size, axis=1)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)
    correlation /= np.maximum(correlation[:, :1], 1e-20)
    lags = slice(round(reduced / 400), round(reduced / 70))
    return float(np.mean(correlation[:, lags].max(axis=1) > 0.45))


def whisperize(dub: np.ndarray, rate: int, amount: float, *, seed: int = 0) -> np.ndarray:
    """Turn voiced speech towards a whisper by swapping the vocal-fold buzz for breath.

    Each frame keeps its spectral envelope (the formants that carry the words) but
    loses its harmonics and takes a random phase, which is what a whisper is: the
    same mouth shape excited by turbulent air.  ``amount`` blends the result with
    the untouched line.  Low frequencies, which a whisper barely has, are thinned.
    """
    dub = np.asarray(dub, dtype=np.float32)
    amount = float(np.clip(amount, 0.0, 1.0))
    if not len(dub) or amount <= 0:
        return dub
    spectra = _frames(dub)
    magnitude = np.abs(spectra)
    # Cepstral smoothing: keep quefrencies below 1.6 ms, i.e. drop pitch above ~600 Hz
    # spacing and everything finer, leaving the formant envelope.
    log_magnitude = np.log(np.maximum(magnitude, 1e-9))
    cepstrum = np.fft.irfft(log_magnitude, n=_SIZE, axis=1)
    keep = max(8, round(rate * 0.0016))
    lifter = np.zeros(_SIZE)
    lifter[:keep] = 1.0
    lifter[-keep + 1 :] = 1.0
    envelope = np.exp(np.fft.rfft(cepstrum * lifter, axis=1).real)
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    tilt = np.sqrt(1.0 / (1.0 + (350.0 / np.maximum(hz, 1.0)) ** 4))
    random = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * random.random(envelope.shape))
    breath = _overlap_add(envelope * tilt * phase, len(dub))
    # Follow the original loudness contour frame by frame (20 ms), not just overall.
    hop = max(1, rate // 50)
    count = max(1, len(dub) // hop)

    def contour(x: np.ndarray) -> np.ndarray:
        usable = x[: count * hop].astype(np.float64).reshape(count, hop)
        return np.sqrt(np.mean(usable**2, axis=1))

    if len(dub) >= hop:
        ratio = contour(dub) / np.maximum(contour(breath), 1e-9)
        positions = (np.arange(len(dub)) - hop / 2) / hop
        # Log-domain smoothing sits in the valleys between harmonics, so the
        # breath starts far quieter than the voice; the contour restores it.
        gain = np.interp(positions, np.arange(count), np.clip(ratio, 0.0, 1000.0))
        breath = breath * gain
    # The two parts are uncorrelated, so blend by power to keep the loudness.
    mixed = dub.astype(np.float64) * np.sqrt(1.0 - amount) + breath * np.sqrt(amount)
    return mixed.astype(np.float32)


def noise_floor(audio: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    """Per-channel magnitude spectrum of the quietest frames: the room and mic noise."""
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    floors = []
    for channel in data.T:
        power = np.abs(_frames(channel)) ** 2
        level = power.sum(axis=1)
        positive = level[level > 0]
        if not len(positive):
            floors.append(np.zeros(_SIZE // 2 + 1))
            continue
        threshold = np.quantile(positive, quantile)
        quiet = power[(level > 0) & (level <= threshold)]
        floors.append(np.sqrt(np.median(quiet, axis=0)))
    return np.stack(floors, axis=1)


def synthesize_floor(floor: np.ndarray, length: int, seed: int = 0) -> np.ndarray:
    """Stationary noise with the given per-channel magnitude spectrum."""
    random = np.random.default_rng(seed)
    count = 1 + (length + _SIZE) // _HOP + 1
    channels = []
    for column in floor.T:
        phase = np.exp(2j * np.pi * random.random((count, len(column))))
        channels.append(_overlap_add(column[None, :] * phase, length))
    output = np.stack(channels, axis=1)
    # Independent random phases in overlapping frames add incoherently; measure the
    # result with the same analysis and correct each channel back to the target.
    measured = noise_floor(output, quantile=0.5)
    wanted = np.sqrt(np.sum(floor**2, axis=0))
    have = np.sqrt(np.sum(measured**2, axis=0))
    output *= np.where(have > 0, wanted / np.maximum(have, 1e-20), 0.0)[None, :]
    return output.astype(np.float32)
