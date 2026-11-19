"""Make a synthesized line sit in the recording like the voice it replaces.

All steps are driven by the separated original voice:

* ``match_spectrum``          - long-term spectrum & proximity slope towards original
* ``extend_bandwidth``        - rebuild the air band that a 22 kHz TTS model omits
* ``whisperize``              - harmonic-selective physiological breath conversion
* ``process_voice_match_fused`` - single-pass complex STFT fusion of all three steps
* ``noise_floor``             - the recording's own room noise, to refill windows

None of this copies phonemes: only smoothed envelopes and transient protections apply.
"""

from __future__ import annotations

import numpy as np

_SIZE = 2048
_HOP = 512
# Incoherent random-phase frames overlapped 4x with sqrt(Hann) cancel by ~1/sqrt(1.91).
_INCOHERENT_OLA_GAIN = 1.38
# The binaural RTF stage that follows smooths ~1.4 dB off the rebuilt band; across the
# 67 evaluation lines a +2 dB rebuild left 8-12 kHz within 0.5 dB of the original voice
# (16 kHz references: -2.1 -> -0.1 dB) and lifted 12-20 kHz from -3.7 to -1.7 dB.
_EXTENSION_GAIN = 10 ** (2.0 / 20)


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


def _spectral_eq_gain(
    own_raw: np.ndarray,
    target_spectrum: np.ndarray,
    rate: int,
    *,
    strength: float,
    max_db: float = 9.0,
    sentence_spectrum: np.ndarray | None = None,
) -> np.ndarray:
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    own = _third_octave_smooth(own_raw, rate)
    target = _third_octave_smooth(target_spectrum, rate)
    difference = 10 * np.log10(np.maximum(target, 1e-30) / np.maximum(own, 1e-30))
    core = (hz >= 200) & (hz <= 4000)
    difference -= np.mean(difference[core])

    # Proximity compensation (60-260 Hz relative to 400-2000 Hz): close-ear whispers
    # and soft speech carry warm binaural proximity bass that whole-track averaging
    # otherwise dilutes by ~80%.
    if sentence_spectrum is not None:
        sent_smooth = _third_octave_smooth(sentence_spectrum, rate)
        mid_mask = (hz >= 400) & (hz <= 2000)
        low_mask = (hz >= 60) & (hz <= 260)
        if np.any(mid_mask) and np.any(low_mask):
            sent_mid = float(np.mean(sent_smooth[mid_mask]))
            own_mid = float(np.mean(own[mid_mask]))
            if sent_mid > 1e-18 and own_mid > 1e-18:
                sent_rel_db = 10 * np.log10(np.maximum(sent_smooth[low_mask], 1e-30) / sent_mid)
                own_rel_db = 10 * np.log10(np.maximum(own[low_mask], 1e-30) / own_mid)
                prox_delta = np.clip(sent_rel_db - own_rel_db, -6.0, 7.5)
                t_prox = np.clip((hz[low_mask] - 140.0) / (260.0 - 140.0), 0.0, 1.0)
                blend = 0.85 * 0.5 * (1.0 + np.cos(np.pi * t_prox))
                difference[low_mask] = (1.0 - blend) * difference[low_mask] + blend * prox_delta

    limit = min(bandwidth_hz(own_raw, rate), bandwidth_hz(target_spectrum, rate))
    min_hz = 60.0 if sentence_spectrum is not None else 80.0
    if limit <= min_hz:
        difference[:] = 0.0
    else:
        t_lo = np.clip((hz - min_hz) / max(min_hz * 0.5, 1.0), 0.0, 1.0)
        win_lo = 0.5 * (1.0 - np.cos(np.pi * t_lo))
        hi_start = limit * 0.85
        t_hi = np.clip((limit - hz) / max(limit - hi_start, 1.0), 0.0, 1.0)
        win_hi = 0.5 * (1.0 - np.cos(np.pi * t_hi))
        difference *= win_lo * win_hi
    return 10 ** (np.clip(difference * strength, -max_db, max_db) / 20)


def match_spectrum(
    dub: np.ndarray,
    rate: int,
    target_spectrum: np.ndarray,
    *,
    strength: float = 0.8,
    max_db: float = 9.0,
) -> np.ndarray:
    """Equalize ``dub`` so its long-term spectrum follows ``target_spectrum``."""
    dub = np.asarray(dub, dtype=np.float32)
    if not len(dub) or strength <= 0:
        return dub
    spectra = _frames(dub)
    power = np.abs(spectra) ** 2
    level = power.sum(axis=1)
    own_raw = (
        power[level >= level.max() * 1e-3].mean(axis=0) + 1e-20
        if len(level) and level.max() > 0
        else np.full(_SIZE // 2 + 1, 1e-20)
    )
    gain = _spectral_eq_gain(own_raw, target_spectrum, rate, strength=strength, max_db=max_db)
    return _overlap_add(spectra * gain, len(dub)).astype(np.float32)


def _extend_spectra_highband(
    spectra: np.ndarray,
    own_raw: np.ndarray,
    target_spectrum: np.ndarray,
    rate: int,
    *,
    strength: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    cutoff = bandwidth_hz(own_raw, rate, 25.0)
    ceiling = min(bandwidth_hz(target_spectrum, rate), rate * 0.46)
    if cutoff < 3000 or ceiling < cutoff * 1.15 or strength <= 0:
        return spectra
    own = _third_octave_smooth(own_raw, rate)
    target = _third_octave_smooth(target_spectrum, rate)
    edge = int(np.searchsorted(hz, cutoff * 0.92))
    width = edge - int(np.searchsorted(hz, cutoff * 0.55))
    top = int(np.searchsorted(hz, ceiling))
    if width < 8 or top <= edge:
        return spectra
    anchor = slice(edge - width, edge - width // 3)
    offset = np.mean(own[anchor]) / max(float(np.mean(target[anchor])), 1e-30)
    random = np.random.default_rng(seed)
    result = spectra.copy()
    bins = np.arange(edge, top)
    source_bins = edge - width + (bins - edge) % width
    frame_level = np.sum(np.abs(spectra) ** 2, axis=1)
    max_level = float(frame_level.max(initial=0.0))
    active = frame_level >= max_level * 1e-3 if max_level > 0 else np.ones(len(spectra), dtype=bool)
    if not np.any(active):
        return spectra
    magnitude = np.abs(spectra[:, source_bins])
    wanted = np.sqrt(np.maximum(target[bins] * offset, 0.0))
    have = np.sqrt(np.mean(magnitude[active] ** 2, axis=0)) + 1e-12
    phase = np.exp(2j * np.pi * random.random(magnitude.shape))
    fade = np.clip((bins - edge) / max(1, width // 8), 0, 1)
    patch = magnitude * (wanted / have) * phase * (strength * _INCOHERENT_OLA_GAIN)
    result[:, bins] = spectra[:, bins] * (1 - fade) + patch * fade
    return result


def extend_bandwidth(
    dub: np.ndarray,
    rate: int,
    target_spectrum: np.ndarray,
    *,
    strength: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Rebuild the band above the TTS model's cutoff from the band just below it."""
    dub = np.asarray(dub, dtype=np.float32)
    if not len(dub) or strength <= 0:
        return dub
    spectra = _frames(dub)
    power = np.abs(spectra) ** 2
    level = power.sum(axis=1)
    own_raw = (
        power[level >= level.max() * 1e-3].mean(axis=0) + 1e-20
        if len(level) and level.max() > 0
        else np.full(_SIZE // 2 + 1, 1e-20)
    )
    extended = _extend_spectra_highband(
        spectra, own_raw, target_spectrum, rate, strength=strength, seed=seed
    )
    if extended is spectra:
        return dub
    return _overlap_add(extended, len(dub)).astype(np.float32)


def voiced_ratio(signal: np.ndarray, rate: int) -> float:
    """Share of active 40 ms frames with a clear pitch. A whisper scores near zero."""
    mono = np.asarray(signal, dtype=np.float64).reshape(-1)
    step = max(1, round(rate / 16000))
    if step > 1:
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


def _whisperize_spectra(
    spectra: np.ndarray,
    rate: int,
    amount: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Harmonic-selective physiological whisper conversion in the STFT domain.

    Unvoiced consonants and sibilant transients (s, sh, x, q, ch, t, k) already
    carry high-frequency turbulent energy without vocal-fold harmonics; smearing
    them with cepstral random phase blurs articulation. We detect harmonic vowel
    frames via cepstral pitch peak + low/high energy balance and convert only the
    voiced harmonic structure into formant-shaped glottal airflow.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0 or not len(spectra):
        return spectra
    magnitude = np.abs(spectra)
    frame_power = np.sum(magnitude**2, axis=1)
    if float(frame_power.max(initial=0.0)) <= 1e-18:
        return spectra

    log_magnitude = np.log(np.maximum(magnitude, 1e-9))
    cepstrum = np.fft.irfft(log_magnitude, n=_SIZE, axis=1)
    keep = max(8, round(rate * 0.0016))
    lifter = np.zeros(_SIZE)
    lifter[:keep] = 1.0
    lifter[-keep + 1 :] = 1.0
    envelope = np.exp(np.fft.rfft(cepstrum * lifter, axis=1).real)

    hz = np.fft.rfftfreq(_SIZE, 1 / rate)
    # Detect sibilant/transient frames where >2.8 kHz energy already dominates <1.6 kHz.
    low_band = (hz >= 120.0) & (hz <= 1600.0)
    sibilant_band = (hz >= 2800.0) & (hz <= min(10_000.0, rate * 0.45))
    low_e = np.sum(magnitude[:, low_band] ** 2, axis=1) + 1e-18
    sib_e = np.sum(magnitude[:, sibilant_band] ** 2, axis=1) + 1e-18
    harmonic_gate = np.clip((1.8 - sib_e / low_e) / 1.4, 0.0, 1.0)
    frame_amount = (amount * harmonic_gate)[:, None]

    tilt = np.sqrt(1.0 / (1.0 + (320.0 / np.maximum(hz, 1.0)) ** 4))[None, :]
    random = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * random.random(envelope.shape))
    breath_mag = envelope * tilt
    # Restore each frame's power before incoherent overlap-add compensation.
    orig_rms = np.sqrt(np.mean(magnitude**2, axis=1, keepdims=True))
    breath_rms = np.sqrt(np.mean(breath_mag**2, axis=1, keepdims=True)) + 1e-12
    breath_spec = breath_mag * (orig_rms / breath_rms) * _INCOHERENT_OLA_GAIN * phase
    # Semi-whisper keeps soft sub-450 Hz glottal murmur while replacing upper harmonics.
    if amount < 0.85:
        murmur_band = np.clip((480.0 - hz) / 220.0, 0.0, 1.0)[None, :] * (1.0 - amount)
        effective = np.clip(frame_amount * (1.0 - 0.65 * murmur_band), 0.0, 1.0)
    else:
        effective = frame_amount
    return spectra * np.sqrt(1.0 - effective) + breath_spec * np.sqrt(effective)


def whisperize(dub: np.ndarray, rate: int, amount: float, *, seed: int = 0) -> np.ndarray:
    """Turn voiced speech towards a whisper while keeping sibilants crisp."""
    dub = np.asarray(dub, dtype=np.float32)
    amount = float(np.clip(amount, 0.0, 1.0))
    if not len(dub) or amount <= 0:
        return dub
    in_rms = float(np.sqrt(np.mean(dub.astype(np.float64) ** 2)))
    if in_rms <= 1e-9:
        return dub
    spectra = _frames(dub)
    whispered = _whisperize_spectra(spectra, rate, amount, seed=seed)
    output = _overlap_add(whispered, len(dub))
    out_rms = float(np.sqrt(np.mean(output**2)))
    if out_rms > 1e-9:
        output *= in_rms / out_rms
    return output.astype(np.float32)


def process_voice_match_fused(
    dub: np.ndarray,
    rate: int,
    target_spectrum: np.ndarray,
    *,
    sentence_spectrum: np.ndarray | None = None,
    match_strength: float = 0.8,
    extend: bool = True,
    whisper_amount: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Single-STFT pipeline fusing whisperize, proximity/timbre EQ, and air extension."""
    dub = np.asarray(dub, dtype=np.float32)
    if not len(dub):
        return dub
    in_rms = float(np.sqrt(np.mean(dub.astype(np.float64) ** 2)))
    if in_rms <= 1e-8:
        return dub
    if whisper_amount <= 0.05 and match_strength <= 0.0 and not extend:
        return dub

    spectra = _frames(dub)
    if whisper_amount > 0.05:
        spectra = _whisperize_spectra(spectra, rate, whisper_amount, seed=seed)

    power = np.abs(spectra) ** 2
    level = power.sum(axis=1)
    own_raw = (
        power[level >= level.max() * 1e-3].mean(axis=0) + 1e-20
        if len(level) and level.max() > 0
        else np.full(_SIZE // 2 + 1, 1e-20)
    )
    if match_strength > 0.0:
        gain = _spectral_eq_gain(
            own_raw,
            target_spectrum,
            rate,
            strength=match_strength,
            sentence_spectrum=sentence_spectrum,
        )
        spectra = spectra * gain[None, :]
        own_raw = own_raw * (gain**2)

    if extend:
        spectra = _extend_spectra_highband(
            spectra, own_raw, target_spectrum, rate, strength=_EXTENSION_GAIN, seed=seed + 17
        )

    output = _overlap_add(spectra, len(dub))
    out_rms = float(np.sqrt(np.mean(output**2)))
    if out_rms > 1e-8:
        output *= in_rms / out_rms
    return output.astype(np.float32)


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
    measured = noise_floor(output, quantile=0.5)
    wanted = np.sqrt(np.sum(floor**2, axis=0))
    have = np.sqrt(np.sum(measured**2, axis=0))
    output *= np.where(have > 0, wanted / np.maximum(have, 1e-20), 0.0)[None, :]
    return output.astype(np.float32)
