from __future__ import annotations

import numpy as np
import soundfile as sf

from asmr_dubber.audio import (
    dominant_channel_mono,
    probe_audio,
    sentence_events,
    speech_bounds,
)
from asmr_dubber.experimental_mix import (
    compose_speech_replacement,
    replacement_windows,
    vocal_weighted_reference,
)
from asmr_dubber.models import DubProject, ProjectSettings, Sentence


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _sentence(sid, start, end, **kwargs):
    return Sentence(
        id=sid,
        start_seconds=start,
        end_seconds=end,
        source_text="原文",
        zh_text="中文",
        tts_file=f"{sid}.wav",
        **kwargs,
    )


def _project(tmp_path, sentences, seconds=2, rate=16000, **settings):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros((rate * seconds, 2)), rate, subtype="FLOAT")
    return DubProject(
        source=probe_audio(path), sentences=sentences, settings=ProjectSettings(**settings)
    )


def _burst(rng, seconds, rate, amplitude=0.2):
    return rng.uniform(-amplitude, amplitude, (int(seconds * rate), 2)).astype(np.float32)


def test_compose_speech_replacement_only_touches_dubbed_windows(tmp_path):
    rate = 16000
    rng = np.random.default_rng(1)
    bed = (rng.uniform(-0.02, 0.02, (rate * 4, 2))).astype(np.float32)
    vocals = np.zeros_like(bed)
    vocals[int(1.0 * rate) : int(1.8 * rate)] = _burst(rng, 0.8, rate)
    vocals[int(2.0 * rate) : int(2.8 * rate)] = _burst(rng, 0.8, rate)
    source_path = tmp_path / "source.wav"
    vocals_path = tmp_path / "vocals.wav"
    sf.write(source_path, bed + vocals, rate, subtype="FLOAT")
    sf.write(vocals_path, vocals, rate, subtype="FLOAT")
    project = DubProject(
        source=probe_audio(source_path),
        sentences=[
            _sentence("s1", 1.0, 1.8),
            _sentence("s2", 2.0, 2.8, original_audio_enabled=True),
        ],
    )
    output = compose_speech_replacement(project, tmp_path, source_path, vocals_path)
    mixed, out_rate = sf.read(output, dtype="float32", always_2d=True)
    original, _ = sf.read(source_path, dtype="float32", always_2d=True)
    assert out_rate == rate and mixed.shape == original.shape
    # Before the window and from the kept-original sentence on: identical samples.
    assert np.array_equal(mixed[: int(0.8 * rate)], original[: int(0.8 * rate)])
    assert np.array_equal(mixed[int(2.02 * rate) :], original[int(2.02 * rate) :])
    # Inside the dubbed sentence the voice is subtracted and only the bed remains.
    inner = slice(int(1.1 * rate), int(1.7 * rate))
    assert _rms(original[inner]) > 5 * _rms(mixed[inner])
    assert np.allclose(mixed[inner], bed[inner], atol=1e-5)
    # The kept sentence still carries its voice.
    assert _rms(mixed[int(2.1 * rate) : int(2.7 * rate)]) > 0.08


def test_replacement_windows_merge_close_sentences_and_split_distant_ones(tmp_path):
    project = _project(
        tmp_path,
        [_sentence("s1", 1.0, 2.0), _sentence("s2", 2.1, 3.0), _sentence("s3", 4.0, 5.0)],
        seconds=6,
    )
    envelope = np.zeros(600)
    windows, ids = replacement_windows(project, envelope, 0.01)
    assert ids == ["s1", "s2", "s3"]
    assert len(windows) == 2
    assert abs(windows[0][0] - 0.96) < 0.02 and abs(windows[0][1] - 3.04) < 0.02
    assert abs(windows[1][0] - 3.96) < 0.02 and abs(windows[1][1] - 5.04) < 0.02


def test_replacement_windows_do_not_eat_neighbours_that_keep_original(tmp_path):
    project = _project(
        tmp_path,
        [
            _sentence("before", 0.0, 0.95, original_audio_enabled=True),
            _sentence("mid", 1.0, 2.0),
            _sentence("after", 2.05, 3.0, original_audio_enabled=True),
        ],
        seconds=4,
        separation_keep_padding_ms=100,
    )
    windows, ids = replacement_windows(project, np.zeros(400), 0.01)
    assert ids == ["mid"]
    assert len(windows) == 1
    assert windows[0][0] >= 0.95 - 1e-9
    # Snapping returns the centre of a 10 ms envelope bin, so the edge may overshoot the
    # neighbour's start by up to half a hop; the 30 ms fade covers it.
    assert windows[0][1] <= 2.05 + 0.01


def _padded_tone(path, rate=16000, lead=0.5, tone=1.0, tail=0.5):
    t = np.arange(int(tone * rate)) / rate
    body = 0.3 * np.sin(2 * np.pi * 440 * t)
    audio = np.concatenate([np.zeros(int(lead * rate)), body, np.zeros(int(tail * rate))]).astype(
        np.float32
    )
    sf.write(path, audio, rate, subtype="FLOAT")
    return audio


def test_speech_bounds_strip_silence_and_keep_all_silent_files(tmp_path):
    padded = tmp_path / "padded.wav"
    _padded_tone(padded)
    start, end = speech_bounds(padded)
    assert 0.4 < start < 0.5
    assert 1.5 < end < 1.6
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16000 * 2, dtype=np.float32), 16000, subtype="FLOAT")
    assert speech_bounds(silent) == (0.0, 2.0)


def test_dominant_channel_mono_prefers_the_louder_side():
    rng = np.random.default_rng(2)
    voice = rng.uniform(-0.3, 0.3, 48000).astype(np.float32)
    stereo = np.stack([voice, voice * 0.2], axis=1)
    assert np.array_equal(dominant_channel_mono(stereo, 48000), voice)
    swapped = stereo[:, ::-1].copy()
    assert np.array_equal(dominant_channel_mono(swapped, 48000), voice)


def test_dominant_channel_mono_aligns_delayed_channels_without_comb_loss():
    rate, delay = 48000, 24  # 0.5 ms: a plain sum would cancel 1 kHz completely
    rng = np.random.default_rng(3)
    voice = rng.normal(0, 0.1, rate * 2)
    left = np.concatenate([np.zeros(delay), voice[:-delay]])
    stereo = np.stack([left, voice], axis=1).astype(np.float32)
    mono = dominant_channel_mono(stereo, rate)
    plain = stereo.mean(axis=1)
    assert _rms(mono) > 0.95 * _rms(voice)
    assert _rms(plain) < 0.8 * _rms(voice)
    assert np.corrcoef(mono[200:-200], left[200:-200])[0, 1] > 0.99


def test_vocal_weighted_reference_silences_the_ear_without_voice(tmp_path):
    rate = 48000
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.05, rate * 3).astype(np.float32)
    voice = rng.normal(0, 0.05, rate * 3).astype(np.float32)
    mix = np.stack([noise, voice], axis=1)
    vocals = np.stack([np.zeros_like(voice), voice], axis=1)
    source, vocal_file = tmp_path / "source.wav", tmp_path / "vocals.wav"
    sf.write(source, mix, rate, subtype="FLOAT")
    sf.write(vocal_file, vocals, rate, subtype="FLOAT")
    result = vocal_weighted_reference(source, vocal_file, tmp_path)
    weighted, out_rate = sf.read(result, dtype="float32", always_2d=True)
    assert out_rate == rate and weighted.shape[1] == 2
    inner = slice(rate // 2, -rate // 2)
    assert _rms(weighted[inner, 0]) < 0.1 * _rms(mix[inner, 0])
    assert _rms(weighted[inner, 1]) > 0.8 * _rms(mix[inner, 1])


def test_sentence_events_trim_silence_uses_trimmed_duration(tmp_path):
    _padded_tone(tmp_path / "s1.wav")
    sentence = _sentence("s1", 0.0, 3.0, tts_duration_seconds=2.0)
    plain = sentence_events(tmp_path, [sentence])[0]
    assert plain.trim_start_seconds == 0.0 and plain.trim_end_seconds is None
    trimmed = sentence_events(tmp_path, [sentence], trim_silence=True)[0]
    assert 0.4 < trimmed.trim_start_seconds < 0.5
    assert trimmed.trim_end_seconds is not None and 1.5 < trimmed.trim_end_seconds < 1.6
    expected = trimmed.trim_end_seconds - trimmed.trim_start_seconds
    assert abs(trimmed.effective_duration_seconds * trimmed.speed_factor - expected) < 1e-6
    assert trimmed.effective_duration_seconds < plain.effective_duration_seconds
