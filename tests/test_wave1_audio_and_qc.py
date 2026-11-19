from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.dub_qc import (
    character_error_rate,
    detect_uncovered_vocal_sounds,
    dub_asr_problem,
    meaningful_japanese_residual,
    normalize_for_cer,
    scan_residual_japanese_windows,
    snap_sentences_to_silence,
)
from asmr_dubber.indextts2_worker import (
    _edge_silence_seconds,
    _infer_kwargs,
    _install_runtime_hooks,
    _load_tasks,
    _parser,
    _spoken_duration_seconds,
)
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence
from asmr_dubber.time_stretch import stretch_speech
from asmr_dubber.timing import plan_dubbing_timing
from asmr_dubber.voice_match import (
    bandwidth_hz,
    long_term_spectrum,
    process_voice_match_fused,
    voiced_ratio,
    whisperize,
)
from asmr_dubber.voice_reference import (
    annotate_sentence_voice_modes,
    effective_index_emotion_weight,
    effective_project_reference_sentence,
    pick_reference_sentence,
    pick_whisper_reference_sentence,
)


def _harmonic_wave(rate: int, seconds: float, f0: float = 190.0) -> np.ndarray:
    t = np.arange(round(rate * seconds), dtype=np.float64) / rate
    sig = np.zeros_like(t)
    for k in range(1, 14):
        sig += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
    sig /= max(float(np.max(np.abs(sig))), 1e-9)
    return (sig * 0.35).astype(np.float32)


def _whisper_wave(rate: int, seconds: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(round(rate * seconds))
    # Band-shape around 1.2-6 kHz to resemble close-mic whisper airflow.
    spec = np.fft.rfft(noise)
    hz = np.fft.rfftfreq(len(noise), 1.0 / rate)
    band = np.exp(-(((hz - 2600.0) / 1800.0) ** 2)) + 0.35 * np.exp(
        -(((hz - 5200.0) / 2200.0) ** 2)
    )
    shaped = np.fft.irfft(spec * band, n=len(noise))
    shaped /= max(float(np.max(np.abs(shaped))), 1e-9)
    return (shaped * 0.22).astype(np.float32)


def test_stretch_speech_preserves_pitch_rms_and_stereo_shape() -> None:
    rate = 24_000
    mono = _harmonic_wave(rate, 1.2, f0=200.0)
    faster = stretch_speech(mono, rate, 1.20, seed=3)
    slower = stretch_speech(mono, rate, 0.90, seed=3)

    assert abs(len(faster) - round(len(mono) / 1.20)) <= 2
    assert abs(len(slower) - round(len(mono) / 0.90)) <= 2
    assert voiced_ratio(faster, rate) > 0.85
    assert voiced_ratio(slower, rate) > 0.85

    in_rms = float(np.sqrt(np.mean(mono**2)))
    out_rms = float(np.sqrt(np.mean(faster**2)))
    assert abs(out_rms - in_rms) / in_rms < 0.08

    stereo = np.column_stack([mono, mono * 0.4])
    stretched_stereo = stretch_speech(stereo, rate, 1.15)
    assert stretched_stereo.shape == (round(len(mono) / 1.15), 2)
    assert np.allclose(stretch_speech(mono, rate, 1.0), mono)
    assert stretch_speech(np.zeros(0, dtype=np.float32), rate, 1.2).size == 0

    with pytest.raises(ValueError, match="speed_factor"):
        stretch_speech(mono, rate, 0.0)


def test_plan_dubbing_timing_respects_kept_sound_obstacles_and_min_speed() -> None:
    s1 = Sentence(
        id="s000001",
        start_seconds=1.0,
        end_seconds=2.0,
        source_text="ねえ",
        zh_text="呐",
        tts_duration_seconds=1.5,
    )
    s2 = Sentence(
        id="s000002",
        start_seconds=3.5,
        end_seconds=5.0,
        source_text="おやすみ",
        zh_text="晚安",
        tts_duration_seconds=1.1,
    )
    # Kept kiss between 2.2s and 2.9s plus leading breath 0.4s..0.95s.
    planned = plan_dubbing_timing(
        [s1, s2],
        offset_ms=-150,
        max_auto_speed=1.35,
        min_auto_speed=0.90,
        mode="fit_then_shift",
        obstacles=[(0.40, 0.95), (2.20, 2.90)],
    )
    assert len(planned) == 2
    first, second = planned
    # Negative offset (-150 ms -> 0.85s) is clamped to the end of the leading breath (0.95s).
    assert first.start_seconds == pytest.approx(0.95)
    # First dub (1.5s) speeds up so it finishes right at the 2.20s kept kiss obstacle!
    assert first.start_seconds + first.effective_duration_seconds <= 2.20 + 1e-6
    assert first.speed_factor == pytest.approx(1.5 / (2.20 - 0.95))
    # Trailing unconstrained line can gently slow toward min_auto_speed=0.90 to fill its window.
    assert 0.90 <= second.speed_factor <= 1.0


def test_voice_match_fused_and_harmonic_selective_whisperize() -> None:
    rate = 48_000
    vowel = _harmonic_wave(rate, 0.7, f0=210.0)
    # Append a crisp high-frequency sibilant burst (5-9 kHz) to verify transient protection.
    rng = np.random.default_rng(11)
    sib_raw = rng.standard_normal(round(rate * 0.25))
    sib_spec = np.fft.rfft(sib_raw)
    hz = np.fft.rfftfreq(len(sib_raw), 1.0 / rate)
    sib = np.fft.irfft(sib_spec * ((hz >= 4500) & (hz <= 9000)), n=len(sib_raw)).astype(np.float32)
    sib = sib / max(float(np.max(np.abs(sib))), 1e-9) * 0.25
    dub = np.concatenate([vowel, sib])

    # Band-limit dub below 10.5 kHz like 22.05 kHz TTS, while target has full 18 kHz air + bass.
    dub_spec = np.fft.rfft(dub)
    dub_hz = np.fft.rfftfreq(len(dub), 1.0 / rate)
    dub_22k = np.fft.irfft(dub_spec * (dub_hz <= 10_500), n=len(dub)).astype(np.float32)

    whispered = whisperize(vowel, rate, 0.92, seed=5)
    assert voiced_ratio(vowel, rate) > 0.85
    assert voiced_ratio(whispered, rate) < 0.20

    target_noise = _whisper_wave(rate, 1.0, seed=19)
    # Add strong close-ear proximity bass (110 Hz) + 16 kHz air to target.
    t = np.arange(len(target_noise), dtype=np.float64) / rate
    target_sig = (
        target_noise
        + 0.35 * np.sin(2 * np.pi * 110.0 * t)
        + 0.08 * np.sin(2 * np.pi * 14_500.0 * t)
    ).astype(np.float32)
    target_spec = long_term_spectrum(target_sig)

    fused = process_voice_match_fused(
        dub_22k,
        rate,
        target_spec,
        sentence_spectrum=target_spec,
        match_strength=0.85,
        extend=True,
        whisper_amount=0.75,
        seed=42,
    )
    assert len(fused) == len(dub_22k)
    assert bandwidth_hz(long_term_spectrum(fused), rate) > 12_500.0


def test_rebuilt_air_band_stays_close_to_the_original_voice() -> None:
    """16 kHz references leave nothing above 8 kHz; the rebuild must not leave the dub dull."""
    rate = 48_000
    noise = np.random.default_rng(3).standard_normal(rate * 3).astype(np.float32) * 0.05
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(len(noise), 1 / rate) > 8000] = 0
    band_limited = np.fft.irfft(spectrum, len(noise)).astype(np.float32)

    def air_db(x: np.ndarray) -> float:
        power = long_term_spectrum(x)
        hz = np.fft.rfftfreq(2048, 1 / rate)
        air = power[(hz >= 8500) & (hz <= 12_000)].mean()
        return float(10 * np.log10(air / power[(hz >= 1000) & (hz <= 4000)].mean()))

    rebuilt = process_voice_match_fused(
        band_limited, rate, long_term_spectrum(noise), match_strength=0.0, extend=True, seed=1
    )
    assert air_db(band_limited) < air_db(noise) - 40
    assert air_db(rebuilt) == pytest.approx(air_db(noise), abs=2.0)


def test_dual_modal_and_whisper_voice_anchor_selection(tmp_path: Path) -> None:
    rate = 24_000
    silence = np.zeros((round(rate * 1.0), 2), dtype=np.float32) + 1e-4
    modal = np.column_stack([_harmonic_wave(rate, 5.5, 205.0), _harmonic_wave(rate, 5.5, 205.0)])
    # Single-ear whisper (left channel active, right near floor) to test ear-aware classification.
    whisp_mono = _whisper_wave(rate, 5.0, seed=21)
    whisper_clip = np.column_stack([whisp_mono, np.zeros_like(whisp_mono) + 1e-4])
    full = np.concatenate([silence, modal, silence, whisper_clip, silence], axis=0)
    wav_path = tmp_path / "vocals.wav"
    sf.write(wav_path, full, rate)

    project = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="0" * 64,
            duration_seconds=13.5,
            sample_rate=rate,
            channels=2,
        ),
        settings=ProjectSettings(tts_dual_voice_anchor=True, tts_index_emo_alpha=0.55),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=1.0,
                end_seconds=6.5,
                source_text="今日も一日お疲れ様でした",
                zh_text="今天也辛苦你啦",
            ),
            Sentence(
                id="s000002",
                start_seconds=7.5,
                end_seconds=12.5,
                source_text="耳元でこっそり教えてあげるね",
                zh_text="在耳边悄悄告诉你哦",
            ),
        ],
    )
    modal_pick = pick_reference_sentence(project, wav_path)
    whisper_pick = pick_whisper_reference_sentence(project, wav_path)
    assert modal_pick is not None and modal_pick.id == "s000001"
    assert whisper_pick is not None and whisper_pick.id == "s000002"

    project.settings.tts_reference_sentence_id = modal_pick.id
    project.settings.tts_whisper_reference_sentence_id = whisper_pick.id
    annotate_sentence_voice_modes(project, wav_path)
    assert project.sentences[0].voice_mode == "modal"
    assert project.sentences[1].voice_mode == "whisper"

    ref_modal, is_whisp_0 = effective_project_reference_sentence(project, project.sentences[0])
    ref_whisp, is_whisp_1 = effective_project_reference_sentence(project, project.sentences[1])
    assert (ref_modal.id, is_whisp_0) == ("s000001", False)
    assert (ref_whisp.id, is_whisp_1) == ("s000002", True)
    assert effective_index_emotion_weight(project, project.sentences[0]) == pytest.approx(0.55)
    assert effective_index_emotion_weight(project, project.sentences[1]) == pytest.approx(0.85)


def test_dub_qc_cer_normalization_b12_and_b13_residual_scan(tmp_path: Path) -> None:
    assert normalize_for_cer("「聽風聲，12次！」", "zh") == "听风声一二次"
    assert character_error_rate("今天辛苦你啦！", "今天辛苦你啦", "zh") == 0.0
    assert character_error_rate("Good night, sweet dreams!", "good night sweet dreams", "en") == 0.0

    dummy_wav = tmp_path / "dub.wav"
    sf.write(dummy_wav, _harmonic_wave(16_000, 0.8), 16_000)
    settings = ProjectSettings()

    with patch("asmr_dubber.dub_qc.transcribe_dub_clip", return_value="完全念错别的句子"):
        warn = dub_asr_problem(dummy_wav, "今天也辛苦你啦", settings, is_final_attempt=True)
        assert warn is not None and "字错率" in warn

    with patch("asmr_dubber.dub_qc.transcribe_dub_clip", return_value="今天也辛苦你啦"):
        assert dub_asr_problem(dummy_wav, "今天也辛苦你啦", settings, is_final_attempt=True) is None

    # 1-char interjections skip CER gating; missing local recognizer degrades gracefully.
    assert dub_asr_problem(dummy_wav, "嗯！", settings, is_final_attempt=True) is None
    with patch("asmr_dubber.dub_qc.transcribe_dub_clip", return_value=None):
        assert dub_asr_problem(dummy_wav, "今天也辛苦你啦", settings, is_final_attempt=True) is None

    # Homophones ("在"/"再", "甜耳朵"/"舔耳朵", "做的"/"做得") have 0% phonetic error,
    # whereas wrong words ("念错"), dropped words ("漏字"), and pure noise ("") are caught.
    assert (
        character_error_rate("舔耳朵，做得还好吗？再试一次", "甜耳朵，做的还好吗？在试一次", "zh")
        == 0.0
    )
    for bad_hyp in ("完全无关的另一句话", "今天也", ""):
        with patch("asmr_dubber.dub_qc.transcribe_dub_clip", return_value=bad_hyp):
            assert (
                dub_asr_problem(
                    dummy_wav, "今天也辛苦你啦，好好休息吧", settings, is_final_attempt=True
                )
                is not None
            )

    assert meaningful_japanese_residual("はぁ…んっ") == ""
    assert meaningful_japanese_residual("あ") == ""
    assert meaningful_japanese_residual("大好きだよ") == "大好きだよ"

    bed_wav = tmp_path / "bed.wav"
    sf.write(bed_wav, _harmonic_wave(16_000, 3.0), 16_000)
    entries = [
        {"id": "s000001", "start": 0.2, "end": 1.2, "reasons": [], "exceeds": False},
        {"id": "s000002", "start": 1.4, "end": 2.4, "reasons": [], "exceeds": False},
    ]
    with (
        patch("asmr_dubber.dub_qc.cached_model_path", return_value=tmp_path),
        patch(
            "asmr_dubber.environment.resolve_transformers_model_source",
            return_value=(str(tmp_path), None),
        ),
        patch(
            "asmr_dubber.dub_qc.resolve_transformers_model_source",
            return_value=(str(tmp_path), None),
        ),
        patch("asmr_dubber.asr._session_model", return_value="fake_pipe"),
        patch(
            "asmr_dubber.asr._run_transformers_asr_pipeline",
            return_value=[{"text": "はぁ…"}, {"text": "おやすみなさい"}],
        ),
    ):
        # Ensure test passes even in lightweight .venv without torch/transformers installed.
        import sys

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float16="fp16",
            float32="fp32",
        )
        fake_tf = SimpleNamespace(
            AutoModelForSpeechSeq2Seq=object,
            AutoProcessor=object,
            pipeline=object,
        )
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_tf}):
            ran = scan_residual_japanese_windows(bed_wav, entries, settings)
    assert ran is True
    assert entries[0]["residual_chars"] == 0
    assert entries[0]["exceeds"] is False
    assert entries[1]["residual_chars"] >= 2
    assert entries[1]["exceeds"] is True
    assert "residual_speech" in entries[1]["reasons"]


def test_snap_sentences_to_silence_and_detect_uncovered_vocal_sounds(tmp_path: Path) -> None:
    rate = 16_000
    audio = np.zeros(rate * 5, dtype=np.float32)
    # Active speech bursts at [0.50, 1.40] and [1.55, 2.40], plus uncovered kiss at [3.20, 4.00].
    audio[int(0.50 * rate) : int(1.40 * rate)] = 0.25
    audio[int(1.55 * rate) : int(2.40 * rate)] = 0.25
    audio[int(3.20 * rate) : int(4.00 * rate)] = 0.12
    wav = tmp_path / "vocals.wav"
    sf.write(wav, audio, rate)

    jittered = [
        Sentence(id="s000001", start_seconds=0.54, end_seconds=1.37, source_text="おはよう"),
        Sentence(id="s000002", start_seconds=1.52, end_seconds=2.44, source_text="おやすみ"),
    ]
    snapped = snap_sentences_to_silence(jittered, wav, outward_ms=90.0, inward_ms=50.0)
    assert snapped[0].start_seconds <= 0.50
    assert snapped[0].end_seconds >= 1.39
    assert snapped[0].end_seconds <= snapped[1].start_seconds

    with_sounds = detect_uncovered_vocal_sounds(snapped, wav, min_duration_seconds=0.50)
    assert len(with_sounds) == 3
    kiss = with_sounds[2]
    assert kiss.status == "skipped_sound"
    assert kiss.enabled is False
    assert kiss.start_seconds == pytest.approx(3.20, abs=0.03)
    assert kiss.end_seconds == pytest.approx(4.00, abs=0.03)


def test_indextts2_worker_batch_loading_kwargs_and_daemon_cli(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    out = tmp_path / "out.wav"
    sf.write(ref, _harmonic_wave(22_050, 1.0), 22_050)
    batch = tmp_path / "batch.jsonl"
    batch.write_text(
        json.dumps(
            {
                "id": "t1",
                "sentence_id": "s000001",
                "text": "晚安啦",
                "voice": str(ref),
                "output": str(out),
                "target_duration_seconds": 1.25,
                "emotion_vector": [0.1] * 8,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = _load_tasks(batch)
    assert len(tasks) == 1
    kwargs = _infer_kwargs(tasks[0])
    assert kwargs["_target_duration_seconds"] == pytest.approx(1.25)
    assert len(kwargs["emo_vector"]) == 8

    args = _parser().parse_args(["--daemon", "--model-dir", str(tmp_path)])
    assert args.daemon is True

    padded_speech = np.concatenate([np.zeros(2205), _harmonic_wave(22_050, 0.8), np.zeros(2205)])
    sf.write(out, padded_speech, 22_050)
    assert _spoken_duration_seconds(out) == pytest.approx(0.80, abs=0.03)

    class _FakeRow:
        def detach(self) -> _FakeRow:
            return self

        def cpu(self) -> _FakeRow:
            return self

        def tolist(self) -> list[int]:
            return [52, 52, 10, 11, 52, 999]

    fake_codes = [_FakeRow()]
    edge = _edge_silence_seconds(fake_codes, 999)
    assert edge == pytest.approx(3 * 1.72 * 256 / 22_050, rel=1e-3)

    # Verify _install_runtime_hooks wires GPT & length_regulator without mutating third-party code.
    fake_lr = SimpleNamespace(forward=lambda x, ylens=None, *a, **kw: (x, ylens))
    fake_model = SimpleNamespace(
        stop_mel_token=999,
        gpt=SimpleNamespace(inference_speech=lambda *a, **kw: (fake_codes, "latent")),
        s2mel=SimpleNamespace(models={"length_regulator": fake_lr}),
    )
    import sys

    fake_torch = SimpleNamespace(clamp=lambda t, min=8: t)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        state = _install_runtime_hooks(fake_model)
    assert state["duration_factor"] == 1.0
