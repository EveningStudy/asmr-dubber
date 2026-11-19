from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import (
    audio,
    dub_qc,
    experimental_mix,
    experimental_spatial,
    separation,
    separation_download,
    spatial_rtf,
    tts,
    tts_backends,
    tts_daemon,
    tts_http_runners,
    tts_indextts,
    tts_indextts25,
    voice_match,
    voice_reference,
    zh_phonetics,
)
from asmr_dubber.audio_dsp import _fold_by_energy, dominant_channel_mono, downmix_to_stereo
from asmr_dubber.errors import OperationCancelledError, SynthesisError
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence
from asmr_dubber.timing import plan_dubbing_timing


def _audio_info(path: Path, rate: int, duration: float, channels: int = 2) -> AudioInfo:
    return AudioInfo(
        path=path.name,
        sha256="0" * 64,
        duration_seconds=duration,
        sample_rate=rate,
        channels=channels,
    )


def _tone(seconds: float, rate: int, freq: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * rate), dtype=np.float64) / rate
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_rtf_zero_phase_inversion_with_sub80hz_noise() -> None:
    rate = 48_000
    rng = np.random.default_rng(101)
    t = np.arange(rate * 2, dtype=np.float64) / rate
    voice = 0.15 * (np.sin(2 * np.pi * 240 * t) + 0.5 * np.sin(2 * np.pi * 620 * t))
    # Uncorrelated -45 dB sub-80 Hz rumble in L and R.
    noise_l = rng.normal(0, 1.0, len(t))
    noise_r = rng.normal(0, 1.0, len(t))
    hz = np.fft.rfftfreq(len(t), 1.0 / rate)
    lp = hz < 80.0
    rumble_l = np.fft.irfft(np.fft.rfft(noise_l) * lp, n=len(t))
    rumble_r = np.fft.irfft(np.fft.rfft(noise_r) * lp, n=len(t))
    rumble_scale = 0.15 * 10 ** (-45.0 / 20.0) / max(float(np.std(rumble_l)), 1e-12)
    reference = np.column_stack((voice + rumble_l * rumble_scale, voice + rumble_r * rumble_scale))

    response, _cues, report = spatial_rtf.filters(reference, rate)
    phase_diff = np.angle(response[:, :, 1] * response[:, :, 0].conj())
    near_pi = np.abs(np.abs(phase_diff) - np.pi) < 0.3
    assert float(np.mean(near_pi)) == 0.0
    assert report["residual_phase_rms_radians"] < 0.05


def test_rtf_polar_interpolation_and_comb_free_strength(tmp_path: Path) -> None:
    rate = 24_000
    rng = np.random.default_rng(7)
    sig = rng.normal(0, 0.05, rate * 2).astype(np.float32)
    # 12-sample ITD (~0.5 ms) across ears, half-amplitude on right.
    ref = np.column_stack((sig, np.pad(sig, (12, 0))[: len(sig)] * 0.6))
    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, ref, rate, subtype="FLOAT")

    # Different mono length exercises polar interpolation across time.
    mono = _tone(1.2, rate, freq=1200.0, amp=0.15)
    out_polar, _ = spatial_rtf.render_rtf(mono, ref, rate, strength=0.5)
    assert out_polar.shape == (len(mono), 2)
    assert np.isfinite(out_polar).all()

    # Broadband mono with strength=0.5 must NOT exhibit comb notches at 1/(2*ITD) ~ 1 kHz.
    broad_mono = rng.normal(0, 0.05, rate).astype(np.float32)
    settings = ProjectSettings(spatial_rtf_enabled=True, spatial_rtf_strength=0.5)
    event = experimental_mix.spatialize_clip(
        broad_mono,
        rate,
        audio.StemEvent("s1", 0.0, ref_path, 0.0, 2.0),
        ref_path,
        settings,
    )
    spec_l = np.abs(np.fft.rfft(event[:, 0]))
    freqs = np.fft.rfftfreq(len(event), 1.0 / rate)
    notch_band = spec_l[(freqs >= 850) & (freqs <= 1150)]
    pass_band = spec_l[(freqs >= 400) & (freqs <= 700)]
    assert float(np.mean(notch_band)) > 0.4 * float(np.mean(pass_band))


def test_stereo_reference_preserves_44100_and_uses_cached_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "source.wav"
    sf.write(src, np.zeros((4410, 2), dtype=np.float32), 44_100, subtype="FLOAT")
    recorded_args: list[list[str]] = []

    def fake_ffmpeg(args: list[str]) -> None:
        recorded_args.append(args)
        sf.write(Path(args[-1]), np.zeros((4410, 2), dtype=np.float32), 44_100, subtype="FLOAT")

    monkeypatch.setattr(experimental_mix, "_run_ffmpeg", fake_ffmpeg)
    ref = experimental_spatial.stereo_reference(src, tmp_path)
    assert ref.is_file()
    ar_idx = recorded_args[0].index("-ar")
    assert recorded_args[0][ar_idx + 1] == "44100"


def test_build_chinese_stem_overshoot_and_edge_fade_after_voice_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate = 48_000
    src = tmp_path / "src.wav"
    sf.write(src, np.zeros((rate, 2), dtype=np.float32), rate, subtype="FLOAT")
    clip = tmp_path / "clip.wav"
    # Constant pulse at edge so edge fade is observable when normalize_loudness=False.
    sf.write(clip, np.full(rate // 4, 0.2, dtype=np.float32), rate, subtype="FLOAT")

    # Simulate _read_adjusted_mono returning +3000 samples more than sf.info pre-estimate.
    monkeypatch.setattr(
        audio,
        "_read_adjusted_mono",
        lambda *_a, **_k: np.full(rate // 4 + 3000, 0.2, dtype=np.float32),
    )

    events = [audio.StemEvent("s000001", 0.9, clip)]
    dest = tmp_path / "stem.wav"
    audio.build_chinese_stem(
        dest,
        events,
        _audio_info(src, rate, 1.0, channels=2),
        0.0,
        normalize_loudness=False,
        stem_peak_dbfs=None,
    )
    rendered, sr = sf.read(dest, dtype="float32", always_2d=True)
    assert sr == rate
    assert len(rendered) >= round(0.9 * rate) + rate // 4 + 3000
    start_idx = round(0.9 * rate)
    assert (
        abs(float(rendered[start_idx, 0])) < 1e-4
    )  # edge fade applied even with normalize_loudness=False


def test_voice_match_active_frame_highband_and_smooth_eq() -> None:
    rate = 48_000
    rng = np.random.default_rng(13)
    speech = rng.normal(0, 0.05, rate).astype(np.float32)
    hz_full = np.fft.rfftfreq(len(speech), 1.0 / rate)
    target_spec = voice_match.long_term_spectrum(speech)
    band_limited = np.fft.irfft(np.fft.rfft(speech) * (hz_full <= 8000.0), n=len(speech)).astype(
        np.float32
    )

    # Pad with 2 seconds of silence (67% silent frames).
    padded = np.concatenate(
        [np.zeros(rate, dtype=np.float32), band_limited, np.zeros(rate, dtype=np.float32)]
    )
    out_clean = voice_match.extend_bandwidth(band_limited, rate, target_spec, strength=1.0, seed=1)
    out_padded = voice_match.extend_bandwidth(padded, rate, target_spec, strength=1.0, seed=1)

    spec_clean = voice_match.long_term_spectrum(out_clean)
    spec_padded = voice_match.long_term_spectrum(out_padded[rate : rate * 2])
    hz = np.fft.rfftfreq(2048, 1.0 / rate)
    hi = (hz >= 9000.0) & (hz <= 15000.0)
    diff_db = 10.0 * np.log10(float(np.mean(spec_padded[hi])) / float(np.mean(spec_clean[hi])))
    assert abs(diff_db) < 1.5

    # Check smooth cosine transitions in _spectral_eq_gain (no jump > 2 dB per bin at boundaries).
    own_raw = voice_match.long_term_spectrum(band_limited)
    gain = voice_match._spectral_eq_gain(
        own_raw, target_spec * 2.0, rate, strength=1.0, sentence_spectrum=target_spec * 2.0
    )
    gain_db = 20.0 * np.log10(np.maximum(gain, 1e-6))
    assert float(np.max(np.abs(np.diff(gain_db)))) < 2.5


def test_5_1_center_channel_preservation_and_stereo_whisper_protection() -> None:
    rate = 16_000
    center_voice = _tone(0.5, rate, freq=1000.0, amp=0.3)
    surround_51 = np.zeros((len(center_voice), 6), dtype=np.float32)
    surround_51[:, 2] = center_voice  # Center channel only (FC = index 2)

    stereo = downmix_to_stereo(surround_51)
    assert stereo.shape == (len(center_voice), 2)
    assert float(np.sqrt(np.mean(stereo**2))) > 0.14

    mono_dom = dominant_channel_mono(surround_51, rate)
    assert float(np.sqrt(np.mean(mono_dom**2))) > 0.14

    folded_51 = _fold_by_energy(surround_51, 800, rate=rate)
    assert float(np.sqrt(np.mean(folded_51**2))) > 0.14

    # Stereo whisper protection: loud 60 Hz foley in Left, quiet 1 kHz whisper in Right.
    t = np.arange(rate, dtype=np.float64) / rate
    foley_l = (0.8 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)
    whisper_r = (0.04 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    folded = _fold_by_energy(np.column_stack((foley_l, whisper_r)), 800, rate=rate)
    # 1 kHz whisper component must be boosted toward 0.8 (not crushed by 60 Hz foley!).
    spec = np.abs(np.fft.rfft(folded))
    freqs = np.fft.rfftfreq(len(folded), 1.0 / rate)
    whisper_bin = spec[int(np.argmin(np.abs(freqs - 1000.0)))]
    orig_bin = np.abs(np.fft.rfft(whisper_r))[int(np.argmin(np.abs(freqs - 1000.0)))]
    assert float(whisper_bin / max(orig_bin, 1e-9)) > 0.75


def test_experimental_mix_positive_gain_no_noise_boost_150ms_gap_and_rate(tmp_path: Path) -> None:
    rate = 48_000
    duration = 2.0
    n = round(rate * duration)
    rng = np.random.default_rng(31)
    # Constant ambient noise floor + quiet breath on kept sentence s2.
    noise = rng.normal(0, 0.002, (n, 2)).astype(np.float32)
    vocals = np.zeros((n, 2), dtype=np.float32)
    vocals[round(0.2 * rate) : round(0.8 * rate)] = 0.1
    vocals[round(1.02 * rate) : round(1.08 * rate)] = 0.02  # kept breath (60 ms)
    vocals[round(1.10 * rate) : round(1.8 * rate)] = 0.1
    src = tmp_path / "source.wav"
    voc_path = tmp_path / "vocals.wav"
    bed_path = tmp_path / "bed.wav"
    sf.write(src, noise + vocals, rate, subtype="FLOAT")
    sf.write(voc_path, vocals, rate, subtype="FLOAT")
    sf.write(bed_path, noise, rate, subtype="FLOAT")

    s1 = Sentence(
        id="s1",
        start_seconds=0.2,
        end_seconds=0.8,
        source_text="a",
        zh_text="甲",
        tts_file="c1.wav",
    )
    s2 = Sentence(
        id="s2",
        start_seconds=1.02,
        end_seconds=1.08,
        source_text="はぁ",
        zh_text="",
        enabled=False,
        chinese_audio_enabled=False,
        original_audio_enabled=True,
        original_audio_gain_db=9.5,
        status="skipped_sound",
    )
    s3 = Sentence(
        id="s3",
        start_seconds=1.10,
        end_seconds=1.8,
        source_text="b",
        zh_text="乙",
        tts_file="c3.wav",
    )
    project = DubProject(
        source=_audio_info(src, rate, duration, 2),
        settings=ProjectSettings(
            separation_enabled=True,
            separation_mix_mode="replace",
            replacement_scope="dubbed_speech",
            replacement_noise_fill=True,
            replacement_residual_suppression=0.5,
            separation_keep_padding_ms=0,
        ),
        sentences=[s1, s2, s3],
    )

    env, hop = experimental_mix._vocal_envelope(voc_path)
    windows, _ = experimental_mix.replacement_windows(project, env, hop)
    # 1.02..1.08 kept breath sits in <150 ms gap -> windows must NOT merge across it.
    assert len(windows) == 2

    out = experimental_mix.compose_speech_replacement(project, tmp_path, src, voc_path)
    out_data, out_sr = sf.read(out, dtype="float32", always_2d=True)
    assert out_sr == rate
    # Kept breath s2 (+9.5 dB) scales vocals without adding negative synthetic floor noise.
    kept_slice = out_data[round(1.04 * rate) : round(1.06 * rate)]
    assert float(np.mean(kept_slice)) > 0.045

    # compose_replacement_bed preserves source sample rate (e.g. 96 kHz) even if stems are 44.1 kHz.
    proj_96 = DubProject(
        source=_audio_info(src, 96_000, 1.0, 2),
        settings=ProjectSettings(separation_enabled=True, separation_mix_mode="replace"),
        sentences=[],
    )
    bed_44 = tmp_path / "bed44.wav"
    voc_44 = tmp_path / "voc44.wav"
    sf.write(bed_44, np.zeros((44_100, 2), dtype=np.float32), 44_100, subtype="FLOAT")
    sf.write(voc_44, np.zeros((44_100, 2), dtype=np.float32), 44_100, subtype="FLOAT")
    composed = experimental_mix.compose_replacement_bed(proj_96, tmp_path, voc_44, bed_44)
    assert sf.info(composed).samplerate == 96_000


def test_timing_skips_disabled_chinese_audio_and_dodges_subsequent_obstacles() -> None:
    s1 = Sentence(
        id="s1",
        start_seconds=0.0,
        end_seconds=1.0,
        source_text="a",
        zh_text="甲",
        tts_duration_seconds=1.8,
    )
    s_muted = Sentence(
        id="s2",
        start_seconds=1.2,
        end_seconds=1.8,
        source_text="b",
        zh_text="乙",
        chinese_audio_enabled=False,
        tts_duration_seconds=1.0,
    )
    s3 = Sentence(
        id="s3",
        start_seconds=1.6,
        end_seconds=2.2,
        source_text="c",
        zh_text="丙",
        tts_duration_seconds=0.6,
    )
    # Kept breath at (2.0, 2.7) lies AFTER s3.source_start (1.6), so shifted s3 must dodge it.
    plan = plan_dubbing_timing(
        [s1, s_muted, s3],
        max_auto_speed=1.0,
        mode="sequential",
        obstacles=[(2.0, 2.7)],
    )
    assert [p.sentence_id for p in plan] == ["s1", "s3"]
    assert plan[1].start_seconds >= 2.7 - 1e-6


def test_tts_daemon_closes_worker_on_on_line_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tts_daemon._ResidentWorker(("k",), "IndexTTS2", ["python"], tmp_path, {})
    worker.ready = True

    class FakeStdin:
        def write(self, _s: str) -> int:
            return 1

        def flush(self) -> None:
            pass

    class FakeProc:
        stdin = FakeStdin()
        pid = 12345

        def poll(self):
            return None

    worker.process = FakeProc()  # type: ignore[assignment]
    worker.messages.put("Generated: line1\n")
    closed = []
    monkeypatch.setattr(worker, "close", lambda: closed.append(True))
    monkeypatch.setattr(tts_daemon, "register_process", lambda *_a, **_k: None)
    monkeypatch.setattr(tts_daemon, "unregister_process", lambda *_a, **_k: None)

    manifest = tmp_path / "m.jsonl"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(SynthesisError, match="bad wav"):
        worker.run_batch(
            manifest,
            total_items=1,
            timeout_seconds=10.0,
            progress=None,
            cancel_event=None,
            on_line=lambda _line: (_ for _ in ()).throw(SynthesisError("bad wav")),
            output_tail=deque(),
        )
    assert closed == [True]


def test_indextts_and_indextts25_per_line_commit_error_does_not_abort_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.wav"
    sf.write(src, _tone(2.0, 16_000), 16_000, subtype="FLOAT")
    (tmp_path / "chinese").mkdir()
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    good_wav = _tone(0.5, 16_000)

    def make_project(backend: str) -> DubProject:
        s1 = Sentence(
            id="s000001", start_seconds=0.0, end_seconds=0.8, source_text="a", zh_text="甲"
        )
        s2 = Sentence(
            id="s000002", start_seconds=0.9, end_seconds=1.7, source_text="b", zh_text="乙"
        )
        settings = ProjectSettings(
            tts_backend=backend,  # type: ignore[arg-type]
            tts_model_path=str(model_dir),
            tts_config_path=str(model_dir / "config.yaml"),
            tts_index25_model_path=str(model_dir),
            tts_index25_config_path=str(model_dir / "config.yaml"),
        )
        return DubProject(
            source=_audio_info(src, 16_000, 2.0, 1), settings=settings, sentences=[s1, s2]
        )

    ref = voice_reference.VoiceReference(path=src, text="a", identity="id", language="ja")
    monkeypatch.setattr(tts_backends, "prepare_index_speaker_reference", lambda *a: ref)
    monkeypatch.setattr(tts_backends, "prepare_index_emotion_reference", lambda *a: None)
    monkeypatch.setattr(tts_backends, "daemon_pool_active", lambda: True)

    # Test IndexTTS2 daemon _on_v2_line
    proj2 = make_project("indextts2")
    monkeypatch.setattr(
        tts_backends, "_indextts_command", lambda _p: ["python", "-m", "indextts.cli_v2"]
    )

    def fake_run_v2(**kwargs):
        lines = Path(kwargs["manifest"]).read_text(encoding="utf-8").splitlines()
        out2 = json.loads(lines[1])["output"]
        sf.write(out2, good_wav, 16_000, subtype="FLOAT")
        kwargs["on_line"](
            f"Generated: {json.loads(lines[0])['output']}\n"
        )  # missing -> SynthesisError
        kwargs["on_line"](f"Generated: {out2}\n")
        return 0

    monkeypatch.setattr(tts_backends, "run_resident_with_one_restart", fake_run_v2)
    failures2 = tts_indextts._synthesize_indextts_cli_batch(
        proj2, tmp_path, src, proj2.sentences, None, None, None
    )
    assert len(failures2) == 1
    assert proj2.sentences[0].status == "error"
    assert proj2.sentences[1].status == "synthesized"

    # Test IndexTTS-2.5 daemon _on_v25_line
    proj25 = make_project("indextts2_5")
    monkeypatch.setattr(tts_backends, "_indextts25_command", lambda _p: ["python", "worker.py"])

    def fake_run_v25(**kwargs):
        lines = Path(kwargs["manifest"]).read_text(encoding="utf-8").splitlines()
        out2 = json.loads(lines[1])["output"]
        sf.write(out2, good_wav, 16_000, subtype="FLOAT")
        kwargs["on_line"]("Generated: s000001\n")  # missing -> SynthesisError
        kwargs["on_line"]("Generated: s000002\n")
        return 0

    monkeypatch.setattr(tts_backends, "run_resident_with_one_restart", fake_run_v25)
    failures25 = tts_indextts25._synthesize_indextts25_batch(
        proj25, tmp_path, src, proj25.sentences, None, None, None
    )
    assert len(failures25) == 1
    assert proj25.sentences[0].status == "error"
    assert proj25.sentences[1].status == "synthesized"


def test_english_normalize_pauses_dub_problem_and_indextts2_cache_key(tmp_path: Path) -> None:
    assert (
        tts.normalize_pauses("Hello... world——right?!", target_language="en")
        == "Hello, world, right?!"
    )
    assert "，" not in tts.normalize_pauses("Wait... really...", target_language="en")

    # Long-lettered 4-word English sentence (~0.6 s) must not fail letter-count check.
    wav = tmp_path / "en.wav"
    sf.write(wav, _tone(0.65, 16_000), 16_000, subtype="FLOAT")
    assert tts.dub_problem(wav, "extraordinary appreciation", target_language="en") is None

    # All 9 IndexTTS2 generation parameters must invalidate tts_cache_key.
    src = tmp_path / "src.wav"
    sf.write(src, _tone(1.0, 16_000), 16_000)
    sent = Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="a", zh_text="甲")
    base_settings = ProjectSettings(
        tts_backend="indextts2",
        tts_index_speaker_source="sentence_reference",
        tts_index_emotion_source="speaker_reference",
    )
    proj = DubProject(
        source=_audio_info(src, 16_000, 1.0, 1), settings=base_settings, sentences=[sent]
    )
    base_key = tts.tts_cache_key(proj, sent)
    for field, alt in [
        ("random_seed", 42),
        ("tts_index25_num_beams", 5),
        ("tts_index25_do_sample", False),
        ("tts_index25_top_k", 15),
        ("tts_index25_repetition_penalty", 5.0),
        ("tts_index25_length_penalty", 0.5),
        ("tts_index25_max_mel_tokens", 900),
        ("tts_index25_interval_silence_ms", 100),
        ("tts_index25_max_text_tokens", 80),
    ]:
        changed = proj.model_copy(
            update={"settings": base_settings.model_copy(update={field: alt})}
        )
        assert tts.tts_cache_key(changed, sent) != base_key


def test_tts_qc_retry_scope_cancel_restore_and_vram_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.wav"
    sf.write(src, _tone(2.0, 16_000), 16_000)
    s_good = Sentence(
        id="good", start_seconds=0.0, end_seconds=0.8, source_text="a", zh_text="你好呀"
    )
    s_bad = Sentence(
        id="bad", start_seconds=1.0, end_seconds=1.8, source_text="b", zh_text="再见啦"
    )
    proj = DubProject(
        source=_audio_info(src, 16_000, 2.0, 1),
        settings=ProjectSettings(tts_quality_check=True, tts_asr_quality_check=True),
        sentences=[s_good, s_bad],
    )
    checked_ids: list[str] = []

    def fake_synth(project, project_dir, source, force=False, sentence_ids=None, **kwargs):
        for s in project.sentences:
            if sentence_ids is not None and s.id not in sentence_ids:
                continue
            if s.id == "bad" and s.tts_attempt >= 1:
                s.status = "error"
                s.error = "retry failed"
                return ["bad: retry failed"]
            out = project_dir / f"{s.id}-{s.tts_attempt}.wav"
            sf.write(out, _tone(0.6, 16_000), 16_000)
            s.tts_file = out.name
            s.status = "synthesized"
        return []

    def fake_asr_check(path, _expected, _settings):
        sid = Path(path).stem.split("-")[0]
        checked_ids.append(sid)
        return ("bad cer", 0.5) if sid == "bad" else (None, 0.0)

    evicted = []
    monkeypatch.setattr(tts_backends, "synthesize_with_selected_backend", fake_synth)
    monkeypatch.setattr(dub_qc, "dub_asr_check", fake_asr_check)
    monkeypatch.setattr(tts, "_evict_tts_daemons_if_low_vram", lambda *_a: evicted.append(True))

    tts.synthesize_sentences(proj, tmp_path, src)
    assert evicted
    assert checked_ids == ["good", "bad"]  # failed retry restored without re-checking good!
    assert s_bad.status == "synthesized"
    assert s_bad.tts_attempt == 0
    assert s_bad.tts_file == "bad-0.wav"

    # Cancellation check inside QC loop
    signal = SimpleNamespace(is_set=lambda: True)
    with pytest.raises(OperationCancelledError):
        tts.synthesize_sentences(proj, tmp_path, src, cancel_event=signal)  # type: ignore[arg-type]


def test_http_runners_retry_mimo_default_indextts2_api_emotion_and_glossary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        zh_phonetics.apply_spoken_normalization("ok, look at book", glossary="ok=好的")
        == "好的, look at book"
    )

    # mimo_tts with empty tts_model defaults to voiceclone -> requires reference audio.
    p_mimo = DubProject(
        source=_audio_info(tmp_path / "s.wav", 16_000, 1.0, 1),
        settings=ProjectSettings(tts_backend="mimo_tts", tts_model=""),
    )
    assert tts_backends._uses_reference_audio(p_mimo) is True

    # _post_with_retry retries 429/503 up to 3 times with exponential backoff.
    sleeps: list[float] = []
    monkeypatch.setattr(tts_http_runners.time, "sleep", sleeps.append)
    attempts = {"n": 0}

    def flaky_post():
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    resp = tts_http_runners._post_with_retry(flaky_post)
    assert resp.status_code == 200
    assert sleeps == [0.5, 1.0]

    # indextts2_api passes emotion_audio when prepare_index_emotion_reference returns one.
    ref_wav = tmp_path / "ref.wav"
    emo_wav = tmp_path / "emo.wav"
    sf.write(ref_wav, _tone(1.2, 16_000), 16_000)
    sf.write(emo_wav, _tone(1.2, 16_000), 16_000)
    monkeypatch.setattr(
        tts_backends,
        "prepare_index_speaker_reference",
        lambda *a: voice_reference.VoiceReference(path=ref_wav, text="x", identity="s"),
    )
    monkeypatch.setattr(
        tts_backends,
        "prepare_index_emotion_reference",
        lambda *a: voice_reference.VoiceReference(path=emo_wav, text="x", identity="e"),
    )
    sent = Sentence(id="s1", start_seconds=0.0, end_seconds=1.2, source_text="a", zh_text="甲")
    p_api = DubProject(
        source=_audio_info(ref_wav, 16_000, 1.2, 1),
        settings=ProjectSettings(tts_backend="indextts2_api"),
        sentences=[sent],
    )
    combined_ref = tts_backends.prepare_index_voice_and_emotion_references(
        p_api, tmp_path, ref_wav, sent
    )
    assert combined_ref.emotion_path == emo_wav


def test_dub_qc_logs_warning_and_discovers_faster_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    wav = tmp_path / "dub.wav"
    sf.write(wav, _tone(0.5, 16_000), 16_000)
    monkeypatch.setattr(dub_qc, "cached_model_path", lambda _id: None)
    with caplog.at_level("WARNING"):
        assert dub_qc.transcribe_dub_clip(wav, "zh", ProjectSettings()) is None
    assert "跳过配音回听检查" in caplog.text


def test_voice_reference_speaker_and_mode_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate = 16_000
    s_short = Sentence(id="s01", start_seconds=0.0, end_seconds=0.12, source_text="ん")
    s_spk1 = Sentence(
        id="s02", start_seconds=0.2, end_seconds=2.2, speaker_id="S01", source_text="おはよう"
    )
    s_spk2_filler = Sentence(
        id="s03",
        start_seconds=2.3,
        end_seconds=5.5,
        speaker_id="S02",
        source_text="はぁ…んっ…",
        status="skipped_filler",
    )
    s_spk2_a = Sentence(
        id="s04", start_seconds=5.6, end_seconds=7.6, speaker_id="S02", source_text="おやすみなさい"
    )
    s_spk2_b = Sentence(
        id="s05", start_seconds=7.7, end_seconds=10.0, speaker_id="S02", source_text="大丈夫だよ"
    )
    src = tmp_path / "src.m4a"
    src.write_bytes(b"fake-m4a")
    decoded = tmp_path / "decoded.wav"
    sf.write(decoded, _tone(10.5, rate), rate, subtype="FLOAT")

    # Simulate sf.SoundFile failing on .m4a and falling back via open_audio_for_analysis.
    monkeypatch.setattr(
        audio,
        "_run_ffmpeg",
        lambda args: sf.write(Path(args[-1]), _tone(10.5, rate), rate, subtype="FLOAT"),
    )
    proj = DubProject(
        source=_audio_info(src, rate, 10.5, 1),
        settings=ProjectSettings(tts_index_emotion_source="project_reference"),
        sentences=[s_short, s_spk1, s_spk2_filler, s_spk2_a, s_spk2_b],
    )
    voice_reference.annotate_sentence_voice_modes(proj, src)
    assert s_short.voice_mode == "modal"

    # speaker_reference_sentence skips filler s03 and picks real S02 speech.
    picked_spk, _ = voice_reference.speaker_reference_sentence(proj, s_spk2_a)
    assert picked_spk.id in {"s04", "s05"}

    # fallback_reference_sentence respects speaker_id S02.
    fb = voice_reference.fallback_reference_sentence(proj, s_spk2_a)
    assert fb is not None and fb.speaker_id == "S02" and fb.id != s_spk2_a.id


def test_separation_process_registration_cache_digest_and_download_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. SeparationSession registers and unregisters subprocess.
    registered: list[object] = []
    unregistered: list[object] = []
    monkeypatch.setattr(separation, "register_process", lambda p: registered.append(p) or "sig")
    monkeypatch.setattr(separation, "unregister_process", lambda p, s: unregistered.append((p, s)))
    monkeypatch.setattr(separation, "runtime_python", lambda: tmp_path / "python")

    class FakePopen:
        returncode = 0
        pid = 999

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakePopen())
    result_file = tmp_path / "res.json"
    out_wav = tmp_path / "vocals.wav"
    out_wav.write_bytes(b"wav")
    result_file.write_text(json.dumps({"vocals": str(out_wav)}), encoding="utf-8")

    with separation.SeparationSession(tmp_path) as sess:
        sess.separate({"input": "in.wav", "output": str(tmp_path), "result": str(result_file)}, 10)
    assert len(registered) == 1
    assert len(unregistered) == 1 and unregistered[0][1] == "sig"

    # 2. separation_timeout_seconds is excluded from cache digest.
    src = tmp_path / "src.wav"
    sf.write(src, np.zeros((1600, 2), dtype=np.float32), 16_000, subtype="FLOAT")
    monkeypatch.setattr(separation, "verify_local_model", lambda _m: "modelhash")
    created_dirs: list[Path] = []
    real_mkdir = Path.mkdir

    def track_mkdir(self, *a, **k):
        if self.parent.name == "separation":
            created_dirs.append(self)
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", track_mkdir)
    for timeout in (60.0, 900.0):
        p = DubProject(
            source=_audio_info(src, 16_000, 0.1, 2),
            settings=ProjectSettings(separation_enabled=True, separation_timeout_seconds=timeout),
        )
        # Pre-seed complete cache for both runs so we just observe the directory digest.
        with contextlib.suppress(Exception):
            separation._ensure_separation_impl(p, tmp_path, src)
    assert len({d.name for d in created_dirs}) == 1

    # 3. Block overseas fallback when allow_external=False, keep .partial, send headers.
    monkeypatch.setenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", "0")
    monkeypatch.setattr(
        separation_download,
        "source_manifest",
        lambda: {
            "files": {
                "m.ckpt": {
                    "sha256": hashlib.sha256(b"abcdef").hexdigest(),
                    "upstream": "https://huggingface.co/m.ckpt",
                }
            }
        },
    )
    seen_headers: list[tuple[str, dict[str, str]]] = []

    class PartialResp:
        status_code = 200

        def __init__(self) -> None:
            self.headers = {"Content-Length": "6"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            yield b"abc"
            raise OSError("network drop")

    def fake_get(url, headers=None, **_):
        seen_headers.append((url, dict(headers or {})))
        return PartialResp()

    target = tmp_path / "m.ckpt"
    with pytest.raises(RuntimeError, match="下载失败"):
        separation_download.download_model_file(
            "https://github.com/m.ckpt", target, models=tmp_path, source="modelscope", get=fake_get
        )
    assert len(seen_headers) == 1  # overseas huggingface/github never attempted
    assert seen_headers[0][1]["User-Agent"] == "curl/8.0"
    assert target.with_name("m.ckpt.partial").read_bytes() == b"abc"
