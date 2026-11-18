from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from asmr_dubber.campplus import find_campplus_checkpoint, kaldi_fbank_80
from asmr_dubber.diarization import (
    acoustic_timbre_embedding,
    cluster_speaker_embeddings,
    diarize_sentences,
    stitch_moss_chunk_speakers,
)
from asmr_dubber.models import Sentence


def _synth_voice(
    rate: int,
    seconds: float,
    *,
    f0: float,
    formants: tuple[float, float, float],
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a deterministic vowel-like harmonic signal with distinct vocal-tract formants."""
    n = round(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    glottal = np.zeros(n, dtype=np.float64)
    max_k = int((rate * 0.45) // f0)
    for k in range(1, max(2, min(max_k, 40))):
        freq = k * f0
        gain = sum(
            math.exp(-0.5 * ((freq - fc) / (0.14 * fc)) ** 2) for fc in formants
        ) / math.sqrt(k)
        glottal += gain * np.sin(2.0 * math.pi * freq * t + 0.1 * k)
    rng = np.random.default_rng(seed)
    glottal += rng.standard_normal(n) * 0.004
    glottal /= max(float(np.max(np.abs(glottal))), 1e-9)
    return (glottal * 0.35).astype(np.float32)


def test_kaldi_fbank_80_shape_and_zero_mean_cmn() -> None:
    sig = _synth_voice(16_000, 1.2, f0=180.0, formants=(500.0, 1500.0, 2500.0))
    feats = kaldi_fbank_80(sig, 16_000)
    assert feats.ndim == 2
    assert feats.shape[1] == 80
    assert feats.shape[0] >= 115
    assert np.allclose(np.mean(feats, axis=0), 0.0, atol=1e-4)
    # Path lookup returns Path or None cleanly without raising in dev environment.
    ckpt = find_campplus_checkpoint()
    assert ckpt is None or ckpt.is_file()


def test_diarize_sentences_separates_two_speakers_under_5pct_error(tmp_path: Path) -> None:
    rate = 16_000
    gap = np.zeros(round(rate * 0.35), dtype=np.float32)
    # Speaker A: higher female voice (F0 ~ 255 Hz, higher formants)
    # Speaker B: lower male/alto voice (F0 ~ 115 Hz, lower formants)
    turns = [
        ("A", "おはよう、起きた？"),
        ("B", "うん、今起きたところ。"),
        ("A", "朝ごはん一緒に食べようね。"),
        ("B", "ありがとう、コーヒー淹れるよ。"),
        ("A", "今日はいい天気だね。"),
        ("B", "散歩に行こうか。"),
        ("A", "うん、楽しみ！"),
        ("B", "じゃあ支度しよう。"),
    ]
    pieces: list[np.ndarray] = [gap]
    sentences: list[Sentence] = []
    cursor = len(gap) / rate
    for i, (spk, text) in enumerate(turns, start=1):
        clip = (
            _synth_voice(rate, 1.8, f0=255.0 + i, formants=(680.0, 2050.0, 3150.0), seed=i)
            if spk == "A"
            else _synth_voice(
                rate, 1.8, f0=115.0 + i, formants=(410.0, 1220.0, 2250.0), seed=i + 20
            )
        )
        start = round(cursor, 3)
        end = round(cursor + len(clip) / rate, 3)
        sentences.append(
            Sentence(id=f"s{i:06d}", start_seconds=start, end_seconds=end, source_text=text)
        )
        pieces.extend([clip, gap])
        cursor += (len(clip) + len(gap)) / rate

    wav = tmp_path / "duet.wav"
    sf.write(wav, np.concatenate(pieces), rate)

    diarized = diarize_sentences(sentences, wav)
    expected = ["S01" if spk == "A" else "S02" for spk, _ in turns]
    errors = sum(s.speaker_id != exp for s, exp in zip(diarized, expected, strict=True))
    assert errors / len(turns) < 0.05


def test_diarize_single_speaker_monologue_stays_unified(tmp_path: Path) -> None:
    rate = 16_000
    gap = np.zeros(round(rate * 0.3), dtype=np.float32)
    pieces: list[np.ndarray] = [gap]
    sentences: list[Sentence] = []
    cursor = len(gap) / rate
    for i in range(1, 7):
        clip = _synth_voice(
            rate,
            1.6,
            f0=225.0 + (i % 3) * 4.0,
            formants=(620.0, 1900.0, 2950.0),
            seed=100 + i,
        )
        sentences.append(
            Sentence(
                id=f"s{i:06d}",
                start_seconds=round(cursor, 3),
                end_seconds=round(cursor + len(clip) / rate, 3),
                source_text=f"台詞{i}",
            )
        )
        pieces.extend([clip, gap])
        cursor += (len(clip) + len(gap)) / rate

    wav = tmp_path / "mono.wav"
    sf.write(wav, np.concatenate(pieces), rate)
    diarized = diarize_sentences(sentences, wav)
    assert {s.speaker_id for s in diarized} == {"S01"}


def test_stitch_moss_chunk_speakers_across_windows(tmp_path: Path) -> None:
    rate = 16_000
    gap = np.zeros(round(rate * 0.25), dtype=np.float32)
    # Chunk 1 has A=c0001:S01, B=c0001:S02; Chunk 2 swaps order so B speaks first as c0002:S01!
    specs = [
        ("c0001:S01", 250.0, (670.0, 2020.0, 3100.0)),
        ("c0001:S02", 118.0, (420.0, 1240.0, 2260.0)),
        ("c0002:S01", 120.0, (425.0, 1250.0, 2270.0)),
        ("c0002:S02", 252.0, (675.0, 2030.0, 3110.0)),
    ]
    pieces: list[np.ndarray] = [gap]
    sentences: list[Sentence] = []
    cursor = len(gap) / rate
    for i, (tag, f0, formants) in enumerate(specs, start=1):
        clip = _synth_voice(rate, 1.7, f0=f0, formants=formants, seed=50 + i)
        sentences.append(
            Sentence(
                id=f"s{i:06d}",
                start_seconds=round(cursor, 3),
                end_seconds=round(cursor + len(clip) / rate, 3),
                source_text=f"文{i}",
                speaker_id=tag,
            )
        )
        pieces.extend([clip, gap])
        cursor += (len(clip) + len(gap)) / rate

    wav = tmp_path / "moss_chunks.wav"
    sf.write(wav, np.concatenate(pieces), rate)
    stitched = stitch_moss_chunk_speakers(sentences, wav, num_speakers=2)
    assert [s.speaker_id for s in stitched] == ["S01", "S02", "S02", "S01"]


def test_cluster_embeddings_and_acoustic_edge_cases() -> None:
    assert cluster_speaker_embeddings(np.zeros((0, 16), dtype=np.float32)) == []
    assert acoustic_timbre_embedding(np.zeros(100, dtype=np.float32)).shape == (36,)


def test_per_speaker_voice_anchors_ui_override_translation_tags_and_moss_compat(
    tmp_path: Path,
) -> None:
    from asmr_dubber import moss
    from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings
    from asmr_dubber.services.project_records import apply_table, project_rows
    from asmr_dubber.translation import _json_lines, _translation_memory
    from asmr_dubber.tts import tts_cache_key
    from asmr_dubber.voice_reference import (
        effective_project_reference_sentence,
        prepare_voice_reference,
    )

    # Switching ASR backend to MOSS from anime-whisper auto-normalizes asr_model to moss.MODEL_ID.
    moss_cfg = ProjectSettings(
        asr_backend="moss_transcribe_diarize",
        asr_model="litagin/anime-whisper",
    )
    assert moss_cfg.asr_model == moss.MODEL_ID

    rate = 16_000
    wav = tmp_path / "source.wav"
    sf.write(wav, _synth_voice(rate, 12.0, f0=210.0, formants=(600.0, 1800.0, 2800.0)), rate)
    proj = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="a" * 64,
            duration_seconds=12.0,
            sample_rate=rate,
            channels=1,
        ),
        settings=ProjectSettings(tts_backend="indextts2_5", tts_model="IndexTTS-2.5"),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.5,
                end_seconds=4.5,
                speaker_id="S01",
                source_text="おはよう、よく眠れた？",
                zh_text="早安，睡得好吗？",
            ),
            Sentence(
                id="s000002",
                start_seconds=5.0,
                end_seconds=9.5,
                speaker_id="S02",
                source_text="うん、すごくよく眠れたよ。",
                zh_text="嗯，睡得特别好哦。",
            ),
            Sentence(
                id="s000003",
                start_seconds=9.8,
                end_seconds=11.5,
                speaker_id="S01",
                source_text="よかったね。",
                zh_text="太好了呢。",
            ),
        ],
    )
    ref_s1, _ = effective_project_reference_sentence(proj, proj.sentences[2])
    ref_s2, _ = effective_project_reference_sentence(proj, proj.sentences[1])
    assert ref_s1.id == "s000001"
    assert ref_s2.id == "s000002"

    vr1 = prepare_voice_reference(proj, tmp_path, wav, proj.sentences[2])
    vr2 = prepare_voice_reference(proj, tmp_path, wav, proj.sentences[1])
    assert vr1.path != vr2.path
    assert "speaker_S01" in vr1.path.name
    assert "speaker_S02" in vr2.path.name

    key_before = tts_cache_key(proj, proj.sentences[2])
    proj.sentences[2].tts_file = "tts/s000003.wav"
    proj.sentences[2].tts_cache_key = key_before

    # UI 11th-column speaker_id override invalidates stale dub and switches voice anchor!
    rows = [
        (*r, s.speaker_id or "") for r, s in zip(project_rows(proj), proj.sentences, strict=True)
    ]
    rows[2] = [*rows[2][:10], "S02"]
    assert apply_table(proj, rows) is True
    assert proj.sentences[2].speaker_id == "S02"
    assert proj.sentences[2].tts_file is None
    assert not proj.sentences[2].review_locked
    assert tts_cache_key(proj, proj.sentences[2]) != key_before

    prompt_json = _json_lines(proj.sentences)
    mem_json = _translation_memory(proj.sentences)
    assert '"speaker":"S01"' in prompt_json and '"speaker":"S02"' in prompt_json
    assert '"speaker":"S01"' in mem_json
