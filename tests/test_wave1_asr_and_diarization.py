from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import asr, asr_candidates, asr_runners, asr_token_timing, pipeline
from asmr_dubber.asr_review import AudioWindow, _candidate, compare_window, replace_window
from asmr_dubber.diarization import (
    cluster_speaker_embeddings,
    diarize_sentences,
    stitch_moss_chunk_speakers,
)
from asmr_dubber.errors import AsmrDubberError, OperationCancelledError
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence
from asmr_dubber.segmentation import (
    TimedToken,
    restore_punctuation,
    split_sound_runs,
    split_timed_tokens,
)
from asmr_dubber.transcript_import import parse_transcript
from asmr_dubber.translation_script import reconcile_script_sentences
from asmr_dubber.translation_script_spans import _validated_script_spans
from asmr_dubber.vad import _speech_segments_from_probabilities, _whisper_drc_waveform


def _sent(sid: str, s: float, e: float, txt: str, **kw: Any) -> Sentence:
    return Sentence(id=sid, start_seconds=s, end_seconds=e, source_text=txt, **kw)


def _proj(src: Path, **kw: Any) -> DubProject:
    return DubProject(
        source=AudioInfo(
            path=src.name, sha256="a" * 64, duration_seconds=10, sample_rate=16_000, channels=1
        ),
        **kw,
    )


def test_vad_hysteresis_and_quiet_whisper_drc() -> None:
    probs = np.zeros(500, dtype=np.float32)
    probs[10:25] = 0.75
    probs[25:27] = 0.20  # brief 40 ms dip (< min_silence_ms)
    probs[27:427] = 0.42  # 8 s soft whisper in [negative_threshold=0.35, threshold=0.5)
    probs[427] = 0.20  # single dip must not truncate frames 25..427
    probs[428:440] = 0.42
    segs = _speech_segments_from_probabilities(
        probs,
        total_samples=500 * 320,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=200,
        speech_pad_ms=0,
    )
    assert len(segs) == 1
    assert (segs[0].start_sample, segs[0].end_sample) == (10 * 320, 440 * 320)

    t = np.arange(32_000, dtype=np.float32) / 16_000
    whisper = (np.sin(2.0 * np.pi * 220.0 * t) * 0.004).astype(np.float32)
    boosted = _whisper_drc_waveform(whisper, 16_000)
    orig_peak, new_peak = float(np.max(np.abs(whisper))), float(np.max(np.abs(boosted)))
    assert new_peak > orig_peak * 4.0
    assert new_peak <= 10.0 ** (-12.0 / 20.0) + 1e-4
    assert new_peak / orig_peak <= 10.0 ** (18.0 / 20.0) + 1e-3


def test_evict_recognition_session_forces_cleanup_and_session_model_evicts_low_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"gc": 0, "empty_cache": 0, "sync": 0}
    monkeypatch.setattr(asr.gc, "collect", lambda: calls.__setitem__("gc", calls["gc"] + 1))

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (2 * 1024**3, 8 * 1024**3),
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
        synchronize=lambda: calls.__setitem__("sync", calls["sync"] + 1),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    with asr.recognition_session():
        assert asr._session_model("m1", lambda: object()) is not None
        asr.evict_recognition_session_if_low_vram(min_free_gb=5.0)
        assert calls["gc"] >= 1 and calls["empty_cache"] >= 1 and calls["sync"] >= 1
        asr._session_model("m2", lambda: object())
        prev_cache = calls["empty_cache"]
        asr._session_model("m3", lambda: object())
        assert calls["empty_cache"] > prev_cache


def test_vad_clips_review_empty_and_active_frames_monologue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    monkeypatch.setattr("asmr_dubber.vad.detect_asmr_speech", lambda *a, **k: [])
    with pytest.raises(AsmrDubberError, match="没有检测到语音"):
        asr_candidates._vad_clips(wav, ProjectSettings(), None, None)
    with asr.recognition_session():
        assert asr_candidates._vad_clips(wav, ProjectSettings(), None, None) == (16_000, [], [])

    t = np.arange(16_000 * 8, dtype=np.float32) / 16_000
    for amp in (0.08, 0.004):
        act, _ = asr_token_timing.active_frames(
            (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 16_000
        )
        assert float(np.mean(act)) == 1.0


def test_anime_whisper_alignment_fallback_and_oom_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(32_000, 0.05, dtype=np.float32), 16_000)
    clip0 = np.zeros(16_000, dtype=np.float32)
    clip0[round(0.2 * 16_000) : round(0.9 * 16_000)] = 0.05
    clip1 = np.full(16_000, 0.05, dtype=np.float32)
    monkeypatch.setattr(asr_candidates, "_require_model", lambda *a, **k: None)
    monkeypatch.setattr(
        asr_candidates, "resolve_transformers_model_source", lambda model_id: (model_id, None)
    )
    monkeypatch.setattr(asr_candidates, "_pretrained", lambda *a, **k: ("m", {"dtype": "float32"}))
    monkeypatch.setattr(
        asr_candidates,
        "_vad_clips",
        lambda *a, **k: (16_000, [(0, 16_000), (16_000, 32_000)], [clip0, clip1]),
    )
    monkeypatch.setattr(asr, "_cleanup_cuda", lambda force=False: None)
    monkeypatch.setattr(
        asr,
        "_run_transformers_asr_pipeline",
        lambda *a, **k: [{"text": "おはよう。"}, {"text": "おやすみ。"}],
    )
    fake_torch = SimpleNamespace(
        float16="f16", float32="f32", cuda=SimpleNamespace(is_available=lambda: False)
    )
    fake_tf = SimpleNamespace(
        AutoModelForSpeechSeq2Seq=SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace(to=lambda d: None)
        ),
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace(tokenizer=None, feature_extractor=None)
        ),
        pipeline=lambda *a, **k: object(),
    )

    def _aligner(fn: Any) -> Any:
        return SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(align=fn))

    def fallback_align(audio, text, language):
        if len(text) == 2:
            raise ValueError("bad clip in batch")
        if text[0] == "おはよう。":
            return [
                SimpleNamespace(
                    items=[SimpleNamespace(text="おはよう", start_time=0.2, end_time=0.9)]
                )
            ]
        raise ValueError("bad clip 2")

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(
        sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=_aligner(fallback_align))
    )
    settings = ProjectSettings(asr_backend="anime_whisper", asr_device="cpu")
    sentences, _ = asr_candidates.transcribe_anime_whisper(wav, settings, None)
    assert [s.source_text for s in sentences] == ["おはよう。", "おやすみ。"]
    assert sentences[0].start_seconds == pytest.approx(0.2)

    cleanup_forces: list[bool] = []

    def _raise(exc: BaseException) -> Any:
        return lambda *_a, **_k: (_ for _ in ()).throw(exc)

    monkeypatch.setattr(asr, "_cleanup_cuda", lambda force=False: cleanup_forces.append(force))
    for exc, err_type in [
        (RuntimeError("CUDA out of memory"), RuntimeError),
        (OperationCancelledError("cancelled"), OperationCancelledError),
    ]:
        monkeypatch.setitem(
            sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=_aligner(_raise(exc)))
        )
        with pytest.raises(err_type):
            asr_candidates.transcribe_anime_whisper(wav, settings, None)
    assert True in cleanup_forces


def test_generic_api_review_empty_cancel_prompt_extra_body_and_chunking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "short.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    captured: list[dict[str, Any]] = []

    def fake_client(responses: list[Any]) -> Any:
        it = iter(responses)

        def post(url, data=None, files=None):
            captured.append({"data": dict(data or {})})
            nxt = next(it)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

        return SimpleNamespace(post=post, close=lambda: None)

    settings = ProjectSettings(
        asr_backend="generic_asr_api",
        asr_model="whisper-1",
        asr_api_base_url="https://api.example.com/v1",
        asr_initial_prompt="whisper ASMR",
        asr_api_extra_body='{"timestamp_granularities[]": "word", "temperature": 0.2}',
    )
    empty_resp = httpx.Response(200, json={"text": "", "segments": [], "language": "ja"})
    monkeypatch.setattr(asr, "logged_http_client", lambda *a, **k: fake_client([empty_resp]))
    with asr.recognition_session():
        assert asr_runners._transcribe_generic_api(wav, settings, None, api_key="k") == ([], "ja")
    assert captured[-1]["data"]["prompt"] == "whisper ASMR"
    assert captured[-1]["data"]["timestamp_granularities[]"] == "word"
    assert captured[-1]["data"]["temperature"] == "0.2"

    sig = threading.Event()
    sig.set()
    monkeypatch.setattr(
        asr, "logged_http_client", lambda *a, **k: fake_client([httpx.ReadError("closed")])
    )
    with pytest.raises(OperationCancelledError):
        asr_runners._transcribe_generic_api(wav, settings, None, cancel_event=sig, api_key="k")

    monkeypatch.setattr(asr_runners, "_GENERIC_API_MAX_BYTES", 10_000)
    monkeypatch.setattr(asr_runners, "_GENERIC_API_CHUNK_TARGET_BYTES", 32_000)
    monkeypatch.setattr(
        "asmr_dubber.audio._run_ffmpeg",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg")),
    )
    long_wav = tmp_path / "long.wav"
    sf.write(long_wav, np.zeros(16_000 * 65, dtype=np.float32), 16_000)
    chunk_responses = [
        httpx.Response(200, json={"text": t, "segments": [{"text": t, "start": 0.5, "end": 2.0}]})
        for t in ("First.", "Second.", "Third.")
    ]
    monkeypatch.setattr(asr, "logged_http_client", lambda *a, **k: fake_client(chunk_responses))
    sentences, _ = asr_runners._transcribe_generic_api(
        long_wav, settings, None, source_language="en", api_key="k"
    )
    assert (
        len(sentences) >= 2
        and sentences[0].source_text == "First."
        and sentences[1].start_seconds > 25.0
    )


def test_diarization_n300_speed_moss_2centroid_cannot_link_and_fallback(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    base = np.eye(16, dtype=np.float32)
    embs = np.vstack([base[i % 3] + rng.normal(0, 0.05, 16).astype(np.float32) for i in range(300)])
    t0 = time.perf_counter()
    labels = cluster_speaker_embeddings(embs)
    assert time.perf_counter() - t0 < 0.2
    assert len(set(labels)) == 3

    ortho = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert cluster_speaker_embeddings(ortho, min_separation_cosine=0.20) == [0, 1]
    same = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert cluster_speaker_embeddings(same, cannot_link=[(0, 1)]) == [0, 1]

    rate = 16_000
    t = np.arange(rate * 2, dtype=np.float32) / rate
    a = (0.2 * np.sin(2 * np.pi * 260 * t)).astype(np.float32)
    b = (0.2 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    wav = tmp_path / "moss_short.wav"
    sf.write(wav, np.concatenate([a, b, np.zeros(round(rate * 0.08), dtype=np.float32), a]), rate)
    sents = [
        _sent("s1", 0.0, 2.0, "A1", speaker_id="c0001:S01"),
        _sent("s2", 2.0, 4.0, "B1", speaker_id="c0001:S02"),
        _sent("s3", 4.0, 4.08, "うん", speaker_id="c0002:S01"),
        _sent("s4", 4.08, 6.08, "A2", speaker_id="c0002:S02"),
    ]
    stitched = stitch_moss_chunk_speakers(sents, wav, num_speakers=2)
    assert stitched[3].speaker_id == stitched[0].speaker_id == "S01"
    assert stitched[2].speaker_id == "S03"

    sig = threading.Event()
    sig.set()
    with pytest.raises(OperationCancelledError):
        diarize_sentences(sents[:2], wav, cancel_event=sig)


def test_segmentation_quotes_nfkc_english_spaces_and_edge_sound_words() -> None:
    toks = [TimedToken("おはよう", 0.0, 0.8), TimedToken("ねえ", 1.0, 1.5)]
    sents = split_timed_tokens(restore_punctuation(toks, "おはよう。「ねえ」"), pause_seconds=0.5)
    assert [s.source_text for s in sents] == ["おはよう。", "「ねえ」"]

    en_toks = [TimedToken("hello", 0.0, 0.5), TimedToken("world", 0.5, 1.0)]
    assert "World!" in "".join(t.text for t in restore_punctuation(en_toks, "Ｈｅｌｌｏ, World!"))

    joined = split_timed_tokens(
        [
            TimedToken(w, i * 0.3, (i + 1) * 0.3)
            for i, w in enumerate(["Hello", "world", "how", "are", "you."])
        ]
    )
    assert [s.source_text for s in joined] == ["Hello world how are you."]

    for word in ["あったかいね", "スーパーに行く", "おっぱい", "おおきい", "はっきり"]:
        chars = [TimedToken(ch, i * 0.25, (i + 1) * 0.25) for i, ch in enumerate(word)]
        assert [s.source_text for s in split_timed_tokens(chars)] == [word]

    paused = [
        TimedToken("は", 0.0, 0.25),
        TimedToken("ぁ", 0.25, 0.50),
        TimedToken("そ", 0.65, 0.85),
        TimedToken("う", 0.85, 1.05),
    ]
    assert len(split_sound_runs(paused)) == 2


def test_asr_review_breath_filter_inbounds_text_and_speaker_preservation() -> None:
    w = AudioWindow("w1", 0.0, 8.0, 0.0, 8.0, "end")
    baseline = [
        _sent("s1", 1.0, 3.0, "こんにちは。", speaker_id="S02", voice_mode="whisper"),
        _sent("s2", 4.0, 5.0, "（吐息）", enabled=False, status="skipped_sound"),
    ]
    cand_rows = [_sent("c1", 1.0, 3.0, "こんにちは。"), _sent("c2", 9.0, 10.0, "範囲外")]
    cand = _candidate(cand_rows, w, "parakeet_nemo|m", "ja")
    assert cand["text"] == "こんにちは。"
    comp = compare_window(w, baseline, [cand], "parakeet_nemo|m", "ja")
    assert comp["baseline_text"] == "こんにちは。" and comp["status"] == "single_family"

    comp2 = compare_window(
        w,
        baseline,
        [_candidate([cand_rows[0]], w, "parakeet_nemo|m", "ja")],
        "parakeet_nemo|m",
        "ja",
    )
    replaced = replace_window(baseline, comp2, comp2["candidates"][0], ProjectSettings())
    assert replaced[0].speaker_id == "S02" and replaced[0].voice_mode == "whisper"


def test_transcript_import_encoding_lrc_plain_ass_and_zh_source(tmp_path: Path) -> None:
    sjis = tmp_path / "sjis.txt"
    sjis.write_bytes("あいうえお！".encode("cp932"))
    assert len(sjis.read_bytes()) % 2 == 0
    assert (
        parse_transcript(duration_seconds=10, path=sjis).sentences[0].source_text == "あいうえお！"
    )

    gbk = tmp_path / "gbk.txt"
    gbk.write_bytes("你好世界".encode("gb18030"))
    assert len(gbk.read_bytes()) % 2 == 0
    parsed_zh = parse_transcript(duration_seconds=10, path=gbk, language="zh", role="source")
    assert (
        parsed_zh.sentences[0].source_text == "你好世界"
        and parsed_zh.sentences[0].target_text == ""
    )

    lrc = "[01:23:45]おはよう。\n[01:23:45]早安。\n[01:25:00]\n[01:30:00]またね。\n"
    parsed_lrc = parse_transcript(duration_seconds=120, pasted_text=lrc)
    assert parsed_lrc.sentences[0].start_seconds == pytest.approx(83.45)
    assert parsed_lrc.sentences[0].end_seconds == pytest.approx(85.00)
    assert (
        "おはよう。" in parsed_lrc.sentences[0].source_text
        and "早安。" in parsed_lrc.sentences[0].source_text
    )
    assert parsed_lrc.sentences[1].start_seconds == pytest.approx(90.00)

    plain = tmp_path / "script.txt"
    plain.write_text(
        "[01:23] Track note\nA --> B\n<whisper> {pause} こんにちは。\n", encoding="utf-8"
    )
    parsed_plain = parse_transcript(duration_seconds=30, path=plain)
    assert parsed_plain.source_format == "纯文本"
    assert "<whisper> {pause} こんにちは。" in [s.source_text for s in parsed_plain.sentences]

    ass = (
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,Alice,0,0,0,,Hello\n"
    )
    assert (
        parse_transcript(duration_seconds=10, pasted_text=ass, language="en")
        .sentences[0]
        .speaker_id
        == "Alice"
    )


def test_zh_en_source_script_skip_after_http_error_and_keep_unmatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "source.wav"
    sf.write(src, np.zeros(16_000, dtype=np.float32), 16_000)
    proj = _proj(src, source_language="zh", settings=ProjectSettings(tts_target_language="en"))
    monkeypatch.setattr(pipeline, "verify_source", lambda *a: src)
    monkeypatch.setattr(pipeline, "make_analysis_copy", lambda *a: src)
    monkeypatch.setattr(
        pipeline,
        "align_script_sentences_with_qwen",
        lambda *a, **k: [{"sentence_id": "s000001", "fallback": False}],
    )
    res = pipeline.import_project_transcript(
        proj,
        tmp_path,
        pasted_text="你好，请坐。",
        plain_timing="qwen",
        script_language="zh",
        script_kind="source",
    )
    assert (
        res["qwen_aligned_sentences"] == 1
        and proj.sentences[0].source_text == "你好，请坐。"
        and proj.sentences[0].zh_text == ""
    )
    q1, q2 = (
        [{"id": "p000001", "quote": "「おはよう」", "skip_after": "（耳元へ移動）"}],
        [{"id": "p000002", "quote": "「おやすみ」"}],
    )
    resp = json.dumps(
        {
            "corrections": [
                {"id": "s000001", "text": "「おはよう」", "script_quotes": q1},
                {"id": "s000002", "text": "「おやすみ」", "script_quotes": q2},
            ]
        },
        ensure_ascii=False,
    )
    sents = [_sent("s000001", 0, 1, "おはよう"), _sent("s000002", 1, 2, "おやすみ")]
    out, ends, allocs = _validated_script_spans(
        resp, sents, [("p000001", "「おはよう」（耳元へ移動）"), ("p000002", "「おやすみ」")], {}
    )
    assert out == {"s000001": "「おはよう」", "s000002": "「おやすみ」"}
    assert ends == {"p000001": len("「おはよう」（耳元へ移動）"), "p000002": len("「おやすみ」")}
    assert (
        isinstance(allocs[0]["skipped_after"], dict)
        and allocs[0]["skipped_after"]["text"] == "（耳元へ移動）"
    )

    def _chat(payload: dict[str, Any]) -> httpx.Response:
        msg = {"content": json.dumps(payload, ensure_ascii=False)}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] <= 3:
            bad = [{"id": "p000001", "start": 0, "end": 99}]
            items = [{"id": s.id, "text": "x", "script_spans": bad} for s in sents]
            return _chat({"corrections": items})
        if calls["n"] == 4:
            raise httpx.ConnectError("transient drop")
        q_ok = [{"id": "p000001", "quote": "おやすみ。"}]
        return _chat(
            {"corrections": [{"id": "s000002", "text": "おやすみ。", "script_quotes": q_ok}]}
        )

    monkeypatch.setattr("asmr_dubber.translation_script.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        kw: dict[str, Any] = {"source_language": "ja", "target": "source", "provider": "deepseek"}
        corr, report = reconcile_script_sentences(
            sents, ["おやすみ。"], model="test", api_key="test", client=client, **kw
        )
    assert corr == {"s000001": "おはよう", "s000002": "おやすみ。"}
    assert report[0]["needs_review_ids"] == ["s000001"]

    proj2 = _proj(src, source_language="ja")
    asr_rows = [_sent("s000001", 0.0, 1.5, "残す台詞。"), _sent("s000002", 2.0, 4.0, "誤認識。")]
    monkeypatch.setattr(
        pipeline, "_analyze_project_impl", lambda w, *a, **kw: setattr(w, "sentences", asr_rows)
    )
    monkeypatch.setattr(pipeline, "resolve_api_key", lambda *a, **kw: "test-key")
    monkeypatch.setattr(
        pipeline,
        "reconcile_script_sentences",
        lambda *a, **kw: ({"s000001": "", "s000002": "正しい台詞。"}, []),
    )
    kept, _ = pipeline._reconcile_untimed_script(
        proj2, tmp_path, script_lines=["正しい台詞。"], script_language="ja", role="source"
    )
    assert kept[0].source_text == "残す台詞。" and kept[0].enabled
    assert kept[0].script_review_note is not None and kept[1].source_text == "正しい台詞。"
