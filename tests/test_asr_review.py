"""Quality failure regressions and executable contracts for audio-window review."""

from __future__ import annotations

import json
import threading
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.asr_review import (
    AudioWindow,
    _candidate,
    compare_window,
    normalize_text,
    plan_windows,
    replace_window,
    run_audio_review,
)
from asmr_dubber.errors import OperationCancelledError, ProjectError
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.services.review import apply_review, undo_review


def row(text: str, start: float = 1, end: float = 3, identifier: str = "s1") -> Sentence:
    return Sentence(id=identifier, start_seconds=start, end_seconds=end, source_text=text)


def settings(**kwargs) -> ProjectSettings:
    return ProjectSettings(
        asr_backend="parakeet_nemo",
        asr_model="model",
        asr_review_enabled=True,
        asr_review_models=["faster_whisper|large-v2"],
        **kwargs,
    )


def audio(tmp_path: Path, duration: int = 8) -> Path:
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(16000 * duration, dtype=np.float32), 16000)
    return path


def fake_runner(texts: dict[str, list[Sentence]], calls: list | None = None):
    def run(jobs, configured, language, cancel, progress):
        for identifier, path in jobs:
            if calls is not None:
                calls.append((configured.asr_backend, identifier, path.read_bytes()))
            yield identifier, texts[configured.asr_backend], None

    return run


def test_different_segmentation_and_timestamps_are_not_a_conflict(tmp_path):
    baseline = [row("今日は寒い。", 1, 2), row("早く帰ろう。", 2.5, 5, "s2")]
    before = [s.model_dump() for s in baseline]
    runner = fake_runner(
        {"parakeet_nemo": baseline, "faster_whisper": [row("今日は寒い、早く帰ろう。", 0.3, 6)]}
    )
    calls = []
    runner = fake_runner(
        {"parakeet_nemo": baseline, "faster_whisper": [row("今日は寒い、早く帰ろう。", 0.3, 6)]},
        calls,
    )
    report = tmp_path / "analysis/asr_review.json"
    output = run_audio_review(audio(tmp_path), baseline, settings(), report, runner=runner)
    assert [s.model_dump() for s in output] == before
    result = json.loads(report.read_text(encoding="utf-8"))["results"][0]
    assert result["status"] == "agreement"
    assert len(result["candidates"]) == 1
    assert result["confidence"] is None
    assert calls[0][2] == calls[1][2]


@pytest.mark.parametrize("mode", ["suggest", "conservative"])
def test_negation_is_never_fuzzy_consensus_or_automatic_edit(tmp_path, mode):
    baseline = [row("明日はこの場所に来てください。")]
    other = [row("明日はこの場所に来ないでください。")]
    report = tmp_path / "analysis/asr_review.json"
    output = run_audio_review(
        audio(tmp_path),
        baseline,
        settings(asr_review_mode=mode),
        report,
        runner=fake_runner({"parakeet_nemo": other, "faster_whisper": other}),
    )
    assert output[0].source_text == baseline[0].source_text
    result = json.loads(report.read_text(encoding="utf-8"))["results"][0]
    assert result["status"] == "disagreement"
    assert result["candidates"][0]["protected_change"]


def test_near_match_candidates_remain_distinct():
    w = AudioWindow("w1", 0, 8, 0, 8, "end")
    baseline = [row("明日はこの場所に来てください。")]
    evidence = [
        _candidate(baseline, w, "parakeet_nemo|m", "ja"),
        _candidate([row("明日はこの場所に来ないでください。")], w, "faster_whisper|m", "ja"),
    ]
    assert len(compare_window(w, baseline, evidence, "parakeet_nemo|m", "ja")["candidates"]) == 2


def test_conservative_requires_primary_relisten_and_independent_family(tmp_path):
    baseline = [row("ここは綺麗な花が咲いています。")]
    other = [row("ここは綺麗な桜が咲いています。")]
    output = run_audio_review(
        audio(tmp_path),
        baseline,
        settings(asr_review_mode="conservative"),
        tmp_path / "review.json",
        runner=fake_runner({"parakeet_nemo": other, "faster_whisper": other}),
    )
    assert output[0].source_text == "ここは綺麗な桜が咲いています。"
    assert output[0].review_locked
    cfg = settings(asr_review_mode="conservative")
    cfg.asr_backend = "faster_whisper"
    cfg.asr_review_models = ["kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2"]
    output = run_audio_review(
        tmp_path / "source.wav",
        baseline,
        cfg,
        tmp_path / "other-review.json",
        runner=fake_runner({"faster_whisper": other, "kotoba_whisper": other}),
    )
    assert output[0].source_text == baseline[0].source_text


def test_window_plan_covers_silence_and_has_nonoverlapping_cores(tmp_path):
    windows = plan_windows(audio(tmp_path, 95), 30, 0.5)
    assert windows[0].start == 0 and windows[-1].end == 95
    assert all(a.end == b.start for a, b in pairwise(windows))
    assert all(w.audio_start <= w.start < w.end <= w.audio_end for w in windows)


def test_cross_boundary_proposals_cannot_duplicate_or_drop_neighbors():
    w = AudioWindow("w1", 0, 10, 0, 11, "forced")
    baseline = [row("またね。", 8, 12)]
    result = compare_window(
        w,
        baseline,
        [_candidate([row("またね。", 8, 10)], w, "parakeet_nemo|m", "ja")],
        "parakeet_nemo|m",
        "ja",
    )
    assert result["status"] == "boundary_review"
    with pytest.raises(ProjectError, match="边界"):
        replace_window(baseline, result, result["candidates"][0], settings())


def test_failure_and_retry_preserve_primary_and_reuse_successful_windows(tmp_path):
    baseline = [row("主稿。")]
    report = tmp_path / "analysis/asr_review.json"

    def fail(jobs, cfg, *args):
        for identifier, _ in jobs:
            if cfg.asr_backend == "parakeet_nemo":
                yield identifier, baseline, None
            else:
                yield identifier, None, "backend unavailable"

    output = run_audio_review(audio(tmp_path), baseline, settings(), report, runner=fail)
    assert output == baseline
    assert json.loads(report.read_text(encoding="utf-8"))["state"] == "completed_with_warnings"
    calls = []
    run_audio_review(
        tmp_path / "source.wav",
        baseline,
        settings(),
        report,
        runner=fake_runner({"parakeet_nemo": baseline, "faster_whisper": baseline}, calls),
    )
    assert [c[0] for c in calls] == ["faster_whisper"]


def test_cancellation_checkpoints_completed_evidence(tmp_path):
    baseline = [row("主稿。")]
    report = tmp_path / "analysis/asr_review.json"
    signal = threading.Event()

    def cancel(jobs, *args):
        for identifier, _ in jobs:
            yield identifier, baseline, None
        signal.set()

    with pytest.raises(OperationCancelledError):
        run_audio_review(
            audio(tmp_path), baseline, settings(), report, runner=cancel, cancel_event=signal
        )
    assert json.loads(report.read_text(encoding="utf-8"))["state"] == "cancelled"
    assert list((tmp_path / "analysis/review-v2-cache").glob("*.json"))


def test_cache_invalidates_for_acoustic_settings(tmp_path):
    source = audio(tmp_path)
    baseline = [row("主稿。")]
    report = tmp_path / "analysis/asr_review.json"
    run = fake_runner({"parakeet_nemo": baseline, "faster_whisper": baseline})
    run_audio_review(source, baseline, settings(), report, runner=run)
    calls = []
    run_audio_review(
        source,
        baseline,
        settings(asr_beam_size=7),
        report,
        runner=fake_runner({"parakeet_nemo": baseline, "faster_whisper": baseline}, calls),
    )
    assert len(calls) == 2


def test_apply_undo_and_stale_table_guard(tmp_path, monkeypatch):
    source = audio(tmp_path)
    baseline = [row("こんにちは。")]
    p = DubProject(
        source=AudioInfo(
            path=source.name, sha256="a" * 64, duration_seconds=8, sample_rate=16000, channels=1
        ),
        settings=settings(),
        sentences=baseline,
    )
    save_project(p, tmp_path)
    report_path = tmp_path / "analysis/asr_review.json"
    run_audio_review(
        source,
        baseline,
        p.settings,
        report_path,
        runner=fake_runner({"parakeet_nemo": baseline, "faster_whisper": [row("こんばんは。")]}),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = next(c for c in report["results"][0]["candidates"] if c["text"] == "こんばんは。")
    monkeypatch.setattr("asmr_dubber.services.review.view", lambda p, *a: p)
    accepted = apply_review(str(tmp_path), "w000001", candidate["id"])
    assert (
        accepted.sentences[0].source_text == "こんばんは。" and accepted.sentences[0].review_locked
    )
    restored = undo_review(str(tmp_path))
    assert restored.sentences[0].source_text == "こんにちは。"
    restored.sentences[0].source_text = "人工修改。"
    save_project(restored, tmp_path)
    with pytest.raises(ProjectError, match="主稿已改变"):
        apply_review(str(tmp_path), "w000001", candidate["id"])


def test_numbers_signs_and_decimals_are_not_collapsed():
    assert normalize_text("1.5") != normalize_text("15")
    assert normalize_text("-3") != normalize_text("3")
    assert normalize_text("can't") != normalize_text("cant")


def test_recognition_session_reuses_and_releases_model(monkeypatch):
    from asmr_dubber.asr import _session_model, recognition_session

    created = []
    monkeypatch.setattr("asmr_dubber.asr._cleanup_cuda", lambda: None)

    def factory():
        obj = object()
        created.append(obj)
        return obj

    with recognition_session():
        assert _session_model("test", factory) is _session_model("test", factory)
    with recognition_session():
        _session_model("test", factory)
    assert len(created) == 2


def test_primary_is_persisted_before_review_failure(tmp_path, monkeypatch):
    from asmr_dubber import pipeline

    source = audio(tmp_path)
    cfg = settings()
    cfg.asr_model = "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
    p = DubProject(
        source=AudioInfo(
            path=source.name, sha256="a" * 64, duration_seconds=8, sample_rate=16000, channels=1
        ),
        settings=cfg,
    )
    monkeypatch.setattr(pipeline, "verify_source", lambda *args: source)
    monkeypatch.setattr(
        pipeline, "transcribe_source", lambda *args, **kwargs: ([row("主稿已完成。")], "ja")
    )

    def fail(audio, baseline, settings, path, **kwargs):
        saved, _ = load_project(tmp_path)
        assert saved.sentences[0].source_text == "主稿已完成。"
        raise OperationCancelledError("cancelled")

    monkeypatch.setattr(pipeline, "run_audio_review", fail)
    with pytest.raises(OperationCancelledError):
        pipeline._analyze_project_impl(p, tmp_path)
    assert load_project(tmp_path)[0].sentences[0].source_text == "主稿已完成。"


def test_new_row_inside_window_prevents_old_proposal_application():
    baseline = [row("原稿。")]
    w = AudioWindow("w1", 0, 8, 0, 8, "end")
    result = compare_window(
        w,
        baseline,
        [_candidate([row("候補。")], w, "faster_whisper|m", "ja")],
        "parakeet_nemo|m",
        "ja",
    )
    with pytest.raises(ProjectError, match="新增"):
        replace_window(
            [*baseline, row("新增。", 5, 6, "s2")], result, result["candidates"][0], settings()
        )


def test_main_model_omission_is_visible_without_automatic_insert(tmp_path):
    baseline = []
    path = tmp_path / "review.json"
    output = run_audio_review(
        audio(tmp_path),
        baseline,
        settings(asr_review_mode="conservative"),
        path,
        runner=fake_runner(
            {"parakeet_nemo": [row("聞こえる。")], "faster_whisper": [row("聞こえる。")]}
        ),
    )
    result = json.loads(path.read_text(encoding="utf-8"))["results"][0]
    assert output == [] and result["status"] == "disagreement"
    assert result["candidates"] and result["action"] == "keep_baseline"


def test_large_rewrite_is_never_auto_applied(tmp_path):
    baseline = [row("今日はここに来ました。")]
    other = [row("それでは明日の予定を説明しましょう。")]
    result = run_audio_review(
        audio(tmp_path),
        baseline,
        settings(asr_review_mode="conservative"),
        tmp_path / "review.json",
        runner=fake_runner({"parakeet_nemo": other, "faster_whisper": other}),
    )
    assert result == baseline


def test_empty_review_window_is_valid_not_a_backend_failure(monkeypatch):
    from asmr_dubber.asr import _finish_tokens, recognition_session

    monkeypatch.setattr("asmr_dubber.asr._cleanup_cuda", lambda: None)
    with recognition_session():
        assert _finish_tokens([], "", "ja", settings()) == ([], "ja")


def test_english_word_boundaries_are_meaningful():
    assert normalize_text("now here") != normalize_text("nowhere")
    assert normalize_text("Today is cold. We should go.") == normalize_text(
        "Today is cold We should go"
    )


def test_alignment_preserves_ids_text_and_rejects_large_drift(tmp_path, monkeypatch):
    from asmr_dubber.audio import probe_audio
    from asmr_dubber.services.project_records import project_rows
    from asmr_dubber.services.review import align_review

    source = audio(tmp_path)
    info = probe_audio(source)
    info.path = source.name
    p = DubProject(
        source=info, settings=settings(), sentences=[row("第一句。"), row("第二句。", 4, 5, "s2")]
    )
    save_project(p, tmp_path)

    def align(audio, rows, *args, **kwargs):
        rows[0].start_seconds = 6
        rows[0].end_seconds = 7
        rows[1].start_seconds = 4.1
        for i, r in enumerate(rows):
            r.id = f"renamed{i}"
        return []

    monkeypatch.setattr("asmr_dubber.forced_alignment.align_sentences_with_qwen", align)
    monkeypatch.setattr("asmr_dubber.services.review.view", lambda p, *args: p)
    monkeypatch.setattr("asmr_dubber.services.review.portable_home", lambda: tmp_path)
    result = align_review(str(tmp_path), project_rows(p))
    assert [(r.id, r.source_text) for r in result.sentences] == [
        ("s1", "第一句。"),
        ("s2", "第二句。"),
    ]
    assert result.sentences[0].start_seconds == 1
    assert result.sentences[1].start_seconds == 4.1
