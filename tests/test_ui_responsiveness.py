from __future__ import annotations

import time

from asmr_dubber.asr_progress import model_heartbeat
from asmr_dubber.asr_review import AudioWindow, _candidate, compare_window, recognize_windows
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence
from asmr_dubber.ui_services import ProjectView, apply_table


def test_heartbeat_is_truthful_and_stops():
    events = []
    with model_heartbeat(lambda *args: events.append(args), "加载", interval=0.01) as progress:
        assert progress
        progress("解码", 3, 10)
        time.sleep(0.035)
    count = len(events)
    time.sleep(0.03)
    assert len(events) == count
    assert any("仍在处理" in event[0] for event in events)
    assert all(event[1:] == (3, 10) for event in events[1:])


def test_parakeet_one_bad_window_does_not_discard_good_ones(monkeypatch, tmp_path):
    sentence = Sentence(id="s1", start_seconds=0, end_seconds=1, source_text="测试")

    def process(*args, window_results, window_errors, **kwargs):
        window_results["w1"] = [sentence]
        window_results["w3"] = []
        window_errors["w2"] = "bad timestamps"

    monkeypatch.setattr("asmr_dubber.asr._transcribe_parakeet", process)
    results = list(
        recognize_windows(
            [(f"w{i}", tmp_path / f"w{i}.wav") for i in range(1, 4)], ProjectSettings(), "ja", None
        )
    )
    assert results[0][1] == [sentence]
    assert results[1][1] is None
    assert results[2][1] == []


def test_bad_timestamp_preserves_text_but_cannot_auto_apply():
    window = AudioWindow("w1", 0, 10, 0, 10, "end")
    rows = [Sentence(id="s1", start_seconds=2, end_seconds=12, source_text="听到的文字")]
    candidate = _candidate(rows, window, "kotoba_whisper|model", "ja")
    assert candidate["text"] == "听到的文字"
    assert candidate["timing_warning"]
    assert candidate["sentences"][0]["end_seconds"] == 10
    compared = compare_window(window, [], [candidate], "parakeet_nemo|m", "ja")
    assert not compared["candidates"][0]["applicable"]


def test_text_table_keeps_booleans_numbers_and_full_content():
    project = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="a" * 64, duration_seconds=2000, sample_rate=16000, channels=1
        )
    )
    long_text = "完整文本" * 200
    rows = [
        [f"s{i}", "true" if i % 2 else "false", str(i), str(i + 0.8), long_text, ""]
        for i in range(500)
    ]
    apply_table(project, rows)
    assert len(project.sentences) == 500
    assert sum(s.enabled for s in project.sentences) == 250
    assert all(s.source_text == long_text for s in project.sentences)
    assert project.sentences[499].end_seconds == 499.8


def test_view_does_not_decode_media_to_return_table(monkeypatch):
    import asmr_dubber.ui as ui

    def picker(path, *, include_preview=True):
        assert not include_preview
        return [], None, None

    monkeypatch.setattr(ui, "reference_picker", picker)
    view = ProjectView(
        "project.json",
        "ja",
        [["s1", True, 0, 1, "source", ""]],
        None,
        None,
        None,
        [],
        None,
        "",
        "ok",
    )
    assert ui._view_values(view)[2] == [["s1", "true", "0", "1", "source", ""]]
