from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import probe_audio
from asmr_dubber.experimental_mix import compose_speech_replacement
from asmr_dubber.models import DubProject, ProjectSettings, Sentence
from asmr_dubber.services import project_operations, project_records, project_review

RATE = 16000


def _sentence(sid, start, end, **kwargs):
    kwargs.setdefault("tts_file", f"{sid}.wav")
    return Sentence(
        id=sid, start_seconds=start, end_seconds=end, source_text="原文", zh_text="中文", **kwargs
    )


def _noise(rng, seconds, amplitude):
    return rng.uniform(-amplitude, amplitude, (int(seconds * RATE), 2)).astype(np.float32)


def _scene(tmp_path):
    """Four dubbed lines.  The separator removed s1/s2/s3 cleanly but left s4's voice in the
    background, and there is a stretch of unexplained voice after s2."""
    rng = np.random.default_rng(3)
    bed = _noise(rng, 12, 0.003)
    vocals = np.zeros_like(bed)
    for start in (1.0, 3.0, 6.5, 8.5):
        vocals[int(start * RATE) : int((start + 0.8) * RATE)] = _noise(rng, 0.8, 0.2)
    # Loose voice after s2 that no sentence covers.
    vocals[int(4.3 * RATE) : int(5.5 * RATE) - 100] = _noise(rng, 1.2, 0.15)[:-100]
    source = tmp_path / "source.wav"
    voice = tmp_path / "vocals.wav"
    sf.write(source, bed + vocals, RATE, subtype="FLOAT")
    # The estimate of s4 is weak: most of its energy is still in source - estimate.
    estimate = vocals.copy()
    estimate[int(8.5 * RATE) : int(9.3 * RATE)] *= 0.3
    sf.write(voice, estimate, RATE, subtype="FLOAT")
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(replacement_noise_fill=False),
        sentences=[
            _sentence("s1", 1.0, 1.8),
            _sentence("s2", 3.0, 3.8),
            _sentence("s3", 6.5, 7.3),
            _sentence("s4", 8.5, 9.3),
        ],
    )
    return project, source, voice


def test_replacement_report_flags_leftover_voice_and_uncovered_activity(tmp_path):
    project, source, voice = _scene(tmp_path)
    compose_speech_replacement(project, tmp_path, source, voice)
    report = json.loads((tmp_path / "mix/replacement-report.json").read_text("utf-8"))
    assert report["kind"] == "energy_estimate"
    assert "不是识别结果" in report["note"]
    rows = {row["id"]: row for row in report["sentences"]}
    assert set(rows) == {"s1", "s2", "s3", "s4"}
    # Well separated lines are fine; the line whose voice stayed in the background is not.
    assert rows["s1"]["residual_db"] < -30 and not rows["s1"]["exceeds"]
    assert "residual_voice" in rows["s4"]["reasons"]
    assert rows["s4"]["residual_db"] > -9
    # Voice at 4.3-5.5 s has no sentence: s2 sees it after its window, s3 before its own.
    assert rows["s2"]["uncovered_after_seconds"] >= 0.8
    assert "uncovered_voice" in rows["s2"]["reasons"]
    assert rows["s3"]["uncovered_before_seconds"] >= 0.8
    assert report["flagged_ids"] == ["s2", "s3", "s4"]
    assert all(
        {"id", "start", "end", "residual_db", "reasons", "exceeds"} <= set(row)
        for row in rows.values()
    )


def test_replacement_still_works_when_the_report_cannot_be_built(tmp_path, monkeypatch):
    project, source, voice = _scene(tmp_path)

    def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr("asmr_dubber.experimental_mix.residual_voice_report", broken)
    output = compose_speech_replacement(project, tmp_path, source, voice)
    assert output.is_file()
    assert not (tmp_path / "mix/replacement-report.json").exists()


def _write_report(directory, entries):
    path = directory / "mix/replacement-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sentences": entries}), encoding="utf-8")


def test_flags_skip_sentences_the_user_already_keeps_or_removed(tmp_path):
    project = DubProject(
        source=probe_audio(_silent(tmp_path)),
        sentences=[
            _sentence("a", 0, 1),
            _sentence("b", 1, 2, original_audio_enabled=True),
            _sentence("c", 2, 3),
        ],
    )
    flagged = {"exceeds": True, "reasons": ["residual_voice"], "residual_db": -3.0}
    _write_report(
        tmp_path,
        [
            {"id": "a", **flagged},
            {"id": "b", **flagged},
            {"id": "gone", **flagged},
            {"id": "c", "exceeds": False, "reasons": []},
        ],
    )
    assert list(project_review.replacement_flags(project, tmp_path)) == ["a"]
    assert project_review.replacement_flags(project, tmp_path / "missing") == {}
    (tmp_path / "mix/replacement-report.json").write_text("not json", encoding="utf-8")
    assert project_review.replacement_flags(project, tmp_path) == {}


def _silent(tmp_path, seconds=3):
    path = tmp_path / "source.wav"
    if not path.exists():
        sf.write(path, np.zeros((RATE * seconds, 2)), RATE, subtype="FLOAT")
    return path


def test_compare_tracks_list_only_files_that_exist(tmp_path):
    source = _silent(tmp_path)
    project = DubProject(source=probe_audio(source), sentences=[_sentence("a", 0, 1)])
    register = lambda path: f"/media/{path.parent.name}/{path.name}"  # noqa: E731
    assert project_review.compare_tracks(project, tmp_path, "/media/src", register) == [
        {"key": "source", "url": "/media/src"}
    ]
    for variant in ("bilingual", "replace"):
        folder = tmp_path / "output" / variant
        folder.mkdir(parents=True)
        sf.write(folder / "mix.wav", np.zeros((10, 2)), RATE)
        sf.write(folder / "mix__chinese-voice-stem.wav", np.zeros((10, 2)), RATE)
    tracks = project_review.compare_tracks(project, tmp_path, "/media/src", register)
    assert [track["key"] for track in tracks] == ["source", "bilingual", "replacement"]
    assert all("stem" not in track["url"] for track in tracks)
    # The project's current output wins for the mode it was mixed in.
    project.settings.separation_mix_mode = "replace"
    sf.write(tmp_path / "output/current.wav", np.zeros((10, 2)), RATE)
    project.output_file = "output/current.wav"
    tracks = {
        t["key"]: t["url"] for t in project_review.compare_tracks(project, tmp_path, None, register)
    }
    assert tracks["replacement"].endswith("current.wav") and "source" not in tracks
    assert tracks["bilingual"].endswith("mix.wav")


def test_editing_the_translation_drops_the_dub_warning():
    project = DubProject(
        source=probe_audio_stub(),
        sentences=[_sentence("a", 0, 1, tts_warning="配音几乎没有声音")],
    )
    rows = project_records.project_rows(project)
    assert project_records.apply_table(project, rows) is False
    assert project.sentences[0].tts_warning == "配音几乎没有声音"
    rows[0][5] = "新的译文"
    project_records.apply_table(project, rows)
    assert project.sentences[0].tts_warning is None


def probe_audio_stub():
    from asmr_dubber.models import AudioInfo

    return AudioInfo(
        path="source.wav", sha256="0" * 64, duration_seconds=10.0, sample_rate=RATE, channels=2
    )


def test_regenerating_a_line_forces_a_new_take_for_only_that_line(tmp_path, monkeypatch):
    project = DubProject(
        source=probe_audio_stub(),
        sentences=[
            _sentence("a", 0, 1, tts_attempt=1, tts_warning="配音几乎没有声音"),
            _sentence("b", 1, 2, tts_warning="keep"),
        ],
    )
    calls = {}

    def fake_synthesize(active, directory, **kwargs):
        calls.update(kwargs, attempts={s.id: s.tts_attempt for s in active.sentences})

    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.reload_project",
        lambda path: (project, tmp_path),
    )
    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.synthesize_project", fake_synthesize
    )
    monkeypatch.setattr("asmr_dubber.services.project_operations.view", lambda *args: args[-1])
    rows = project_records.project_rows(project)
    message = project_operations.regenerate_dubs(str(tmp_path / "project.json"), rows, ["a"])
    assert calls["force"] is True and calls["sentence_ids"] == ["a"]
    assert calls["attempts"] == {"a": 2, "b": 0}
    assert project.sentences[0].tts_warning is None and project.sentences[1].tts_warning == "keep"
    assert "1" in message
    with pytest.raises(Exception, match="没有可生成配音"):
        project_operations.regenerate_dubs(str(tmp_path / "project.json"), rows, ["zzz"])


def test_frontend_wires_review_flags_filter_and_compare_player():
    from importlib.resources import files

    root = files("asmr_dubber").joinpath("frontend")
    html = root.joinpath("index.html").read_text(encoding="utf-8")
    projects = root.joinpath("projects.js").read_text(encoding="utf-8")
    revise = root.joinpath("project_revise.js").read_text(encoding="utf-8")
    review = root.joinpath("project_review.js").read_text(encoding="utf-8")
    assert 'data-filter="check"' in html and "state.filter === 'check'" in revise
    assert "comparePlayer" in html and "initializeCompare()" in projects
    assert "initializeReview(" in projects and "renderCompare(project)" in projects
    # One-click redo uses the ordinary dubbing task, limited to one sentence.
    assert "startTask('synthesize', { sentence_ids: [sentence.id], force: true })" in review
    for value in ("'default'", "'on'", "'off'"):
        assert value in review


def test_task_contract_accepts_a_single_sentence_redo():
    from asmr_dubber.api_contract import TaskRequest

    request = TaskRequest(
        kind="synthesize", project="p.json", revision=3, sentence_ids=["a"], force=True
    )
    assert request.sentence_ids == ["a"] and request.force


def test_retranslate_only_the_flagged_sentences(tmp_path, monkeypatch):
    project = DubProject(
        source=probe_audio_stub(),
        sentences=[_sentence("a", 0, 1), _sentence("b", 1, 2)],
    )
    calls = {}
    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.reload_project",
        lambda path: (project, tmp_path),
    )
    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.translate_project",
        lambda active, directory, **kwargs: calls.update(kwargs),
    )
    monkeypatch.setattr("asmr_dubber.services.project_operations.resolve_api_key", lambda *a: "k")
    monkeypatch.setattr("asmr_dubber.services.project_operations.view", lambda *args: args[-1])
    rows = project_records.project_rows(project)
    message = project_operations.retranslate(str(tmp_path / "project.json"), rows, ["b", "zzz"])
    assert calls["sentence_ids"] == ["b"] and calls["overwrite_edited"] is True
    assert "1" in message
    review = (
        __import__("importlib.resources")
        .resources.files("asmr_dubber")
        .joinpath("frontend/project_review.js")
        .read_text(encoding="utf-8")
    )
    assert "startTask('translate', { sentence_ids: [again.dataset.retranslate] })" in review
