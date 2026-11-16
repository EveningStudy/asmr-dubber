from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dubber.autoflow import engine, project_cli
from asmr_dubber.autoflow.catalog import _subtitle_content_language, scan_work
from asmr_dubber.models import AudioInfo, DubProject, Sentence, load_project, save_project


def project(tmp_path):
    value = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="a" * 64, duration_seconds=20, sample_rate=48000, channels=1
        ),
        sentences=[
            Sentence(
                id="old", start_seconds=0, end_seconds=1, source_text="旧识别", zh_text="旧重复结果"
            )
        ],
    )
    save_project(value, tmp_path)
    return tmp_path / "project.json"


def transcript(tmp_path, name, text, language, offset):
    path = tmp_path / name
    path.write_text(f"WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n{text}\n", encoding="utf-8")
    return {
        "transcript": str(path),
        "transcript_language": language,
        "transcript_timed": True,
        "transcript_mode": "direct",
        "start_samples": offset * engine.SAMPLE_RATE,
        "duration_samples": 10 * engine.SAMPLE_RATE,
    }


def test_chinese_middle_dot_is_not_kana(tmp_path):
    path = tmp_path / "03.mp3.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n作者莫理斯・盧布朗\n", encoding="utf-8"
    )
    assert _subtitle_content_language(path) == "zh"


def test_japanese_kana_still_detected(tmp_path):
    path = tmp_path / "sub.vtt"
    path.write_text("これは日本語です。", encoding="utf-8")
    assert _subtitle_content_language(path) == "ja"


@pytest.mark.parametrize("second_language", ["zh", "ja", "en"])
def test_all_tracks_with_timed_subtitles_never_fall_back_to_asr(
    tmp_path, monkeypatch, second_language
):
    manifest = project(tmp_path)
    timeline = [
        transcript(tmp_path, "01.mp3.vtt", "第一句。", "zh", 0),
        transcript(tmp_path, "02.mp3.vtt", "Second cue.", second_language, 10),
    ]
    monkeypatch.setattr(
        project_cli,
        "run_asmr_cli",
        lambda *a, **k: pytest.fail("must not call ASR/translation during import"),
    )
    result = engine.import_available_source_transcript(SimpleNamespace(), manifest, timeline)
    imported, _ = load_project(manifest)
    assert result["kind"] == "direct"
    assert len(imported.sentences) == 2
    assert imported.sentences[0].source_text == ""
    assert imported.sentences[0].zh_text == "第一句。"
    assert [(r.start_seconds, r.end_seconds) for r in imported.sentences] == [(1, 2.5), (11, 12.5)]
    assert imported.source_language == second_language
    if second_language == "zh":
        assert all(r.zh_text for r in imported.sentences)
    else:
        assert imported.sentences[1].source_text == "Second cue."
        assert not imported.sentences[1].zh_text


def test_invalid_authoritative_subtitle_fails_without_overwriting_project(tmp_path):
    manifest = project(tmp_path)
    timeline = [
        transcript(tmp_path, "01.vtt", "第一句", "zh", 0),
        transcript(tmp_path, "02.vtt", "第二句", "zh", 10),
    ]
    Path(timeline[1]["transcript"]).write_text("WEBVTT\ninvalid", encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(engine.VideoPreparerError):
        engine.import_available_source_transcript(SimpleNamespace(), manifest, timeline)
    assert manifest.read_bytes() == before


def test_double_extension_pairing_and_chinese_language(tmp_path):
    for index in range(7):
        (tmp_path / f"{index:02}.mp3").write_bytes(b"audio" * 100)
        transcript(tmp_path, f"{index:02}.mp3.vtt", "中文作者莫理斯・盧布朗", "zh", 0)
    scan = scan_work(tmp_path)
    tracks = next(e.tracks for e in scan.editions if len(e.tracks) == 7)
    assert len(tracks) == 7
    assert all(
        t.transcript and t.transcript.language == "zh" and t.transcript.timed for t in tracks
    )


def test_complete_chinese_pipeline_enters_tts_without_asr_or_translation(tmp_path, monkeypatch):
    manifest = project(tmp_path)
    timeline = [
        transcript(tmp_path, "01.vtt", "第一句。", "zh", 0),
        transcript(tmp_path, "02.vtt", "第二句。", "zh", 10),
    ]
    state = {"status": "project_created", "project_json": str(manifest), "timeline": timeline}
    calls = []

    class ReachedTTS(Exception):
        pass

    def cli(paths, command, *args):
        calls.append(command)
        if command == "synthesize":
            loaded, _ = load_project(manifest)
            assert loaded.source_language == "zh"
            assert [r.zh_text for r in loaded.sentences] == ["第一句。", "第二句。"]
            raise ReachedTTS
        pytest.fail(f"Unexpected stage before TTS: {command}")

    monkeypatch.setattr(project_cli, "run_asmr_cli", cli)
    monkeypatch.setattr(project_cli, "project_reference_id", lambda _: "s000001")
    with pytest.raises(ReachedTTS):
        engine.execute_task(SimpleNamespace(), tmp_path, tmp_path / "state.json", state, [])
    assert calls == ["synthesize"]


def test_old_complete_overlay_task_is_not_silently_resumed(tmp_path, monkeypatch):
    manifest = project(tmp_path)
    timeline = [
        transcript(tmp_path, "01.vtt", "第一句。", "zh", 0),
        transcript(tmp_path, "02.vtt", "第二句。", "ja", 10),
    ]
    state = {
        "status": "analyzed",
        "project_json": str(manifest),
        "timeline": timeline,
        "transcript_import": {"kind": "zh_overlay_partial", "count": 2},
    }
    original = manifest.read_bytes()
    monkeypatch.setattr(
        project_cli, "run_asmr_cli", lambda *a: pytest.fail("must stop before synthesis")
    )
    with pytest.raises(engine.VideoPreparerError, match="旧版"):
        engine.execute_task(SimpleNamespace(), tmp_path, tmp_path / "state.json", state, [])
    assert manifest.read_bytes() == original
