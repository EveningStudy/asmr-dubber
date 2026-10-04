from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dubber.autoflow import engine
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import ProjectSettings
from asmr_dubber.services.batch_catalog import (
    scan_for_ui,
)
from asmr_dubber.services.batch_configuration import config_from_settings
from asmr_dubber.services.batch_plan import build_plan_for_ui, deserialize_plan, edit_plan_for_ui
from asmr_dubber.services.batch_queue import (
    add_plan_to_queue,
    queue_items_for_ui,
    remove_plan_from_queue,
    reorder_queue_for_ui,
    replace_plan_in_queue,
    toggle_plan_rebuild,
)
from asmr_dubber.user_settings import UserSettings


def _work(root: Path) -> Path:
    work = root / "RJ测试作品"
    wav = work / "WAV"
    mp3 = work / "MP3"
    bonus = work / "特典"
    wav.mkdir(parents=True)
    mp3.mkdir()
    bonus.mkdir()
    (wav / "Track01 開場.wav").write_bytes(b"wav-1")
    (wav / "Track02 耳かき.wav").write_bytes(b"wav-2")
    (mp3 / "Track01 開場.mp3").write_bytes(b"mp3-1")
    (mp3 / "Track02 耳かき.mp3").write_bytes(b"mp3-2")
    (bonus / "EX01 おまけ.wav").write_bytes(b"bonus")
    (wav / "Track01 開場.wav.vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\n始まります。\n",
        encoding="utf-8",
    )
    (wav / "Track02 耳かき.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n现在开始掏耳朵。\n",
        encoding="utf-8",
    )
    (wav / "Track01 開場.txt").write_text("这个文本文件不会作为字幕。", encoding="utf-8")
    (wav / "Track02 耳かき.pdf").write_bytes(b"not-a-real-pdf")
    (work / "cover.jpg").write_bytes(b"image")
    return work


def test_autoflow_queue_can_reorder_edit_remove_and_restart(tmp_path: Path) -> None:
    settings = UserSettings()
    first_work = _work(tmp_path / "first")
    second_work = _work(tmp_path / "second")
    first_scan = scan_for_ui(first_work, settings=settings)
    second_scan = scan_for_ui(second_work, settings=settings)
    first_payload = build_plan_for_ui(
        first_work,
        first_scan.selected_edition,
        first_scan.source_payloads,
        "audio",
        "merged",
        "black",
        False,
        False,
        settings=settings,
    )
    second_payload = build_plan_for_ui(
        second_work,
        second_scan.selected_edition,
        second_scan.source_payloads,
        "audio",
        "merged",
        "black",
        False,
        False,
        settings=settings,
    )
    queue = add_plan_to_queue(add_plan_to_queue([], first_payload), second_payload)
    first_id = str(first_payload["plan_id"])
    second_id = str(second_payload["plan_id"])

    queue = reorder_queue_for_ui(queue, [second_id, first_id])
    assert [item["id"] for item in queue_items_for_ui(queue)] == [second_id, first_id]

    runtime_items = queue_items_for_ui(
        queue,
        runtime={
            second_id: {
                "reference_ready": True,
                "request_id": "request-1",
                "status": "等待参考音频",
            }
        },
    )
    assert runtime_items[0]["reference_ready"] is True
    assert runtime_items[0]["reference_request_id"] == "request-1"
    assert runtime_items[0]["reference_status"] == "等待参考音频"
    assert runtime_items[1]["reference_ready"] is False

    queue = toggle_plan_rebuild(queue, second_id)
    assert queue_items_for_ui(queue)[0]["rebuild"] is True
    edit_view = edit_plan_for_ui(queue, second_id, settings=settings)
    assert edit_view.folder == str(second_work.resolve())
    assert edit_view.rebuild is True

    updated_second = build_plan_for_ui(
        second_work,
        second_scan.selected_edition,
        second_scan.source_payloads,
        "audio",
        "separate",
        "black",
        False,
        False,
        settings=settings,
    )
    queue = replace_plan_in_queue(queue, second_id, updated_second)
    assert deserialize_plan(queue[0]).layout == engine.LAYOUT_SEPARATE

    queue = remove_plan_from_queue(queue, updated_second["plan_id"])
    assert [deserialize_plan(item).plan_id for item in queue] == [first_id]


def test_autoflow_can_keep_original_work_and_track_titles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _work(tmp_path)
    scanned = scan_for_ui(work, settings=UserSettings())
    sources = [engine.deserialize_audio_source(item) for item in scanned.source_payloads]
    saved: dict[str, object] = {}
    monkeypatch.setattr(engine, "load_plan_metadata", lambda _plan_id: dict(saved))
    monkeypatch.setattr(engine, "save_plan_metadata", lambda _plan_id, value: saved.update(value))
    monkeypatch.setattr(
        engine,
        "translate_titles",
        lambda *_args, **_kwargs: pytest.fail("关闭标题翻译时不应调用翻译服务"),
    )

    folder_title, track_titles = engine.translated_plan_titles(
        "no-title-translation",
        work,
        sources,
        SimpleNamespace(),
        translate_work_title=False,
        translate_track_titles=False,
    )

    assert folder_title == work.name
    assert list(track_titles.values()) == [source.title_ja for source in sources]
    assert saved["title_translation_policy"] == {"work": False, "tracks": False}


def test_autoflow_title_translation_uses_language_neutral_sentence_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "日语作品"
    work.mkdir()
    captured: list[str] = []

    def fake_translate(sentences, **_kwargs) -> None:
        captured.extend(sentence.source_text for sentence in sentences)
        sentences[0].zh_text = "中文作品"
        sentences[1].zh_text = "第一轨"

    monkeypatch.setattr("asmr_dubber.translation.translate_sentences", fake_translate)
    monkeypatch.setattr("asmr_dubber.user_settings.load_user_settings", UserSettings)
    monkeypatch.setattr("asmr_dubber.user_settings.resolve_api_key", lambda _provider: "key")
    state = {
        "source_folder": str(work),
        "folder_name_original": work.name,
        "timeline": [
            {
                "filename": "01.wav",
                "relative_path": "01.wav",
                "title_ja": "一番目",
                "source_language": "ja",
            }
        ],
    }

    translated = engine.translate_titles(state, SimpleNamespace())

    assert captured == ["日语作品", "一番目"]
    assert state["folder_name_translation"] == "中文作品"
    assert translated == {"01.wav": "第一轨"}


@pytest.mark.parametrize(("language", "expected"), [("zh", "zh"), ("invalid", "auto")])
def test_autoflow_shared_reference_normalizes_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    expected: str,
) -> None:
    from asmr_dubber import models

    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"reference")
    project = SimpleNamespace(settings=ProjectSettings())
    saved: list[object] = []
    monkeypatch.setattr(models, "load_project", lambda _path: (project, tmp_path))
    monkeypatch.setattr(models, "save_project", lambda value, _directory: saved.append(value))

    engine.apply_shared_reference(
        tmp_path / "project.json",
        {"audio": str(audio), "text": "参考", "language": language},
    )

    assert project.settings.tts_external_reference_language == expected
    assert saved == [project]


def test_autoflow_settings_reject_unsafe_output_folder_name() -> None:
    with pytest.raises(ProjectError, match="Windows 文件名"):
        config_from_settings(UserSettings(autoflow_output_folder_name="bad:name"))


def test_single_timed_chinese_subtitle_is_imported_without_asr_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "Track01 中文.srt"
    transcript.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
        encoding="utf-8",
    )
    project_json = tmp_path / "project.json"
    project_json.write_text("{}", encoding="utf-8")
    project = SimpleNamespace()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "asmr_dubber.models.load_project",
        lambda _path: (project, tmp_path),
    )

    def fake_import(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"format": "SRT", "language": "zh", "timed": True, "sentences": 1}

    monkeypatch.setattr("asmr_dubber.pipeline.import_project_transcript", fake_import)
    result = engine.import_available_source_transcript(
        SimpleNamespace(asmr_home=tmp_path),
        project_json,
        [
            {
                "transcript": str(transcript),
                "transcript_language": "zh",
                "transcript_timed": True,
                "start_samples": 0,
                "duration_samples": 2 * engine.SAMPLE_RATE,
            }
        ],
    )

    assert result["kind"] == "direct"
    assert result["language"] == "zh"
    assert captured["script_language"] == "zh"
    assert captured["transcript_path"] == transcript.resolve()


def test_single_timed_chinese_subtitle_can_use_asr_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "Track01 中文.lrc"
    transcript.write_text("[00:00.00]你好\n", encoding="utf-8")
    project_json = tmp_path / "project.json"
    project_json.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "asmr_dubber.models.load_project",
        lambda _path: (SimpleNamespace(), tmp_path),
    )

    def fake_import(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "format": "LRC（仅文字）",
            "language": "zh",
            "timed": False,
            "sentences": 1,
            "script_reconciled": True,
        }

    monkeypatch.setattr("asmr_dubber.pipeline.import_project_transcript", fake_import)
    result = engine.import_available_source_transcript(
        SimpleNamespace(asmr_home=tmp_path),
        project_json,
        [
            {
                "transcript": str(transcript),
                "transcript_language": "zh",
                "transcript_timed": True,
                "transcript_mode": engine.TRANSCRIPT_MODE_ASR_RECONCILE,
                "start_samples": 0,
                "duration_samples": 2 * engine.SAMPLE_RATE,
            }
        ],
    )

    assert result["script_reconciled"] is True
    assert captured["plain_timing"] == "script_review"
    assert captured["use_embedded_timing"] is False


def test_multitrack_plain_scripts_wait_for_asr_reconciliation(tmp_path: Path) -> None:
    first = tmp_path / "01.txt"
    second = tmp_path / "02.txt"
    first.write_text("第一句。", encoding="utf-8")
    second.write_text("第二句。", encoding="utf-8")

    result = engine.import_available_source_transcript(
        SimpleNamespace(asmr_home=tmp_path),
        tmp_path / "project.json",
        [
            {
                "transcript": str(first),
                "transcript_language": "zh",
                "transcript_timed": False,
                "transcript_mode": engine.TRANSCRIPT_MODE_ASR_RECONCILE,
            },
            {
                "transcript": str(second),
                "transcript_language": "zh",
                "transcript_timed": False,
                "transcript_mode": engine.TRANSCRIPT_MODE_ASR_RECONCILE,
            },
        ],
    )

    assert result == {"kind": "reconcile_after_asr", "count": 2}


def test_reconcile_timeline_scripts_uses_each_track_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "02.txt"
    script.write_text("第二句。", encoding="utf-8")
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "asmr_dubber.models.load_project",
        lambda _path: (SimpleNamespace(), tmp_path),
    )

    def fake_reconcile(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.append(dict(kwargs))
        return {"matched_sentences": 1}

    monkeypatch.setattr("asmr_dubber.pipeline.reconcile_analyzed_project_script", fake_reconcile)
    results = engine.reconcile_timeline_scripts(
        tmp_path / "project.json",
        [
            {
                "transcript": str(script),
                "transcript_language": "zh",
                "transcript_mode": engine.TRANSCRIPT_MODE_DIRECT,
                "start_samples": 0,
                "duration_samples": engine.SAMPLE_RATE,
            },
            {
                "transcript": str(script),
                "transcript_language": "zh",
                "transcript_mode": engine.TRANSCRIPT_MODE_ASR_RECONCILE,
                "start_samples": 2 * engine.SAMPLE_RATE,
                "duration_samples": 3 * engine.SAMPLE_RATE,
            },
        ],
    )

    assert results == [{"matched_sentences": 1}]
    assert captured[0]["start_seconds"] == 2.0
    assert captured[0]["end_seconds"] == 5.0


def test_complete_multitrack_chinese_subtitles_replace_asr_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "Track01 中文.srt"
    second = tmp_path / "Track02 中文.srt"
    first.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n第一句\n",
        encoding="utf-8",
    )
    second.write_text(
        "1\n00:00:00,500 --> 00:00:01,500\n第二句\n",
        encoding="utf-8",
    )
    project_json = tmp_path / "project.json"
    project_json.write_text("{}", encoding="utf-8")
    project = SimpleNamespace(
        sentences=[],
        source_language="ja",
        settings=ProjectSettings(),
        asr_language=None,
        asr_settings_dirty=True,
        chinese_stem_file="old.wav",
        output_file="old.wav",
        output_video_file="old.mp4",
        subtitle_srt_file="old.srt",
        subtitle_lrc_file="old.lrc",
        subtitle_video_file="old-subtitle.mp4",
    )
    saved: list[object] = []
    exported: list[object] = []
    monkeypatch.setattr(
        "asmr_dubber.models.load_project",
        lambda _path: (project, tmp_path),
    )
    monkeypatch.setattr(
        "asmr_dubber.models.save_project",
        lambda value, _directory: saved.append(value),
    )
    monkeypatch.setattr(
        "asmr_dubber.pipeline.export_transcript",
        lambda value, _directory: exported.append(value),
    )

    result = engine.import_available_source_transcript(
        SimpleNamespace(asmr_home=tmp_path),
        project_json,
        [
            {
                "transcript": str(first),
                "transcript_language": "zh",
                "transcript_timed": True,
                "start_samples": 0,
                "duration_samples": 2 * engine.SAMPLE_RATE,
            },
            {
                "transcript": str(second),
                "transcript_language": "zh",
                "transcript_timed": True,
                "start_samples": 2 * engine.SAMPLE_RATE,
                "duration_samples": 2 * engine.SAMPLE_RATE,
            },
        ],
    )

    assert result == {"kind": "direct", "language": "zh", "sentences": 2}
    assert project.source_language == "zh"
    assert [sentence.zh_text for sentence in project.sentences] == ["第一句", "第二句"]
    assert [sentence.source_text for sentence in project.sentences] == ["", ""]
    assert [(sentence.start_seconds, sentence.end_seconds) for sentence in project.sentences] == [
        (0.0, 1.0),
        (2.5, 3.5),
    ]
    assert project.chinese_stem_file is None
    assert project.output_file is None
    assert saved == [project]
    assert exported == [project]


def test_legacy_complete_chinese_overlay_is_rewound_for_direct_import() -> None:
    state = {
        "status": "completed",
        "transcript_import": {"kind": "zh_overlay"},
        "timeline": [
            {
                "transcript": "Track01.srt",
                "transcript_language": "zh",
                "transcript_timed": True,
            }
        ],
        "chinese_transcript_overlay_done": True,
        "outputs": {"audio": "mixed.wav"},
    }

    assert engine.legacy_chinese_timeline_needs_reimport(state) is True
    engine.reset_legacy_chinese_timeline_state(state)

    assert state["status"] == "project_created"
    assert "chinese_transcript_overlay_done" not in state
    assert "outputs" not in state
