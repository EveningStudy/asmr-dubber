from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dubber.autoflow import engine
from asmr_dubber.autoflow.ui_services import (
    add_plan_to_queue,
    build_plan_for_ui,
    config_from_settings,
    deserialize_plan,
    edit_plan_for_ui,
    preview_edition_for_ui,
    queue_choices,
    queue_items_for_ui,
    queue_rows,
    remove_plan_from_queue,
    reorder_queue_for_ui,
    reorder_tracks_for_ui,
    replace_plan_in_queue,
    scan_for_ui,
    set_track_subtitle_for_ui,
    subtitle_output_rows,
    toggle_plan_rebuild,
)
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import ProjectSettings
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


def test_autoflow_scan_prefers_configured_format_and_previews_bonus(tmp_path: Path) -> None:
    work = _work(tmp_path)
    settings = UserSettings(
        autoflow_preferred_audio_formats="wav,flac,mp3",
        autoflow_include_bonus=False,
    )

    scanned = scan_for_ui(work, settings=settings)

    assert scanned.folder == str(work.resolve())
    assert "WAV" in scanned.edition_choices[0][0]
    assert [item["path"] for item in scanned.track_items] == [
        "WAV/Track01 開場.wav",
        "WAV/Track02 耳かき.wav",
    ]
    assert scanned.selected_background == "cover.jpg"
    assert scanned.selected_background_preview == str((work / "cover.jpg").resolve())
    assert "auto" not in {value for _label, value in scanned.background_choices}
    assert len(scanned.source_payloads) == 2
    subtitle_values = {
        choice["value"]
        for item in scanned.track_items
        for choice in item["transcript_choices"]
        if choice["value"]
    }
    assert any(value.endswith(".txt") for value in subtitle_values)
    assert all(not value.endswith(".pdf") for value in subtitle_values)

    view = preview_edition_for_ui(
        work,
        scanned.selected_edition,
        True,
        settings=settings,
    )
    assert len(view.track_items) == 3
    assert view.track_items[-1]["category"] == "特典"
    assert "3 条音轨" in view.summary


def test_autoflow_video_only_folder_has_specific_guidance(tmp_path: Path) -> None:
    work = tmp_path / "只有视频"
    work.mkdir()
    (work / "01.mp4").write_bytes(b"video")

    with pytest.raises(ProjectError, match=r"检测到视频文件.*单个视频请使用‘单个作品’"):
        scan_for_ui(work, settings=UserSettings())


def test_autoflow_plain_script_uses_asr_timing_and_timed_subtitle_can_opt_in(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    scanned = scan_for_ui(work, settings=UserSettings())
    track = scanned.track_items[0]

    plain = set_track_subtitle_for_ui(
        work,
        scanned.source_payloads,
        track["id"],
        "WAV/Track01 開場.txt",
        "zh",
        "direct",
        settings=UserSettings(),
    )
    plain_source = engine.deserialize_audio_source(plain.source_payloads[0])
    assert plain_source.transcript_timed is False
    assert plain_source.transcript_mode == engine.TRANSCRIPT_MODE_ASR_RECONCILE

    timed = set_track_subtitle_for_ui(
        work,
        plain.source_payloads,
        track["id"],
        "WAV/Track01 開場.wav.vtt",
        "ja",
        "asr_reconcile",
        settings=UserSettings(),
    )
    timed_source = engine.deserialize_audio_source(timed.source_payloads[0])
    assert timed_source.transcript_timed is True
    assert timed_source.transcript_mode == engine.TRANSCRIPT_MODE_ASR_RECONCILE


def test_subtitle_output_rows_lists_srt_and_lrc_paths(tmp_path: Path) -> None:
    output = tmp_path / "AutoFlow输出"
    nested = output / "合并版"
    nested.mkdir(parents=True)
    (nested / "双语版.srt").write_text("srt", encoding="utf-8")
    (nested / "双语版.lrc").write_text("lrc", encoding="utf-8")
    (nested / "双语版.mp4").write_bytes(b"video")

    rows = subtitle_output_rows([str(output)])

    assert [row[1] for row in rows] == ["LRC", "SRT"]
    assert all(Path(row[2]).is_file() for row in rows)


def test_autoflow_reference_wait_defaults_to_one_minute_and_can_be_disabled() -> None:
    default_config = config_from_settings(UserSettings())
    custom_config = config_from_settings(UserSettings(autoflow_reference_wait_seconds=180))
    disabled_config = config_from_settings(
        UserSettings(
            autoflow_reference_wait_enabled=False,
            autoflow_reference_wait_seconds=180,
        )
    )

    assert default_config.reference_wait_seconds == 60
    assert custom_config.reference_wait_seconds == 180
    assert disabled_config.reference_wait_seconds == 0


def test_reference_wait_emits_best_effort_ui_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_json = tmp_path / "project.json"
    project_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine, "project_reference_id", lambda _path: "s000007")
    monkeypatch.setattr(engine, "project_has_external_reference", lambda _path: False)
    events: list[dict[str, object]] = []

    assert engine.wait_for_reference(
        SimpleNamespace(),  # type: ignore[arg-type]
        project_json,
        timeout_seconds=60,
        event_callback=events.append,
        event_context={"plan_id": "plan-1", "work": "测试作品"},
        launch_ui=False,
    )

    assert [event["kind"] for event in events] == ["ready", "selected"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[0]["plan_id"] == "plan-1"
    assert events[1]["sentence_id"] == "s000007"


def test_reference_notification_failure_never_stops_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_json = tmp_path / "project.json"
    project_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine, "project_reference_id", lambda _path: "s000001")
    monkeypatch.setattr(engine, "project_has_external_reference", lambda _path: False)
    monkeypatch.setattr(engine, "log_event", lambda _message: None)

    def broken_notification(_payload: dict[str, object]) -> None:
        raise RuntimeError("browser closed")

    assert engine.wait_for_reference(
        SimpleNamespace(),  # type: ignore[arg-type]
        project_json,
        timeout_seconds=60,
        event_callback=broken_notification,
        launch_ui=False,
    )


def test_autoflow_plan_preserves_track_order_and_per_track_subtitles(tmp_path: Path) -> None:
    work = _work(tmp_path)
    settings = UserSettings(
        autoflow_include_bonus=False,
        autoflow_translate_work_title=False,
        autoflow_translate_track_titles=False,
    )
    scanned = scan_for_ui(work, settings=settings)
    reversed_ids = [item["id"] for item in reversed(scanned.track_items)]
    reordered = reorder_tracks_for_ui(
        work,
        scanned.source_payloads,
        reversed_ids,
        settings=settings,
    )

    payload = build_plan_for_ui(
        work,
        scanned.selected_edition,
        reordered.source_payloads,
        "audio",
        "separate",
        "black",
        True,
        False,
        settings=settings,
    )
    plan = deserialize_plan(payload)

    assert plan.mode == engine.MODE_AUDIO
    assert plan.layout == engine.LAYOUT_SEPARATE
    assert plan.background is None
    assert plan.embed_subtitles is False
    assert [source.path.name for source in plan.sources] == [
        "Track02 耳かき.wav",
        "Track01 開場.wav",
    ]
    assert [source.order for source in plan.sources] == [1, 2]
    assert plan.sources[0].transcript_language == "zh"
    assert plan.sources[1].transcript_language == "ja"
    assert plan.translate_work_title is False
    assert plan.translate_track_titles is False

    without_subtitle = set_track_subtitle_for_ui(
        work,
        reordered.source_payloads,
        reordered.track_items[0]["id"],
        "",
        "zh",
        settings=settings,
    )
    without_subtitle_payload = build_plan_for_ui(
        work,
        scanned.selected_edition,
        without_subtitle.source_payloads,
        "audio",
        "separate",
        "black",
        True,
        False,
        settings=settings,
    )
    assert deserialize_plan(without_subtitle_payload).plan_id != plan.plan_id

    queue = add_plan_to_queue([], payload)
    assert queue_rows(queue)[0][1:5] == [
        "RJ测试作品",
        2,
        "纯音频",
        "每条音轨分别处理并输出（不合并）",
    ]
    assert queue_choices(queue)[0][1] == plan.plan_id
    with pytest.raises(ProjectError, match="已经在队列"):
        add_plan_to_queue(queue, payload)


def test_autoflow_subtitle_only_plan_is_persisted_and_has_distinct_identity(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    settings = UserSettings()
    scanned = scan_for_ui(work, settings=settings)
    regular = build_plan_for_ui(
        work,
        scanned.selected_edition,
        scanned.source_payloads,
        "video_normal",
        "merged",
        scanned.selected_background,
        True,
        False,
        False,
        settings=settings,
    )
    subtitle_only = build_plan_for_ui(
        work,
        scanned.selected_edition,
        scanned.source_payloads,
        "video_normal",
        "merged",
        scanned.selected_background,
        True,
        False,
        True,
        settings=settings,
    )

    plan = deserialize_plan(subtitle_only)
    assert plan.subtitles_only is True
    assert plan.embed_subtitles is True
    assert plan.plan_id != deserialize_plan(regular).plan_id
    assert queue_items_for_ui([subtitle_only])[0]["mode"] == "仅字幕 · 普通静态视频"


def test_autoflow_subtitle_only_execution_skips_reference_tts_and_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_json = tmp_path / "project.json"
    project_json.write_text("{}", encoding="utf-8")
    original = tmp_path / "原声.flac"
    original.write_bytes(b"audio")
    master = tmp_path / "master.wav"
    master.write_bytes(b"master")
    state_file = tmp_path / "state.json"
    state = {
        "schema": 1,
        "status": "awaiting_reference",
        "mode": engine.MODE_AUDIO,
        "subtitles_only": True,
        "embed_subtitles": False,
        "project_json": str(project_json),
        "original_media": str(original),
        "master_audio": str(master),
        "harmonized_delay_seconds": 0,
        "harmonized_volume_db": -10.0,
        "timeline": [
            {
                "filename": "01.wav",
                "relative_path": "01.wav",
                "title_ja": "第一轨",
                "start_samples": 0,
                "duration_samples": engine.SAMPLE_RATE,
            }
        ],
        "title_translations": {"01.wav": "第一轨"},
        "folder_name_original": "作品",
        "folder_name_translation": "作品",
        "source_folder": str(tmp_path),
        "outputs": {},
    }
    commands: list[str] = []

    monkeypatch.setattr(
        engine,
        "run_asmr_cli",
        lambda _paths, command, *_arguments: commands.append(command),
    )
    monkeypatch.setattr(
        engine,
        "wait_for_reference",
        lambda *_args, **_kwargs: pytest.fail("仅字幕任务不应等待参考音频"),
    )
    monkeypatch.setattr(
        engine,
        "copy_subtitle_only_outputs",
        lambda *_args, **_kwargs: {
            "original": str(original),
            "srt": str(tmp_path / "双语版.srt"),
            "lrc": str(tmp_path / "双语版.lrc"),
        },
    )

    def fake_timestamp(_state: dict[str, object], folder: Path) -> Path:
        path = folder / "时间戳.txt"
        path.write_text("00:00:00 第一轨\n", encoding="utf-8")
        return path

    monkeypatch.setattr(engine, "write_timestamp_document", fake_timestamp)

    engine.execute_task(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path,
        state_file,
        state,
        [],
    )

    assert commands == ["subtitles"]
    assert state["status"] == "completed"
    assert "audio" not in state["outputs"]


def test_autoflow_requires_explicit_rebuild_before_replacing_other_plan(tmp_path: Path) -> None:
    work = _work(tmp_path)
    settings = UserSettings()
    scanned = scan_for_ui(work, settings=settings)
    output = work / settings.autoflow_output_folder_name
    output.mkdir()
    (output / "处理清单.json").write_text(
        json.dumps({"plan_id": "another-plan"}),
        encoding="utf-8",
    )

    with pytest.raises(ProjectError, match=r"勾选.*重做"):
        build_plan_for_ui(
            work,
            scanned.selected_edition,
            scanned.source_payloads,
            "audio",
            "merged",
            "black",
            False,
            False,
            settings=settings,
        )

    payload = build_plan_for_ui(
        work,
        scanned.selected_edition,
        scanned.source_payloads,
        "audio",
        "merged",
        "black",
        False,
        True,
        settings=settings,
    )
    assert deserialize_plan(payload).force is True


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
