"""Behavior checks for the reported 2.1 batch and missing-translation failures."""

import json
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber.api import API
from asmr_dubber.autoflow import engine as e
from asmr_dubber.errors import TranslationError
from asmr_dubber.models import Sentence, load_project, save_project
from asmr_dubber.project_origin import project_inventory, record_batch_origin
from asmr_dubber.services import batch_queue, settings
from asmr_dubber.services.application import Application
from asmr_dubber.services.batch_plan import deserialize_plan
from asmr_dubber.translation import translate_sentences


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "home/config"))
    monkeypatch.setattr(e, "STATE_ROOT", tmp_path / "state")
    return Application(tmp_path / "tasks.json")


def work_at(path):
    path.mkdir()
    sf.write(path / "01-original.wav", np.zeros(32000), 16000)
    (path / "01-original.srt").write_text(
        "1\n00:00:00,100 --> 00:00:01,500\nテストです。\n", encoding="utf-8"
    )
    return path


def add_work(app, folder, content="subtitles"):
    scan = app.batch.scan(str(folder))
    return app.batch.save(
        str(folder),
        scan["selected_edition"],
        scan["source_payloads"],
        "video_normal",
        "both",
        content=content,
    )[-1]


def test_subtitle_defaults_export_only_named_translation_and_translate_titles(
    app, tmp_path, monkeypatch
):
    settings.update(
        {
            "autoflow_default_content": "subtitles",
            "autoflow_subtitle_language": "zh",
            "autoflow_subtitle_naming": "original",
            "autoflow_translate_work_title": True,
            "autoflow_translate_track_titles": True,
        }
    )
    folder = work_at(tmp_path / "作品-original")
    plan = deserialize_plan(add_work(app, folder))
    assert plan.source_subtitles_only and plan.subtitle_language == "zh"
    assert plan.translate_work_title and plan.translate_track_titles
    titles_calls, commands, manifests = [], [], []

    def titles(*args, **kwargs):
        titles_calls.append(kwargs)
        return "作品译名", {"01-original.wav": "曲目译名"}

    def create(paths, source, source_language, project_name=None):
        _, directory = pipeline.create_project(
            source,
            projects_root=tmp_path / "projects",
            source_language=source_language,
            project_name=project_name,
        )
        manifests.append(directory / "project.json")
        return manifests[-1]

    def cli(paths, command, manifest, *args):
        commands.append(command)
        project, directory = load_project(manifest)
        if command == "translate":
            for sentence in project.sentences:
                sentence.zh_text = "这是译文。"
            save_project(project, directory)
        elif command == "subtitles":
            assert args == ("--language", "zh")
            pipeline.generate_subtitles(project, directory, language="zh")
        else:
            pytest.fail(f"unexpected ASR/TTS/media operation: {command}")

    monkeypatch.setattr(e, "translated_plan_titles", titles)
    monkeypatch.setattr(e, "create_asmr_project", create)
    monkeypatch.setattr(e, "run_asmr_cli", cli)
    paths = SimpleNamespace(ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe()))
    e.execute_prepared_smart_plan(paths, e.AppConfig(None, -10, 0, ""), plan)
    assert titles_calls == [{"translate_work_title": True, "translate_track_titles": True}]
    assert commands == ["translate", "subtitles"]
    assert not any(p.suffix in {".wav", ".flac", ".mp4"} for p in plan.output_root.rglob("*"))
    assert "これは" not in (plan.output_root / "合并版/01-original.srt").read_text(encoding="utf-8")
    assert "这是译文。" in (plan.output_root / "分轨/001/01-original.srt").read_text(
        encoding="utf-8"
    )
    assert "曲目译名" in (plan.output_root / "曲目清单.txt").read_text(encoding="utf-8-sig")
    assert "作品-original" in manifests[0].parent.name
    inventory = project_inventory(tmp_path / "projects")
    assert "作品-original" in inventory[0]["label"] and "01-original" in inventory[0]["label"]


def test_legacy_queue_preserves_selected_meaning_and_old_output(app, tmp_path):
    folder = work_at(tmp_path / "old-work")
    payload = add_work(app, folder)
    payload["edition"]["output_policy"].pop("content_version")
    payload.update(plan_id="old-plan", source_subtitles_only=False)
    root = Path(payload["output_root"])
    root.mkdir()
    (root / "原声.flac").write_bytes(b"previous-result")
    (root / "处理清单.json").write_text('{"plan_id":"old-plan"}', encoding="utf-8")
    app.batch._save([payload])
    upgraded = app.batch.list()[0]
    assert upgraded["source_subtitles_only"] and upgraded["plan_id"] != "old-plan"
    assert app.batch.list() == [upgraded]
    with pytest.raises(e.VideoPreparerError):
        e.prepare_smart_output_root(root, plan_id=upgraded["plan_id"], force=False, rebuild=False)
    assert (root / "原声.flac").read_bytes() == b"previous-result"


@pytest.mark.parametrize("mode", ["audio", "video_normal"])
def test_original_with_subtitles_exports_media_without_tts(app, tmp_path, monkeypatch, mode):
    folder = work_at(tmp_path / "original-media")
    settings.update(
        {"autoflow_translate_work_title": False, "autoflow_translate_track_titles": False}
    )
    scan = app.batch.scan(str(folder))
    payload = app.batch.save(
        str(folder),
        scan["selected_edition"],
        scan["source_payloads"],
        mode,
        "merged",
        content="source_subtitles",
        subtitle_language="zh",
    )[-1]
    plan = deserialize_plan(payload)
    calls = []

    def create(paths, source, source_language, project_name=None):
        _, directory = pipeline.create_project(
            source,
            projects_root=tmp_path / "projects",
            source_language=source_language,
            project_name=project_name,
        )
        return directory / "project.json"

    def cli(paths, command, manifest, *args):
        calls.append(command)
        project, directory = load_project(manifest)
        if command == "translate":
            for sentence in project.sentences:
                sentence.zh_text = "仅译文字幕。"
            save_project(project, directory)
        elif command == "subtitles":
            pipeline.generate_subtitles(project, directory, language="zh")
        else:
            pytest.fail(f"unexpected inference: {command}")

    monkeypatch.setattr(e, "create_asmr_project", create)
    monkeypatch.setattr(e, "run_asmr_cli", cli)
    monkeypatch.setattr(
        e, "translated_plan_titles", lambda *a, **k: (folder.name, {"01-original.wav": "original"})
    )
    paths = SimpleNamespace(
        ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe()), video_encoder_options=("-c:v", "libx264")
    )
    e.execute_prepared_smart_plan(paths, e.AppConfig(None, -10, 0, ""), plan)
    expected = "原声.flac" if mode == "audio" else "原声.mp4"
    assert (plan.output_root / "合并版" / expected).is_file()
    assert "仅译文字幕。" in (plan.output_root / "合并版/双语版.srt").read_text(
        encoding="utf-8-sig"
    )
    assert "synthesize" not in calls and "mix" not in calls


@pytest.mark.parametrize("recover", [True, False])
def test_blank_semantic_translation_retries_without_disabling_or_losing_success(recover):
    sentences = [
        Sentence(
            id=f"s{i}",
            start_seconds=i,
            end_seconds=i + 1,
            source_text="これは大切な長い文章なので最後まで翻訳してください。",
        )
        for i in (1, 2)
    ]
    calls, checkpoints = [], []

    def handler(request):
        calls.append(request)
        rows = (
            [{"id": "s1", "zh": "第一句。"}, {"id": "s2", "zh": ""}]
            if len(calls) == 1
            else [{"id": "s2", "zh": "第二句。" if recover else ""}]
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"translations": rows}, ensure_ascii=False)
                        },
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:

        def run():
            translate_sentences(
                sentences,
                "not-real-test-key",
                "deepseek-flash",
                client=client,
                on_batch=lambda: checkpoints.append([s.zh_text for s in sentences]),
            )

        if recover:
            run()
        else:
            with pytest.raises(TranslationError, match="s2"):
                run()
    assert len(calls) == 2 and sentences[0].zh_text == "第一句。"
    assert sentences[1].enabled
    assert checkpoints and checkpoints[-1][0] == "第一句。"
    assert sentences[1].status == ("translated" if recover else "translation_failed")
    if not recover:
        assert sentences[1].error and not sentences[1].zh_text


def wait_task(app, identifier):
    for _ in range(300):
        task = app.tasks.get(identifier)
        if task["status"] not in {"queued", "running", "cancelling"}:
            return task
        time.sleep(0.01)
    pytest.fail("batch did not finish")


def test_queue_has_individual_statuses_and_preserves_partial_success(app, tmp_path, monkeypatch):
    entered, release = Event(), Event()
    payloads = [
        {
            "plan_id": str(i),
            "folder": str(tmp_path / str(i)),
            "output_root": str(tmp_path / f"out{i}"),
        }
        for i in range(3)
    ]
    app.batch._save(payloads)
    for name in ("save_failed_task", "remove_failed_task", "log_event"):
        monkeypatch.setattr(e, name, lambda *a, **k: None)

    def run(payloads, *, cancel_event, reference_event_callback):
        plans = [
            SimpleNamespace(
                plan_id=p["plan_id"],
                folder=Path(p["folder"]),
                output_root=Path(p["output_root"]),
                retry_of=None,
            )
            for p in payloads
        ]

        def executor(paths, config, plan):
            if plan.plan_id == "1":
                entered.set()
                assert release.wait(3)
                raise e.VideoPreparerError("second work failed")

        result = e.execute_smart_queue(
            None, None, plans, executor=executor, reference_event_callback=reference_event_callback
        )
        return result, []

    monkeypatch.setattr(batch_queue, "run_queue", run)
    task = app.start({"kind": "batch"})
    assert entered.wait(3)
    statuses = {key: value["status"] for key, value in app.tasks.get(task["id"])["plans"].items()}
    release.set()
    assert statuses == {"0": "completed", "1": "running", "2": "queued"}
    finished = wait_task(app, task["id"])
    assert finished["status"] == "failed"
    assert {key: value["status"] for key, value in finished["plans"].items()} == {
        "0": "completed",
        "1": "failed",
        "2": "completed",
    }
    assert finished["result"]["outcomes"]["1"]["error"] == "second work failed"


def test_empty_queue_is_rejected_before_task_creation_and_history_is_hideable(app):
    with pytest.raises(ValueError, match="未启动"):
        app.start({"kind": "batch"})
    assert app.tasks.list() == []
    app.tasks.items["old-empty"] = {"id": "old-empty", "kind": "batch", "status": "failed"}
    assert API(app).call("tasks/dismiss", {"identifier": "old-empty"})["hidden"]
    # Retain outcomes for queue badges even after hiding the historical card.
    assert app.tasks.list()[0]["hidden"]


def test_reenable_legacy_semantic_skip_clears_state_and_labels_old_projects(app, tmp_path):
    source = tmp_path / "original.wav"
    sf.write(source, np.zeros(16000), 16000)
    project, directory = pipeline.create_project(source, projects_root=tmp_path / "projects")
    project.sentences = [
        Sentence(
            id="s1",
            start_seconds=0,
            end_seconds=1,
            source_text="これは意味のある長い文章です。",
            enabled=False,
            status="skipped_filler",
        )
    ]
    save_project(project, directory)
    snapshot = app.projects.get(str(directory / "project.json"))
    assert snapshot["recoverable_translations"] == ["s1"]
    row = snapshot["rows"][0]
    row[1] = True
    updated = app.projects.sentence(snapshot["manifest"], row, snapshot["revision"])
    assert updated["sentences"][0]["enabled"] and updated["sentences"][0]["status"] == "pending"
    record_batch_origin(
        directory / "project.json",
        source_folder=tmp_path / "作品名",
        plan_id="12345678abcdef",
        job_id="track-1",
        track="音轨名",
    )
    assert project_inventory(tmp_path / "projects")[0]["label"] == "作品名 · 音轨名 [12345678]"
    assert directory.is_dir()  # Existing project paths are never moved by the display upgrade.
