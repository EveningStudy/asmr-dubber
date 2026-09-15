from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber.autoflow import engine as e
from asmr_dubber.autoflow.source_subtitles import execute_source_subtitles
from asmr_dubber.autoflow.ui_services import (
    build_plan_for_ui,
    deserialize_plan,
    edit_plan_for_ui,
    queue_items_for_ui,
    scan_for_ui,
    serialize_plan,
)
from asmr_dubber.models import Sentence, load_project, save_project
from asmr_dubber.user_settings import UserSettings


def make_work(tmp_path):
    work = tmp_path / "RJ999999"
    work.mkdir()
    for i in range(2):
        sf.write(work / f"0{i + 1}.wav", np.zeros(32000, dtype=np.float32), 16000)
    return work


def make_plan(work, layout="both"):
    settings = UserSettings(
        autoflow_subtitle_language="source", autoflow_subtitle_naming="standard"
    )
    scanned = scan_for_ui(work, settings=settings)
    payload = build_plan_for_ui(
        work,
        scanned.selected_edition,
        scanned.source_payloads,
        "video_harmonized",
        layout,
        scanned.selected_background,
        True,
        False,
        source_subtitles_only=True,
        settings=settings,
    )
    return deserialize_plan(payload)


def test_new_plan_serialization_identity_and_queue(tmp_path):
    work = make_work(tmp_path)
    plan = make_plan(work)
    payload = serialize_plan(plan)
    assert deserialize_plan(payload) == plan
    assert plan.subtitles_only and plan.source_subtitles_only
    assert plan.mode == "audio" and not plan.embed_subtitles and plan.background is None
    assert not plan.translate_work_title and not plan.translate_track_titles
    assert queue_items_for_ui([payload])[0]["mode"] == "仅字幕文件 · 原文"
    assert edit_plan_for_ui([payload], plan.plan_id, settings=UserSettings()).source_subtitles_only
    assert e.failed_task_payload(plan)["source_subtitles_only"] is True
    normal_id = e.plan_identity(
        work,
        mode=plan.mode,
        layout=plan.layout,
        edition=plan.edition,
        sources=list(plan.sources),
        output_root=plan.output_root,
        background=None,
        embed_subtitles=False,
        subtitles_only=True,
        translate_work_title=False,
        translate_track_titles=False,
    )
    assert normal_id != plan.plan_id


def mock_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        e, "planned_state_path", lambda folder, plan_id, job: tmp_path / "state" / f"{job}.json"
    )
    commands = []

    def create(paths, media, source_language):
        _, folder = pipeline.create_project(
            media, projects_root=tmp_path / "projects", source_language=source_language
        )
        return folder / "project.json"

    def cli(paths, command, manifest, *args):
        commands.append(command)
        project, directory = load_project(manifest)
        if command == "analyze":
            project.sentences = [
                Sentence(
                    id="s000001", start_seconds=0.2, end_seconds=1.5, source_text="こんにちは。"
                )
            ]
            save_project(project, directory)
        elif command == "subtitles":
            assert args == ("--language", "source")
            pipeline.generate_subtitles(project, directory, language="source")
        else:
            pytest.fail(f"forbidden command: {command}")

    monkeypatch.setattr(e, "create_asmr_project", create)
    monkeypatch.setattr(e, "run_asmr_cli", cli)
    for name in (
        "translated_plan_titles",
        "render_static_video",
        "render_static_bilingual_video",
        "normalize_and_concat",
    ):
        monkeypatch.setattr(
            e, name, lambda *a, **k: pytest.fail("unexpected media/translation work")
        )
    return commands


@pytest.mark.parametrize("layout", ["merged", "separate", "both"])
def test_source_subtitles_skip_all_translation_and_media_exports(tmp_path, monkeypatch, layout):
    work = make_work(tmp_path)
    plan = make_plan(work, layout)
    commands = mock_runtime(tmp_path, monkeypatch)
    config = e.AppConfig(None, -10, 1200, "note", original_hard_subtitles=True)
    execute_source_subtitles(SimpleNamespace(), config, plan)
    assert commands.count("analyze") == 2
    assert not any(p.suffix in {".mp4", ".wav", ".flac"} for p in plan.output_root.rglob("*"))
    if layout != "separate":
        text = (plan.output_root / "合并版" / "原文字幕.srt").read_text(encoding="utf-8")
        assert "00:00:00,200" in text and "00:00:02,200" in text
        assert text.count("こんにちは。") == 2
    if layout != "merged":
        assert len(list((plan.output_root / "分轨").rglob("*.srt"))) == 2
    execute_source_subtitles(SimpleNamespace(), config, plan)
    assert commands.count("analyze") == 2
    assert sorted(p.name for p in work.glob("*.wav")) == ["01.wav", "02.wav"]


def test_matching_timed_source_subtitle_bypasses_asr(tmp_path, monkeypatch):
    work = make_work(tmp_path)
    plan = make_plan(work, "merged")
    subtitle = work / "script.srt"
    subtitle.write_text("1\n00:00:00,100 --> 00:00:01,500\nはい。\n", encoding="utf-8")
    plan = replace(
        plan,
        sources=tuple(
            replace(
                s,
                transcript_path=subtitle,
                transcript_language="ja",
                transcript_timed=True,
                transcript_mode=e.TRANSCRIPT_MODE_DIRECT,
            )
            for s in plan.sources
        ),
    )
    commands = mock_runtime(tmp_path, monkeypatch)
    execute_source_subtitles(SimpleNamespace(), e.AppConfig(None, -10, 1200, ""), plan)
    assert "analyze" not in commands
    assert "はい。" in (plan.output_root / "合并版" / "原文字幕.srt").read_text(encoding="utf-8")


@pytest.mark.integration
def test_timed_subtitles_real_cli_end_to_end(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    import imageio_ffmpeg

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    work = make_work(tmp_path)
    plan = make_plan(work, "both")
    subtitle = work / "test.srt"
    subtitle.write_text("1\n00:00:00,100 --> 00:00:01,500\nテストです。\n", encoding="utf-8")
    plan = replace(
        plan,
        sources=tuple(
            replace(s, transcript_path=subtitle, transcript_language="ja", transcript_timed=True)
            for s in plan.sources
        ),
    )
    monkeypatch.setattr(
        e, "planned_state_path", lambda folder, plan_id, job: tmp_path / "state" / f"{job}.json"
    )
    repo = Path(__file__).parents[1]
    paths = e.ToolPaths(
        repo,
        tmp_path / "home",
        Path(sys.executable),
        Path(imageio_ffmpeg.get_ffmpeg_exe()),
        (sys.executable, "-m", "asmr_dubber.cli"),
        (),
        None,
        (),
    )
    e.execute_prepared_smart_plan(paths, e.AppConfig(repo, -10, 1200, ""), plan)
    assert len(list(plan.output_root.rglob("*.srt"))) == 3
    assert "テストです。" in (plan.output_root / "合并版" / "原文字幕.srt").read_text(
        encoding="utf-8"
    )
    assert not list(plan.output_root.rglob("*.mp4"))
    assert not list(plan.output_root.rglob("*.flac"))
