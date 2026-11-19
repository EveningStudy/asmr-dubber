from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from asmr_dubber import app_logging, cache_cleanup, models, runtime_health
from asmr_dubber.autoflow import (
    constants,
    engine,
    fileio,
    inprocess_pipeline,
    media_video,
    merged_output_files,
    output_files,
    output_policy,
    plan_files,
    plan_run,
    project_cli,
    runlog,
    source_subtitles,
    title_translation,
)
from asmr_dubber.autoflow import task_run as autoflow_task_run
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import AudioInfo, DubProject, Sentence, save_project
from asmr_dubber.services import (
    application as app_service,
)
from asmr_dubber.services import (
    batch_plan,
    batch_preflight,
    batch_queue,
    diagnostics,
)
from asmr_dubber.services import (
    tasks as tasks_service,
)
from asmr_dubber.user_settings import UserSettings


def test_corrupt_tasks_json_backed_up_and_startup_clears_reference(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text("{corrupt-json", encoding="utf-8")

    tasks = tasks_service.Tasks(lambda *a: {}, tasks_path)
    assert tasks.items == {}
    backups = list(tmp_path.glob("tasks.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{corrupt-json"

    # Startup clears stale reference on interrupted active tasks and historical tasks
    tasks_path.write_text(
        json.dumps(
            {
                "t1": {
                    "id": "t1",
                    "kind": "batch",
                    "resource": "batch",
                    "status": "running",
                    "reference": {"kind": "ready", "plan_id": "p1"},
                    "plans": {"p1": {"status": "running"}},
                    "updated_at": 10.0,
                },
                "t2": {
                    "id": "t2",
                    "kind": "batch",
                    "resource": "batch",
                    "status": "failed",
                    "reference": {"kind": "ready", "plan_id": "p2"},
                    "plans": {},
                    "updated_at": 5.0,
                },
            }
        ),
        encoding="utf-8",
    )
    reloaded = tasks_service.Tasks(lambda *a: {}, tasks_path)
    assert reloaded.items["t1"]["status"] == "interrupted"
    assert reloaded.items["t1"]["reference"] is None
    assert reloaded.items["t1"]["plans"]["p1"]["status"] == "interrupted"
    assert reloaded.items["t2"]["reference"] is None


def test_tasks_report_survives_transient_save_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}
    real_write = tasks_service.atomic_write_text

    def flaky_write(path, text):
        calls["count"] += 1
        # Fail only during mid-run throttled save (after queued + running saves)
        if calls["count"] == 3:
            raise ProjectError("antivirus lock")
        return real_write(path, text)

    monkeypatch.setattr(tasks_service, "atomic_write_text", flaky_write)

    def execute(_req, report, _token, _ref):
        report("tick 1", 1, 2)
        return {"ok": True}

    tasks = tasks_service.Tasks(execute, tmp_path / "tasks.json")
    started = tasks.start({"kind": "analyze"}, "res")
    import time

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        snap = tasks.get(started["id"])
        if snap["status"] == "completed":
            break
        time.sleep(0.01)
    assert tasks.get(started["id"])["status"] == "completed"


def test_batch_resets_rebuild_and_preserves_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    app = app_service.Application(tmp_path / "tasks.json")
    plan_payload = {
        "plan_id": "p-rebuild",
        "folder": str(tmp_path / "workA"),
        "output_root": str(tmp_path / "outA"),
        "edition_label": "默认",
        "sources": [
            {
                "order": 1,
                "path": str(tmp_path / "workA" / "01.wav"),
                "title_ja": "01",
                "size": 100,
                "mtime_ns": 1,
            }
        ],
        "edition": {"output_policy": {"variants": ["bilingual"], "settings": {}}},
        "mode": "audio",
        "layout": "merged",
        "rebuild": True,
        "force": True,
    }
    app.batch._save([plan_payload])

    def fake_run_queue(queue, **kwargs):
        cb = kwargs["reference_event_callback"]
        cb({"kind": "plan_rebuild_started", "plan_id": "p-rebuild"})
        # Queue file must already have rebuild/force cleared before track finishes
        saved = app.batch.list()
        assert saved[0]["rebuild"] is False
        assert saved[0]["force"] is False
        cb(
            {
                "kind": "plan_finished",
                "plan_id": "p-rebuild",
                "status": "failed",
                "work": "workA",
                "output": str(tmp_path / "outA"),
                "error": "尚未保存 API Key",
                "error_code": "API_KEY_MISSING",
                "error_hint": "请填写 API Key",
            }
        )
        return 1, [str(tmp_path / "outA")]

    monkeypatch.setattr(batch_queue, "run_queue", fake_run_queue)
    result = app.batch.run({}, lambda *_a: None, None, lambda *_a: None)
    assert result["error_code"] == "API_KEY_MISSING"
    assert "尚未保存 API Key" in result["error"]


def test_batch_plan_rebuild_error_wording_uses_reprocess_label(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / "处理清单.json").write_text(json.dumps({"plan_id": "other"}), encoding="utf-8")
    plan = cast(Any, SimpleNamespace(plan_id="mine", output_root=out_root, rebuild=False))
    with pytest.raises(ProjectError, match="重新处理"):
        batch_plan._guard_output_replacement(plan)


def test_short_clean_video_path_in_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "WORK_ROOT", tmp_path / "w")
    seen_destinations: list[Path] = []

    def fake_static(_paths, _audio, _bg, dest, **_kw):
        seen_destinations.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"clean")

    monkeypatch.setattr(media_video, "render_static_video", fake_static)
    monkeypatch.setattr(
        media_video.processes,
        "run_ffmpeg",
        lambda _p, args, **_kw: Path(args[-1]).write_bytes(b"final"),
    )
    deep_dest = tmp_path / ("a" * 50) / ("b" * 50) / "out.mp4"
    sub = tmp_path / "sub.srt"
    sub.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    media_video.render_static_bilingual_video(
        cast(Any, SimpleNamespace(video_encoder_options=())),
        tmp_path / "a.wav",
        None,
        sub,
        deep_dest,
    )
    assert deep_dest.is_file()
    assert seen_destinations and seen_destinations[0].parent == tmp_path / "w"
    assert seen_destinations[0].name.startswith("clean-")
    assert len(seen_destinations[0].name) < 30
    assert not seen_destinations[0].exists()


def test_output_files_uses_output_stem_for_subtitles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "mixed.flac").write_bytes(b"flac")
    (proj_dir / "sub.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    (proj_dir / "sub.lrc").write_text("[00:00.00]hi\n", encoding="utf-8")
    pj = proj_dir / "project.json"
    pj.write_text(
        json.dumps(
            {
                "output_file": "mixed.flac",
                "subtitle_srt_file": "sub.srt",
                "subtitle_lrc_file": "sub.lrc",
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "out"
    outputs = output_files.copy_final_outputs(
        cast(Any, SimpleNamespace()),
        pj,
        dest,
        "audio",
        harmonized_delay_seconds=0,
        harmonized_volume_db=0.0,
        embed_subtitles=False,
        output_stem="配音版",
    )
    assert Path(outputs["srt"]).name == "配音版.srt"
    assert Path(outputs["lrc"]).name == "配音版.lrc"
    assert Path(outputs["audio"]).name == "配音版.flac"


def test_merged_output_files_reuses_original_across_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "WORK_ROOT", tmp_path / "work")
    render_calls: list[Path] = []

    def fake_static(_paths, _audio, _bg, dest, **_kw):
        render_calls.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4-data")

    monkeypatch.setattr(media_video, "render_static_video", fake_static)
    monkeypatch.setattr(
        merged_output_files,
        "concat_audio_files",
        lambda _p, files, work, **_kw: (files[0], []),
    )
    monkeypatch.setattr(
        merged_output_files,
        "combine_srt_files",
        lambda _e, dst, **_kw: (dst.write_text("srt", encoding="utf-8") and dst) or dst,
    )
    monkeypatch.setattr(
        merged_output_files,
        "combine_lrc_files",
        lambda _e, dst, **_kw: (dst.write_text("lrc", encoding="utf-8") and dst) or dst,
    )
    master = tmp_path / "master.wav"
    master.write_bytes(b"m")
    mix_bi = tmp_path / "bi.wav"
    mix_bi.write_bytes(b"b")
    mix_rep = tmp_path / "rep.wav"
    mix_rep.write_bytes(b"r")
    srt = tmp_path / "t.srt"
    srt.write_text("1", encoding="utf-8")
    lrc = tmp_path / "t.lrc"
    lrc.write_text("1", encoding="utf-8")
    state = {
        "mode": "video_normal",
        "harmonized_delay_seconds": 0,
        "timeline": [
            {
                "filename": "01.wav",
                "relative_path": "01.wav",
                "title_ja": "t1",
                "start_samples": 0,
                "duration_samples": 48000,
            }
        ],
        "master_audio": str(master),
        "mix_variants": {
            "bilingual": {"audio": str(mix_bi), "outputs": {"srt": str(srt), "lrc": str(lrc)}},
            "replace": {"audio": str(mix_rep), "outputs": {"srt": str(srt), "lrc": str(lrc)}},
        },
    }
    src = engine.AudioSource(order=1, path=tmp_path / "01.wav", title_ja="t1", size=1, mtime_ns=1)
    cfg = engine.AppConfig(
        None,
        0.0,
        0,
        "",
        output_policy={"variants": ["bilingual", "replace"], "settings": {}},
    )
    out = merged_output_files.build_merged_outputs(
        cast(Any, SimpleNamespace()),
        cfg,
        source_folder=tmp_path / "src",
        output_folder=tmp_path / "merged",
        mode="video_normal",
        sources=[src],
        states=[state],
        background=None,
        folder_translation="作品",
        title_translations={"01.wav": "第一轨"},
        embed_subtitles=False,
        subtitles_only=False,
    )
    # 1 original + 2 variant dubs = 3 renders (NOT 4!)
    assert [p.name for p in render_calls] == ["原声.mp4", "双语版.mp4", "配音版.mp4"]
    assert Path(out["replace_original"]).is_file()


def test_plan_run_reuse_merged_writes_timestamps_in_variant_subdirs(tmp_path: Path) -> None:
    merged_root = tmp_path / "合并版"
    merged_root.mkdir()
    bi_dir = merged_root / "双语版"
    bi_dir.mkdir()
    rep_dir = merged_root / "完全配音版"
    rep_dir.mkdir()
    outputs = {}
    for prefix, folder, stem in (("bilingual", bi_dir, "双语版"), ("replace", rep_dir, "配音版")):
        for key, name in (
            ("original", "原声.mp4"),
            ("video", f"{stem}.mp4"),
            ("srt", f"{stem}.srt"),
            ("lrc", f"{stem}.lrc"),
            ("timestamps", "时间戳.txt"),
        ):
            p = folder / name
            p.write_text("x", encoding="utf-8")
            outputs[f"{prefix}_{key}"] = str(p)
            if prefix == "bilingual":
                outputs[key] = str(p)
    plan = SimpleNamespace(
        folder=tmp_path / "work",
        mode="video_normal",
        rebuild=False,
        subtitles_only=False,
        embed_subtitles=False,
    )
    cfg = engine.AppConfig(
        None,
        0.0,
        0,
        "",
        output_policy={"variants": ["bilingual", "replace"], "settings": {}},
    )
    descriptors = [{"kind": "merged", "output": str(merged_root)}]
    previous_jobs = [{"kind": "merged", "status": "completed", "outputs": outputs}]
    states = [
        {
            "timeline": [
                {
                    "filename": "01.wav",
                    "relative_path": "01.wav",
                    "title_ja": "t1",
                    "start_samples": 0,
                    "duration_samples": 48000,
                }
            ]
        }
    ]
    plan_run._finish_merged_from_tracks(
        cast(Any, SimpleNamespace()),
        cfg,
        cast(Any, plan),
        descriptors,
        previous_jobs,
        states,
        "作品",
        {"01.wav": "第一轨"},
    )
    assert not (merged_root / "时间戳.txt").exists()
    assert (bi_dir / "时间戳.txt").is_file()
    assert (rep_dir / "时间戳.txt").is_file()


def test_subtitles_skips_video_render_when_no_video_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "source.mp4").write_bytes(b"mp4")
    from asmr_dubber.audio import sha256_file

    project = DubProject(
        source=AudioInfo(
            path="source.mp4",
            sha256=sha256_file(proj_dir / "source.mp4"),
            media_type="video",
            duration_seconds=1.0,
            sample_rate=48000,
            channels=2,
        ),
        sentences=[
            Sentence(
                id="s1",
                start_seconds=0.0,
                end_seconds=1.0,
                source_text="こんにちは",
                target_text="你好",
            )
        ],
    )
    manifest = save_project(project, proj_dir)

    seen_args: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        autoflow_task_run.project_cli,
        "run_asmr_cli",
        lambda _p, *args: seen_args.append(args),
    )
    state_file = tmp_path / "state.json"
    autoflow_task_run._subtitles_stage(
        cast(Any, SimpleNamespace()),
        state_file,
        {"mode": "video_harmonized", "embed_subtitles": True, "subtitles_only": False},
        manifest,
    )
    assert "--no-video" in seen_args[0]

    # Also test output_policy.render_variants passes --no-video when embed_subtitles=False
    seen_policy_args: list[tuple[str, ...]] = []

    def fake_cli(_p, cmd, pj, *args):
        seen_policy_args.append((cmd, *args))
        p, d = models.load_project(pj)
        if cmd == "mix":
            (d / "out.wav").write_bytes(b"w")
            p.output_file = "out.wav"
            (d / "out.mp4").write_bytes(b"v")
            p.output_video_file = "out.mp4"
        elif cmd == "subtitles":
            (d / "sub.srt").write_text("1", encoding="utf-8")
            (d / "sub.lrc").write_text("1", encoding="utf-8")
            p.subtitle_srt_file = "sub.srt"
            p.subtitle_lrc_file = "sub.lrc"
        save_project(p, d)

    monkeypatch.setattr(project_cli, "run_asmr_cli", fake_cli)
    monkeypatch.setattr(
        output_files, "copy_final_outputs", lambda *a, **k: {"video": "v", "srt": "s", "lrc": "l"}
    )
    state = {
        "output_policy": {"variants": ["bilingual"], "subtitle": "zh", "settings": {}},
        "mode": "video_normal",
        "embed_subtitles": False,
        "harmonized_delay_seconds": 0,
        "harmonized_volume_db": 0.0,
    }
    output_policy.render_variants(
        cast(Any, SimpleNamespace()), manifest, tmp_path / "out_var", state
    )
    assert any(a[0] == "subtitles" and "--no-video" in a for a in seen_policy_args)

    # Also test inprocess_pipeline handles --no-video without mutating saved media_type
    from asmr_dubber import pipeline

    rendered_video = {"called": False}
    monkeypatch.setattr(
        pipeline,
        "render_subtitled_video",
        lambda *a, **k: rendered_video.__setitem__("called", True),
    )
    inprocess_pipeline.run_stage_in_process(
        ("subtitles", str(manifest), "--language", "bilingual", "--no-video")
    )
    assert not rendered_video["called"]
    reloaded, _ = models.load_project(manifest)
    assert reloaded.source.media_type == "video"
    assert reloaded.subtitle_srt_file


def test_safe_filename_component_guards_windows_reserved_names() -> None:
    for name in ("CON", "prn", "Aux.txt", "nul", "COM1", "com9.srt", "LPT1", "lpt9.lrc"):
        safe = fileio.safe_filename_component(name)
        assert safe.startswith("_"), f"{name} -> {safe}"


def test_title_translation_merges_metadata_and_survives_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "STATE_ROOT", tmp_path / "state")
    existing = {
        "jobs": [{"job_id": "merged", "project_json": "p.json"}],
        "shared_reference": {"audio": "ref.wav"},
    }
    plan_files.save_plan_metadata("p-meta", existing)

    monkeypatch.setattr(
        "asmr_dubber.user_settings.load_user_settings",
        lambda: UserSettings(translation_provider="deepseek"),
    )
    monkeypatch.setattr(
        "asmr_dubber.user_settings.resolve_api_key",
        lambda _p: (_ for _ in ()).throw(ProjectError("尚未保存 DeepSeek API Key")),
    )
    src = engine.AudioSource(
        order=1, path=tmp_path / "01.wav", title_ja="トラック1", size=1, mtime_ns=1
    )
    folder_t, track_t = title_translation.translated_plan_titles(
        "p-meta", tmp_path / "作品A", [src], cast(Any, SimpleNamespace())
    )
    assert folder_t == "作品A"
    assert track_t == {"01.wav": "トラック1"}
    saved = plan_files.load_plan_metadata("p-meta")
    assert saved["jobs"] == existing["jobs"]
    assert saved["shared_reference"] == existing["shared_reference"]

    # Non-auth failure still merges cached metadata instead of wiping jobs/shared_reference
    monkeypatch.setattr("asmr_dubber.user_settings.resolve_api_key", lambda _p: "sk-key-123456")
    monkeypatch.setattr(
        "asmr_dubber.translation.translate_sentences",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("network timeout")),
    )
    with pytest.raises(engine.VideoPreparerError):
        title_translation.translated_plan_titles(
            "p-meta", tmp_path / "作品A", [src], cast(Any, SimpleNamespace())
        )
    after_err = plan_files.load_plan_metadata("p-meta")
    assert after_err["jobs"] == existing["jobs"]
    assert after_err["shared_reference"] == existing["shared_reference"]


def test_source_subtitles_saves_jobs_metadata_and_uses_budgeted_translate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "STATE_ROOT", tmp_path / "state")
    work = tmp_path / "work"
    work.mkdir()
    wav = work / "01.wav"
    import wave

    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    proj_dir = tmp_path / "proj"
    project = DubProject(
        source=AudioInfo(
            path="01.wav", sha256="abc", duration_seconds=1.0, sample_rate=48000, channels=2
        ),
        sentences=[Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="a")],
    )
    manifest = save_project(project, proj_dir)
    monkeypatch.setattr(project_cli, "create_asmr_project", lambda *a, **k: manifest)
    monkeypatch.setattr(
        title_translation, "translated_plan_titles", lambda *a, **k: ("work", {"01.wav": "01"})
    )

    budgeted_calls: list[str] = []

    def fake_budgeted(_p, state, stage, pj):
        budgeted_calls.append(stage)
        p, d = models.load_project(pj)
        (d / "s.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n译文\n", encoding="utf-8")
        (d / "s.lrc").write_text("[00:00.00]译文\n", encoding="utf-8")
        p.subtitle_srt_file = "s.srt"
        p.subtitle_lrc_file = "s.lrc"
        p.sentences[0].zh_text = "译文"
        save_project(p, d)

    monkeypatch.setattr(project_cli, "run_budgeted_stage", fake_budgeted)
    monkeypatch.setattr(project_cli, "run_asmr_cli", lambda *a, **k: None)

    plan = engine.SmartTaskPlan(
        folder=work,
        output_root=tmp_path / "out",
        edition_label="默认",
        sources=(engine.AudioSource(order=1, path=wav, title_ja="01", size=3, mtime_ns=1),),
        edition={},
        mode="audio",
        layout="merged",
        background=None,
        embed_subtitles=False,
        plan_id="p-sub",
        rebuild=False,
        force=False,
        subtitles_only=True,
        source_subtitles_only=True,
        subtitle_language="zh",
    )
    source_subtitles.execute_source_subtitles(
        cast(Any, SimpleNamespace()), engine.AppConfig(None, 0.0, 0, ""), plan
    )
    assert budgeted_calls == ["translate"]
    meta = plan_files.load_plan_metadata("p-sub")
    assert meta["jobs"] and meta["jobs"][0]["project_json"] == str(manifest)


def test_preflight_disk_uses_projects_root_and_accounts_layout_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_projects = tmp_path / "custom_projects"
    user = UserSettings(projects_root=str(user_projects))
    plan_bi = SimpleNamespace(
        source_subtitles_only=False,
        subtitles_only=False,
        mode="video_normal",
        layout="merged",
        edition={"output_policy": {"variants": ["bilingual"], "settings": {}}},
        output_root=tmp_path / "out",
        sources=(),
    )
    plan_both = SimpleNamespace(
        source_subtitles_only=False,
        subtitles_only=False,
        mode="video_harmonized",
        layout="both",
        edition={"output_policy": {"variants": ["bilingual"], "settings": {}}},
        output_root=tmp_path / "out",
        sources=(),
    )
    plan_src = SimpleNamespace(
        source_subtitles_only=True,
        subtitles_only=True,
        mode="audio",
        layout="merged",
        edition={},
        output_root=tmp_path / "out",
        sources=(),
    )
    assert batch_preflight.estimate_bytes(plan_both, 100.0) > batch_preflight.estimate_bytes(
        plan_bi, 100.0
    )
    assert batch_preflight.estimate_bytes(plan_src, 100.0) > 0

    probed_paths: list[Path] = []
    monkeypatch.setattr(batch_preflight, "plan_seconds", lambda _p: 1000.0)

    def fake_disk(path: Path):
        probed_paths.append(path)
        return (500 * 1024**3, len(probed_paths))

    monkeypatch.setattr(batch_preflight, "_disk_free", fake_disk)
    batch_preflight.check_disk([plan_bi], user)
    assert user_projects in probed_paths


def test_secret_redaction_patterns_and_logging_call_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "custom-env-deepseek-token")
    sample = (
        "gemini=AIzaSyDummyKey123456789012345 "
        "rep=r8_DummyReplicateToken1234567890 "
        "https://alice:topsecret@api.example.com/v1?apiKey=querySecret123&x=1 "
        "raw=custom-env-deepseek-token"
    )
    redacted = app_logging.redact_sensitive(sample)
    for leak in (
        "AIzaSyDummyKey",
        "r8_DummyReplicateToken",
        "topsecret",
        "querySecret123",
        "custom-env-deepseek-token",
    ):
        assert leak not in redacted

    monkeypatch.setattr(constants, "LOG_FILE", tmp_path / "autoflow.log")
    work_log = tmp_path / "work.log"
    with runlog.bind_autoflow_callbacks(work_log=work_log):
        runlog.emit(f"error {sample}")
    assert "topsecret" not in (tmp_path / "autoflow.log").read_text(encoding="utf-8")
    assert "AIzaSyDummyKey" not in work_log.read_text(encoding="utf-8")
    assert "r8_DummyReplicateToken" not in batch_queue.recent_log_text()

    log_file = app_logging.application_log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(sample + "\n", encoding="utf-8")
    monkeypatch.setattr(app_logging, "_CONFIGURED_PATH", log_file.resolve())
    app = app_service.Application(tmp_path / "tasks.json")
    logs_res = app.logs()
    assert "topsecret" not in logs_res["text"]
    assert "r8_DummyReplicateToken" not in logs_res["text"]

    # Diagnostics includes .log.1 backups and env keys
    log_file.with_suffix(".log.1").write_text(
        "backup custom-env-deepseek-token\n", encoding="utf-8"
    )
    constants.LOG_FILE.with_suffix(".log.1").write_text("af backup\n", encoding="utf-8")
    zip_path, names = diagnostics.build_bundle([], {}, doctor=lambda: "ok")
    assert "logs/application.log.1" in names
    assert "logs/autoflow.log.1" in names
    with zipfile.ZipFile(zip_path) as zf:
        assert "custom-env-deepseek-token" not in zf.read("logs/application.log.1").decode("utf-8")


def test_repair_respects_download_source_and_diagnostics_searches_all_batch_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    app = app_service.Application(tmp_path / "tasks.json")
    app.update_settings({"download_source": "modelscope"})

    seen_env: list[str | None] = []
    monkeypatch.setattr(
        runtime_health,
        "repair_runtime_health",
        lambda *_a: seen_env.append(os.environ.get("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS")) or "ok",
    )
    app.execute(
        {"kind": "repair", "backend": "parakeet_nemo", "action": "backend"}, None, None, None
    )
    assert seen_env == ["0"]

    app.update_settings({"download_source": "original"})
    app.execute(
        {"kind": "repair", "backend": "parakeet_nemo", "action": "backend"}, None, None, None
    )
    assert seen_env == ["0", "1"]

    # Diagnostics merges older batch task outcomes and falls back when plan removed from queue
    app.tasks.items = {
        "old": {
            "id": "old",
            "kind": "batch",
            "status": "failed",
            "started_at": 100.0,
            "plans": {
                "removed-plan": {
                    "work": "RemovedWork",
                    "output": str(tmp_path / "out1"),
                    "status": "failed",
                    "error": "boom",
                }
            },
        },
        "new": {
            "id": "new",
            "kind": "batch",
            "status": "completed",
            "started_at": 200.0,
            "plans": {
                "other-plan": {
                    "work": "Other",
                    "output": str(tmp_path / "out2"),
                    "status": "completed",
                }
            },
        },
    }
    monkeypatch.setattr(diagnostics, "run_doctor", lambda: "ok")
    diag = app.diagnostics("removed-plan")
    with zipfile.ZipFile(diag["path"]) as zf:
        payload = json.loads(zf.read("failed-sentences.json").decode("utf-8"))
    assert len(payload["works"]) == 1
    assert payload["works"][0]["plan_id"] == "removed-plan"
    assert payload["works"][0]["outcome"]["error"] == "boom"


def test_cache_cleanup_fast_signature_vocal_ref_and_safe_app_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(home))
    proj = home / "projects" / "p1"
    proj.mkdir(parents=True)
    (proj / "source.wav").write_bytes(b"src")
    save_project(
        DubProject(
            source=AudioInfo(
                path="source.wav", sha256="s", duration_seconds=1.0, sample_rate=48000, channels=2
            ),
            sentences=[],
        ),
        proj,
    )
    sep = proj / "analysis" / "separation" / "abc"
    sep.mkdir(parents=True)
    # Create > 4 MiB sparse stems so sha256_file MUST NOT be called
    for name in ("vocals.wav", "background.wav"):
        with (sep / name).open("wb") as handle:
            handle.truncate(5 * 1024 * 1024)
    (sep / "source.wav").write_bytes(b"norm")
    (sep / "complete.json").write_text(
        json.dumps({"vocals.wav": "a" * 64, "background.wav": "b" * 64}),
        encoding="utf-8",
    )
    (proj / "analysis" / "spatial-reference-vocal.wav").write_bytes(b"vr")
    (proj / "analysis" / "spatial-reference-vocal.key").write_text("k", encoding="utf-8")

    # Orphaned/completed autoflow workspace + resumable workspace + diagnostic zip
    work_root = home / "autoflow" / ".work"
    done_ws = work_root / "smart-done"
    done_ws.mkdir(parents=True)
    (done_ws / "tmp.wav").write_bytes(b"12345")
    active_ws = work_root / "smart-active"
    active_ws.mkdir(parents=True)
    (active_ws / "keep.wav").write_bytes(b"keep")
    state_root = home / "autoflow" / ".state"
    state_root.mkdir(parents=True)
    (state_root / "smart-active.json").write_text(
        json.dumps({"status": "analyzed", "workspace": str(active_ws)}), encoding="utf-8"
    )
    temp_dir = home / "temp"
    temp_dir.mkdir(parents=True)
    diag_zip = temp_dir / "ASMR-Dubber-diagnostics-20261119-010203.zip"
    diag_zip.write_bytes(b"zip")

    app = app_service.Application(tmp_path / "tasks.json")
    storage_info = app.storage()
    assert storage_info["sizes"]["缓存"] >= 5 + 4 + 3

    monkeypatch.setattr(
        cache_cleanup,
        "sha256_file",
        lambda _p: pytest.fail("sha256_file must not run on multi-MB stems"),
    )
    safe_plan = cache_cleanup.scan_caches(str(home / "projects"), ["safe"], app_home=home)
    safe_selected = [e["path"] for e in safe_plan["projects"]]
    removed, skipped = cache_cleanup.clean_caches(safe_plan, safe_selected, True)
    assert removed > 0 and not skipped
    assert not (sep / "source.wav").exists()
    assert not (done_ws / "tmp.wav").exists()
    assert (active_ws / "keep.wav").exists()
    assert not diag_zip.exists()

    sep_plan = cache_cleanup.scan_caches(str(home / "projects"), ["separation"], app_home=home)
    cache_cleanup.clean_caches(sep_plan, [str(proj)], True)
    assert not (proj / "analysis" / "spatial-reference-vocal.wav").exists()
    assert not (proj / "analysis" / "spatial-reference-vocal.key").exists()


def test_frontend_batch_controls_present() -> None:
    root = Path(__file__).resolve().parents[1] / "src/asmr_dubber/frontend"
    batch_js = (root / "batch.js").read_text(encoding="utf-8")
    view_js = (root / "batch_view.js").read_text(encoding="utf-8")
    assert len(batch_js.splitlines()) <= 500
    assert len(view_js.splitlines()) <= 500
    for marker in (
        "保存修改",
        "toggleAllTracks",
        "data-track-down",
        "data-open-batch-project",
        "syncTaskFieldVisibility",
        "batchRefCountdown",
    ):
        assert marker in batch_js
    for marker in ("ACTIVE_STATUSES", "data-open-batch-project", "打开项目", "重新处理"):
        assert marker in view_js
