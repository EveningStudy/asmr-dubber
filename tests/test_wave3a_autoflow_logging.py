"""Wave 3.A: ContextVar autoflow logging/stage callbacks and non-interactive CLI guards."""

from __future__ import annotations

import ast
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dubber.autoflow import (
    app_settings,
    cli_main,
    configure,
    constants,
    engine,
    failed_tasks,
    interactive_queue,
    planned_run,
    project_cli,
    prompts,
    runlog,
    smart_queue,
    source_scan,
    track_prompts,
    work_state,
)
from asmr_dubber.autoflow.domain import AppConfig
from asmr_dubber.services import batch, batch_preflight, batch_queue
from asmr_dubber.services.media import MediaStore


@pytest.fixture
def isolated_autoflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "home" / "config"))
    state_root = tmp_path / "state"
    monkeypatch.setattr(constants, "STATE_ROOT", state_root)
    monkeypatch.setattr(constants, "LOG_FILE", state_root / "autoflow.log")
    monkeypatch.setattr(constants, "LAST_OPTIONS_FILE", state_root / "last-options.json")
    monkeypatch.setattr(constants, "FAILED_TASKS_FILE", state_root / "failed-tasks.json")
    return tmp_path


def _make_work(root: Path, name: str = "RJ123456 测试作品") -> Path:
    work = root / name
    wav = work / "WAV"
    mp3 = work / "MP3"
    bonus = work / "特典"
    wav.mkdir(parents=True)
    mp3.mkdir()
    bonus.mkdir()
    (wav / "Track01 開場.wav").write_bytes(b"wav-1")
    (wav / "Track02 耳かき.wav").write_bytes(b"wav-2")
    (mp3 / "Track01 開場.mp3").write_bytes(b"mp3-1")
    (bonus / "EX01 おまけ.wav").write_bytes(b"bonus")
    (work / "cover.jpg").write_bytes(b"jpg")
    return work


def _default_config() -> AppConfig:
    return AppConfig(
        asmr_root=None,
        harmonized_volume_db=-10.0,
        harmonized_delay_seconds=1200,
        timestamp_footer="",
    )


def test_autoflow_has_zero_builtin_print_input_or_global_stdout_redirect() -> None:
    autoflow_dir = Path(runlog.__file__).parent
    offenders: list[str] = []
    for path in sorted(autoflow_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"print", "input"}
            ):
                offenders.append(f"{path.name}:{node.lineno}:{node.func.id}")
            if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
                offenders.extend(
                    f"{path.name}:{node.lineno}:{alias.name}"
                    for alias in node.names
                    if alias.name in {"redirect_stdout", "redirect_stderr"}
                )
    assert offenders == []

    batch_source = Path(batch.__file__).read_text(encoding="utf-8")
    assert "redirect_stdout" not in batch_source
    assert "redirect_stderr" not in batch_source
    assert "STAGE_LISTENER" not in batch_source
    assert "LogStream" not in batch_source


def test_contextvar_callbacks_and_work_logs_are_isolated_across_threads(
    isolated_autoflow: Path,
) -> None:
    barrier = threading.Barrier(2)

    def worker(tag: str) -> tuple[list[str], list[str], str]:
        messages: list[str] = []
        stages_seen: list[str] = []
        work_log = isolated_autoflow / "logs" / f"{tag}.log"
        with runlog.bind_autoflow_callbacks(
            on_message=messages.append,
            on_stage=stages_seen.append,
        ):
            barrier.wait(timeout=5)
            runlog.emit(f"outer-{tag}")
            with runlog.bind_autoflow_callbacks(work_log=work_log):
                barrier.wait(timeout=5)
                runlog.emit(f"inner-{tag}")
                project_cli.announce_stage([f"stage-{tag}", "project.json"])
                # Broken stage callback never breaks announce_stage.
            runlog.emit(f"after-{tag}")
        assert runlog.current_work_log() is None
        return messages, stages_seen, work_log.read_text(encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(worker, "alpha")
        future_b = pool.submit(worker, "beta")
        msgs_a, stages_a, log_a = future_a.result()
        msgs_b, stages_b, log_b = future_b.result()

    assert msgs_a == ["outer-alpha", "inner-alpha", "after-alpha"]
    assert msgs_b == ["outer-beta", "inner-beta", "after-beta"]
    assert stages_a == ["stage-alpha"]
    assert stages_b == ["stage-beta"]
    assert "inner-alpha" in log_a and "inner-beta" not in log_a and "outer-alpha" not in log_a
    assert "inner-beta" in log_b and "inner-alpha" not in log_b and "outer-beta" not in log_b
    assert engine._WORK_LOG is None

    def broken_stage(_stage: str) -> None:
        raise ValueError("broken stage listener")

    with runlog.bind_autoflow_callbacks(on_stage=broken_stage):
        project_cli.announce_stage(["mix", "project.json"])


def test_batch_run_streams_messages_and_stages_without_touching_stdout_or_global_listener(
    isolated_autoflow: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("save_failed_task", "remove_failed_task"):
        monkeypatch.setattr(failed_tasks, name, lambda *_a, **_k: None)

    work_dir = isolated_autoflow / "work-1"
    out_dir = isolated_autoflow / "out-1"
    plan_obj = SimpleNamespace(
        plan_id="plan-1",
        folder=work_dir,
        output_root=out_dir,
        retry_of=None,
    )
    monkeypatch.setattr(batch_queue, "deserialize_plan", lambda _payload: plan_obj)
    monkeypatch.setattr(batch_queue, "config_from_settings", _default_config)
    monkeypatch.setattr(app_settings, "find_tool_paths", lambda _config: SimpleNamespace())
    monkeypatch.setattr(app_settings, "validate_asmr_version", lambda: "3.0.0")
    monkeypatch.setattr(
        batch_preflight,
        "run_preflight",
        lambda _plans: batch_preflight.PreflightReport(
            [batch_preflight.Check("vram", batch_preflight.WARN, "显存", "使用共享显存")]
        ),
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def fake_prepared(_paths, _config, plan, *, reference_event_callback=None):
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        assert project_cli.STAGE_LISTENER is None
        runlog.emit(f"正在处理 {plan.folder.name}")
        project_cli.announce_stage(["analyze", "project.json"])
        project_cli.announce_stage(["synthesize", "project.json"])
        return []

    monkeypatch.setattr(smart_queue, "execute_prepared_smart_plan", fake_prepared)

    service = batch.Batch(MediaStore())
    service._save([{"plan_id": "plan-1", "folder": str(work_dir), "output_root": str(out_dir)}])
    reported: list[str] = []
    events: list[dict[str, object]] = []

    result = service.run(
        {},
        lambda message, _current, _total: reported.append(message),
        threading.Event(),
        events.append,
    )

    assert result["exit_code"] == 0
    assert any("开始前检查" in item for item in reported)
    assert any("提示：显存：使用共享显存" in item for item in reported)
    assert any("正在处理 work-1" in item for item in reported)
    stage_events = [item["stage"] for item in events if item.get("kind") == "plan_stage"]
    assert stage_events == ["analyze", "synthesize"]
    assert project_cli.STAGE_LISTENER is None


def test_cli_without_folder_fails_fast_instead_of_blocking_on_stdin(
    isolated_autoflow: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_main, "load_app_config", _default_config)
    monkeypatch.setattr(app_settings, "find_tool_paths", lambda _config: SimpleNamespace())
    monkeypatch.setattr(app_settings, "validate_asmr_version", lambda: "3.0.0")

    code = cli_main.main([])
    captured = capsys.readouterr()
    assert code == 1
    assert "ASMR-Dubber AutoFlow" in captured.out
    assert "交互式终端问答已移除" in captured.err

    legacy_code = cli_main.main(["--scan", "legacy"])
    legacy_captured = capsys.readouterr()
    assert legacy_code == 1
    assert "未指定作品文件夹路径" in legacy_captured.err


def test_noninteractive_defaults_and_injected_input_reader_cover_plan_builders(
    isolated_autoflow: Path,
) -> None:
    work = _make_work(isolated_autoflow)
    config = _default_config()

    # Non-interactive (input_reader=None): picks WAV edition, video_normal, merged, cover.
    auto_plan = configure.prepare_smart_plan(
        config,
        work,
        mode_argument=None,
        layout_argument=None,
        edition_argument=None,
        include_bonus=False,
        output_root_argument=None,
        background_argument=None,
        embed_subtitles_argument=None,
        rebuild=False,
        force=False,
    )
    assert auto_plan.mode == constants.MODE_VIDEO_NORMAL
    assert auto_plan.layout == constants.LAYOUT_MERGED
    assert auto_plan.embed_subtitles is True
    assert auto_plan.background == (work / "cover.jpg").resolve()
    assert [src.path.suffix for src in auto_plan.sources] == [".wav", ".wav"]

    # Injected input_reader: exercises manual track selection, harmonized video, separate, black bg.
    scripted = iter(("m", "1,3", "2", "b", "2", "2", "2", "0", "n"))
    custom_plan = configure.prepare_smart_plan(
        config,
        work,
        mode_argument=None,
        layout_argument=None,
        edition_argument=None,
        include_bonus=False,
        output_root_argument=None,
        background_argument=None,
        embed_subtitles_argument=None,
        rebuild=False,
        force=False,
        input_reader=lambda _prompt: next(scripted),
    )
    assert custom_plan.edition_label == "自定义音轨"
    assert custom_plan.mode == constants.MODE_VIDEO_HARMONIZED
    assert custom_plan.layout == constants.LAYOUT_SEPARATE
    assert custom_plan.background is None
    assert custom_plan.embed_subtitles is False

    # Queue collection works with injected input_reader and rejects None.
    args = cli_main.build_parser().parse_args([])
    with pytest.raises(engine.VideoPreparerError, match="交互式终端问答已移除"):
        interactive_queue.collect_interactive_smart_plans(config, args)

    queue_inputs = iter((str(work), "0"))
    queued = interactive_queue.collect_interactive_smart_plans(
        config,
        args,
        input_reader=lambda _prompt: next(queue_inputs),
    )
    assert len(queued) == 1 and queued[0].folder == work.resolve()


def test_legacy_prepare_or_resume_and_overwrite_guards_without_blocking(
    isolated_autoflow: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = isolated_autoflow / "legacy-work"
    legacy.mkdir()
    track = legacy / "1 开场.wav"
    track.write_bytes(b"wav")
    existing_out = legacy / "原声.flac"
    existing_out.write_bytes(b"existing")
    config = _default_config()

    # Overwrite without --force refuses safely when input_reader is None.
    with pytest.raises(engine.VideoPreparerError, match="--force"):
        prompts.confirm_overwrite(legacy, constants.MODE_AUDIO, force=False)

    # Interactive rejection raises KeyboardInterrupt; approval succeeds.
    with pytest.raises(KeyboardInterrupt):
        prompts.confirm_overwrite(
            legacy, constants.MODE_AUDIO, force=False, input_reader=lambda _p: "n"
        )
    prompts.confirm_overwrite(
        legacy, constants.MODE_AUDIO, force=False, input_reader=lambda _p: "y"
    )

    # Changed fingerprint without --rebuild raises VideoPreparerError non-interactively.
    sources = source_scan.discover_audio(legacy)
    state_file = work_state.state_path(legacy)
    work_state.save_state(
        state_file,
        {
            "schema": 1,
            "mode": constants.MODE_AUDIO,
            "status": "analyzed",
            "fingerprint": {"stale": True},
            "embed_subtitles": False,
        },
    )
    with pytest.raises(engine.VideoPreparerError, match="--rebuild"):
        planned_run.prepare_or_resume(
            SimpleNamespace(),  # type: ignore[arg-type]
            config,
            legacy,
            "audio",
            None,
            rebuild=False,
            force=True,
        )

    # Missing resume artifacts without --rebuild raises VideoPreparerError non-interactively.
    work_state.save_state(
        state_file,
        {
            "schema": 1,
            "mode": constants.MODE_AUDIO,
            "status": "analyzed",
            "fingerprint": work_state.fingerprint(sources, None),
            "embed_subtitles": False,
            "timestamp_schema": constants.TIMESTAMP_SCHEMA,
        },
    )
    with pytest.raises(engine.VideoPreparerError, match="--rebuild"):
        planned_run.prepare_or_resume(
            SimpleNamespace(),  # type: ignore[arg-type]
            config,
            legacy,
            "audio",
            None,
            rebuild=False,
            force=True,
        )

    # Unfinished matching task with intact artifacts resumes automatically without prompting.
    project_json = isolated_autoflow / "project" / "project.json"
    project_json.parent.mkdir(parents=True)
    project_json.write_text("{}", encoding="utf-8")
    resumed: list[str] = []
    monkeypatch.setattr(
        planned_run,
        "execute_task",
        lambda _paths, _folder, _sf, state, _srcs: resumed.append(str(state["status"])),
    )
    work_state.save_state(
        state_file,
        {
            "schema": 1,
            "mode": constants.MODE_AUDIO,
            "status": "analyzed",
            "fingerprint": work_state.fingerprint(sources, None),
            "embed_subtitles": False,
            "project_json": str(project_json),
            "timestamp_schema": constants.TIMESTAMP_SCHEMA,
        },
    )
    planned_run.prepare_or_resume(
        SimpleNamespace(),  # type: ignore[arg-type]
        config,
        legacy,
        "audio",
        None,
        rebuild=False,
        force=True,
    )
    assert resumed == ["analyzed"]
    assert track_prompts.choose_tracks(
        configure.scan_work(_make_work(isolated_autoflow, "RJ999")),
        config,
    )[1]
