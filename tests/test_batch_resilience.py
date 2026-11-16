"""Batch errors, failure budget, preflight checks, per-work logs and the diagnostics bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from asmr_dubber import cli, stage_result
from asmr_dubber.audio import sha256_file
from asmr_dubber.autoflow import (
    app_settings,
    constants,
    failed_tasks,
    plan_files,
    processes,
    project_cli,
    smart_queue,
    stages,
)
from asmr_dubber.autoflow import engine as e
from asmr_dubber.errors import ProjectError, TranslationError
from asmr_dubber.experimental_mix import sentence_is_replaced
from asmr_dubber.localization import catalog
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence, save_project
from asmr_dubber.pipeline import mix_project
from asmr_dubber.services import batch_preflight, diagnostics
from asmr_dubber.services.batch_preflight import FAIL, OK, WARN


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "home/config"))
    monkeypatch.setattr(constants, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(constants, "LOG_FILE", tmp_path / "state" / "autoflow.log")
    return tmp_path


def make_project(directory: Path, count: int = 10, **fields) -> Path:
    sentences = [
        Sentence(
            id=f"s{index:06d}",
            start_seconds=index,
            end_seconds=index + 0.9,
            source_text="こんにちは。",
            zh_text="你好。",
            status="synthesized",
        )
        for index in range(1, count + 1)
    ]
    project = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="a" * 64,
            duration_seconds=count + 2,
            sample_rate=16000,
            channels=2,
        ),
        settings=ProjectSettings(),
        sentences=sentences,
    )
    save_project(project, directory)
    return directory / "project.json"


# --- failure budget -------------------------------------------------------------


def test_budget_is_the_larger_of_the_count_and_the_percentage():
    assert stage_result.failure_budget(50) == 3
    assert stage_result.failure_budget(400) == 8
    assert stage_result.failure_budget(401) == 9  # rounded up
    assert stage_result.failure_budget(10, 0, 50.0) == 5
    assert stage_result.failure_budget(0) == 3


def result(stage, failed, total, error=None):
    marker = stage_result.PARTIAL_MARKERS[stage]
    return {
        "ok": False,
        "stage": stage,
        "error": error if error is not None else f"{len(failed)} 句{marker}x",
        "total": total,
        "failed": [{"id": item, "reason": "无译文"} for item in failed],
    }


def test_within_budget_failures_are_partial_and_listed():
    record = stage_result.judge_partial("translate", result("translate", ["s1", "s2"], 40))
    assert record["count"] == 2 and record["total"] == 40
    assert [item["id"] for item in record["items"]] == ["s1", "s2"]


def test_over_budget_or_unrelated_errors_are_failures():
    four = result("translate", ["s1", "s2", "s3", "s4"], 40)
    assert stage_result.judge_partial("translate", four) is None
    assert stage_result.judge_partial("translate", four, count=4) is not None
    # A whole-stage failure (bad key, service down) never counts as a few lost lines.
    assert (
        stage_result.judge_partial("translate", result("translate", ["s1"], 40, error="401"))
        is None
    )
    # When every sentence failed there is nothing to deliver.
    assert stage_result.judge_partial("translate", result("translate", ["s1", "s2"], 2)) is None
    assert stage_result.judge_partial("mix", {"ok": False, "error": "x"}) is None
    assert stage_result.judge_partial("synthesize", None) is None


def test_synthesis_failures_use_the_same_budget():
    record = stage_result.judge_partial("synthesize", result("synthesize", ["s000003"], 200))
    assert record["stage"] == "synthesize"
    assert "未配音" in stage_result.describe_partial(record)


# --- stage result file written by the command line ------------------------------


def test_write_result_lists_failed_sentences_without_secrets(home):
    manifest = make_project(home / "p")
    from asmr_dubber.models import load_project

    project, directory = load_project(manifest)
    project.sentences[1].status = "translation_failed"
    project.sentences[1].error = "单句重试失败：key sk-abcdefghijklmnop"
    save_project(project, directory)
    target = home / "out" / "r.json"
    stage_result.write_result(
        target, "translate", manifest, TranslationError("1 句未取得译文：s000002：x")
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is False and payload["total"] == 10
    assert payload["failed"][0]["id"] == "s000002"
    assert stage_result.judge_partial("translate", payload) is not None


def test_cli_stage_writes_result_json_for_failure_and_success(home, monkeypatch):
    manifest = make_project(home / "p")
    out = home / "result.json"

    def failing(project, directory, **kwargs):
        project.sentences[0].status = "translation_failed"
        project.sentences[0].error = "没有译文"
        save_project(project, directory)
        raise TranslationError("1 句未取得译文：s000001：没有译文。")

    monkeypatch.setattr(cli, "translate_project", failing)
    run = CliRunner().invoke(cli.app, ["translate", str(manifest), "--result-json", str(out)])
    assert run.exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["failed"] == [{"id": "s000001", "reason": "没有译文"}]
    assert "未取得译文" in payload["error"]

    monkeypatch.setattr(cli, "translate_project", lambda *a, **k: None)
    ok = CliRunner().invoke(cli.app, ["translate", str(manifest), "--result-json", str(out)])
    assert ok.exit_code == 0 and json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_cli_mix_passes_allow_missing(home, monkeypatch):
    manifest = make_project(home / "p")
    seen = {}

    def fake_mix(project, directory, **kwargs):
        seen.update(kwargs)
        return directory / "mix.wav"

    monkeypatch.setattr(cli, "mix_project", fake_mix)
    run = CliRunner().invoke(cli.app, ["mix", str(manifest), "--allow-missing"])
    assert run.exit_code == 0 and seen["allow_missing"] is True


# --- batch engine ---------------------------------------------------------------


def fake_paths(tmp_path):
    return SimpleNamespace(cli_command=("python", "-m", "asmr_dubber.cli"), asmr_root=tmp_path)


def test_batch_error_shows_the_real_reason_not_the_exit_code(home, monkeypatch):
    def fake_run_process(command, *, cwd=None, tail=None):
        target = Path(command[command.index("--result-json") + 1])
        target.write_text(
            json.dumps({"ok": False, "error": "DeepSeek 尚未保存 API Key", "failed": []}),
            encoding="utf-8",
        )
        raise e.VideoPreparerError("命令执行失败（退出码 1）：python.exe")

    monkeypatch.setattr(processes, "run_process", fake_run_process)
    with pytest.raises(e.StageFailedError) as caught:
        e.run_asmr_cli(fake_paths(home), "translate", "project.json")
    assert str(caught.value) == "翻译失败：DeepSeek 尚未保存 API Key"
    assert "退出码" not in str(caught.value)
    assert not list((home / "state" / "results").glob("*.json"))  # temporary file removed


def test_batch_error_falls_back_to_the_last_output_line(home, monkeypatch):
    def fake_run_process(command, *, cwd=None, tail=None):
        tail.extend(["loading", "ModuleNotFoundError: No module named 'torch'"])
        raise e.VideoPreparerError("命令执行失败（退出码 1）：python.exe")

    monkeypatch.setattr(processes, "run_process", fake_run_process)
    with pytest.raises(e.StageFailedError, match="No module named 'torch'"):
        e.run_asmr_cli(fake_paths(home), "synthesize", "project.json")


def test_other_commands_are_not_given_a_result_file(home, monkeypatch):
    seen = []
    monkeypatch.setattr(processes, "run_process", lambda command, **kw: seen.append(command))
    assert e.run_asmr_cli(fake_paths(home), "subtitles", "p.json", "--language", "zh") is None
    assert "--result-json" not in seen[0]


def raising(stage, failed, total, *, error=None):
    def run(paths, command, manifest, *args):
        data = result(stage, failed, total, error=error)
        raise e.StageFailedError(stage, f"{stage_result.STAGE_LABELS[stage]}失败：raw", data)

    return run


@pytest.mark.parametrize("stage", ["translate", "synthesize"])
def test_failures_within_budget_continue_and_are_recorded(stage, monkeypatch):
    state = {"failure_budget": {"count": 3, "percent": 2.0}}
    monkeypatch.setattr(project_cli, "run_asmr_cli", raising(stage, ["s000012"], 51))
    e.run_budgeted_stage(None, state, stage, Path("project.json"))
    assert state["partial"][0]["stage"] == stage
    assert state["partial"][0]["items"][0]["id"] == "s000012"


def test_failures_over_budget_fail_and_list_the_sentences(monkeypatch):
    state = {"failure_budget": {"count": 3, "percent": 2.0}}
    ids = [f"s{n:06d}" for n in range(1, 8)]
    monkeypatch.setattr(project_cli, "run_asmr_cli", raising("translate", ids, 51))
    with pytest.raises(e.StageFailedError) as caught:
        e.run_budgeted_stage(None, state, "translate", Path("project.json"))
    assert "超过批量允许的 3 句" in str(caught.value) and "s000001" in str(caught.value)
    assert "partial" not in state


def test_a_whole_stage_failure_is_never_partial(monkeypatch):
    state = {}
    monkeypatch.setattr(
        project_cli,
        "run_asmr_cli",
        raising("translate", [], 5, error="连续 3 批翻译失败"),
    )
    with pytest.raises(e.StageFailedError, match="连续 3 批翻译失败"):
        e.run_budgeted_stage(None, state, "translate", Path("project.json"))


def test_mix_uses_allow_missing_only_after_a_dubbing_loss():
    assert stages.mix_arguments({}) == ()
    assert stages.mix_arguments({"partial": [{"stage": "translate"}]}) == ()
    assert stages.mix_arguments({"partial": [{"stage": "synthesize"}]}) == ("--allow-missing",)


def test_queue_reports_partial_works_with_their_sentences_and_own_log(home, monkeypatch):
    for name in ("save_failed_task", "remove_failed_task"):
        monkeypatch.setattr(failed_tasks, name, lambda *a, **k: None)
    plans = [
        SimpleNamespace(plan_id=pid, folder=home / pid, output_root=home / f"o{pid}", retry_of=None)
        for pid in ("a1", "b2", "c3")
    ]
    record = {"stage": "synthesize", "count": 1, "total": 20, "items": [{"id": "s000007"}]}
    events = []

    def executor(paths, config, plan):
        e.append_log(f"hello from {plan.plan_id}")
        if plan.plan_id == "b2":
            return [record]
        if plan.plan_id == "c3":
            raise e.VideoPreparerError("配音失败：boom")
        return []

    code = e.execute_smart_queue(
        None, None, plans, executor=executor, reference_event_callback=events.append
    )
    finished = {ev["plan_id"]: ev for ev in events if ev["kind"] == "plan_finished"}
    assert code == 1  # a failed work still fails the queue
    assert {k: v["status"] for k, v in finished.items()} == {
        "a1": "completed",
        "b2": "partial",
        "c3": "failed",
    }
    assert finished["b2"]["warnings"] == [record] and finished["b2"]["error"] == ""
    texts = {pid: Path(finished[pid]["log"]).read_text(encoding="utf-8") for pid in finished}
    assert "hello from b2" in texts["b2"] and "hello from a1" not in texts["b2"]
    assert "配音失败：boom" in finished["c3"]["error"]
    assert e._WORK_LOG is None  # later messages no longer go to the last work's log


def test_partial_only_queue_succeeds(home, monkeypatch):
    plan = SimpleNamespace(plan_id="p", folder=home / "w", output_root=home / "o", retry_of=None)
    for name in ("save_failed_task", "remove_failed_task"):
        monkeypatch.setattr(failed_tasks, name, lambda *a, **k: None)
    code = e.execute_smart_queue(None, None, [plan], executor=lambda *a: [{"stage": "translate"}])
    assert code == 0


def test_work_log_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "WORK_LOG_MAX_BYTES", 100)
    path = stages.work_log_path(tmp_path, "plan/../x", "my:work*")
    assert path.parent == tmp_path / "logs" and ".." not in path.name and ":" not in path.name
    for _ in range(10):
        stages.append_work_log(path, "x" * 60)
    assert path.stat().st_size < 400 and path.with_suffix(".log.1").is_file()


def test_failure_budget_setting_reaches_the_engine_config(home):
    from asmr_dubber.services import settings
    from asmr_dubber.services.batch_configuration import config_from_settings

    config = config_from_settings()
    assert (config.failure_budget_count, config.failure_budget_percent) == (3, 2.0)
    settings.update({"batch_failure_budget_count": 5, "batch_failure_budget_percent": 10})
    config = config_from_settings()
    assert (config.failure_budget_count, config.failure_budget_percent) == (5, 10.0)
    with pytest.raises(ValueError):
        settings.update({"batch_failure_budget_percent": 500})


# --- mixing with missing dubs ---------------------------------------------------


def project_with_one_missing_dub(directory: Path, monkeypatch) -> DubProject:
    rate = 16_000
    tone = (0.03 * np.sin(2 * np.pi * 220 * np.arange(rate * 3) / rate)).astype(np.float32)
    sf.write(directory / "source.wav", np.column_stack([tone, tone]), rate, subtype="FLOAT")
    (directory / "chinese").mkdir()
    voice = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)).astype(np.float32)
    sf.write(directory / "chinese" / "s000001.wav", voice, rate, subtype="FLOAT")
    sentences = [
        Sentence(
            id="s000001",
            start_seconds=0,
            end_seconds=1,
            source_text="a",
            zh_text="你好。",
            tts_file="chinese/s000001.wav",
            tts_duration_seconds=0.5,
            tts_cache_key="valid",
            status="synthesized",
        ),
        Sentence(
            id="s000002",
            start_seconds=1.5,
            end_seconds=2.5,
            source_text="b",
            zh_text="再见。",
            status="error",
            error="合成失败",
        ),
    ]
    project = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256=sha256_file(directory / "source.wav"),
            duration_seconds=3.0,
            sample_rate=rate,
            channels=2,
        ),
        source_language="en",
        settings=ProjectSettings(
            tts_backend="gpt_sovits",
            tts_model="GPT-SoVITS-v4",
            tts_clone_mode="reference_only",
            normalize_chinese_loudness=False,
            match_source_loudness=False,
            mix_peak_protection=False,
        ),
        sentences=sentences,
    )
    monkeypatch.setattr("asmr_dubber.pipeline.tts_cache_key", lambda *_a: "valid")
    return project


def test_mix_refuses_missing_dubs_unless_told_to_leave_them_out(tmp_path, monkeypatch):
    project = project_with_one_missing_dub(tmp_path, monkeypatch)
    with pytest.raises(ProjectError, match="仍有 1 句的中文配音缺失"):
        mix_project(project, tmp_path)
    output = mix_project(project, tmp_path, allow_missing=True)
    assert output.is_file()
    missing = project.sentences[1]
    assert not sentence_is_replaced(missing)  # the fully dubbed version keeps its original voice
    assert sentence_is_replaced(project.sentences[0])
    assert missing.status == "error" and missing.error == "合成失败"  # still retryable


# --- preflight ------------------------------------------------------------------


def make_plan(tmp_path, *, subtitles_only=False, source_only=False, variants=("bilingual",)):
    source = SimpleNamespace(
        path=tmp_path / "a.wav",
        size=1000,
        transcript_path=None,
        transcript_timed=False,
        transcript_mode="direct",
    )
    return SimpleNamespace(
        plan_id="p1",
        folder=tmp_path / "work",
        output_root=tmp_path / "out",
        sources=(source,),
        subtitles_only=subtitles_only,
        source_subtitles_only=source_only,
        edition={"output_policy": {"variants": list(variants), "settings": {}}},
    )


def Hardware(vram, cuda=True):
    from asmr_dubber.runtime_manager import HardwareProfile

    return HardwareProfile(
        platform_id="windows",
        platform_label="Windows",
        architecture="x64",
        cpu="测试",
        memory_gb=32.0,
        gpu="测试显卡" if cuda else None,
        vram_gb=vram,
        driver="1.0",
        cuda_available=cuda,
    )


@pytest.fixture
def quiet_machine(home, monkeypatch):
    from asmr_dubber import runtime_manager

    monkeypatch.setattr(batch_preflight, "plan_seconds", lambda plan: 3600.0)
    monkeypatch.setattr(batch_preflight, "_disk_free", lambda path: (500 * 1024**3, 1))
    monkeypatch.setattr(runtime_manager, "detect_hardware", lambda: Hardware(24))
    monkeypatch.setattr(
        runtime_manager,
        "backend_status",
        lambda spec, settings=None: runtime_manager.BackendStatus("ready", "可用"),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    return home


def by_id(report):
    return {check.id: check for check in report.checks}


def test_preflight_passes_on_a_healthy_machine_and_never_uses_the_network(
    quiet_machine, monkeypatch
):
    from asmr_dubber import backend_diagnostics

    sent = []
    monkeypatch.setattr(
        backend_diagnostics, "test_translation_api", lambda settings, key: sent.append(key) or "ok"
    )
    report = batch_preflight.run_preflight([make_plan(quiet_machine)])
    assert report.ok and sent == ["test-not-a-real-key"]
    assert {check.status for check in report.checks} <= {OK, batch_preflight.SKIP}
    assert "够用" in by_id(report)["disk"].message


def test_preflight_blocks_on_low_disk_with_a_clear_message(quiet_machine, monkeypatch):
    monkeypatch.setattr(batch_preflight, "_disk_free", lambda path: (2 * 1024**3, 1))
    report = batch_preflight.run_preflight([make_plan(quiet_machine)], network=False)
    assert not report.ok
    message = by_id(report)["disk"].message
    assert "剩余 2.0 GB" in message and "预计需要约" in message and "1 个作品" in message
    assert "没有处理任何文件" in report.summary()


def test_preflight_vram_levels(quiet_machine, monkeypatch):
    from asmr_dubber import runtime_manager

    plans = [make_plan(quiet_machine)]
    monkeypatch.setattr(runtime_manager, "detect_hardware", lambda: Hardware(1.5))
    low = by_id(batch_preflight.run_preflight(plans, network=False))["vram"]
    assert low.status == FAIL and "显存" in low.message and "1.5 GB" in low.message
    monkeypatch.setattr(runtime_manager, "detect_hardware", lambda: Hardware(None, cuda=False))
    nogpu = by_id(batch_preflight.run_preflight(plans, network=False))["vram"]
    assert nogpu.status in {WARN, FAIL} and "NVIDIA" in nogpu.message


def test_preflight_reports_a_bad_translation_key_without_leaking_it(quiet_machine, monkeypatch):
    from asmr_dubber import backend_diagnostics

    def refuse(settings, key):
        raise ProjectError("HTTP 401 for key sk-abcdefghijklmnopqrstuvwx")

    monkeypatch.setattr(backend_diagnostics, "test_translation_api", refuse)
    report = batch_preflight.run_preflight([make_plan(quiet_machine)])
    check = by_id(report)["translation"]
    assert check.status == FAIL and "401" in check.message
    assert "sk-abcdefghijkl" not in check.message and "没有处理任何文件" in report.summary()
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    missing = by_id(batch_preflight.run_preflight([make_plan(quiet_machine)]))["translation"]
    assert missing.status == FAIL and "API Key" in missing.message


def test_preflight_reports_missing_models(quiet_machine, monkeypatch):
    from asmr_dubber import runtime_manager

    monkeypatch.setattr(
        runtime_manager,
        "backend_status",
        lambda spec, settings=None: runtime_manager.BackendStatus("missing", "未安装", "缺少模型"),
    )
    report = batch_preflight.run_preflight([make_plan(quiet_machine)], network=False)
    failed = [c for c in report.checks if c.id == "models"]
    assert failed and all(c.status == FAIL and "设备与模型" in c.message for c in failed)


def test_replacement_needs_the_separation_model(quiet_machine, monkeypatch):
    report = batch_preflight.run_preflight(
        [make_plan(quiet_machine, variants=("replace",))], network=False
    )
    assert by_id(report)["separation"].status == FAIL


def test_subtitle_only_work_needs_nothing(quiet_machine):
    report = batch_preflight.run_preflight([make_plan(quiet_machine, source_only=True)])
    assert report.ok and all(check.status == batch_preflight.SKIP for check in report.checks)


def test_a_broken_check_is_a_warning_not_a_crash(quiet_machine, monkeypatch):
    monkeypatch.setattr(batch_preflight, "plan_seconds", lambda plan: 1 / 0)
    report = batch_preflight.run_preflight([make_plan(quiet_machine)], network=False)
    assert report.ok and any("没能完成" in check.message for check in report.checks)


def test_run_queue_stops_before_processing_when_preflight_fails(home, monkeypatch):
    from asmr_dubber.services import batch_queue

    plan = make_plan(home)
    monkeypatch.setattr(batch_queue, "deserialize_plan", lambda payload: plan)
    monkeypatch.setattr(batch_queue, "config_from_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(app_settings, "find_tool_paths", lambda config: None)
    monkeypatch.setattr(app_settings, "validate_asmr_version", lambda: "x")
    monkeypatch.setattr(
        batch_preflight,
        "run_preflight",
        lambda plans: batch_preflight.PreflightReport(
            [batch_preflight.Check("disk", FAIL, "磁盘空间", "空间不足")]
        ),
    )
    monkeypatch.setattr(
        smart_queue,
        "execute_smart_queue",
        lambda *a, **k: pytest.fail("must not start processing"),
    )
    with pytest.raises(ProjectError, match="空间不足"):
        batch_queue.run_queue([{"plan_id": "p1"}])


def test_every_preflight_message_has_an_english_translation():
    translations = catalog()
    missing = [
        t for t in (*batch_preflight.TEMPLATES, *batch_preflight.TITLES) if t not in translations
    ]
    assert not missing
    for template in batch_preflight.TEMPLATES:
        names = {part.split("}")[0] for part in template.split("{")[1:]}
        assert names == {part.split("}")[0] for part in translations[template].split("{")[1:]}


# --- diagnostics bundle ---------------------------------------------------------


def test_bundle_has_logs_settings_failures_and_doctor_but_never_secrets(home, monkeypatch):
    from asmr_dubber import app_logging
    from asmr_dubber.models import load_project
    from asmr_dubber.user_settings import save_api_key

    secret = "sk-ThisIsAFakeKeyForTests123456"
    save_api_key("deepseek", secret)
    secrets_file = home / "home/config/secrets.json"
    assert secrets_file.is_file() and secret in secrets_file.read_text(encoding="utf-8")

    manifest = make_project(home / "project")
    project, directory = load_project(manifest)
    project.sentences[2].status = "translation_failed"
    project.sentences[2].error = f"请求失败 api_key={secret}"
    save_project(project, directory)
    plan = SimpleNamespace(plan_id="plan1", folder=home / "作品A", output_root=home / "out")
    monkeypatch.setattr(
        plan_files,
        "load_plan_metadata",
        lambda plan_id: {"jobs": [{"job_id": "merged", "project_json": str(manifest)}]},
    )
    log = stages.work_log_path(e.STATE_ROOT, "plan1", "作品A")
    stages.append_work_log(log, f"运行命令 Authorization: Bearer {secret}\n本作品日志")
    e.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    e.LOG_FILE.write_text(f"autoflow {secret}", encoding="utf-8")
    app_logging.application_log_path().parent.mkdir(parents=True, exist_ok=True)
    app_logging.application_log_path().write_text("app log\n", encoding="utf-8")

    settings_values = {
        "translation_provider": "deepseek",
        "tts_api_key": secret,
        "translation_max_output_tokens": 4096,
        "translation_extra_body": f'{{"Authorization": "Bearer {secret}"}}',
    }
    path, names = diagnostics.build_bundle(
        [plan], settings_values, doctor=lambda: f"doctor ok {secret}"
    )
    assert path.parent == home / "home" / "temp"
    with zipfile.ZipFile(path) as bundle:
        assert sorted(bundle.namelist()) == sorted(names)
        everything = "\n".join(bundle.read(name).decode("utf-8") for name in bundle.namelist())
        assert not any("secrets" in name for name in bundle.namelist())
        settings_json = json.loads(bundle.read("settings.json"))
        failed = json.loads(bundle.read("failed-sentences.json"))
        assert "doctor ok" in bundle.read("doctor.txt").decode("utf-8")
    assert secret not in everything and "sk-ThisIs" not in everything
    assert settings_json["translation_max_output_tokens"] == 4096
    assert settings_json["tts_api_key"] == "[已隐藏]"
    assert "logs/application.log" in names and "logs/autoflow.log" in names
    assert any(name.startswith("logs/work-") for name in names)
    assert "本作品日志" in everything
    row = failed["sentences"][0]
    assert (row["sentence"], row["stage"], row["start"]) == ("s000003", "translate", 3)
    assert "こんにちは" not in everything  # sentence text is not exported


def test_diagnostics_route_is_registered_and_returns_a_download_link(home, monkeypatch):
    from asmr_dubber.api import API
    from asmr_dubber.services.application import Application

    application = Application(home / "tasks.json")
    api = API(application)
    monkeypatch.setattr(diagnostics, "run_doctor", lambda: "doctor")
    result = api.call("batch/diagnostics", {"identifier": ""})
    assert Path(result["path"]).is_file() and result["url"] and "doctor.txt" in result["files"]
    with pytest.raises(ValueError):
        api.call("batch/preflight", {})
