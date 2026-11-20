import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.api import API
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import ProjectSettings, load_project
from asmr_dubber.services import models, parameters, settings
from asmr_dubber.services.application import Application
from asmr_dubber.services.tasks import ACTIVE, Tasks


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return Application(tmp_path / "tasks.json")


def completed(tasks, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        task = tasks.get(identifier)
        if task["status"] not in ACTIVE:
            return task
        time.sleep(0.01)
    pytest.fail("Task failed to finish")


def create(application, tmp_path):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000 * 3), 16000)
    request = {"kind": "create", "source": str(audio), "source_language": "en"}
    task = application.start(request)
    result = completed(application.tasks, task["id"])
    assert result["status"] == "completed", result
    assert set(result["result"]) == {"manifest"}
    return application.projects.get(result["result"]["manifest"])


def test_catalog_has_every_setting_once_and_real_constraints():
    catalog = parameters.catalog()
    by_key = {item["key"]: item for item in catalog}
    assert len(by_key) == len(catalog) == len(settings.Settings.model_fields)
    actual = settings.Settings().model_dump()
    for key, value in actual.items():
        assert by_key[key]["default"] == value
        assert by_key[key]["label"]
        assert by_key[key]["group"]
        assert by_key[key]["scope"] == (
            "project" if key in ProjectSettings.model_fields else "global"
        )
    assert by_key["asr_batch_size"]["maximum"] == 32
    assert by_key["asr_review_models"]["type"] == "array"
    assert by_key["tts_index25_emotion_vector"]["items"]["maximum"] == 1


def test_global_defaults_do_not_change_existing_project(application, tmp_path):
    project = create(application, tmp_path)
    original = project["settings"]["tts_speed"]
    application.update_settings({"tts_speed": 1.2})
    assert load_project(project["manifest"])[0].settings.tts_speed == original
    assert settings.current().tts_speed == 1.2


def test_project_settings_leave_defaults_and_unrelated_parameters_intact(application, tmp_path):
    project = create(application, tmp_path)
    defaults = settings.current().tts_speed
    application.update_settings({"tts_speed": 1.3}, project["manifest"], project["revision"])
    active = load_project(project["manifest"])[0]
    assert active.settings.tts_speed == 1.3
    assert active.settings.asr_model == project["settings"]["asr_model"]
    assert settings.current().tts_speed == defaults
    with pytest.raises(ProjectError):
        application.update_settings({"tts_speed": 1.4}, project["manifest"], project["revision"])
    assert load_project(project["manifest"])[0].settings.tts_speed == 1.3


@pytest.mark.parametrize(
    "changes",
    [{"bogus": 1}, {"tts_speed": 100}, {"asr_batch_size": 0}, {"tts_api_extra_body": "[]"}],
)
def test_settings_reject_invalid_changes(application, changes):
    before = settings.current().model_dump()
    with pytest.raises((ValueError, ProjectError)):
        application.update_settings(changes)
    assert settings.current().model_dump() == before


def test_table_preserves_sentence_controls_and_rejects_stale_save(application, tmp_path):
    project = create(application, tmp_path)
    rows = [["s1", True, 0, 1, "hello", "你好", "off", -3, True, 2]]
    saved = application.save_table(project["manifest"], rows, project["revision"])
    assert saved["rows"] == rows
    with pytest.raises(ProjectError):
        application.save_table(project["manifest"], rows, project["revision"])


def test_sentence_patch_keeps_other_rows_and_rejects_unknown_sentence(application, tmp_path):
    project = create(application, tmp_path)
    rows = [
        ["s1", True, 0, 1, "hello", "你好", "off", -3, True, 2],
        ["s2", True, 1, 2, "bye", "再见", "default", 0, True, 0],
    ]
    saved = application.save_table(project["manifest"], rows, project["revision"])
    changed = [*rows[0]]
    changed[5] = "您好"
    saved = application.save_sentence(project["manifest"], changed, saved["revision"])
    assert saved["rows"] == [changed, rows[1]]
    changed[0] = "missing"
    with pytest.raises(ValueError, match="已不存在"):
        application.save_sentence(project["manifest"], changed, saved["revision"])
    assert load_project(project["manifest"])[0].sentences[0].original_audio_enabled is False


def test_cancel_resume_and_restart_use_persisted_request(tmp_path):
    began = threading.Event()
    runs = []

    def execute(request, report, token, reference):
        runs.append(request)
        report("progress", 1, 4)
        reference({"kind": "ready", "project_json": "example"})
        began.set()
        if len(runs) == 1:
            token.wait(2)
        from asmr_dubber.task_control import check_cancelled

        check_cancelled(token)
        return {"ok": True}

    tasks = Tasks(execute, tmp_path / "tasks.json")
    task = tasks.start({"kind": "example", "value": 3}, "project")
    assert began.wait(3)
    assert tasks.get(task["id"])["reference"]["kind"] == "ready"
    with pytest.raises(ValueError):
        tasks.start({"kind": "example"}, "project")
    tasks.cancel(task["id"])
    assert completed(tasks, task["id"])["status"] == "cancelled"
    restored = Tasks(execute, tmp_path / "tasks.json")
    restored.resume(task["id"])
    result = completed(restored, task["id"])
    assert result["status"] == "completed"
    assert result["result"] == {"ok": True}
    assert runs == [{"kind": "example", "value": 3}] * 2
    assert result["reference"] is None


def test_restart_marks_unfinished_tasks_interrupted(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"t": {"id": "t", "status": "running"}}))
    tasks = Tasks(lambda *args: None, path)
    assert tasks.get("t")["status"] == "interrupted"
    assert json.loads(path.read_text(encoding="utf-8"))["t"]["status"] == "interrupted"


def test_task_failure_is_saved_and_secrets_are_rejected(tmp_path):
    def execute(*args):
        raise ValueError("failed")

    tasks = Tasks(execute, tmp_path / "tasks.json")
    with pytest.raises(ValueError):
        tasks.start({"kind": "example", "api_key": "secret"}, "p")
    task = tasks.start({"kind": "example"}, "p")
    assert completed(tasks, task["id"])["error"] == "failed"


def test_upload_is_owned_and_truncated_upload_is_removed(application):
    result = application.media.upload(io.BytesIO(b"audio"), 5, "../audio.wav")
    path = Path(result["path"])
    assert path.name == "audio.wav"
    assert path.read_bytes() == b"audio"
    assert application.media.resolve(result["url"].split("/")[-1])[0] == path
    with pytest.raises(ValueError):
        application.media.upload(io.BytesIO(b"a"), 3, "bad.wav")
    assert not list(path.parent.parent.rglob("bad.wav"))
    with pytest.raises(FileNotFoundError):
        application.media.resolve("../secrets.json")


def test_batch_queue_scan_save_edit_reorder_and_subtitles(application, tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    for name in ("01.wav", "02.wav"):
        sf.write(folder / name, np.zeros(16000), 16000)
    (folder / "01.srt").write_text("1\n00:00:00,000 --> 00:00:00,900\nhello\n", encoding="utf-8")
    scan = application.batch.scan(str(folder))
    assert len(scan["source_payloads"]) == 2
    queue = application.batch.save(
        str(folder), scan["selected_edition"], scan["source_payloads"], "audio", "merged"
    )
    assert len(queue) == 1
    identifier = queue[0]["plan_id"]
    edited = application.batch.edit(identifier)
    assert edited["mode"] == "audio"
    assert edited["source_payloads"] == scan["source_payloads"]
    assert application.batch.reorder([identifier]) == queue
    with pytest.raises(ProjectError):
        application.batch.reorder([])
    assert application.batch.remove(identifier) == []


def test_native_api_validates_before_calling_service(application, monkeypatch):
    api = API(application)
    called = []
    api.routes["projects/table"] = (api.routes["projects/table"][0], lambda **kw: called.append(kw))
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        api.call("projects/table", {"project": "p", "revision": "1", "rows": []})
    assert called == []
    with pytest.raises(ValidationError):
        api.call("tasks/start", {"kind": "analyze"})
    assert "parameters" in api.call("bootstrap", {})
    with pytest.raises(KeyError):
        api.call("unrecognized", {})


def test_model_catalog_uses_registry_numbers(application, monkeypatch):
    monkeypatch.setattr(
        models.runtime,
        "backend_status",
        lambda *a, **k: models.runtime.BackendStatus("ready", "ready"),
    )
    result = models.catalog()
    from asmr_dubber.model_registry import ASR_BACKENDS

    item = next(item for item in result["items"] if item["id"] == "parakeet_nemo")
    assert item["disk_gb"] == ASR_BACKENDS["parakeet_nemo"].disk_gb
    assert item["vram_gb"] == ASR_BACKENDS["parakeet_nemo"].recommended_vram_gb


def test_model_delete_refuses_external_path(application, monkeypatch, tmp_path):
    external = tmp_path / "outside-models"
    external.mkdir()
    (external / "weights.bin").write_bytes(b"keep")
    monkeypatch.setattr(models, "cached_model_path", lambda *_: external)
    with pytest.raises(ValueError):
        models.remove("faster_whisper")
    assert (external / "weights.bin").read_bytes() == b"keep"


def test_reference_setting_can_be_selected_cleared_and_rejects_missing_sentence(
    application, monkeypatch, tmp_path
):
    from asmr_dubber.services import project_audio

    monkeypatch.setattr(project_audio, "reference_preview", lambda *args: None)
    project = create(application, tmp_path)
    rows = [["s1", True, 0, 1, "hello", "你好", "default", 0, True, 0]]
    project = application.save_table(project["manifest"], rows, project["revision"])
    application.update_settings(
        {"tts_reference_sentence_id": "s1"}, project["manifest"], project["revision"]
    )
    active, _ = load_project(project["manifest"])
    assert active.settings.tts_reference_sentence_id == "s1"
    application.update_settings(
        {
            "tts_reference_source": "external",
            "tts_external_reference_audio": str(tmp_path / "source.wav"),
        },
        project["manifest"],
        active.revision,
    )
    external = application.projects.get(project["manifest"])
    assert external["reference_external"] is True
    assert external["external_reference_url"].startswith("/media/")
    active, _ = load_project(project["manifest"])
    application.update_settings(
        {"tts_reference_sentence_id": None}, project["manifest"], active.revision
    )
    active, _ = load_project(project["manifest"])
    assert active.settings.tts_reference_sentence_id is None
    before = active.model_dump()
    with pytest.raises(ProjectError):
        application.update_settings(
            {"tts_reference_sentence_id": "missing"}, project["manifest"], active.revision
        )
    assert load_project(project["manifest"])[0].model_dump() == before


def test_cancelling_terminal_task_during_worker_cleanup_preserves_its_state(tmp_path):
    from asmr_dubber.task_control import CancellationToken

    tasks = Tasks(lambda *args: {}, tmp_path / "tasks.json")
    tasks.items["finished"] = {"id": "finished", "status": "cancelled"}
    tasks.tokens["finished"] = CancellationToken()
    assert tasks.cancel("finished")["status"] == "cancelled"
    assert not tasks.tokens["finished"].is_set()


@pytest.mark.parametrize("content", ["subtitles", "source_subtitles"])
def test_batch_subtitle_content_sets_core_execution_flags(application, tmp_path, content):
    folder = tmp_path / "subtitle-work"
    folder.mkdir()
    sf.write(folder / "01.wav", np.zeros(16000), 16000)
    scan = application.batch.scan(str(folder))
    queue = application.batch.save(
        str(folder),
        scan["selected_edition"],
        scan["source_payloads"],
        "video_normal",
        "merged",
        content=content,
    )
    assert queue[0]["subtitles_only"] is True
    assert queue[0]["source_subtitles_only"] == (content == "subtitles")
    assert queue[0]["edition"]["output_policy"]["content"] == content
    if content == "subtitles":
        assert queue[0]["mode"] == "audio"


def test_failed_batch_retains_outputs_and_can_resume(application, monkeypatch, tmp_path):
    from asmr_dubber.services import batch_queue

    monkeypatch.setattr(batch_queue, "run_queue", lambda *args, **kwargs: (1, ["partial-output"]))
    monkeypatch.setattr(batch_queue, "subtitle_output_rows", lambda *args: [])
    application.batch._save(
        [{"plan_id": "one", "folder": str(tmp_path), "output_root": str(tmp_path / "out")}]
    )
    task = application.start({"kind": "batch"})
    failed = completed(application.tasks, task["id"])
    assert failed["status"] == "failed"
    assert failed["result"]["outputs"] == ["partial-output"]
    assert failed["error"]
    monkeypatch.setattr(batch_queue, "run_queue", lambda *args, **kwargs: (0, ["full-output"]))
    application.resume(task["id"])
    completed_task = completed(application.tasks, task["id"])
    assert completed_task["status"] == "completed"
    assert completed_task["result"]["outputs"] == ["full-output"]


def test_model_storage_counts_checkpoints_and_cache_without_overlapping_totals(application):
    from asmr_dubber.platforms import portable_home

    home = portable_home()
    application.update_settings(
        {
            "tts_model_path": str(home / "runtimes/index-tts/checkpoints"),
            "tts_index25_model_path": str(home / "runtimes/index-tts-2.5/checkpoints"),
        }
    )
    for relative, count in (
        ("models/separation/weights.bin", 7),
        ("runtimes/index-tts/checkpoints/weights.bin", 11),
        ("cache/huggingface/hub/models--test/blobs/weights.bin", 13),
        ("cache/downloads/temporary.zip", 17),
    ):
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"a" * count)
    result = models.model_storage(settings.current())
    assert (result["total_bytes"], result["cache_bytes"]) == (31, 13)
    assert application.storage()["sizes"]["模型"] == 31
    assert application.storage()["sizes"]["缓存"] == 17


def test_storage_lists_every_large_folder_once_and_opens_only_known_ones(application, monkeypatch):
    from asmr_dubber.platforms import portable_home
    from asmr_dubber.services import storage_report

    home = portable_home()
    application.update_settings(
        {
            "tts_model_path": str(home / "runtimes/index-tts/checkpoints"),
            "tts_index25_model_path": str(home / "runtimes/index-tts-2.5/checkpoints"),
        }
    )
    for relative, count in (
        ("runtimes/index-tts/checkpoints/gpt.pth", 11),  # counted as a model
        ("runtimes/index-tts/.venv/torch.dll", 23),  # the backend's own environment
        ("projects/work/source.wav", 5),
    ):
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"a" * count)
    storage = application.storage()
    by_key = {item["key"]: item for item in storage["locations"]}
    assert by_key["runtimes"]["bytes"] == 23
    assert by_key["models"]["bytes"] == 11
    assert by_key["projects"]["bytes"] == 5 and by_key["projects"]["exists"]
    assert storage["sizes"]["后端运行环境"] == 23
    assert all(item["hint"] for item in storage["locations"])

    opened = []
    monkeypatch.setattr(storage_report, "open_directory", lambda path: opened.append(path) or path)
    application.open_storage("runtimes")
    assert opened == [home / "runtimes"]
    with pytest.raises(ValueError):
        application.open_storage("../../Windows")
    with pytest.raises(ValueError):
        application.open_storage("model_packs" if not by_key.get("model_packs") else "nothing")


def test_directory_size_skips_entries_it_cannot_read(tmp_path, monkeypatch):
    (tmp_path / "ok.bin").write_bytes(b"abc")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "x.bin").write_bytes(b"zz")
    real_scandir = models.os.scandir

    def scandir(path):
        if str(path) == str(locked):
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr(models.os, "scandir", scandir)
    assert models.directory_bytes(tmp_path) == 3
    assert models.directory_bytes(tmp_path / "missing") == 0


def _dummy_plan(
    tmp_path: Path,
    *,
    plan_id: str = "p1",
    subtitles_only: bool = False,
    source_only: bool = False,
    variants: tuple[str, ...] = ("bilingual",),
    policy_settings: dict | None = None,
    sources: tuple | None = None,
    subtitle_language: str = "bilingual",
):
    from types import SimpleNamespace

    default_source = SimpleNamespace(
        path=tmp_path / "01.wav",
        size=1000,
        transcript_path=None,
        transcript_language=None,
        transcript_timed=False,
        transcript_mode="direct",
        source_language="ja",
    )
    return SimpleNamespace(
        plan_id=plan_id,
        folder=tmp_path / f"work-{plan_id}",
        output_root=tmp_path / f"out-{plan_id}",
        sources=sources if sources is not None else (default_source,),
        subtitles_only=subtitles_only,
        source_subtitles_only=source_only,
        subtitle_language=subtitle_language,
        edition={
            "output_policy": {
                "variants": list(variants),
                "subtitle": subtitle_language,
                "settings": dict(policy_settings or {}),
            }
        },
    )


def test_preflight_caches_translation_probe_across_preflight_and_run_queue(
    application, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from asmr_dubber import backend_diagnostics, runtime_manager
    from asmr_dubber.autoflow import app_settings, smart_queue
    from asmr_dubber.services import batch_preflight, batch_queue

    batch_preflight.clear_translation_probe_cache()
    monkeypatch.setattr(batch_preflight, "plan_seconds", lambda plan: 60.0)
    monkeypatch.setattr(batch_preflight, "_disk_free", lambda path: (200 * 1024**3, 1))
    monkeypatch.setattr(
        runtime_manager,
        "detect_hardware",
        lambda: runtime_manager.HardwareProfile(
            platform_id="windows",
            platform_label="Windows",
            architecture="x64",
            cpu="cpu",
            memory_gb=32.0,
            gpu="RTX",
            vram_gb=16.0,
            driver="1",
            cuda_available=True,
        ),
    )
    monkeypatch.setattr(
        runtime_manager,
        "backend_status",
        lambda *a, **k: runtime_manager.BackendStatus("ready", "可用"),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-cache-test")

    plan = _dummy_plan(tmp_path)
    calls: list[str] = []

    def flaky_then_ok(_settings, key):
        calls.append(key)
        if len(calls) == 1:
            raise ProjectError("temporary network failure")
        return "ok"

    monkeypatch.setattr(backend_diagnostics, "test_translation_api", flaky_then_ok)
    # Failure must NOT be cached.
    first = batch_preflight.run_preflight([plan])
    assert not first.ok and len(calls) == 1

    # Next preflight succeeds and caches for 300s; immediate run_queue reuses cached probe.
    second = batch_preflight.run_preflight([plan])
    assert second.ok and len(calls) == 2

    monkeypatch.setattr(batch_queue, "deserialize_plan", lambda payload: plan)
    monkeypatch.setattr(batch_queue, "config_from_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(app_settings, "find_tool_paths", lambda config: None)
    monkeypatch.setattr(app_settings, "validate_asmr_version", lambda: "ok")
    monkeypatch.setattr(smart_queue, "execute_smart_queue", lambda *a, **k: 0)
    exit_code, outputs = batch_queue.run_queue([{"plan_id": "p1"}], preflight=True)
    assert exit_code == 0 and outputs == [str(plan.output_root)]
    assert len(calls) == 2

    batch_preflight.clear_translation_probe_cache()
    assert batch_preflight.run_preflight([plan]).ok and len(calls) == 3


def test_preflight_vram_accounts_for_separation_aligner_and_per_plan_peaks(
    application, monkeypatch, tmp_path
):
    from asmr_dubber import runtime_manager
    from asmr_dubber.services import batch_preflight
    from asmr_dubber.user_settings import UserSettings

    def hw(vram: float):
        return runtime_manager.HardwareProfile(
            platform_id="windows",
            platform_label="Windows",
            architecture="x64",
            cpu="cpu",
            memory_gb=32.0,
            gpu="RTX",
            vram_gb=vram,
            driver="1",
            cuda_available=True,
        )

    # Local RoFormer separation (~3.0 GB min) is counted even when ASR & TTS need no dedicated VRAM.
    online_user = UserSettings(
        asr_backend="generic_asr_api",
        tts_backend="edge_tts",
        separation_backend="local",
        separation_model="vocals_mel_band_roformer.ckpt",
    )
    replace_plan = _dummy_plan(tmp_path, plan_id="rep", variants=("replace",))
    monkeypatch.setattr(runtime_manager, "detect_hardware", lambda: hw(2.0))
    sep_low = batch_preflight.check_vram([replace_plan], online_user)
    assert sep_low.status == batch_preflight.FAIL and "Mel-Band RoFormer" in sep_low.message

    # Forced alignment (Qwen3-ForcedAligner ~2.5 GB min) is counted when enabled on Parakeet.
    align_user = UserSettings(
        asr_backend="parakeet_nemo",
        asr_forced_alignment_enabled=True,
        tts_backend="edge_tts",
    )
    align_low = batch_preflight.check_vram([_dummy_plan(tmp_path)], align_user)
    assert align_low.status == batch_preflight.FAIL and "Qwen3 时间对齐" in align_low.message

    # Earlier heavy plan must NOT be masked by a later online plan in the queue.
    heavy_plan = _dummy_plan(
        tmp_path,
        plan_id="heavy",
        variants=("replace",),
        policy_settings={"tts_backend": "indextts2_5", "separation_enabled": True},
    )
    light_plan = _dummy_plan(
        tmp_path,
        plan_id="light",
        variants=("bilingual",),
        policy_settings={"tts_backend": "edge_tts", "separation_enabled": False},
    )
    monkeypatch.setattr(runtime_manager, "detect_hardware", lambda: hw(4.0))
    mixed = batch_preflight.check_vram([heavy_plan, light_plan], online_user)
    assert mixed.status == batch_preflight.FAIL and "IndexTTS-2.5" in mixed.message


def test_preflight_skips_translation_when_tracks_have_target_subtitles_or_same_language(
    application, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from asmr_dubber.services import batch_preflight
    from asmr_dubber.user_settings import PROVIDER_PRESETS, UserSettings

    for preset in PROVIDER_PRESETS.values():
        env_name = preset.get("env")
        if env_name:
            monkeypatch.delenv(env_name, raising=False)
    user = UserSettings(translation_provider="deepseek", tts_target_language="zh")

    srt = tmp_path / "01.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    zh_sub_source = SimpleNamespace(
        path=tmp_path / "01.wav",
        size=1000,
        transcript_path=srt,
        transcript_language="zh",
        transcript_timed=True,
        transcript_mode="direct",
        source_language="ja",
    )
    direct_sub_plan = _dummy_plan(tmp_path, sources=(zh_sub_source,))
    assert (
        batch_preflight.check_translation([direct_sub_plan], user, network=True).status
        == batch_preflight.SKIP
    )

    same_lang_source = SimpleNamespace(
        path=tmp_path / "02.wav",
        size=1000,
        transcript_path=None,
        transcript_language=None,
        transcript_timed=False,
        transcript_mode="direct",
        source_language="zh",
    )
    same_lang_plan = _dummy_plan(tmp_path, sources=(same_lang_source,))
    assert (
        batch_preflight.check_translation([same_lang_plan], user, network=True).status
        == batch_preflight.SKIP
    )

    # Untimed script reconciled against ASR may leave unmatched lines to translate -> require key.
    untimed_source = SimpleNamespace(
        path=tmp_path / "03.wav",
        size=1000,
        transcript_path=tmp_path / "script.txt",
        transcript_language="zh",
        transcript_timed=False,
        transcript_mode="asr_reconcile",
        source_language="ja",
    )
    untimed_plan = _dummy_plan(tmp_path, sources=(untimed_source,))
    assert (
        batch_preflight.check_translation([untimed_plan], user, network=False).status
        == batch_preflight.FAIL
    )
