"""First-run guide, presets, per-work batch stages and the global task panel."""

import re
from dataclasses import replace
from importlib.resources import files

import pytest

from asmr_dubber import runtime_manager as runtime
from asmr_dubber.autoflow import project_cli
from asmr_dubber.localization import catalog as locale_catalog
from asmr_dubber.models import ProjectSettings
from asmr_dubber.services import onboarding, presets, settings
from asmr_dubber.services.tasks import Tasks
from tests.test_native_services import completed

GPU = runtime.HardwareProfile(
    platform_id="windows",
    platform_label="Windows",
    architecture="x64",
    cpu="cpu",
    memory_gb=32.0,
    gpu="RTX",
    vram_gb=16.0,
    driver="1",
    cuda_available=True,
    compute_capability="8.9",
)


def test_presets_are_valid_settings_and_best_quality_is_the_defaults():
    entries = {item["id"]: item for item in presets.catalog()}
    assert set(entries) == {"fast", "balanced", "best", "bilingual", "replacement"}
    defaults = settings.Settings().model_dump()
    assert all(defaults[key] == value for key, value in entries["best"]["changes"].items())
    assert set(entries["best"]["changes"]) == set(entries["fast"]["changes"])
    for item in entries.values():
        values = {
            **ProjectSettings().model_dump(),
            **{k: item["changes"][k] for k in item["project_keys"]},
        }
        ProjectSettings.model_validate(values).validate_mix_dependencies()
        # Presets never switch the recognition or dubbing backend.
        assert not {"asr_backend", "tts_backend", "asr_model"} & set(item["changes"])
    assert "autoflow_default_content" not in entries["replacement"]["project_keys"]


def test_preset_roundtrips_through_the_settings_service(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    fast = next(item for item in presets.catalog() if item["id"] == "fast")
    saved = settings.update(fast["changes"])
    assert saved["translation_context_sentences"] == 8
    assert settings.current().tts_quality_check is False
    best = next(item for item in presets.catalog() if item["id"] == "best")
    settings.update(best["changes"])
    assert settings.current().translation_context_sentences == 24


def test_onboarding_flag_is_stored_with_the_interface_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    assert settings.current().onboarding_completed is False
    settings.update({"onboarding_completed": True})
    assert settings.current().onboarding_completed is True
    assert "onboarding_completed" in (tmp_path / "config" / "interface.json").read_text("utf-8")
    assert not any(
        item["key"] == "onboarding_completed" and not item.get("hidden")
        for item in __import__("asmr_dubber.services.parameters", fromlist=["x"]).catalog()
    )


@pytest.mark.parametrize(
    ("profile", "asr", "local_tts"),
    [
        (GPU, "anime_whisper", True),
        (replace(GPU, gpu=None, vram_gb=None, cuda_available=False), "parakeet_nemo", False),
    ],
)
def test_onboarding_recommendation_follows_the_hardware_probe(monkeypatch, profile, asr, local_tts):
    monkeypatch.setattr(runtime, "detect_hardware", lambda: profile)
    result = onboarding.recommendation()
    assert result["asr"]["id"] == asr
    assert bool(result["tts"]["id"]) is local_tts
    assert result["title"] and result["detail"]


def test_every_recommendation_text_has_an_english_translation(monkeypatch):
    translations = locale_catalog()
    for gpu, vram, capability in [
        ("RTX", 16.0, "8.9"),
        ("RTX", 9.0, "8.9"),
        ("GTX", 6.5, "8.6"),
        ("GTX", 16.0, "6.1"),
        (None, None, None),
    ]:
        profile = replace(
            GPU,
            gpu=gpu,
            vram_gb=vram,
            cuda_available=gpu is not None,
            compute_capability=capability,
        )
        lines = runtime.recommended_stack_markdown(profile).splitlines()
        title = lines[0].removeprefix("### ").removeprefix("本机建议：")
        assert title in translations, title
        assert "\n".join(lines[1:]) in translations, lines[1:]


def test_new_frontend_strings_are_translated_and_files_stay_small():
    root = files("asmr_dubber").joinpath("frontend")
    translations = locale_catalog()
    missing = set()
    for name in ("wizard.js", "presets.js", "task_panel.js", "notify.js", "batch_view.js"):
        source = root.joinpath(name).read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 500
        for text in re.findall(r"\bt\('([^'$]*[一-鿿][^']*)'", source):
            if text not in translations:
                missing.add(text)
    assert not missing
    for item in presets.catalog():
        assert item["label"] in translations and item["description"] in translations, item["id"]


def test_plan_stage_events_are_kept_per_work(tmp_path):
    def execute(request, report, token, reference):
        reference({"kind": "queue", "plans": [{"plan_id": "p", "work": "W", "output": "o"}]})
        reference({"kind": "plan_started", "plan_id": "p", "work": "W"})
        reference({"kind": "plan_stage", "plan_id": "p", "stage": "analyze"})
        reference({"kind": "plan_stage", "plan_id": "p", "stage": "translate"})
        reference({"kind": "plan_stage", "plan_id": "p", "stage": "translate"})
        return {"ok": True}

    tasks = Tasks(execute, tmp_path / "tasks.json")
    task = tasks.start({"kind": "batch"}, "batch")
    plan = completed(tasks, task["id"])["plans"]["p"]
    assert plan["stages"] == ["analyze", "translate"]
    assert plan["stage"] == "translate"
    assert plan["status"] == "running"


def test_stage_listener_hears_cli_command_names_and_never_breaks_the_engine():
    heard = []
    project_cli.STAGE_LISTENER = heard.append
    try:
        project_cli.announce_stage(["synthesize", "project.json"])
        project_cli.STAGE_LISTENER = lambda stage: 1 / 0
        project_cli.announce_stage(["mix"])
    finally:
        project_cli.STAGE_LISTENER = None
    assert heard == ["synthesize"]


def test_single_work_retry_is_a_plan_scoped_batch_request():
    from asmr_dubber.api_contract import TaskRequest

    assert TaskRequest.model_validate({"kind": "batch", "plan": "abc"}).plan == "abc"
