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
        # Fixed preset changes never switch the recognition or dubbing backend.
        assert not {"asr_backend", "tts_backend", "asr_model"} & set(item["changes"])
    assert "autoflow_default_content" not in entries["replacement"]["project_keys"]
    # The full dub only prefers IndexTTS-2.5, switched to when it is installed.
    replacement = entries["replacement"]
    assert replacement["prefer"] == {"tts_backend": "indextts2_5", "tts_model": "IndexTTS-2.5"}
    assert replacement["prefer_requires"] == "indextts2_5"
    ProjectSettings.model_validate(
        {**ProjectSettings().model_dump(), **replacement["prefer"]}
    ).validate_mix_dependencies()
    assert all(not item["prefer"] for key, item in entries.items() if key != "replacement")


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


def test_onboarding_is_skipped_automatically_for_existing_users(tmp_path, monkeypatch):
    from asmr_dubber.services.application import Application
    from asmr_dubber.user_settings import PROVIDER_PRESETS

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    for preset in PROVIDER_PRESETS.values():
        env_var = preset.get("env_var")
        if isinstance(env_var, str) and env_var:
            monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        "asmr_dubber.services.models.catalog",
        lambda: {"items": [{"id": "parakeet_nemo", "state": "missing"}]},
    )

    assert onboarding.required(settings.current(), settings.key_statuses()) is True
    assert Application(tmp_path / "tasks.json").bootstrap()["onboarding_required"] is True
    # The user's explicit wizard completion flag remains False until they act on it.
    assert settings.current().onboarding_completed is False

    # Configured model -> skip wizard automatically.
    assert (
        onboarding.required(
            settings.current(),
            settings.key_statuses(),
            model_items=[{"id": "parakeet_nemo", "state": "ready"}],
        )
        is False
    )

    # Configured API key -> skip wizard automatically even with no models downloaded.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-existing")
    assert onboarding.required(settings.current(), settings.key_statuses()) is False
    assert Application(tmp_path / "tasks.json").bootstrap()["onboarding_required"] is False

    wizard = files("asmr_dubber").joinpath("frontend/wizard.js").read_text(encoding="utf-8")
    assert "onboarding_required" in wizard


def test_translation_api_falls_back_to_preset_base_url_and_avoids_zh_shortcut(monkeypatch):
    import asmr_dubber.backend_diagnostics as diagnostics
    from asmr_dubber.errors import ProjectError
    from asmr_dubber.user_settings import PROVIDER_PRESETS, UserSettings

    calls: list[dict[str, object]] = []

    def fake_translate(sentences, **kwargs):
        calls.append(kwargs)
        assert kwargs["source_language"] != kwargs["target_language"]
        sentences[0].zh_text = "Hello." if kwargs["target_language"] == "en" else "你好。"

    monkeypatch.setattr(diagnostics, "translate_sentences", fake_translate)
    monkeypatch.setattr(diagnostics, "resolve_api_key", lambda provider, key: "sk-test")

    # Empty translation_base_url falls back to PROVIDER_PRESETS[provider]["base_url"].
    fresh = UserSettings(translation_provider="deepseek", translation_base_url="")
    out_ja = diagnostics.test_translation_api(fresh)
    assert "こんにちは。 → 你好。" in out_ja
    assert calls[-1]["base_url"] == PROVIDER_PRESETS["deepseek"]["base_url"]
    assert calls[-1]["target_language"] == "zh"

    # Chinese default_source_language translates zh -> en over the network, never short-circuiting.
    zh_settings = UserSettings(
        default_source_language="zh",
        translation_provider="deepseek",
        translation_base_url="",
    )
    out_zh = diagnostics.test_translation_api(zh_settings)
    assert "你好。 → Hello." in out_zh
    assert calls[-1]["source_language"] == "zh"
    assert calls[-1]["target_language"] == "en"
    assert calls[-1]["base_url"] == PROVIDER_PRESETS["deepseek"]["base_url"]

    # Real translate_sentences must not hit source_language == target_language bypass for zh.
    network_hits = []

    class FakeDeepSeek:
        max_retries = 1

        def __init__(self, **kwargs):
            network_hits.append(kwargs["base_url"])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def close(self):
            return None

        def translate_chunk(self, chunk, *_args, **_kwargs):
            return {chunk.sentences[0].id: "   "}

    monkeypatch.undo()
    monkeypatch.setattr(diagnostics, "resolve_api_key", lambda provider, key: "sk-test")
    monkeypatch.setattr("asmr_dubber.translation.DeepSeekTranslator", FakeDeepSeek)
    with pytest.raises(ProjectError):
        diagnostics.test_translation_api(
            UserSettings(default_source_language="zh", translation_base_url="")
        )
    assert network_hits == [PROVIDER_PRESETS["deepseek"]["base_url"]]


def test_models_catalog_and_recommended_bundle_align_with_onboarding(tmp_path, monkeypatch):
    from asmr_dubber.services import models

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    monkeypatch.setattr(
        models.runtime,
        "backend_status",
        lambda *a, **k: models.runtime.BackendStatus("missing", "未安装"),
    )
    installed: list[str] = []
    monkeypatch.setattr(
        models.runtime,
        "install_backend",
        lambda backend, **_kw: installed.append(backend) or f"ok:{backend}",
    )

    class Token:
        def is_set(self):
            return False

    monkeypatch.setattr(runtime, "detect_hardware", lambda: GPU)
    gpu_catalog = models.catalog()
    assert gpu_catalog["recommended"] == ["anime_whisper", "indextts2"]
    assert {item["id"] for item in gpu_catalog["items"] if item.get("recommended")} == {
        "anime_whisper",
        "indextts2",
    }
    models._download({"model": "recommended"}, lambda *a, **k: None, Token())
    assert installed == ["anime_whisper", "indextts2"]

    installed.clear()
    cpu_profile = replace(GPU, gpu=None, vram_gb=None, cuda_available=False)
    monkeypatch.setattr(runtime, "detect_hardware", lambda: cpu_profile)
    cpu_catalog = models.catalog()
    assert cpu_catalog["recommended"] == ["parakeet_nemo"]
    models._download({"model": "recommended"}, lambda *a, **k: None, Token())
    assert installed == ["parakeet_nemo"]

    models_js = files("asmr_dubber").joinpath("frontend/models.js").read_text(encoding="utf-8")
    assert "anime-whisper" in models_js and "catalog.recommended" in models_js
    translations = locale_catalog()
    for name in ("session.js", "app.js", "projects.js", "compare_player.js", "models.js"):
        source = files("asmr_dubber").joinpath("frontend", name).read_text(encoding="utf-8")
        assert len(source.splitlines()) < 500, name
        for text in re.findall(r"\bt\('([^'$]*[一-鿿][^']*)'", source):
            assert text in translations, f"{name}: {text}"
