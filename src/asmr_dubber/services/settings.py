"""Global defaults, project settings and secrets have distinct ownership."""

import json
from pathlib import Path

from pydantic import Field

from ..errors import ProjectError
from ..lifecycle import browser_revision_scope
from ..model_registry import ASR_BACKENDS, TTS_BACKENDS
from ..models import ProjectSettings, load_project
from ..storage import exclusive_file_lock
from ..user_settings import (
    PROVIDER_PRESETS,
    UserSettings,
    api_key_status,
    clear_api_key,
    clear_service_key,
    config_dir,
    load_user_settings,
    save_api_key,
    save_service_key,
    save_user_settings,
    service_key_status,
    store_reference_audio,
)
from . import project_operations


class Settings(UserSettings):
    ui_language: str = Field(default="zh", pattern="^(zh|en)$")
    open_browser: bool = True
    download_source: str = Field(default="modelscope", pattern="^(modelscope|original)$")


def current():
    values = load_user_settings().model_dump()
    path = config_dir() / "interface.json"
    if path.is_file():
        values.update(json.loads(path.read_text(encoding="utf-8")))
    return Settings.model_validate(values)


def update(changes, project=None, revision=None):
    from .parameters import expand_changes, validate_changes

    changes = expand_changes(changes)
    validate_changes(changes, project=bool(project))
    changes = dict(changes)
    for backend_key, model_key, registry in (
        ("asr_backend", "asr_model", ASR_BACKENDS),
        ("tts_backend", "tts_model", TTS_BACKENDS),
    ):
        if backend_key in changes and changes[backend_key] in registry:
            spec = registry[changes[backend_key]]
            changes.setdefault(model_key, spec.default_model)
            if backend_key == "tts_backend":
                if (
                    spec.id in {"indextts2", "indextts2_5"}
                    and changes[model_key] not in spec.models
                ):
                    changes[model_key] = spec.default_model
                changes.setdefault("tts_voice", spec.default_voice)
    if "translation_provider" in changes and changes["translation_provider"] in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[changes["translation_provider"]]
        changes.setdefault("translation_model", preset["default_model"])
        changes.setdefault("translation_base_url", preset["base_url"])
    if project:
        active, _ = load_project(project)
        selected_reference = changes.get("tts_reference_sentence_id")
        if selected_reference and not any(
            sentence.id == selected_reference for sentence in active.sentences
        ):
            raise ProjectError(f"项目中找不到参考句：{selected_reference}")
        payload = active.settings.model_dump()
        payload.update(changes)
        configured = ProjectSettings.model_validate(payload)
        configured.validate_mix_dependencies()
        # The existing invalidation rules are retained; only project values enter this call.
        defaults = UserSettings.model_validate(configured.model_dump())
        defaults.translation_prompt_ja = active.settings.translation_prompt
        defaults.translation_prompt_en = active.settings.translation_prompt
        defaults.translation_prompt_zh = active.settings.translation_prompt
        if "translation_prompt" in changes:
            setattr(
                defaults,
                f"translation_prompt_{active.source_language}",
                changes["translation_prompt"],
            )
        with browser_revision_scope(project, revision):
            project_operations.apply_global_settings(project, defaults)
            if "tts_reference_sentence_id" in changes:
                from .project_audio import select_reference

                select_reference(project, changes["tts_reference_sentence_id"] or "")
        return {"saved": True}
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(directory / ".interface.lock"):
        values = current().model_dump()
        values.update(changes)
        configured = Settings.model_validate(values)
        save_user_settings(UserSettings.model_validate(configured.model_dump()))
        from ..storage import atomic_write_text

        preferences = {
            name: getattr(configured, name)
            for name in ("ui_language", "open_browser", "download_source")
        }
        atomic_write_text(directory / "interface.json", json.dumps(preferences))
    return configured.model_dump()


def reference_upload(path):
    return str(store_reference_audio(Path(path)))


def key_statuses():
    return {
        "translation": {provider: api_key_status(provider) for provider in PROVIDER_PRESETS},
        "services": {
            name: service_key_status(name, True)
            for name in (
                "asr:generic_asr_api",
                "tts:mimo_tts",
                "tts:minimax",
                "tts:fish_speech",
                "tts:generic_tts_api",
                "tts:indextts2_api",
                "tts:gpt_sovits",
                "tts:cosyvoice",
                "separation:replicate",
                "separation:http",
            )
        },
    }


def key(kind, name, value="", clear=False):
    if kind == "translation":
        if name not in PROVIDER_PRESETS:
            raise ValueError("Unknown translation provider")
        if clear:
            clear_api_key(name)
        else:
            save_api_key(name, value)
        return api_key_status(name)
    valid = key_statuses()["services"]
    if name not in valid:
        raise ValueError("Unknown service")
    if clear:
        clear_service_key(name)
    else:
        save_service_key(name, value)
    return service_key_status(name, True)
