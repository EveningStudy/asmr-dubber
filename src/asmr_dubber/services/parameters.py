"""One schema drives project forms, global defaults and JSON validation."""

import json
from importlib.resources import files

from ..model_registry import ASR_BACKENDS, TTS_BACKENDS
from ..models import ProjectSettings
from ..user_settings import PROVIDER_PRESETS
from .settings import Settings


def catalog():
    layout = json.loads(
        files(__package__).joinpath("parameter_layout.json").read_text(encoding="utf-8")
    )
    schema = Settings.model_json_schema()["properties"]
    defaults = Settings().model_dump(mode="json")
    result = []
    for name, specification in schema.items():
        value = dict(specification)
        if "anyOf" in value:
            value.update(next(item for item in value["anyOf"] if item.get("type") != "null"))
            value["nullable"] = True
        value.update(layout[name])
        value.update(
            key=name,
            default=defaults[name],
            scope="project" if name in ProjectSettings.model_fields else "global",
        )
        if name in {"asr_backend", "tts_backend"}:
            registry = ASR_BACKENDS if name == "asr_backend" else TTS_BACKENDS
            value["options"] = [[spec.label, spec.id] for spec in registry.values()]
        elif name == "translation_provider":
            value["options"] = [[item["label"], key] for key, item in PROVIDER_PRESETS.items()]
        elif "enum" in value and "options" not in value:
            value["options"] = [[str(option), option] for option in value["enum"]]
        result.append(value)
    return result


def validate_changes(changes, *, project=False):
    allowed = ProjectSettings.model_fields if project else Settings.model_fields
    unknown = changes.keys() - allowed.keys()
    if unknown:
        raise ValueError("Unknown parameters: " + ", ".join(sorted(unknown)))
    # Full model validation runs after merging with the correct persisted scope.
    for key, value in changes.items():
        if key.endswith(("extra_body", "params")):
            parsed = json.loads(value or "{}")
            if not isinstance(parsed, dict):
                raise ValueError(f"{key} must contain a JSON object")


def choices():
    return {
        "asr": {
            key: {"models": list(spec.models), "default_model": spec.default_model}
            for key, spec in ASR_BACKENDS.items()
        },
        "tts": {
            key: {
                "models": list(spec.models),
                "default_model": spec.default_model,
                "voices": list(spec.voices),
                "default_voice": spec.default_voice,
                "reference": spec.reference_audio,
            }
            for key, spec in TTS_BACKENDS.items()
        },
        "translation": PROVIDER_PRESETS,
    }
