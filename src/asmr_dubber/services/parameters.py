"""One schema drives project forms, global defaults and JSON validation."""

import json
from importlib.resources import files

from ..model_registry import ASR_BACKENDS, TTS_BACKENDS
from ..models import ProjectSettings
from ..runtime_manager import asmr_vad_status, ordered_backends
from ..translation import default_translation_prompt
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
            value["options"] = [[spec.label, spec.id] for spec in ordered_backends(registry)]
        elif name == "translation_provider":
            value["options"] = [[item["label"], key] for key, item in PROVIDER_PRESETS.items()]
        elif name == "asr_review_models":
            value["options"] = [
                [f"{spec.label} · {model}", f"{spec.id}|{model}"]
                for spec in ASR_BACKENDS.values()
                if spec.id != "generic_asr_api"
                for model in spec.models
            ]
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


def visible(parameter, values):
    def matches(condition):
        return all(values.get(key) in allowed for key, allowed in condition.items())

    return matches(parameter["visible_when"]) and (
        not parameter.get("visible_any") or any(matches(item) for item in parameter["visible_any"])
    )


def expand_changes(changes):
    by_key = {item["key"]: item for item in catalog()}
    expanded = dict(changes)
    for key, value in changes.items():
        mapping = by_key.get(key, {}).get("display_values", {})
        if isinstance(value, str) and mapping:
            if value not in mapping:
                raise ValueError("Unknown display option")
            expanded.update(mapping[value])
    return expanded


def display_value(parameter, values):
    for label, fields in parameter.get(
        "display_conditions", parameter.get("display_values", {})
    ).items():
        if all(values.get(key) == value for key, value in fields.items()):
            return label
    return values[parameter["key"]]


def choices():
    return {
        "translation_prompts": {
            language: default_translation_prompt(language) for language in ("ja", "en", "zh")
        },
        "transcript_timing": {
            "source": ["estimate", "qwen", "script_review"],
            "zh": ["estimate", "script_review"],
        },
        "asmr_vad_ready": asmr_vad_status().state == "ready",
        "asr": {
            key: {
                "models": list(spec.models),
                "default_model": spec.default_model,
                "models_by_language": {
                    language: [
                        model
                        for model in spec.models
                        if language == "ja"
                        or (
                            not model.startswith("kotoba-tech/")
                            and not (
                                language == "zh"
                                and (model.endswith(".en") or "distil" in model.lower())
                            )
                        )
                    ]
                    for language in ("ja", "en", "zh")
                },
            }
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
