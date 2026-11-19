"""First-run guide: what to install for this machine, from the existing hardware probe."""

from .. import runtime_manager as runtime
from ..constants import FALLBACK_ASR_BACKEND, RECOMMENDED_ASR_BACKEND
from ..model_registry import ASR_BACKENDS, TTS_BACKENDS
from ..user_settings import PROVIDER_PRESETS


def recommendation():
    profile = runtime.detect_hardware()
    asr = ASR_BACKENDS[RECOMMENDED_ASR_BACKEND]
    asr_note = runtime.compatibility_note(asr, profile)
    if not profile.cuda_available:
        # It can run on a CPU, but slowly; Parakeet is the sensible choice there.
        asr_note = "需要 NVIDIA GPU"
    elif not profile.full_cuda_stack_supported:
        asr_note = "需要更新架构的 NVIDIA GPU（计算能力 ≥ 7.5）"
    # The recommended recogniser may not fit this machine; Parakeet is the light fallback.
    incompatible = asr_note in {
        "不兼容",
        "需要 NVIDIA GPU",
        "需要更新架构的 NVIDIA GPU（计算能力 ≥ 7.5）",
    } or asr_note.startswith("显存不足")
    asr_id = FALLBACK_ASR_BACKEND if incompatible else RECOMMENDED_ASR_BACKEND
    tts = TTS_BACKENDS["indextts2"]
    tts_note = runtime.compatibility_note(tts, profile)
    if not profile.cuda_available:
        tts_note = "需要 NVIDIA GPU"  # local voice cloning is not practical on a CPU
    elif not profile.full_cuda_stack_supported:
        tts_note = "需要更新架构的 NVIDIA GPU（计算能力 ≥ 7.5）"
    local_tts = not (
        tts_note in {"不兼容", "需要 NVIDIA GPU", "需要更新架构的 NVIDIA GPU（计算能力 ≥ 7.5）"}
        or tts_note.startswith("显存不足")
    )
    text = runtime.recommended_stack_markdown(profile).splitlines()
    return {
        "title": text[0].removeprefix("### ").removeprefix("本机建议：") if text else "",
        "detail": "\n".join(text[1:]),
        "device": profile.recommended_device,
        "asr": {"id": asr_id, "note": asr_note if incompatible else ""},
        "tts": {"id": "indextts2" if local_tts else "", "note": "" if local_tts else tts_note},
    }


def required(saved_settings, keys=None, *, model_items=None) -> bool:
    """Only show the first-run wizard automatically on an unconfigured install."""
    if getattr(saved_settings, "onboarding_completed", False):
        return False
    if keys is None:
        from .settings import key_statuses

        keys = key_statuses()
    available = keys.get("translation_available") or {}
    if any(
        bool(available.get(provider))
        and not PROVIDER_PRESETS.get(provider, {}).get("key_optional", False)
        for provider in available
    ):
        return False
    for group_key in ("translation", "services"):
        if any(
            "以便携式明文保存" in str(status) for status in (keys.get(group_key) or {}).values()
        ):
            return False
    if model_items is None:
        from .models import catalog

        model_items = catalog()["items"]
    return not any(item.get("state") == "ready" for item in model_items)


LEGACY_UPGRADE_ASR_BACKENDS = frozenset({"faster_whisper", "parakeet_nemo", "kotoba_whisper"})


def _saved_explicit_asr_backend() -> str | None:
    import json

    from ..user_settings import config_dir

    path = config_dir() / "settings.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("asr_backend") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def asr_upgrade_prompt(
    saved_settings,
    *,
    onboarding_needed: bool = False,
    model_items=None,
    rec=None,
):
    """Prompt upgraded 2.x Japanese-ASR users once to switch to anime-whisper."""
    if onboarding_needed or getattr(saved_settings, "asr_upgrade_prompt_dismissed", False):
        return None
    if getattr(saved_settings, "default_source_language", "ja") != "ja":
        return None
    current_backend = getattr(saved_settings, "asr_backend", "")
    if current_backend not in LEGACY_UPGRADE_ASR_BACKENDS:
        return None
    explicit_saved = _saved_explicit_asr_backend()
    if explicit_saved not in LEGACY_UPGRADE_ASR_BACKENDS:
        # Ignore implicit fresh-install fallback unless the legacy backend is actually installed.
        if model_items is not None:
            legacy_installed = any(
                item.get("id") == current_backend and item.get("state") == "ready"
                for item in model_items
            )
        else:
            legacy_installed = (
                runtime.backend_status(ASR_BACKENDS[current_backend], settings=saved_settings).state
                == "ready"
            )
        if not legacy_installed:
            return None
    choice = rec if rec is not None else recommendation()
    if choice["asr"]["id"] != RECOMMENDED_ASR_BACKEND:
        return None
    recommended_spec = ASR_BACKENDS[RECOMMENDED_ASR_BACKEND]
    if model_items is not None:
        ready = any(
            item.get("id") == RECOMMENDED_ASR_BACKEND and item.get("state") == "ready"
            for item in model_items
        )
    else:
        ready = runtime.backend_status(recommended_spec, settings=saved_settings).state == "ready"
    current_spec = ASR_BACKENDS[current_backend]
    return {
        "current_backend": current_backend,
        "current_label": current_spec.label,
        "recommended_backend": RECOMMENDED_ASR_BACKEND,
        "recommended_label": recommended_spec.label,
        "recommended_ready": ready,
        "disk_gb": recommended_spec.disk_gb,
    }
