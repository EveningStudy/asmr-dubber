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
    # The recommended recogniser may not fit this machine; Parakeet is the light fallback.
    incompatible = asr_note in {"不兼容", "需要 NVIDIA GPU"} or asr_note.startswith("显存不足")
    asr_id = FALLBACK_ASR_BACKEND if incompatible else RECOMMENDED_ASR_BACKEND
    tts = TTS_BACKENDS["indextts2"]
    tts_note = runtime.compatibility_note(tts, profile)
    if not profile.cuda_available:
        tts_note = "需要 NVIDIA GPU"  # local voice cloning is not practical on a CPU
    local_tts = not (tts_note in {"不兼容", "需要 NVIDIA GPU"} or tts_note.startswith("显存不足"))
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
            "已以便携式明文保存" in str(status) for status in (keys.get(group_key) or {}).values()
        ):
            return False
    if model_items is None:
        from .models import catalog

        model_items = catalog()["items"]
    return not any(item.get("state") == "ready" for item in model_items)
