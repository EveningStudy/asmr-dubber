"""First-run guide: what to install for this machine, from the existing hardware probe."""

from .. import runtime_manager as runtime
from ..constants import FALLBACK_ASR_BACKEND, RECOMMENDED_ASR_BACKEND
from ..model_registry import ASR_BACKENDS, TTS_BACKENDS


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
