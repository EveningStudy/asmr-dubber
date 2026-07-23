from asmr_dubber.model_registry import ASR_BACKENDS, TTS_BACKENDS
from asmr_dubber.models import ProjectSettings
from asmr_dubber.runtime_manager import installable_backend_ids


def test_default_backends_use_verified_recommendations() -> None:
    settings = ProjectSettings()
    assert settings.asr_backend == "parakeet_nemo"
    assert settings.asr_model == ASR_BACKENDS["parakeet_nemo"].default_model
    assert settings.tts_backend == "indextts2"
    assert settings.tts_model == TTS_BACKENDS["indextts2"].default_model
    assert ASR_BACKENDS["parakeet_nemo"].tested_default is True
    assert TTS_BACKENDS["indextts2"].support_level == "verified"


def test_common_asr_and_tts_backends_are_registered() -> None:
    assert {
        "parakeet_nemo",
        "kotoba_whisper",
        "qwen3_asr",
        "faster_whisper",
        "openai_whisper",
        "whisperx",
        "funasr",
        "openai_compatible_asr",
    } <= ASR_BACKENDS.keys()
    assert ASR_BACKENDS["parakeet_nemo"].models[0] == (
        "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
    )
    assert "large-v2" in ASR_BACKENDS["faster_whisper"].models
    assert "kotoba-tech/kotoba-whisper-v2.0-faster" in (ASR_BACKENDS["faster_whisper"].models)
    assert {
        "voxcpm2",
        "qwen3_tts",
        "indextts2",
        "gpt_sovits",
        "cosyvoice",
        "f5_tts",
        "fish_speech",
        "xtts_v2",
    } <= TTS_BACKENDS.keys()
    assert TTS_BACKENDS["indextts2"].reference_text == "unused"
    assert TTS_BACKENDS["qwen3_tts"].reference_text == "required"
    assert "stable_hifi" in TTS_BACKENDS["voxcpm2"].clone_modes
    assert "stable_hifi" not in TTS_BACKENDS["indextts2"].clone_modes


def test_installable_backends_come_from_registry_capabilities() -> None:
    declared = {
        item.id
        for item in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values())
        if item.installer is not None
    }
    assert set(installable_backend_ids()) == declared
    assert all(
        item.installer != "python-extra" or item.python_extra
        for item in (*ASR_BACKENDS.values(), *TTS_BACKENDS.values())
    )


def test_backend_registry_declares_execution_capabilities() -> None:
    qwen_asr = ASR_BACKENDS["qwen3_asr"]
    assert qwen_asr.execution.batch_strategy == "native_list"
    assert qwen_asr.execution.quality_sensitive_batch is True
    assert "细微结果差异" in qwen_asr.execution_label

    qwen_tts = TTS_BACKENDS["qwen3_tts"]
    assert qwen_tts.execution.reusable_reference_conditioning is True
    assert qwen_tts.execution.persistent_session is True

    index = TTS_BACKENDS["indextts2"]
    assert index.execution.progress_strategy == "streamed_process"
