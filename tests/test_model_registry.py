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


def test_only_supported_asr_and_tts_backends_are_registered() -> None:
    assert set(ASR_BACKENDS) == {
        "generic_asr_api",
        "parakeet_nemo",
        "kotoba_whisper",
        "faster_whisper",
    }
    assert ASR_BACKENDS["parakeet_nemo"].models[0] == (
        "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
    )
    assert "large-v2" in ASR_BACKENDS["faster_whisper"].models
    assert "kotoba-tech/kotoba-whisper-v2.0-faster" in (ASR_BACKENDS["faster_whisper"].models)
    assert set(TTS_BACKENDS) == {
        "indextts2",
        "indextts2_api",
        "generic_tts_api",
        "gpt_sovits",
        "cosyvoice",
        "fish_speech",
        "edge_tts",
        "mimo_tts",
        "minimax",
    }
    assert TTS_BACKENDS["indextts2"].reference_text == "unused"
    assert TTS_BACKENDS["gpt_sovits"].reference_text == "required"
    assert TTS_BACKENDS["fish_speech"].api_key is True
    assert TTS_BACKENDS["edge_tts"].api_key is False
    assert TTS_BACKENDS["mimo_tts"].default_model == "mimo-v2.5-tts-voiceclone"
    assert TTS_BACKENDS["minimax"].default_voice == "female-shaonv"
    assert TTS_BACKENDS["indextts2_api"].reference_audio is True
    assert ASR_BACKENDS["generic_asr_api"].runtime == "http"


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


def test_project_settings_preserve_supported_generic_api_backends() -> None:
    settings = ProjectSettings.model_validate(
        {
            "asr_backend": "generic_asr_api",
            "asr_model": "remote-asr",
            "tts_backend": "indextts2_api",
            "tts_model": "IndexTTS2",
        }
    )

    assert settings.asr_backend == "generic_asr_api"
    assert settings.asr_model == "remote-asr"
    assert settings.tts_backend == "indextts2_api"


def test_backend_registry_declares_execution_capabilities() -> None:
    faster = ASR_BACKENDS["faster_whisper"]
    assert faster.execution.batch_strategy == "internal_chunks"
    assert faster.execution.quality_sensitive_batch is True
    assert "细微结果差异" in faster.execution_label

    external = TTS_BACKENDS["gpt_sovits"]
    assert external.execution.reusable_reference_conditioning is True
    assert external.execution.batch_strategy == "request_concurrency"

    index = TTS_BACKENDS["indextts2"]
    assert index.execution.progress_strategy == "streamed_process"
