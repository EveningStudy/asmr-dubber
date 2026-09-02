from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .asr import _transcribe_generic_api
from .errors import ProjectError
from .languages import SpeechSourceLanguage
from .model_registry import TTS_BACKENDS
from .models import Sentence, load_project, settings_for_source_language
from .platforms import portable_home
from .translation import default_translation_prompt, translate_sentences
from .tts_backends import (
    _empty_reference,
    _require_reference_text,
    _runner,
    _uses_reference_audio,
    _validate_output,
)
from .user_settings import UserSettings, resolve_api_key
from .voice_reference import prepare_voice_reference

logger = logging.getLogger(__name__)


def _diagnostic_root() -> Path:
    root = portable_home() / "cache" / "api-tests"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_asr_api(settings: UserSettings, api_key: str = "") -> str:
    """Send a tiny real request to the configured generic ASR endpoint."""

    if settings.asr_backend != "generic_asr_api":
        raise ProjectError("当前选择的是本地 ASR（语音识别）后端，不需要测试 API。")
    language: SpeechSourceLanguage = settings.default_source_language
    project_settings = settings_for_source_language(
        settings.to_project_settings(source_language=language),
        language,
    )
    logger.info(
        "开始测试 ASR API：后端=%s 模型=%s",
        project_settings.asr_backend,
        project_settings.asr_model,
    )
    with tempfile.TemporaryDirectory(prefix="asr-", dir=_diagnostic_root()) as temporary:
        sample = Path(temporary) / "connection-test.wav"
        sf.write(sample, np.zeros(8_000, dtype=np.float32), 16_000, subtype="PCM_16")
        _transcribe_generic_api(
            sample,
            project_settings,
            None,
            source_language=language,
            api_key=api_key,
            probe_only=True,
        )
    logger.info("ASR API 测试通过：后端=%s", project_settings.asr_backend)
    return "ASR API 测试通过：服务接受了测试音频，并返回了有效 JSON。"


def test_translation_api(settings: UserSettings, api_key: str = "") -> str:
    """Translate one short sentence with the values currently shown in the form."""

    language: SpeechSourceLanguage = settings.default_source_language
    source_text = "Hello." if language == "en" else "こんにちは。"
    sentence = Sentence(
        id="api-test",
        start_seconds=0.0,
        end_seconds=1.0,
        source_text=source_text,
    )
    key = resolve_api_key(settings.translation_provider, api_key)
    logger.info(
        "开始测试翻译 API：服务=%s 模型=%s 源语言=%s",
        settings.translation_provider,
        settings.translation_model,
        language,
    )
    translate_sentences(
        [sentence],
        api_key=key,
        model=settings.translation_model,
        base_url=settings.translation_base_url,
        provider=settings.translation_provider,
        source_language=language,
        system_prompt=(
            settings.translation_prompt_for(language) or default_translation_prompt(language)
        ),
        temperature=settings.translation_temperature,
        top_p=settings.translation_top_p,
        max_output_tokens=settings.translation_max_output_tokens,
        deepl_formality=settings.translation_deepl_formality,
        microsoft_region=settings.translation_microsoft_region,
        send_context=False,
        context_sentences=0,
        memory_sentences=0,
        job_id="api-connection-test",
        extra_body=settings.translation_extra_body,
    )
    if not sentence.zh_text.strip():
        raise ProjectError("翻译服务响应成功，但没有返回中文文本。")
    logger.info("翻译 API 测试通过：服务=%s", settings.translation_provider)
    return f"翻译 API 测试通过：{source_text} → {sentence.zh_text}"


def test_tts_api(project_path: str, settings: UserSettings, api_key: str = "") -> str:
    """Generate and validate one short clip without modifying the project."""

    backend = settings.tts_backend
    spec = TTS_BACKENDS[backend]
    if spec.runtime != "http":
        raise ProjectError("当前选择的是本地 TTS（语音合成）后端，不需要测试 API。")
    normalized_manifest = str(project_path or "").strip()
    if not normalized_manifest:
        raise ProjectError("请先新建或打开项目，再测试 TTS 服务。")

    project, project_dir = load_project(normalized_manifest)
    project = project.model_copy(deep=True)
    project.settings = settings_for_source_language(
        settings.to_project_settings(project.settings, source_language=project.source_language),
        project.source_language,
    )
    candidates = [item for item in project.sentences if item.enabled]
    if not candidates:
        raise ProjectError("当前项目还没有可用于测试 TTS 的句子。")
    sentence = max(
        candidates,
        key=lambda item: (item.end_seconds - item.start_seconds, -item.start_seconds),
    )
    sentence.zh_text = "这是一次语音接口测试。"
    source = (project_dir / project.source.path).resolve()
    if not source.is_file():
        raise ProjectError(f"找不到当前项目的源音频：{source}")

    logger.info("开始测试 TTS API：后端=%s 模型=%s", backend, project.settings.tts_model)
    with tempfile.TemporaryDirectory(prefix="tts-", dir=_diagnostic_root()) as temporary:
        temporary_dir = Path(temporary)
        reference = (
            prepare_voice_reference(project, temporary_dir, source, sentence)
            if _uses_reference_audio(project)
            else _empty_reference()
        )
        _require_reference_text(project, reference)
        output = temporary_dir / "connection-test.wav"
        run, cleanup = _runner(project, api_key)
        try:
            run(sentence, reference, output)
            frames, sample_rate = _validate_output(output)
        finally:
            cleanup()
    duration = frames / sample_rate
    logger.info("TTS API 测试通过：后端=%s 时长=%.2f秒", backend, duration)
    return f"TTS 服务测试通过：已生成并验证 {duration:.2f} 秒音频。测试文件已清理。"
