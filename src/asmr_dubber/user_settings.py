from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .audio import probe_audio, sha256_file
from .constants import (
    DEFAULT_ASR_REVIEW_PROMPT,
    DEFAULT_INDEXTTS_CONFIG,
    DEFAULT_INDEXTTS_MODEL_DIR,
    DEFAULT_TRANSLATION_MODEL,
    RECOMMENDED_ASR_BACKEND,
    RECOMMENDED_ASR_MODEL,
    RECOMMENDED_TTS_BACKEND,
    RECOMMENDED_TTS_MODEL,
)
from .errors import ProjectError
from .models import ProjectSettings
from .platforms import current_platform, portable_home, user_config_dir
from .translation import SYSTEM_PROMPT

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek（推荐默认）",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-pro",
        "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
        "env": "DEEPSEEK_API_KEY",
        "help": "OpenAI 兼容接口。默认 V4 Pro，保留完整上下文并强制逐句 JSON 输出。",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5.6-sol",
        "models": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4"],
        "env": "OPENAI_API_KEY",
        "help": (
            "使用 OpenAI Chat Completions。Sol 质量优先，Terra 平衡质量与成本，Luna 适合"
            "高吞吐；模型下拉框允许直接输入账号可用的其他模型 ID。"
        ),
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-fable-5",
        "models": [
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ],
        "env": "ANTHROPIC_API_KEY",
        "help": "使用 Claude Messages API。Fable/Opus 质量优先，Sonnet 更均衡，Haiku 更快。",
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-3.6-flash",
        "models": [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
        ],
        "env": "GEMINI_API_KEY",
        "help": "使用 Gemini generateContent 与 JSON 输出。可输入其他有效 Gemini 模型 ID。",
    },
    "openai_compatible": {
        "label": "本地/自定义 OpenAI-compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "local-model",
        "models": ["local-model", "qwen3:30b-a3b", "gpt-oss:20b"],
        "env": "OPENAI_COMPATIBLE_API_KEY",
        "help": "适用于 Ollama、LM Studio、vLLM 等。程序只调用接口，不会下载或启动模型。",
        "key_optional": True,
    },
    "deepl": {
        "label": "DeepL API",
        "base_url": "https://api.deepl.com",
        "default_model": "prefer_quality_optimized",
        "models": ["prefer_quality_optimized", "quality_optimized", "latency_optimized"],
        "env": "DEEPL_API_KEY",
        "help": "专业机器翻译 API。Free 用户把地址改为 https://api-free.deepl.com。",
    },
    "google_translate": {
        "label": "Google Cloud Translation",
        "base_url": "https://translation.googleapis.com/language/translate/v2",
        "default_model": "nmt",
        "models": ["nmt", "base"],
        "env": "GOOGLE_TRANSLATE_API_KEY",
        "help": "Cloud Translation Basic v2，支持 API Key；逐句翻译，不使用大模型 Prompt。",
    },
    "microsoft_translate": {
        "label": "Microsoft Azure Translator",
        "base_url": "https://api.cognitive.microsofttranslator.com",
        "default_model": "general",
        "models": ["general"],
        "env": "AZURE_TRANSLATOR_KEY",
        "help": "Azure Translator Text v3。区域资源还需要填写订阅区域。",
    },
}

_PORTABLE_PATH_TOKEN = "${ASMR_DUBBER_HOME}"
_PORTABLE_PATH_FIELDS = (
    "projects_root",
    "tts_model_path",
    "tts_config_path",
    "tts_external_reference_audio",
)


class UserSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    projects_root: str = ""
    huggingface_endpoint: str = ""
    pypi_index_url: str = ""
    asr_backend: str = RECOMMENDED_ASR_BACKEND
    asr_model: str = RECOMMENDED_ASR_MODEL
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_batch_size: int = Field(default=1, ge=1, le=32)
    asr_beam_size: int = Field(default=5, ge=1, le=100)
    asr_vad_filter: bool = False
    asr_vad_min_silence_ms: int = Field(default=500, ge=50, le=10_000)
    asr_condition_on_previous_text: bool = True
    asr_initial_prompt: str = ""
    asr_api_base_url: str = "http://127.0.0.1:8080/v1"
    asr_timeout_seconds: float = Field(default=600.0, ge=10.0, le=7200.0)
    asr_funasr_vad_model: str = "fsmn-vad"
    asr_funasr_punc_model: str = "ct-punc"
    asr_parakeet_decoder: str = "tdt"
    asr_chunk_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    asr_kotoba_chunk_seconds: float = Field(default=15.0, ge=5.0, le=30.0)
    asr_review_enabled: bool = False
    asr_review_models: list[str] = Field(
        default_factory=lambda: [
            ("parakeet_nemo|grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"),
            "faster_whisper|kotoba-tech/kotoba-whisper-v2.0-faster",
        ],
        max_length=6,
    )
    asr_review_background: str = ""
    asr_review_prompt: str = DEFAULT_ASR_REVIEW_PROMPT
    asr_review_max_drift_seconds: float = Field(default=1.5, ge=0.1, le=10.0)
    global_overlap_seconds: float = Field(default=5.0, ge=-30.0, le=30.0)
    global_overlap_percentage: float = Field(default=50.0, ge=0.0, le=100.0)
    chinese_gain_db: float = Field(default=0.0, ge=-40.0, le=20.0)
    tts_clone_mode: str = "stable_reference"
    tts_backend: str = RECOMMENDED_TTS_BACKEND
    tts_model: str = RECOMMENDED_TTS_MODEL
    tts_device: str = "cuda"
    tts_reference_source: str = "project_sentence"
    tts_external_reference_audio: str = ""
    tts_external_reference_text: str = ""
    tts_api_base_url: str = "http://127.0.0.1:9880"
    tts_timeout_seconds: float = Field(default=600.0, ge=10.0, le=7200.0)
    tts_model_path: str = str(DEFAULT_INDEXTTS_MODEL_DIR)
    tts_config_path: str = str(DEFAULT_INDEXTTS_CONFIG)
    tts_executable: str = "f5-tts_infer-cli"
    tts_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    tts_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    tts_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    tts_qwen_x_vector_only: bool = False
    tts_index_use_fp16: bool = True
    tts_index_emo_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    tts_index_use_emo_text: bool = False
    tts_index_emo_text: str = ""
    tts_gpt_top_k: int = Field(default=15, ge=1, le=100)
    tts_gpt_text_split_method: str = "cut5"
    tts_gpt_sample_steps: int = Field(default=32, ge=1, le=64)
    tts_cosyvoice_mode: str = "zero_shot"
    tts_f5_nfe_steps: int = Field(default=32, ge=4, le=128)
    tts_f5_cfg_strength: float = Field(default=2.0, ge=0.0, le=10.0)
    match_source_loudness: bool = True
    chinese_relative_loudness_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    chinese_min_active_rms_dbfs: float = Field(default=-42.0, ge=-60.0, le=-20.0)
    chinese_max_active_rms_dbfs: float = Field(default=-30.0, ge=-50.0, le=-16.0)
    retain_chinese_stem: bool = False
    tts_cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    tts_inference_timesteps: int = Field(default=30, ge=1, le=100)
    tts_control_instruction: str = ""
    translation_provider: str = "deepseek"
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_base_url: str = "https://api.deepseek.com"
    translation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    translation_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    translation_max_output_tokens: int = Field(default=16_384, ge=1_024, le=131_072)
    translation_prompt: str = SYSTEM_PROMPT
    translation_deepl_formality: str = "default"
    translation_microsoft_region: str = ""

    def to_project_settings(self, base: ProjectSettings | None = None) -> ProjectSettings:
        values = base.model_dump() if base is not None else {}
        values.update(
            asr_backend=self.asr_backend,
            asr_model=self.asr_model,
            aligner_model=self.aligner_model,
            asr_device=self.asr_device,
            asr_compute_type=self.asr_compute_type,
            asr_batch_size=self.asr_batch_size,
            asr_beam_size=self.asr_beam_size,
            asr_vad_filter=self.asr_vad_filter,
            asr_vad_min_silence_ms=self.asr_vad_min_silence_ms,
            asr_condition_on_previous_text=self.asr_condition_on_previous_text,
            asr_initial_prompt=self.asr_initial_prompt,
            asr_api_base_url=self.asr_api_base_url,
            asr_timeout_seconds=self.asr_timeout_seconds,
            asr_funasr_vad_model=self.asr_funasr_vad_model,
            asr_funasr_punc_model=self.asr_funasr_punc_model,
            asr_parakeet_decoder=self.asr_parakeet_decoder,
            asr_chunk_seconds=self.asr_chunk_seconds,
            asr_kotoba_chunk_seconds=self.asr_kotoba_chunk_seconds,
            asr_review_enabled=self.asr_review_enabled,
            asr_review_models=self.asr_review_models,
            asr_review_background=self.asr_review_background,
            asr_review_prompt=self.asr_review_prompt,
            asr_review_max_drift_seconds=self.asr_review_max_drift_seconds,
            global_overlap_seconds=self.global_overlap_seconds,
            global_overlap_percentage=self.global_overlap_percentage,
            chinese_gain_db=self.chinese_gain_db,
            tts_backend=self.tts_backend,
            tts_model=self.tts_model,
            tts_device=self.tts_device,
            tts_reference_source=self.tts_reference_source,
            tts_external_reference_audio=self.tts_external_reference_audio,
            tts_external_reference_text=self.tts_external_reference_text,
            tts_api_base_url=self.tts_api_base_url,
            tts_timeout_seconds=self.tts_timeout_seconds,
            tts_model_path=self.tts_model_path,
            tts_config_path=self.tts_config_path,
            tts_executable=self.tts_executable,
            tts_speed=self.tts_speed,
            tts_temperature=self.tts_temperature,
            tts_top_p=self.tts_top_p,
            tts_qwen_x_vector_only=self.tts_qwen_x_vector_only,
            tts_index_use_fp16=self.tts_index_use_fp16,
            tts_index_emo_alpha=self.tts_index_emo_alpha,
            tts_index_use_emo_text=self.tts_index_use_emo_text,
            tts_index_emo_text=self.tts_index_emo_text,
            tts_gpt_top_k=self.tts_gpt_top_k,
            tts_gpt_text_split_method=self.tts_gpt_text_split_method,
            tts_gpt_sample_steps=self.tts_gpt_sample_steps,
            tts_cosyvoice_mode=self.tts_cosyvoice_mode,
            tts_f5_nfe_steps=self.tts_f5_nfe_steps,
            tts_f5_cfg_strength=self.tts_f5_cfg_strength,
            tts_clone_mode=self.tts_clone_mode,
            match_source_loudness=self.match_source_loudness,
            chinese_relative_loudness_db=self.chinese_relative_loudness_db,
            chinese_min_active_rms_dbfs=self.chinese_min_active_rms_dbfs,
            chinese_target_active_rms_dbfs=self.chinese_max_active_rms_dbfs,
            retain_chinese_stem=self.retain_chinese_stem,
            tts_cfg_value=self.tts_cfg_value,
            tts_inference_timesteps=self.tts_inference_timesteps,
            tts_control_instruction=self.tts_control_instruction,
            translation_provider=self.translation_provider,
            translation_model=self.translation_model,
            translation_base_url=self.translation_base_url,
            translation_temperature=self.translation_temperature,
            translation_top_p=self.translation_top_p,
            translation_max_output_tokens=self.translation_max_output_tokens,
            translation_prompt=self.translation_prompt,
            translation_deepl_formality=self.translation_deepl_formality,
            translation_microsoft_region=self.translation_microsoft_region,
        )
        return ProjectSettings.model_validate(values)


class _Secrets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_keys: dict[str, str] = Field(default_factory=dict)
    service_keys: dict[str, str] = Field(default_factory=dict)


def config_dir() -> Path:
    return user_config_dir()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"无法读取本地设置 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProjectError(f"本地设置格式无效：{path}")
    home = portable_home()
    for field in _PORTABLE_PATH_FIELDS:
        raw = value.get(field)
        if isinstance(raw, str) and raw.startswith(_PORTABLE_PATH_TOKEN):
            relative = raw[len(_PORTABLE_PATH_TOKEN) :].lstrip("/\\")
            value[field] = str(home / Path(relative))
    return value


def _portable_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = dict(payload)
    home = portable_home()
    for field in _PORTABLE_PATH_FIELDS:
        raw = encoded.get(field)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            relative = Path(raw).expanduser().resolve().relative_to(home.resolve())
        except (OSError, ValueError):
            continue
        suffix = relative.as_posix()
        encoded[field] = (
            _PORTABLE_PATH_TOKEN if suffix == "." else f"{_PORTABLE_PATH_TOKEN}/{suffix}"
        )
    return encoded


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    windows = current_platform().is_windows
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        if not windows:
            path.parent.chmod(0o700)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not windows:
            temporary.chmod(0o600)
        temporary.replace(path)
        if not windows:
            path.chmod(0o600)
    except OSError as exc:
        raise ProjectError(f"无法保存本地设置 {path}: {exc}") from exc


def load_user_settings() -> UserSettings:
    path = config_dir() / "settings.json"
    try:
        settings = UserSettings.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ProjectError(f"本地设置校验失败 {path}: {exc}") from exc
    if settings.huggingface_endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface_endpoint
    if settings.pypi_index_url:
        os.environ["UV_DEFAULT_INDEX"] = settings.pypi_index_url
    return settings


def save_user_settings(settings: UserSettings) -> Path:
    path = config_dir() / "settings.json"
    _write_private_json(path, _portable_json_payload(settings.model_dump()))
    return path


def _load_secrets() -> _Secrets:
    path = config_dir() / "secrets.json"
    try:
        return _Secrets.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ProjectError(f"本地密钥文件校验失败 {path}: {exc}") from exc


def saved_api_key(provider: str) -> str:
    return _load_secrets().api_keys.get(provider, "").strip()


def save_api_key(provider: str, api_key: str) -> Path:
    provider = provider.strip()
    key = api_key.strip()
    if provider not in PROVIDER_PRESETS:
        raise ProjectError(f"未知翻译服务：{provider}")
    if not key:
        raise ProjectError("API Key 为空；如需删除，请使用“清除当前服务密钥”。")
    secrets = _load_secrets()
    secrets.api_keys[provider] = key
    path = config_dir() / "secrets.json"
    _write_private_json(path, secrets.model_dump())
    return path


def clear_api_key(provider: str) -> None:
    secrets = _load_secrets()
    secrets.api_keys.pop(provider, None)
    _write_private_json(config_dir() / "secrets.json", secrets.model_dump())


def saved_service_key(service: str) -> str:
    return _load_secrets().service_keys.get(service.strip(), "").strip()


def save_service_key(service: str, api_key: str) -> Path:
    service_id = service.strip()
    key = api_key.strip()
    if not service_id or not key:
        raise ProjectError("服务名称和 API Key 均不能为空。")
    secrets = _load_secrets()
    secrets.service_keys[service_id] = key
    path = config_dir() / "secrets.json"
    _write_private_json(path, secrets.model_dump())
    return path


def clear_service_key(service: str) -> None:
    secrets = _load_secrets()
    secrets.service_keys.pop(service.strip(), None)
    _write_private_json(config_dir() / "secrets.json", secrets.model_dump())


def service_key_status(service: str, required: bool = False) -> str:
    if saved_service_key(service):
        return "已在本地配置中保存该服务的 API Key。"
    return "当前服务尚未保存 API Key。" if required else "当前本地服务通常不需要 API Key。"


def store_reference_audio(source: str | os.PathLike[str]) -> Path:
    """Persist a Gradio upload below the current user's private config directory."""
    candidate = Path(source).expanduser().resolve()
    if not candidate.is_file():
        raise ProjectError(f"找不到外部参考音频：{candidate}")
    try:
        probe_audio(candidate)
    except Exception as exc:
        raise ProjectError(f"外部参考音频无法读取：{candidate}: {exc}") from exc
    suffix = candidate.suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".wav"
    reference_dir = config_dir() / "references"
    reference_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not current_platform().is_windows:
        reference_dir.chmod(0o700)
    destination = reference_dir / f"{sha256_file(candidate)}{suffix}"
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copy2(candidate, temporary)
            if not current_platform().is_windows:
                temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if not current_platform().is_windows:
        destination.chmod(0o600)
    return destination.resolve()


def resolve_api_key(provider: str, supplied: str | None = None) -> str:
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    key = (supplied or "").strip() or saved_api_key(provider)
    env_name = str(preset.get("env", ""))
    key = key or os.getenv(env_name, "").strip()
    if not key and not preset.get("key_optional", False):
        raise ProjectError(f"{preset['label']} 尚未保存 API Key；请前往设置 → 翻译服务。")
    return key


def api_key_status(provider: str) -> str:
    preset = PROVIDER_PRESETS.get(provider, {})
    key = saved_api_key(provider)
    if key:
        protection = "（文件权限 600）" if not current_platform().is_windows else ""
        return f"已在本地配置中保存 {preset.get('label', provider)} 的 API Key{protection}。"
    if preset.get("key_optional", False):
        return "当前服务可不填写 API Key；如服务端要求认证，也可以保存。"
    return "当前服务尚未保存 API Key。"
