from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .audio import probe_audio, sha256_file
from .errors import ProjectError
from .models import ProjectSettings
from .platforms import current_platform, portable_home, user_config_dir
from .storage import atomic_write_text, exclusive_file_lock

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
    "tts_index_external_emotion_audio",
)


class UserSettings(ProjectSettings):
    """Portable global defaults plus paths that do not belong to a project."""

    model_config = ConfigDict(extra="ignore")

    projects_root: str = ""
    huggingface_endpoint: str = ""
    pypi_index_url: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            if (
                "chinese_target_active_rms_dbfs" not in value
                and "chinese_max_active_rms_dbfs" in value
            ):
                value["chinese_target_active_rms_dbfs"] = value["chinese_max_active_rms_dbfs"]
            if "asr_backend" in value and value.get("asr_backend") not in {
                "parakeet_nemo",
                "kotoba_whisper",
                "faster_whisper",
            }:
                value["asr_backend"] = "parakeet_nemo"
                value["asr_model"] = "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
            if "tts_backend" in value and value.get("tts_backend") not in {
                "indextts2",
                "gpt_sovits",
                "cosyvoice",
                "fish_speech",
            }:
                value["tts_backend"] = "indextts2"
                value["tts_model"] = "IndexTTS2"
        return value

    def to_project_settings(self, base: ProjectSettings | None = None) -> ProjectSettings:
        values = base.model_dump() if base is not None else {}
        values.update(
            {
                name: getattr(self, name)
                for name in ProjectSettings.model_fields
                if base is None or name != "tts_reference_sentence_id"
            }
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
        with exclusive_file_lock(path.with_name(f".{path.name}.lock")):
            atomic_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                mode=0o600,
            )
    except OSError as exc:
        raise ProjectError(f"无法保存本地设置 {path}: {exc}") from exc


def _write_private_json_unlocked(path: Path, payload: dict[str, Any]) -> None:
    try:
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            mode=0o600,
        )
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


def _update_secrets(mutator: Any) -> Path:
    path = config_dir() / "secrets.json"
    with exclusive_file_lock(path.with_name(".secrets.json.lock")):
        try:
            secrets = _Secrets.model_validate(_read_json(path))
        except ValidationError as exc:
            raise ProjectError(f"本地密钥文件校验失败 {path}: {exc}") from exc
        mutator(secrets)
        _write_private_json_unlocked(path, secrets.model_dump())
    return path


def saved_api_key(provider: str) -> str:
    return _load_secrets().api_keys.get(provider, "").strip()


def save_api_key(provider: str, api_key: str) -> Path:
    provider = provider.strip()
    key = api_key.strip()
    if provider not in PROVIDER_PRESETS:
        raise ProjectError(f"未知翻译服务：{provider}")
    if not key:
        raise ProjectError("API Key 为空；如需删除，请使用“清除当前服务密钥”。")
    return _update_secrets(lambda secrets: secrets.api_keys.__setitem__(provider, key))


def clear_api_key(provider: str) -> None:
    _update_secrets(lambda secrets: secrets.api_keys.pop(provider, None))


def saved_service_key(service: str) -> str:
    return _load_secrets().service_keys.get(service.strip(), "").strip()


def save_service_key(service: str, api_key: str) -> Path:
    service_id = service.strip()
    key = api_key.strip()
    if not service_id or not key:
        raise ProjectError("服务名称和 API Key 均不能为空。")
    return _update_secrets(lambda secrets: secrets.service_keys.__setitem__(service_id, key))


def clear_service_key(service: str) -> None:
    service_id = service.strip()
    _update_secrets(lambda secrets: secrets.service_keys.pop(service_id, None))


def service_key_status(service: str, required: bool = False) -> str:
    if saved_service_key(service):
        return "API Key 已以便携式明文保存在程序目录中；删除程序文件夹时会一并删除。"
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
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
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
        return (
            f"已将 {preset.get('label', provider)} 的 API Key 以便携式明文保存在程序目录中；"
            "删除程序文件夹时会一并删除。"
        )
    if preset.get("key_optional", False):
        return "当前服务可不填写 API Key；如服务端要求认证，也可以保存。"
    return "当前服务尚未保存 API Key。"
