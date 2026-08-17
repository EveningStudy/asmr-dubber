from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ASR_REVIEW_MODELS,
    DEFAULT_ASR_REVIEW_PROMPT,
    DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
    DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY,
    DEFAULT_CHINESE_DUBBING_OFFSET_MS,
    DEFAULT_CHINESE_MAX_AUTO_SPEED,
    DEFAULT_CHINESE_RELATIVE_LOUDNESS_DB,
    DEFAULT_INDEXTTS_CONFIG,
    DEFAULT_INDEXTTS_EMOTION_WEIGHT,
    DEFAULT_INDEXTTS_MODEL_DIR,
    DEFAULT_TRANSLATION_MODEL,
    MAX_CHINESE_AUTO_SPEED,
    PROJECT_SCHEMA_VERSION,
    RECOMMENDED_ASR_BACKEND,
    RECOMMENDED_ASR_MODEL,
    RECOMMENDED_TTS_BACKEND,
    RECOMMENDED_TTS_MODEL,
)
from .errors import ProjectConflictError, ProjectError
from .languages import SourceLanguage
from .storage import atomic_write_text, exclusive_file_lock


class AudioInfo(BaseModel):
    path: str
    sha256: str
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    channel_layout: str | None = None
    codec: str | None = None
    media_type: Literal["audio", "video"] = "audio"
    video_codec: str | None = None
    video_width: int | None = Field(default=None, gt=0)
    video_height: int | None = Field(default=None, gt=0)
    video_frame_rate: float | None = Field(default=None, gt=0)


class Sentence(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    source_text: str = Field(validation_alias=AliasChoices("source_text", "ja_text"))
    zh_text: str = ""
    enabled: bool = True
    reference_file: str | None = None
    tts_file: str | None = None
    tts_duration_seconds: float | None = None
    tts_cache_key: str | None = None
    status: str = "pending"
    error: str | None = None

    @field_validator("source_text", "zh_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @property
    def ja_text(self) -> str:
        """Compatibility alias for callers and projects created before schema v3."""

        return self.source_text

    @ja_text.setter
    def ja_text(self, value: str) -> None:
        self.source_text = value.strip()

    @model_validator(mode="after")
    def valid_range(self) -> Sentence:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("sentence end must be after start")
        return self


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    asr_backend: Literal["parakeet_nemo", "kotoba_whisper", "faster_whisper"] = (
        RECOMMENDED_ASR_BACKEND
    )
    asr_model: str = RECOMMENDED_ASR_MODEL
    asr_batch_size: int = Field(default=1, ge=1, le=32)
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_beam_size: int = Field(default=5, ge=1, le=100)
    asr_vad_filter: bool = False
    asr_vad_mode: Literal["off", "backend", "asmr"] = "off"
    asr_vad_min_silence_ms: int = Field(default=500, ge=50, le=10_000)
    asr_asmr_vad_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    asr_asmr_vad_min_speech_ms: int = Field(default=250, ge=20, le=10_000)
    asr_asmr_vad_min_silence_ms: int = Field(default=100, ge=20, le=10_000)
    asr_asmr_vad_speech_pad_ms: int = Field(default=200, ge=0, le=5_000)
    aligner_model: str = DEFAULT_ALIGNER_MODEL
    asr_forced_alignment_enabled: bool = False
    asr_condition_on_previous_text: bool = True
    asr_initial_prompt: str = ""
    asr_timeout_seconds: float = Field(default=600.0, ge=10.0, le=7200.0)
    asr_parakeet_decoder: Literal["tdt", "ctc"] = "tdt"
    asr_chunk_seconds: float = Field(default=120.0, ge=15.0, le=600.0)
    asr_kotoba_chunk_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    asr_review_enabled: bool = False
    asr_review_models: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ASR_REVIEW_MODELS),
        max_length=6,
    )
    asr_review_text_priority_model: str = DEFAULT_ASR_REVIEW_TEXT_PRIORITY
    asr_review_timestamp_priority_model: str = DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY
    asr_review_background: str = ""
    asr_review_prompt: str = DEFAULT_ASR_REVIEW_PROMPT
    asr_review_max_drift_seconds: float = Field(default=1.5, ge=0.1, le=10.0)
    pause_split_seconds: float = Field(default=0.55, ge=0.1, le=5.0)
    max_sentence_seconds: float = Field(default=15.0, ge=2.0, le=60.0)
    translation_provider: str = "deepseek"
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_base_url: str = ""
    translation_prompt: str = ""
    translation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    translation_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    translation_max_output_tokens: int = Field(default=16_384, ge=1_024, le=131_072)
    translation_send_context: bool = True
    translation_context_sentences: int = Field(default=24, ge=0, le=200)
    translation_memory_sentences: int = Field(default=50, ge=0, le=500)
    translation_deepl_formality: str = "default"
    translation_microsoft_region: str = ""
    tts_backend: Literal["indextts2", "gpt_sovits", "cosyvoice", "fish_speech"] = (
        RECOMMENDED_TTS_BACKEND
    )
    tts_model: str = RECOMMENDED_TTS_MODEL
    tts_device: str = "cuda"
    tts_reference_source: Literal["project_sentence", "external"] = "project_sentence"
    tts_external_reference_audio: str = ""
    tts_external_reference_text: str = ""
    tts_external_reference_language: Literal["auto", "ja", "en", "zh"] = "auto"
    tts_api_base_url: str = "http://127.0.0.1:9880"
    tts_timeout_seconds: float = Field(default=600.0, ge=10.0, le=7200.0)
    tts_request_concurrency: int = Field(default=2, ge=1, le=8)
    tts_model_path: str = str(DEFAULT_INDEXTTS_MODEL_DIR)
    tts_config_path: str = str(DEFAULT_INDEXTTS_CONFIG)
    tts_executable: str = ""
    tts_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    tts_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    tts_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    tts_index_use_fp16: bool = True
    tts_index_emo_alpha: float = Field(
        default=DEFAULT_INDEXTTS_EMOTION_WEIGHT,
        ge=0.0,
        le=1.0,
    )
    tts_index_speaker_source: Literal[
        "project_reference",
        "sentence_reference",
        "external",
    ] = "project_reference"
    tts_index_emotion_source: Literal[
        "sentence_reference",
        "project_reference",
        "speaker_reference",
        "external",
        "text",
    ] = "sentence_reference"
    tts_index_external_emotion_audio: str = ""
    tts_index_emo_text: str = ""
    tts_gpt_top_k: int = Field(default=15, ge=1, le=100)
    tts_gpt_text_split_method: str = "cut5"
    tts_gpt_sample_steps: int = Field(default=32, ge=1, le=64)
    tts_cosyvoice_mode: Literal["zero_shot", "cross_lingual"] = "zero_shot"
    tts_clone_mode: Literal["stable_reference", "reference_only"] = "stable_reference"
    tts_reference_sentence_id: str | None = None

    chinese_dubbing_offset_ms: int = Field(
        default=DEFAULT_CHINESE_DUBBING_OFFSET_MS,
        ge=-30_000,
        le=30_000,
    )
    chinese_max_auto_speed: float = Field(
        default=DEFAULT_CHINESE_MAX_AUTO_SPEED,
        ge=1.0,
        le=MAX_CHINESE_AUTO_SPEED,
    )
    chinese_dubbing_timing_mode: Literal["fit_window", "sequential"] = "fit_window"
    chinese_gain_db: float = Field(default=0.0, ge=-40.0, le=20.0)
    normalize_chinese_loudness: bool = True
    match_source_loudness: bool = True
    chinese_relative_loudness_db: float = Field(
        default=DEFAULT_CHINESE_RELATIVE_LOUDNESS_DB,
        ge=-24.0,
        le=24.0,
    )
    chinese_min_active_rms_dbfs: float = Field(default=-42.0, ge=-60.0, le=-20.0)
    chinese_target_active_rms_dbfs: float = Field(default=-30.0, ge=-50.0, le=-16.0)
    chinese_max_loudness_boost_db: float = Field(default=12.0, ge=0.0, le=30.0)
    chinese_line_peak_dbfs: float = Field(default=-9.0, ge=-20.0, le=-1.0)
    chinese_stem_peak_dbfs: float = Field(default=-3.0, ge=-12.0, le=-0.1)
    chinese_fade_ms: float = Field(default=8.0, ge=0.0, le=100.0)
    chinese_channel_routing: Literal["auto", "all"] = "auto"
    mix_peak_protection: bool = True
    mix_peak_limit_dbfs: float = Field(default=-1.0, ge=-6.0, le=-0.1)
    mix_output_mode: Literal["mixed", "stem", "both"] = "both"
    skip_japanese_fillers: bool = True
    reference_padding_seconds: float = Field(default=0.0, ge=0.0, le=2.0)
    random_seed: int = Field(default=20260722, ge=0)
    subtitle_timeline: Literal["source", "dubbing"] = "source"
    subtitle_max_chars_per_line: int = Field(default=22, ge=8, le=60)
    subtitle_min_duration_seconds: float = Field(default=1.0, ge=0.2, le=10.0)
    subtitle_max_cps: float = Field(default=18.0, ge=5.0, le=40.0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_index_emotion(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            supported_asr = {"parakeet_nemo", "kotoba_whisper", "faster_whisper"}
            supported_tts = {"indextts2", "gpt_sovits", "cosyvoice", "fish_speech"}
            if "asr_backend" in value and value.get("asr_backend") not in supported_asr:
                value["asr_backend"] = RECOMMENDED_ASR_BACKEND
                value["asr_model"] = RECOMMENDED_ASR_MODEL
            if "tts_backend" in value and value.get("tts_backend") not in supported_tts:
                value["tts_backend"] = RECOMMENDED_TTS_BACKEND
                value["tts_model"] = RECOMMENDED_TTS_MODEL
            review_models = [
                str(item)
                for item in (value.get("asr_review_models") or [])
                if str(item).partition("|")[0] in supported_asr
            ]
            if "asr_review_models" in value:
                value["asr_review_models"] = review_models
                if value.get("asr_review_enabled") and not review_models:
                    value["asr_review_enabled"] = False
            priority_defaults = {
                "asr_review_text_priority_model": DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
                "asr_review_timestamp_priority_model": (DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY),
            }
            for field, default in priority_defaults.items():
                configured = str(value.get(field, ""))
                if (
                    field in value
                    and configured.partition("|")[0] not in supported_asr
                    and not configured.startswith("qwen_forced_aligner|")
                ):
                    value[field] = default
            if (
                "tts_index_emotion_source" not in value
                and value.get("tts_index_use_emo_text") is True
            ):
                value["tts_index_emotion_source"] = "text"
            if "asr_vad_mode" not in value:
                value["asr_vad_mode"] = "backend" if value.get("asr_vad_filter") else "off"
            if value.get("asr_vad_mode") not in {"off", "backend", "asmr"}:
                value["asr_vad_mode"] = "off"
            # Older releases used 0 to mean automatic Parakeet chunking. The
            # current runner creates bounded inputs for one model process and
            # needs an explicit, validated 120-second default.
            try:
                legacy_chunk_seconds = float(value.get("asr_chunk_seconds", 120.0))
            except (TypeError, ValueError):
                legacy_chunk_seconds = 120.0
            if legacy_chunk_seconds <= 0:
                value["asr_chunk_seconds"] = 120.0
            if value.get("tts_clone_mode") not in {"stable_reference", "reference_only"}:
                value["tts_clone_mode"] = "stable_reference"
            if "mix_output_mode" not in value:
                value["mix_output_mode"] = (
                    "both" if value.get("retain_chinese_stem", True) else "mixed"
                )
            value.pop("retain_chinese_stem", None)
        return value

    @model_validator(mode="after")
    def valid_loudness_range(self) -> ProjectSettings:
        self.asr_vad_filter = self.asr_vad_mode == "backend"
        if (
            self.normalize_chinese_loudness
            and self.match_source_loudness
            and self.chinese_min_active_rms_dbfs > self.chinese_target_active_rms_dbfs
        ):
            raise ValueError("Chinese loudness floor must not exceed its ceiling")
        if self.asr_review_enabled and not self.asr_review_models:
            raise ValueError("ASR review requires at least one comparison model")
        return self


def settings_for_source_language(
    settings: ProjectSettings,
    source_language: SourceLanguage,
) -> ProjectSettings:
    """Return settings that cannot dispatch a source to a language-incompatible ASR."""

    if source_language != "en":
        return settings
    values = settings.model_dump()
    model = str(values.get("asr_model", ""))
    if values.get("asr_backend") != "faster_whisper" or model.startswith("kotoba-tech/"):
        values["asr_backend"] = "faster_whisper"
        values["asr_model"] = "large-v2"
    primary = f"{values['asr_backend']}|{values['asr_model']}"
    review_models = [
        str(item)
        for item in values.get("asr_review_models") or []
        if str(item).startswith("faster_whisper|") and "kotoba-tech/kotoba-whisper" not in str(item)
    ]
    values["asr_review_models"] = review_models
    if not any(item != primary for item in review_models):
        values["asr_review_enabled"] = False
    text_priority = str(values.get("asr_review_text_priority_model", ""))
    if not text_priority.startswith("faster_whisper|") or "kotoba-tech/" in text_priority:
        values["asr_review_text_priority_model"] = primary
    if values.get("asr_vad_mode") == "asmr":
        values["asr_vad_mode"] = "off"
    review_prompt = str(values.get("asr_review_prompt", ""))
    if review_prompt.startswith("你是日语语音识别校对专家"):
        values["asr_review_prompt"] = DEFAULT_ASR_REVIEW_PROMPT
    return ProjectSettings.model_validate(values)


class DubProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PROJECT_SCHEMA_VERSION
    app_version: str = "1.0.1"
    revision: int = Field(default=0, ge=0)
    migration_warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: AudioInfo
    source_language: SourceLanguage = "ja"
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    sentences: list[Sentence] = Field(default_factory=list)
    asr_language: str | None = None
    asr_settings_dirty: bool = False
    chinese_stem_file: str | None = None
    output_file: str | None = None
    output_video_file: str | None = None
    subtitle_language: Literal["bilingual", "zh", "source"] = "bilingual"
    subtitle_srt_file: str | None = None
    subtitle_lrc_file: str | None = None
    subtitle_video_file: str | None = None

    @field_validator("schema_version")
    @classmethod
    def supported_schema(cls, value: int) -> int:
        if value != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported project schema {value}; expected {PROJECT_SCHEMA_VERSION}"
            )
        return value

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC).isoformat()


def manifest_path(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    if isinstance(raw, str):
        value = raw.strip(" \t\r\n\ufeff\u200b")
        quote_pairs = {
            '"': '"',
            "'": "'",
            "`": "`",
            "“": "”",
            "‘": "’",
            "「": "」",
            "『": "』",
        }
        for _ in range(4):
            closing = quote_pairs.get(value[:1])
            if closing is None or not value.endswith(closing) or len(value) < 2:
                break
            value = value[1:-1].strip(" \t\r\n\ufeff\u200b")
        if not value:
            raise ProjectError("项目路径不能为空。")
        raw = value
    candidate = Path(raw).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "project.json"
    return candidate


def load_project(path: str | os.PathLike[str]) -> tuple[DubProject, Path]:
    manifest = manifest_path(path)
    if not manifest.is_file():
        raise ProjectError(f"找不到项目文件：{manifest}")
    try:
        data: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        data = _migrate_project_payload(data)
        project = DubProject.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectError(f"无法读取项目文件 {manifest}: {exc}") from exc
    return project, manifest.parent


def save_project(project: DubProject, project_dir: str | os.PathLike[str]) -> Path:
    directory = Path(project_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "project.json"
    lock_path = directory / ".project.lock"
    with exclusive_file_lock(lock_path):
        current_revision = 0
        current_payload: dict[str, Any] | None = None
        if destination.is_file():
            try:
                loaded = json.loads(destination.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("project manifest root must be an object")
                current_payload = loaded
                current_revision = int(current_payload.get("revision", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ProjectError(f"现有项目文件损坏，拒绝覆盖：{destination}") from exc
        if current_revision != project.revision:
            raise ProjectConflictError(
                "项目已被另一个窗口或命令修改。请重新加载项目后再保存；"
                f"当前版本 {project.revision}，磁盘版本 {current_revision}。"
            )
        if (
            current_payload is not None
            and int(current_payload.get("schema_version", 1)) < PROJECT_SCHEMA_VERSION
        ):
            backup_dir = directory / "backups"
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            old_schema = int(current_payload.get("schema_version", 1))
            backup = backup_dir / f"project-schema-v{old_schema}-{stamp}.json"
            if not backup.exists():
                atomic_write_text(
                    backup,
                    json.dumps(current_payload, ensure_ascii=False, indent=2) + "\n",
                )
        previous_revision = project.revision
        previous_updated_at = project.updated_at
        project.touch()
        project.revision += 1
        payload = project.model_dump_json(indent=2, exclude_none=False) + "\n"
        try:
            atomic_write_text(destination, payload)
        except Exception:
            project.revision = previous_revision
            project.updated_at = previous_updated_at
            raise
    return destination


def _migrate_project_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate historical project manifests without modifying them on load."""

    payload = dict(data)
    version = int(payload.get("schema_version", 1))
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"project schema {version} is newer than supported {PROJECT_SCHEMA_VERSION}"
        )
    warnings = list(payload.get("migration_warnings") or [])
    if version == 1:
        settings = dict(payload.get("settings") or {})
        supported_asr = {"parakeet_nemo", "kotoba_whisper", "faster_whisper"}
        supported_tts = {"indextts2", "gpt_sovits", "cosyvoice", "fish_speech"}
        if settings.get("asr_backend") not in supported_asr:
            old = settings.get("asr_backend", "未知")
            settings["asr_backend"] = RECOMMENDED_ASR_BACKEND
            settings["asr_model"] = RECOMMENDED_ASR_MODEL
            warnings.append(f"旧语音识别后端 {old} 已停止支持，已切换为 Parakeet。")
        if settings.get("tts_backend") not in supported_tts:
            old = settings.get("tts_backend", "未知")
            settings["tts_backend"] = RECOMMENDED_TTS_BACKEND
            settings["tts_model"] = RECOMMENDED_TTS_MODEL
            warnings.append(f"旧语音合成后端 {old} 已停止支持，已切换为 IndexTTS2。")
        if settings.get("asr_vad_mode") not in {"off", "backend", "asmr"}:
            settings["asr_vad_mode"] = "off"
            warnings.append("未知 VAD（语音活动检测）模式已关闭。")
        review_models = [
            item
            for item in settings.get("asr_review_models", [])
            if str(item).partition("|")[0] in supported_asr
        ]
        settings["asr_review_models"] = review_models or [
            DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
            "faster_whisper|large-v2",
        ]
        for field in ("asr_review_text_priority_model", "asr_review_timestamp_priority_model"):
            value = str(settings.get(field, ""))
            if value.partition("|")[0] not in supported_asr and not value.startswith(
                "qwen_forced_aligner|"
            ):
                settings[field] = DEFAULT_ASR_REVIEW_TEXT_PRIORITY
        payload["settings"] = settings
        payload["schema_version"] = 2
        payload["app_version"] = "1.0.1"
        payload["revision"] = int(payload.get("revision", 0))
        payload["migration_warnings"] = warnings
        version = 2
    if version == 2:
        sentences = []
        for raw_sentence in payload.get("sentences") or []:
            sentence = dict(raw_sentence)
            if "source_text" not in sentence:
                sentence["source_text"] = sentence.pop("ja_text", "")
            else:
                sentence.pop("ja_text", None)
            sentences.append(sentence)
        settings = dict(payload.get("settings") or {})
        if "mix_output_mode" not in settings:
            settings["mix_output_mode"] = (
                "both" if settings.get("retain_chinese_stem", True) else "mixed"
            )
        settings.pop("retain_chinese_stem", None)
        source_language: SourceLanguage = "ja"
        if (
            sentences
            and all(not str(item.get("source_text", "")).strip() for item in sentences)
            and any(str(item.get("zh_text", "")).strip() for item in sentences)
        ):
            source_language = "zh"
        payload["sentences"] = sentences
        payload["settings"] = settings
        payload["source_language"] = source_language
        if payload.get("subtitle_language") == "ja":
            payload["subtitle_language"] = "source"
        payload["schema_version"] = 3
        payload["revision"] = int(payload.get("revision", 0))
        payload["migration_warnings"] = warnings
        version = 3
    if version != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"unsupported project schema {version}")
    return payload
