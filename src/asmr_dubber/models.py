from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ASR_REVIEW_PROMPT,
    DEFAULT_INDEXTTS_CONFIG,
    DEFAULT_INDEXTTS_MODEL_DIR,
    DEFAULT_TRANSLATION_MODEL,
    PROJECT_SCHEMA_VERSION,
    RECOMMENDED_ASR_BACKEND,
    RECOMMENDED_ASR_MODEL,
    RECOMMENDED_TTS_BACKEND,
    RECOMMENDED_TTS_MODEL,
)
from .errors import ProjectError


class AudioInfo(BaseModel):
    path: str
    sha256: str
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    channel_layout: str | None = None
    codec: str | None = None


class Sentence(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    ja_text: str
    zh_text: str = ""
    enabled: bool = True
    overlap_seconds: float | None = None
    reference_file: str | None = None
    tts_file: str | None = None
    tts_duration_seconds: float | None = None
    tts_cache_key: str | None = None
    status: str = "pending"
    error: str | None = None

    @field_validator("ja_text", "zh_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("overlap_seconds")
    @classmethod
    def finite_overlap(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            return None
        return value

    @model_validator(mode="after")
    def valid_range(self) -> Sentence:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("sentence end must be after start")
        return self

    def effective_overlap_seconds(
        self,
        global_overlap_seconds: float,
        global_overlap_percentage: float = 50.0,
    ) -> float:
        overlap = (
            self.overlap_seconds if self.overlap_seconds is not None else global_overlap_seconds
        )
        if overlap <= 0.0:
            # Zero still means sentence end; a negative value remains an
            # explicit post-sentence delay for manual timing adjustments.
            return overlap
        duration = self.end_seconds - self.start_seconds
        percentage_limit = duration * global_overlap_percentage / 100.0
        return min(overlap, percentage_limit)

    def chinese_start_seconds(
        self,
        global_overlap_seconds: float,
        global_overlap_percentage: float = 50.0,
    ) -> float:
        overlap = self.effective_overlap_seconds(
            global_overlap_seconds,
            global_overlap_percentage,
        )
        return max(0.0, self.end_seconds - overlap)


class ProjectSettings(BaseModel):
    asr_backend: str = RECOMMENDED_ASR_BACKEND
    asr_model: str = RECOMMENDED_ASR_MODEL
    aligner_model: str = DEFAULT_ALIGNER_MODEL
    # Keep quality-first inference sequential.  Some backends can use a larger
    # explicit batch for speed, but floating-point ordering can cause small
    # punctuation or segmentation differences.
    asr_batch_size: int = Field(default=1, ge=1, le=32)
    asr_max_new_tokens: int = Field(default=4096, ge=256, le=32768)
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_beam_size: int = Field(default=5, ge=1, le=100)
    asr_vad_filter: bool = False
    asr_vad_min_silence_ms: int = Field(default=500, ge=50, le=10_000)
    asr_condition_on_previous_text: bool = True
    asr_initial_prompt: str = ""
    asr_api_base_url: str = "http://127.0.0.1:8080/v1"
    asr_timeout_seconds: float = Field(default=600.0, ge=10.0, le=7200.0)
    asr_funasr_vad_model: str = "fsmn-vad"
    asr_funasr_punc_model: str = "ct-punc"
    asr_parakeet_decoder: Literal["tdt", "ctc"] = "tdt"
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
    pause_split_seconds: float = Field(default=0.55, ge=0.1, le=5.0)
    max_sentence_seconds: float = Field(default=15.0, ge=2.0, le=60.0)
    translation_provider: str = "deepseek"
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_base_url: str = ""
    translation_prompt: str = ""
    translation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    translation_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    translation_max_output_tokens: int = Field(default=16_384, ge=1_024, le=131_072)
    translation_deepl_formality: str = "default"
    translation_microsoft_region: str = ""
    # Used by local Transformers providers (currently Hunyuan Hy-MT2). Mirrors
    # asr_device/tts_device so a single runtime can be selected per project.
    translation_device: str = "cuda"
    tts_backend: str = RECOMMENDED_TTS_BACKEND
    tts_model: str = RECOMMENDED_TTS_MODEL
    tts_device: str = "cuda"
    tts_reference_source: Literal["project_sentence", "external"] = "project_sentence"
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
    # Retained so projects created before the split still load. New code derives
    # this value from tts_index_emotion_source.
    tts_index_use_emo_text: bool = False
    tts_index_emo_text: str = ""
    tts_gpt_top_k: int = Field(default=15, ge=1, le=100)
    tts_gpt_text_split_method: str = "cut5"
    tts_gpt_sample_steps: int = Field(default=32, ge=1, le=64)
    tts_cosyvoice_mode: Literal["zero_shot", "cross_lingual"] = "zero_shot"
    tts_f5_nfe_steps: int = Field(default=32, ge=4, le=128)
    tts_f5_cfg_strength: float = Field(default=2.0, ge=0.0, le=10.0)
    # Reusing one project-level reference is the default because per-sentence
    # ASMR clips can be extremely short or contain mostly breath/effects, which
    # makes the inferred age, pitch, and even gender drift between calls.
    tts_clone_mode: Literal[
        "stable_hifi",
        "stable_reference",
        "stable_voice_sentence_style",
        "reference_only",
        "ultimate",
    ] = "stable_reference"
    tts_reference_sentence_id: str | None = None
    tts_cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    # VoxCPM2 documents 4–30 as the recommended range and says more steps
    # improve detail/naturalness.  Use the maximum recommended value for the
    # quality-first workflow requested here.
    tts_inference_timesteps: int = Field(default=30, ge=1, le=100)
    tts_control_instruction: str = ""
    global_overlap_seconds: float = Field(default=5.0, ge=-30.0, le=30.0)
    global_overlap_percentage: float = Field(default=50.0, ge=0.0, le=100.0)
    chinese_gain_db: float = Field(default=0.0, ge=-40.0, le=20.0)
    normalize_chinese_loudness: bool = True
    match_source_loudness: bool = True
    chinese_relative_loudness_db: float = Field(default=-4.0, ge=-24.0, le=24.0)
    chinese_min_active_rms_dbfs: float = Field(default=-42.0, ge=-60.0, le=-20.0)
    chinese_target_active_rms_dbfs: float = Field(default=-30.0, ge=-50.0, le=-16.0)
    chinese_max_loudness_boost_db: float = Field(default=12.0, ge=0.0, le=30.0)
    chinese_line_peak_dbfs: float = Field(default=-9.0, ge=-20.0, le=-1.0)
    chinese_stem_peak_dbfs: float = Field(default=-3.0, ge=-12.0, le=-0.1)
    chinese_fade_ms: float = Field(default=8.0, ge=0.0, le=100.0)
    retain_chinese_stem: bool = False
    skip_japanese_fillers: bool = True
    reference_padding_seconds: float = Field(default=0.0, ge=0.0, le=2.0)
    random_seed: int = Field(default=20260722, ge=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_index_emotion(cls, value: Any) -> Any:
        if (
            isinstance(value, dict)
            and "tts_index_emotion_source" not in value
            and value.get("tts_index_use_emo_text") is True
        ):
            value = dict(value)
            value["tts_index_emotion_source"] = "text"
        return value

    @model_validator(mode="after")
    def valid_loudness_range(self) -> ProjectSettings:
        if self.chinese_min_active_rms_dbfs > self.chinese_target_active_rms_dbfs:
            raise ValueError("Chinese loudness floor must not exceed its ceiling")
        if self.asr_review_enabled and not self.asr_review_models:
            raise ValueError("ASR review requires at least one comparison model")
        return self


class DubProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PROJECT_SCHEMA_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: AudioInfo
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    sentences: list[Sentence] = Field(default_factory=list)
    asr_language: str | None = None
    chinese_stem_file: str | None = None
    output_file: str | None = None

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
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "project.json"
    return candidate


def load_project(path: str | os.PathLike[str]) -> tuple[DubProject, Path]:
    manifest = manifest_path(path)
    if not manifest.is_file():
        raise ProjectError(f"找不到项目文件：{manifest}")
    try:
        data: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        project = DubProject.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectError(f"无法读取项目文件 {manifest}: {exc}") from exc
    return project, manifest.parent


def save_project(project: DubProject, project_dir: str | os.PathLike[str]) -> Path:
    directory = Path(project_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    project.touch()
    destination = directory / "project.json"
    temporary = directory / ".project.json.tmp"
    payload = project.model_dump_json(indent=2, exclude_none=False)
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
