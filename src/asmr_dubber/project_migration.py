"""Centralized project manifest and settings migration (v1 / v2 / v3 -> v4)."""

from __future__ import annotations

import json
from typing import Any

from .constants import (
    DEFAULT_ASR_REVIEW_PROMPT,
    DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
    DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY,
    FALLBACK_ASR_BACKEND,
    FALLBACK_ASR_MODEL,
    PROJECT_SCHEMA_VERSION,
    RECOMMENDED_ASR_MODEL,
    RECOMMENDED_TTS_BACKEND,
    RECOMMENDED_TTS_MODEL,
)
from .languages import SourceLanguage
from .settings_groups import flatten_grouped_settings

VALID_SENTENCE_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "review_accepted",
        "translated",
        "translation_failed",
        "synthesized",
        "skipped_filler",
        "skipped_hallucination",
        "skipped_sound",
        "skipped_tts",
        "error",
    }
)

PRE_2_3_SETTINGS: dict[str, Any] = {
    "reference_audio_mode": "legacy",
    "tts_normalize_pauses": False,
    "tts_fit_source_duration": False,
    "tts_dual_voice_anchor": False,
    "tts_spoken_normalization": False,
}


def is_legacy_asr_review_prompt(value: str) -> bool:
    prompt = str(value or "").strip()
    return (
        prompt.startswith("你是语音识别校对专家。你会收到按时间窗口组织的多个 ASR 候选。")
        or prompt.startswith("你是日语语音识别校对专家。你会收到按时间窗口组织的多个 ASR 候选。")
        or ('"evidence_ids"' in prompt and ('"ja"' in prompt or "校对后的日文" in prompt))
    )


def default_sentence_error_code(status: str) -> str:
    if status == "translation_failed":
        return "TRANSLATION_FAILED"
    if status == "error":
        return "SYNTHESIS_FAILED"
    return "SENTENCE_ERROR"


def normalize_project_settings_dict(value: Any) -> Any:
    """Normalize legacy and grouped ``ProjectSettings`` input dictionaries."""

    if not isinstance(value, dict):
        return value
    value = flatten_grouped_settings(value)
    supported_asr = {
        "moss_transcribe_diarize",
        "generic_asr_api",
        "parakeet_nemo",
        "kotoba_whisper",
        "faster_whisper",
        "qwen3_asr",
        "anime_whisper",
    }
    supported_tts = {
        "indextts2_5",
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
    if "asr_backend" in value and value.get("asr_backend") not in supported_asr:
        value["asr_backend"] = FALLBACK_ASR_BACKEND
        value["asr_model"] = FALLBACK_ASR_MODEL
    if value.get("asr_backend") == "moss_transcribe_diarize":
        from . import moss

        cur_asr_model = str(value.get("asr_model") or "").strip()
        other_defaults = {
            RECOMMENDED_ASR_MODEL,
            FALLBACK_ASR_MODEL,
            "Qwen/Qwen3-ASR-1.7B",
            "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
            "kotoba-tech/kotoba-whisper-v2.2",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "",
        }
        if cur_asr_model in other_defaults:
            value["asr_model"] = moss.MODEL_ID
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
    legacy_review_prompt = str(value.get("asr_review_prompt", "") or "").strip()
    if is_legacy_asr_review_prompt(legacy_review_prompt):
        value["asr_review_prompt"] = DEFAULT_ASR_REVIEW_PROMPT
    priority_defaults = {
        "asr_review_text_priority_model": DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
        "asr_review_timestamp_priority_model": DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY,
    }
    for field, default in priority_defaults.items():
        configured = str(value.get(field, ""))
        if (
            field in value
            and configured.partition("|")[0] not in supported_asr
            and not configured.startswith("qwen_forced_aligner|")
        ):
            value[field] = default
    if "tts_index_emotion_source" not in value and value.get("tts_index_use_emo_text") is True:
        value["tts_index_emotion_source"] = "text"
    if "asr_vad_mode" not in value:
        value["asr_vad_mode"] = "backend" if value.get("asr_vad_filter") else "off"
    if value.get("asr_vad_mode") not in {"off", "backend", "asmr"}:
        value["asr_vad_mode"] = "off"
    try:
        legacy_chunk_seconds = float(value.get("asr_chunk_seconds", 120.0))
    except (TypeError, ValueError):
        legacy_chunk_seconds = 120.0
    if legacy_chunk_seconds <= 0:
        value["asr_chunk_seconds"] = 120.0
    if value.get("tts_clone_mode") not in {"stable_reference", "reference_only"}:
        value["tts_clone_mode"] = "stable_reference"
    if "mix_output_mode" not in value:
        value["mix_output_mode"] = "both" if value.get("retain_chinese_stem", True) else "mixed"
    value.pop("retain_chinese_stem", None)
    return value


def migrate_project_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate historical project manifests (v1, v2, v3) to schema v4 without modifying disk."""

    payload = dict(data)
    version = int(payload.get("schema_version", 1))
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"project schema {version} is newer than supported {PROJECT_SCHEMA_VERSION}"
        )
    warnings = list(payload.get("migration_warnings") or [])
    if version == 1:
        settings = dict(payload.get("settings") or {})
        supported_asr = {
            "moss_transcribe_diarize",
            "generic_asr_api",
            "parakeet_nemo",
            "kotoba_whisper",
            "faster_whisper",
        }
        supported_tts = {
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
        if settings.get("asr_backend") not in supported_asr:
            old = settings.get("asr_backend", "未知")
            settings["asr_backend"] = FALLBACK_ASR_BACKEND
            settings["asr_model"] = FALLBACK_ASR_MODEL
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
            val = str(settings.get(field, ""))
            if val.partition("|")[0] not in supported_asr and not val.startswith(
                "qwen_forced_aligner|"
            ):
                settings[field] = DEFAULT_ASR_REVIEW_TEXT_PRIORITY
        payload["settings"] = settings
        payload["schema_version"] = 2
        payload["app_version"] = "2.2.1"
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
            and any(
                str(item.get("target_text", item.get("zh_text", ""))).strip() for item in sentences
            )
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
    if version == 3:
        sentences = []
        for raw_sentence in payload.get("sentences") or []:
            sentence = dict(raw_sentence)
            if "source_text" not in sentence and "ja_text" in sentence:
                sentence["source_text"] = sentence.pop("ja_text", "")
            else:
                sentence.pop("ja_text", None)
            if "target_text" not in sentence:
                sentence["target_text"] = sentence.pop("zh_text", "")
            else:
                sentence.pop("zh_text", None)
            raw_status = str(sentence.get("status") or "pending").strip()
            if raw_status not in VALID_SENTENCE_STATUSES:
                raw_status = "error" if sentence.get("error") else "pending"
            sentence["status"] = raw_status
            err = sentence.get("error")
            if err and not sentence.get("error_code"):
                sentence["error_code"] = default_sentence_error_code(raw_status)
            elif not err:
                sentence["error_code"] = None
            sentences.append(sentence)
        payload["sentences"] = sentences
        payload["schema_version"] = 4
        payload["revision"] = int(payload.get("revision", 0))
        payload["migration_warnings"] = warnings
        version = 4
    if version != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"unsupported project schema {version}")
    settings = payload.get("settings")
    if isinstance(settings, dict):
        updated_settings = flatten_grouped_settings(settings)
        prompt = str(updated_settings.get("translation_prompt", "") or "").strip()
        if prompt:
            from .user_settings import _is_legacy_builtin_translation_prompt

            if _is_legacy_builtin_translation_prompt(prompt):
                updated_settings["translation_prompt"] = ""
        legacy = {
            key: value for key, value in PRE_2_3_SETTINGS.items() if key not in updated_settings
        }
        if legacy or updated_settings != settings:
            payload["settings"] = {**updated_settings, **legacy}
    return payload


def validate_separation_json_object(value: str, field_name: str) -> str:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("分离参数必须为 JSON 对象。")
    allowed = {
        "separation_common_params": {
            "invert_using_spec",
            "use_torch_compile",
            "use_native_fp16",
            "amplification_threshold",
        },
        "separation_mdx_params": {
            "hop_length",
            "segment_size",
            "overlap",
            "batch_size",
            "enable_denoise",
        },
        "separation_vr_params": {
            "batch_size",
            "window_size",
            "aggression",
            "enable_tta",
            "enable_post_process",
            "post_process_threshold",
            "high_end_process",
        },
        "separation_demucs_params": {"segment_size", "shifts", "overlap", "segments_enabled"},
        "separation_mdxc_params": {
            "segment_size",
            "override_model_segment_size",
            "batch_size",
            "overlap",
            "pitch_shift",
        },
    }.get(field_name)
    if allowed is not None and parsed.keys() - allowed:
        raise ValueError("不支持的分离参数：" + ", ".join(sorted(parsed.keys() - allowed)))
    if allowed is not None:
        for key, parameter in parsed.items():
            if (
                key in {"batch_size", "window_size", "hop_length", "segment_size"}
                and parameter != "Default"
            ) and (isinstance(parameter, bool) or not isinstance(parameter, int) or parameter < 1):
                raise ValueError(f"{key} 必须是正整数。")
            if key == "overlap":
                minimum, maximum = (1, 32) if field_name == "separation_mdxc_params" else (0, 0.99)
                if not isinstance(parameter, (int, float)) or not minimum <= parameter <= maximum:
                    raise ValueError(f"overlap 必须在 {minimum} 到 {maximum} 之间。")
    return json.dumps(parsed, ensure_ascii=False)


def validate_moss_generation_json(value: str) -> str:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("MOSS generation parameters must be a JSON object")
    protected = {
        "input_ids",
        "inputs",
        "attention_mask",
        "input_features",
        "audio_feature_lengths",
        "audio_chunk_mapping",
        "streamer",
        "generation_config",
        "return_dict_in_generate",
        "max_length",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
        "repetition_penalty",
        "use_cache",
        "num_return_sequences",
    }
    if protected.intersection(parsed):
        raise ValueError("Use the dedicated MOSS controls for reserved generation parameters")
    return value


def adapt_settings_values_for_source_language(
    values: dict[str, Any], source_language: SourceLanguage
) -> dict[str, Any]:
    out = dict(values)
    model = str(out.get("asr_model", ""))
    if (
        out.get("asr_backend")
        not in {"faster_whisper", "generic_asr_api", "moss_transcribe_diarize", "qwen3_asr"}
        or model.startswith("kotoba-tech/")
        or (source_language == "zh" and (model.endswith(".en") or "distil" in model.lower()))
    ):
        out["asr_backend"] = "faster_whisper"
        out["asr_model"] = "large-v2"
    primary = f"{out['asr_backend']}|{out['asr_model']}"
    review_models = [
        str(item)
        for item in out.get("asr_review_models") or []
        if str(item).startswith(("faster_whisper|", "moss_transcribe_diarize|"))
        and "kotoba-tech/kotoba-whisper" not in str(item)
    ]
    if source_language == "zh":
        review_models = [
            item
            for item in review_models
            if not item.endswith(".en") and "distil" not in item.lower()
        ]
    out["asr_review_models"] = review_models
    if not any(item != primary for item in review_models):
        out["asr_review_enabled"] = False
    text_priority = str(out.get("asr_review_text_priority_model", ""))
    if not text_priority.startswith("faster_whisper|") or "kotoba-tech/" in text_priority:
        out["asr_review_text_priority_model"] = primary
    if out.get("asr_vad_mode") == "asmr":
        out["asr_vad_mode"] = "off"
    review_prompt = str(out.get("asr_review_prompt", ""))
    if is_legacy_asr_review_prompt(review_prompt):
        out["asr_review_prompt"] = DEFAULT_ASR_REVIEW_PROMPT
    return out
