"""Structured domain submodels for ``ProjectSettings`` (W9 / H4).

Provides typed domain views (``asr``, ``translation``, ``tts``, ``mix``, ``separation``,
``subtitles``) and a unified ``MixVariantSettings`` for bilingual vs. full-dub replacement,
while preserving flat-attribute and ``parameter_layout.json`` compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class MixVariantSettings(BaseModel):
    """Unified loudness, timing, and voice-matching configuration for one mix mode."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["bilingual", "replace"] = "bilingual"
    dubbing_offset_ms: int = 500
    max_auto_speed: float = 1.4
    timing_mode: Literal["fit_window", "sequential", "fit_then_shift"] = "fit_window"
    gain_db: float = 0.0
    normalize_loudness: bool = True
    match_source_loudness: bool = True
    relative_loudness_db: float = -8.0
    min_active_rms_dbfs: float = -42.0
    target_active_rms_dbfs: float = -30.0
    max_loudness_boost_db: float = 12.0
    line_peak_dbfs: float = -9.0
    stem_peak_dbfs: float = -3.0
    fade_ms: float = 8.0
    scope: Literal["dubbed_speech", "all_vocals"] = "dubbed_speech"
    voice_match_strength: float = 0.0
    bandwidth_extension: bool = False
    noise_fill: bool = False
    residual_suppression: float = 0.0
    residual_asr_check: bool = False
    whisper_match: float = 0.0


class MixSettingsGroup(BaseModel):
    """Structured mix configuration with bilingual and replacement variants."""

    model_config = ConfigDict(frozen=True)

    bilingual: MixVariantSettings
    replacement: MixVariantSettings
    chinese_channel_routing: Literal["auto", "all"] = "auto"
    mix_peak_protection: bool = True
    mix_peak_limit_dbfs: float = -1.0
    mix_output_mode: Literal["mixed", "stem", "both"] = "both"
    spatial_rtf_enabled: bool = False
    spatial_rtf_strength: float = 1.0
    spatial_rtf_level_strength: float = 0.7
    spatial_rtf_color_strength: float = 0.35
    spatial_rtf_fft_size: int = 0
    spatial_rtf_hop_divisor: int = 8
    spatial_rtf_block_seconds: float = 10.0
    spatial_rtf_vocal_weighting: bool = True

    def variant(self, replacing: bool = False) -> MixVariantSettings:
        return self.replacement if replacing else self.bilingual


class AsrSettingsGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str
    model: str
    device: str
    compute_type: str
    batch_size: int
    beam_size: int
    vad_mode: Literal["off", "backend", "asmr"]
    forced_alignment_enabled: bool
    aligner_model: str
    diarize_speakers: bool
    speaker_count: int
    review_enabled: bool
    review_mode: Literal["suggest", "conservative"]
    review_models: list[str]
    pause_split_seconds: float
    max_sentence_seconds: float
    skip_japanese_fillers: bool


class TranslationSettingsGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    base_url: str
    prompt: str
    target_language: Literal["zh", "en"]
    temperature: float
    top_p: float
    max_output_tokens: int
    send_context: bool
    context_sentences: int
    memory_sentences: int
    sound_policy: Literal["skip", "romaji", "keep_source"]
    length_budget: bool
    chars_per_second: float


class TtsSettingsGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str
    model: str
    device: str
    target_language: Literal["zh", "en"]
    clone_mode: Literal["stable_reference", "reference_only"]
    reference_source: Literal["project_sentence", "external"]
    reference_audio_mode: Literal["legacy", "fullband", "fullband_vocals"]
    reference_sentence_id: str | None
    whisper_reference_sentence_id: str | None
    dual_voice_anchor: bool
    fit_source_duration: bool
    normalize_pauses: bool
    spoken_normalization: bool
    pronunciation_glossary: str
    quality_check: bool
    asr_quality_check: bool
    speed: float
    random_seed: int


class SeparationSettingsGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    backend: Literal["local", "replicate", "http"]
    model: str
    device: Literal["cuda", "cpu"]
    mix_mode: Literal["bilingual", "replace"]
    keep_original: Literal["none", "unvoiced", "manual"]
    keep_ids: str
    keep_padding_ms: float
    chunk_seconds: float
    overlap_seconds: float


class SubtitleSettingsGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeline: Literal["source", "dubbing"]
    max_chars_per_line: int
    min_duration_seconds: float
    max_cps: float


_BILINGUAL_VARIANT_MAP = {
    "dubbing_offset_ms": "chinese_dubbing_offset_ms",
    "max_auto_speed": "chinese_max_auto_speed",
    "timing_mode": "chinese_dubbing_timing_mode",
    "gain_db": "chinese_gain_db",
    "normalize_loudness": "normalize_chinese_loudness",
    "match_source_loudness": "match_source_loudness",
    "relative_loudness_db": "chinese_relative_loudness_db",
    "min_active_rms_dbfs": "chinese_min_active_rms_dbfs",
    "target_active_rms_dbfs": "chinese_target_active_rms_dbfs",
    "max_loudness_boost_db": "chinese_max_loudness_boost_db",
    "line_peak_dbfs": "chinese_line_peak_dbfs",
    "stem_peak_dbfs": "chinese_stem_peak_dbfs",
    "fade_ms": "chinese_fade_ms",
}

_REPLACEMENT_VARIANT_MAP = {
    "dubbing_offset_ms": "replacement_dubbing_offset_ms",
    "max_auto_speed": "replacement_max_auto_speed",
    "timing_mode": "replacement_timing_mode",
    "gain_db": "replacement_chinese_gain_db",
    "normalize_loudness": "replacement_normalize_chinese_loudness",
    "match_source_loudness": "replacement_match_source_loudness",
    "relative_loudness_db": "replacement_chinese_relative_loudness_db",
    "min_active_rms_dbfs": "replacement_chinese_min_active_rms_dbfs",
    "target_active_rms_dbfs": "replacement_chinese_target_active_rms_dbfs",
    "max_loudness_boost_db": "replacement_chinese_max_loudness_boost_db",
    "line_peak_dbfs": "replacement_chinese_line_peak_dbfs",
    "stem_peak_dbfs": "replacement_chinese_stem_peak_dbfs",
    "fade_ms": "replacement_chinese_fade_ms",
    "scope": "replacement_scope",
    "voice_match_strength": "replacement_voice_match_strength",
    "bandwidth_extension": "replacement_bandwidth_extension",
    "noise_fill": "replacement_noise_fill",
    "residual_suppression": "replacement_residual_suppression",
    "residual_asr_check": "replacement_residual_asr_check",
    "whisper_match": "replacement_whisper_match",
}


def flatten_grouped_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Expand nested group dicts (``asr``, ``translation``, ``tts``, ``mix``) into flat keys."""

    out = dict(raw)
    prefix_groups = {
        "asr": "asr_",
        "translation": "translation_",
        "tts": "tts_",
        "separation": "separation_",
        "subtitles": "subtitle_",
    }
    direct_fields = {
        "aligner_model",
        "pause_split_seconds",
        "max_sentence_seconds",
        "skip_japanese_fillers",
        "reference_audio_mode",
        "reference_padding_seconds",
        "random_seed",
        "tts_target_language",
    }
    for group_key, prefix in prefix_groups.items():
        nested = out.pop(group_key, None)
        if not isinstance(nested, Mapping):
            continue
        for k, v in nested.items():
            target_key = (
                k
                if (k.startswith(prefix) or k in direct_fields)
                else (
                    "tts_target_language"
                    if group_key == "translation" and k == "target_language"
                    else f"{prefix}{k}"
                )
            )
            out.setdefault(target_key, v)
    mix_nested = out.pop("mix", None)
    if isinstance(mix_nested, Mapping):
        for variant_key, mapping in (
            ("bilingual", _BILINGUAL_VARIANT_MAP),
            ("replacement", _REPLACEMENT_VARIANT_MAP),
        ):
            vdict = mix_nested.get(variant_key)
            if isinstance(vdict, Mapping):
                for k, v in vdict.items():
                    if k in mapping:
                        out.setdefault(mapping[k], v)
        for k, v in mix_nested.items():
            if k not in {"bilingual", "replacement"}:
                out.setdefault(k, v)
    return out


def build_asr_group(s: Any) -> AsrSettingsGroup:
    return AsrSettingsGroup(
        backend=s.asr_backend,
        model=s.asr_model,
        device=s.asr_device,
        compute_type=s.asr_compute_type,
        batch_size=s.asr_batch_size,
        beam_size=s.asr_beam_size,
        vad_mode=s.asr_vad_mode,
        forced_alignment_enabled=s.asr_forced_alignment_enabled,
        aligner_model=s.aligner_model,
        diarize_speakers=s.asr_diarize_speakers,
        speaker_count=s.asr_speaker_count,
        review_enabled=s.asr_review_enabled,
        review_mode=s.asr_review_mode,
        review_models=list(s.asr_review_models),
        pause_split_seconds=s.pause_split_seconds,
        max_sentence_seconds=s.max_sentence_seconds,
        skip_japanese_fillers=s.skip_japanese_fillers,
    )


def build_translation_group(s: Any) -> TranslationSettingsGroup:
    return TranslationSettingsGroup(
        provider=s.translation_provider,
        model=s.translation_model,
        base_url=s.translation_base_url,
        prompt=s.translation_prompt,
        target_language=s.tts_target_language,
        temperature=s.translation_temperature,
        top_p=s.translation_top_p,
        max_output_tokens=s.translation_max_output_tokens,
        send_context=s.translation_send_context,
        context_sentences=s.translation_context_sentences,
        memory_sentences=s.translation_memory_sentences,
        sound_policy=s.translation_sound_policy,
        length_budget=s.translation_length_budget,
        chars_per_second=s.translation_chars_per_second,
    )


def build_tts_group(s: Any) -> TtsSettingsGroup:
    return TtsSettingsGroup(
        backend=s.tts_backend,
        model=s.tts_model,
        device=s.tts_device,
        target_language=s.tts_target_language,
        clone_mode=s.tts_clone_mode,
        reference_source=s.tts_reference_source,
        reference_audio_mode=s.reference_audio_mode,
        reference_sentence_id=s.tts_reference_sentence_id,
        whisper_reference_sentence_id=s.tts_whisper_reference_sentence_id,
        dual_voice_anchor=s.tts_dual_voice_anchor,
        fit_source_duration=s.tts_fit_source_duration,
        normalize_pauses=s.tts_normalize_pauses,
        spoken_normalization=s.tts_spoken_normalization,
        pronunciation_glossary=s.tts_pronunciation_glossary,
        quality_check=s.tts_quality_check,
        asr_quality_check=s.tts_asr_quality_check,
        speed=s.tts_speed,
        random_seed=s.random_seed,
    )


def build_mix_group(s: Any) -> MixSettingsGroup:
    bilingual = MixVariantSettings(
        mode="bilingual",
        **{k: getattr(s, field) for k, field in _BILINGUAL_VARIANT_MAP.items()},
    )
    replacement = MixVariantSettings(
        mode="replace",
        **{k: getattr(s, field) for k, field in _REPLACEMENT_VARIANT_MAP.items()},
    )
    return MixSettingsGroup(
        bilingual=bilingual,
        replacement=replacement,
        chinese_channel_routing=s.chinese_channel_routing,
        mix_peak_protection=s.mix_peak_protection,
        mix_peak_limit_dbfs=s.mix_peak_limit_dbfs,
        mix_output_mode=s.mix_output_mode,
        spatial_rtf_enabled=s.spatial_rtf_enabled,
        spatial_rtf_strength=s.spatial_rtf_strength,
        spatial_rtf_level_strength=s.spatial_rtf_level_strength,
        spatial_rtf_color_strength=s.spatial_rtf_color_strength,
        spatial_rtf_fft_size=s.spatial_rtf_fft_size,
        spatial_rtf_hop_divisor=s.spatial_rtf_hop_divisor,
        spatial_rtf_block_seconds=s.spatial_rtf_block_seconds,
        spatial_rtf_vocal_weighting=s.spatial_rtf_vocal_weighting,
    )


def build_separation_group(s: Any) -> SeparationSettingsGroup:
    return SeparationSettingsGroup(
        enabled=s.separation_enabled,
        backend=s.separation_backend,
        model=s.separation_model,
        device=s.separation_device,
        mix_mode=s.separation_mix_mode,
        keep_original=s.separation_keep_original,
        keep_ids=s.separation_keep_ids,
        keep_padding_ms=s.separation_keep_padding_ms,
        chunk_seconds=s.separation_chunk_seconds,
        overlap_seconds=s.separation_overlap_seconds,
    )


def build_subtitles_group(s: Any) -> SubtitleSettingsGroup:
    return SubtitleSettingsGroup(
        timeline=s.subtitle_timeline,
        max_chars_per_line=s.subtitle_max_chars_per_line,
        min_duration_seconds=s.subtitle_min_duration_seconds,
        max_cps=s.subtitle_max_cps,
    )
