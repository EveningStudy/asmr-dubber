"""Parameter presets: two independent groups, each a set of conservative setting changes.

A preset only touches settings that need no extra installation.  Its fixed
changes never switch the recognition or dubbing backend; the full-dub preset only
*prefers* IndexTTS-2.5, applied by the front end when that model is already
installed.  The "best quality" preset is the shipped defaults, computed from the
model so it cannot drift from them.
"""

from ..models import ProjectSettings
from .settings import Settings

# Quality group: speed and video memory against detail.  Values for the two
# lighter presets are explicit; "best" is read from the defaults.
FAST = {
    "separation_chunk_seconds": 20.0,
    "separation_overlap_seconds": 0.5,
    "translation_context_sentences": 8,
    "translation_memory_sentences": 20,
    "tts_quality_check": False,
    "tts_index25_num_beams": 1,
    "replacement_voice_match_strength": 0.5,
    "replacement_bandwidth_extension": False,
    "replacement_whisper_match": 0.0,
}
BALANCED = {
    "separation_chunk_seconds": 30.0,
    "separation_overlap_seconds": 0.5,
    "translation_context_sentences": 16,
    "translation_memory_sentences": 30,
    "tts_quality_check": True,
    "tts_index25_num_beams": 2,
    "replacement_voice_match_strength": 0.8,
    "replacement_bandwidth_extension": True,
    "replacement_whisper_match": 0.5,
}
QUALITY_KEYS = tuple(FAST)

# Output group: which product the project aims at.  ``autoflow_default_content``
# exists only as a global default (it picks the batch product).
BILINGUAL = {
    "separation_enabled": False,
    "separation_mix_mode": "bilingual",
    "chinese_dubbing_timing_mode": "fit_window",
    "autoflow_default_content": "dubbing",
}
REPLACEMENT = {
    "separation_enabled": True,
    "separation_mix_mode": "replace",
    "spatial_rtf_enabled": True,
    "tts_fit_source_duration": True,
    "tts_dual_voice_anchor": True,
    "tts_asr_quality_check": True,
    "replacement_scope": "dubbed_speech",
    "replacement_timing_mode": "fit_then_shift",
    "replacement_max_auto_speed": 1.25,
    "replacement_voice_match_strength": 0.85,
    "replacement_whisper_match": 0.65,
    "replacement_noise_fill": True,
    "replacement_residual_suppression": 0.5,
    "replacement_residual_asr_check": True,
    "autoflow_default_content": "replacement",
}
# Blind tests and the timing benchmark favoured IndexTTS-2.5 for the full dub; it is
# a separate install, so it is only switched to when present.
REPLACEMENT_PREFERRED = {"tts_backend": "indextts2_5", "tts_model": "IndexTTS-2.5"}


def _entry(identifier, group, label, description, changes, requires="", prefer=None):
    return {
        "id": identifier,
        "group": group,
        "label": label,
        "description": description,
        "changes": changes,
        "requires": requires,
        # Extra changes applied only when the backend they name is installed.
        "prefer": dict(prefer or {}),
        "prefer_requires": (prefer or {}).get("tts_backend", ""),
    }


def catalog():
    defaults = Settings().model_dump(mode="json")
    best = {key: defaults[key] for key in QUALITY_KEYS}
    project_keys = set(ProjectSettings.model_fields)
    result = [
        _entry(
            "fast",
            "quality",
            "快速",
            "更快、更省显存：分块更短，翻译带的前后文更少，不做配音质检，完全配音版少做几步修饰。",
            FAST,
        ),
        _entry(
            "balanced",
            "quality",
            "均衡",
            "介于两者之间：保留质检和音色贴合，翻译上下文适中。",
            BALANCED,
        ),
        _entry(
            "best",
            "quality",
            "最佳质量",
            "出厂默认值。最慢、最占显存，细节最好。",
            best,
        ),
        _entry(
            "bilingual",
            "output",
            "双语版",
            "原声保留，叠加中文配音和双语字幕。",
            BILINGUAL,
        ),
        _entry(
            "replacement",
            "output",
            "完全配音版",
            "在有配音的句子处去掉原人声，需要先装好人声分离模型。",
            REPLACEMENT,
            requires="separation_roformer",
            prefer=REPLACEMENT_PREFERRED,
        ),
    ]
    for item in result:
        known = set(item["changes"])
        unknown = (known | set(item["prefer"])) - set(defaults)
        if unknown:  # a typo in a preset must fail loudly in tests, not silently at apply time
            raise KeyError(sorted(unknown))
        item["project_keys"] = sorted(known & project_keys)
    return result
