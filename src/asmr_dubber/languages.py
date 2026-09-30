from __future__ import annotations

from typing import Literal

SourceLanguage = Literal["ja", "en", "zh"]
SpeechSourceLanguage = Literal["ja", "en", "zh"]
TargetLanguage = Literal["zh", "en"]

SOURCE_LANGUAGE_LABELS: dict[str, str] = {
    "ja": "日语",
    "en": "英语",
    "zh": "中文",
}

QWEN_LANGUAGE_NAMES: dict[SourceLanguage, str] = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
}

MACHINE_TRANSLATION_LANGUAGE_CODES: dict[str, dict[SpeechSourceLanguage, str]] = {
    "deepl": {"ja": "JA", "en": "EN", "zh": "ZH"},
    "google_translate": {"ja": "ja", "en": "en", "zh": "zh-CN"},
    "microsoft_translate": {"ja": "ja", "en": "en", "zh": "zh-Hans"},
}


def source_language_label(language: str) -> str:
    return SOURCE_LANGUAGE_LABELS.get(language, language or "未知")


def qwen_language_name(language: SourceLanguage) -> str:
    return QWEN_LANGUAGE_NAMES[language]


def reference_language_code(language: SourceLanguage) -> str:
    """Return the language code used by external voice-cloning APIs."""

    return language
