"""Prompt templates, language/sound rules, context builders, and sentence status helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from importlib.resources import files
from typing import Literal

from .filtering import (
    has_speakable_text,
    is_japanese_filler_only,
    kana_to_romaji,
)
from .languages import (
    SourceLanguage,
    SpeechSourceLanguage,
    source_language_label,
)
from .models import Sentence
from .translation_schema import TranslationChunk, TranslationEntry

SoundPolicy = Literal["skip", "romaji", "keep_source"]
SOUND_POLICIES: tuple[SoundPolicy, ...] = ("skip", "romaji", "keep_source")


def empty_translation_is_expected(sentence: Sentence) -> bool:
    return not has_speakable_text(sentence.source_text) or is_japanese_filler_only(
        sentence.source_text
    )


_BUDGET_RULE = (
    "每项的 {key} 是按原句说话时长估算的{unit}上限，用于让配音和原声对齐。"
    "在不丢失意思的前提下尽量不超过它；需要删减时，先删重复、语气词和可以省略的修饰。"
)


def budget_rule(chunk: TranslationChunk) -> str:
    """Prompt sentence explaining the length budget, or "" when there is none."""
    if not chunk.budgets:
        return ""
    unit = "英文单词数" if chunk.budget_key == "max_words" else "字数（不含标点）"
    rule = "\n" + _BUDGET_RULE.format(key=chunk.budget_key, unit=unit)
    if chunk.compress:
        rule += (
            f"\n以上句子当前 existing_zh 超出了 {chunk.budget_key} 上限；"
            "请在不丢失核心意思、人物称谓和语气的前提下精简译文，"
            f"把 zh 压缩到 {chunk.budget_key} 以内。"
        )
    return rule


def _load_prompt(name: str) -> str:
    return files("asmr_dubber.prompts").joinpath(name).read_text(encoding="utf-8").strip()


_SYSTEM_PROMPT_TEMPLATE = _load_prompt("translation.md")
_DEEPSEEK_STRUCTURE_PROMPT = _load_prompt("translation-structure.md")
_SCRIPT_RECONCILIATION_PROMPT = _load_prompt("script-reconciliation.md")

_SOURCE_PROMPT_VALUES: dict[SpeechSourceLanguage, tuple[str, str, str]] = {
    "ja": (
        "日语",
        "日语中的「あ」「え」「う」「ん」「ふふ」「えーと」",
        "\n   日语中的「しこしこ」「シコシコ」及其罗马字转写 “shiko shiko” 或 “shikoshiko” "
        '一律按动作或摩擦拟声处理（`kind` 为 "sound"），不得翻译成“撸”“撸撸”等会被合成语音的文字；'
        "同项还有实义台词时只翻译台词。",
    ),
    "en": ("英语", "英语中的 “uh”“um”“erm”", ""),
    "zh": ("中文", "中文中的“呃”“嗯……”", ""),
}

_SOUND_POLICY_RULES: dict[str, str] = {
    "skip": '`kind` 为 "sound" 时 `zh` 必须是空字符串。',
    "romaji": (
        '`kind` 为 "sound" 时，`zh` 写这个声音的罗马字（小写，例如「き」写 ki），'
        "不得翻成中文拟声词；无法拼写的呼吸声、笑声写空字符串。"
    ),
    "keep_source": '`kind` 为 "sound" 时，`zh` 原样抄写 `source`，不翻译。',
}


def sound_policy_rule(policy: str) -> str:
    return _SOUND_POLICY_RULES.get(policy, _SOUND_POLICY_RULES["skip"])


def default_translation_prompt(
    source_language: SourceLanguage = "ja",
    target_language: str = "zh",
    sound_policy: str = "skip",
) -> str:
    """Render the packaged prompt for one source language and sound policy."""

    prompt_language = source_language
    language, filler_examples, sound_rule = _SOURCE_PROMPT_VALUES[prompt_language]
    template = _SYSTEM_PROMPT_TEMPLATE
    if target_language == "en":
        template = template.replace("简体中文", "英语").replace("中文", "英语")
    prompt = (
        template.replace("{{SOURCE_LANGUAGE}}", language)
        .replace("{{FILLER_EXAMPLES}}", filler_examples)
        .replace("{{SOURCE_SPECIFIC_SOUND_RULE}}", sound_rule)
        .replace("{{SOUND_POLICY_RULE}}", sound_policy_rule(sound_policy))
    )
    if target_language == "en":
        prompt += "\n输出英语台词。JSON 键 zh 是兼容字段名，不表示输出中文；不得修改键名。"
    return prompt


def _builtin_translation_prompts(source_language: SourceLanguage) -> set[str]:
    return {
        default_translation_prompt(source_language, target_lang, sound_policy=policy).strip()
        for target_lang in ("zh", "en")
        for policy in SOUND_POLICIES
    }


def _resolve_system_prompt(
    system_prompt: str,
    source_language: SourceLanguage,
    target_language: str,
    sound_policy: str,
) -> str:
    default_prompts = _builtin_translation_prompts(source_language)
    if not system_prompt.strip() or system_prompt.strip() in default_prompts:
        resolved = default_translation_prompt(source_language, target_language, sound_policy)
    elif target_language == "en":
        resolved = (
            system_prompt
            + "\n本次覆盖目标语言为英语，输入原文语言为"
            + source_language_label(source_language)
            + "；忽略此前输出中文的要求。"
            + "\n目标语言必须是英语；JSON 的 zh 字段存放英文译文，这是兼容字段名。"
        )
    else:
        resolved = system_prompt
    if not resolved.strip():
        return default_translation_prompt(source_language, target_language, sound_policy)
    return resolved


# Backward-compatible Japanese default for callers that imported this name.
SYSTEM_PROMPT = default_translation_prompt("ja")


def _deepseek_translation_request(
    target: str,
    expected_ids: list[str],
    attempt: int,
    sound_policy: str = "skip",
) -> str:
    output_schema = json.dumps(
        {
            "translations": [
                {"id": sentence_id, "zh": "", "kind": "speech"} for sentence_id in expected_ids
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    retry_note = (
        ""
        if attempt == 1
        else (
            f"这是第 {attempt} 次校验重试。上一次输出没有通过结构校验；"
            "这次必须逐字沿用结构模板中的键名、id 和顺序。"
        )
    )
    return (
        _DEEPSEEK_STRUCTURE_PROMPT.replace("{{SOUND_POLICY_RULE}}", sound_policy_rule(sound_policy))
        .replace("{{RETRY_NOTE}}", retry_note)
        .replace("{{OUTPUT_SCHEMA}}", output_schema)
        .replace("{{TARGET_JSON}}", target)
        .strip()
    )


def _json_lines(
    sentences: Iterable[Sentence],
    field: str = "source",
    budgets: dict[str, int] | None = None,
    budget_key: str = "max_chars",
) -> str:
    items = []
    for sentence in sentences:
        item: dict[str, object] = {"id": sentence.id, "source": sentence.source_text}
        if sentence.speaker_id:
            item["speaker"] = sentence.speaker_id
        if budgets and sentence.id in budgets:
            item[budget_key] = budgets[sentence.id]
        if field == "both" and sentence.zh_text:
            item["existing_zh"] = sentence.zh_text
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _translation_memory(sentences: Iterable[Sentence]) -> str:
    items = []
    for sentence in sentences:
        if not sentence.zh_text:
            continue
        entry: dict[str, object] = {
            "id": sentence.id,
            "source": sentence.source_text,
            "zh": sentence.zh_text,
        }
        if sentence.speaker_id:
            entry["speaker"] = sentence.speaker_id
        items.append(entry)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _context_for_chunk(
    sentences: list[Sentence],
    chunk: TranslationChunk,
    maximum_sentences: int,
) -> str:
    if maximum_sentences <= 0:
        return "[]"
    positions = {sentence.id: index for index, sentence in enumerate(sentences)}
    indices = [positions[item.id] for item in chunk.sentences if item.id in positions]
    if not indices:
        return "[]"
    first, last = min(indices), max(indices)
    room = max(0, maximum_sentences - (last - first + 1))
    start = max(0, first - room // 2)
    end = min(len(sentences), last + 1 + (room - (first - start)))
    if end - start < maximum_sentences:
        start = max(0, end - maximum_sentences)
    return _json_lines(sentences[start:end], field="both")


def _bounded_translation_memory(
    sentences: list[Sentence],
    chunk: TranslationChunk,
    maximum_sentences: int,
) -> str:
    if maximum_sentences <= 0:
        return "[]"
    current_ids = {sentence.id for sentence in chunk.sentences}
    confirmed = [
        sentence for sentence in sentences if sentence.zh_text and sentence.id not in current_ids
    ]
    return _translation_memory(confirmed[-maximum_sentences:])


def _as_entry(value: TranslationEntry | str) -> TranslationEntry:
    """Machine translators and test doubles may still return plain strings."""

    return value if isinstance(value, TranslationEntry) else TranslationEntry(str(value))


def _clear_audio(sentence: Sentence) -> None:
    sentence.tts_file = None
    sentence.tts_cache_key = None
    sentence.tts_duration_seconds = None
    sentence.tts_candidates = []
    sentence.tts_warning = None
    sentence.tts_attempt = 0


def _mark_translated(sentence: Sentence, text: str) -> None:
    if sentence.zh_text != text:
        _clear_audio(sentence)
    if sentence.status == "skipped_sound" and not sentence.chinese_audio_enabled:
        # The row was a sound under an earlier run and is a real line now.
        sentence.chinese_audio_enabled = True
    sentence.zh_text = text
    sentence.status = "translated"
    sentence.error = None


def _mark_filler(sentence: Sentence) -> None:
    sentence.zh_text = ""
    sentence.enabled = False
    _clear_audio(sentence)
    sentence.status = "skipped_filler"
    sentence.error = None


def _looks_like_romaji(text: str) -> bool:
    return bool(text) and all(ord(character) < 128 or character in "…’‘—–" for character in text)


def _sound_text(sentence: Sentence, model_text: str, policy: str) -> str:
    """Subtitle text for a sound row under the romaji and keep_source policies."""

    if policy == "keep_source":
        return sentence.source_text
    if _looks_like_romaji(model_text):
        return model_text
    converted = kana_to_romaji(sentence.source_text)
    # Kanji cannot be spelled locally; the original is the honest subtitle then.
    return sentence.source_text if re.search(r"[㐀-鿿]", converted) else converted


def _mark_sound(sentence: Sentence, model_text: str, policy: str) -> None:
    """Apply the sound policy: skip it, or keep a subtitle that is never voiced."""

    _clear_audio(sentence)
    text = "" if policy == "skip" else _sound_text(sentence, model_text, policy)
    sentence.zh_text = text
    sentence.status = "skipped_sound"
    sentence.error = None
    if text:
        # Visible in subtitles, but the Chinese TTS must not read "ki" aloud.
        sentence.chinese_audio_enabled = False
    else:
        sentence.enabled = False
