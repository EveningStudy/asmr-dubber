"""Length budgets, batch chunking, and post-translation quality checks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from itertools import pairwise

from .filtering import has_speakable_text, speakable_character_count
from .languages import SourceLanguage
from .models import Sentence
from .translation_schema import TranslationChunk, TranslationFlag


def length_budgets(
    sentences: list[Sentence], targets: list[Sentence], rate: float
) -> dict[str, int]:
    """How many characters (or words) each line can take when dubbed at ``rate`` per second.

    A dub may run into the pause before the next line, but at most one second past
    the line's own end, so a line before a long silence is not given a paragraph.
    """
    ordered = sorted(sentences, key=lambda s: (s.start_seconds, s.end_seconds, s.id))
    next_start = {current.id: later.start_seconds for current, later in pairwise(ordered)}
    budgets = {}
    for sentence in targets:
        own = sentence.end_seconds - sentence.start_seconds
        available = min(
            next_start.get(sentence.id, sentence.end_seconds + 1.0), sentence.end_seconds + 1.0
        )
        seconds = max(own, available - sentence.start_seconds)
        budgets[sentence.id] = max(2, round(seconds * rate))
    return budgets


def _chunks(
    sentences: list[Sentence],
    max_lines: int = 16,
    max_chars: int = 4_000,
) -> list[TranslationChunk]:
    result: list[TranslationChunk] = []
    current: list[Sentence] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.source_text)
        if current and (len(current) >= max_lines or count + length > max_chars):
            result.append(TranslationChunk(current))
            current = []
            count = 0
        current.append(sentence)
        count += length
    if current:
        result.append(TranslationChunk(current))
    return result


_KANA = re.compile(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\uff66-\uff9d]")
_ANY_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\uff66-\uff9f]")
_SHARED_HANZI_ASCII_KEY = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff0-9a-z]+$")
_HANZI_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# A line over its length budget by more than this share is flagged.
LENGTH_OVERRUN_RATIO = 1.2


def _same_text_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def _allow_same_as_source(
    source_text: str,
    target_text: str,
    key: str,
    *,
    source_language: SourceLanguage,
    target_language: str,
) -> bool:
    normalized_source = unicodedata.normalize("NFKC", source_text)
    normalized_target = unicodedata.normalize("NFKC", target_text)
    if _ANY_KANA.search(normalized_source) or _ANY_KANA.search(normalized_target):
        return False
    if not _SHARED_HANZI_ASCII_KEY.fullmatch(key):
        return False
    if _HANZI_CHAR.search(key) and (target_language != "zh" or source_language == "zh"):
        return False
    words = re.findall(r"[A-Za-z]+", normalized_target)
    if not words:
        return True
    if len(words) > 1:
        return False
    word = words[0]
    if word.casefold() == "ok" or (word.isupper() and len(word) <= 6):
        return True
    return source_language == "ja" and len(word) <= 4


def _used_length(text: str, budget_key: str) -> int:
    return len(text.split()) if budget_key == "max_words" else speakable_character_count(text)


def check_translations(
    sentences: list[Sentence],
    budgets: dict[str, int] | None = None,
    budget_key: str = "max_chars",
    only_ids: Iterable[str] | None = None,
    *,
    source_language: SourceLanguage = "ja",
    target_language: str | None = None,
) -> list[TranslationFlag]:
    """Self-check finished translations; nothing is written to the sentences.

    Flags leftover kana, a length over ``budgets`` by more than 20%, text equal to
    the source, and text equal to the previous line's translation (unless the
    source lines are equal too, which is ordinary repetition).  The check can be
    run again on any project, so the result never needs to be stored.
    """

    wanted = None if only_ids is None else set(only_ids)
    resolved_target = (
        target_language
        if target_language is not None
        else ("en" if budget_key == "max_words" else "zh")
    )
    ordered = sorted(
        (s for s in sentences if s.enabled and s.zh_text),
        key=lambda s: (s.start_seconds, s.end_seconds, s.id),
    )
    flags: list[TranslationFlag] = []
    previous: Sentence | None = None
    for sentence in ordered:
        before, previous = previous, sentence
        if (wanted is not None and sentence.id not in wanted) or sentence.status != "translated":
            continue
        text = sentence.zh_text
        if _KANA.search(text):
            flags.append(
                TranslationFlag(sentence.id, "kana_left", "译文里还有日文假名，可能漏译或没译完。")
            )
        budget = (budgets or {}).get(sentence.id)
        if budget:
            used = _used_length(text, budget_key)
            if used > budget * LENGTH_OVERRUN_RATIO:
                flags.append(
                    TranslationFlag(
                        sentence.id,
                        "too_long",
                        f"译文 {used} 超出时长预算 {budget} 的 20% 以上，配音可能赶不上。",
                    )
                )
        key = _same_text_key(text)
        if (
            key
            and key == _same_text_key(sentence.source_text)
            and has_speakable_text(text)
            and not _allow_same_as_source(
                sentence.source_text,
                text,
                key,
                source_language=source_language,
                target_language=resolved_target,
            )
        ):
            flags.append(TranslationFlag(sentence.id, "same_as_source", "译文与原文完全相同。"))
        if (
            before is not None
            and key
            and key == _same_text_key(before.zh_text)
            and _same_text_key(sentence.source_text) != _same_text_key(before.source_text)
        ):
            flags.append(
                TranslationFlag(
                    sentence.id, "repeats_previous", "译文与上一句完全重复，原文并不相同。"
                )
            )
    return flags
