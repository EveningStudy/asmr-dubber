from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .models import Sentence

_SENTENCE_END = re.compile(r"[。！？!?]+[」』】）)]*$")
_SOFT_END = re.compile(r"[、，,；;：:…]+[」』】）)]*$")
_CJK = r"\u3040-\u30ff\u3400-\u9fff"
_NON_CONTENT = re.compile(rf"[^A-Za-z0-9{_CJK}]")


@dataclass(frozen=True)
class TimedToken:
    text: str
    start_seconds: float
    end_seconds: float


def clean_japanese_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(rf"(?<=[{_CJK}]) (?=[{_CJK}])", "", value)
    value = re.sub(rf"(?<=[{_CJK}]) (?=[。！？、，」』】）])", "", value)
    value = re.sub(rf"(?<=[「『【（]) (?=[{_CJK}])", "", value)
    return value


def restore_punctuation(tokens: list[TimedToken], full_text: str) -> list[TimedToken]:
    """Attach punctuation from Qwen's transcript to aligner tokens.

    The forced aligner timestamps Japanese words/characters but can omit punctuation.
    Qwen's ASR transcript retains it. Exact monotonic matching lets us keep timestamps
    while recovering sentence boundaries without asking another model to retime text.
    """
    if not tokens or not full_text.strip():
        return tokens

    normalized_chars: list[str] = []
    raw_positions: list[int] = []
    for raw_index, char in enumerate(full_text):
        if not _NON_CONTENT.match(char):
            normalized_chars.append(char)
            raw_positions.append(raw_index)
    normalized_full = "".join(normalized_chars)
    if not normalized_full:
        return tokens

    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        needle = _NON_CONTENT.sub("", token.text)
        if not needle:
            spans.append((-1, -1))
            continue
        found = normalized_full.find(needle, cursor)
        if found < 0:
            return tokens
        raw_start = raw_positions[found]
        end_normalized = found + len(needle)
        raw_end = raw_positions[end_normalized - 1] + 1
        spans.append((raw_start, raw_end))
        cursor = end_normalized

    restored: list[TimedToken] = []
    for index, token in enumerate(tokens):
        raw_start, raw_end = spans[index]
        if raw_start < 0:
            restored.append(token)
            continue
        next_start = len(full_text)
        for later_start, _ in spans[index + 1 :]:
            if later_start >= 0:
                next_start = later_start
                break
        if index == 0:
            raw_start = 0
        display = full_text[raw_start : max(raw_end, next_start)]
        restored.append(
            TimedToken(
                text=display or token.text,
                start_seconds=token.start_seconds,
                end_seconds=token.end_seconds,
            )
        )
    return restored


def _is_content(text: str) -> bool:
    return bool(re.search(rf"[A-Za-z0-9{_CJK}]", text))


def _make_sentence(index: int, tokens: list[TimedToken]) -> Sentence | None:
    text = clean_japanese_text("".join(token.text for token in tokens))
    if not text or not _is_content(text):
        return None
    return Sentence(
        id=f"s{index:06d}",
        start_seconds=max(0.0, tokens[0].start_seconds),
        end_seconds=max(tokens[0].start_seconds + 0.01, tokens[-1].end_seconds),
        ja_text=text,
    )


def split_timed_tokens(
    tokens: Iterable[TimedToken],
    pause_seconds: float = 0.7,
    max_sentence_seconds: float = 15.0,
) -> list[Sentence]:
    ordered = sorted(tokens, key=lambda item: (item.start_seconds, item.end_seconds))
    sentences: list[Sentence] = []
    current: list[TimedToken] = []

    def flush() -> None:
        nonlocal current
        if current:
            sentence = _make_sentence(len(sentences) + 1, current)
            if sentence is not None:
                sentences.append(sentence)
            current = []

    for position, token in enumerate(ordered):
        if not token.text or token.end_seconds <= token.start_seconds:
            continue
        current.append(token)
        text = clean_japanese_text("".join(part.text for part in current))
        next_token = ordered[position + 1] if position + 1 < len(ordered) else None
        gap = (
            max(0.0, next_token.start_seconds - token.end_seconds)
            if next_token is not None
            else float("inf")
        )
        duration = token.end_seconds - current[0].start_seconds

        should_split = bool(_SENTENCE_END.search(text))
        should_split = should_split or gap >= pause_seconds
        if duration >= max_sentence_seconds:
            should_split = should_split or bool(_SOFT_END.search(text)) or len(current) > 1
        if next_token is None:
            should_split = True
        if should_split:
            flush()
    flush()
    return sentences
