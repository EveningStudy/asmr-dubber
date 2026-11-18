from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .filtering import has_speakable_text, is_japanese_filler_only, normalized_japanese_utterance
from .models import Sentence

_SENTENCE_END = re.compile(r"[。！？!?\.]+[」』】）)]*$")
_SOFT_END = re.compile(r"[、，,；;：:…]+[」』】）)]*$")
_JA_BOUND_HEAD = re.compile(
    r"^[のをにがはでとへもやねよなっーぁぃぅぇぉゃゅょ]|^(?:でも|から|けど|ちゃ|じゃ|てる|てい|ない|ます|です)"
)
_SMALL_KANA_HEAD = re.compile(r"^[ぁぃぅぇぉゃゅょゎっーァィゥェォャュョヮッ]")
_TRAILING_SINGLE_KANA = re.compile(r"(?:^|[、，,。！？…\s])[ぁ-ゖァ-ヶ]$")
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
    """Attach punctuation from a full transcript to timestamped tokens.

    Some recognition backends timestamp Japanese words or characters but omit
    punctuation. Exact monotonic matching keeps their timestamps while recovering
    sentence boundaries from the backend's full transcript.
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


# Onomatopoeia repeated three or more times (しこしこしこしこ, ちゅぱちゅぱちゅぱ) is a sound,
# wherever it sits in the line.  Two repeats are only a sound outside this list of
# ordinary spoken doublings (そうそう, ほらほら, もうもう).
_REDUPLICATED = re.compile(r"([ぁ-ゖー]{2,3}?)\1{2,}")
_DOUBLED = re.compile(r"([ぁ-ゖー]{2,3}?)\1")
_SPOKEN_DOUBLES = {
    "そう",
    "ほら",
    "もう",
    "どう",
    "まだ",
    "いや",
    "ねえ",
    "はい",
    "うん",
    "よし",
    "ごめ",
    "だめ",
    "いい",
    "ここ",
    "これ",
    "それ",
    "あれ",
    "なに",
    "ほん",
    "まあ",
    "ほーら",
    "ねー",
    "すき",
    "もっと",
    "いく",
    "きて",
    "やだ",
    "ちょ",
}
# Breath, moans and kisses at the edge of a line: only multi-character forms, so the
# あ of あなた or the ん of んで is never cut away.
_EDGE_SOUND = re.compile(
    r"(?:"
    r"(?:あ|ぁ){2,}|(?:う|ぅ){2,}|(?:お|ぉ){2,}|(?:ん|ー){2,}|(?:あ?は|ぁ){2,}|(?:う?ふ){2,}"
    r"|はぁ|はあ|ふぅ|ふう|んっ|あっ|おっ|うっ|はっ|ちゅっ?|ちゅぱ|れろ|ぺろ|すー|ふー|はー"
    r"|[っーぁぃぅぇぉ]"
    r")+"
)
_MIN_SOUND_SECONDS = 0.4


# A mimetic word followed by a particle is part of the sentence, not a sound effect:
# 「トロトロなお顔」「ぬるぬるにして」「とろとろの」.
_USED_AS_WORD = "なにのとでだすしく"


def _is_used_as_word(compact: str, end: int) -> bool:
    return end < len(compact) and compact[end] in _USED_AS_WORD


def _sound_mask(compact: str) -> list[bool]:
    mask = [False] * len(compact)
    for match in _REDUPLICATED.finditer(compact):
        if match.group(1) not in _SPOKEN_DOUBLES and not _is_used_as_word(compact, match.end()):
            mask[match.start() : match.end()] = [True] * (match.end() - match.start())
    for match in _DOUBLED.finditer(compact):
        # Two repeats are often an ordinary mimetic word; only cut them at a line's edge.
        at_edge = match.start() == 0 or match.end() == len(compact)
        if (
            at_edge
            and match.group(1) not in _SPOKEN_DOUBLES
            and not _is_used_as_word(compact, match.end())
        ):
            mask[match.start() : match.end()] = [True] * (match.end() - match.start())
    for match in _EDGE_SOUND.finditer(compact):
        at_edge = match.start() == 0 or match.end() == len(compact)
        if at_edge and match.end() - match.start() >= 2:
            mask[match.start() : match.end()] = [True] * (match.end() - match.start())
    return mask


def split_sound_runs(tokens: list[TimedToken]) -> list[list[TimedToken]]:
    """Cut onomatopoeia, breath and moans out of a line that also has real speech.

    In a full dub the original voice of a dubbed line is removed.  When a line such
    as 「しこしこしこしこ、そうです」 is translated, only 「そうです」 is spoken in
    Chinese, so the sound part must become its own line: it is then skipped by
    translation and its original audio stays in the mix.  Only runs of at least
    0.4 s are split, and only when the rest of the line still has speakable text.
    """
    if len(tokens) < 2:
        return [tokens]
    compact_parts = [normalized_japanese_utterance(token.text) for token in tokens]
    compact = "".join(compact_parts)
    if not compact:
        return [tokens]
    mask = _sound_mask(compact)
    flags: list[bool | None] = []
    cursor = 0
    for part in compact_parts:
        if not part:
            flags.append(None)  # punctuation: belongs to whatever precedes it
        else:
            flags.append(all(mask[cursor : cursor + len(part)]))
        cursor += len(part)
    groups: list[tuple[bool, list[TimedToken]]] = []
    for token, flag in zip(tokens, flags, strict=True):
        if flag is None:
            if groups:
                groups[-1][1].append(token)
            else:
                groups.append((False, [token]))
            continue
        if groups and groups[-1][0] == flag:
            groups[-1][1].append(token)
        else:
            groups.append((flag, [token]))

    def text(group: list[TimedToken]) -> str:
        return "".join(token.text for token in group)

    # A short sound run is not worth its own line: fold it into a neighbour.
    merged: list[tuple[bool, list[TimedToken]]] = []
    for flag, group in groups:
        duration = group[-1].end_seconds - group[0].start_seconds
        if flag and duration < _MIN_SOUND_SECONDS:
            flag = False
        if merged and merged[-1][0] == flag:
            merged[-1][1].extend(group)
        else:
            merged.append((flag, list(group)))
    speech = [g for flag, g in merged if not flag]
    if not speech or not any(
        has_speakable_text(text(g)) and not is_japanese_filler_only(text(g)) for g in speech
    ):
        return [tokens]
    if len(merged) == 1:
        return [tokens]
    return [group for _, group in merged]


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
        source_text=text,
    )


def split_timed_tokens(
    tokens: Iterable[TimedToken],
    pause_seconds: float = 0.7,
    max_sentence_seconds: float = 15.0,
    split_sounds: bool = True,
) -> list[Sentence]:
    ordered = sorted(tokens, key=lambda item: (item.start_seconds, item.end_seconds))
    sentences: list[Sentence] = []
    current: list[TimedToken] = []

    def flush() -> None:
        nonlocal current
        if current:
            for part in split_sound_runs(current) if split_sounds else [current]:
                sentence = _make_sentence(len(sentences) + 1, part)
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

        has_hard_end = bool(_SENTENCE_END.search(text))
        has_soft_end = bool(_SOFT_END.search(text))
        next_clean = next_token.text.strip() if next_token is not None else ""
        starts_small_kana = bool(next_clean) and bool(_SMALL_KANA_HEAD.match(next_clean))
        mid_word_continuation = (
            not has_hard_end
            and not has_soft_end
            and bool(next_clean)
            and (
                starts_small_kana
                or bool(_JA_BOUND_HEAD.match(next_clean))
                or bool(_TRAILING_SINGLE_KANA.search(text))
            )
        )
        effective_pause = max(pause_seconds * 3.2, 1.8) if mid_word_continuation else pause_seconds

        should_split = has_hard_end or gap >= effective_pause
        # Split long sentences gracefully at natural comma/ellipsis pauses before hitting hard cap.
        if duration >= max_sentence_seconds * 0.55 and has_soft_end:
            should_split = True
        if duration >= max_sentence_seconds and not starts_small_kana:
            should_split = should_split or has_soft_end or len(current) > 1
        elif duration >= max_sentence_seconds * 1.25:
            should_split = True
        if next_token is None:
            should_split = True
        if should_split:
            flush()
    flush()
    return sentences
