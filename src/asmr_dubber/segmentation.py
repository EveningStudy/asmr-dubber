from __future__ import annotations

import re
import unicodedata
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
_OPENING_BRACKETS = frozenset("「『（(“‘【〈《[")
_SOUND_SPLIT_PUNCT = frozenset("、，,…・！？!?—―,.~")
_EDGE_SOUND_MIN_GAP_SECONDS = 0.12


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
        for nchar in unicodedata.normalize("NFKC", char).casefold():
            if not _NON_CONTENT.match(nchar):
                normalized_chars.append(nchar)
                raw_positions.append(raw_index)
    normalized_full = "".join(normalized_chars)
    if not normalized_full:
        return tokens

    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        needle = _NON_CONTENT.sub("", unicodedata.normalize("NFKC", token.text).casefold())
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

    matched_indices = [idx for idx, (s, _) in enumerate(spans) if s >= 0]
    slice_bounds: dict[int, tuple[int, int]] = {}
    if matched_indices:
        slice_starts: list[int] = [0]
        slice_ends: list[int] = []
        for pos in range(len(matched_indices) - 1):
            cur_end = spans[matched_indices[pos]][1]
            nxt_start = spans[matched_indices[pos + 1]][0]
            cut = max(cur_end, nxt_start)
            while cut > cur_end and full_text[cut - 1] in _OPENING_BRACKETS:
                cut -= 1
            slice_ends.append(cut)
            slice_starts.append(cut)
        slice_ends.append(len(full_text))
        for idx, start, end in zip(matched_indices, slice_starts, slice_ends, strict=True):
            slice_bounds[idx] = (start, end)

    restored: list[TimedToken] = []
    for index, token in enumerate(tokens):
        if index not in slice_bounds:
            restored.append(token)
            continue
        raw_start, raw_end = slice_bounds[index]
        display = full_text[raw_start:raw_end]
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


def _has_edge_boundary(
    tokens: list[TimedToken],
    compact_parts: list[str],
    boundary: int,
) -> bool:
    """Return True if an edge sound run is separated by punctuation or >=0.12s pause."""
    char_owners: list[int] = []
    for tok_idx, part in enumerate(compact_parts):
        char_owners.extend([tok_idx] * len(part))
    if boundary <= 0 or boundary >= len(char_owners):
        return False
    left_tok = char_owners[boundary - 1]
    right_tok = char_owners[boundary]
    if left_tok == right_tok:
        text = tokens[left_tok].text
        # Check if any split punctuation lies inside the token between content chars.
        left_offset = boundary - char_owners.index(left_tok)
        seen_content = 0
        for ch in text:
            norm_len = len(normalized_japanese_utterance(ch))
            if seen_content >= left_offset and ch in _SOUND_SPLIT_PUNCT:
                return True
            seen_content += norm_len
            if seen_content > left_offset:
                break
        return False
    gap = tokens[right_tok].start_seconds - tokens[left_tok].end_seconds
    if gap >= _EDGE_SOUND_MIN_GAP_SECONDS:
        return True
    for between in range(left_tok + 1, right_tok):
        if any(ch in _SOUND_SPLIT_PUNCT for ch in tokens[between].text):
            return True
    left_text = tokens[left_tok].text
    right_text = tokens[right_tok].text
    for ch in reversed(left_text):
        if normalized_japanese_utterance(ch):
            break
        if ch in _SOUND_SPLIT_PUNCT:
            return True
    for ch in right_text:
        if normalized_japanese_utterance(ch):
            break
        if ch in _SOUND_SPLIT_PUNCT:
            return True
    return False


def _sound_mask(
    compact: str,
    tokens: list[TimedToken] | None = None,
    compact_parts: list[str] | None = None,
) -> list[bool]:
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
        at_start = match.start() == 0 and match.end() < len(compact)
        at_end = match.end() == len(compact) and match.start() > 0
        if (at_start or at_end) and match.end() - match.start() >= 2:
            if tokens is not None and compact_parts is not None:
                boundary = match.end() if at_start else match.start()
                if not _has_edge_boundary(tokens, compact_parts, boundary):
                    continue
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
    mask = _sound_mask(compact, tokens, compact_parts)
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


def _join_token_texts(tokens: list[TimedToken]) -> str:
    if not tokens:
        return ""
    has_latin_words = any(len(re.sub(r"[^A-Za-z0-9]", "", token.text)) >= 2 for token in tokens)
    parts: list[str] = [tokens[0].text]
    for token in tokens[1:]:
        prev = parts[-1]
        cur = token.text
        if (
            has_latin_words
            and prev
            and cur
            and not prev[-1].isspace()
            and not cur[0].isspace()
            and prev[-1] not in "'’-「『（("
            and cur[0] not in ".,!?;:'’)）」』"
            and re.search(r"[A-Za-z0-9.,!?;:]$", prev)
            and re.match(r"^[A-Za-z0-9]", cur)
            and not (prev[-1] == "." and len(prev) >= 2 and prev[-2].isdigit() and cur[0].isdigit())
        ):
            parts.append(" ")
        parts.append(cur)
    return clean_japanese_text("".join(parts))


def _make_sentence(index: int, tokens: list[TimedToken]) -> Sentence | None:
    text = _join_token_texts(tokens)
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
        text = _join_token_texts(current)
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
