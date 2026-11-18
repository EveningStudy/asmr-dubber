"""Mandarin syllable/phonetic matching for TTS pronunciation QC (B12) and text normalization."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def zh_syllable_map() -> dict[str, frozenset[str]]:
    path = Path(__file__).with_name("zh_syllables.json")
    if not path.is_file():
        return {}
    raw: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    by_char: dict[str, set[str]] = {}
    for syllable, chars in raw.items():
        for ch in chars:
            by_char.setdefault(ch, set()).add(syllable)
    return {ch: frozenset(syls) for ch, syls in by_char.items()}


def canonical_mandarin_syllable(syl: str) -> str:
    """Normalize front/back nasal codas and retroflex initials for soft/whispered ASR."""
    for prefix in ("zh", "ch", "sh"):
        if syl.startswith(prefix):
            syl = prefix[0] + syl[2:]
            break
    if syl.endswith("ng") and not syl.endswith("ong"):
        syl = syl[:-1]
    return syl


_INTERJECTIONS = frozenset(
    {"嗯", "唔", "啊", "哦", "噢", "喔", "哎", "欸", "诶", "嘿", "哈", "呐", "呀"}
)


def zh_phonetic_char_cost(ch_a: str, ch_b: str, table: dict[str, frozenset[str]]) -> float:
    """Phonetic substitution distance in [0.0, 1.0] between two Chinese characters."""
    if ch_a == ch_b:
        return 0.0
    if ch_a in _INTERJECTIONS and ch_b in _INTERJECTIONS:
        return 0.25
    sa = table.get(ch_a)
    sb = table.get(ch_b)
    if not sa or not sb:
        return 1.0
    if sa & sb:
        return 0.0
    ca = {canonical_mandarin_syllable(s) for s in sa}
    cb = {canonical_mandarin_syllable(s) for s in sb}
    if ca & cb:
        return 0.25
    if any(a[:1] == b[:1] and len(a) > 1 and len(b) > 1 for a in ca for b in cb):
        return 0.50
    return 1.0
