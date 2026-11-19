"""Mandarin syllable/phonetic matching for TTS QC (B12) and spoken normalization (W15)."""

from __future__ import annotations

import json
import re
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


_GLOSSARY_SPLIT = re.compile(r"\s*(?:->|=>|=|：|:)\s*")
_ZH_DIGITS = "零一二三四五六七八九"
_PERCENT_RE = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*[%％]")
_YEAR_RE = re.compile(r"(?<!\d)([12]\d{3})\s*年")
_DECIMAL_RE = re.compile(r"(?<![A-Za-z\d])(\d+)\.(\d+)(?!\.\d)")
_CARDINAL_RE = re.compile(r"(?<![A-Za-z\d])(\d{1,8})(?![A-Za-z\d])")


@lru_cache(maxsize=64)
def parse_pronunciation_glossary(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse ``原词=读法`` / ``原词 -> 读法`` lines, sorted longest-source-first."""
    rules: dict[str, str] = {}
    for line in str(raw or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = _GLOSSARY_SPLIT.split(stripped, maxsplit=1)
        if len(parts) != 2:
            continue
        src, dst = parts[0].strip(), parts[1].strip()
        if src:
            rules[src] = dst
    return tuple(sorted(rules.items(), key=lambda pair: len(pair[0]), reverse=True))


def _int_under_10000_to_zh(n: int) -> str:
    if n == 0:
        return "零"
    units = ("", "十", "百", "千")
    digits = [int(c) for c in str(n)]
    length = len(digits)
    out: list[str] = []
    zero_pending = False
    for idx, d in enumerate(digits):
        pos = length - 1 - idx
        if d == 0:
            zero_pending = True
        else:
            if zero_pending and out:
                out.append("零")
            zero_pending = False
            if not (d == 1 and pos == 1 and not out):
                out.append(_ZH_DIGITS[d])
            if units[pos]:
                out.append(units[pos])
    return "".join(out)


def int_to_spoken_zh(n: int) -> str:
    """Convert non-negative integer ``0 <= n < 100_000_000`` into natural Mandarin."""
    if n == 0:
        return "零"
    if n >= 100_000_000:
        return "".join(_ZH_DIGITS[int(c)] for c in str(n))
    high, low = divmod(n, 10_000)
    if high == 0:
        return _int_under_10000_to_zh(low)
    prefix = _int_under_10000_to_zh(high) + "万"
    if low == 0:
        return prefix
    if low < 1000:
        return prefix + "零" + _int_under_10000_to_zh(low)
    return prefix + _int_under_10000_to_zh(low)


def normalize_chinese_spoken_numbers(text: str) -> str:
    """Expand years, percentages, decimals, and standalone numbers for Chinese TTS."""

    def repl_year(match: re.Match[str]) -> str:
        return "".join(_ZH_DIGITS[int(c)] for c in match.group(1)) + "年"

    def repl_percent(match: re.Match[str]) -> str:
        num = match.group(1)
        if "." in num:
            left, right = num.split(".", 1)
            spoken = int_to_spoken_zh(int(left)) + "点" + "".join(_ZH_DIGITS[int(c)] for c in right)
        else:
            val = int(num)
            spoken = "百" if val == 100 else int_to_spoken_zh(val)
        return "百分之" + spoken

    def repl_decimal(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        return int_to_spoken_zh(int(left)) + "点" + "".join(_ZH_DIGITS[int(c)] for c in right)

    def repl_cardinal(match: re.Match[str]) -> str:
        digits = match.group(1)
        if len(digits) >= 5 and digits.startswith("0"):
            return "".join(_ZH_DIGITS[int(c)] for c in digits)
        if len(digits) > 1 and digits.startswith("0"):
            return "".join(_ZH_DIGITS[int(c)] for c in digits)
        return int_to_spoken_zh(int(digits))

    out = _YEAR_RE.sub(repl_year, text)
    out = _PERCENT_RE.sub(repl_percent, out)
    out = _DECIMAL_RE.sub(repl_decimal, out)
    out = _CARDINAL_RE.sub(repl_cardinal, out)
    return out


def _glossary_pattern(src: str) -> str:
    escaped = re.escape(src)
    if src.isascii() and any(ch.isalnum() for ch in src):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    return escaped


def apply_spoken_normalization(
    text: str,
    *,
    glossary: str = "",
    normalize_numbers: bool = False,
    target_language: str = "zh",
) -> str:
    """Apply project pronunciation glossary and optional spoken number expansion."""
    result = text
    rules = parse_pronunciation_glossary(glossary)
    if rules:
        pattern = re.compile("|".join(_glossary_pattern(src) for src, _ in rules))
        lookup = dict(rules)
        result = pattern.sub(lambda m: lookup[m.group(0)], result)
    if normalize_numbers and target_language == "zh":
        result = normalize_chinese_spoken_numbers(result)
    return result
