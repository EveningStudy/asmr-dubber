from __future__ import annotations

import re
import unicodedata

_PUNCTUATION = re.compile(r"[\s、。,.!?！？…‥・「」『』（）()【】\[\]｢｣]+")
_TRAILING_SOUND = re.compile(r"[っッー―～〜~]+$")
_ASR_CONTENT = re.compile(r"[0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]")

# Deliberately conservative: only a whole ASR sentence made entirely from a
# non-lexical vocalisation is skipped.  Meaningful lines such as
# 「あ、よし」or「うわ、怒りっぽい」do not match and remain enabled.
_STANDALONE_FILLERS = {
    "あ",
    "ああ",
    "あっ",
    "あー",
    "え",
    "えっ",
    "えー",
    "えーと",
    "えっと",
    "ええと",
    "う",
    "うう",
    "うん",
    "うーん",
    "うわ",
    "お",
    "おお",
    "おっ",
    "ん",
    "んん",
    "はぁ",
    "はあ",
    "ふぅ",
    "ふう",
    "ふふ",
    "ふふふ",
    "うふふ",
    "へへ",
    "えへへ",
    "あは",
    "あはは",
    "わはは",
    "くすくす",
    "笑",
    "笑い",
    "笑い声",
    "呼吸",
    "息",
    "吐息",
    "喘ぎ",
    "喘ぎ声",
    "呻吟",
    "きす",
    "きす音",
    "ちゅ",
    "ちゅっ",
    "music",
    "laughter",
    "breathing",
    "moaning",
    "んあ",
    "んああ",
}

_NON_LEXICAL_PATTERNS = (
    r"(?:あ|ぁ){2,}",
    r"(?:う|ぅ){2,}",
    r"(?:え|ぇ){2,}",
    r"(?:お|ぉ){2,}",
    r"ん+(?:あ|ぁ)+ん*",
    r"(?:あ|ぁ)+ん+(?:(?:あ|ぁ)|ん)*",
    r"(?:あ?は){2,}",
    r"(?:う?ふ){2,}",
    r"(?:え?へ){2,}",
    r"(?:くす){2,}",
    r"(?:きゃ){2,}",
)


def _katakana_to_hiragana(text: str) -> str:
    return "".join(chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char for char in text)


def normalized_japanese_utterance(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return _katakana_to_hiragana(_PUNCTUATION.sub("", value).strip())


def is_japanese_filler_only(text: str) -> bool:
    compact = normalized_japanese_utterance(text)
    if not compact:
        return False
    if compact in _STANDALONE_FILLERS:
        return True
    without_tail = _TRAILING_SOUND.sub("", compact)
    if without_tail in _STANDALONE_FILLERS:
        return True
    if any(re.fullmatch(pattern, without_tail) for pattern in _NON_LEXICAL_PATTERNS):
        return True
    # A sequence such as 「うん、ああ」 becomes one compact string.
    return bool(re.fullmatch(r"(?:うん|あ+|え+|うわ+|お+|ん+){2,}", without_tail))


def implausible_asr_reason(text: str, duration_seconds: float) -> str | None:
    """Return a user-facing reason for physically implausible ASR output.

    This deliberately catches only extreme text-density failures.  A legitimate
    fast sentence is kept; a multi-sentence paragraph forced into a two-second
    timestamp is retained in the table but disabled before translation/TTS.
    """

    if duration_seconds <= 0:
        return None
    content_characters = len(_ASR_CONTENT.findall(unicodedata.normalize("NFKC", text)))
    characters_per_second = content_characters / duration_seconds
    if content_characters >= 24 and characters_per_second > 20.0:
        return (
            f"自动停用：{duration_seconds:.2f} 秒内识别出 {content_characters} 个字符"
            f"（{characters_per_second:.1f} 字/秒），疑似 ASR 幻觉；"
            "请试听核对后再手动启用。"
        )
    return None
