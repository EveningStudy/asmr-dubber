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


def has_speakable_text(text: str) -> bool:
    """Return whether a TTS backend has lexical content to pronounce.

    Letters cover Chinese, Japanese, Korean and alphabetic scripts; numbers
    are also pronounceable. Punctuation, whitespace, emoji and decorative
    symbols alone must not be sent to any local or remote TTS backend.
    """

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in normalized)


# Phrases Whisper-family models invent on silence or noise (video outros learned
# from subtitles).  Matched as substrings of the normalised text, and only for
# short segments, so a long line that merely mentions one is left alone.
_HALLUCINATION_MARKERS = (
    "ご視聴",
    "ご清聴",
    "チャンネル登録",
    "高評価",
    "次の動画",
    "字幕",
    "thankyouforwatching",
    "thanksforwatching",
    "amara.org",
)
_HALLUCINATION_MAX_CHARACTERS = 40
# Shorter repeats are ordinary ASMR speech (「そうそう」「すきすき」).
_ADJACENT_REPEAT_MIN_CHARACTERS = 8


def _comparison_key(text: str) -> str:
    return _PUNCTUATION.sub("", unicodedata.normalize("NFKC", str(text or ""))).casefold()


def known_hallucination_phrase(text: str) -> bool:
    key = _comparison_key(text)
    if not key or len(_ASR_CONTENT.findall(key)) > _HALLUCINATION_MAX_CHARACTERS:
        return False
    return any(marker in key for marker in _HALLUCINATION_MARKERS)


def implausible_asr_reason(
    text: str, duration_seconds: float, previous_text: str | None = None
) -> str | None:
    """Return a user-facing reason for physically implausible ASR output.

    Catches extreme text-density failures, known silence hallucinations and a
    long line repeated verbatim right after ``previous_text`` (the preceding
    sentence, when the caller has it).  A legitimate fast sentence is kept; short
    repeated speech and onomatopoeia are never treated as repeats.  The row stays
    in the table but is disabled before translation/TTS.
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
    if known_hallucination_phrase(text):
        return "自动停用：识别结果是常见的静音幻觉用语（如视听感谢、频道订阅），请试听核对。"
    if previous_text is not None:
        key = _comparison_key(text)
        if (
            key
            and key == _comparison_key(previous_text)
            and len(_ASR_CONTENT.findall(key)) >= _ADJACENT_REPEAT_MIN_CHARACTERS
        ):
            return "自动停用：与上一句完全相同，疑似 ASR 重复幻觉，请试听核对后再手动启用。"
    return None


_SPEAKABLE_CATEGORIES = {"Lu", "Ll", "Lt", "Lo", "Nd", "Nl", "No"}


def speakable_character_count(text: str) -> int:
    """Count letters and digits that carry lexical content.

    Prolonged-sound marks, punctuation and symbols are deliberately excluded, so
    「あー…」 counts one character.
    """

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return sum(
        1 for character in normalized if unicodedata.category(character) in _SPEAKABLE_CATEGORIES
    )


def is_negligible_sound_segment(text: str, duration_seconds: float) -> bool:
    """Recognise ASR fragments that are too short to be a spoken line.

    A 0.08 second 「き」 is a click or breath, not a sentence.  The result is only
    used to disable the row before translation; the user can enable it again.
    """

    return 0.0 < duration_seconds < 0.15 and speakable_character_count(text) <= 1


def looks_like_sound_effect(text: str, duration_seconds: float) -> bool:
    """Fallback when a model returns an empty translation without a ``kind``.

    At most four content characters within one second is a vocalisation far more
    often than a missed line; longer or slower text stays a translation failure.
    """

    return duration_seconds <= 1.0 and speakable_character_count(text) <= 4


_ROMAJI_BASE = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}  # fmt: skip
_ROMAJI_SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
_ROMAJI_SMALL_VOWEL = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
_ROMAJI_SPECIAL_PAIRS = {
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "しぇ": "she", "じぇ": "je", "ちぇ": "che",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
}  # fmt: skip
_ROMAJI_PUNCTUATION = {"、": ", ", "。": ". ", "！": "! ", "？": "? ", "・": " ", "　": " "}
_VOWELS = "aiueo"


def kana_to_romaji(text: str) -> str:
    """Convert hiragana and katakana to lower-case Hepburn romaji.

    Characters that are not kana (kanji, Latin letters, punctuation) are kept, so
    the caller can detect an unconvertible line by looking for remaining CJK.
    """

    value = _katakana_to_hiragana(unicodedata.normalize("NFKC", str(text or "")).strip())
    output: list[str] = []
    geminate = False
    index = 0
    while index < len(value):
        character = value[index]
        following = value[index + 1] if index + 1 < len(value) else ""
        piece: str | None = None
        step = 1
        if character in {"っ", "ッ"}:
            geminate = True
            index += 1
            continue
        if character == "ー":
            last_vowel = next((c for c in reversed("".join(output)) if c in _VOWELS), "")
            piece = last_vowel
        elif character + following in _ROMAJI_SPECIAL_PAIRS:
            piece = _ROMAJI_SPECIAL_PAIRS[character + following]
            step = 2
        elif character in _ROMAJI_BASE and following in _ROMAJI_SMALL_Y:
            stem = _ROMAJI_BASE[character]
            if stem.endswith("i") and character not in "ゃゅょ":
                stem = stem[:-1]
                glide = "" if stem in {"sh", "ch", "j"} else "y"
                piece = stem + glide + _ROMAJI_SMALL_Y[following]
                step = 2
        if piece is None:
            piece = _ROMAJI_BASE.get(character)
        if piece is None:
            geminate = False
            output.append(_ROMAJI_PUNCTUATION.get(character, character))
            index += 1
            continue
        if geminate and piece and piece[0] not in _VOWELS + "n":
            piece = ("t" if piece.startswith("ch") else piece[0]) + piece
        geminate = False
        output.append(piece)
        index += step
    return re.sub(r"\s+", " ", "".join(output)).strip().lower()
