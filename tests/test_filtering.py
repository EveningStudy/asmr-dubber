import pytest

from asmr_dubber.filtering import (
    has_speakable_text,
    implausible_asr_reason,
    is_japanese_filler_only,
)


@pytest.mark.parametrize(
    "text",
    [
        "あ、",
        "うん。",
        "うん、ああ、",
        "うわ！",
        "お、",
        "えーと……",
        "ふふっ",
        "はははは",
        "アアアア",
        "んあああ……",
        "くすくす",
        "（笑い声）",
        "[吐息]",
        "キス音",
        "[laughter]",
    ],
)
def test_standalone_japanese_fillers_are_detected(text: str) -> None:
    assert is_japanese_filler_only(text)


@pytest.mark.parametrize(
    "text",
    ["さあ始めましょう", "はい、分かりました", "そうだよね", "うわ、怒りっぽい", "あ、よし"],
)
def test_meaningful_japanese_sentences_are_not_filtered(text: str) -> None:
    assert not is_japanese_filler_only(text)


def test_impossible_asr_text_density_is_flagged() -> None:
    pathological = (
        "のおちんちじゃなく家族でいおちんちんします起きなくて猫おこまで"
        "ぐちゃぐちゃ疲れまくる腰の動き全くないよ何かに引っ張られちゃった"
    )
    reason = implausible_asr_reason(pathological, 2.573)
    assert reason is not None
    assert "疑似 ASR 幻觉" in reason


def test_fast_but_plausible_asr_sentence_is_kept() -> None:
    assert implausible_asr_reason("今日は一緒に始めましょう", 1.2) is None


@pytest.mark.parametrize("text", ["", "   ", "……", "——", "?!。", "😊🌸", "『……』"])
def test_non_lexical_tts_text_is_not_speakable(text: str) -> None:
    assert not has_speakable_text(text)


@pytest.mark.parametrize("text", ["你好……", "Hello!", "123", "１号", "καλημέρα"])
def test_letters_and_numbers_are_speakable(text: str) -> None:
    assert has_speakable_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございました。",
        "チャンネル登録お願いします",
        "最後までご視聴いただきありがとうございました",
        "Thank you for watching!",
    ],
)
def test_known_silence_hallucination_phrases_are_flagged(text: str) -> None:
    assert implausible_asr_reason(text, 3.0) is not None


@pytest.mark.parametrize("text", ["すきすきすき", "そうそう", "ちゅ、ちゅ、ちゅ", "うん、うん"])
def test_short_repeated_asmr_speech_is_never_a_repeat_hallucination(text: str) -> None:
    assert implausible_asr_reason(text, 2.0) is None
    assert implausible_asr_reason(text, 2.0, previous_text=text) is None


@pytest.mark.parametrize("text", ["ありがとうございました", "はい、おやすみなさい"])
def test_ordinary_closing_phrases_are_not_hallucinations(text: str) -> None:
    assert implausible_asr_reason(text, 2.0) is None


def test_long_line_repeated_verbatim_is_flagged_only_with_previous_text() -> None:
    line = "今日はゆっくりお耳をマッサージしていきますね"
    assert implausible_asr_reason(line, 4.0) is None
    assert implausible_asr_reason(line, 4.0, previous_text=line + "。") is not None
    assert implausible_asr_reason(line, 4.0, previous_text="別の台詞です") is None
