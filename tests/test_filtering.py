import pytest

from asmr_dubber.filtering import implausible_asr_reason, is_japanese_filler_only


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
