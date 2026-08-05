from asmr_dubber.segmentation import (
    TimedToken,
    restore_punctuation,
    split_timed_tokens,
)


def test_restores_qwen_punctuation_and_splits_sentences() -> None:
    tokens = [
        TimedToken("さあ", 0.0, 0.35),
        TimedToken("始めましょう", 0.35, 1.2),
        TimedToken("次", 1.25, 1.5),
        TimedToken("ですよ", 1.5, 2.0),
    ]
    restored = restore_punctuation(tokens, "さあ始めましょう。次ですよ！")
    sentences = split_timed_tokens(restored, pause_seconds=0.5)
    assert [item.ja_text for item in sentences] == ["さあ始めましょう。", "次ですよ！"]
    assert sentences[0].start_seconds == 0.0
    assert sentences[0].end_seconds == 1.2
    assert sentences[1].start_seconds == 1.25


def test_splits_on_pause_without_punctuation() -> None:
    tokens = [
        TimedToken("おはよう", 0.0, 0.7),
        TimedToken("今日は", 1.5, 2.0),
        TimedToken("いい天気", 2.0, 2.8),
    ]
    sentences = split_timed_tokens(tokens, pause_seconds=0.55)
    assert [item.ja_text for item in sentences] == ["おはよう", "今日はいい天気"]


def test_splits_english_sentences_on_ascii_period() -> None:
    tokens = [
        TimedToken("Hello.", 0.0, 0.5),
        TimedToken(" Please", 0.55, 0.9),
        TimedToken(" sit", 0.9, 1.1),
        TimedToken(" down.", 1.1, 1.5),
    ]

    sentences = split_timed_tokens(tokens, pause_seconds=0.8)

    assert [item.source_text for item in sentences] == ["Hello.", "Please sit down."]
