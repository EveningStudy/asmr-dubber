from asmr_dubber.asr_review import _build_windows, _evidence_range
from asmr_dubber.models import Sentence


def _sentence(identifier: str, start: float, end: float, text: str) -> Sentence:
    return Sentence(
        id=identifier,
        start_seconds=start,
        end_seconds=end,
        ja_text=text,
    )


def test_review_windows_align_overlapping_candidates_and_keep_secondary_only() -> None:
    primary = [
        _sentence("p1", 1.0, 2.0, "始めましょう"),
        _sentence("p2", 4.0, 5.0, "次です"),
    ]
    secondary = [
        _sentence("s1", 1.2, 2.2, "さあ始めましょう"),
        _sentence("s2", 2.8, 3.2, "聞こえますか"),
    ]

    windows = _build_windows(
        [("primary", primary), ("secondary", secondary)],
        max_drift_seconds=0.5,
    )

    assert len(windows) == 3
    assert [len(window.evidence) for window in windows] == [2, 1, 1]
    assert windows[0].evidence[0].id == "w000001-c01"
    assert windows[1].evidence[0].text == "聞こえますか"


def test_evidence_time_uses_median_across_models() -> None:
    primary = [_sentence("p1", 10.0, 12.0, "こんにちは")]
    alternatives = [
        _sentence("a1", 10.2, 12.2, "こんにちは"),
        _sentence("b1", 9.8, 11.8, "こんにちは"),
    ]
    window = _build_windows(
        [
            ("primary", primary),
            ("alternative-a", alternatives[:1]),
            ("alternative-b", alternatives[1:]),
        ],
        max_drift_seconds=1.0,
    )[0]

    start, end = _evidence_range(window, [item.id for item in window.evidence])

    assert start == 10.0
    assert end == 12.0


def test_evidence_time_honors_timestamp_priority_source() -> None:
    primary = [_sentence("p1", 10.0, 12.0, "こんにちは")]
    alternative = [_sentence("a1", 10.4, 11.7, "こんにちは")]
    window = _build_windows(
        [("primary", primary), ("time-priority", alternative)],
        max_drift_seconds=1.0,
    )[0]

    start, end = _evidence_range(
        window,
        [window.evidence[0].id],
        "time-priority",
    )

    assert start == 10.4
    assert end == 11.7
