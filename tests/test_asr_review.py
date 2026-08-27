import json
from pathlib import Path

import httpx

from asmr_dubber.asr_review import _build_windows, _evidence_range, review_transcriptions
from asmr_dubber.models import ProjectSettings, Sentence


def _sentence(identifier: str, start: float, end: float, text: str) -> Sentence:
    return Sentence(
        id=identifier,
        start_seconds=start,
        end_seconds=end,
        ja_text=text,
    )


def test_review_windows_keep_primary_timeline_and_ignore_unmatched_secondary() -> None:
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

    assert len(windows) == 2
    assert [len(window.evidence) for window in windows] == [2, 1]
    assert windows[0].evidence[0].id == "w000001-c01"
    assert all(
        evidence.text != "聞こえますか" for window in windows for evidence in window.evidence
    )


def test_evidence_time_keeps_primary_window_by_default() -> None:
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


def _settings() -> ProjectSettings:
    return ProjectSettings(
        asr_backend="faster_whisper",
        asr_model="large-v2",
        asr_review_enabled=True,
        asr_review_models=["kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2"],
        asr_review_text_priority_model="faster_whisper|large-v2",
        asr_review_timestamp_priority_model="faster_whisper|large-v2",
    )


def _transcriptions(*texts: tuple[str, str]) -> list[tuple[str, list[Sentence]]]:
    return [
        (label, [_sentence(f"{index}", 1.0, 2.0, text)])
        for index, (label, text) in enumerate(texts, start=1)
    ]


def test_consensus_skips_llm_and_records_decision(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "asmr_dubber.asr_review._request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )
    report = tmp_path / "review.json"

    sentences = review_transcriptions(
        _transcriptions(
            ("faster_whisper|large-v2", "こんにちは。"),
            ("kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2", "こんにちは"),
        ),
        _settings(),
        report,
    )

    assert [item.source_text for item in sentences] == ["こんにちは。"]
    result = json.loads(report.read_text(encoding="utf-8"))["results"][0]
    assert result["decision"] == "consensus"
    assert result["selected_candidate"] == 1


def test_majority_vote_beats_dissent_without_llm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "asmr_dubber.asr_review._request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )

    sentences = review_transcriptions(
        _transcriptions(
            ("faster_whisper|large-v2", "正しい文"),
            ("kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2", "正しい文。"),
            ("parakeet_nemo|model", "全く違う文"),
        ),
        _settings(),
        tmp_path / "review.json",
    )

    assert [item.source_text for item in sentences] == ["正しい文"]


def test_llm_can_only_select_an_existing_candidate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "asmr_dubber.asr_review._request_json",
        lambda *_args, **_kwargs: json.dumps(
            {
                "results": [
                    {
                        "window_id": "w000001",
                        "selected_candidate": 2,
                        "confidence": 0.88,
                    }
                ]
            }
        ),
    )
    report = tmp_path / "review.json"

    sentences = review_transcriptions(
        _transcriptions(
            ("faster_whisper|large-v2", "候補一"),
            ("kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2", "候補二"),
        ),
        _settings(),
        report,
    )

    assert [item.source_text for item in sentences] == ["候補二"]
    assert json.loads(report.read_text(encoding="utf-8"))["results"][0]["decision"] == "llm_choice"


def test_invalid_llm_selection_falls_back_to_primary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "asmr_dubber.asr_review._request_json",
        lambda *_args, **_kwargs: json.dumps(
            {
                "results": [
                    {
                        "window_id": "w000001",
                        "selected_candidate": 99,
                    }
                ]
            }
        ),
    )
    report = tmp_path / "review.json"

    sentences = review_transcriptions(
        _transcriptions(
            ("faster_whisper|large-v2", "主模型文字"),
            ("kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2", "其它文字"),
        ),
        _settings(),
        report,
    )

    assert [item.source_text for item in sentences] == ["主模型文字"]
    result = json.loads(report.read_text(encoding="utf-8"))["results"][0]
    assert result["decision"] == "fallback"
    assert "回退主模型" in result["reason"]


def test_llm_network_failure_does_not_abort_review(monkeypatch, tmp_path: Path) -> None:
    def fail_request(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("asmr_dubber.asr_review._request_json", fail_request)

    sentences = review_transcriptions(
        _transcriptions(
            ("faster_whisper|large-v2", "主模型文字"),
            ("kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2", "其它文字"),
        ),
        _settings(),
        tmp_path / "review.json",
    )

    assert [item.source_text for item in sentences] == ["主模型文字"]


def test_one_invalid_window_falls_back_without_losing_other_llm_choice(
    monkeypatch,
    tmp_path: Path,
) -> None:
    primary = [
        _sentence("p1", 1.0, 2.0, "主一"),
        _sentence("p2", 3.0, 4.0, "主二"),
    ]
    secondary = [
        _sentence("s1", 1.0, 2.0, "复核一"),
        _sentence("s2", 3.0, 4.0, "复核二"),
    ]

    def response(_settings, messages, _job_id):
        target_message = messages[-1]["content"]
        if '"target_window_ids":["w000001","w000002"]' in target_message:
            return '{"results":[]}'
        if '"target_window_ids":["w000001"]' in target_message:
            return json.dumps(
                {
                    "results": [
                        {
                            "window_id": "w000001",
                            "selected_candidate": 2,
                            "confidence": 0.9,
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "results": [
                    {
                        "window_id": "w000002",
                        "selected_candidate": 99,
                    }
                ]
            }
        )

    monkeypatch.setattr("asmr_dubber.asr_review._request_json", response)
    report = tmp_path / "review.json"

    sentences = review_transcriptions(
        [
            ("faster_whisper|large-v2", primary),
            ("kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2", secondary),
        ],
        _settings(),
        report,
    )

    assert [item.source_text for item in sentences] == ["复核一", "主二"]
    decisions = [
        item["decision"] for item in json.loads(report.read_text(encoding="utf-8"))["results"]
    ]
    assert decisions == ["llm_choice", "fallback"]
