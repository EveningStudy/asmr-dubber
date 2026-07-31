import json

import pytest

from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence
from asmr_dubber.pipeline import export_transcript
from asmr_dubber.timing import dubbing_start_seconds, plan_dubbing_timing


def _sentence(
    sentence_id: str,
    start: float,
    duration: float | None,
    *,
    enabled: bool = True,
    chinese: str = "中文",
) -> Sentence:
    return Sentence(
        id=sentence_id,
        start_seconds=start,
        end_seconds=start + 0.8,
        ja_text="日本語",
        zh_text=chinese,
        enabled=enabled,
        tts_duration_seconds=duration,
    )


def test_dubbing_starts_at_source_time_with_global_millisecond_offset() -> None:
    sentence = _sentence("s1", 0.1, 1.0)

    assert dubbing_start_seconds(sentence) == pytest.approx(0.1)
    assert dubbing_start_seconds(sentence, 500) == pytest.approx(0.6)
    assert dubbing_start_seconds(sentence, -200) == 0.0


def test_no_conflict_keeps_original_tempo_and_last_sentence_is_not_accelerated() -> None:
    planned = plan_dubbing_timing([_sentence("s1", 1.0, 1.5), _sentence("s2", 3.0, 8.0)])

    assert [item.start_seconds for item in planned] == [1.0, 3.0]
    assert [item.speed_factor for item in planned] == [1.0, 1.0]
    assert planned[0].remaining_overlap_seconds == 0.0


def test_conflict_uses_only_the_speed_needed_to_fit_the_window() -> None:
    planned = plan_dubbing_timing(
        [_sentence("s1", 1.0, 2.2), _sentence("s2", 3.0, 1.0)],
        max_auto_speed=1.2,
    )

    assert planned[0].speed_factor == pytest.approx(1.1)
    assert planned[0].effective_duration_seconds == pytest.approx(2.0)
    assert planned[0].remaining_overlap_seconds == pytest.approx(0.0)


def test_conflict_stops_at_maximum_speed_and_preserves_remaining_overlap() -> None:
    planned = plan_dubbing_timing(
        [_sentence("s1", 1.0, 3.0), _sentence("s2", 3.0, 1.0)],
        max_auto_speed=1.2,
    )

    assert planned[0].speed_factor == pytest.approx(1.2)
    assert planned[0].effective_duration_seconds == pytest.approx(2.5)
    assert planned[0].remaining_overlap_seconds == pytest.approx(0.5)


def test_rows_without_available_chinese_tts_do_not_create_a_boundary() -> None:
    planned = plan_dubbing_timing(
        [
            _sentence("s1", 1.0, 2.5),
            _sentence("disabled", 2.0, 1.0, enabled=False),
            _sentence("empty", 2.5, 1.0, chinese=""),
            _sentence("missing", 3.0, None),
            _sentence("s2", 4.0, 1.0),
        ],
        max_auto_speed=1.5,
    )

    assert [item.sentence_id for item in planned] == ["s1", "s2"]
    assert planned[0].speed_factor == 1.0


def test_negative_offset_clamps_multiple_early_starts_and_uses_speed_limit() -> None:
    planned = plan_dubbing_timing(
        [_sentence("s1", 0.1, 1.0), _sentence("s2", 0.15, 1.0)],
        offset_ms=-200,
        max_auto_speed=1.4,
    )

    assert [item.start_seconds for item in planned] == [0.0, 0.0]
    assert planned[0].speed_factor == pytest.approx(1.4)
    assert planned[0].remaining_overlap_seconds == pytest.approx(1.0 / 1.4)


def test_transcript_export_records_effective_schedule_without_legacy_fields(tmp_path) -> None:
    project = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="a" * 64,
            duration_seconds=8.0,
            sample_rate=16_000,
            channels=1,
        ),
        settings=ProjectSettings(
            chinese_dubbing_offset_ms=500,
            chinese_max_auto_speed=1.5,
        ),
        sentences=[_sentence("s1", 1.0, 3.0), _sentence("s2", 3.0, 1.0)],
    )

    export_transcript(project, tmp_path)

    rows = json.loads((tmp_path / "exports" / "transcript.json").read_text(encoding="utf-8"))
    assert rows[0]["zh_start_seconds"] == pytest.approx(1.5)
    assert rows[0]["auto_speed_factor"] == pytest.approx(1.5)
    assert rows[0]["zh_effective_duration_seconds"] == pytest.approx(2.0)
    assert rows[0]["remaining_overlap_seconds"] == pytest.approx(0.0)
    assert "overlap_seconds" not in rows[0]
