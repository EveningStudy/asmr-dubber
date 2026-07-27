import numpy as np
import pytest

from asmr_dubber.vad import (
    TimelinePiece,
    _speech_segments_from_probabilities,
    map_analysis_time,
)


def test_asmr_vad_hysteresis_and_padding() -> None:
    probabilities = np.zeros(100, dtype=np.float32)
    probabilities[10:30] = 0.9
    segments = _speech_segments_from_probabilities(
        probabilities,
        total_samples=100 * 320,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=100,
        speech_pad_ms=20,
    )

    assert len(segments) == 1
    assert segments[0].start_sample == 9 * 320
    assert segments[0].end_sample == 31 * 320


def test_condensed_timeline_maps_speech_and_separator_to_original_time() -> None:
    timeline = [
        TimelinePiece(0.0, 1.0, 10.0, 11.0),
        TimelinePiece(1.7, 2.7, 20.0, 21.0),
    ]

    assert map_analysis_time(0.25, timeline, end=False) == pytest.approx(10.25)
    assert map_analysis_time(1.4, timeline, end=True) == pytest.approx(11.0)
    assert map_analysis_time(1.4, timeline, end=False) == pytest.approx(20.0)
    assert map_analysis_time(2.5, timeline, end=True) == pytest.approx(20.8)
