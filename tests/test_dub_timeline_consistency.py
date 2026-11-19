"""The fully dubbed mix, dubbing-timeline subtitles and timing table agree, and the
last line stays inside the recording when it can."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber.audio import probe_audio, sentence_events, sha256_file
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, save_project
from asmr_dubber.timing import plan_dubbing_timing
from asmr_dubber.tts import tts_cache_key

RATE = 16_000


def _line(sentence_id: str, start: float, end: float, seconds: float) -> Sentence:
    return Sentence(
        id=sentence_id,
        start_seconds=start,
        end_seconds=end,
        source_text="セリフ",
        zh_text="台词",
        tts_duration_seconds=seconds,
    )


@pytest.mark.parametrize("mode", ["fit_then_shift", "fit_window"])
def test_last_line_speeds_up_to_end_with_the_recording(mode: str) -> None:
    lines = [_line("a", 0.0, 1.0, 1.0), _line("b", 2.0, 3.0, 2.4)]
    loose = plan_dubbing_timing(lines, max_auto_speed=1.25, mode=mode)
    assert loose[-1].speed_factor == 1.0  # without the end it plays at its natural pace

    fitted = plan_dubbing_timing(lines, max_auto_speed=1.25, mode=mode, end_seconds=4.0)
    assert fitted[-1].speed_factor == pytest.approx(1.2)
    assert fitted[-1].start_seconds + fitted[-1].effective_duration_seconds == pytest.approx(4.0)

    capped = plan_dubbing_timing(lines, max_auto_speed=1.25, mode=mode, end_seconds=3.5)
    assert capped[-1].speed_factor == pytest.approx(1.25)  # never above the user's limit
    assert capped[-1].start_seconds + capped[-1].effective_duration_seconds > 3.5


def _tone(path: Path, seconds: float, channels: int = 1) -> Path:
    t = np.arange(round(RATE * seconds)) / RATE
    wave = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    sf.write(path, np.column_stack([wave] * channels) if channels > 1 else wave, RATE)
    return path


def test_replacement_subtitles_and_table_follow_the_mix_plan(tmp_path: Path) -> None:
    source = _tone(tmp_path / "source.wav", 4.0, channels=2)
    project = DubProject(
        source=probe_audio(source, sha256=sha256_file(source)),
        settings=ProjectSettings(
            tts_backend="edge_tts",
            separation_enabled=True,
            separation_mix_mode="replace",
            subtitle_timeline="dubbing",
            subtitle_min_duration_seconds=0.2,
            subtitle_max_cps=40.0,
            replacement_dubbing_offset_ms=0,
            replacement_max_auto_speed=1.25,
            replacement_timing_mode="fit_then_shift",
        ),
    )
    (tmp_path / "chinese").mkdir()
    _tone(tmp_path / "chinese" / "first.wav", 1.0)
    _tone(tmp_path / "chinese" / "last.wav", 1.5)
    first = _line("s1", 0.2, 1.0, 1.0)
    first.tts_file = "chinese/first.wav"
    kept = Sentence(  # an untranslated breath whose original voice stays in the mix
        id="s2", start_seconds=1.1, end_seconds=1.7, source_text="はぁ", status="skipped_sound"
    )
    last = _line("s3", 2.6, 3.4, 1.5)
    last.tts_file = "chinese/last.wav"
    project.sentences = [first, kept, last]
    for sentence in (first, last):
        sentence.tts_cache_key = tts_cache_key(project, sentence)
    save_project(project, tmp_path)

    s = project.settings
    events = {  # exactly how the fully dubbed mix plans
        event.sentence_id: event
        for event in sentence_events(
            tmp_path,
            project.sentences,
            s.replacement_dubbing_offset_ms,
            s.replacement_max_auto_speed,
            s.replacement_timing_mode,
            trim_silence=True,
            source_duration_seconds=project.source.duration_seconds,
        )
    }
    # The kept breath stops the first line early; the last line ends with the recording.
    assert events["s1"].start_seconds + events["s1"].effective_duration_seconds <= 1.1 + 1e-6
    assert events["s3"].start_seconds + events["s3"].effective_duration_seconds == pytest.approx(
        4.0, abs=0.02
    )

    pipeline.export_transcript(project, tmp_path)
    rows = {
        row["id"]: row
        for row in json.loads((tmp_path / "exports" / "transcript.json").read_text("utf-8"))
    }
    for sentence_id, event in events.items():
        assert rows[sentence_id]["zh_start_seconds"] == pytest.approx(event.start_seconds)
        assert rows[sentence_id]["auto_speed_factor"] == pytest.approx(event.speed_factor)

    srt, _, _ = pipeline.generate_subtitles(project, tmp_path, language="zh")
    starts = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt.read_text(encoding="utf-8"))
    seconds = [int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000 for h, m, sec, ms in starts]
    for event in events.values():  # the kept breath also gets a (source-text) cue
        assert any(abs(start - event.start_seconds) < 0.002 for start in seconds), event
