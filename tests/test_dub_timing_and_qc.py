import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import tts
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, load_project, save_project
from asmr_dubber.timing import plan_dubbing_timing
from asmr_dubber.tts import dub_problem, normalize_pauses, spoken_text, tts_cache_key


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("身体的最深处都是……米娅的精油……", "身体的最深处都是，米娅的精油"),
        ("可算不上惩罚哦……对吧？", "可算不上惩罚哦，对吧？"),
        ("嘿……来……", "嘿，来"),
        ("就这样。。。好吗", "就这样。。。好吗"),
        ("好——吧", "好，吧"),
        ("嗯——", "嗯"),
        ("……", "……"),
    ],
)
def test_long_pause_punctuation_becomes_commas(text, expected) -> None:
    assert normalize_pauses(text) == expected
    assert normalize_pauses(expected) == expected


def test_spoken_text_follows_the_setting() -> None:
    assert spoken_text(ProjectSettings(tts_normalize_pauses=False), "嘿……") == "嘿……"
    assert spoken_text(ProjectSettings(), "嘿……") == "嘿"


def _line(index: int, start: float, end: float) -> Sentence:
    return Sentence(
        id=f"s{index:06d}", start_seconds=start, end_seconds=end, source_text="x", zh_text="好"
    )


def test_fit_then_shift_does_not_let_one_long_line_delay_the_rest() -> None:
    lines = [_line(1, 0.0, 1.0), _line(2, 1.0, 2.0), _line(3, 5.0, 6.0), _line(4, 6.0, 7.0)]
    durations = {"s000001": 2.0, "s000002": 1.0, "s000003": 1.0, "s000004": 1.0}
    plan = plan_dubbing_timing(
        lines, max_auto_speed=1.2, durations=durations, mode="fit_then_shift"
    )
    starts = [round(item.start_seconds, 3) for item in plan]
    # Line 1 speeds up to 1.2x (1.667 s) and still overruns line 2's start: line 2 waits.
    assert plan[0].speed_factor == pytest.approx(1.2)
    assert starts[1] == pytest.approx(2.0 / 1.2, abs=1e-3)
    # The gap before line 3 absorbs the delay: line 3 is back on its own start.
    assert starts[2:] == [5.0, 6.0]
    for previous, current in pairwise(plan):
        assert (
            previous.start_seconds + previous.effective_duration_seconds
            <= current.start_seconds + 1e-9
        )

    sequential = plan_dubbing_timing(
        lines, max_auto_speed=1.2, durations=durations, mode="sequential"
    )
    assert sequential[1].start_seconds == pytest.approx(2.0)  # no speed-up, full delay


def _wav(path: Path, seconds: float, level: float = 0.1, rate: int = 16000) -> Path:
    samples = np.zeros(int(rate * (seconds + 0.4)), dtype=np.float32)
    voiced = np.sin(np.arange(int(rate * seconds)) * 0.05).astype(np.float32) * level
    samples[int(rate * 0.2) : int(rate * 0.2) + len(voiced)] = voiced
    sf.write(path, samples, rate)
    return path


def test_dub_problem_catches_gross_failures(tmp_path) -> None:
    assert dub_problem(_wav(tmp_path / "ok.wav", 1.2), "好治愈了呢") is None
    assert "没有声音" in dub_problem(_wav(tmp_path / "silent.wav", 1.0, level=0.0), "好治愈了呢")
    assert "没有声音" in dub_problem(_wav(tmp_path / "blip.wav", 0.03), "浸透了")
    assert "短太多" in dub_problem(_wav(tmp_path / "short.wav", 0.4), "只是舒服的话可算不上惩罚哦")
    assert "长太多" in dub_problem(_wav(tmp_path / "long.wav", 9.0), "来")
    clipped = tmp_path / "clipped.wav"
    sf.write(clipped, np.ones(16000, dtype=np.float32), 16000)
    assert "削波" in dub_problem(clipped, "好治愈了呢")


def test_broken_dubs_are_regenerated_with_a_new_seed_then_flagged(monkeypatch, tmp_path) -> None:
    project = DubProject(
        source={
            "path": "source.wav",
            "sha256": "0" * 64,
            "duration_seconds": 10.0,
            "sample_rate": 16000,
            "channels": 1,
        },
        sentences=[
            Sentence(
                id="good", start_seconds=0, end_seconds=1, source_text="a", zh_text="好治愈了呢"
            ),
            Sentence(
                id="flaky", start_seconds=2, end_seconds=3, source_text="b", zh_text="被治愈了呢"
            ),
            Sentence(
                id="broken", start_seconds=4, end_seconds=5, source_text="c", zh_text="浸透了呢"
            ),
        ],
    )
    calls = []

    def fake_backend(project, project_dir, source, force=False, sentence_ids=None, **kwargs):
        calls.append((force, sentence_ids))
        for sentence in project.sentences:
            if sentence_ids is not None and sentence.id not in sentence_ids:
                continue
            ok = sentence.id == "good" or (sentence.id == "flaky" and sentence.tts_attempt >= 1)
            name = f"{sentence.id}-{sentence.tts_attempt}.wav"
            _wav(project_dir / name, 1.2 if ok else 0.03)
            sentence.tts_file = name
        return []

    monkeypatch.setattr("asmr_dubber.tts_backends.synthesize_with_selected_backend", fake_backend)
    tts.synthesize_sentences(project, tmp_path, tmp_path / "source.wav")
    by_id = {s.id: s for s in project.sentences}
    assert by_id["good"].tts_attempt == 0 and by_id["good"].tts_warning is None
    assert by_id["flaky"].tts_attempt == 1 and by_id["flaky"].tts_warning is None
    assert by_id["broken"].tts_attempt == 2 and "没有声音" in by_id["broken"].tts_warning
    assert calls == [(False, None), (True, ["flaky", "broken"]), (True, ["broken"])]


def test_when_every_retry_fails_the_least_wrong_take_is_kept(monkeypatch, tmp_path) -> None:
    project = DubProject(
        source={
            "path": "source.wav",
            "sha256": "0" * 64,
            "duration_seconds": 10.0,
            "sample_rate": 16000,
            "channels": 1,
        },
        sentences=[
            Sentence(
                id="hard", start_seconds=0, end_seconds=1, source_text="a", zh_text="被治愈了哦"
            ),
        ],
    )
    project.settings.tts_asr_quality_check = True
    error_by_attempt = {0: 0.5, 1: 0.2, 2: 0.6}

    def fake_backend(project, project_dir, source, force=False, sentence_ids=None, **kwargs):
        for sentence in project.sentences:
            name = f"{sentence.id}-{sentence.tts_attempt}.wav"
            _wav(project_dir / name, 1.2)
            sentence.tts_file = name
            sentence.tts_cache_key = f"key-{sentence.tts_attempt}"
            sentence.tts_duration_seconds = 1.2 + sentence.tts_attempt
        return []

    def fake_check(path, expected, settings):
        cer = error_by_attempt[int(Path(path).stem.rsplit("-", 1)[1])]
        return f"配音回听字错率 {cer * 100:.0f}%", cer

    monkeypatch.setattr("asmr_dubber.tts_backends.synthesize_with_selected_backend", fake_backend)
    monkeypatch.setattr("asmr_dubber.dub_qc.dub_asr_check", fake_check)
    tts.synthesize_sentences(project, tmp_path, tmp_path / "source.wav")
    hard = project.sentences[0]
    assert (hard.tts_attempt, hard.tts_file, hard.tts_cache_key) == (1, "hard-1.wav", "key-1")
    assert hard.tts_duration_seconds == 2.2
    assert hard.tts_warning == "配音回听字错率 20%"


def test_retry_attempt_changes_the_cache_key_only_after_a_retry() -> None:
    project = DubProject(
        source={
            "path": "s.wav",
            "sha256": "0" * 64,
            "duration_seconds": 5.0,
            "sample_rate": 16000,
            "channels": 1,
        },
        sentences=[Sentence(id="a", start_seconds=0, end_seconds=1, source_text="a", zh_text="好")],
    )
    first = tts_cache_key(project, project.sentences[0])
    project.sentences[0].tts_attempt = 1
    assert tts_cache_key(project, project.sentences[0]) != first


def test_projects_written_before_2_3_keep_their_dubbing_behaviour(tmp_path) -> None:
    project = DubProject(
        source={
            "path": "s.wav",
            "sha256": "0" * 64,
            "duration_seconds": 5.0,
            "sample_rate": 16000,
            "channels": 1,
        },
    )
    save_project(project, tmp_path)
    manifest = tmp_path / "project.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("reference_audio_mode", "tts_normalize_pauses"):
        data["settings"].pop(key)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    old, _ = load_project(tmp_path)
    assert old.settings.reference_audio_mode == "legacy"
    assert old.settings.tts_normalize_pauses is False
    # Full-band references made IndexTTS crackle; new projects use 16 kHz ones again.
    assert ProjectSettings().reference_audio_mode == "legacy"
    assert ProjectSettings().tts_normalize_pauses is True


def test_reference_pick_prefers_a_clear_voiced_line_over_the_longest(tmp_path) -> None:
    from asmr_dubber.voice_reference import pick_reference_sentence

    rate = 16000
    audio = np.random.default_rng(0).normal(0, 0.002, rate * 40).astype(np.float32)
    t = np.arange(rate * 8) / rate
    audio[rate * 1 : rate * 9] += (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)  # voiced
    noise = np.random.default_rng(1).normal(0, 0.01, rate * 18).astype(np.float32)
    audio[rate * 12 : rate * 30] += noise  # long, quiet, breathy
    source = tmp_path / "source.wav"
    sf.write(source, audio, rate)
    project = DubProject(
        source={
            "path": "source.wav",
            "sha256": "0" * 64,
            "duration_seconds": 40.0,
            "sample_rate": rate,
            "channels": 1,
        },
        sentences=[
            Sentence(id="clear", start_seconds=1, end_seconds=9, source_text="はっきり"),
            Sentence(id="whisper", start_seconds=12, end_seconds=30, source_text="ささやき"),
        ],
    )
    assert pick_reference_sentence(project, source).id == "clear"


def test_reference_is_fixed_only_before_the_first_dub(monkeypatch, tmp_path) -> None:
    project = DubProject(
        source={
            "path": "s.wav",
            "sha256": "0" * 64,
            "duration_seconds": 5.0,
            "sample_rate": 16000,
            "channels": 1,
        },
        sentences=[
            Sentence(
                id="a",
                start_seconds=0,
                end_seconds=3,
                source_text="あ",
                zh_text="好",
                tts_file="a.wav",
            )
        ],
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.synthesize_with_selected_backend", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "asmr_dubber.voice_reference.pick_reference_sentence", lambda *a: pytest.fail("picked")
    )
    tts.synthesize_sentences(
        project.model_copy(update={"settings": ProjectSettings(tts_quality_check=False)}),
        tmp_path,
        tmp_path / "s.wav",
    )
