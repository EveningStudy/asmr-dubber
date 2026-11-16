from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import asr, asr_candidates
from asmr_dubber.models import ProjectSettings
from asmr_dubber.segmentation import TimedToken, split_sound_runs, split_timed_tokens


def _chars(text: str, start: float = 1.0, step: float = 0.15) -> list[TimedToken]:
    tokens, time = [], start
    for char in text:
        tokens.append(TimedToken(char, time, time + step))
        time += step
    return tokens


def _texts(text: str) -> list[str]:
    return [s.source_text for s in split_timed_tokens(_chars(text), pause_seconds=0.7)]


def test_onomatopoeia_before_speech_becomes_its_own_line() -> None:
    sentences = split_timed_tokens(_chars("しこしこしこしこ、しこしこしこしこ、そうです、"))
    assert [s.source_text for s in sentences] == [
        "しこしこしこしこ、しこしこしこしこ、",
        "そうです、",
    ]
    assert sentences[0].end_seconds <= sentences[1].start_seconds
    assert [s.id for s in sentences] == ["s000001", "s000002"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("はぁ…気持ちいい", ["はぁ…", "気持ちいい"]),
        ("んっ、んっ、そこ、だめ", ["んっ、んっ、", "そこ、だめ"]),
        ("ねえ、ちゅぱちゅぱちゅぱ、好き", ["ねえ、", "ちゅぱちゅぱちゅぱ、", "好き"]),
    ],
)
def test_sound_runs_at_edges_or_repeated_inside_are_split(text, expected) -> None:
    assert _texts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ほらほら、ほーら",
        "そうそう、いいよ",
        "あなたのこと",
        "ちゅぱちゅぱちゅぱ",
        "ここ、ここだよ",
        "もっとトロトロなお顔になろうね",
        "ぬるぬるにしてあげる",
    ],
)
def test_spoken_doubles_plain_words_and_pure_sounds_stay_whole(text) -> None:
    assert _texts(text) == [text]


def test_short_sound_runs_are_not_worth_a_line() -> None:
    tokens = _chars("はぁ、そうです", step=0.05)  # はぁ lasts 0.1 s
    assert len(split_sound_runs(tokens)) == 1


def test_split_can_be_disabled() -> None:
    tokens = _chars("しこしこしこしこ、そうです")
    assert len(split_timed_tokens(tokens, split_sounds=False)) == 1


def test_anime_whisper_groups_vad_segments_into_short_clips() -> None:
    rate = 16000
    segments = [
        SimpleNamespace(start_sample=0, end_sample=rate),
        SimpleNamespace(start_sample=int(1.2 * rate), end_sample=2 * rate),  # 0.2 s gap: joined
        SimpleNamespace(start_sample=3 * rate, end_sample=4 * rate),  # 1 s gap: new group
        SimpleNamespace(start_sample=4 * rate, end_sample=20 * rate),  # would exceed 15 s
    ]
    assert asr_candidates._speech_groups(segments, rate) == [
        (0, 2 * rate),
        (3 * rate, 4 * rate),
        (4 * rate, 20 * rate),
    ]


def test_aligned_items_become_offset_tokens() -> None:
    items = [
        SimpleNamespace(text="そ", start_time=0.1, end_time=0.2),
        SimpleNamespace(text="う", start_time=0.2, end_time=float("nan")),
    ]
    assert asr_candidates._items_to_tokens(items, 5.0) == [TimedToken("そ", 5.1, 5.2)]


@pytest.mark.parametrize("backend", ["qwen3_asr", "anime_whisper"])
def test_new_backends_are_dispatched(monkeypatch, tmp_path, backend) -> None:
    audio = tmp_path / "speech.wav"
    sf.write(audio, np.full(16000, 0.01, dtype=np.float32), 16000)
    seen = {}

    def fake(path, settings, progress, *args, **kwargs):
        seen["path"] = path
        return [], "ja"

    monkeypatch.setattr(asr_candidates, f"transcribe_{backend}", fake)
    # anime-whisper runs the ASMR VAD itself, so the outer VAD must not condense the audio.
    settings = ProjectSettings(asr_backend=backend, asr_vad_mode="off")
    if backend == "anime_whisper":
        settings = settings.model_copy(update={"asr_vad_mode": "asmr"})
        monkeypatch.setattr(asr, "detect_asmr_speech", lambda *a, **k: pytest.fail("outer VAD"))
    asr.transcribe_source(audio, settings)
    assert seen["path"] == audio


def test_repeated_spoken_words_are_not_sounds() -> None:
    assert _texts("すきすきすき、くちゅくちゅくちゅ") == ["すきすきすき、", "くちゅくちゅくちゅ"]


def test_long_vad_clips_are_cut_at_their_quietest_moment() -> None:
    rate = 1000
    clip = np.ones(20 * rate, dtype=np.float32)
    clip[6 * rate : 6 * rate + 100] = 0  # a pause 6 s in
    pieces = asr_candidates._split_long_clip(clip, rate)
    assert pieces[0] == (0, 6 * rate + 50)
    assert all(b - a <= 8 * rate for a, b in pieces)
    assert pieces[-1][1] == len(clip)


def _voiced_clip(rate: int, seconds: float, voiced: tuple[float, float]) -> np.ndarray:
    clip = np.full(round(seconds * rate), 1e-4, dtype=np.float32)
    clip[round(voiced[0] * rate) : round(voiced[1] * rate)] = 0.1
    return clip


def test_collapsed_alignment_is_replaced_by_the_voiced_span() -> None:
    rate = 1000
    clip = _voiced_clip(rate, 8.0, (1.0, 7.0))  # voice from 1 s to 7 s of the clip
    squeezed = [
        SimpleNamespace(text=c, start_time=1.0 + i * 0.05, end_time=1.05 + i * 0.05)
        for i, c in enumerate("みあのおいる")
    ]
    tokens = asr_candidates._group_tokens(
        "みあのおいる", squeezed, (10 * rate, 18 * rate), rate, clip
    )
    assert tokens[0].start_seconds == pytest.approx(11.0, abs=0.05)
    assert tokens[-1].end_seconds == pytest.approx(17.0, abs=0.05)
    assert "".join(t.text for t in tokens) == "みあのおいる"


def test_plausible_alignment_is_kept() -> None:
    rate = 1000
    clip = _voiced_clip(rate, 4.0, (0.5, 3.5))
    items = [
        SimpleNamespace(text=c, start_time=0.5 + i, end_time=1.5 + i)
        for i, c in enumerate("そうで")
    ]
    tokens = asr_candidates._group_tokens("そうで", items, (0, 4 * rate), rate, clip)
    assert [t.start_seconds for t in tokens] == [0.5, 1.5, 2.5]


def test_asr_copy_follows_the_louder_ear() -> None:
    from asmr_dubber.audio import _fold_by_energy

    rate = 16000
    voice = np.sin(np.arange(rate) * 0.1).astype(np.float32) * 0.2
    one_sided = np.column_stack((voice * 0.01, voice))
    folded = _fold_by_energy(one_sided, 800)
    assert np.sqrt(np.mean(folded**2)) > 0.95 * np.sqrt(np.mean(voice**2))  # no -6 dB loss
    centred = np.column_stack((voice, voice))
    assert np.allclose(_fold_by_energy(centred, 800), voice, atol=1e-6)
