"""Randomised property tests for plan_dubbing_timing (stdlib random, no hypothesis)."""

import random
from itertools import pairwise

import pytest

from asmr_dubber.constants import MAX_CHINESE_AUTO_SPEED
from asmr_dubber.models import Sentence
from asmr_dubber.timing import plan_dubbing_timing

MODES = ["fit_window", "sequential", "fit_then_shift"]
EPSILON = 1e-9


def _random_case(seed: int):
    rng = random.Random(seed)
    sentences, durations = [], {}
    cursor = 0.0
    for index in range(rng.randint(0, 40)):
        # Gaps may be zero and lines may overlap or arrive unsorted, like ASR output.
        start = max(0.0, cursor + rng.choice([-0.5, 0.0, 0.0, rng.uniform(0.0, 4.0)]))
        end = start + rng.uniform(0.05, 6.0)
        cursor = start if rng.random() < 0.2 else end
        enabled = rng.random() > 0.1
        text = "好" if rng.random() > 0.1 else ""
        sentence = Sentence(
            id=f"s{index:06d}", start_seconds=start, end_seconds=end, source_text="x", zh_text=text
        )
        sentence.enabled = enabled
        sentences.append(sentence)
        if rng.random() > 0.1:
            durations[sentence.id] = rng.choice([rng.uniform(0.01, 8.0), 0.01, 12.0])
    rng.shuffle(sentences)
    return rng, sentences, durations


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(60))
def test_random_plans_have_sane_times(mode, seed):
    rng, sentences, durations = _random_case(seed)
    offset = rng.choice([0, 0, -300, 250, -5000])
    limit = rng.choice([1.0, 1.1, 1.2, MAX_CHINESE_AUTO_SPEED])
    plan = plan_dubbing_timing(
        sentences, offset_ms=offset, max_auto_speed=limit, durations=durations, mode=mode
    )
    expected = {
        s.id for s in sentences if s.enabled and s.zh_text and durations.get(s.id, 0.0) > 0.0
    }
    assert {item.sentence_id for item in plan} == expected
    for item in plan:
        assert item.start_seconds >= 0.0
        assert item.effective_duration_seconds > 0.0
        assert item.original_duration_seconds > 0.0
        assert 1.0 - EPSILON <= item.speed_factor <= limit + EPSILON
        assert item.remaining_overlap_seconds >= 0.0
        assert item.effective_duration_seconds <= item.original_duration_seconds + EPSILON
    # Starts follow the source order.
    assert [i.start_seconds for i in plan] == sorted(i.start_seconds for i in plan)


@pytest.mark.parametrize("mode", ["sequential", "fit_then_shift"])
@pytest.mark.parametrize("seed", range(60))
def test_sequential_modes_never_overlap(mode, seed):
    rng, sentences, durations = _random_case(seed + 1000)
    limit = rng.choice([1.0, 1.2, MAX_CHINESE_AUTO_SPEED])
    plan = plan_dubbing_timing(sentences, max_auto_speed=limit, durations=durations, mode=mode)
    for previous, current in pairwise(plan):
        end = previous.start_seconds + previous.effective_duration_seconds
        assert end <= current.start_seconds + EPSILON


@pytest.mark.parametrize("seed", range(60))
def test_fit_then_shift_respects_speed_limit_and_never_starts_early(seed):
    rng, sentences, durations = _random_case(seed + 2000)
    limit = rng.choice([1.0, 1.15, 1.2, MAX_CHINESE_AUTO_SPEED])
    plan = plan_dubbing_timing(
        sentences, max_auto_speed=limit, durations=durations, mode="fit_then_shift"
    )
    by_id = {s.id: s for s in sentences}
    for item in plan:
        assert item.speed_factor <= limit + EPSILON
        # A line is only ever delayed, never moved ahead of its own source start.
        assert item.start_seconds >= by_id[item.sentence_id].start_seconds - EPSILON


def test_speed_limit_outside_the_allowed_range_is_rejected():
    with pytest.raises(ValueError):
        plan_dubbing_timing([], max_auto_speed=MAX_CHINESE_AUTO_SPEED + 0.5)
    with pytest.raises(ValueError):
        plan_dubbing_timing([], mode="nope")
