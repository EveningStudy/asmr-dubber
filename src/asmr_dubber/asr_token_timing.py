"""Character times for clip-wise recognition, checked against where the voice is.

The Qwen3 ForcedAligner times every character of a recognised clip, but on
whispered ASMR it sometimes squeezes a line into part of the voice.  Plausible
alignments are kept, with collapsed vowel tails stretched over the voiced
frames; otherwise the text is spread evenly over the voiced time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .segmentation import TimedToken


def items_to_tokens(items: Any, offset: float) -> list[TimedToken]:
    from .asr import _offset_finite_token

    tokens: list[TimedToken] = []
    for item in items or []:
        token = _offset_finite_token(
            getattr(item, "text", None),
            getattr(item, "start_time", None),
            getattr(item, "end_time", None),
            offset,
        )
        if token is not None:
            tokens.append(token)
    return tokens


_ACTIVE_FRAME_SECONDS = 0.02
_ACTIVE_DROP = 0.032  # frames within ~30 dB of the clip's 98th-percentile frame count as voice
_MIN_ALIGNED_SHARE = 0.45
_MAX_EDGE_MISS_SECONDS = 0.45


def active_frames(clip: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    hop = max(1, round(sample_rate * _ACTIVE_FRAME_SECONDS))
    count = len(clip) // hop
    if count == 0:
        return np.zeros(0, dtype=bool), hop / sample_rate
    rms = np.sqrt(np.mean(clip[: count * hop].astype(np.float64).reshape(count, hop) ** 2, axis=1))
    ref = float(np.percentile(rms, 98)) if len(rms) >= 4 else float(rms.max())
    floor = float(np.percentile(rms, 10)) * 2.2 if len(rms) >= 8 else 1e-6
    return rms >= max(ref * _ACTIVE_DROP, floor, 1e-6), hop / sample_rate


def _alignment_is_plausible(
    aligned: list[TimedToken], active: np.ndarray, hop_seconds: float, offset: float
) -> bool:
    """Reject alignments that squeeze the text into part of the voice or miss its edges.

    On whispered ASMR the aligner sometimes packs a whole line into the first
    second of a clip, or into 0.1 s.  The rest of the spoken line then falls
    outside every sentence, and a full dub leaves that Japanese untouched.
    """
    if not active.any():
        return True
    indices = np.flatnonzero(active)
    voice_start = offset + indices[0] * hop_seconds
    voice_end = offset + (indices[-1] + 1) * hop_seconds
    voice_seconds = active.sum() * hop_seconds
    start, end = aligned[0].start_seconds, aligned[-1].end_seconds
    covered = (
        active[
            max(0, int((start - offset) / hop_seconds)) : max(
                0, int(np.ceil((end - offset) / hop_seconds))
            )
        ].sum()
        * hop_seconds
    )
    if covered < _MIN_ALIGNED_SHARE * voice_seconds:
        return False
    return (
        start - voice_start <= _MAX_EDGE_MISS_SECONDS and voice_end - end <= _MAX_EDGE_MISS_SECONDS
    )


def _spread_over_voice(
    text: str, active: np.ndarray, hop_seconds: float, offset: float, end: float
) -> list[TimedToken]:
    """Give each character an equal share of the clip's voiced time, in order."""
    chars = [char for char in text if not char.isspace()]
    indices = np.flatnonzero(active)
    if not chars or not len(indices):
        return [TimedToken(text, offset, end)]
    times = offset + indices * hop_seconds
    tokens = []
    for n, char in enumerate(chars):
        a = int(n * len(indices) / len(chars))
        b = max(a + 1, int((n + 1) * len(indices) / len(chars)))
        tokens.append(TimedToken(char, float(times[a]), float(times[b - 1] + hop_seconds)))
    return tokens


def _extend_tokens_over_active_voice(
    aligned: list[TimedToken],
    active: np.ndarray,
    hop_seconds: float,
    offset: float,
    end: float,
) -> list[TimedToken]:
    """Bridge whispered vowel tails that ForcedAligner collapses to point-onsets."""
    if not aligned or not active.any():
        return aligned
    indices = np.flatnonzero(active)
    voice_start = max(offset, float(offset + indices[0] * hop_seconds))
    voice_end = min(end, float(offset + (indices[-1] + 1) * hop_seconds))
    out = list(aligned)
    if out[0].start_seconds > voice_start:
        out[0] = TimedToken(
            out[0].text,
            max(voice_start, out[0].start_seconds - 1.2),
            max(out[0].end_seconds, voice_start + 0.02),
        )
    # Bridge intra-group active-voice tails between consecutive tokens so sustained
    # whispered vowels ("メアァ…の") never look like multi-second silences.
    for idx in range(len(out) - 1):
        cur = out[idx]
        nxt = out[idx + 1]
        if nxt.start_seconds <= cur.end_seconds + hop_seconds:
            continue
        f_lo = max(0, int((cur.end_seconds - offset) / hop_seconds))
        f_hi = min(len(active), int(np.ceil((nxt.start_seconds - offset) / hop_seconds)))
        if f_hi > f_lo and float(np.mean(active[f_lo:f_hi])) >= 0.45:
            bridged_end = nxt.start_seconds - 0.01
            if bridged_end > cur.end_seconds:
                out[idx] = TimedToken(cur.text, cur.start_seconds, bridged_end)
        else:
            # Extend across contiguous trailing voice up to a 160 ms silence gap.
            quiet = 0
            tail = cur.end_seconds
            max_quiet = max(1, round(0.16 / hop_seconds))
            for f in range(f_lo, f_hi):
                if active[f]:
                    quiet = 0
                    tail = min(nxt.start_seconds - 0.01, offset + (f + 1) * hop_seconds)
                else:
                    quiet += 1
                    if quiet > max_quiet:
                        break
            if tail > cur.end_seconds:
                out[idx] = TimedToken(cur.text, cur.start_seconds, tail)
    if out[-1].end_seconds < voice_end:
        out[-1] = TimedToken(
            out[-1].text,
            out[-1].start_seconds,
            min(voice_end, out[-1].end_seconds + 2.2),
        )
    return out


def group_tokens(
    text: str,
    items: Any,
    group: tuple[int, int],
    sample_rate: int,
    clip: np.ndarray | None = None,
) -> list[TimedToken]:
    """Aligned characters of one clip, checked against where the voice actually is."""
    offset = group[0] / sample_rate
    end = group[1] / sample_rate
    aligned = items_to_tokens(items, offset)
    if clip is None:
        return aligned or [TimedToken(text, offset, end)]
    active, hop_seconds = active_frames(clip, sample_rate)
    if aligned and _alignment_is_plausible(aligned, active, hop_seconds, offset):
        return _extend_tokens_over_active_voice(aligned, active, hop_seconds, offset, end)
    return _spread_over_voice(text, active, hop_seconds, offset, end)
