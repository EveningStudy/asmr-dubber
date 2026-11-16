"""Experimental RTF rendering and explicit residual/original-voice composition."""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .audio import _run_ffmpeg
from .errors import ProjectError
from .filtering import has_speakable_text
from .models import DubProject, ProjectSettings
from .spatial_cues import rms
from .spatial_rtf import render_rtf
from .storage import atomic_write_text
from .task_control import check_cancelled

logger = logging.getLogger(__name__)


def _short_reference_fallback(mono, reference, settings, sentence_id):
    """Preserve the utterance without inventing phase from insufficient evidence."""
    ratio = 1.0
    if reference.size and rms(reference) >= 1e-7:
        levels = np.sqrt(np.mean(reference.astype(np.float64) ** 2, axis=0))
        ratio = float(np.clip(levels[1] / max(levels[0], 1e-8), 1 / 16, 16))
    gains = np.array([1, ratio]) * np.sqrt(2 / (1 + ratio**2))
    strength = settings.spatial_rtf_strength
    result = mono[:, None] * (gains * strength + (1 - strength))
    peak = float(np.max(np.abs(result), initial=0))
    ceiling = 10 ** (settings.chinese_line_peak_dbfs / 20)
    if peak > ceiling:
        result *= ceiling / peak
    logger.warning(
        "RTF 降级：%s 参考过短或无有效信号；保留中文、时长和位置，"
        "仅使用有限左右电平差（无信号时居中），不估计相位。",
        sentence_id,
    )
    return result.astype(np.float32)


def stereo_reference(source: Path, directory: Path) -> Path:
    destination = directory / "analysis/spatial-reference.wav"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{uuid.uuid4().hex}.wav")
        try:
            _run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-c:a",
                    "pcm_f32le",
                    "-rf64",
                    "auto",
                    str(temporary),
                ]
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def vocal_weighted_reference(source: Path, vocals: Path, directory: Path) -> Path:
    """Spatial reference that keeps the mix's true ear-to-ear phase but only where the voice is.

    Separation models do not promise to preserve binaural cues, so the separated
    voice is used only as a weight: a real time-frequency mask
    ``voice / (voice + rest)`` per ear is applied to the original mix.  The phase
    at each ear stays exactly that of the recording, while effects, music, hiss and
    whatever else is happening at the *other* ear stop steering the estimate.
    Binaural ASMR routinely has the voice at one ear and a different sound at the
    other, which a mask shared by both ears cannot tell apart.
    """
    from .audio import sha256_file
    from .spatial_rtf import istft, stft

    base = stereo_reference(source, directory)
    key = f"vocal-weighted-v2|{sha256_file(base)}|{sha256_file(vocals)}"
    destination = directory / "analysis/spatial-reference-vocal.wav"
    receipt = destination.with_suffix(".key")
    if destination.is_file() and receipt.is_file() and receipt.read_text("utf-8") == key:
        return destination
    token = uuid.uuid4().hex
    temporary = destination.with_name(f".{token}.wav")
    voice_file = destination.with_name(f".{token}.vocals.wav")
    size, hop = 2048, 512
    try:
        with sf.SoundFile(base) as mix:
            rate, frames = mix.samplerate, mix.frames
            _resample_file(vocals, voice_file, rate, frames)
            with (
                sf.SoundFile(voice_file) as voice,
                sf.SoundFile(
                    temporary, "w", samplerate=rate, channels=2, format="RF64", subtype="FLOAT"
                ) as out,
            ):
                block, pad = rate * 20, rate // 2
                for start in range(0, frames, block):
                    check_cancelled()
                    lo, hi = max(0, start - pad), min(frames, start + block + pad)
                    mix.seek(lo)
                    voice.seek(lo)
                    original = mix.read(hi - lo, dtype="float32", always_2d=True)
                    estimate = voice.read(hi - lo, dtype="float32", always_2d=True)
                    if estimate.shape[1] == 1:
                        estimate = np.repeat(estimate, 2, axis=1)
                    spectrum = stft(original.astype(np.float64), size, hop)
                    voiced = stft(estimate.astype(np.float64), size, hop)
                    voice_power = np.abs(voiced) ** 2
                    rest_power = np.abs(spectrum - voiced) ** 2
                    mask = voice_power / np.maximum(voice_power + rest_power, 1e-14)
                    weighted = istft(spectrum * mask, size, hop, len(original))
                    keep = slice(start - lo, min(frames, start + block) - lo)
                    out.write(weighted[keep].astype(np.float32))
        temporary.replace(destination)
        atomic_write_text(receipt, key)
    finally:
        temporary.unlink(missing_ok=True)
        voice_file.unlink(missing_ok=True)
    return destination


def spatialize_clip(
    mono: np.ndarray, rate: int, event, reference: Path, settings: ProjectSettings
) -> np.ndarray:
    """Second-version RTF in bounded overlapping blocks, after tempo/loudness processing."""
    if not np.isfinite(mono).all():
        raise ProjectError(f"{event.sentence_id} 中文音频包含非有限采样。")
    if not len(mono) or rms(mono) < 1e-8:
        return np.column_stack((mono, mono))
    with sf.SoundFile(reference) as handle:
        sr = handle.samplerate
        if handle.channels != 2:
            raise ProjectError("RTF 参考必须是双声道。")
        lo = min(handle.frames, max(0, round(event.source_start_seconds * sr)))
        hi = max(lo, min(handle.frames, round(event.source_end_seconds * sr)))
        if hi - lo < sr // 10:
            handle.seek(lo)
            ref = handle.read(hi - lo, dtype="float32", always_2d=True)
            if not np.isfinite(ref).all():
                raise ProjectError(f"{event.sentence_id} 原声参考包含非有限采样。")
            return _short_reference_fallback(mono, ref, settings, event.sentence_id)
        chunk = max(1, round(settings.spatial_rtf_block_seconds * rate))
        pad = min(round(rate * 0.2), chunk // 4)
        result = np.zeros((len(mono), 2), dtype=np.float32)
        weights = np.zeros(len(mono), dtype=np.float32)
        for start in range(0, len(mono), chunk):
            check_cancelled()
            end = min(len(mono), start + chunk)
            a, b = max(0, start - pad), min(len(mono), end + pad)
            ref_start = lo + round((hi - lo) * a / len(mono))
            ref_end = lo + round((hi - lo) * b / len(mono))
            handle.seek(ref_start)
            ref = handle.read(max(1, ref_end - ref_start), dtype="float32", always_2d=True)
            if not np.isfinite(ref).all():
                raise ProjectError(f"{event.sentence_id} 原声参考包含非有限采样。")
            if sr != rate:
                ref = soxr.resample(ref, sr, rate)
            try:
                if len(ref) < rate // 10 or rms(ref) < 1e-7:
                    audio = _short_reference_fallback(
                        mono[a:b],
                        ref,
                        settings.model_copy(update={"spatial_rtf_strength": 1.0}),
                        event.sentence_id,
                    )
                elif rms(mono[a:b]) < 1e-8:
                    audio = np.column_stack((mono[a:b], mono[a:b]))
                else:
                    audio, _ = render_rtf(
                        mono[a:b],
                        ref,
                        rate,
                        level_strength=settings.spatial_rtf_level_strength,
                        color_strength=settings.spatial_rtf_color_strength,
                        fft_size=settings.spatial_rtf_fft_size,
                        hop_divisor=settings.spatial_rtf_hop_divisor,
                    )
            except ValueError as exc:
                # Quiet ASMR phrases can pass the coarse RMS check but still
                # contain no frame that is loud enough for reliable spatial
                # cue estimation.  RTF is an enhancement, so one quiet
                # reference block must not fail the whole batch item.  Keep
                # the Chinese clip and apply only the safe level fallback.
                if "参考片段电平过低" in str(exc):
                    audio = _short_reference_fallback(
                        mono[a:b],
                        ref,
                        settings.model_copy(update={"spatial_rtf_strength": 1.0}),
                        event.sentence_id,
                    )
                else:
                    raise ProjectError(f"{event.sentence_id} RTF 失败：{exc}") from exc
            # Leave overall loudness to the application's existing loudness controls.
            audio *= rms(mono[a:b]) / max(rms(audio), 1e-8)
            weight = np.ones(b - a, dtype=np.float32)
            if a < start:
                weight[: start - a] = np.linspace(0, 1, start - a)
            if b > end:
                weight[end - a :] = np.linspace(1, 0, b - end)
            result[a:b] += audio * weight[:, None]
            weights[a:b] += weight
        result /= np.maximum(weights[:, None], 1e-8)
    strength = settings.spatial_rtf_strength
    result = result * strength + mono[:, None] * (1 - strength)
    peak = float(np.max(np.abs(result)))
    ceiling = 10 ** (settings.chinese_line_peak_dbfs / 20)
    if peak > ceiling:
        result *= ceiling / peak
    return result


def compose_replacement_bed(
    project: DubProject, directory: Path, vocals: Path, background: Path
) -> Path:
    """Retain only explicit/undubbed vocal windows, never duplicate the background."""
    settings = project.settings
    rows = {s.id: s for s in project.sentences}
    ids = set(re.split(r"[\s,，;；]+", settings.separation_keep_ids.strip())) - {""}
    if settings.separation_keep_original == "manual" and ids - rows.keys():
        raise ProjectError("原声回填 ID 不存在：" + ", ".join(sorted(ids - rows.keys())))
    bilingual = settings.separation_mix_mode == "bilingual"
    selected = [
        s
        for s in project.sentences
        if not bilingual
        and (
            (settings.separation_keep_original == "manual" and s.id in ids)
            or (
                settings.separation_keep_original == "unvoiced"
                and (
                    not s.enabled
                    or not s.chinese_audio_enabled
                    or not has_speakable_text(s.zh_text)
                )
            )
        )
    ]
    rate = sf.info(background).samplerate
    padding = settings.separation_keep_padding_ms / 1000
    ranges = sorted(
        (max(0, round((s.start_seconds - padding) * rate)), round((s.end_seconds + padding) * rate))
        for s in selected
    )
    merged: list[list[int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    target = directory / "mix/replacement-background.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{uuid.uuid4().hex}.wav")
    fade = max(1, round(rate * 0.01))
    selected_ids = {s.id for s in selected}
    overrides = sorted(
        (
            s
            for s in project.sentences
            if s.original_audio_enabled is not None or s.original_audio_gain_db != 0
        ),
        key=lambda s: (s.original_audio_enabled is False, s.start_seconds, s.id),
    )
    try:
        with (
            sf.SoundFile(background) as bed,
            sf.SoundFile(vocals) as voice,
            sf.SoundFile(
                temporary,
                "w",
                samplerate=rate,
                channels=bed.channels,
                format="RF64",
                subtype="FLOAT",
            ) as out,
        ):
            if (
                voice.frames != bed.frames
                or voice.channels != bed.channels
                or voice.samplerate != rate
            ):
                raise ProjectError("分离人声与背景时间轴不一致。")
            for offset in range(0, bed.frames, rate * 10):
                check_cancelled()
                base = bed.read(rate * 10, dtype="float32", always_2d=True)
                original_voice = voice.read(len(base), dtype="float32", always_2d=True)
                mask = np.full(len(base), float(bilingual), dtype=np.float32)
                for lo, hi in merged:
                    a, b = max(offset, lo), min(offset + len(base), hi)
                    if a < b:
                        positions = np.arange(a, b)
                        mask[a - offset : b - offset] = np.minimum(
                            1, np.minimum((positions - lo) / fade, (hi - positions) / fade)
                        )
                for sentence in overrides:
                    lo = max(0, round(sentence.start_seconds * rate))
                    hi = min(bed.frames, round(sentence.end_seconds * rate))
                    a, b = max(offset, lo), min(offset + len(base), hi)
                    if a >= b:
                        continue
                    present = sentence.original_audio_enabled
                    if present is None:
                        present = bilingual or sentence.id in selected_ids
                    gain = 10 ** (sentence.original_audio_gain_db / 20) if present else 0
                    positions = np.arange(a, b)
                    ramp = np.minimum(
                        1, np.minimum((positions - lo) / fade, (hi - positions) / fade)
                    )
                    section = slice(a - offset, b - offset)
                    mask[section] = mask[section] * (1 - ramp) + gain * ramp
                out.write(base + original_voice * mask[:, None])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write_text(
        directory / "mix/experimental-report.json",
        json.dumps(
            {
                "experimental": True,
                "mode": settings.separation_mix_mode,
                "sentence_overrides": [
                    {
                        "id": s.id,
                        "enabled": s.original_audio_enabled,
                        "gain_db": s.original_audio_gain_db,
                    }
                    for s in overrides
                ],
                "rtf": settings.spatial_rtf_enabled,
                "retained_ids": [s.id for s in selected],
                "policy": settings.separation_keep_original,
                "warning": "未配音不等于非语言声音；背景可能残留日语。请人工试听。",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return target


# --- Replacement 2.0: remove the original voice only where a dub replaces it ---

_ENVELOPE_HOP_SECONDS = 0.010
_WINDOW_FADE_SECONDS = 0.030
_WINDOW_MERGE_GAP_SECONDS = 0.150
_SNAP_OUTWARD_SECONDS = (0.150, 0.250)  # before the sentence start / after its end
_SNAP_INWARD_SECONDS = 0.030
# ASR timestamps often stop early (or start late) on soft speech.  The window follows
# the separated voice while it stays within _FOLLOW_DROP_DB of the line's peak,
# bridging syllable gaps up to _FOLLOW_GAP_SECONDS, but never past these limits.
_FOLLOW_AFTER_SECONDS = 1.5
_FOLLOW_BEFORE_SECONDS = 0.6
_FOLLOW_DROP_DB = 24.0
_FOLLOW_GAP_SECONDS = 0.12


def _vocal_envelope(vocals: Path) -> tuple[np.ndarray, float]:
    """Short-time RMS of the separated voice, one value per 10 ms."""
    with sf.SoundFile(vocals) as handle:
        hop = max(1, round(handle.samplerate * _ENVELOPE_HOP_SECONDS))
        values: list[np.ndarray] = []
        for block in handle.blocks(blocksize=hop * 1000, dtype="float32", always_2d=True):
            check_cancelled()
            power = np.max(block.astype(np.float64) ** 2, axis=1)
            padding = (-len(power)) % hop
            if padding:
                power = np.pad(power, (0, padding))
            values.append(np.sqrt(power.reshape(-1, hop).mean(axis=1)))
        envelope = np.concatenate(values) if values else np.zeros(0)
        return envelope, hop / handle.samplerate


def _snap_boundary(
    envelope: np.ndarray, hop_seconds: float, nominal: float, lower: float, upper: float
) -> float:
    """Move a window edge to the quietest nearby point of the voice track."""
    lo = max(0, int(np.floor(lower / hop_seconds)))
    hi = min(len(envelope), int(np.ceil(upper / hop_seconds)) + 1)
    if hi - lo < 2:
        return nominal
    section = envelope[lo:hi]
    quiet = np.flatnonzero(section <= float(section.min()) * 1.5 + 1e-9)
    times = (lo + quiet + 0.5) * hop_seconds
    # Frame centres can sit half a hop past the limit; never cross a kept neighbour.
    return float(np.clip(times[np.argmin(np.abs(times - nominal))], lower, upper))


def _follow_voice(
    envelope: np.ndarray,
    hop_seconds: float,
    start: float,
    end: float,
    limit: float,
    *,
    forward: bool,
) -> float:
    """Where the voice that a sentence belongs to actually fades out (or in)."""
    if not len(envelope):
        return end if forward else start
    a = max(0, int(start / hop_seconds))
    b = min(len(envelope), max(a + 1, int(np.ceil(end / hop_seconds))))
    peak = float(envelope[a:b].max(initial=0.0))
    if peak <= 1e-7:
        return end if forward else start
    threshold = peak * 10 ** (-_FOLLOW_DROP_DB / 20)
    gap = max(1, round(_FOLLOW_GAP_SECONDS / hop_seconds))
    if forward:
        indices = range(b, min(len(envelope), int(limit / hop_seconds)))
        edge = end
    else:
        indices = range(a - 1, max(-1, int(np.ceil(limit / hop_seconds)) - 1), -1)
        edge = start
    quiet = 0
    for index in indices:
        if envelope[index] >= threshold:
            quiet = 0
            edge = (index + 1) * hop_seconds if forward else index * hop_seconds
        else:
            quiet += 1
            if quiet > gap:
                break
    return edge


def sentence_is_replaced(sentence) -> bool:
    """Whether the original voice of this sentence is removed in replacement mode."""
    if sentence.original_audio_enabled is True:
        return False
    if sentence.original_audio_enabled is False:
        return True
    return bool(
        sentence.enabled
        and sentence.chinese_audio_enabled
        and sentence.tts_file
        and has_speakable_text(sentence.zh_text)
    )


def replacement_windows(
    project: DubProject, envelope: np.ndarray, hop_seconds: float
) -> tuple[list[tuple[float, float]], list[str]]:
    """Return merged [start, end] windows (seconds) whose original voice is replaced."""
    ordered = sorted(project.sentences, key=lambda s: (s.start_seconds, s.end_seconds, s.id))
    padding = project.settings.separation_keep_padding_ms / 1000
    raw: list[tuple[float, float]] = []
    ids: list[str] = []
    for index, sentence in enumerate(ordered):
        if not sentence_is_replaced(sentence):
            continue
        ids.append(sentence.id)
        before = ordered[index - 1] if index else None
        after = ordered[index + 1] if index + 1 < len(ordered) else None
        lower = sentence.start_seconds - max(padding, _SNAP_OUTWARD_SECONDS[0])
        upper = sentence.end_seconds + max(padding, _SNAP_OUTWARD_SECONDS[1])
        # Never eat into a neighbouring sentence that keeps its original voice.
        if before is not None and not sentence_is_replaced(before):
            lower = max(lower, min(before.end_seconds, sentence.start_seconds))
        if after is not None and not sentence_is_replaced(after):
            upper = min(upper, max(after.start_seconds, sentence.end_seconds))
        # Follow the voice past early/late ASR timestamps, up to any neighbour.
        follow_limit = sentence.end_seconds + _FOLLOW_AFTER_SECONDS
        if after is not None:
            follow_limit = min(follow_limit, max(after.start_seconds, sentence.end_seconds))
        voice_end = _follow_voice(
            envelope,
            hop_seconds,
            sentence.start_seconds,
            sentence.end_seconds,
            follow_limit,
            forward=True,
        )
        follow_floor = sentence.start_seconds - _FOLLOW_BEFORE_SECONDS
        if before is not None:
            follow_floor = max(follow_floor, min(before.end_seconds, sentence.start_seconds))
        voice_start = _follow_voice(
            envelope,
            hop_seconds,
            sentence.start_seconds,
            sentence.end_seconds,
            follow_floor,
            forward=False,
        )
        lower = min(lower, voice_start - _SNAP_OUTWARD_SECONDS[0])
        upper = max(upper, voice_end + _SNAP_OUTWARD_SECONDS[1])
        if before is not None and not sentence_is_replaced(before):
            lower = max(lower, min(before.end_seconds, sentence.start_seconds))
        if after is not None and not sentence_is_replaced(after):
            upper = min(upper, max(after.start_seconds, sentence.end_seconds))
        start = _snap_boundary(
            envelope,
            hop_seconds,
            min(sentence.start_seconds - padding, voice_start),
            lower,
            min(sentence.start_seconds, voice_start) + _SNAP_INWARD_SECONDS,
        )
        end = _snap_boundary(
            envelope,
            hop_seconds,
            max(sentence.end_seconds + padding, voice_end),
            max(sentence.end_seconds, voice_end) - _SNAP_INWARD_SECONDS,
            upper,
        )
        raw.append((max(0.0, start), max(start, end)))
    merged: list[list[float]] = []
    for lo, hi in sorted(raw):
        if merged and lo - merged[-1][1] < _WINDOW_MERGE_GAP_SECONDS:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged], ids


def _ramp(times: np.ndarray, lo: float, hi: float, fade: float) -> np.ndarray:
    """Raised-cosine gate that is 1 inside [lo, hi] and fades across each edge."""
    rise = np.clip((times - (lo - fade / 2)) / fade, 0, 1)
    fall = np.clip(((hi + fade / 2) - times) / fade, 0, 1)
    return np.sin(np.minimum(rise, fall) * (np.pi / 2)) ** 2


def _decode_float(source: Path, destination: Path, rate: int, channels: int) -> Path:
    _run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            "-rf64",
            "auto",
            str(destination),
        ]
    )
    return destination


def _resample_file(source: Path, destination: Path, rate: int, frames: int) -> Path:
    """Stream-resample to ``rate`` and pad/trim to exactly ``frames``."""
    with sf.SoundFile(source) as handle:
        written = 0
        with sf.SoundFile(
            destination,
            "w",
            samplerate=rate,
            channels=handle.channels,
            format="RF64",
            subtype="FLOAT",
        ) as out:
            stream = (
                None
                if handle.samplerate == rate
                else soxr.ResampleStream(
                    handle.samplerate, rate, handle.channels, dtype="float32", quality="VHQ"
                )
            )
            block = handle.samplerate * 10
            position = 0
            while position < handle.frames:
                check_cancelled()
                data = handle.read(block, dtype="float32", always_2d=True)
                position += len(data)
                if stream is not None:
                    data = stream.resample_chunk(data, last=position >= handle.frames)
                data = data[: max(0, frames - written)]
                out.write(data)
                written += len(data)
            if written < frames:
                out.write(np.zeros((frames - written, handle.channels), dtype=np.float32))
    return destination


# --- Residual-voice report: an energy estimate, not a recognition result ---
# All values below are rules of thumb chosen by listening to a handful of ASMR clips.
# They have not been calibrated against labelled data; the report only says
# "listen here", it never claims that Japanese speech was heard.
_RESIDUAL_RATIO_DB = -9.0  # leftover (source - separated voice) vs. separated voice, in dB
_RESIDUAL_ABOVE_MEDIAN_DB = 8.0  # ... and this far above the median replaced sentence
_UNCOVERED_LOOK_SECONDS = 2.0  # how far outside a replaced window to look for loose voice
_UNCOVERED_DROP_DB = 20.0  # voice counts as active within this many dB of the line's own peak
_UNCOVERED_FLOOR = 10 ** (-70 / 20)  # ... and never below this absolute level
_UNCOVERED_MIN_SECONDS = 0.8  # active voice with no sentence this long gets flagged
_COVERED_PAD_SECONDS = 0.05  # sentence timestamps are loose; tolerate this much


def _db(power: float) -> float:
    return round(10 * math.log10(max(power, 1e-12)), 2)


def residual_voice_report(
    project: DubProject,
    decoded: Path,
    voice_file: Path,
    envelope: np.ndarray,
    hop_seconds: float,
    windows: list[tuple[float, float]],
    replaced_ids: list[str],
) -> dict:
    """Per replaced sentence: how much non-voice energy is left, and loose voice nearby.

    ``decoded`` is the untouched source and ``voice_file`` the separated voice, both at the
    source sample rate and length.  Inside a sentence the background is ``source - voice``;
    if the separator left part of the speech in it, the background gets loud compared with
    the voice estimate.  Outside the sentence the separated voice envelope shows whether
    there is voice activity that no sentence (and no replacement window) covers, which
    usually means speech the recogniser missed.
    """
    rows = {s.id: s for s in project.sentences}
    merged: list[list[float]] = []
    for a, b in sorted(
        [
            (s.start_seconds - _COVERED_PAD_SECONDS, s.end_seconds + _COVERED_PAD_SECONDS)
            for s in rows.values()
        ]
        + list(windows)
    ):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    starts, ends = np.array([m[0] for m in merged]), np.array([m[1] for m in merged])

    def uncovered(lo: float, hi: float, threshold: float) -> float:
        first, last = max(0, int(lo / hop_seconds)), min(len(envelope), int(hi / hop_seconds))
        if last <= first or not len(starts):
            return 0.0
        middles = (np.arange(first, last) + 0.5) * hop_seconds
        found = np.searchsorted(starts, middles, side="right") - 1
        inside = (found >= 0) & (middles <= ends[np.maximum(found, 0)])
        return float(np.count_nonzero((envelope[first:last] >= threshold) & ~inside)) * hop_seconds

    entries = []
    with sf.SoundFile(decoded) as original, sf.SoundFile(voice_file) as voice:
        rate = original.samplerate
        for sentence_id in replaced_ids:
            check_cancelled()
            sentence = rows[sentence_id]
            lo, hi = (
                max(0, round(sentence.start_seconds * rate)),
                round(sentence.end_seconds * rate),
            )
            hi = min(original.frames, hi)
            entry: dict = {
                "id": sentence_id,
                "start": round(sentence.start_seconds, 3),
                "end": round(sentence.end_seconds, 3),
                "voice_dbfs": None,
                "background_dbfs": None,
                "residual_db": None,
                "uncovered_before_seconds": 0.0,
                "uncovered_after_seconds": 0.0,
                "reasons": [],
            }
            if hi - lo >= rate // 20:
                original.seek(lo)
                voice.seek(lo)
                mix = original.read(hi - lo, dtype="float64", always_2d=True)
                estimate = voice.read(len(mix), dtype="float64", always_2d=True)
                voice_power = float(np.mean(estimate**2))
                background_power = float(np.mean((mix - estimate) ** 2))
                entry.update(
                    voice_dbfs=_db(voice_power),
                    background_dbfs=_db(background_power),
                    residual_db=round(_db(background_power) - _db(voice_power), 2),
                )
            first, last = (
                int(sentence.start_seconds / hop_seconds),
                int(sentence.end_seconds / hop_seconds),
            )
            peak = float(envelope[first : max(first + 1, last)].max(initial=0.0))
            threshold = max(peak * 10 ** (-_UNCOVERED_DROP_DB / 20), _UNCOVERED_FLOOR)
            if peak > _UNCOVERED_FLOOR:
                window = next(
                    ((a, b) for a, b in windows if a - 0.01 <= sentence.start_seconds <= b + 0.01),
                    (sentence.start_seconds, sentence.end_seconds),
                )
                entry["uncovered_before_seconds"] = round(
                    uncovered(window[0] - _UNCOVERED_LOOK_SECONDS, window[0], threshold), 2
                )
                entry["uncovered_after_seconds"] = round(
                    uncovered(window[1], window[1] + _UNCOVERED_LOOK_SECONDS, threshold), 2
                )
                if max(entry["uncovered_before_seconds"], entry["uncovered_after_seconds"]) >= (
                    _UNCOVERED_MIN_SECONDS
                ):
                    entry["reasons"].append("uncovered_voice")
            entries.append(entry)
    # Material full of effects (oil, brushes) has a loud background everywhere, so a
    # sentence only counts when it also stands out from the other replaced sentences.
    ratios = [e["residual_db"] for e in entries if e["residual_db"] is not None]
    median = float(np.median(ratios)) if len(ratios) >= 3 else -math.inf
    for entry in entries:
        ratio = entry["residual_db"]
        if (
            ratio is not None
            and ratio > _RESIDUAL_RATIO_DB
            and ratio > median + _RESIDUAL_ABOVE_MEDIAN_DB
        ):
            entry["reasons"].insert(0, "residual_voice")
        entry["exceeds"] = bool(entry["reasons"])
    return {
        "version": 1,
        "kind": "energy_estimate",
        "note": (
            "能量估计，不是识别结果：没有对原声做语音识别。residual_db 是“原文件减去分离人声”"
            "与分离人声的能量比；uncovered_*_seconds 是替换区间紧邻处有人声活动、但没有任何句子"
            "覆盖的时长。阈值是经验值，只提示哪里值得试听。"
        ),
        "thresholds": {
            "residual_db": _RESIDUAL_RATIO_DB,
            "residual_above_median_db": _RESIDUAL_ABOVE_MEDIAN_DB,
            "uncovered_seconds": _UNCOVERED_MIN_SECONDS,
            "uncovered_look_seconds": _UNCOVERED_LOOK_SECONDS,
            "uncovered_drop_db": _UNCOVERED_DROP_DB,
        },
        "sentences": entries,
        "flagged_ids": [e["id"] for e in entries if e["exceeds"]],
    }


def _write_residual_report(
    project, directory, decoded, voice_file, envelope, hop_seconds, windows, ids
):
    """Write mix/replacement-report.json; the report never decides whether the mix succeeds."""
    from .errors import OperationCancelledError

    try:
        report = residual_voice_report(
            project, decoded, voice_file, envelope, hop_seconds, windows, ids
        )
        atomic_write_text(
            directory / "mix/replacement-report.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    except OperationCancelledError:
        raise
    except Exception:
        logger.warning("残留原声报告生成失败，已跳过。", exc_info=True)


def compose_speech_replacement(
    project: DubProject, directory: Path, source: Path, vocals: Path
) -> Path:
    """Subtract the separated voice from the untouched source only inside dubbed windows.

    Outside the windows the output equals the decoded source sample for sample, so
    breaths, mouth sounds, effects and the room between sentences stay exactly as
    recorded.  Inside a window the voice estimate is removed at the source sample rate.
    """
    settings = project.settings
    rate, channels = project.source.sample_rate, project.source.channels
    envelope, hop_seconds = _vocal_envelope(vocals)
    windows, replaced_ids = replacement_windows(project, envelope, hop_seconds)
    replaced_set = set(replaced_ids)
    gains = [
        (s.start_seconds, s.end_seconds, 10 ** (s.original_audio_gain_db / 20))
        for s in project.sentences
        if s.original_audio_gain_db != 0 and s.id not in replaced_set
    ]
    target = directory / "mix/replacement-background.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = target.with_name(f".{token}.wav")
    decoded = target.with_name(f".{token}.source.wav")
    voice_file = target.with_name(f".{token}.vocals.wav")
    fade = _WINDOW_FADE_SECONDS
    try:
        _decode_float(source, decoded, rate, channels)
        frames = sf.info(decoded).frames
        _resample_file(vocals, voice_file, rate, frames)
        floor_noise: np.ndarray | None = None
        floor_level = 0.0
        if settings.replacement_noise_fill and windows:
            from .voice_match import noise_floor, synthesize_floor

            # The separator files the room and microphone noise under "voice", so a
            # replaced window would otherwise drop to digital silence.
            with sf.SoundFile(decoded) as probe:
                sample = probe.read(min(frames, rate * 180), dtype="float32", always_2d=True)
            floor = noise_floor(sample)
            floor_noise = synthesize_floor(floor, rate * 10 + 1)
            floor_level = float(np.sqrt(np.mean(floor_noise.astype(np.float64) ** 2)))
            del sample
        with (
            sf.SoundFile(decoded) as original,
            sf.SoundFile(voice_file) as voice,
            sf.SoundFile(
                temporary, "w", samplerate=rate, channels=channels, format="RF64", subtype="FLOAT"
            ) as out,
        ):
            if voice.channels != channels:
                raise ProjectError("分离人声与原音频声道数不一致。")
            block = rate * 10
            for offset in range(0, frames, block):
                check_cancelled()
                raw = original.read(block, dtype="float32", always_2d=True)
                estimate = voice.read(len(raw), dtype="float32", always_2d=True)
                lo_time, hi_time = offset / rate, (offset + len(raw)) / rate
                times = (offset + np.arange(len(raw))) / rate
                removal = np.zeros(len(raw), dtype=np.float64)
                for lo, hi in windows:
                    if lo - fade <= hi_time and hi + fade >= lo_time:
                        removal = np.maximum(removal, _ramp(times, lo, hi, fade))
                for lo, hi, gain in gains:
                    if lo - fade <= hi_time and hi + fade >= lo_time:
                        removal += (1 - removal) * (1 - gain) * _ramp(times, lo, hi, fade)
                active = removal != 0
                if active.any():
                    raw = raw.copy()
                    raw[active] -= estimate[active] * removal[active, None].astype(np.float32)
                    full = removal >= 0.99
                    if floor_noise is not None and floor_level > 0 and full.sum() > rate // 20:
                        # Add only what is missing: nothing if the bed kept its own floor.
                        hop = max(1, rate // 50)
                        inside = raw[full]
                        count = len(inside) // hop
                        levels = np.sqrt(
                            np.mean(
                                inside[: count * hop].reshape(count, hop, -1).astype(np.float64)
                                ** 2,
                                axis=(1, 2),
                            )
                        )
                        left = float(np.percentile(levels, 10)) if count else 0.0
                        missing = math.sqrt(max(0.0, 1.0 - (left / floor_level) ** 2))
                        if missing > 0.05:
                            shift = offset % max(1, len(floor_noise) - len(raw))
                            noise = floor_noise[shift : shift + len(raw)]
                            raw += noise * (removal * missing)[:, None].astype(np.float32)
                out.write(raw)
        temporary.replace(target)
        _write_residual_report(
            project, directory, decoded, voice_file, envelope, hop_seconds, windows, replaced_ids
        )
    finally:
        for path in (temporary, decoded, voice_file):
            path.unlink(missing_ok=True)
    atomic_write_text(
        directory / "mix/experimental-report.json",
        json.dumps(
            {
                "experimental": True,
                "mode": settings.separation_mix_mode,
                "scope": "dubbed_speech",
                "rtf": settings.spatial_rtf_enabled,
                "replaced_ids": replaced_ids,
                "windows": [[round(lo, 4), round(hi, 4)] for lo, hi in windows],
                "retained_ids": [s.id for s in project.sentences if s.id not in replaced_set],
                "warning": "只替换有配音的语音区间；其余时间与原文件一致。",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return target
