"""Experimental RTF rendering and explicit residual/original-voice composition."""

from __future__ import annotations

import json
import logging
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
        ratio = float(np.clip(levels[1] / max(levels[0], 1e-8), 0.25, 4))
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
