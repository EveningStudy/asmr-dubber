from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .constants import ASMR_VAD_MODEL, OPTIONAL_ASR_MODEL_REVISIONS
from .environment import cached_model_path
from .errors import AsmrDubberError

_SAMPLE_RATE = 16_000
_CHUNK_SECONDS = 30
_FRAME_SECONDS = 0.02


@dataclass(frozen=True)
class VadSegment:
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class TimelinePiece:
    analysis_start: float
    analysis_end: float
    original_start: float
    original_end: float


def _cache_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}.asmr-vad.json")


def _cache_key(
    audio_path: Path,
    *,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> dict[str, Any]:
    stat = audio_path.stat()
    return {
        "schema": 1,
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "model": ASMR_VAD_MODEL,
        "revision": OPTIONAL_ASR_MODEL_REVISIONS[ASMR_VAD_MODEL],
        "threshold": threshold,
        "min_speech_ms": min_speech_ms,
        "min_silence_ms": min_silence_ms,
        "speech_pad_ms": speech_pad_ms,
    }


def _read_cached_segments(path: Path, key: dict[str, Any]) -> list[VadSegment] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("key") != key:
            return None
        segments = [VadSegment(int(item[0]), int(item[1])) for item in payload.get("segments", [])]
        if any(item.end_sample <= item.start_sample for item in segments):
            return None
        return segments
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_segments(path: Path, key: dict[str, Any], segments: list[VadSegment]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "key": key,
                "segments": [[item.start_sample, item.end_sample] for item in segments],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _speech_segments_from_probabilities(
    probabilities: np.ndarray,
    *,
    total_samples: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[VadSegment]:
    min_speech_frames = max(1, math.ceil(min_speech_ms / (_FRAME_SECONDS * 1000)))
    min_silence_frames = max(1, math.ceil(min_silence_ms / (_FRAME_SECONDS * 1000)))
    pad_samples = round(speech_pad_ms * _SAMPLE_RATE / 1000)
    frame_samples = round(_FRAME_SECONDS * _SAMPLE_RATE)
    negative_threshold = max(0.01, threshold - 0.15)
    raw: list[tuple[int, int]] = []
    triggered = False
    start_frame = 0
    possible_end: int | None = None
    for index, probability in enumerate(probabilities):
        if not triggered and probability >= threshold:
            triggered = True
            start_frame = index
            possible_end = None
            continue
        if not triggered:
            continue
        if probability < negative_threshold:
            if possible_end is None:
                possible_end = index
            if index - possible_end >= min_silence_frames:
                if possible_end - start_frame >= min_speech_frames:
                    raw.append((start_frame * frame_samples, possible_end * frame_samples))
                triggered = False
                possible_end = None
        elif probability >= threshold:
            possible_end = None
    if triggered:
        end_frame = len(probabilities)
        if end_frame - start_frame >= min_speech_frames:
            raw.append((start_frame * frame_samples, total_samples))

    padded = [
        [max(0, start - pad_samples), min(total_samples, end + pad_samples)] for start, end in raw
    ]
    for index in range(len(padded) - 1):
        if padded[index][1] > padded[index + 1][0]:
            midpoint = (raw[index][1] + raw[index + 1][0]) // 2
            padded[index][1] = midpoint
            padded[index + 1][0] = midpoint
    return [VadSegment(start, end) for start, end in padded if end > start]


def detect_asmr_speech(
    audio_path: Path,
    *,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    progress: Any | None = None,
) -> list[VadSegment]:
    """Detect quiet/whispered speech with the pinned ASMR-specialized ONNX VAD."""

    model_root = cached_model_path(ASMR_VAD_MODEL)
    if model_root is None:
        raise AsmrDubberError(
            "日语 ASMR 专用 VAD 模型尚未安装。请重新运行“进阶”安装，或导入对应离线模型包。"
        )
    try:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
    except ImportError as exc:
        raise AsmrDubberError(
            "日语 ASMR 专用 VAD 运行依赖尚未安装；请重新运行“进阶”安装。"
        ) from exc

    key = _cache_key(
        audio_path,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    cache = _cache_path(audio_path)
    cached = _read_cached_segments(cache, key)
    if cached is not None:
        if progress:
            progress(f"复用 ASMR VAD：检测到 {len(cached)} 个语音区间", 1, 1)
        return cached

    model_path = model_root / "model.onnx"
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise AsmrDubberError(f"无法加载日语 ASMR 专用 VAD：{exc}") from exc

    extractor = WhisperFeatureExtractor()
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    chunk_samples = _SAMPLE_RATE * _CHUNK_SECONDS
    probabilities: list[np.ndarray] = []
    total_samples = 0
    with sf.SoundFile(audio_path) as audio:
        if audio.samplerate != _SAMPLE_RATE or audio.channels != 1:
            raise AsmrDubberError("ASMR VAD 需要程序生成的 16 kHz 单声道分析音频。")
        total_samples = len(audio)
        total_chunks = max(1, math.ceil(total_samples / chunk_samples))
        for chunk_index in range(total_chunks):
            if progress:
                progress(
                    f"ASMR VAD 分析 {chunk_index + 1}/{total_chunks}",
                    chunk_index,
                    total_chunks,
                )
            waveform = audio.read(chunk_samples, dtype="float32", always_2d=False)
            actual_samples = len(waveform)
            if actual_samples < chunk_samples:
                waveform = np.pad(waveform, (0, chunk_samples - actual_samples))
            features = extractor(
                waveform,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="np",
            ).input_features
            logits = np.asarray(session.run(output_names, {input_name: features})[0][0])
            valid_frames = min(
                len(logits),
                math.ceil(actual_samples / (_FRAME_SECONDS * _SAMPLE_RATE)),
            )
            logits = np.clip(logits[:valid_frames], -30.0, 30.0)
            probabilities.append(1.0 / (1.0 + np.exp(-logits)))

    combined = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    segments = _speech_segments_from_probabilities(
        combined,
        total_samples=total_samples,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    _write_cached_segments(cache, key, segments)
    if progress:
        progress(f"ASMR VAD 完成：检测到 {len(segments)} 个语音区间", 1, 1)
    return segments


def build_condensed_analysis_audio(
    source_path: Path,
    destination: Path,
    segments: list[VadSegment],
    *,
    separator_seconds: float,
) -> list[TimelinePiece]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    separator_samples = max(0, round(separator_seconds * _SAMPLE_RATE))
    separator = np.zeros(separator_samples, dtype=np.float32)
    timeline: list[TimelinePiece] = []
    analysis_cursor = 0
    with (
        sf.SoundFile(source_path) as source,
        sf.SoundFile(
            destination,
            mode="w",
            samplerate=_SAMPLE_RATE,
            channels=1,
            subtype="PCM_16",
        ) as output,
    ):
        for index, segment in enumerate(segments):
            source.seek(segment.start_sample)
            waveform = source.read(
                segment.end_sample - segment.start_sample,
                dtype="float32",
                always_2d=False,
            )
            if index and separator_samples:
                output.write(separator)
                analysis_cursor += separator_samples
            analysis_start = analysis_cursor / _SAMPLE_RATE
            output.write(waveform)
            analysis_cursor += len(waveform)
            timeline.append(
                TimelinePiece(
                    analysis_start=analysis_start,
                    analysis_end=analysis_cursor / _SAMPLE_RATE,
                    original_start=segment.start_sample / _SAMPLE_RATE,
                    original_end=segment.end_sample / _SAMPLE_RATE,
                )
            )
    return timeline


def map_analysis_time(value: float, timeline: list[TimelinePiece], *, end: bool) -> float:
    if not timeline:
        return value
    previous: TimelinePiece | None = None
    for piece in timeline:
        if piece.analysis_start <= value <= piece.analysis_end:
            return piece.original_start + (value - piece.analysis_start)
        if value < piece.analysis_start:
            return previous.original_end if end and previous is not None else piece.original_start
        previous = piece
    return timeline[-1].original_end
