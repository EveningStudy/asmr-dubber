from __future__ import annotations

import gc
import logging
import math
import shutil
import subprocess as subprocess
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .api_logging import logged_http_client as logged_http_client
from .asr_moss_runner import _transcribe_moss as _transcribe_moss
from .asr_parakeet import (
    _crispasr_executable as _crispasr_executable,
)
from .asr_parakeet import (
    _parakeet_input_path as _parakeet_input_path,
)
from .asr_parakeet import (
    _parakeet_model_path as _parakeet_model_path,
)
from .asr_parakeet import (
    _transcribe_parakeet as _transcribe_parakeet,
)
from .asr_runners import (
    _transcribe_faster_whisper as _transcribe_faster_whisper,
)
from .asr_runners import (
    _transcribe_generic_api as _transcribe_generic_api,
)
from .asr_runners import (
    _transcribe_kotoba_whisper as _transcribe_kotoba_whisper,
)
from .constants import PROJECT_ROOT as PROJECT_ROOT
from .environment import (
    cached_model_path as cached_model_path,
)
from .environment import (
    require_cuda as require_cuda,
)
from .environment import (
    resolve_model_source as resolve_model_source,
)
from .environment import (
    resolve_transformers_model_source as resolve_transformers_model_source,
)
from .errors import AsmrDubberError
from .languages import SpeechSourceLanguage, source_language_label
from .models import ProjectSettings, Sentence
from .platforms import (
    current_platform as current_platform,
)
from .platforms import (
    isolated_runtime_environment as isolated_runtime_environment,
)
from .platforms import (
    portable_home as portable_home,
)
from .segmentation import TimedToken, restore_punctuation, split_timed_tokens
from .task_control import (
    CancellationSignal,
    check_cancelled,
)
from .task_control import (
    register_process as register_process,
)
from .task_control import (
    terminate_process_tree as terminate_process_tree,
)
from .task_control import (
    unregister_process as unregister_process,
)
from .user_settings import saved_service_key as saved_service_key
from .vad import (
    build_condensed_analysis_audio,
    detect_asmr_speech,
    map_analysis_time,
)

Progress = Callable[[str, int, int], None]

_PARAKEET_AUTO_CHUNK_SECONDS = 120.0
_PARAKEET_BOUNDARY_SEARCH_SECONDS = 5.0
_PARAKEET_BOUNDARY_WINDOW_SECONDS = 0.1
_TRANSFORMERS_ASR_PIPELINE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)
_REVIEW_MODELS: ContextVar[dict[str, Any] | None] = ContextVar("review_asr_models", default=None)


@contextmanager
def recognition_session():
    """Keep selected Python ASR models resident across windows and batch tracks."""
    existing = _REVIEW_MODELS.get()
    if existing is not None:
        yield
        return
    models: dict[str, Any] = {}
    marker = _REVIEW_MODELS.set(models)
    try:
        yield
    finally:
        models.clear()
        _REVIEW_MODELS.reset(marker)
        _cleanup_cuda()


def evict_recognition_session_if_low_vram(min_free_gb: float = 5.0) -> None:
    """Release cached in-process ASR models only when CUDA VRAM is tight."""
    models = _REVIEW_MODELS.get()
    if not models:
        return
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= min_free_gb * (1024**3):
                return
    except Exception:
        return
    models.clear()
    _cleanup_cuda(force=True)


def _session_model(key: str, factory: Callable[[], Any]) -> Any:
    models = _REVIEW_MODELS.get()
    if models is None:
        return factory()
    if key not in models:
        if models:
            try:
                import torch

                if torch.cuda.is_available():
                    free_bytes, _total_bytes = torch.cuda.mem_get_info()
                    if free_bytes < 4.0 * (1024**3):
                        models.clear()
                        _cleanup_cuda(force=True)
            except Exception:
                pass
        models[key] = factory()
    return models[key]


def _run_transformers_asr_pipeline(pipe: Any, inputs: Any, **kwargs: Any) -> Any:
    """Run an already-decoded waveform without importing optional TorchCodec."""
    from transformers.pipelines import automatic_speech_recognition as asr_pipeline

    with _TRANSFORMERS_ASR_PIPELINE_LOCK:
        availability_check = asr_pipeline.is_torchcodec_available
        asr_pipeline.is_torchcodec_available = lambda: False
        try:
            return pipe(inputs, **kwargs)
        finally:
            asr_pipeline.is_torchcodec_available = availability_check


def _finite_token(text: Any, start: Any, end: Any) -> TimedToken | None:
    value = str(text or "").strip()
    if not value:
        return None
    try:
        start_value = max(0.0, float(start))
        end_value = float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start_value) or not math.isfinite(end_value) or end_value <= start_value:
        return None
    return TimedToken(text=value, start_seconds=start_value, end_seconds=end_value)


def _offset_finite_token(text: Any, start: Any, end: Any, offset: float) -> TimedToken | None:
    try:
        return _finite_token(text, offset + float(start), offset + float(end))
    except (TypeError, ValueError):
        return None


def _audio_file_chunk_ranges(
    audio_path: Path,
    chunk_seconds: float,
    search_seconds: float = _PARAKEET_BOUNDARY_SEARCH_SECONDS,
    window_seconds: float = _PARAKEET_BOUNDARY_WINDOW_SECONDS,
) -> tuple[int, list[tuple[int, int]]]:
    """Split an audio file near quiet boundaries without loading it all into RAM."""
    with sf.SoundFile(audio_path) as audio:
        sample_rate = audio.samplerate
        total_samples = len(audio)
        target_samples = max(1, round(chunk_seconds * sample_rate))
        if total_samples <= target_samples:
            return sample_rate, [(0, total_samples)]
        search_samples = max(0, round(search_seconds * sample_rate))
        window_samples = max(1, round(window_seconds * sample_rate))
        ranges: list[tuple[int, int]] = []
        start = 0
        while total_samples - start > target_samples:
            target = start + target_samples
            left = max(start + 1, target - search_samples)
            right = min(total_samples, target + search_samples)
            audio.seek(left)
            region = audio.read(right - left, dtype="float32", always_2d=True)
            amplitudes = np.mean(np.abs(region), axis=1)
            if amplitudes.size <= window_samples:
                boundary = target
            else:
                prefix = np.empty(amplitudes.size + 1, dtype=np.float64)
                prefix[0] = 0.0
                np.cumsum(amplitudes, dtype=np.float64, out=prefix[1:])
                energies = prefix[window_samples:] - prefix[:-window_samples]
                boundary = left + int(np.argmin(energies)) + (window_samples // 2)
            boundary = min(total_samples, max(start + 1, boundary))
            ranges.append((start, boundary))
            start = boundary
        ranges.append((start, total_samples))
        return sample_rate, ranges


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def _finish_tokens(
    tokens: Iterable[TimedToken],
    full_text: str,
    language: str,
    settings: ProjectSettings,
    source_language: SpeechSourceLanguage = "ja",
    *,
    restore: bool = True,
) -> tuple[list[Sentence], str]:
    values = list(tokens)
    if not values:
        if _REVIEW_MODELS.get() is not None and not full_text.strip():
            return [], language or source_language
        if not full_text.strip():
            raise AsmrDubberError("未检测到可识别的语音；请检查音频或调整语音检测设置。")
        raise AsmrDubberError("识别出了文字，但所选 ASR（语音识别）后端没有返回可用时间戳。")
    punctuated = restore_punctuation(values, full_text) if restore else values
    sentences = split_timed_tokens(
        punctuated,
        pause_seconds=settings.pause_split_seconds,
        max_sentence_seconds=settings.max_sentence_seconds,
    )
    if not sentences:
        raise AsmrDubberError(
            f"没有在音频中识别到带有效时间戳的{source_language_label(source_language)}句子。"
        )
    return sentences, language or source_language


def _cleanup_cuda(force: bool = False) -> None:
    if not force and _REVIEW_MODELS.get() is not None:
        return
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _crispasr_payload_tokens(
    payload: Mapping[str, Any],
) -> tuple[list[TimedToken], str, str]:
    def timed_items(items: Any) -> list[TimedToken]:
        parsed: list[TimedToken] = []
        expected = 0
        for item in items or []:
            if not isinstance(item, Mapping):
                return []
            text = str(item.get("text", "") or "").strip()
            # Some CTC punctuation passes append an empty terminal token.
            if not text:
                continue
            expected += 1
            offsets = item.get("offsets") or {}
            if not isinstance(offsets, Mapping):
                return []
            token = _finite_token(
                text,
                float(offsets.get("from", 0)) / 1000,
                float(offsets.get("to", 0)) / 1000,
            )
            if token is None:
                return []
            parsed.append(token)
        return parsed if expected and len(parsed) == expected else []

    tokens: list[TimedToken] = []
    full_text_parts: list[str] = []
    for segment in payload.get("transcription", []) or []:
        if not isinstance(segment, Mapping):
            continue
        text = str(segment.get("text", "") or "")
        full_text_parts.append(text)
        # TDT normally exposes merged `words`; FastConformer CTC 1.1B exposes
        # only timestamped `tokens`. Prefer words when complete, otherwise use
        # tokens before falling back to the coarse segment range. The previous
        # implementation ignored `tokens`, which collapsed an entire CTC
        # recording into one sentence.
        item_tokens = timed_items(segment.get("words"))
        if not item_tokens:
            item_tokens = timed_items(segment.get("tokens"))
        if item_tokens:
            tokens.extend(item_tokens)
            continue
        if text:
            offsets = segment.get("offsets") or {}
            if isinstance(offsets, Mapping):
                token = _finite_token(
                    text,
                    float(offsets.get("from", 0)) / 1000,
                    float(offsets.get("to", 0)) / 1000,
                )
                if token:
                    tokens.append(token)
    metadata = payload.get("crispasr") or {}
    language = str(metadata.get("language", "Japanese")) if isinstance(metadata, Mapping) else "ja"
    return tokens, "".join(full_text_parts), language


def _parse_crispasr_payload(
    payload: Mapping[str, Any],
    settings: ProjectSettings,
) -> tuple[list[Sentence], str]:
    tokens, full_text, language = _crispasr_payload_tokens(payload)
    return _finish_tokens(tokens, full_text, language, settings, "ja")


def _transcribe_qwen3_asr(*args: Any, **kwargs: Any) -> tuple[list[Sentence], str]:
    from .asr_candidates import transcribe_qwen3_asr

    return transcribe_qwen3_asr(*args, **kwargs)


def _transcribe_anime_whisper(*args: Any, **kwargs: Any) -> tuple[list[Sentence], str]:
    from .asr_candidates import transcribe_anime_whisper

    return transcribe_anime_whisper(*args, **kwargs)


def transcribe_source(
    analysis_audio: Path,
    settings: ProjectSettings,
    source_language: SpeechSourceLanguage = "ja",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[list[Sentence], str]:
    """Run a registered recognition backend with cancellation and timeline mapping."""

    if source_language != "ja" and settings.asr_backend not in {
        "faster_whisper",
        "generic_asr_api",
        "moss_transcribe_diarize",
        "qwen3_asr",
    }:
        raise AsmrDubberError("英语识别可使用 Faster-Whisper 或通用 ASR API。")

    # Only reject digital silence, never a loudness threshold that might remove whispers.
    check_cancelled(cancel_event)
    with sf.SoundFile(analysis_audio) as audio:
        signal_present = False
        for block in audio.blocks(blocksize=262_144, dtype="float32", always_2d=True):
            check_cancelled(cancel_event)
            if np.any(block != 0):
                signal_present = True
                break
    if not signal_present:
        if _REVIEW_MODELS.get() is not None:
            return [], source_language
        raise AsmrDubberError("未检测到语音：音频没有有效信号（数字静音）。原有句子表已保留。")

    runners = {
        "moss_transcribe_diarize": _transcribe_moss,
        "generic_asr_api": _transcribe_generic_api,
        "parakeet_nemo": _transcribe_parakeet,
        "kotoba_whisper": _transcribe_kotoba_whisper,
        "faster_whisper": _transcribe_faster_whisper,
        "qwen3_asr": _transcribe_qwen3_asr,
        "anime_whisper": _transcribe_anime_whisper,
    }
    try:
        runner = runners[settings.asr_backend]
    except KeyError as exc:
        raise AsmrDubberError(
            f"不支持的 ASR（语音识别）后端：{settings.asr_backend}。"
            "请选择 Parakeet、Kotoba-Whisper、Faster-Whisper 或通用 ASR API。"
        ) from exc
    # anime-whisper and Japanese Qwen3-ASR cut speech with the ASMR VAD themselves.
    self_segmenting = settings.asr_backend == "anime_whisper" or (
        settings.asr_backend == "qwen3_asr" and source_language == "ja"
    )
    if settings.asr_vad_mode == "asmr" and not self_segmenting:
        if source_language != "ja":
            raise AsmrDubberError("日语 ASMR 专用 VAD 不用于英语项目；请选择关闭或后端 VAD。")
        cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        segments = detect_asmr_speech(
            analysis_audio,
            threshold=settings.asr_asmr_vad_threshold,
            min_speech_ms=settings.asr_asmr_vad_min_speech_ms,
            min_silence_ms=settings.asr_asmr_vad_min_silence_ms,
            speech_pad_ms=settings.asr_asmr_vad_speech_pad_ms,
            progress=progress,
            **cancel_kwargs,
        )
        if not segments:
            raise AsmrDubberError(
                "日语 ASMR 专用 VAD 没有检测到语音。请降低阈值、增加边界保留，或关闭 VAD。"
            )
        run_directory = portable_home() / "temp" / "asr" / f"asmr-vad-{uuid.uuid4().hex}"
        run_directory.mkdir(parents=True, exist_ok=False)
        condensed = run_directory / "speech.wav"
        try:
            timeline = build_condensed_analysis_audio(
                analysis_audio,
                condensed,
                segments,
                separator_seconds=max(0.65, settings.pause_split_seconds + 0.1),
                **cancel_kwargs,
            )
            inner_settings = settings.model_copy(
                update={"asr_vad_mode": "off", "asr_vad_filter": False}
            )
            check_cancelled(cancel_event)
            if cancel_event is None:
                sentences, language = runner(
                    condensed,
                    inner_settings,
                    progress,
                    source_language=source_language,
                )
            else:
                sentences, language = runner(
                    condensed,
                    inner_settings,
                    progress,
                    cancel_event,
                    source_language=source_language,
                )
            remapped: list[Sentence] = []
            for sentence in sentences:
                start = map_analysis_time(sentence.start_seconds, timeline, end=False)
                end = map_analysis_time(sentence.end_seconds, timeline, end=True)
                if end <= start:
                    continue
                remapped.append(
                    sentence.model_copy(
                        update={
                            "id": f"s{len(remapped) + 1:06d}",
                            "start_seconds": start,
                            "end_seconds": end,
                        }
                    )
                )
            if not remapped:
                raise AsmrDubberError("ASMR VAD 识别结果无法映射回原始时间轴。")
            result = remapped, language
        finally:
            shutil.rmtree(run_directory, ignore_errors=True)
    else:
        if cancel_event is None:
            result = runner(
                analysis_audio,
                settings,
                progress,
                source_language=source_language,
            )
        else:
            result = runner(
                analysis_audio,
                settings,
                progress,
                cancel_event,
                source_language=source_language,
            )
    if progress:
        progress(f"语音识别完成：{len(result[0])} 句", 1, 1)
    return result


def transcribe_japanese(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[list[Sentence], str]:
    """Backward-compatible Japanese recognition entry point."""

    return transcribe_source(
        analysis_audio,
        settings,
        source_language="ja",
        progress=progress,
        cancel_event=cancel_event,
    )
