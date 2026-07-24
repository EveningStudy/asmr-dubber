from __future__ import annotations

import gc
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .constants import PROJECT_ROOT
from .environment import (
    cached_model_path,
    require_cuda,
    resolve_model_source,
    resolve_transformers_model_source,
)
from .errors import AsmrDubberError
from .model_registry import ASR_BACKENDS
from .models import ProjectSettings, Sentence
from .platforms import current_platform, isolated_runtime_environment, portable_home
from .segmentation import TimedToken, restore_punctuation, split_timed_tokens
from .user_settings import saved_service_key

Progress = Callable[[str, int, int], None]

_QWEN_CHUNK_SECONDS = 90.0
_QWEN_BOUNDARY_SEARCH_SECONDS = 5.0
_QWEN_BOUNDARY_WINDOW_SECONDS = 0.1
_TRANSFORMERS_ASR_PIPELINE_LOCK = threading.Lock()


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


def _qwen_chunk_ranges(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = _QWEN_CHUNK_SECONDS,
    search_seconds: float = _QWEN_BOUNDARY_SEARCH_SECONDS,
    window_seconds: float = _QWEN_BOUNDARY_WINDOW_SECONDS,
) -> list[tuple[int, int]]:
    """Split long ASR input near quiet boundaries without gaps or overlap."""
    total_samples = int(waveform.shape[0])
    if total_samples <= 0 or sample_rate <= 0:
        return []
    target_samples = max(1, int(round(chunk_seconds * sample_rate)))
    if total_samples <= target_samples:
        return [(0, total_samples)]

    search_samples = max(0, int(round(search_seconds * sample_rate)))
    window_samples = max(1, int(round(window_seconds * sample_rate)))
    ranges: list[tuple[int, int]] = []
    start = 0
    while total_samples - start > target_samples:
        target = start + target_samples
        left = max(start + 1, target - search_samples)
        right = min(total_samples, target + search_samples)
        # Only the boundary search window needs absolute amplitudes.  Building
        # abs(waveform) for a multi-hour recording used to duplicate the entire
        # decoded signal in memory.
        region = np.abs(waveform[left:right])
        if region.size <= window_samples:
            boundary = target
        else:
            # A cumulative sum finds the quietest short window in linear time.
            prefix = np.empty(region.size + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(region, dtype=np.float64, out=prefix[1:])
            energies = prefix[window_samples:] - prefix[:-window_samples]
            boundary = left + int(np.argmin(energies)) + (window_samples // 2)
        boundary = min(total_samples, max(start + 1, boundary))
        ranges.append((start, boundary))
        start = boundary

    ranges.append((start, total_samples))
    return ranges


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
) -> tuple[list[Sentence], str]:
    values = list(tokens)
    if not values:
        raise AsmrDubberError("识别出了文字，但所选 ASR 没有返回可用时间戳。")
    punctuated = restore_punctuation(values, full_text)
    sentences = split_timed_tokens(
        punctuated,
        pause_seconds=settings.pause_split_seconds,
        max_sentence_seconds=settings.max_sentence_seconds,
    )
    if not sentences:
        raise AsmrDubberError("没有在音频中识别到带有效时间戳的日语句子。")
    return sentences, language or "Japanese"


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _transcribe_qwen3(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    use_cuda = settings.asr_device.startswith("cuda")
    if use_cuda:
        require_cuda()
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise AsmrDubberError("缺少 qwen-asr；请在“设备与模型”页安装 Qwen3-ASR。") from exc

    if progress:
        progress(f"加载 Qwen3-ASR 与强制对齐器：{settings.asr_model}", 0, 1)
    torch.set_float32_matmul_precision("high")
    model = None
    try:
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "cuda:0" if use_cuda else "cpu"
        model_source, model_revision = resolve_transformers_model_source(settings.asr_model)
        aligner_source, aligner_revision = resolve_transformers_model_source(settings.aligner_model)
        model = Qwen3ASRModel.from_pretrained(
            model_source,
            revision=model_revision,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=settings.asr_batch_size,
            max_new_tokens=settings.asr_max_new_tokens,
            forced_aligner=aligner_source,
            forced_aligner_kwargs={
                "dtype": dtype,
                "device_map": device_map,
                "revision": aligner_revision,
            },
        )
        try:
            waveform, sample_rate = sf.read(
                analysis_audio,
                dtype="float32",
                always_2d=False,
            )
        except (OSError, RuntimeError) as exc:
            raise AsmrDubberError(f"无法读取 ASR 分析音频：{exc}") from exc
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1, dtype=np.float32)
        elif waveform.ndim != 1:
            raise AsmrDubberError(f"ASR 分析音频的维度无效：{waveform.ndim}")
        chunk_ranges = _qwen_chunk_ranges(waveform, int(sample_rate))
        if not chunk_ranges:
            raise AsmrDubberError("ASR 分析音频为空。")

        tokens: list[TimedToken] = []
        full_text_parts: list[str] = []
        languages: list[str] = []
        chunk_total = len(chunk_ranges)

        def consume_result(result: Any, chunk_index: int, start_sample: int) -> None:
            offset_seconds = start_sample / sample_rate
            text = str(getattr(result, "text", "") or "")
            language = str(getattr(result, "language", "") or "")
            if text:
                full_text_parts.append(text)
            if language and language not in languages:
                languages.append(language)
            for item in getattr(result, "time_stamps", None) or []:
                token = _offset_finite_token(
                    getattr(item, "text", ""),
                    getattr(item, "start_time", None),
                    getattr(item, "end_time", None),
                    offset_seconds,
                )
                if token is not None:
                    tokens.append(token)
            if progress:
                progress(
                    f"已完成日语识别：{chunk_index}/{chunk_total} 段",
                    chunk_index,
                    chunk_total,
                )

        # Batch size 1 remains the quality-first default.  Values above 1 are
        # an explicit user opt-in: Qwen's batch path is faster but floating
        # point differences can produce small punctuation/segmentation changes.
        requested_batch = max(1, settings.asr_batch_size)
        cursor = 0
        while cursor < chunk_total:
            batch_ranges = chunk_ranges[cursor : cursor + requested_batch]
            first_start, _ = batch_ranges[0]
            _, last_end = batch_ranges[-1]
            if progress:
                first_index = cursor + 1
                last_index = cursor + len(batch_ranges)
                segment_label = (
                    f"第 {first_index}/{chunk_total} 段"
                    if first_index == last_index
                    else f"第 {first_index}–{last_index}/{chunk_total} 段"
                )
                progress(
                    (
                        f"识别日语并生成时间戳：{segment_label}"
                        f"（{_clock(first_start / sample_rate)}–"
                        f"{_clock(last_end / sample_rate)}）"
                    ),
                    cursor,
                    chunk_total,
                )

            batch_audio = [
                (waveform[start_sample:end_sample], int(sample_rate))
                for start_sample, end_sample in batch_ranges
            ]
            audio_input: Any = batch_audio[0] if len(batch_audio) == 1 else batch_audio
            try:
                results = model.transcribe(
                    audio=audio_input,
                    language="Japanese",
                    return_time_stamps=True,
                )
            except torch.cuda.OutOfMemoryError:
                if len(batch_audio) == 1:
                    raise
                # An explicitly requested batch may be too large for a
                # particular model/GPU.  Falling back to one chunk at a time
                # preserves resumability and the quality-first behavior.
                torch.cuda.empty_cache()
                if progress:
                    progress(
                        "ASR 批处理显存不足，自动改为逐段识别",
                        cursor,
                        chunk_total,
                    )
                requested_batch = 1
                continue
            if len(results) != len(batch_ranges):
                raise AsmrDubberError(
                    f"Qwen3-ASR 请求 {len(batch_ranges)} 段但返回 {len(results)} 个结果。"
                )
            for offset, (result, (start_sample, _)) in enumerate(
                zip(results, batch_ranges, strict=True),
                start=1,
            ):
                consume_result(result, cursor + offset, start_sample)
            cursor += len(batch_ranges)

        return _finish_tokens(
            tokens,
            "".join(full_text_parts),
            ",".join(languages) or "Japanese",
            settings,
        )
    finally:
        del model
        _cleanup_cuda()


def _transcribe_faster_whisper(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise AsmrDubberError(
            "Faster-Whisper 未安装。请在兼容环境安装 faster-whisper 后重试。"
        ) from exc
    if progress:
        progress(f"加载 Faster-Whisper：{settings.asr_model}", 0, 1)
    model = None
    try:
        model_source = settings.asr_model
        is_kotoba_faster = model_source == "kotoba-tech/kotoba-whisper-v2.0-faster"
        if model_source == "large-v2":
            model_source = "Systran/faster-whisper-large-v2"
        model_source = resolve_model_source(model_source)
        model = WhisperModel(
            model_source,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
        transcribe_kwargs = {
            "language": "ja",
            "beam_size": settings.asr_beam_size,
            # Kotoba's distilled decoder has two layers, but its converted
            # config still carries the teacher model's alignment heads
            # (layers 7..25). Asking CTranslate2 for word alignment can
            # therefore crash the native process instead of raising Python.
            # Native segment timestamps remain valid and avoid that path.
            "word_timestamps": not is_kotoba_faster,
            "vad_filter": settings.asr_vad_filter,
            "vad_parameters": {"min_silence_duration_ms": settings.asr_vad_min_silence_ms},
            "condition_on_previous_text": (
                False if is_kotoba_faster else settings.asr_condition_on_previous_text
            ),
            "initial_prompt": settings.asr_initial_prompt or None,
        }
        if is_kotoba_faster:
            transcribe_kwargs["chunk_length"] = settings.asr_kotoba_chunk_seconds
        if settings.asr_batch_size > 1:
            # Faster-Whisper's official batched pipeline defaults VAD to on;
            # pass every ASMR-sensitive option explicitly so a speed opt-in
            # does not silently discard whispers or breathy speech.
            batched_model = BatchedInferencePipeline(model=model)
            segments, info = batched_model.transcribe(
                str(analysis_audio),
                without_timestamps=False,
                batch_size=settings.asr_batch_size,
                **transcribe_kwargs,
            )
        else:
            segments, info = model.transcribe(
                str(analysis_audio),
                **transcribe_kwargs,
            )
        values = list(segments)
        tokens: list[TimedToken] = []
        for segment in values:
            words = list(getattr(segment, "words", None) or [])
            if words:
                for word in words:
                    token = _finite_token(word.word, word.start, word.end)
                    if token:
                        tokens.append(token)
            else:
                token = _finite_token(segment.text, segment.start, segment.end)
                if token:
                    tokens.append(token)
        full_text = "".join(str(segment.text) for segment in values)
        return _finish_tokens(tokens, full_text, str(getattr(info, "language", "ja")), settings)
    finally:
        del model
        _cleanup_cuda()


def _transcribe_kotoba_whisper(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    configured_model = Path(settings.asr_model).expanduser()
    if not configured_model.is_dir() and cached_model_path(settings.asr_model) is None:
        raise AsmrDubberError(
            f"Kotoba-Whisper 模型尚未完整下载：{settings.asr_model}。"
            "识别流程不会在后台自动下载模型；请先安装该模型，"
            "或切换到已经完整安装的 ASR 模型。未完成的下载文件已保留。"
        )
    try:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
    except ImportError as exc:
        raise AsmrDubberError(
            "Kotoba-Whisper 运行环境未安装；请在“设备与模型”页一键安装。"
        ) from exc
    use_cuda = settings.asr_device.startswith("cuda")
    if use_cuda:
        require_cuda()
    if progress:
        progress(f"加载 Kotoba-Whisper：{settings.asr_model}", 0, 1)
    model = processor = pipe = None
    try:
        source, revision = resolve_transformers_model_source(settings.asr_model)
        dtype = torch.float16 if use_cuda else torch.float32
        processor = AutoProcessor.from_pretrained(source, revision=revision)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            source,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(settings.asr_device if use_cuda else "cpu")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=settings.asr_device if use_cuda else "cpu",
            chunk_length_s=settings.asr_kotoba_chunk_seconds,
            batch_size=max(1, settings.asr_batch_size),
            ignore_warning=True,
        )
        waveform, sample_rate = sf.read(
            analysis_audio,
            dtype="float32",
            always_2d=False,
        )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        result = _run_transformers_asr_pipeline(
            pipe,
            {"array": waveform, "sampling_rate": sample_rate},
            # Kotoba's distilled two-layer decoder inherits alignment-head
            # indices from the 32-layer Whisper teacher.  Word timestamp
            # extraction therefore indexes nonexistent layers.  Native
            # Whisper timestamp tokens remain valid and provide stable
            # segment ranges for our punctuation/pause splitter.
            return_timestamps=True,
            generate_kwargs={
                "language": "ja",
                "task": "transcribe",
                "condition_on_prev_tokens": settings.asr_condition_on_previous_text,
            },
        )
        if not isinstance(result, Mapping):
            raise AsmrDubberError("Kotoba-Whisper 返回的结果格式无效。")
        tokens: list[TimedToken] = []
        for item in result.get("chunks", []) or []:
            if not isinstance(item, Mapping):
                continue
            timestamp = item.get("timestamp")
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            token = _finite_token(item.get("text"), timestamp[0], timestamp[1])
            if token:
                tokens.append(token)
        return _finish_tokens(tokens, str(result.get("text", "")), "Japanese", settings)
    finally:
        del pipe, processor, model
        _cleanup_cuda()


def _crispasr_executable() -> Path:
    name = "crispasr.exe" if current_platform().is_windows else "crispasr"
    return portable_home() / "runtimes" / "crispasr" / "bin" / name


def _parakeet_model_path(model_id: str) -> Path:
    directory = portable_home() / "models" / "parakeet"
    filenames = {
        "nvidia/parakeet-tdt_ctc-0.6b-ja": "parakeet-tdt-0.6b-ja.gguf",
        (
            "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
        ): "parakeet-ctc-1.1b-ja-f16.gguf",
    }
    filename = filenames.get(model_id)
    if filename is None:
        raise AsmrDubberError(f"不支持的 Parakeet 日语模型：{model_id}")
    return directory / filename


def _parakeet_input_path(analysis_audio: Path, run_directory: Path) -> Path:
    """Give CrispASR an ASCII path on Windows without changing the analysis copy."""
    resolved = analysis_audio.resolve()
    if not current_platform().is_windows or str(resolved).isascii():
        return resolved
    staged = run_directory / "input.wav"
    try:
        # This is instant when the project and portable runtime share a volume.
        os.link(resolved, staged)
    except OSError:
        # Cross-volume projects cannot be hard-linked.  The temporary copy is
        # still isolated from the source and is removed with run_directory.
        shutil.copyfile(resolved, staged)
    return staged


def _parse_crispasr_payload(
    payload: Mapping[str, Any],
    settings: ProjectSettings,
) -> tuple[list[Sentence], str]:
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
    return _finish_tokens(tokens, "".join(full_text_parts), language, settings)


def _transcribe_parakeet(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    executable = _crispasr_executable()
    model_path = _parakeet_model_path(settings.asr_model)
    if not executable.is_file() or not model_path.is_file():
        raise AsmrDubberError(
            "Parakeet/CrispASR 未安装完整；请在“设备与模型”页选择 Parakeet 后点击安装。"
        )
    run_directory = portable_home() / "temp" / "asr" / f"parakeet-{uuid.uuid4().hex}"
    run_directory.mkdir(parents=True, exist_ok=False)
    result_base = run_directory / "result"
    result_file = result_base.with_suffix(".json")
    cache_directory = portable_home() / "cache" / "crispasr"
    cache_directory.mkdir(parents=True, exist_ok=True)
    native_input = _parakeet_input_path(analysis_audio, run_directory)
    command = [
        str(executable),
        "--backend",
        "parakeet",
        "--cache-dir",
        str(cache_directory),
        "-m",
        str(model_path),
        "-f",
        str(native_input),
        "-l",
        "ja",
        "-ojf",
        "-of",
        str(result_base),
        "-pp",
        "-t",
        str(max(1, min(os.cpu_count() or 4, settings.asr_batch_size * 4))),
    ]
    if settings.asr_model == "nvidia/parakeet-tdt_ctc-0.6b-ja":
        command.extend(("--parakeet-decoder", settings.asr_parakeet_decoder))
    if settings.asr_chunk_seconds > 0:
        command.extend(("--chunk-seconds", str(settings.asr_chunk_seconds)))
    if settings.asr_vad_filter:
        command.extend(
            (
                "--vad",
                "-vm",
                "silero",
                "-vsd",
                str(settings.asr_vad_min_silence_ms),
            )
        )
    if settings.asr_initial_prompt:
        command.extend(("--hotwords", settings.asr_initial_prompt))
    if not settings.asr_device.startswith("cuda"):
        command.append("--no-gpu")
    environment = isolated_runtime_environment("crispasr")
    environment["CRISPASR_CACHE_DIR"] = str(portable_home() / "cache" / "crispasr")
    if progress:
        progress(f"启动 Parakeet：{settings.asr_model}", 0, 1)
    process = None
    output_lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output,
            name="parakeet-output-reader",
            daemon=True,
        )
        reader.start()
        deadline = time.monotonic() + settings.asr_timeout_seconds
        last_heartbeat = 0.0
        output_closed = False
        while not output_closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise AsmrDubberError(
                    f"Parakeet 识别超过 {settings.asr_timeout_seconds:g} 秒，已安全停止。"
                    "可在设置中增加 ASR 超时，或缩短输入音频。"
                )
            try:
                line = output_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                now = time.monotonic()
                if progress and now - last_heartbeat >= 2:
                    elapsed = settings.asr_timeout_seconds - remaining
                    progress(f"Parakeet 正在识别（已运行 {elapsed:.0f} 秒）", 0, 1)
                    last_heartbeat = now
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                output_closed = True
                continue
            value = line.strip()
            if not value:
                continue
            output_lines.append(value)
            if progress and ("progress" in value.lower() or "%" in value):
                progress(f"Parakeet：{value[-160:]}", 0, 1)
        reader.join(timeout=1)
        return_code = process.wait()
        if return_code != 0:
            if len(output_lines) > 42:
                diagnostic_lines = output_lines[:12] + ["…"] + output_lines[-30:]
            else:
                diagnostic_lines = output_lines
            detail = "\n".join(diagnostic_lines).strip()
            raise AsmrDubberError(f"Parakeet 识别进程失败（退出码 {return_code}）：{detail[:4000]}")
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AsmrDubberError(f"Parakeet 没有生成有效结果：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise AsmrDubberError("Parakeet JSON 顶层格式无效。")
        return _parse_crispasr_payload(payload, settings)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        shutil.rmtree(run_directory, ignore_errors=True)
        _cleanup_cuda()


def _transcribe_openai_whisper(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    try:
        import whisper
    except ImportError as exc:
        raise AsmrDubberError("官方 OpenAI Whisper 未安装。请安装 openai-whisper。") from exc
    if progress:
        progress(f"加载 OpenAI Whisper：{settings.asr_model}", 0, 1)
    model = None
    try:
        model = whisper.load_model(settings.asr_model, device=settings.asr_device)
        result = model.transcribe(
            str(analysis_audio),
            language="ja",
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=settings.asr_condition_on_previous_text,
            initial_prompt=settings.asr_initial_prompt or None,
            beam_size=settings.asr_beam_size,
            verbose=False,
        )
        tokens: list[TimedToken] = []
        for segment in result.get("segments", []):
            words = segment.get("words") or []
            if words:
                for word in words:
                    token = _finite_token(word.get("word"), word.get("start"), word.get("end"))
                    if token:
                        tokens.append(token)
            else:
                token = _finite_token(segment.get("text"), segment.get("start"), segment.get("end"))
                if token:
                    tokens.append(token)
        return _finish_tokens(
            tokens, str(result.get("text", "")), str(result.get("language", "ja")), settings
        )
    finally:
        del model
        _cleanup_cuda()


def _transcribe_whisperx(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    try:
        import whisperx
    except ImportError as exc:
        raise AsmrDubberError("WhisperX 未安装。请在独立兼容环境安装 whisperx。") from exc
    if progress:
        progress(f"加载 WhisperX：{settings.asr_model}", 0, 2)
    model = align_model = None
    try:
        audio = whisperx.load_audio(str(analysis_audio))
        model = whisperx.load_model(
            settings.asr_model,
            settings.asr_device,
            compute_type=settings.asr_compute_type,
            language="ja",
        )
        result = model.transcribe(audio, batch_size=settings.asr_batch_size, language="ja")
        if progress:
            progress("WhisperX 日语强制对齐", 1, 2)
        align_model, metadata = whisperx.load_align_model(
            language_code=str(result.get("language", "ja")), device=settings.asr_device
        )
        aligned = whisperx.align(
            result.get("segments", []),
            align_model,
            metadata,
            audio,
            settings.asr_device,
            return_char_alignments=False,
        )
        tokens: list[TimedToken] = []
        for segment in aligned.get("segments", []):
            words = segment.get("words") or []
            if words:
                for word in words:
                    token = _finite_token(word.get("word"), word.get("start"), word.get("end"))
                    if token:
                        tokens.append(token)
            else:
                token = _finite_token(segment.get("text"), segment.get("start"), segment.get("end"))
                if token:
                    tokens.append(token)
        full_text = "".join(str(item.get("text", "")) for item in aligned.get("segments", []))
        return _finish_tokens(tokens, full_text, str(result.get("language", "ja")), settings)
    finally:
        del model, align_model
        _cleanup_cuda()


def _transcribe_funasr(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    try:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError as exc:
        raise AsmrDubberError("FunASR 未安装。请在兼容环境安装官方 funasr。") from exc
    if progress:
        progress(f"加载 FunASR：{settings.asr_model}", 0, 1)
    kwargs: dict[str, Any] = {"model": settings.asr_model, "device": settings.asr_device}
    if settings.asr_vad_filter:
        kwargs["vad_model"] = settings.asr_funasr_vad_model
        kwargs["punc_model"] = settings.asr_funasr_punc_model
    model = None
    try:
        model = AutoModel(**kwargs)
        results = model.generate(
            input=str(analysis_audio),
            batch_size_s=max(1, settings.asr_batch_size) * 60,
            language="ja",
        )
        if not results:
            raise AsmrDubberError("FunASR 没有返回结果。")
        result = results[0]
        full_text = rich_transcription_postprocess(str(result.get("text", "")))
        tokens: list[TimedToken] = []
        for segment in result.get("sentence_info", []) or []:
            text = segment.get("sentence") or segment.get("text") or ""
            text = rich_transcription_postprocess(str(text))
            token = _finite_token(
                text, float(segment.get("start", 0)) / 1000, float(segment.get("end", 0)) / 1000
            )
            if token:
                tokens.append(token)
        return _finish_tokens(tokens, full_text, "ja", settings)
    finally:
        del model
        _cleanup_cuda()


def _mapping_tokens(payload: Mapping[str, Any]) -> list[TimedToken]:
    tokens: list[TimedToken] = []
    for item in payload.get("words", []) or []:
        if isinstance(item, Mapping):
            token = _finite_token(
                item.get("word") or item.get("text"), item.get("start"), item.get("end")
            )
            if token:
                tokens.append(token)
    if tokens:
        return tokens
    for item in payload.get("segments", []) or []:
        if isinstance(item, Mapping):
            token = _finite_token(item.get("text"), item.get("start"), item.get("end"))
            if token:
                tokens.append(token)
    return tokens


def _transcribe_openai_compatible(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
) -> tuple[list[Sentence], str]:
    import httpx

    base = settings.asr_api_base_url.rstrip("/")
    url = base if base.endswith("/audio/transcriptions") else f"{base}/audio/transcriptions"
    key = saved_service_key(f"asr:{settings.asr_backend}")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if progress:
        progress(f"调用 ASR 服务：{url}", 0, 1)
    try:
        with analysis_audio.open("rb") as handle:
            response = httpx.post(
                url,
                headers=headers,
                files={"file": (analysis_audio.name, handle, "audio/wav")},
                data={
                    "model": settings.asr_model,
                    "language": "ja",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                    "prompt": settings.asr_initial_prompt,
                },
                timeout=settings.asr_timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        raise AsmrDubberError(
            f"ASR HTTP 服务失败：{exc}{': ' + detail[:500] if detail else ''}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AsmrDubberError("ASR 服务返回的 JSON 不是对象。")
    return _finish_tokens(
        _mapping_tokens(payload),
        str(payload.get("text", "")),
        str(payload.get("language", "ja")),
        settings,
    )


def transcribe_japanese(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None = None,
) -> tuple[list[Sentence], str]:
    """Dispatch Japanese ASR through a capability-registered backend."""
    backend = settings.asr_backend
    if backend not in ASR_BACKENDS:
        raise AsmrDubberError(f"未知 ASR 模型后端：{backend}")
    runners = {
        "parakeet_nemo": _transcribe_parakeet,
        "kotoba_whisper": _transcribe_kotoba_whisper,
        "qwen3_asr": _transcribe_qwen3,
        "faster_whisper": _transcribe_faster_whisper,
        "openai_whisper": _transcribe_openai_whisper,
        "whisperx": _transcribe_whisperx,
        "funasr": _transcribe_funasr,
        "openai_compatible_asr": _transcribe_openai_compatible,
    }
    result = runners[backend](analysis_audio, settings, progress)
    if progress:
        progress(f"识别完成：{len(result[0])} 句", 1, 1)
    return result
