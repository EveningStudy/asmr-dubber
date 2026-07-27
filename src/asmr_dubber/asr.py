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
from .models import ProjectSettings, Sentence
from .platforms import current_platform, isolated_runtime_environment, portable_home
from .segmentation import TimedToken, restore_punctuation, split_timed_tokens
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
) -> tuple[list[Sentence], str]:
    values = list(tokens)
    if not values:
        raise AsmrDubberError("识别出了文字，但所选 ASR（语音识别）后端没有返回可用时间戳。")
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
            "vad_filter": settings.asr_vad_mode == "backend",
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
            "或切换到已经完整安装的 ASR（语音识别）模型。未完成的下载文件已保留。"
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
        sample_rate, chunk_ranges = _audio_file_chunk_ranges(
            analysis_audio,
            settings.asr_kotoba_chunk_seconds,
        )
        tokens: list[TimedToken] = []
        full_text: list[str] = []
        with sf.SoundFile(analysis_audio) as audio:
            for index, (start_sample, end_sample) in enumerate(chunk_ranges, start=1):
                audio.seek(start_sample)
                waveform = audio.read(
                    end_sample - start_sample,
                    dtype="float32",
                    always_2d=False,
                )
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                if progress:
                    progress(
                        f"Kotoba-Whisper 正在识别第 {index}/{len(chunk_ranges)} 段",
                        index - 1,
                        len(chunk_ranges),
                    )
                result = _run_transformers_asr_pipeline(
                    pipe,
                    {"array": waveform, "sampling_rate": sample_rate},
                    return_timestamps=True,
                    generate_kwargs={
                        "language": "ja",
                        "task": "transcribe",
                        "condition_on_prev_tokens": settings.asr_condition_on_previous_text,
                    },
                )
                if not isinstance(result, Mapping):
                    raise AsmrDubberError("Kotoba-Whisper 返回的结果格式无效。")
                offset = start_sample / sample_rate
                full_text.append(str(result.get("text", "")))
                for item in result.get("chunks", []) or []:
                    if not isinstance(item, Mapping):
                        continue
                    timestamp = item.get("timestamp")
                    if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                        continue
                    token = _offset_finite_token(
                        item.get("text"), timestamp[0], timestamp[1], offset
                    )
                    if token:
                        tokens.append(token)
        return _finish_tokens(tokens, "".join(full_text), "Japanese", settings)
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
    return _finish_tokens(tokens, full_text, language, settings)


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
    cache_directory = portable_home() / "cache" / "crispasr"
    cache_directory.mkdir(parents=True, exist_ok=True)
    native_input = _parakeet_input_path(analysis_audio, run_directory)
    base_command = [
        str(executable),
        "--backend",
        "parakeet",
        "--cache-dir",
        str(cache_directory),
        "-m",
        str(model_path),
        "-l",
        "ja",
        "-ojf",
        "-pp",
        "-t",
        str(max(1, min(os.cpu_count() or 4, settings.asr_batch_size * 4))),
    ]
    if settings.asr_model == "nvidia/parakeet-tdt_ctc-0.6b-ja":
        base_command.extend(("--parakeet-decoder", settings.asr_parakeet_decoder))
    if settings.asr_vad_mode == "backend":
        base_command.extend(
            (
                "--vad",
                "-vm",
                "silero",
                "-vsd",
                str(settings.asr_vad_min_silence_ms),
            )
        )
    if settings.asr_initial_prompt:
        base_command.extend(("--hotwords", settings.asr_initial_prompt))
    if not settings.asr_device.startswith("cuda"):
        base_command.append("--no-gpu")
    environment = isolated_runtime_environment("crispasr")
    environment["CRISPASR_CACHE_DIR"] = str(portable_home() / "cache" / "crispasr")
    # CrispASR currently starts once per application-managed chunk. A larger,
    # explicit default drastically reduces model reloads on long recordings,
    # while the validated 15–600 second range still bounds peak memory.
    chunk_seconds = settings.asr_chunk_seconds or _PARAKEET_AUTO_CHUNK_SECONDS
    sample_rate, chunk_ranges = _audio_file_chunk_ranges(native_input, chunk_seconds)
    if not chunk_ranges or chunk_ranges[-1][1] <= 0:
        raise AsmrDubberError("Parakeet 输入音频为空。")
    if progress:
        progress(f"Parakeet 将音频分为 {len(chunk_ranges)} 段", 0, len(chunk_ranges))
    deadline = time.monotonic() + settings.asr_timeout_seconds
    active_process: subprocess.Popen[str] | None = None

    def run_chunk(command: list[str], result_file: Path, index: int) -> Mapping[str, Any]:
        nonlocal active_process
        output_lines: list[str] = []
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
        active_process = process
        stdout = process.stdout
        assert stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output,
            name="parakeet-output-reader",
            daemon=True,
        )
        reader.start()
        last_heartbeat = 0.0
        output_closed = False
        while not output_closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise AsmrDubberError(
                    f"Parakeet 识别超过 {settings.asr_timeout_seconds:g} 秒，已安全停止。"
                    "可在设置中增加 ASR（语音识别）超时，或缩短输入音频。"
                )
            try:
                line = output_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                now = time.monotonic()
                if progress and now - last_heartbeat >= 2:
                    elapsed = settings.asr_timeout_seconds - remaining
                    progress(
                        f"Parakeet 正在识别第 {index}/{len(chunk_ranges)} 段"
                        f"（已运行 {elapsed:.0f} 秒）",
                        index - 1,
                        len(chunk_ranges),
                    )
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
                progress(
                    f"Parakeet 第 {index}/{len(chunk_ranges)} 段：{value[-140:]}",
                    index - 1,
                    len(chunk_ranges),
                )
        reader.join(timeout=1)
        return_code = process.wait()
        active_process = None
        if return_code != 0:
            if len(output_lines) > 42:
                diagnostic_lines = [*output_lines[:12], "…", *output_lines[-30:]]
            else:
                diagnostic_lines = output_lines
            detail = "\n".join(diagnostic_lines).strip()
            raise AsmrDubberError(
                f"Parakeet 第 {index}/{len(chunk_ranges)} 段识别失败"
                f"（退出码 {return_code}）：{detail[:4000]}"
            )
        if not result_file.is_file():
            # CrispASR returns success without creating JSON when a chunk is
            # entirely silent or contains no decodable speech. This is normal
            # for quiet ASMR gaps; skip it and continue with adjacent chunks.
            return {"crispasr": {"language": "ja"}, "transcription": []}
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AsmrDubberError(
                f"Parakeet 第 {index}/{len(chunk_ranges)} 段没有生成有效结果：{exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise AsmrDubberError("Parakeet JSON 顶层格式无效。")
        return payload

    all_tokens: list[TimedToken] = []
    full_text_parts: list[str] = []
    language = "Japanese"
    try:
        with sf.SoundFile(native_input) as source_audio:
            for index, (start_sample, end_sample) in enumerate(chunk_ranges, start=1):
                if len(chunk_ranges) == 1:
                    chunk_file = native_input
                else:
                    chunk_file = run_directory / f"chunk-{index:04d}.wav"
                    source_audio.seek(start_sample)
                    waveform = source_audio.read(
                        end_sample - start_sample,
                        dtype="float32",
                        always_2d=False,
                    )
                    sf.write(chunk_file, waveform, sample_rate, subtype="PCM_16")
                result_base = run_directory / f"result-{index:04d}"
                command = [
                    *base_command,
                    "-f",
                    str(chunk_file),
                    "-of",
                    str(result_base),
                ]
                if progress:
                    progress(
                        f"Parakeet 正在识别第 {index}/{len(chunk_ranges)} 段",
                        index - 1,
                        len(chunk_ranges),
                    )
                payload = run_chunk(command, result_base.with_suffix(".json"), index)
                tokens, full_text, chunk_language = _crispasr_payload_tokens(payload)
                offset_seconds = start_sample / sample_rate
                all_tokens.extend(
                    TimedToken(
                        text=token.text,
                        start_seconds=token.start_seconds + offset_seconds,
                        end_seconds=token.end_seconds + offset_seconds,
                    )
                    for token in tokens
                )
                full_text_parts.append(full_text)
                if chunk_language:
                    language = chunk_language
                if chunk_file != native_input:
                    chunk_file.unlink(missing_ok=True)
                if progress:
                    progress(
                        f"Parakeet 已完成第 {index}/{len(chunk_ranges)} 段",
                        index,
                        len(chunk_ranges),
                    )
        return _finish_tokens(all_tokens, "".join(full_text_parts), language, settings)
    finally:
        if active_process is not None and active_process.poll() is None:
            active_process.kill()
        shutil.rmtree(run_directory, ignore_errors=True)
        _cleanup_cuda()


def transcribe_japanese(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None = None,
) -> tuple[list[Sentence], str]:
    """Run one of the three deliberately supported recognition families."""

    runners = {
        "parakeet_nemo": _transcribe_parakeet,
        "kotoba_whisper": _transcribe_kotoba_whisper,
        "faster_whisper": _transcribe_faster_whisper,
    }
    try:
        runner = runners[settings.asr_backend]
    except KeyError as exc:
        raise AsmrDubberError(
            f"不支持的 ASR（语音识别）后端：{settings.asr_backend}。"
            "请选择 Parakeet、Kotoba-Whisper 或 Faster-Whisper。"
        ) from exc
    if settings.asr_vad_mode == "asmr":
        segments = detect_asmr_speech(
            analysis_audio,
            threshold=settings.asr_asmr_vad_threshold,
            min_speech_ms=settings.asr_asmr_vad_min_speech_ms,
            min_silence_ms=settings.asr_asmr_vad_min_silence_ms,
            speech_pad_ms=settings.asr_asmr_vad_speech_pad_ms,
            progress=progress,
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
            )
            inner_settings = settings.model_copy(
                update={"asr_vad_mode": "off", "asr_vad_filter": False}
            )
            sentences, language = runner(condensed, inner_settings, progress)
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
        result = runner(analysis_audio, settings, progress)
    if progress:
        progress(f"语音识别完成：{len(result[0])} 句", 1, 1)
    return result
