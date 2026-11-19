from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import soundfile as sf

from .api_contracts import APIContractError, bearer_headers, endpoint_url, merge_extra_body
from .api_logging import safe_api_url
from .errors import AsmrDubberError
from .languages import SpeechSourceLanguage
from .models import ProjectSettings, Sentence
from .segmentation import TimedToken, restore_punctuation
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)

Progress = Callable[[str, int, int], None]
_GENERIC_API_MAX_BYTES = 20 * 1024 * 1024
_GENERIC_API_CHUNK_TARGET_BYTES = 16 * 1024 * 1024


def _prepare_generic_api_uploads(
    analysis_audio: Path,
    *,
    probe_only: bool,
    temp_dir: Path,
) -> list[tuple[Path, str, float]]:
    if analysis_audio.stat().st_size <= _GENERIC_API_MAX_BYTES:
        return [(analysis_audio, "audio/wav", 0.0)]
    if probe_only:
        probe_wav = temp_dir / "probe.wav"
        with sf.SoundFile(analysis_audio) as audio:
            frames = min(audio.frames, max(1, audio.samplerate))
            samples = audio.read(frames, dtype="float32", always_2d=True)
            sf.write(probe_wav, samples, audio.samplerate, subtype="PCM_16", format="WAV")
        return [(probe_wav, "audio/wav", 0.0)]
    try:
        from . import audio as audio_mod

        compact_mp3 = temp_dir / "analysis.mp3"
        audio_mod._run_ffmpeg(
            [
                "-y",
                "-i",
                str(analysis_audio),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
                str(compact_mp3),
            ]
        )
        if compact_mp3.is_file() and compact_mp3.stat().st_size <= _GENERIC_API_MAX_BYTES:
            return [(compact_mp3, "audio/mpeg", 0.0)]
    except Exception:
        pass

    from . import asr

    info = sf.info(analysis_audio)
    bytes_per_second = max(1, info.samplerate * info.channels * 2)
    chunk_seconds = max(30.0, _GENERIC_API_CHUNK_TARGET_BYTES / bytes_per_second)
    sample_rate, ranges = asr._audio_file_chunk_ranges(analysis_audio, chunk_seconds)
    uploads: list[tuple[Path, str, float]] = []
    with sf.SoundFile(analysis_audio) as audio:
        for index, (start, end) in enumerate(ranges, start=1):
            audio.seek(start)
            samples = audio.read(end - start, dtype="float32", always_2d=True)
            chunk_path = temp_dir / f"chunk_{index:04d}.wav"
            sf.write(chunk_path, samples, sample_rate, subtype="PCM_16", format="WAV")
            uploads.append((chunk_path, "audio/wav", start / sample_rate))
    return uploads


def _transcribe_generic_api(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
    api_key: str | None = None,
    probe_only: bool = False,
) -> tuple[list[Sentence], str]:
    import httpx

    from . import asr

    if progress:
        progress(f"上传音频到通用 ASR API：{settings.asr_model}", 0, 1)
    try:
        url = endpoint_url(settings.asr_api_base_url, "/audio/transcriptions")
        base_payload: dict[str, object] = {
            "model": settings.asr_model,
            "language": source_language,
            "response_format": "verbose_json",
        }
        if settings.asr_initial_prompt and settings.asr_initial_prompt.strip():
            base_payload["prompt"] = settings.asr_initial_prompt.strip()
        payload = merge_extra_body(
            base_payload,
            settings.asr_api_extra_body,
            label="ASR API 附加请求参数",
            reserved={"file", "model"},
        )
    except APIContractError as exc:
        raise AsmrDubberError(str(exc)) from exc
    form = {
        key: (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        )
        for key, value in payload.items()
        if value is not None
    }
    if "timestamp_granularities" not in form:
        form.setdefault("timestamp_granularities[]", "segment")
    key = (api_key or "").strip() or asr.saved_service_key("asr:generic_asr_api")
    asr.logger.info(
        "ASR 后端开始：后端=generic_asr_api 模型=%s 地址=%s",
        settings.asr_model,
        safe_api_url(url),
    )
    client = asr.logged_http_client(
        "通用 ASR API",
        headers=bearer_headers(key),
        timeout=settings.asr_timeout_seconds,
    )
    callback_signal = register_cancel_callback(client.close, cancel_event)
    results: list[tuple[Mapping[str, object], float]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="asmr-generic-api-") as tmp_name:
            uploads = _prepare_generic_api_uploads(
                analysis_audio,
                probe_only=probe_only,
                temp_dir=Path(tmp_name),
            )
            for upload_path, mime_type, offset in uploads:
                check_cancelled(cancel_event)
                with upload_path.open("rb") as handle:
                    response = client.post(
                        url,
                        data=form,
                        files={"file": (upload_path.name, handle, mime_type)},
                    )
                if response.is_error:
                    detail = response.text[:500]
                    raise AsmrDubberError(
                        f"通用 ASR API 拒绝请求（HTTP {response.status_code}）：{detail}"
                    )
                try:
                    result = response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise AsmrDubberError("通用 ASR API 返回的不是有效 JSON。") from exc
                if not isinstance(result, Mapping):
                    raise AsmrDubberError("通用 ASR API JSON 顶层必须是对象。")
                results.append((result, offset))
    except httpx.HTTPError as exc:
        check_cancelled(cancel_event)
        raise AsmrDubberError(f"通用 ASR API 请求失败：{exc}") from exc
    finally:
        unregister_cancel_callback(client.close, callback_signal)
        client.close()
    language = str(results[0][0].get("language") or source_language) if results else source_language
    if probe_only:
        return [], language
    tokens: list[TimedToken] = []
    text_parts: list[str] = []
    for result, offset in results:
        chunk_tokens: list[TimedToken] = []
        raw_items = result.get("segments") or result.get("words") or []
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, Mapping):
                    continue
                token = asr._offset_finite_token(
                    item.get("text") or item.get("word"),
                    item.get("start"),
                    item.get("end"),
                    offset,
                )
                if token is not None:
                    chunk_tokens.append(token)
        chunk_text = str(result.get("text") or "").strip() or "".join(
            token.text for token in chunk_tokens
        )
        if chunk_text:
            text_parts.append(chunk_text)
        if len(results) > 1 and chunk_tokens and chunk_text:
            tokens.extend(restore_punctuation(chunk_tokens, chunk_text))
        else:
            tokens.extend(chunk_tokens)
    joiner = " " if source_language == "en" else ""
    full_text = joiner.join(text_parts)
    if not full_text:
        if asr._REVIEW_MODELS.get() is not None:
            return [], language
        raise AsmrDubberError("通用 ASR API 没有返回可用文字。")
    return asr._finish_tokens(
        tokens,
        full_text,
        language,
        settings,
        source_language,
        restore=len(results) == 1,
    )


def _transcribe_faster_whisper(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
) -> tuple[list[Sentence], str]:
    from . import asr

    check_cancelled(cancel_event)
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
        if source_language != "ja" and is_kotoba_faster:
            raise AsmrDubberError(
                "Kotoba-Whisper 是日语模型；英语项目请选择其它 Faster-Whisper 模型。"
            )
        if model_source == "large-v2":
            model_source = "Systran/faster-whisper-large-v2"
        model_source = asr.resolve_model_source(model_source)
        compute_type = settings.asr_compute_type
        if not settings.asr_device.startswith("cuda") and compute_type in {
            "float16",
            "int8_float16",
        }:
            # CTranslate2 cannot execute FP16 kernels on most desktop CPUs.
            # Keep projects portable when a user switches an existing CUDA
            # configuration to CPU instead of failing during model loading.
            compute_type = "int8"
            asr.logger.info(
                "Faster-Whisper CPU 自动调整计算精度：%s -> %s",
                settings.asr_compute_type,
                compute_type,
            )
            if progress:
                progress("Faster-Whisper CPU 自动使用 int8 计算精度", 0, 1)
        model = asr._session_model(
            f"whisper:{model_source}:{settings.asr_device}:{compute_type}",
            lambda: WhisperModel(
                model_source,
                device=settings.asr_device,
                compute_type=compute_type,
            ),
        )
        transcribe_kwargs = {
            "language": source_language,
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
        values = []
        for segment in segments:
            check_cancelled(cancel_event)
            values.append(segment)
            if progress:
                duration = float(getattr(info, "duration", 0) or 0)
                consumed = min(duration, max(0.0, float(getattr(segment, "end", 0) or 0)))
                progress(
                    f"Faster-Whisper 已识别 {consumed:.1f}/{duration:.1f} 秒",
                    round(consumed * 1000),
                    max(1, round(duration * 1000)),
                )
        tokens: list[TimedToken] = []
        for segment in values:
            words = list(getattr(segment, "words", None) or [])
            if words:
                for word in words:
                    token = asr._finite_token(word.word, word.start, word.end)
                    if token:
                        tokens.append(token)
            else:
                token = asr._finite_token(segment.text, segment.start, segment.end)
                if token:
                    tokens.append(token)
        full_text = "".join(str(segment.text) for segment in values)
        return asr._finish_tokens(
            tokens,
            full_text,
            str(getattr(info, "language", source_language)),
            settings,
            source_language,
        )
    finally:
        del model
        asr._cleanup_cuda()


def _transcribe_kotoba_whisper(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
) -> tuple[list[Sentence], str]:
    from . import asr

    check_cancelled(cancel_event)
    if source_language != "ja":
        raise AsmrDubberError("Kotoba-Whisper 仅支持日语；英语项目请使用 Faster-Whisper。")
    configured_model = Path(settings.asr_model).expanduser()
    if not configured_model.is_dir() and asr.cached_model_path(settings.asr_model) is None:
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
        asr.require_cuda()
    if progress:
        progress(f"加载 Kotoba-Whisper：{settings.asr_model}", 0, 1)
    model = processor = pipe = None
    try:
        source, revision = asr.resolve_transformers_model_source(settings.asr_model)
        dtype = torch.float16 if use_cuda else torch.float32
        processor = asr._session_model(
            f"processor:{source}:{revision}",
            lambda: AutoProcessor.from_pretrained(source, revision=revision),
        )
        model = asr._session_model(
            f"kotoba:{source}:{revision}:{settings.asr_device}",
            lambda: AutoModelForSpeechSeq2Seq.from_pretrained(
                source,
                revision=revision,
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(settings.asr_device if use_cuda else "cpu"),
        )
        pipe = asr._session_model(
            f"pipeline:{source}:{revision}:{settings.asr_device}",
            lambda model=model, processor=processor: pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=settings.asr_device if use_cuda else "cpu",
                chunk_length_s=settings.asr_kotoba_chunk_seconds,
                batch_size=max(1, settings.asr_batch_size),
                ignore_warning=True,
            ),
        )
        sample_rate, chunk_ranges = asr._audio_file_chunk_ranges(
            analysis_audio,
            settings.asr_kotoba_chunk_seconds,
        )
        tokens: list[TimedToken] = []
        full_text: list[str] = []
        with sf.SoundFile(analysis_audio) as audio:
            for index, (start_sample, end_sample) in enumerate(chunk_ranges, start=1):
                check_cancelled(cancel_event)
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
                result = asr._run_transformers_asr_pipeline(
                    pipe,
                    {"array": waveform, "sampling_rate": sample_rate},
                    return_timestamps=True,
                    generate_kwargs={
                        "language": "ja",
                        "task": "transcribe",
                        "condition_on_prev_tokens": settings.asr_condition_on_previous_text,
                    },
                )
                check_cancelled(cancel_event)
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
                    token = asr._offset_finite_token(
                        item.get("text"), timestamp[0], timestamp[1], offset
                    )
                    if token:
                        tokens.append(token)
        return asr._finish_tokens(tokens, "".join(full_text), "Japanese", settings, source_language)
    finally:
        del pipe, processor, model
        asr._cleanup_cuda()
