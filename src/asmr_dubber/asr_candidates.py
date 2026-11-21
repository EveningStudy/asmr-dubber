"""Qwen3-ASR and anime-whisper recognition backends.

Both models give better text than Parakeet on whispered Japanese, but neither
produces dependable timestamps on its own.  Each backend therefore pairs the
recogniser with the pinned Qwen3 ForcedAligner, which returns per-character
times; those feed the same punctuation restore and sentence splitter as every
other backend (including the onomatopoeia split in ``segmentation``).

* ``qwen3_asr``: Qwen3-ASR (base or a fine-tune in the same format) through the
  ``qwen_asr`` package, aligned per window by its built-in aligner hook.
* ``anime_whisper``: litagin/anime-whisper (a kotoba-whisper fine-tune trained
  on anime/game voices).  It was trained on single short lines without
  timestamps, so the audio is first cut with the Japanese ASMR VAD, every
  speech group is recognised on its own, and each group is aligned afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .asr_progress import model_heartbeat
from .asr_token_timing import active_frames, group_tokens
from .constants import DEFAULT_ALIGNER_MODEL
from .environment import cached_model_path, require_cuda, resolve_transformers_model_source
from .errors import AsmrDubberError, OperationCancelledError
from .languages import SpeechSourceLanguage
from .models import ProjectSettings, Sentence
from .segmentation import TimedToken, restore_punctuation
from .task_control import CancellationSignal, check_cancelled

logger = logging.getLogger(__name__)
Progress = Callable[[str, int, int], None]

QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN3_ASR_ANIME_MODEL = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
ANIME_WHISPER_MODEL = "litagin/anime-whisper"
_QWEN_LANGUAGES = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
# qwen_asr aligns at most 180 s per piece; stay below it so our progress and
# cancellation checks see every window.
_QWEN_WINDOW_SECONDS = 150.0
# anime-whisper was trained on single game lines and stops early on long clips
# (a 20 s clip lost its last two sentences): keep every clip short.
_ANIME_GROUP_GAP_SECONDS = 0.35
_ANIME_GROUP_MAX_SECONDS = 8.0
_SPLIT_SEARCH_FROM = 0.5  # cut a long clip in its second half, at the quietest 100 ms
_ALIGN_BATCH = 8


def _require_model(model_id: str, label: str) -> None:
    if Path(model_id).expanduser().is_dir() or cached_model_path(model_id) is not None:
        return
    raise AsmrDubberError(
        f"{label} 模型尚未完整下载：{model_id}。识别流程不会在后台自动下载模型；"
        "请先在“设备与模型”页安装，或切换到已经安装的识别模型。"
    )


def _pretrained(torch: Any, settings: ProjectSettings, model_id: str) -> tuple[str, dict]:
    from .forced_alignment import _alignment_device_map, _alignment_dtype

    source, revision = resolve_transformers_model_source(model_id)
    kwargs: dict[str, Any] = {
        "dtype": _alignment_dtype(torch, settings.asr_device),
        "device_map": _alignment_device_map(settings.asr_device),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if Path(source).is_dir():
        kwargs["local_files_only"] = True
    return source, kwargs


def _aligner_id(settings: ProjectSettings) -> str:
    model_id = settings.aligner_model or DEFAULT_ALIGNER_MODEL
    _require_model(model_id, "Qwen3 ForcedAligner")
    return model_id


def _vad_clips(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None,
) -> tuple[int, list[tuple[int, int]], list[np.ndarray]]:
    """Cut speech with the Japanese ASMR VAD into short clips (<= 15 s each)."""
    from .vad import detect_asmr_speech

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
    sample_rate = sf.info(analysis_audio).samplerate
    groups = _speech_groups(segments, sample_rate)
    if not groups:
        from .asr import _REVIEW_MODELS

        if _REVIEW_MODELS.get() is not None:
            return sample_rate, [], []
        raise AsmrDubberError("日语 ASMR 专用 VAD 没有检测到语音；请降低 VAD 阈值后重试。")
    pieces: list[tuple[int, int]] = []
    clips = []
    with sf.SoundFile(analysis_audio) as audio:
        for start, end in groups:
            audio.seek(start)
            clip = audio.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
            for a, b in _split_long_clip(clip, sample_rate):
                pieces.append((start + a, start + b))
                clips.append(_level_whisper_clip(clip[a:b], sample_rate))
    return sample_rate, pieces, clips


def _level_whisper_clip(clip: np.ndarray, sample_rate: int) -> np.ndarray:
    """Bring very quiet whispered VAD clips up into the ASR encoder's sweet spot."""
    samples = np.asarray(clip, dtype=np.float64)
    if samples.size < max(64, sample_rate // 20):
        return clip
    hop = max(1, sample_rate // 50)
    count = samples.size // hop
    if count < 2:
        return clip
    rms = np.sqrt(np.mean(samples[: count * hop].reshape(count, hop) ** 2, axis=1))
    peak_rms = float(rms.max(initial=0.0))
    if peak_rms <= 1e-6:
        return clip
    active_rms = float(np.sqrt(np.mean(rms[rms >= peak_rms * 0.2] ** 2)))
    target = 10 ** (-20.0 / 20.0)
    if active_rms >= target * 0.8:
        return clip
    gain = min(target / max(active_rms, 1e-6), 12.0)
    boosted = np.tanh(samples * gain * 1.05) / 1.05
    return boosted.astype(np.float32)


def _split_long_clip(clip: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Cut a continuous VAD segment longer than the clip limit at its quietest moments."""
    limit = round(_ANIME_GROUP_MAX_SECONDS * sample_rate)
    frame = max(1, sample_rate // 10)
    cuts = [0]
    while len(clip) - cuts[-1] > limit:
        lo = cuts[-1] + round(limit * _SPLIT_SEARCH_FROM)
        hi = cuts[-1] + limit
        frames = (hi - lo) // frame
        if frames < 1:
            cuts.append(hi)
            continue
        energy = np.square(clip[lo : lo + frames * frame].astype(np.float64)).reshape(frames, frame)
        cuts.append(lo + int(np.argmin(energy.mean(axis=1))) * frame + frame // 2)
    cuts.append(len(clip))
    return [(a, b) for a, b in pairwise(cuts) if b > a]


def transcribe_qwen3_asr(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
) -> tuple[list[Sentence], str]:
    from .asr import _audio_file_chunk_ranges, _cleanup_cuda, _finish_tokens, _session_model

    check_cancelled(cancel_event)
    language = _QWEN_LANGUAGES.get(source_language)
    if language is None:
        raise AsmrDubberError(f"Qwen3-ASR 不支持该源语言：{source_language}")
    _require_model(settings.asr_model, "Qwen3-ASR")
    aligner = _aligner_id(settings)
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise AsmrDubberError("Qwen3-ASR 运行依赖尚未安装；请重新运行“进阶”安装。") from exc
    if settings.asr_device.startswith("cuda"):
        require_cuda()
    if progress:
        progress(f"加载 Qwen3-ASR：{settings.asr_model}", 0, 1)
    source, model_kwargs = _pretrained(torch, settings, settings.asr_model)
    aligner_source, aligner_kwargs = _pretrained(torch, settings, aligner)
    model = None
    try:
        with model_heartbeat(progress, f"加载 Qwen3-ASR：{settings.asr_model}"):
            model = _session_model(
                f"qwen3-asr:{source}:{aligner_source}:{settings.asr_device}",
                lambda: Qwen3ASRModel.from_pretrained(
                    source,
                    forced_aligner=aligner_source,
                    forced_aligner_kwargs=aligner_kwargs,
                    max_inference_batch_size=max(1, settings.asr_batch_size),
                    max_new_tokens=1024,
                    **model_kwargs,
                ),
            )
        tokens: list[TimedToken] = []
        texts: list[str] = []
        if source_language == "ja":
            # Whispered ASMR aligns badly over minutes of audio; short VAD clips
            # keep the aligner on the words it is given.
            sample_rate, groups, clips = _vad_clips(
                analysis_audio, settings, progress, cancel_event
            )
            batch = max(1, settings.asr_batch_size) * 4
            for first in range(0, len(clips), batch):
                check_cancelled(cancel_event)
                if progress:
                    progress("Qwen3-ASR 正在识别语音段", first, len(clips))
                indices = range(first, min(len(clips), first + batch))
                results = model.transcribe(
                    audio=[(clips[i], sample_rate) for i in indices],
                    context=settings.asr_initial_prompt or "",
                    language=language,
                    return_time_stamps=True,
                )
                for i, result in zip(indices, results, strict=True):
                    text = (result.text or "").strip()
                    if not text:
                        continue
                    texts.append(text)
                    clip_tokens = group_tokens(
                        text,
                        getattr(result.time_stamps, "items", None),
                        groups[i],
                        sample_rate,
                        clips[i],
                    )
                    tokens.extend(restore_punctuation(clip_tokens, text))
            return _finish_tokens(
                tokens, "".join(texts), language, settings, source_language, restore=False
            )
        sample_rate, ranges = _audio_file_chunk_ranges(analysis_audio, _QWEN_WINDOW_SECONDS)
        with sf.SoundFile(analysis_audio) as audio:
            for index, (start, end) in enumerate(ranges, start=1):
                check_cancelled(cancel_event)
                if progress:
                    progress(
                        f"Qwen3-ASR 正在识别第 {index}/{len(ranges)} 段", index - 1, len(ranges)
                    )
                audio.seek(start)
                waveform = audio.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
                if not np.any(waveform):
                    continue
                result = model.transcribe(
                    audio=(waveform, sample_rate),
                    context=settings.asr_initial_prompt or "",
                    language=language,
                    return_time_stamps=True,
                )[0]
                text = (result.text or "").strip()
                if not text:
                    continue
                texts.append(text)
                window_tokens = group_tokens(
                    text, getattr(result.time_stamps, "items", None), (start, end), sample_rate
                )
                tokens.extend(restore_punctuation(window_tokens, text))
        joiner = " " if source_language == "en" else ""
        return _finish_tokens(
            tokens, joiner.join(texts), language, settings, source_language, restore=False
        )
    finally:
        del model
        _cleanup_cuda()


def _speech_groups(segments: list[Any], sample_rate: int) -> list[tuple[int, int]]:
    groups: list[list[int]] = []
    gap = round(_ANIME_GROUP_GAP_SECONDS * sample_rate)
    limit = round(_ANIME_GROUP_MAX_SECONDS * sample_rate)
    for segment in segments:
        start, end = int(segment.start_sample), int(segment.end_sample)
        if groups and start - groups[-1][1] <= gap and end - groups[-1][0] <= limit:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return [(start, end) for start, end in groups if end > start]


def _is_cuda_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).casefold()


def transcribe_anime_whisper(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
) -> tuple[list[Sentence], str]:
    from .asr import _cleanup_cuda, _finish_tokens, _run_transformers_asr_pipeline, _session_model

    check_cancelled(cancel_event)
    if source_language != "ja":
        raise AsmrDubberError("anime-whisper 仅支持日语。")
    _require_model(settings.asr_model, "anime-whisper")
    aligner = _aligner_id(settings)
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise AsmrDubberError(
            "anime-whisper 运行依赖尚未安装（需要 Kotoba-Whisper 与时间戳对齐组件）；"
            "请重新运行“进阶”安装。"
        ) from exc
    use_cuda = settings.asr_device.startswith("cuda")
    if use_cuda:
        require_cuda()
    sample_rate, groups, clips = _vad_clips(analysis_audio, settings, progress, cancel_event)
    if not clips:
        return [], "Japanese"

    # 1. Recognise every speech group on its own (no timestamps; not what it was trained for).
    source, revision = resolve_transformers_model_source(settings.asr_model)
    dtype = torch.float16 if use_cuda else torch.float32
    device = settings.asr_device if use_cuda else "cpu"
    texts: list[str] = []
    pipe = None
    try:
        if progress:
            progress(f"加载 anime-whisper：{settings.asr_model}", 0, len(groups))
        local_only = Path(source).is_dir()

        def _build_anime_pipe() -> Any:
            proc = AutoProcessor.from_pretrained(
                source, revision=revision, local_files_only=local_only
            )
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                source,
                revision=revision,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_only,
            ).to(device)
            return pipeline(
                "automatic-speech-recognition",
                model=mdl,
                tokenizer=proc.tokenizer,
                feature_extractor=proc.feature_extractor,
                dtype=dtype,
                device=device,
            )

        with model_heartbeat(progress, f"加载 anime-whisper：{settings.asr_model}"):
            pipe = _session_model(
                f"anime-whisper:{source}:{revision}:{device}:{dtype}",
                _build_anime_pipe,
            )
        batch = max(1, settings.asr_batch_size) * 4
        if use_cuda:
            try:
                if torch.cuda.is_available():
                    free_bytes, _total_bytes = torch.cuda.mem_get_info()
                    if free_bytes < 2.5 * 1024**3:
                        batch = 1
                    elif free_bytes < 4.5 * 1024**3:
                        batch = min(batch, 2)
            except Exception:
                pass
        for first in range(0, len(clips), batch):
            check_cancelled(cancel_event)
            with (
                model_heartbeat(progress, "anime-whisper 正在识别语音段")
                if first == 0
                else nullcontext()
            ) as hb:
                if hb:
                    hb("anime-whisper 正在识别语音段", 0, len(clips))
                elif progress:
                    progress("anime-whisper 正在识别语音段", first, len(clips))
                outputs = _run_transformers_asr_pipeline(
                    pipe,
                    [
                        {"array": clip, "sampling_rate": sample_rate}
                        for clip in clips[first : first + batch]
                    ],
                    batch_size=batch,
                    generate_kwargs={
                        "language": "ja",
                        "task": "transcribe",
                        # The model card's advice against repetition loops; never an initial prompt.
                        "no_repeat_ngram_size": 5,
                        "repetition_penalty": 1.0,
                    },
                )
            texts.extend(
                str(output.get("text", "")).strip() if isinstance(output, dict) else ""
                for output in (outputs if isinstance(outputs, list) else [outputs])
            )
    finally:
        del pipe
        _cleanup_cuda()

    # 2. Align each recognised group to get per-character times.
    tokens: list[TimedToken] = []
    pending = [i for i, text in enumerate(texts) if text]
    aligner_source, aligner_kwargs = _pretrained(torch, settings, aligner)
    aligner_model = None
    try:
        if progress:
            progress("加载 Qwen3 ForcedAligner（逐字时间戳）", 0, len(pending))
        with model_heartbeat(progress, "加载 Qwen3 ForcedAligner（逐字时间戳）") as hb:
            if hb:
                hb("加载 Qwen3 ForcedAligner（逐字时间戳）", 0, len(pending))
            aligner_model = _session_model(
                f"qwen3-aligner:{aligner_source}:{settings.asr_device}",
                lambda: Qwen3ForcedAligner.from_pretrained(aligner_source, **aligner_kwargs),
            )
        for first in range(0, len(pending), _ALIGN_BATCH):
            check_cancelled(cancel_event)
            chunk = pending[first : first + _ALIGN_BATCH]
            if progress:
                progress("对齐 anime-whisper 识别结果", first, len(pending))
            try:
                results = aligner_model.align(
                    audio=[(clips[i], sample_rate) for i in chunk],
                    text=[texts[i] for i in chunk],
                    language=["Japanese"] * len(chunk),
                )
            except (OperationCancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _cleanup_cuda(force=True)
                    raise
                logger.warning("批量对齐失败（%d 段），降级为逐段对齐：%s", len(chunk), exc)
                results = []
                for item in chunk:
                    check_cancelled(cancel_event)
                    try:
                        single = aligner_model.align(
                            audio=[(clips[item], sample_rate)],
                            text=[texts[item]],
                            language=["Japanese"],
                        )
                        results.append(single[0] if single else None)
                    except (OperationCancelledError, KeyboardInterrupt):
                        raise
                    except Exception as single_exc:
                        if _is_cuda_oom(single_exc):
                            _cleanup_cuda(force=True)
                            raise
                        logger.warning(
                            "单段对齐失败（第 %d 段），按语音区间均匀分布：%s", item, single_exc
                        )
                        results.append(None)
            for i, result in zip(chunk, results, strict=True):
                clip_tokens = group_tokens(
                    texts[i],
                    getattr(result, "items", None),
                    groups[i],
                    sample_rate,
                    clips[i],
                )
                tokens.extend(restore_punctuation(clip_tokens, texts[i]))
    finally:
        del aligner_model
        _cleanup_cuda()
    finished, detected_lang = _finish_tokens(
        tokens, "".join(texts[i] for i in pending), "Japanese", settings, "ja", restore=False
    )
    return (
        _preserve_nonverbal_vad_groups(finished, texts, groups, clips, sample_rate),
        detected_lang,
    )


def _preserve_nonverbal_vad_groups(
    sentences: list[Sentence],
    texts: list[str],
    groups: list[tuple[int, int]],
    clips: list[np.ndarray],
    sample_rate: int,
) -> list[Sentence]:
    """Emit disabled ``skipped_sound`` rows for VAD vocal groups with no lexical words.

    anime-whisper suppresses pure breaths, kisses and sighs to ``""``. Recording
    sustained vocal groups (>= 0.40 s) as ``skipped_sound`` rows preserves their
    original voice in replacement mixes and blocks neighbouring dubs from overlapping.
    """
    extra: list[Sentence] = []
    for idx, text in enumerate(texts):
        if text.strip():
            continue
        start_s = groups[idx][0] / sample_rate
        end_s = groups[idx][1] / sample_rate
        if end_s - start_s < 0.40:
            continue
        active, hop_s = active_frames(clips[idx], sample_rate)
        if active.sum() * hop_s < 0.35:
            continue
        act_idx = np.flatnonzero(active)
        v_start = round(start_s + act_idx[0] * hop_s, 3)
        v_end = round(min(end_s, start_s + (act_idx[-1] + 1) * hop_s), 3)
        if v_end - v_start < 0.35:
            continue
        # Skip if already overlapping an existing sentence.
        if any(min(v_end, s.end_seconds) - max(v_start, s.start_seconds) > 0.08 for s in sentences):
            continue
        extra.append(
            Sentence(
                id=f"s9{idx:05d}",
                start_seconds=v_start,
                end_seconds=v_end,
                source_text="（吐息）",
                enabled=False,
                chinese_audio_enabled=False,
                status="skipped_sound",
            )
        )
    if not extra:
        return sentences
    combined = sorted([*sentences, *extra], key=lambda s: (s.start_seconds, s.end_seconds, s.id))
    for i, s in enumerate(combined, start=1):
        s.id = f"s{i:06d}"
    return combined
