from __future__ import annotations

import gc
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .constants import DEFAULT_ALIGNER_MODEL
from .environment import cached_model_path, require_cuda, resolve_transformers_model_source
from .errors import AsmrDubberError
from .languages import SourceLanguage, qwen_language_name
from .models import ProjectSettings, Sentence
from .task_control import CancellationSignal, check_cancelled

Progress = Callable[[str, int, int], None]


def _alignment_device_map(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


def _alignment_dtype(torch: Any, device: str) -> Any:
    if not device.startswith("cuda"):
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except (AttributeError, RuntimeError):
        pass
    # Turing/Pascal cards can have enough VRAM for the aligner but do not
    # support BF16 kernels. FP16 keeps those GPUs usable instead of binding the
    # runtime to the architecture of the machine that built the portable pack.
    return torch.float16


def _map_group_items_to_sentences(
    processor: Any,
    sentences: list[Sentence],
    items: list[Any],
) -> list[list[Any]]:
    """Map whole-group tokens back to lines without retokenizing each line."""

    # Nagisa can split the same Japanese text differently when neighbouring
    # lines are present. The aligner returns the cleaned token text it actually
    # used, so character spans remain exact even when token counts differ.
    clean_token = processor.clean_token
    sentence_texts = [clean_token(sentence.source_text) for sentence in sentences]
    item_texts = [clean_token(str(item.text)) for item in items]
    expected = "".join(sentence_texts)
    actual = "".join(item_texts)
    if expected != actual:
        raise ValueError("Qwen3 ForcedAligner 返回的文字单元与台本不一致，已保留本组估算时间轴。")

    item_starts: list[int] = []
    item_ends: list[int] = []
    cursor = 0
    for text in item_texts:
        item_starts.append(cursor)
        cursor += len(text)
        item_ends.append(cursor)

    mapped: list[list[Any]] = []
    cursor = 0
    for text in sentence_texts:
        start = cursor
        cursor += len(text)
        if cursor <= start:
            mapped.append([])
            continue
        first = bisect_right(item_ends, start)
        stop = bisect_left(item_starts, cursor)
        mapped.append(items[first:stop])
    return mapped


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def align_sentences_with_qwen(
    audio_path: Path,
    sentences: list[Sentence],
    settings: ProjectSettings,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SourceLanguage = "ja",
) -> list[dict[str, Any]]:
    """Refine sentence boundaries with the pinned standalone Qwen aligner."""

    if not sentences:
        return []
    check_cancelled(cancel_event)
    model_id = settings.aligner_model or DEFAULT_ALIGNER_MODEL
    if cached_model_path(model_id) is None:
        raise AsmrDubberError(
            "Qwen3 ForcedAligner 尚未完整安装；请重新运行“进阶”安装，或导入对应离线模型包。"
        )
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise AsmrDubberError(
            "Qwen3 ForcedAligner 运行依赖尚未安装；请重新运行“进阶”安装。"
        ) from exc

    use_cuda = settings.asr_device.startswith("cuda")
    if use_cuda:
        require_cuda()
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1 or sample_rate <= 0:
        raise AsmrDubberError("Qwen3 ForcedAligner 无法读取 ASR（语音识别）分析音频。")

    source, revision = resolve_transformers_model_source(model_id)
    kwargs: dict[str, Any] = {
        "dtype": _alignment_dtype(torch, settings.asr_device),
        "device_map": _alignment_device_map(settings.asr_device),
    }
    if revision is not None:
        kwargs["revision"] = revision

    model = None
    results: list[dict[str, Any]] = []
    result_sentences: list[Sentence] = []
    try:
        if progress:
            progress("加载 Qwen3 ForcedAligner（时间戳对齐）", 0, len(sentences))
        model = Qwen3ForcedAligner.from_pretrained(source, **kwargs)
        check_cancelled(cancel_event)
        duration = len(waveform) / sample_rate
        padding = max(0.75, min(2.0, settings.asr_review_max_drift_seconds))
        for index, sentence in enumerate(sentences, start=1):
            check_cancelled(cancel_event)
            crop_start = max(0.0, sentence.start_seconds - padding)
            crop_end = min(duration, sentence.end_seconds + padding)
            start_sample = max(0, int(crop_start * sample_rate))
            end_sample = min(len(waveform), int(np.ceil(crop_end * sample_rate)))
            record: dict[str, Any] = {
                "sentence_id": sentence.id,
                "fallback": True,
                "before": [sentence.start_seconds, sentence.end_seconds],
            }
            if progress:
                progress(
                    f"Qwen3 ForcedAligner：{index}/{len(sentences)}",
                    index - 1,
                    len(sentences),
                )
            try:
                aligned = model.align(
                    audio=(waveform[start_sample:end_sample], sample_rate),
                    text=sentence.source_text,
                    language=qwen_language_name(source_language),
                )
                items = list(aligned[0].items) if aligned else []
                starts = [float(item.start_time) for item in items]
                ends = [float(item.end_time) for item in items]
                if starts and ends:
                    aligned_start = crop_start + min(starts)
                    aligned_end = crop_start + max(ends)
                    if 0 <= aligned_start < aligned_end <= duration + 0.01:
                        sentence.start_seconds = aligned_start
                        sentence.end_seconds = min(duration, aligned_end)
                        record["fallback"] = False
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            check_cancelled(cancel_event)
            record["after"] = [sentence.start_seconds, sentence.end_seconds]
            results.append(record)
            result_sentences.append(sentence)

        sentences.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.id))
        for index, sentence in enumerate(sentences, start=1):
            sentence.id = f"s{index:06d}"
        for sentence, record in zip(result_sentences, results, strict=True):
            record["sentence_id"] = sentence.id
        if progress:
            aligned_count = sum(not item["fallback"] for item in results)
            progress(
                f"Qwen3 ForcedAligner 完成：{aligned_count}/{len(sentences)} 句",
                len(sentences),
                len(sentences),
            )
        return results
    finally:
        del model
        _cleanup_cuda()


def align_script_sentences_with_qwen(
    audio_path: Path,
    sentences: list[Sentence],
    settings: ProjectSettings,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SourceLanguage = "ja",
) -> list[dict[str, Any]]:
    """Align an ordered, untimed script in bounded five-minute groups.

    Qwen3 ForcedAligner documents a five-minute maximum input. Initial ranges
    are therefore used only to choose groups and crops; successful model spans
    replace those estimates, while failed groups remain editable fallbacks.
    """

    if not sentences:
        return []
    check_cancelled(cancel_event)
    model_id = settings.aligner_model or DEFAULT_ALIGNER_MODEL
    if cached_model_path(model_id) is None:
        raise AsmrDubberError(
            "Qwen3 ForcedAligner 尚未完整安装。可改用“按台词长度估算”，"
            "或在“设备与模型”中安装进阶组件。"
        )
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise AsmrDubberError(
            "Qwen3 ForcedAligner 运行依赖尚未安装。可改用“按台词长度估算”，"
            "或在“设备与模型”中安装进阶组件。"
        ) from exc

    use_cuda = settings.asr_device.startswith("cuda")
    if use_cuda:
        require_cuda()
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1 or sample_rate <= 0:
        raise AsmrDubberError("Qwen3 ForcedAligner 无法读取台本对齐音频。")
    duration = len(waveform) / sample_rate

    # Leave 30 seconds on each side for dialogue-density differences while
    # keeping every crop within the model's documented 300-second limit.
    groups: list[list[Sentence]] = []
    current: list[Sentence] = []
    for sentence in sentences:
        if current and sentence.end_seconds - current[0].start_seconds > 240.0:
            groups.append(current)
            current = []
        current.append(sentence)
    if current:
        groups.append(current)

    source, revision = resolve_transformers_model_source(model_id)
    kwargs: dict[str, Any] = {
        "dtype": _alignment_dtype(torch, settings.asr_device),
        "device_map": _alignment_device_map(settings.asr_device),
    }
    if revision is not None:
        kwargs["revision"] = revision

    model = None
    report: list[dict[str, Any]] = []
    try:
        if progress:
            progress("加载 Qwen3 ForcedAligner（纯台本时间轴）", 0, len(groups))
        model = Qwen3ForcedAligner.from_pretrained(source, **kwargs)
        check_cancelled(cancel_event)
        for group_index, group in enumerate(groups, start=1):
            check_cancelled(cancel_event)
            crop_start = max(0.0, group[0].start_seconds - 30.0)
            crop_end = min(duration, group[-1].end_seconds + 30.0)
            if crop_end - crop_start > 300.0:
                excess = crop_end - crop_start - 300.0
                crop_start += excess / 2
                crop_end -= excess - (excess / 2)
            start_sample = max(0, int(crop_start * sample_rate))
            end_sample = min(len(waveform), int(np.ceil(crop_end * sample_rate)))
            if progress:
                progress(
                    f"Qwen3 ForcedAligner 对齐台本 {group_index}/{len(groups)} 组",
                    group_index - 1,
                    len(groups),
                )
            group_text = "\n".join(sentence.source_text for sentence in group)
            group_records = [
                {
                    "sentence_id": sentence.id,
                    "before": [sentence.start_seconds, sentence.end_seconds],
                    "fallback": True,
                }
                for sentence in group
            ]
            try:
                aligned = model.align(
                    audio=(waveform[start_sample:end_sample], sample_rate),
                    text=group_text,
                    language=qwen_language_name(source_language),
                )
                items = list(aligned[0].items) if aligned else []
                sentence_items = _map_group_items_to_sentences(
                    model.aligner_processor,
                    group,
                    items,
                )
                for sentence, record, aligned_items in zip(
                    group,
                    group_records,
                    sentence_items,
                    strict=True,
                ):
                    if not aligned_items:
                        continue
                    start = crop_start + float(aligned_items[0].start_time)
                    end = crop_start + float(aligned_items[-1].end_time)
                    if 0 <= start < end <= duration + 0.01:
                        sentence.start_seconds = start
                        sentence.end_seconds = min(duration, end)
                        record["fallback"] = False
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
                for record in group_records:
                    record["error"] = error
            check_cancelled(cancel_event)
            for sentence, record in zip(group, group_records, strict=True):
                record["after"] = [sentence.start_seconds, sentence.end_seconds]
            report.extend(group_records)

        sentences.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.id))
        renamed: dict[str, str] = {}
        for index, sentence in enumerate(sentences, start=1):
            new_id = f"s{index:06d}"
            renamed[sentence.id] = new_id
            sentence.id = new_id
        for record in report:
            old_id = str(record["sentence_id"])
            record["sentence_id"] = renamed.get(old_id, old_id)
        if progress:
            aligned_count = sum(not item["fallback"] for item in report)
            progress(
                f"纯台本时间轴完成：模型对齐 {aligned_count}/{len(sentences)} 句",
                len(groups),
                len(groups),
            )
        return report
    finally:
        del model
        _cleanup_cuda()
