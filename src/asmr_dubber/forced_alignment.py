from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .constants import DEFAULT_ALIGNER_MODEL
from .environment import cached_model_path, require_cuda, resolve_transformers_model_source
from .errors import AsmrDubberError
from .models import ProjectSettings, Sentence

Progress = Callable[[str, int, int], None]


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
) -> list[dict[str, Any]]:
    """Refine sentence boundaries with the pinned standalone Qwen aligner."""

    if not sentences:
        return []
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
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
        "device_map": "cuda:0" if use_cuda else "cpu",
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
        duration = len(waveform) / sample_rate
        padding = max(0.75, min(2.0, settings.asr_review_max_drift_seconds))
        for index, sentence in enumerate(sentences, start=1):
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
                    text=sentence.ja_text,
                    language="Japanese",
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
