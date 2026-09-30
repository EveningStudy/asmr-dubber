"""Review identical audio windows; keep the baseline and propose reversible edits."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .asr_progress import model_heartbeat
from .errors import OperationCancelledError, ProjectError
from .hashing import cached_sha256_file
from .languages import SourceLanguage
from .models import ProjectSettings, Sentence
from .storage import atomic_write_text, require_disk_space
from .task_control import CancellationSignal, check_cancelled

REVIEW_VERSION = 2
Progress = Callable[[str, int, int], None]
WindowRunner = Callable[..., Iterator[tuple[str, list[Sentence] | None, str | None]]]


@dataclass(frozen=True)
class AudioWindow:
    id: str
    start: float
    end: float
    audio_start: float
    audio_end: float
    boundary: str


def normalize_text(text: str) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"(?<!\d)\.(?=\s|$)", "", value)
    # Signs, decimal points, apostrophes and lexical differences are meaningful.
    value = re.sub(r"\s+", " ", value).strip()
    value = "".join(c for c in value if c not in "。、!?！？")
    return re.sub(r"(?<![a-z0-9]) | (?![a-z0-9])", "", value)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_family(source: str) -> str:
    backend = source.partition("|")[0].casefold()
    return "whisper" if "whisper" in backend else "parakeet" if "parakeet" in backend else backend


def rows_fingerprint(rows: list[Sentence]) -> str:
    return _digest(
        [
            {
                k: row.model_dump()[k]
                for k in (
                    "id",
                    "start_seconds",
                    "end_seconds",
                    "source_text",
                    "zh_text",
                    "enabled",
                    "review_locked",
                )
            }
            for row in rows
        ]
    )


def _write(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def plan_windows(audio: Path, seconds: float = 30.0, context: float = 0.5) -> list[AudioWindow]:
    """Disjoint core intervals with acoustic overlap; quiet audio is never removed."""
    if not 10 <= seconds <= 90 or not 0 <= context <= 3:
        raise ValueError("Invalid audio review window settings")
    windows = []
    with sf.SoundFile(audio) as source:
        rate = source.samplerate
        duration = source.frames / rate
        start = 0.0
        while start < duration:
            end = min(duration, start + seconds)
            boundary = "end" if end == duration else "forced"
            if end < duration:
                lo, hi = max(start + seconds * 0.6, end - 4), min(duration, end + 4)
                source.seek(round(lo * rate))
                samples = source.read(round((hi - lo) * rate), dtype="float32", always_2d=True)
                mono = samples.mean(axis=1)
                step = max(1, round(rate * 0.1))
                levels = np.array(
                    [
                        float(np.sqrt(np.mean(mono[i : i + step] ** 2)))
                        for i in range(0, len(mono), step)
                        if len(mono[i : i + step]) == step
                    ]
                )
                quiet = [
                    i
                    for i, level in enumerate(levels)
                    if level <= min(0.003, max(1e-6, float(levels.max(initial=0)) * 0.15))
                ]
                if quiet:
                    index = min(quiet, key=lambda i: abs(lo + (i + 0.5) * 0.1 - end))
                    end = min(duration, lo + (index + 0.5) * 0.1)
                    boundary = "quiet"
            windows.append(
                AudioWindow(
                    f"w{len(windows) + 1:06d}",
                    start,
                    end,
                    max(0.0, start - context),
                    min(duration, end + context),
                    boundary,
                )
            )
            start = end
    return windows


def _window_rows(rows: list[Sentence], window: AudioWindow) -> list[Sentence]:
    return [
        row for row in rows if row.end_seconds > window.start and row.start_seconds < window.end
    ]


def merge_boundary_windows(
    windows: list[AudioWindow], baseline: list[Sentence], context: float
) -> list[AudioWindow]:
    """Merge adjacent audio units, never force a baseline sentence through a cut.

    Bound merges to 90 s. Unresolvable coarse/incorrect baseline timestamps stay
    visible as boundary proposals rather than being split by character counts.
    """
    merged = []
    index = 0
    while index < len(windows):
        first = last = windows[index]
        while index + 1 < len(windows):
            crosses = any(
                row.start_seconds < last.end + context and row.end_seconds > last.end - context
                for row in baseline
            )
            following = windows[index + 1]
            if not crosses or following.end - first.start > 90:
                break
            index += 1
            last = following
        merged.append(
            AudioWindow(
                f"w{len(merged) + 1:06d}",
                first.start,
                last.end,
                first.audio_start,
                last.audio_end,
                last.boundary,
            )
        )
        index += 1
    return merged


def _text(rows: list[Sentence], language: SourceLanguage) -> str:
    return (" " if language == "en" else "").join(row.source_text.strip() for row in rows)


def _candidate(
    rows: list[Sentence], window: AudioWindow, source: str, language: SourceLanguage
) -> dict[str, Any]:
    shifted = []
    timing_warning = False
    duration = window.audio_end - window.audio_start
    for row in rows:
        if not (
            math.isfinite(row.start_seconds)
            and math.isfinite(row.end_seconds)
            and 0 <= row.start_seconds < row.end_seconds
        ):
            raise ValueError("Reviewer returned invalid timestamps")
        if row.end_seconds > duration:
            timing_warning = True
        if row.start_seconds >= duration:
            timing_warning = True
            continue
        shifted.append(
            row.model_copy(
                update={
                    "start_seconds": row.start_seconds + window.audio_start,
                    "end_seconds": min(window.audio_end, row.end_seconds + window.audio_start),
                }
            )
        )
    shifted.sort(key=lambda row: (row.start_seconds, row.end_seconds))
    text = _text(rows, language)
    return {
        "id": "c" + _digest(normalize_text(text))[:16],
        "text": text,
        "sources": [source],
        "families": [source_family(source)],
        "sentences": [row.model_dump() for row in shifted],
        "timing_warning": timing_warning,
    }


def sensitive_change(before: str, after: str) -> bool:
    pattern = r"\d+(?:[.,]\d+)*|ない|ません|ぬ|禁止|いいえ|\b(?:not|no|never|don't|cannot|can't)\b"
    return re.findall(pattern, before.casefold()) != re.findall(pattern, after.casefold())


def automatic_edit_allowed(before: str, after: str) -> bool:
    """No learned calibration yet: only small non-deleting changes are eligible."""
    from difflib import SequenceMatcher

    original, proposed = normalize_text(before), normalize_text(after)
    if sensitive_change(before, after) or not original or not proposed:
        return False
    if len(proposed) < len(original) * 0.8 or len(proposed) > len(original) * 1.2:
        return False
    return SequenceMatcher(None, original, proposed, autojunk=False).ratio() >= 0.8


def compare_window(
    window: AudioWindow,
    baseline: list[Sentence],
    evidence: list[dict[str, Any]],
    primary_source: str,
    language: SourceLanguage,
) -> dict[str, Any]:
    owned = _window_rows(baseline, window)
    outside = [
        row
        for row in baseline
        if row not in owned
        and row.end_seconds > window.audio_start
        and row.start_seconds < window.audio_end
    ]
    safe = not outside and all(
        window.start <= row.start_seconds < row.end_seconds <= window.end for row in owned
    )
    groups: dict[str, dict[str, Any]] = {}
    for item in evidence:
        normalized = normalize_text(item["text"])
        if not normalized:
            continue
        if normalized not in groups:
            groups[normalized] = dict(item)
        else:
            group = groups[normalized]
            group["sources"] = list(dict.fromkeys([*group["sources"], *item["sources"]]))
            group["families"] = list(dict.fromkeys([*group["families"], *item["families"]]))
            if primary_source in item["sources"]:
                group["text"], group["sentences"] = item["text"], item["sentences"]
                group["timing_warning"] = item.get("timing_warning", False)
    baseline_text = _text(owned, language)
    candidates = list(groups.values())
    for candidate in candidates:
        candidate["protected_change"] = sensitive_change(baseline_text, candidate["text"])
        candidate["applicable"] = (
            safe
            and not candidate.get("timing_warning", False)
            and bool(candidate["sentences"])
            and all(
                window.start - 0.05
                <= row["start_seconds"]
                < row["end_seconds"]
                <= window.end + 0.05
                for row in candidate["sentences"]
            )
            and not any(row.review_locked for row in owned)
        )
    agreement = len(groups) == 1 and len(candidates[0]["families"]) >= 2 if candidates else False
    changed = any(normalize_text(c["text"]) != normalize_text(baseline_text) for c in candidates)
    status = (
        "agreement" if agreement and not changed else "disagreement" if changed else "single_family"
    )
    if not candidates:
        status = "unavailable"
    if not safe:
        status = "boundary_review"
    return {
        **asdict(window),
        "status": status,
        "baseline_text": baseline_text,
        "baseline_ids": [row.id for row in owned],
        "baseline_fingerprint": rows_fingerprint(owned),
        "candidates": candidates,
        "confidence": None,
        "action": "keep_baseline",
        "needs_review": status != "agreement",
        "boundary_safe": safe,
        "reason": "上下文或原稿句子跨越片段边界，请试听后在表格校对"
        if not safe
        else "主稿保留；一致性是证据，不是经过校准的正确率",
    }


def recognize_windows(
    jobs: list[tuple[str, Path]],
    settings: ProjectSettings,
    source_language: SourceLanguage,
    cancel_event: CancellationSignal | None,
    progress: Progress | None = None,
) -> Iterator[tuple[str, list[Sentence] | None, str | None]]:
    from .asr import _transcribe_parakeet, recognition_session, transcribe_source

    if source_language not in {"ja", "en", "zh"}:
        raise ProjectError("不支持的音频语言。")

    if settings.asr_backend == "parakeet_nemo" and jobs:
        results: dict[str, list[Sentence]] = {}
        errors: dict[str, str] = {}
        try:
            _transcribe_parakeet(
                jobs[0][1],
                settings,
                progress,
                cancel_event,
                window_inputs=[path for _, path in jobs],
                window_results=results,
                window_errors=errors,
            )
            for identifier, path in jobs:
                yield (
                    identifier,
                    None if path.stem in errors else results.get(path.stem, []),
                    errors.get(path.stem),
                )
        except OperationCancelledError:
            raise
        except Exception as exc:
            for identifier, path in jobs:
                yield identifier, results.get(path.stem), errors.get(path.stem, str(exc))
        return
    with recognition_session():
        consecutive_failures = 0
        for identifier, path in jobs:
            check_cancelled(cancel_event)
            if consecutive_failures >= 3:
                yield identifier, None, "该后端连续失败，已暂停；保留主稿，可重试复核"
                continue
            try:
                with model_heartbeat(
                    progress, f"{settings.asr_backend}：准备模型/解码 {identifier}"
                ) as live_progress:
                    rows, _ = transcribe_source(
                        path,
                        settings,
                        source_language=source_language,
                        progress=live_progress,
                        cancel_event=cancel_event,
                    )
                consecutive_failures = 0
                yield identifier, rows, None
            except OperationCancelledError:
                raise
            except Exception as exc:
                consecutive_failures += 1
                yield identifier, None, str(exc)


def run_audio_review(
    audio: Path,
    baseline: list[Sentence],
    settings: ProjectSettings,
    report_path: Path,
    *,
    source_language: SourceLanguage = "ja",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    runner: WindowRunner | None = None,
) -> list[Sentence]:
    check_cancelled(cancel_event)
    runner = runner or recognize_windows
    primary = f"{settings.asr_backend}|{settings.asr_model}"
    sources = list(dict.fromkeys([primary, *settings.asr_review_models]))
    windows = plan_windows(
        audio, settings.asr_review_window_seconds, settings.asr_review_context_seconds
    )
    windows = merge_boundary_windows(windows, baseline, settings.asr_review_context_seconds)
    source_hash = cached_sha256_file(audio)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    cache_dir = report_path.parent / "review-v2-cache"
    cache_dir.mkdir(exist_ok=True)
    clips_dir = (
        report_path.parent
        / "review-audio"
        / _digest([source_hash, [asdict(w) for w in windows]])[:24]
    )
    clips_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, list[dict[str, Any]]] = {window.id: [] for window in windows}
    warnings: list[dict[str, str]] = []
    by_id = {window.id: window for window in windows}
    report: dict[str, Any] = {
        "schema": REVIEW_VERSION,
        "run_id": run_id,
        "state": "running",
        "audio_sha256": source_hash,
        "baseline_fingerprint": rows_fingerprint(baseline),
        "baseline": [row.model_dump() for row in baseline],
        "primary_source": primary,
        "mode": settings.asr_review_mode,
        "results": [],
        "warnings": warnings,
        "undo": [],
    }
    if report_path.is_file():
        import shutil

        history = report_path.parent / "review-history"
        history.mkdir(exist_ok=True)
        shutil.copyfile(report_path, history / f"{uuid.uuid4().hex}.json")

    def checkpoint(state: str = "running") -> None:
        report["state"] = state
        report["results"] = [
            compare_window(w, baseline, evidence[w.id], primary, source_language) for w in windows
        ]
        _write(report_path, report)

    checkpoint()
    try:
        with sf.SoundFile(audio) as source_audio:
            required_bytes = sum(
                round((w.audio_end - w.audio_start) * source_audio.samplerate)
                * source_audio.channels
                * 2
                for w in windows
                if not (clips_dir / f"{w.id}.wav").is_file()
            )
            require_disk_space(clips_dir, required_bytes)
            for window in windows:
                check_cancelled(cancel_event)
                clip = clips_dir / f"{window.id}.wav"
                expected_frames = round(
                    (window.audio_end - window.audio_start) * source_audio.samplerate
                )
                if clip.is_file():
                    try:
                        if sf.info(clip).frames == expected_frames:
                            continue
                    except (OSError, RuntimeError):
                        pass
                source_audio.seek(round(window.audio_start * source_audio.samplerate))
                samples = source_audio.read(
                    round((window.audio_end - window.audio_start) * source_audio.samplerate),
                    dtype="float32",
                )
                sf.write(
                    clip,
                    samples,
                    source_audio.samplerate,
                    subtype="PCM_16",
                )
        for source_index, label in enumerate(sources):
            check_cancelled(cancel_event)
            backend, separator, model = label.partition("|")
            if not separator:
                warnings.append({"source": label, "error": "复核模型配置无效"})
                continue
            payload = settings.model_dump()
            payload.update(
                asr_backend=backend,
                asr_model=model,
                asr_review_enabled=False,
                asr_vad_mode="off",
                asr_vad_filter=False,
                skip_japanese_fillers=False,
            )
            try:
                configured = ProjectSettings.model_validate(payload)
            except ValueError as exc:
                warnings.append({"source": label, "error": str(exc)})
                continue
            acoustic_settings = {
                k: v
                for k, v in configured.model_dump().items()
                if (k.startswith("asr_") and not k.startswith("asr_review"))
                or k in {"pause_split_seconds", "max_sentence_seconds"}
            }
            pending = []
            paths = {}
            for window in windows:
                key = _digest(
                    [
                        REVIEW_VERSION,
                        source_hash,
                        asdict(window),
                        label,
                        source_language,
                        acoustic_settings,
                    ]
                )
                path = cache_dir / f"{key}.json"
                paths[window.id] = path
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    rows = [Sentence.model_validate(row) for row in cached["sentences"]]
                    evidence[window.id].append(_candidate(rows, window, label, source_language))
                except (OSError, ValueError, KeyError, TypeError):
                    pending.append((window.id, clips_dir / f"{window.id}.wav"))
            if progress:
                progress(
                    f"统一音频复核 {source_index + 1}/{len(sources)}：{label}"
                    f"，待处理 {len(pending)} 段",
                    source_index,
                    len(sources),
                )
            processed = [len(windows) - len(pending), 0]
            pending_count = len(pending)

            def window_progress(
                message: str,
                current: int,
                total: int,
                _label=label,
                _backend=backend,
                _source_index=source_index,
                _processed=processed,
                _pending_count=pending_count,
            ) -> None:
                if progress:
                    partial = max(0.0, min(1.0, current / max(1, total)))
                    if _backend == "parakeet_nemo":
                        partial *= _pending_count
                    completed = min(len(windows), _processed[0] + partial)
                    _processed[1] = max(
                        _processed[1],
                        round((_source_index + completed / max(1, len(windows))) * 10000),
                    )
                    progress(
                        f"{_label} · 已完成 {_processed[0]}/{len(windows)} 段 · {message}",
                        _processed[1],
                        len(sources) * 10000,
                    )

            try:
                for identifier, rows, error in runner(
                    pending, configured, source_language, cancel_event, window_progress
                ):
                    check_cancelled(cancel_event)
                    if rows is None:
                        warnings.append(
                            {"source": label, "window": identifier, "error": error or "复核失败"}
                        )
                    else:
                        try:
                            item = _candidate(rows, by_id[identifier], label, source_language)
                            evidence[identifier].append(item)
                            _write(
                                paths[identifier], {"sentences": [row.model_dump() for row in rows]}
                            )
                        except (ValueError, KeyError) as exc:
                            warnings.append(
                                {"source": label, "window": identifier, "error": str(exc)}
                            )
                    processed[0] += 1
                    window_progress("片段结果已保存", 0, 1)
                checkpoint()
            except OperationCancelledError:
                raise
            except Exception as exc:
                warnings.append({"source": label, "error": str(exc)})
                checkpoint()
        checkpoint("completed_with_warnings" if warnings else "completed")
    except OperationCancelledError:
        checkpoint("cancelled")
        raise
    except Exception as exc:
        warnings.append({"source": "review", "error": str(exc)})
        checkpoint("failed")
    output = [row.model_copy(deep=True) for row in baseline]
    if settings.asr_review_mode == "conservative" and report["state"].startswith("completed"):
        for result in report["results"]:
            viable = [
                candidate
                for candidate in result["candidates"]
                if candidate["applicable"]
                and not candidate["protected_change"]
                and automatic_edit_allowed(result["baseline_text"], candidate["text"])
                and all(row.enabled for row in output if row.id in result["baseline_ids"])
                and len(candidate["families"]) >= 2
                and primary in candidate["sources"]
                and normalize_text(candidate["text"]) != normalize_text(result["baseline_text"])
            ]
            if len(viable) == 1 and result["baseline_ids"]:
                before = [row.model_dump() for row in output if row.id in result["baseline_ids"]]
                output = replace_window(output, result, viable[0], settings, source_language)
                after = [
                    row for row in output if row.id.startswith(f"{result['id']}-{viable[0]['id']}-")
                ]
                report["undo"].append(
                    {
                        "window": result["id"],
                        "before": before,
                        "after_ids": [row.id for row in after],
                        "after_fingerprint": rows_fingerprint(after),
                    }
                )
                result["action"] = "auto_applied"
                result["selected_candidate"] = viable[0]["id"]
    _write(report_path, report)
    if progress:
        count = sum(item["needs_review"] for item in report["results"])
        progress(
            f"音频复核完成：{len(windows)} 段，{count} 段待查看；主稿已保留",
            len(sources),
            len(sources),
        )
    return output


def replace_window(
    current: list[Sentence],
    result: dict[str, Any],
    candidate: dict[str, Any],
    settings: ProjectSettings,
    source_language: SourceLanguage = "ja",
) -> list[Sentence]:
    from .segmentation import TimedToken, split_timed_tokens

    owned = [row for row in current if row.id in result["baseline_ids"]]
    intersecting = [
        row
        for row in current
        if row.end_seconds > result["start"] and row.start_seconds < result["end"]
    ]
    if {row.id for row in intersecting} != set(result["baseline_ids"]):
        raise ProjectError("片段内句子已新增、删除或移动，旧提案不能覆盖当前内容。")
    if rows_fingerprint(owned) != result["baseline_fingerprint"] or any(
        row.review_locked for row in owned
    ):
        raise ProjectError("该片段的主稿已修改或已人工锁定，请重新复核；未覆盖校对内容。")
    if not candidate["applicable"]:
        raise ProjectError("该候选跨越上下文边界，不能安全一键应用；请试听后在表格校对。")
    rows = [Sentence.model_validate(row) for row in candidate["sentences"]]
    if not rows or any(
        not result["start"] - 0.05 <= row.start_seconds < row.end_seconds <= result["end"] + 0.05
        for row in rows
    ):
        raise ProjectError("候选时间范围无效。")
    normalized = split_timed_tokens(
        [
            TimedToken(
                (" " if source_language == "en" else "") + row.source_text,
                row.start_seconds,
                row.end_seconds,
            )
            for row in rows
        ],
        pause_seconds=settings.pause_split_seconds,
        max_sentence_seconds=settings.max_sentence_seconds,
    )
    for index, row in enumerate(normalized):
        row.id = f"{result['id']}-{candidate['id']}-{index + 1}"
        row.review_locked = True
        row.status = "review_accepted"
    kept = [row for row in current if row.id not in result["baseline_ids"]]
    return sorted(
        [*kept, *normalized], key=lambda row: (row.start_seconds, row.end_seconds, row.id)
    )
