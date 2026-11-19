"""Transcript import and LLM script reconciliation pipeline operations."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from .errors import ProjectError
from .languages import SourceLanguage
from .lifecycle import project_operation
from .models import DubProject, Sentence
from .task_control import CancellationSignal
from .transcript_import import TranscriptLanguage

Progress = Callable[[str, int, int], None]


def _chinese_script_timing_warnings(
    sentences: list[Sentence],
    *,
    maximum_speed: float,
) -> list[dict[str, object]]:
    """Flag obvious text/window mismatches without changing ASR time ranges."""
    from . import pipeline

    warnings: list[dict[str, object]] = []
    for sentence in sentences:
        text = sentence.zh_text.strip()
        units = len(pipeline._SCRIPT_VISIBLE_CHARACTER.findall(text))
        window = max(0.001, sentence.end_seconds - sentence.start_seconds)
        # Five visible Chinese characters per second is deliberately
        # conservative. Only warn when the estimate still exceeds the user's
        # configured maximum mix-time acceleration.
        estimated = units / 5.0
        required_speed = estimated / window
        if units >= 8 and required_speed > maximum_speed:
            warnings.append(
                {
                    "id": sentence.id,
                    "text_characters": units,
                    "window_seconds": round(window, 3),
                    "estimated_required_speed": round(required_speed, 2),
                    "maximum_auto_speed": maximum_speed,
                }
            )
    return warnings


def _reconcile_untimed_script(
    project: DubProject,
    project_dir: Path,
    *,
    script_lines: list[str],
    script_language: TranscriptLanguage,
    role: Literal["source", "target"] | None = None,
    script_source: str = "",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[list[Sentence], list[dict[str, object]]]:
    """Run the normal ASR/translation flow in a temporary project, then apply the script text."""
    from . import pipeline

    provider = project.settings.translation_provider
    if provider not in pipeline.LLM_RECONCILIATION_PROVIDERS:
        raise ProjectError(
            "无时间轴台本的智能校对需要大模型翻译服务；请先在设置中选择 DeepSeek、百炼、"
            "豆包、商汤、OpenAI、Claude、Gemini 或本地/自定义 OpenAI-compatible。"
        )
    preset = pipeline.PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    key = pipeline.resolve_api_key(provider)
    base_url = project.settings.translation_base_url.strip() or str(preset["base_url"])
    source = pipeline.verify_source(project_dir, project.source)
    temporary_root = project_dir / "temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="script-reconcile-", dir=temporary_root) as name:
        working_dir = Path(name)
        source_name = Path(project.source.path).name or "source.media"
        working_source = working_dir / source_name
        try:
            os.link(source, working_source)
        except OSError:
            # Hard links avoid copying multi-hour DLsite files on the same volume;
            # a normal copy keeps the feature usable across filesystems.
            shutil.copy2(source, working_source)
        working = project.model_copy(deep=True)
        working.source.path = source_name
        working.sentences = []
        working.revision = 0
        working.asr_language = None
        working.chinese_stem_file = None
        working.output_file = None
        working.output_video_file = None
        working.subtitle_srt_file = None
        working.subtitle_lrc_file = None
        working.subtitle_video_file = None
        working.source_language = project.source_language
        pipeline._analyze_project_impl(
            working,
            working_dir,
            force=True,
            progress=progress,
            cancel_event=cancel_event,
        )
        target: Literal["source", "zh"] = (
            "source" if role == "source" else ("zh" if script_language == "zh" else "source")
        )
        corrections, report = pipeline.reconcile_script_sentences(
            working.sentences,
            script_lines,
            source_language=project.source_language,
            target=target,
            provider=provider,
            api_key=key,
            model=project.settings.translation_model,
            base_url=base_url,
            temperature=project.settings.translation_temperature,
            top_p=project.settings.translation_top_p,
            max_output_tokens=project.settings.translation_max_output_tokens,
            extra_body=project.settings.translation_extra_body,
            job_id=f"asmr_{project.source.sha256[:24]}_script",
            diagnostic_path=project_dir / "imports" / "script-reconciliation-error.json",
            script_source=script_source,
            progress=progress,
            cancel_event=cancel_event,
        )
        unmatched_ids: list[str] = []
        review_ids = {
            str(sid)
            for entry in report
            if isinstance(ids := entry.get("needs_review_ids"), list)
            for sid in ids
        }
        for sentence in working.sentences:
            sentence.script_review_note = (
                "台本校对未确认，保留原识别/译文，请人工校对。"
                if sentence.id in review_ids
                else None
            )
            if sentence.id in review_ids:
                unmatched_ids.append(sentence.id)
                continue
            corrected = str(corrections.get(sentence.id, "")).strip()
            if target == "source":
                if corrected:
                    sentence.source_text = corrected
                    sentence.zh_text = ""
                    sentence.status = "pending"
                elif sentence.source_text.strip():
                    # Keep existing ASR text and flag for review rather than clearing to "".
                    sentence.zh_text = ""
                    sentence.status = "pending"
                    sentence.script_review_note = "台本未匹配该句，保留原识别文本，请人工核对。"
                    unmatched_ids.append(sentence.id)
                else:
                    sentence.status = "skipped_filler"
            else:
                if corrected:
                    sentence.zh_text = corrected
                    sentence.status = "translated"
                else:
                    sentence.zh_text = ""
                    sentence.status = "pending"
                    if sentence.source_text.strip():
                        sentence.script_review_note = "中文台本未匹配该句，需人工核对或后续翻译。"
                        unmatched_ids.append(sentence.id)
            sentence.enabled = bool(sentence.source_text.strip() or sentence.zh_text.strip())
            sentence.tts_file = None
            sentence.tts_cache_key = None
            sentence.tts_duration_seconds = None
            sentence.error = None
        if unmatched_ids:
            report.append(
                {
                    "unmatched_ids": unmatched_ids,
                    "reason": "中文台本没有可靠匹配；后续只翻译这些缺失句子"
                    if target == "zh"
                    else "台本没有可靠匹配；只采用台本文字，不自动混入 ASR 原文，需人工核对",
                }
            )
        timing_warnings = pipeline._chinese_script_timing_warnings(
            working.sentences,
            maximum_speed=project.settings.chinese_max_auto_speed,
        )
        if target == "zh" and timing_warnings:
            report.append(
                {
                    "timing_warnings": timing_warnings,
                    "reason": "部分中文台本相对 ASR 时间窗过长，配音可能达到最大加速后仍有重叠",
                }
            )
        return [item.model_copy(deep=True) for item in working.sentences], report


@project_operation
def reconcile_analyzed_project_script(
    project: DubProject,
    project_dir: Path,
    *,
    transcript_path: str | Path,
    script_language: TranscriptLanguage,
    start_seconds: float,
    end_seconds: float,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> dict[str, object]:
    """Map one track's script onto existing ASR ranges without trusting file timing."""
    from . import pipeline

    if script_language not in {"ja", "en", "zh"}:
        raise ProjectError("台本语言必须是日语、英语或中文。")
    selected = [
        sentence
        for sentence in project.sentences
        if start_seconds <= (sentence.start_seconds + sentence.end_seconds) / 2 < end_seconds
    ]
    if not selected:
        raise ProjectError("这条音轨的时间范围内没有 ASR 句子，无法校对台本。")
    parsed = pipeline.parse_transcript(
        duration_seconds=max(0.001, end_seconds - start_seconds),
        path=transcript_path,
        language=script_language,
        use_embedded_timing=False,
    )
    script_lines = [
        sentence.zh_text if script_language == "zh" else sentence.source_text
        for sentence in parsed.sentences
    ]
    provider = project.settings.translation_provider
    if provider not in pipeline.LLM_RECONCILIATION_PROVIDERS:
        raise ProjectError(
            "ASR 定时的台本校对需要大模型翻译服务；请先在设置中选择 DeepSeek、百炼、"
            "豆包、商汤、OpenAI、Claude、Gemini 或本地/自定义 OpenAI-compatible。"
        )
    preset = pipeline.PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    key = pipeline.resolve_api_key(provider)
    base_url = project.settings.translation_base_url.strip() or str(preset["base_url"])
    target: Literal["source", "zh"] = "zh" if script_language == "zh" else "source"
    corrections, report = pipeline.reconcile_script_sentences(
        selected,
        script_lines,
        source_language=project.source_language,
        target=target,
        provider=provider,
        api_key=key,
        model=project.settings.translation_model,
        base_url=base_url,
        temperature=project.settings.translation_temperature,
        top_p=project.settings.translation_top_p,
        max_output_tokens=project.settings.translation_max_output_tokens,
        extra_body=project.settings.translation_extra_body,
        job_id=(
            f"asmr_{project.source.sha256[:18]}_script_"
            f"{round(start_seconds * 1000)}_{round(end_seconds * 1000)}"
        ),
        diagnostic_path=project_dir
        / "imports"
        / f"script-reconciliation-{round(start_seconds * 1000)}-error.json",
        script_source=str(transcript_path),
        progress=progress,
        cancel_event=cancel_event,
    )
    matched = 0
    unmatched: list[str] = []
    review_ids = {
        str(sid)
        for entry in report
        if isinstance(ids := entry.get("needs_review_ids"), list)
        for sid in ids
    }
    for sentence in selected:
        sentence.script_review_note = (
            "台本校对未确认，保留原识别/译文，请人工校对。" if sentence.id in review_ids else None
        )
        if sentence.id in review_ids:
            unmatched.append(sentence.id)
            continue
        corrected = str(corrections.get(sentence.id, "")).strip()
        if target == "source":
            if corrected:
                matched += 1
                sentence.source_text = corrected
                sentence.zh_text = ""
                sentence.status = "pending"
            else:
                unmatched.append(sentence.id)
                if sentence.source_text.strip():
                    sentence.script_review_note = "台本未匹配该句，保留原识别文本，请人工核对。"
                else:
                    sentence.status = "skipped_filler"
        elif corrected:
            sentence.zh_text = corrected
            sentence.status = "translated"
            matched += 1
        else:
            unmatched.append(sentence.id)
            if not sentence.zh_text:
                sentence.status = "pending"
        sentence.enabled = bool(sentence.source_text.strip() or sentence.zh_text.strip())
        sentence.tts_file = None
        sentence.tts_cache_key = None
        sentence.tts_duration_seconds = None
        sentence.error = None

    project.chinese_stem_file = None
    project.output_file = None
    project.output_video_file = None
    project.subtitle_srt_file = None
    project.subtitle_lrc_file = None
    project.subtitle_video_file = None
    timing_warnings = (
        pipeline._chinese_script_timing_warnings(
            selected,
            maximum_speed=project.settings.chinese_max_auto_speed,
        )
        if target == "zh"
        else []
    )
    if timing_warnings:
        report.append(
            {
                "timing_warnings": timing_warnings,
                "reason": "部分中文台本相对 ASR 时间窗过长，配音可能达到最大加速后仍有重叠",
            }
        )
    pipeline.save_project(project, project_dir)
    pipeline.export_transcript(project, project_dir)
    return {
        "path": str(Path(transcript_path).expanduser().resolve()),
        "language": script_language,
        "asr_sentences": len(selected),
        "matched_sentences": matched,
        "unmatched_ids": unmatched,
        "timing_warning_ids": [str(item["id"]) for item in timing_warnings],
        "report": report,
    }


@project_operation
def import_project_transcript(
    project: DubProject,
    project_dir: Path,
    *,
    transcript_path: str | Path | None = None,
    pasted_text: str = "",
    plain_timing: str = "estimate",
    script_language: str = "ja",
    script_kind: Literal["source", "zh"] | None = None,
    use_embedded_timing: bool = True,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> dict[str, object]:
    """Replace ASR output with a user-provided script or timed subtitle file."""
    from . import pipeline

    pipeline.require_supported_platform()
    pipeline.check_cancelled(cancel_event)
    if plain_timing not in {"estimate", "qwen", "script_review"}:
        raise ProjectError("纯台本时间轴方式无效。")
    if script_language not in {"ja", "en", "zh"}:
        raise ProjectError("台本语言必须是日语、英语或中文。")
    role: Literal["source", "target"] | None = (
        "source" if script_kind == "source" else ("target" if script_kind == "zh" else None)
    )
    is_target_zh = script_language == "zh" and script_kind != "source"
    if is_target_zh and plain_timing == "qwen":
        raise ProjectError("中文纯台本不能使用 Qwen3 自动对齐，请选择按台词长度估算。")
    if plain_timing == "script_review" and project.source_language == "zh" and is_target_zh:
        raise ProjectError(
            "当前项目已经是中文台本项目，不能再次运行源音频识别；请新建原始音频项目。"
        )
    source = pipeline.verify_source(project_dir, project.source)
    parsed = pipeline.parse_transcript(
        duration_seconds=project.source.duration_seconds,
        path=transcript_path,
        pasted_text=pasted_text,
        language=cast(TranscriptLanguage, script_language),
        role=role,
        use_embedded_timing=use_embedded_timing,
    )
    sentences = [sentence.model_copy(deep=True) for sentence in parsed.sentences]
    alignment_report: list[dict[str, object]] = []
    reconciliation_report: list[dict[str, object]] = []
    reconciled = False
    if not parsed.timed and plain_timing == "script_review":
        script_lines = [
            sentence.zh_text if is_target_zh else sentence.source_text
            for sentence in parsed.sentences
        ]
        sentences, reconciliation_report = pipeline._reconcile_untimed_script(
            project,
            project_dir,
            script_lines=script_lines,
            script_language=cast(TranscriptLanguage, script_language),
            role=role,
            script_source=str(transcript_path) if transcript_path else "粘贴台本",
            progress=progress,
            cancel_event=cancel_event,
        )
        reconciled = True
    elif not parsed.timed and plain_timing == "qwen":
        analysis = pipeline.make_analysis_copy(
            source, project_dir / "analysis" / "asr_16k_mono.wav"
        )
        cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        alignment_report = pipeline.align_script_sentences_with_qwen(
            analysis,
            sentences,
            project.settings,
            source_language=cast(SourceLanguage, script_language),
            progress=progress,
            **cancel_kwargs,
        )
    pipeline.check_cancelled(cancel_event)

    import_dir = project_dir / "imports"
    import_dir.mkdir(parents=True, exist_ok=True)
    pipeline.atomic_write_text(import_dir / "latest-transcript.txt", parsed.source_text)
    aligned_count = sum(not bool(item.get("fallback", True)) for item in alignment_report)
    pipeline.atomic_write_text(
        import_dir / "latest-transcript.json",
        json.dumps(
            {
                "format": parsed.source_format,
                "language": parsed.language,
                "timed": parsed.timed,
                "plain_timing": None if parsed.timed else plain_timing,
                "sentences": len(sentences),
                "qwen_aligned_sentences": aligned_count,
                "script_reconciled": reconciled,
                "reconciliation": reconciliation_report,
                "alignment": alignment_report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    project.sentences = sentences
    project.translation_language = project.settings.tts_target_language
    if project.settings.tts_target_language == "en" and is_target_zh:
        for sentence in project.sentences:
            sentence.zh_text = ""
    if not reconciled:
        project.source_language = cast(SourceLanguage, parsed.language)
        project.settings = pipeline.settings_for_source_language(
            project.settings, project.source_language
        )
        language_name = pipeline.source_language_label(parsed.language)
        project.asr_language = f"{language_name} (imported {parsed.source_format})"
    else:
        language_name = pipeline.source_language_label(project.source_language)
        suffix = "中文台本校对" if is_target_zh else "台本校对"
        project.asr_language = f"{language_name}（ASR + {suffix}）"
    project.asr_settings_dirty = False
    project.settings.tts_reference_sentence_id = None
    project.chinese_stem_file = None
    project.output_file = None
    project.output_video_file = None
    project.subtitle_srt_file = None
    project.subtitle_lrc_file = None
    project.subtitle_video_file = None
    pipeline.save_project(project, project_dir)
    pipeline.export_transcript(project, project_dir)
    return {
        "format": parsed.source_format,
        "language": parsed.language,
        "timed": parsed.timed,
        "sentences": len(sentences),
        "qwen_aligned_sentences": aligned_count,
        "script_reconciled": reconciled,
    }
