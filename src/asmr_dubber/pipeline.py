from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from .asr import transcribe_japanese
from .asr_review import review_transcriptions
from .audio import (
    build_chinese_stem,
    copy_source_verbatim,
    make_analysis_copy,
    mix_original_and_stem,
    mux_mixed_video,
    project_file_exists,
    render_subtitled_video,
    resolve_project_path,
    sentence_events,
    verify_source,
)
from .constants import DEFAULT_PROJECTS_DIR
from .errors import ProjectError, SynthesisError
from .filtering import implausible_asr_reason, is_japanese_filler_only
from .forced_alignment import align_script_sentences_with_qwen, align_sentences_with_qwen
from .models import DubProject, ProjectSettings, Sentence, load_project, save_project
from .performance import measure_stage
from .platforms import portable_home, require_supported_platform
from .storage import atomic_write_text, exclusive_file_lock
from .subtitles import SubtitleLanguage, write_subtitle_files
from .task_control import CancellationSignal, check_cancelled
from .transcript_import import TranscriptLanguage, parse_transcript
from .translation import translate_sentences
from .tts import synthesize_sentences, tts_cache_key
from .user_settings import PROVIDER_PRESETS, resolve_api_key

Progress = Callable[[str, int, int], None]
_PROJECT_STAMP = re.compile(r"_\d{8}T\d{6}Z(?:_\d+)?$")
_INVALID_FILENAME = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')
_RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _INVALID_FILENAME.sub("_", normalized)
    normalized = re.sub(r"\s+", "_", normalized).strip(" ._-")
    normalized = normalized[:80].rstrip(" .")
    if not normalized:
        return "audio"
    if normalized.partition(".")[0].upper() in _RESERVED_WINDOWS_NAMES:
        normalized = f"_{normalized}"
    return normalized


def output_filename(project: DubProject, project_dir: Path) -> str:
    """Return a stable, human-readable filename for the current TTS setup."""
    source_label = _PROJECT_STAMP.sub("", project_dir.name) or "audio"
    model_label = Path(project.settings.tts_model.replace("\\", "/")).name
    reference_label = (
        f"{project.settings.tts_index_speaker_source}-{project.settings.tts_index_emotion_source}"
        if project.settings.tts_backend == "indextts2"
        else project.settings.tts_clone_mode
    )
    tts_label = _safe_name(f"{project.settings.tts_backend}-{model_label}-{reference_label}")
    return f"{_safe_name(source_label)}__ja-zh__{tts_label}.wav"


def output_video_filename(project: DubProject, project_dir: Path) -> str:
    return str(Path(output_filename(project, project_dir)).with_suffix(".mp4"))


def subtitle_video_filename(
    project_dir: Path,
    language: SubtitleLanguage,
    *,
    mixed: bool,
) -> str:
    source_label = _PROJECT_STAMP.sub("", project_dir.name) or "media"
    audio_label = "mixed" if mixed else "original"
    return f"{_safe_name(source_label)}__subtitles-{language}__{audio_label}.mp4"


def default_projects_dir() -> Path:
    configured = os.getenv("ASMR_DUBBER_PROJECTS")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_PROJECTS_DIR.resolve()


def create_project(
    source_audio: str | Path,
    projects_root: str | Path | None = None,
    settings: ProjectSettings | None = None,
    project_name: str | None = None,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[DubProject, Path]:
    check_cancelled(cancel_event)
    require_supported_platform()
    source = Path(source_audio).expanduser().resolve()
    root = Path(projects_root or default_projects_dir()).expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = _safe_name(project_name or source.stem)
    project_dir = root / f"{name}_{stamp}"
    suffix = 1
    while project_dir.exists():
        project_dir = root / f"{name}_{stamp}_{suffix}"
        suffix += 1
    project_dir.mkdir(parents=True)
    with measure_stage(project_dir, "project_init", source_bytes=source.stat().st_size):
        _, info = copy_source_verbatim(source, project_dir, progress=progress)
        check_cancelled(cancel_event)
        project = DubProject(source=info, settings=settings or ProjectSettings())
        save_project(project, project_dir)
        export_transcript(project, project_dir)
    return project, project_dir


def import_project_transcript(
    project: DubProject,
    project_dir: Path,
    *,
    transcript_path: str | Path | None = None,
    pasted_text: str = "",
    plain_timing: str = "estimate",
    script_language: str = "ja",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> dict[str, object]:
    """Replace ASR output with a user-provided script or timed subtitle file."""

    require_supported_platform()
    check_cancelled(cancel_event)
    if plain_timing not in {"estimate", "qwen"}:
        raise ProjectError("纯台本时间轴方式必须是按长度估算或 Qwen3 自动对齐。")
    if script_language not in {"ja", "zh"}:
        raise ProjectError("台本语言必须是日语或中文。")
    if script_language == "zh" and plain_timing == "qwen":
        raise ProjectError("中文纯台本不能使用 Qwen3 自动对齐，请选择按台词长度估算。")
    source = verify_source(project_dir, project.source)
    parsed = parse_transcript(
        duration_seconds=project.source.duration_seconds,
        path=transcript_path,
        pasted_text=pasted_text,
        language=cast(TranscriptLanguage, script_language),
    )
    sentences = [sentence.model_copy(deep=True) for sentence in parsed.sentences]
    alignment_report: list[dict[str, object]] = []
    if not parsed.timed and plain_timing == "qwen":
        analysis = make_analysis_copy(source, project_dir / "analysis" / "asr_16k_mono.wav")
        cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        alignment_report = align_script_sentences_with_qwen(
            analysis,
            sentences,
            project.settings,
            progress=progress,
            **cancel_kwargs,
        )
    check_cancelled(cancel_event)

    import_dir = project_dir / "imports"
    import_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(import_dir / "latest-transcript.txt", parsed.source_text)
    aligned_count = sum(not bool(item.get("fallback", True)) for item in alignment_report)
    atomic_write_text(
        import_dir / "latest-transcript.json",
        json.dumps(
            {
                "format": parsed.source_format,
                "language": parsed.language,
                "timed": parsed.timed,
                "plain_timing": None if parsed.timed else plain_timing,
                "sentences": len(sentences),
                "qwen_aligned_sentences": aligned_count,
                "alignment": alignment_report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    project.sentences = sentences
    language_name = "Japanese" if parsed.language == "ja" else "Chinese"
    project.asr_language = f"{language_name} (imported {parsed.source_format})"
    project.asr_settings_dirty = False
    project.settings.tts_reference_sentence_id = None
    project.chinese_stem_file = None
    project.output_file = None
    project.output_video_file = None
    project.subtitle_srt_file = None
    project.subtitle_lrc_file = None
    project.subtitle_video_file = None
    save_project(project, project_dir)
    export_transcript(project, project_dir)
    return {
        "format": parsed.source_format,
        "language": parsed.language,
        "timed": parsed.timed,
        "sentences": len(sentences),
        "qwen_aligned_sentences": aligned_count,
    }


def _analyze_project_impl(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    check_cancelled(cancel_event)
    require_supported_platform()
    if project.sentences and not force:
        if progress:
            progress(f"已存在 {len(project.sentences)} 句识别缓存", 1, 1)
        return
    source = verify_source(project_dir, project.source)
    analysis = make_analysis_copy(source, project_dir / "analysis" / "asr_16k_mono.wav")
    cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
    sentences, language = transcribe_japanese(
        analysis,
        project.settings,
        progress=progress,
        **cancel_kwargs,
    )
    if project.settings.asr_review_enabled:
        transcriptions: list[tuple[str, list[Sentence]]] = [
            (
                f"{project.settings.asr_backend}|{project.settings.asr_model}",
                sentences,
            )
        ]
        selected: list[tuple[str, str]] = []
        for value in project.settings.asr_review_models:
            backend, separator, model = str(value).partition("|")
            if not separator or not backend.strip() or not model.strip():
                raise ProjectError(f"多 ASR（语音识别）模型配置无效：{value}")
            pair = (backend.strip(), model.strip())
            if pair not in selected:
                selected.append(pair)
        primary_pair = (project.settings.asr_backend, project.settings.asr_model)
        comparison = [pair for pair in selected if pair != primary_pair]
        total_models = len(comparison) + 1
        for model_index, (backend, model) in enumerate(comparison, start=2):
            check_cancelled(cancel_event)
            if progress:
                progress(
                    f"多 ASR（语音识别）候选 {model_index}/{total_models}：{backend} · {model}",
                    model_index - 1,
                    total_models,
                )
            candidate_payload = project.settings.model_dump()
            candidate_payload.update(
                asr_backend=backend,
                asr_model=model,
                asr_review_enabled=False,
            )
            candidate_settings = ProjectSettings.model_validate(candidate_payload)
            candidate_sentences, _ = transcribe_japanese(
                analysis,
                candidate_settings,
                progress=progress,
                **cancel_kwargs,
            )
            transcriptions.append((f"{backend}|{model}", candidate_sentences))
        candidates_path = project_dir / "analysis" / "asr_candidates.json"
        candidates_path.write_text(
            json.dumps(
                {
                    "audio": str(analysis.relative_to(project_dir)),
                    "models": [
                        {
                            "source": label,
                            "sentences": [item.model_dump() for item in items],
                        }
                        for label, items in transcriptions
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        review_cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        sentences = review_transcriptions(
            transcriptions,
            project.settings,
            project_dir / "analysis" / "asr_review.json",
            analysis_audio=analysis,
            progress=progress,
            **review_cancel_kwargs,
        )
        check_cancelled(cancel_event)
        language = "Japanese (multi-ASR reviewed)"
    already_qwen_aligned = (
        project.settings.asr_review_enabled
        and project.settings.asr_review_timestamp_priority_model.startswith("qwen_forced_aligner|")
    )
    if project.settings.asr_forced_alignment_enabled and not already_qwen_aligned:
        alignment_report = align_sentences_with_qwen(
            analysis,
            sentences,
            project.settings,
            progress=progress,
            **cancel_kwargs,
        )
        alignment_path = project_dir / "analysis" / "asr_forced_alignment.json"
        alignment_path.parent.mkdir(parents=True, exist_ok=True)
        alignment_path.write_text(
            json.dumps(
                {
                    "model": project.settings.aligner_model,
                    "source": f"{project.settings.asr_backend}|{project.settings.asr_model}",
                    "sentences": alignment_report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        language = f"{language} (Qwen3 ForcedAligner)"
    # Encoded containers and the decoded analysis WAV can differ by a few samples
    # because of codec padding. Keep every reference boundary inside the real source.
    bounded_sentences = []
    for sentence in sentences:
        sentence.end_seconds = min(sentence.end_seconds, project.source.duration_seconds)
        if sentence.end_seconds > sentence.start_seconds:
            duration = sentence.end_seconds - sentence.start_seconds
            hallucination_reason = implausible_asr_reason(sentence.ja_text, duration)
            if hallucination_reason:
                sentence.enabled = False
                sentence.status = "skipped_hallucination"
                sentence.error = hallucination_reason
            elif project.settings.skip_japanese_fillers and is_japanese_filler_only(
                sentence.ja_text
            ):
                # Keep the Japanese row and original audio intact.  `enabled`
                # controls only translation and Chinese dubbing, and the user
                # can opt a row back in from the table.
                sentence.enabled = False
                sentence.status = "skipped_filler"
            bounded_sentences.append(sentence)
    if not bounded_sentences:
        raise ProjectError("识别时间戳全部落在源音频范围之外。")
    bounded_sentences.sort(
        key=lambda sentence: (sentence.start_seconds, sentence.end_seconds, sentence.id)
    )
    for index, sentence in enumerate(bounded_sentences, start=1):
        sentence.id = f"s{index:06d}"
    project.sentences = bounded_sentences
    project.asr_language = language
    project.asr_settings_dirty = False
    project.chinese_stem_file = None
    project.output_file = None
    project.output_video_file = None
    project.subtitle_srt_file = None
    project.subtitle_lrc_file = None
    project.subtitle_video_file = None
    save_project(project, project_dir)
    export_transcript(project, project_dir)


def _analyze_project_unlocked(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    with measure_stage(
        project_dir,
        "asr",
        backend=project.settings.asr_backend,
        model=project.settings.asr_model,
        audio_seconds=project.source.duration_seconds,
        requested_batch=project.settings.asr_batch_size,
    ) as metrics:
        cached = bool(project.sentences and not force)
        _analyze_project_impl(
            project,
            project_dir,
            force=force,
            progress=progress,
            cancel_event=cancel_event,
        )
        metrics["cache_hit"] = cached
        metrics["sentences"] = len(project.sentences)


def analyze_project(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    with exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=30.0):
        _analyze_project_unlocked(
            project,
            project_dir,
            force=force,
            progress=progress,
            cancel_event=cancel_event,
        )


def _translate_project_impl(
    project: DubProject,
    project_dir: Path,
    api_key: str | None = None,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    check_cancelled(cancel_event)
    require_supported_platform()
    if not project.sentences:
        raise ProjectError("项目还没有句子；请先运行识别。")
    will_translate = force or any(
        sentence.enabled and not sentence.zh_text for sentence in project.sentences
    )
    if force:
        for sentence in project.sentences:
            sentence.zh_text = ""
            sentence.tts_file = None
            sentence.tts_cache_key = None
            sentence.tts_duration_seconds = None
            sentence.status = "pending"
            sentence.error = None
    if will_translate:
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_srt_file = None
        project.subtitle_lrc_file = None
        project.subtitle_video_file = None
    provider = project.settings.translation_provider
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    key = resolve_api_key(provider, api_key)
    base_url = project.settings.translation_base_url.strip() or str(preset["base_url"])

    def checkpoint() -> None:
        save_project(project, project_dir)
        export_transcript(project, project_dir)

    translate_sentences(
        project.sentences,
        api_key=key,
        model=project.settings.translation_model,
        base_url=base_url,
        provider=provider,
        system_prompt=project.settings.translation_prompt,
        temperature=project.settings.translation_temperature,
        top_p=project.settings.translation_top_p,
        max_output_tokens=project.settings.translation_max_output_tokens,
        deepl_formality=project.settings.translation_deepl_formality,
        microsoft_region=project.settings.translation_microsoft_region,
        send_context=project.settings.translation_send_context,
        context_sentences=project.settings.translation_context_sentences,
        memory_sentences=project.settings.translation_memory_sentences,
        job_id=f"asmr_{project.source.sha256[:24]}",
        progress=progress,
        on_batch=checkpoint,
        cancel_event=cancel_event,
    )
    checkpoint()


def translate_project(
    project: DubProject,
    project_dir: Path,
    api_key: str | None = None,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    with measure_stage(
        project_dir,
        "translation",
        provider=project.settings.translation_provider,
        model=project.settings.translation_model,
        enabled_sentences=sum(sentence.enabled for sentence in project.sentences),
    ) as metrics:
        before = sum(bool(sentence.zh_text) for sentence in project.sentences)
        _translate_project_impl(
            project,
            project_dir,
            api_key=api_key,
            force=force,
            progress=progress,
            cancel_event=cancel_event,
        )
        metrics["translated_before"] = before
        metrics["translated_after"] = sum(bool(sentence.zh_text) for sentence in project.sentences)


def _synthesize_project_impl(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    sentence_ids: list[str] | None = None,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    check_cancelled(cancel_event)
    require_supported_platform()
    source = verify_source(project_dir, project.source)
    requested = set(sentence_ids) if sentence_ids is not None else None
    needs_generation = any(
        sentence.enabled
        and sentence.zh_text
        and (requested is None or sentence.id in requested)
        and (
            force
            or not sentence.tts_file
            or not project_file_exists(
                project_dir,
                sentence.tts_file,
                f"句子 {sentence.id} 的中文音频",
            )
            or sentence.tts_cache_key != tts_cache_key(project, sentence)
        )
        for sentence in project.sentences
    )
    if needs_generation:
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_video_file = None

    def checkpoint() -> None:
        save_project(project, project_dir)
        export_transcript(project, project_dir)

    cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
    failures = synthesize_sentences(
        project,
        project_dir,
        source,
        force=force,
        sentence_ids=sentence_ids,
        progress=progress,
        on_sentence=checkpoint,
        **cancel_kwargs,
    )
    checkpoint()
    if failures:
        preview = "\n".join(failures[:12])
        remainder = f"\n另有 {len(failures) - 12} 句失败" if len(failures) > 12 else ""
        raise SynthesisError(
            "部分句子的逐句克隆失败，已保留成功缓存。请在表格中修正时间/文本后重试：\n"
            + preview
            + remainder
        )


def _synthesize_project_unlocked(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    sentence_ids: list[str] | None = None,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    with measure_stage(
        project_dir,
        "tts",
        backend=project.settings.tts_backend,
        model=project.settings.tts_model,
        requested_sentences=(
            len(sentence_ids)
            if sentence_ids is not None
            else sum(sentence.enabled and bool(sentence.zh_text) for sentence in project.sentences)
        ),
    ) as metrics:
        before = sum(
            project_file_exists(
                project_dir,
                sentence.tts_file,
                f"句子 {sentence.id} 的中文音频",
            )
            for sentence in project.sentences
        )
        _synthesize_project_impl(
            project,
            project_dir,
            force=force,
            sentence_ids=sentence_ids,
            progress=progress,
            cancel_event=cancel_event,
        )
        after = sum(
            project_file_exists(
                project_dir,
                sentence.tts_file,
                f"句子 {sentence.id} 的中文音频",
            )
            for sentence in project.sentences
        )
        metrics["cached_before"] = before
        metrics["available_after"] = after


def synthesize_project(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    sentence_ids: list[str] | None = None,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    with exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=30.0):
        _synthesize_project_unlocked(
            project,
            project_dir,
            force=force,
            sentence_ids=sentence_ids,
            progress=progress,
            cancel_event=cancel_event,
        )


def _mix_project_impl(
    project: DubProject,
    project_dir: Path,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> Path:
    check_cancelled(cancel_event)
    require_supported_platform()
    source = verify_source(project_dir, project.source)
    missing = [
        sentence.id
        for sentence in project.sentences
        if sentence.enabled
        and sentence.zh_text
        and (not sentence.tts_file or sentence.tts_cache_key != tts_cache_key(project, sentence))
    ]
    if missing:
        raise ProjectError(
            f"仍有 {len(missing)} 句的中文配音缺失或已过期：{', '.join(missing[:12])}"
        )
    events = sentence_events(
        project_dir,
        project.sentences,
        project.settings.global_overlap_seconds,
        project.settings.global_overlap_percentage,
    )
    if not events:
        raise ProjectError("没有可混入的中文配音。")
    stem = project_dir / "mix" / "chinese_stem_float32.wav"
    output = project_dir / "output" / output_filename(project, project_dir)
    loudness_reference = make_analysis_copy(
        source,
        project_dir / "analysis" / "asr_16k_mono.wav",
    )
    if progress:
        progress("按对应日语句子校准中文响度（不移动或修改原音频）", 0, max(1, len(events)))
    build_chinese_stem(
        destination=stem,
        events=events,
        source_info=project.source,
        chinese_gain_db=project.settings.chinese_gain_db,
        normalize_loudness=project.settings.normalize_chinese_loudness,
        source_reference_path=loudness_reference,
        match_source_loudness=project.settings.match_source_loudness,
        relative_loudness_db=project.settings.chinese_relative_loudness_db,
        minimum_active_rms_dbfs=project.settings.chinese_min_active_rms_dbfs,
        target_active_rms_dbfs=project.settings.chinese_target_active_rms_dbfs,
        max_loudness_boost_db=project.settings.chinese_max_loudness_boost_db,
        line_peak_dbfs=project.settings.chinese_line_peak_dbfs,
        stem_peak_dbfs=project.settings.chinese_stem_peak_dbfs,
        fade_ms=project.settings.chinese_fade_ms,
        channel_routing=project.settings.chinese_channel_routing,
        progress=progress,
    )
    check_cancelled(cancel_event)
    if progress:
        progress("正在混合原轨与中文轨，并执行最终峰值保护", 0, 1)
    mix_original_and_stem(
        source,
        stem,
        output,
        project.source,
        output_codec="pcm_s24le",
        peak_protection=project.settings.mix_peak_protection,
        peak_limit_dbfs=project.settings.mix_peak_limit_dbfs,
    )
    check_cancelled(cancel_event)
    if project.settings.retain_chinese_stem:
        project.chinese_stem_file = str(stem.relative_to(project_dir))
    else:
        try:
            stem.unlink()
            project.chinese_stem_file = None
        except OSError:
            # The final output is already complete.  Retain the path when an
            # antivirus/player still holds the intermediate file on Windows.
            project.chinese_stem_file = str(stem.relative_to(project_dir))
            if progress:
                progress("最终音频已完成；中文中间轨暂时被占用，未能自动删除", 1, 1)
    project.output_file = str(output.relative_to(project_dir))
    project.output_video_file = None
    project.subtitle_video_file = None
    save_project(project, project_dir)
    if project.source.media_type == "video":
        if progress:
            progress("保留原画面并封装中文混合音轨", 0, 1)
        video_output = mux_mixed_video(
            source,
            output,
            project_dir / "output" / output_video_filename(project, project_dir),
        )
        project.output_video_file = video_output.relative_to(project_dir).as_posix()
    save_project(project, project_dir)
    export_transcript(project, project_dir)
    if progress:
        progress("混音完成", 1, 1)
    return output


def mix_project(
    project: DubProject,
    project_dir: Path,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> Path:
    with measure_stage(
        project_dir,
        "mix",
        audio_seconds=project.source.duration_seconds,
        sample_rate=project.source.sample_rate,
        channels=project.source.channels,
        retained_stem=project.settings.retain_chinese_stem,
    ) as metrics:
        output = _mix_project_impl(
            project,
            project_dir,
            progress=progress,
            cancel_event=cancel_event,
        )
        metrics["output_bytes"] = output.stat().st_size
        if project.output_video_file:
            video_output = resolve_project_path(
                project_dir,
                project.output_video_file,
                "混音视频",
            )
            if video_output.is_file():
                metrics["video_output_bytes"] = video_output.stat().st_size
        return output


def generate_subtitles(
    project: DubProject,
    project_dir: Path,
    language: SubtitleLanguage = "bilingual",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[Path, Path, Path | None]:
    """Create external subtitles and, for video projects, a subtitled video."""
    require_supported_platform()
    check_cancelled(cancel_event)
    source = verify_source(project_dir, project.source)

    if progress:
        progress("生成 SRT 与 LRC 字幕", 0, 2 if project.source.media_type == "video" else 1)
    srt, lrc = write_subtitle_files(
        project.sentences,
        project_dir / "subtitles",
        language,
        timeline=project.settings.subtitle_timeline,
        maximum_chars=project.settings.subtitle_max_chars_per_line,
        minimum_duration=project.settings.subtitle_min_duration_seconds,
        maximum_cps=project.settings.subtitle_max_cps,
        global_overlap_seconds=project.settings.global_overlap_seconds,
        global_overlap_percentage=project.settings.global_overlap_percentage,
    )
    check_cancelled(cancel_event)

    video_output: Path | None = None
    if project.source.media_type == "video":
        mixed_audio: Path | None = None
        if project.output_file:
            candidate = resolve_project_path(project_dir, project.output_file, "完成音频")
            if candidate.is_file():
                mixed_audio = candidate
        if progress:
            progress("生成带字幕视频", 1, 2)
        video_output = render_subtitled_video(
            source,
            srt,
            project_dir
            / "output"
            / subtitle_video_filename(project_dir, language, mixed=mixed_audio is not None),
            replacement_audio=mixed_audio,
            subtitle_language=language,
        )
        check_cancelled(cancel_event)
    # Commit metadata only after every requested artifact succeeds. Cancellation
    # or FFmpeg failure must leave the previous downloadable outputs visible.
    project.subtitle_language = language
    project.subtitle_srt_file = srt.relative_to(project_dir).as_posix()
    project.subtitle_lrc_file = lrc.relative_to(project_dir).as_posix()
    project.subtitle_video_file = (
        video_output.relative_to(project_dir).as_posix() if video_output is not None else None
    )
    save_project(project, project_dir)
    if progress:
        progress("字幕生成完成", 1, 1)
    return srt, lrc, video_output


def export_transcript(project: DubProject, project_dir: Path) -> None:
    exports = project_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": sentence.id,
            "enabled": sentence.enabled,
            "ja_start_seconds": sentence.start_seconds,
            "ja_end_seconds": sentence.end_seconds,
            "zh_start_seconds": sentence.chinese_start_seconds(
                project.settings.global_overlap_seconds,
                project.settings.global_overlap_percentage,
            ),
            "overlap_seconds": sentence.overlap_seconds,
            "ja": sentence.ja_text,
            "zh": sentence.zh_text,
            "status": sentence.status,
            "error": sentence.error,
        }
        for sentence in project.sentences
    ]
    (exports / "transcript.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (exports / "transcript.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)


def reload_project(path: str | Path) -> tuple[DubProject, Path]:
    require_supported_platform()
    return load_project(path)
