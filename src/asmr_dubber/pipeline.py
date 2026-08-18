from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from .asr import transcribe_source
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
from .languages import SourceLanguage, source_language_label
from .models import (
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
    settings_for_source_language,
)
from .performance import measure_stage
from .platforms import portable_home, require_supported_platform
from .storage import atomic_write_text, exclusive_file_lock
from .subtitles import SubtitleLanguage, write_subtitle_files
from .task_control import CancellationSignal, check_cancelled
from .timing import dubbing_start_seconds, plan_dubbing_timing
from .transcript_import import TranscriptLanguage, parse_transcript
from .translation import (
    LLM_RECONCILIATION_PROVIDERS,
    reconcile_script_sentences,
    translate_sentences,
)
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
    return f"{_safe_name(source_label)}__{project.source_language}-zh__{tts_label}.wav"


def stem_output_filename(project: DubProject, project_dir: Path) -> str:
    mixed = Path(output_filename(project, project_dir))
    return f"{mixed.stem}__chinese-voice-stem.wav"


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
    source_language: SourceLanguage = "ja",
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
        project_settings = settings_for_source_language(
            settings or ProjectSettings(),
            source_language,
        )
        project = DubProject(
            source=info,
            source_language=source_language,
            settings=project_settings,
        )
        save_project(project, project_dir)
        export_transcript(project, project_dir)
    return project, project_dir


def _reconcile_untimed_script(
    project: DubProject,
    project_dir: Path,
    *,
    script_lines: list[str],
    script_language: TranscriptLanguage,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[list[Sentence], list[dict[str, object]]]:
    """Run the normal ASR/translation flow in a temporary project, then apply the script text."""

    provider = project.settings.translation_provider
    if provider not in LLM_RECONCILIATION_PROVIDERS:
        raise ProjectError(
            "无时间轴台本的智能校对需要大模型翻译服务；请先在设置中选择 DeepSeek、百炼、"
            "豆包、OpenAI、Claude、Gemini 或本地/自定义 OpenAI-compatible。"
        )
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    key = resolve_api_key(provider)
    base_url = project.settings.translation_base_url.strip() or str(preset["base_url"])
    source = verify_source(project_dir, project.source)
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
        _analyze_project_impl(
            working,
            working_dir,
            force=True,
            progress=progress,
            cancel_event=cancel_event,
        )
        target: Literal["source", "zh"] = "source"
        if script_language == "zh":
            _translate_project_impl(
                working,
                working_dir,
                api_key=key,
                force=True,
                progress=progress,
                cancel_event=cancel_event,
            )
            target = "zh"
        corrections, report = reconcile_script_sentences(
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
            job_id=f"asmr_{project.source.sha256[:24]}_script",
            progress=progress,
            cancel_event=cancel_event,
        )
        fallback_ids: list[str] = []
        for sentence in working.sentences:
            candidate = sentence.source_text if target == "source" else sentence.zh_text
            corrected = str(corrections.get(sentence.id, "")).strip()
            if not corrected and candidate.strip():
                # A malformed or over-aggressive empty answer must not silently
                # delete a real ASR/translation line.
                corrected = candidate.strip()
                fallback_ids.append(sentence.id)
            if target == "source":
                sentence.source_text = corrected
                sentence.zh_text = ""
                sentence.status = "pending" if corrected else "skipped_filler"
            else:
                sentence.zh_text = corrected
                sentence.status = "translated" if corrected else "skipped_filler"
            sentence.enabled = bool(corrected)
            sentence.tts_file = None
            sentence.tts_cache_key = None
            sentence.tts_duration_seconds = None
            sentence.error = None
        if fallback_ids:
            report.append({"fallback_ids": fallback_ids, "reason": "模型返回空文本，保留候选文本"})
        return [item.model_copy(deep=True) for item in working.sentences], report


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
    if plain_timing not in {"estimate", "qwen", "script_review"}:
        raise ProjectError("纯台本时间轴方式无效。")
    if script_language not in {"ja", "en", "zh"}:
        raise ProjectError("台本语言必须是日语、英语或中文。")
    if script_language == "zh" and plain_timing == "qwen":
        raise ProjectError("中文纯台本不能使用 Qwen3 自动对齐，请选择按台词长度估算。")
    if plain_timing == "script_review" and project.source_language == "zh":
        raise ProjectError(
            "当前项目已经是中文台本项目，不能再次运行源音频识别；请新建原始音频项目。"
        )
    source = verify_source(project_dir, project.source)
    parsed = parse_transcript(
        duration_seconds=project.source.duration_seconds,
        path=transcript_path,
        pasted_text=pasted_text,
        language=cast(TranscriptLanguage, script_language),
    )
    sentences = [sentence.model_copy(deep=True) for sentence in parsed.sentences]
    alignment_report: list[dict[str, object]] = []
    reconciliation_report: list[dict[str, object]] = []
    reconciled = False
    if not parsed.timed and plain_timing == "script_review":
        script_lines = [
            sentence.zh_text if script_language == "zh" else sentence.source_text
            for sentence in parsed.sentences
        ]
        sentences, reconciliation_report = _reconcile_untimed_script(
            project,
            project_dir,
            script_lines=script_lines,
            script_language=cast(TranscriptLanguage, script_language),
            progress=progress,
            cancel_event=cancel_event,
        )
        reconciled = True
    elif not parsed.timed and plain_timing == "qwen":
        analysis = make_analysis_copy(source, project_dir / "analysis" / "asr_16k_mono.wav")
        cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        alignment_report = align_script_sentences_with_qwen(
            analysis,
            sentences,
            project.settings,
            source_language=cast(SourceLanguage, script_language),
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
    if not reconciled:
        project.source_language = cast(SourceLanguage, parsed.language)
        project.settings = settings_for_source_language(project.settings, project.source_language)
        language_name = source_language_label(parsed.language)
        project.asr_language = f"{language_name} (imported {parsed.source_format})"
    else:
        language_name = source_language_label(project.source_language)
        suffix = "翻译 + 台本校对" if parsed.language == "zh" else "台本校对"
        project.asr_language = f"{language_name}（ASR + {suffix}）"
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
        "script_reconciled": reconciled,
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
    if project.source_language == "zh":
        raise ProjectError("中文台本项目不需要运行 ASR（语音识别）。")
    project.settings = settings_for_source_language(project.settings, project.source_language)
    if project.sentences and not force:
        if progress:
            progress(f"已存在 {len(project.sentences)} 句识别缓存", 1, 1)
        return
    source = verify_source(project_dir, project.source)
    analysis = make_analysis_copy(source, project_dir / "analysis" / "asr_16k_mono.wav")
    cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
    sentences, language = transcribe_source(
        analysis,
        project.settings,
        source_language=project.source_language,
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
            candidate_sentences, _ = transcribe_source(
                analysis,
                candidate_settings,
                source_language=project.source_language,
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
            source_language=project.source_language,
            progress=progress,
            **review_cancel_kwargs,
        )
        check_cancelled(cancel_event)
        language = f"{source_language_label(project.source_language)}（多模型校对）"
    already_qwen_aligned = (
        project.settings.asr_review_enabled
        and project.settings.asr_review_timestamp_priority_model.startswith("qwen_forced_aligner|")
    )
    if project.settings.asr_forced_alignment_enabled and not already_qwen_aligned:
        alignment_report = align_sentences_with_qwen(
            analysis,
            sentences,
            project.settings,
            source_language=project.source_language,
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
            hallucination_reason = implausible_asr_reason(sentence.source_text, duration)
            if hallucination_reason:
                sentence.enabled = False
                sentence.status = "skipped_hallucination"
                sentence.error = hallucination_reason
            elif (
                project.source_language == "ja"
                and project.settings.skip_japanese_fillers
                and is_japanese_filler_only(sentence.source_text)
            ):
                # Keep the source row and original audio intact.  `enabled`
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
        source_language=project.source_language,
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
        project.settings.chinese_dubbing_offset_ms,
        project.settings.chinese_max_auto_speed,
        project.settings.chinese_dubbing_timing_mode,
    )
    if not events:
        raise ProjectError("没有可混入的中文配音。")
    output_mode = project.settings.mix_output_mode
    keep_stem = output_mode in {"stem", "both"}
    stem = (
        project_dir / "output" / stem_output_filename(project, project_dir)
        if keep_stem
        else project_dir / "mix" / "chinese_stem_float32.wav"
    )
    output = project_dir / "output" / output_filename(project, project_dir)
    loudness_reference = make_analysis_copy(
        source,
        project_dir / "analysis" / "asr_16k_mono.wav",
    )
    if progress:
        progress("按对应源语言句子校准中文响度（不移动或修改原音频）", 0, max(1, len(events)))
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
    mixed_output: Path | None = None
    if output_mode in {"mixed", "both"}:
        if progress:
            progress("正在把中文克隆音轨加入原音轨，并执行最终峰值保护", 0, 1)
        mix_original_and_stem(
            source,
            stem,
            output,
            project.source,
            output_codec="pcm_s24le",
            peak_protection=project.settings.mix_peak_protection,
            peak_limit_dbfs=project.settings.mix_peak_limit_dbfs,
        )
        mixed_output = output
        check_cancelled(cancel_event)
    if keep_stem:
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
    project.output_file = (
        str(mixed_output.relative_to(project_dir)) if mixed_output is not None else None
    )
    project.output_video_file = None
    project.subtitle_video_file = None
    save_project(project, project_dir)
    if project.source.media_type == "video" and mixed_output is not None:
        if progress:
            progress("保留原画面并封装中文混合音轨", 0, 1)
        video_output = mux_mixed_video(
            source,
            mixed_output,
            project_dir / "output" / output_video_filename(project, project_dir),
        )
        project.output_video_file = video_output.relative_to(project_dir).as_posix()
    save_project(project, project_dir)
    export_transcript(project, project_dir)
    if progress:
        progress("中文配音输出完成", 1, 1)
    return mixed_output or stem


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
        output_mode=project.settings.mix_output_mode,
    ) as metrics:
        output = _mix_project_impl(
            project,
            project_dir,
            progress=progress,
            cancel_event=cancel_event,
        )
        metrics["output_bytes"] = output.stat().st_size
        if project.chinese_stem_file:
            stem_output = resolve_project_path(
                project_dir,
                project.chinese_stem_file,
                "中文克隆音轨",
            )
            if stem_output.is_file():
                metrics["stem_output_bytes"] = stem_output.stat().st_size
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
    language: SubtitleLanguage | Literal["ja"] = "bilingual",
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[Path, Path, Path | None]:
    """Create external subtitles and, for video projects, a subtitled video."""
    require_supported_platform()
    check_cancelled(cancel_event)
    normalized_language: SubtitleLanguage = "source" if language == "ja" else language
    source = verify_source(project_dir, project.source)

    if progress:
        progress("生成 SRT 与 LRC 字幕", 0, 2 if project.source.media_type == "video" else 1)
    srt, lrc = write_subtitle_files(
        project.sentences,
        project_dir / "subtitles",
        normalized_language,
        timeline=project.settings.subtitle_timeline,
        maximum_chars=project.settings.subtitle_max_chars_per_line,
        minimum_duration=project.settings.subtitle_min_duration_seconds,
        maximum_cps=project.settings.subtitle_max_cps,
        chinese_dubbing_offset_ms=project.settings.chinese_dubbing_offset_ms,
        chinese_max_auto_speed=project.settings.chinese_max_auto_speed,
        chinese_dubbing_timing_mode=project.settings.chinese_dubbing_timing_mode,
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
            / subtitle_video_filename(
                project_dir,
                normalized_language,
                mixed=mixed_audio is not None,
            ),
            replacement_audio=mixed_audio,
            subtitle_language=normalized_language,
            source_language=project.source_language,
        )
        check_cancelled(cancel_event)
    # Commit metadata only after every requested artifact succeeds. Cancellation
    # or FFmpeg failure must leave the previous downloadable outputs visible.
    project.subtitle_language = normalized_language
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
    timing_by_id = {
        timing.sentence_id: timing
        for timing in plan_dubbing_timing(
            project.sentences,
            offset_ms=project.settings.chinese_dubbing_offset_ms,
            max_auto_speed=project.settings.chinese_max_auto_speed,
            mode=project.settings.chinese_dubbing_timing_mode,
        )
    }
    rows = [
        {
            "id": sentence.id,
            "enabled": sentence.enabled,
            "source_language": project.source_language,
            "source_start_seconds": sentence.start_seconds,
            "source_end_seconds": sentence.end_seconds,
            "zh_start_seconds": (
                timing_by_id[sentence.id].start_seconds
                if sentence.id in timing_by_id
                else dubbing_start_seconds(
                    sentence,
                    project.settings.chinese_dubbing_offset_ms,
                )
            ),
            "tts_duration_seconds": sentence.tts_duration_seconds,
            "auto_speed_factor": (
                timing_by_id[sentence.id].speed_factor if sentence.id in timing_by_id else 1.0
            ),
            "zh_effective_duration_seconds": (
                timing_by_id[sentence.id].effective_duration_seconds
                if sentence.id in timing_by_id
                else sentence.tts_duration_seconds
            ),
            "remaining_overlap_seconds": (
                timing_by_id[sentence.id].remaining_overlap_seconds
                if sentence.id in timing_by_id
                else 0.0
            ),
            "source": sentence.source_text,
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
