from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .asr import transcribe_source
from .asr_review import run_audio_review
from .audio import (
    ASR_ANALYSIS_NAME,
    copy_source_verbatim,
    make_asr_analysis_copy,
    project_file_exists,
    verify_source,
)
from .audio import build_chinese_stem as build_chinese_stem
from .audio import make_analysis_copy as make_analysis_copy
from .audio import mix_original_and_stem as mix_original_and_stem
from .audio import mux_mixed_video as mux_mixed_video
from .audio import render_subtitled_video as render_subtitled_video
from .audio import resolve_project_path as resolve_project_path
from .audio import sentence_events as sentence_events
from .audio import speech_bounds as speech_bounds
from .constants import DEFAULT_PROJECTS_DIR
from .errors import ProjectError, SynthesisError, TranslationError
from .filtering import (
    has_speakable_text,
    implausible_asr_reason,
    is_japanese_filler_only,
    is_negligible_sound_segment,
)
from .forced_alignment import align_script_sentences_with_qwen as align_script_sentences_with_qwen
from .forced_alignment import align_sentences_with_qwen
from .languages import SourceLanguage
from .languages import source_language_label as source_language_label
from .lifecycle import invalidate_outputs, project_operation
from .models import (
    DubProject,
    ProjectSettings,
    load_project,
    save_project,
    settings_for_source_language,
)
from .models import Sentence as Sentence
from .performance import measure_stage
from .pipeline_mix import _mix_project_impl as _mix_project_impl
from .pipeline_mix import _sentence_needs_dubbing as _sentence_needs_dubbing
from .pipeline_mix import mix_project as mix_project
from .pipeline_subtitles import _replacement_dubbing_durations as _replacement_dubbing_durations
from .pipeline_subtitles import export_transcript
from .pipeline_subtitles import generate_subtitles as generate_subtitles
from .pipeline_transcript import _chinese_script_timing_warnings as _chinese_script_timing_warnings
from .pipeline_transcript import _reconcile_untimed_script as _reconcile_untimed_script
from .pipeline_transcript import import_project_transcript as import_project_transcript
from .pipeline_transcript import (
    reconcile_analyzed_project_script as reconcile_analyzed_project_script,
)
from .platforms import portable_home, require_supported_platform
from .storage import atomic_write_text, exclusive_file_lock
from .subtitles import SubtitleLanguage
from .subtitles import write_subtitle_files as write_subtitle_files
from .task_control import CancellationSignal, check_cancelled
from .timing import dubbing_start_seconds as dubbing_start_seconds
from .timing import plan_dubbing_timing as plan_dubbing_timing
from .transcript_import import TranscriptLanguage as TranscriptLanguage
from .transcript_import import parse_transcript as parse_transcript
from .translation import LLM_RECONCILIATION_PROVIDERS as LLM_RECONCILIATION_PROVIDERS
from .translation import reconcile_script_sentences as reconcile_script_sentences
from .translation import translate_sentences
from .tts import synthesize_sentences, valid_tts_cache
from .tts import tts_cache_key as tts_cache_key
from .user_settings import PROVIDER_PRESETS, resolve_api_key

Progress = Callable[[str, int, int], None]
logger = logging.getLogger(__name__)
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
_SCRIPT_VISIBLE_CHARACTER = re.compile(r"[^\s，。！？、；：,.!?;:'\"“”‘’（）()\[\]{}<>《》…—-]")


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
        if project.settings.tts_backend in {"indextts2_5", "indextts2"}
        else project.settings.tts_clone_mode
    )
    tts_label = _safe_name(f"{project.settings.tts_backend}-{model_label}-{reference_label}")
    return (
        f"{_safe_name(source_label)}__{project.source_language}-"
        f"{project.settings.tts_target_language}__{tts_label}.wav"
    )


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
            translation_language=project_settings.tts_target_language,
        )
        save_project(project, project_dir)
        export_transcript(project, project_dir)
    return project, project_dir


def _analyze_project_impl(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    check_cancelled(cancel_event)
    require_supported_platform()
    project.settings = settings_for_source_language(project.settings, project.source_language)
    if force and any(row.review_locked for row in project.sentences):
        raise ProjectError(
            "项目包含人工确认的句子。请先在复核面板解除确认锁定，再重跑 ASR；也可只重试复核。"
        )
    if project.sentences and not force:
        if progress:
            progress(f"已存在 {len(project.sentences)} 句识别缓存", 1, 1)
        return
    source = verify_source(project_dir, project.source)
    vocals: Path | None = None
    if project.settings.separation_enabled:
        from .separation import ensure_separation

        vocals, _background = ensure_separation(project, project_dir, source, progress)
        analysis = make_asr_analysis_copy(vocals, vocals.parent / ASR_ANALYSIS_NAME)
    else:
        analysis = make_asr_analysis_copy(source, project_dir / "analysis" / ASR_ANALYSIS_NAME)
    cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
    sentences, language = transcribe_source(
        analysis,
        project.settings,
        source_language=project.source_language,
        progress=progress,
        **cancel_kwargs,
    )
    if project.settings.asr_forced_alignment_enabled:
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
    previous_text: str | None = None
    for sentence in sentences:
        sentence.end_seconds = min(sentence.end_seconds, project.source.duration_seconds)
        if sentence.end_seconds > sentence.start_seconds:
            if sentence.status == "skipped_sound":
                sentence.enabled = False
                bounded_sentences.append(sentence)
                continue
            duration = sentence.end_seconds - sentence.start_seconds
            hallucination_reason = implausible_asr_reason(
                sentence.source_text, duration, previous_text
            )
            previous_text = sentence.source_text
            if hallucination_reason:
                sentence.enabled = False
                sentence.status = "skipped_hallucination"
                sentence.set_error(hallucination_reason)
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
            elif is_negligible_sound_segment(sentence.source_text, duration):
                # A fraction of a second with one character is a click or a
                # breath.  Keep the row visible; the user can enable it again.
                sentence.enabled = False
                sentence.status = "skipped_sound"
            bounded_sentences.append(sentence)
    if not bounded_sentences:
        raise ProjectError("识别时间戳全部落在源音频范围之外。")
    bounded_sentences.sort(
        key=lambda sentence: (sentence.start_seconds, sentence.end_seconds, sentence.id)
    )
    from .diarization import diarize_sentences
    from .dub_qc import detect_uncovered_vocal_sounds, snap_sentences_to_silence
    from .voice_reference import annotate_sentence_voice_modes

    bounded_sentences = snap_sentences_to_silence(bounded_sentences, analysis)
    if vocals is not None and project.source_language == "ja":
        bounded_sentences = detect_uncovered_vocal_sounds(bounded_sentences, vocals)
    for index, sentence in enumerate(bounded_sentences, start=1):
        sentence.id = f"s{index:06d}"
    diarize_audio = vocals if vocals is not None else source
    if project.settings.asr_diarize_speakers or any(
        s.speaker_id and s.speaker_id.startswith("c") and ":S" in s.speaker_id
        for s in bounded_sentences
    ):
        spk_count = project.settings.asr_speaker_count or None
        bounded_sentences = diarize_sentences(
            bounded_sentences, diarize_audio, num_speakers=spk_count
        )
    project.sentences = bounded_sentences
    project.translation_proposals = []
    project.settings.tts_reference_sentence_id = None
    project.settings.tts_whisper_reference_sentence_id = None
    annotate_sentence_voice_modes(project, diarize_audio)
    project.translation_language = project.settings.tts_target_language
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

    if project.settings.asr_review_enabled:
        # Commit the usable single-model baseline before optional reviewers run.
        baseline = [row.model_copy(deep=True) for row in project.sentences]
        snapshot = project_dir / "analysis" / f"asr-baseline-r{project.revision}.json"
        atomic_write_text(
            snapshot,
            json.dumps([row.model_dump() for row in baseline], ensure_ascii=False, indent=2),
        )
        project.sentences = run_audio_review(
            analysis,
            baseline,
            project.settings,
            project_dir / "analysis" / "asr_review.json",
            source_language=project.source_language,
            progress=progress,
            cancel_event=cancel_event,
        )
        project.asr_language = f"{language}（音频片段复核；主稿独立保留）"
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


@project_operation
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
    overwrite_edited: bool = False,
    sentence_ids: list[str] | None = None,
    compress: bool = False,
) -> None:
    check_cancelled(cancel_event)
    require_supported_platform()
    if not project.sentences:
        raise ProjectError("项目还没有句子；请先运行识别。")
    if project.translation_language != project.settings.tts_target_language:
        atomic_write_text(
            project_dir / f"translations-{project.translation_language}-{project.revision}.json",
            project.model_dump_json(indent=2),
        )
        preserved_skip_statuses = {"skipped_sound", "skipped_filler", "skipped_hallucination"}
        for sentence in project.sentences:
            sentence.zh_text = ""
            sentence.tts_file = None
            sentence.tts_cache_key = None
            sentence.tts_duration_seconds = None
            sentence.tts_candidates = []
            sentence.tts_warning = None
            sentence.tts_attempt = 0
            sentence.translation_warning = None
            sentence.review_locked = False
            if sentence.status not in preserved_skip_statuses:
                sentence.status = "pending"
                sentence.clear_error()
            elif sentence.status != "skipped_hallucination":
                sentence.clear_error()
        project.translation_proposals = []
        project.translation_language = project.settings.tts_target_language
        save_project(project, project_dir)
    has_selected_ids = bool(sentence_ids)
    will_translate = (
        force
        or compress
        or has_selected_ids
        or any(
            sentence.enabled and (not sentence.zh_text or sentence.status == "translation_failed")
            for sentence in project.sentences
        )
    )
    if force or compress or has_selected_ids:
        # Keep the old text until its replacement has arrived, and leave a copy so
        # an interrupted or disappointing run can be undone.
        from .translation_backup import write_translation_backup

        write_translation_backup(project, project_dir)
    if will_translate and not compress:
        invalidate_outputs(project)
    provider = project.settings.translation_provider
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    key = (
        resolve_api_key(provider, api_key)
        if will_translate and project.source_language != project.settings.tts_target_language
        else ""
    )
    base_url = project.settings.translation_base_url.strip() or str(preset["base_url"])

    before_translation = {sentence.id: sentence.zh_text for sentence in project.sentences}

    def checkpoint() -> None:
        if compress and any(
            sentence.zh_text != before_translation.get(sentence.id)
            for sentence in project.sentences
        ):
            invalidate_outputs(project)
        save_project(project, project_dir)
        export_transcript(project, project_dir)

    budget_key = "max_words" if project.settings.tts_target_language == "en" else "max_chars"
    budget_rate = (
        (
            project.settings.translation_chars_per_second
            if project.settings.tts_target_language != "en"
            else project.settings.translation_chars_per_second * 0.6
        )
        if (project.settings.translation_length_budget or compress)
        else None
    )
    report = None
    try:
        report = translate_sentences(
            project.sentences,
            api_key=key,
            model=project.settings.translation_model,
            base_url=base_url,
            provider=provider,
            source_language=project.source_language,
            system_prompt=project.settings.translation_prompt,
            target_language=project.settings.tts_target_language,
            temperature=project.settings.translation_temperature,
            top_p=project.settings.translation_top_p,
            max_output_tokens=project.settings.translation_max_output_tokens,
            extra_body=project.settings.translation_extra_body,
            deepl_formality=project.settings.translation_deepl_formality,
            microsoft_region=project.settings.translation_microsoft_region,
            send_context=project.settings.translation_send_context,
            context_sentences=project.settings.translation_context_sentences,
            memory_sentences=project.settings.translation_memory_sentences,
            job_id=f"asmr_{project.source.sha256[:24]}",
            progress=progress,
            on_batch=checkpoint,
            cancel_event=cancel_event,
            sound_policy=project.settings.translation_sound_policy,
            retranslate=force,
            overwrite_edited=overwrite_edited,
            sentence_ids=sentence_ids,
            compress=compress,
            length_budget_rate=budget_rate,
        )
    finally:
        changed_sentences = [
            sentence
            for sentence in project.sentences
            if sentence.zh_text != before_translation.get(sentence.id)
        ]
        if changed_sentences:
            if compress:
                invalidate_outputs(project)
            if report is not None:
                flagged = report.flagged
            else:
                from .translation import check_translations, length_budgets

                changed_with_text = [s for s in changed_sentences if s.zh_text.strip()]
                budgets = (
                    length_budgets(list(project.sentences), changed_with_text, budget_rate)
                    if (budget_rate and changed_with_text)
                    else {}
                )
                flagged = (
                    check_translations(
                        project.sentences,
                        budgets,
                        budget_key,
                        only_ids=[s.id for s in changed_with_text],
                        source_language=project.source_language,
                        target_language=project.settings.tts_target_language,
                    )
                    if changed_with_text
                    else []
                )
            reasons: dict[str, list[str]] = {}
            for flag in flagged:
                reasons.setdefault(flag.sentence_id, []).append(flag.reason)
            for sentence in changed_sentences:
                found = reasons.get(sentence.id)
                sentence.translation_warning = "；".join(found) if found else None
            checkpoint()
    if not changed_sentences:
        checkpoint()
    if report is not None and report.failed:
        # Every other sentence is already saved; say which ones are missing.
        raise TranslationError(report.failure_message())


@project_operation
def translate_project(
    project: DubProject,
    project_dir: Path,
    api_key: str | None = None,
    force: bool = False,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    overwrite_edited: bool = False,
    sentence_ids: list[str] | None = None,
    compress: bool = False,
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
            overwrite_edited=overwrite_edited,
            sentence_ids=sentence_ids,
            compress=compress,
        )
        metrics["translated_before"] = before
        metrics["translated_after"] = sum(bool(sentence.zh_text) for sentence in project.sentences)


@project_operation
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
        and sentence.chinese_audio_enabled
        and sentence.zh_text
        and (requested is None or sentence.id in requested)
        and (
            (
                has_speakable_text(sentence.zh_text)
                and (force or not valid_tts_cache(project, sentence, project_dir))
            )
            or (
                not has_speakable_text(sentence.zh_text)
                and (sentence.tts_file is not None or sentence.status != "skipped_tts")
            )
        )
        for sentence in project.sentences
    )
    if needs_generation:
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_video_file = None
        if project.settings.subtitle_timeline == "dubbing":
            invalidate_outputs(project, audio=False)

    last_checkpoint = 0.0
    checkpoint_dirty = False

    def checkpoint() -> None:
        nonlocal last_checkpoint, checkpoint_dirty
        checkpoint_dirty = True
        now = time.monotonic()
        if now - last_checkpoint >= 2.0:
            save_project(project, project_dir)
            last_checkpoint = now
            checkpoint_dirty = False

    cancel_kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
    reference_source = source
    if (
        needs_generation
        and project.settings.reference_audio_mode == "fullband_vocals"
        and project.settings.separation_enabled
        and project.source.channels in {1, 2}
    ):
        # Clone from the isolated voice so effects and music never leak into the timbre.
        from .separation import _ensure_separation_impl

        reference_source, _ = _ensure_separation_impl(project, project_dir, source, progress)
    try:
        failures = synthesize_sentences(
            project,
            project_dir,
            reference_source,
            force=force,
            sentence_ids=sentence_ids,
            progress=progress,
            on_sentence=checkpoint,
            **cancel_kwargs,
        )
    finally:
        if checkpoint_dirty:
            save_project(project, project_dir)
            checkpoint_dirty = False
    save_project(project, project_dir)
    export_transcript(project, project_dir)
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
            else sum(
                sentence.enabled
                and sentence.chinese_audio_enabled
                and has_speakable_text(sentence.zh_text)
                for sentence in project.sentences
            )
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


@project_operation
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


def reload_project(path: str | Path) -> tuple[DubProject, Path]:
    require_supported_platform()
    return load_project(path)
