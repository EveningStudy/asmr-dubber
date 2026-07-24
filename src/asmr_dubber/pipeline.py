from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .asr import transcribe_japanese
from .asr_review import review_transcriptions
from .audio import (
    build_chinese_stem,
    copy_source_verbatim,
    make_analysis_copy,
    mix_original_and_stem,
    project_file_exists,
    sentence_events,
    verify_source,
)
from .constants import DEFAULT_PROJECTS_DIR
from .errors import ProjectError, SynthesisError
from .filtering import implausible_asr_reason, is_japanese_filler_only
from .models import DubProject, ProjectSettings, Sentence, load_project, save_project
from .performance import measure_stage
from .platforms import require_supported_platform
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


def default_projects_dir() -> Path:
    configured = os.getenv("ASMR_DUBBER_PROJECTS")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_PROJECTS_DIR.resolve()


def create_project(
    source_audio: str | Path,
    projects_root: str | Path | None = None,
    settings: ProjectSettings | None = None,
    project_name: str | None = None,
    progress: Progress | None = None,
) -> tuple[DubProject, Path]:
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
        project = DubProject(source=info, settings=settings or ProjectSettings())
        save_project(project, project_dir)
        export_transcript(project, project_dir)
    return project, project_dir


def _analyze_project_impl(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    progress: Progress | None = None,
) -> None:
    require_supported_platform()
    if project.sentences and not force:
        if progress:
            progress(f"已存在 {len(project.sentences)} 句识别缓存", 1, 1)
        return
    source = verify_source(project_dir, project.source)
    analysis = make_analysis_copy(source, project_dir / "analysis" / "asr_16k_mono.wav")
    sentences, language = transcribe_japanese(analysis, project.settings, progress=progress)
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
                raise ProjectError(f"多 ASR 模型配置无效：{value}")
            pair = (backend.strip(), model.strip())
            if pair not in selected:
                selected.append(pair)
        primary_pair = (project.settings.asr_backend, project.settings.asr_model)
        comparison = [pair for pair in selected if pair != primary_pair]
        total_models = len(comparison) + 1
        for model_index, (backend, model) in enumerate(comparison, start=2):
            if progress:
                progress(
                    f"多 ASR 候选 {model_index}/{total_models}：{backend} · {model}",
                    model_index - 1,
                    total_models,
                )
            candidate_settings = project.settings.model_copy(
                update={
                    "asr_backend": backend,
                    "asr_model": model,
                    "asr_review_enabled": False,
                }
            )
            candidate_sentences, _ = transcribe_japanese(
                analysis,
                candidate_settings,
                progress=progress,
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
        sentences = review_transcriptions(
            transcriptions,
            project.settings,
            project_dir / "analysis" / "asr_review.json",
            progress=progress,
        )
        language = "Japanese (multi-ASR reviewed)"
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
    project.sentences = bounded_sentences
    project.asr_language = language
    project.chinese_stem_file = None
    project.output_file = None
    save_project(project, project_dir)
    export_transcript(project, project_dir)


def analyze_project(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    progress: Progress | None = None,
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
        _analyze_project_impl(project, project_dir, force=force, progress=progress)
        metrics["cache_hit"] = cached
        metrics["sentences"] = len(project.sentences)


def _translate_project_impl(
    project: DubProject,
    project_dir: Path,
    api_key: str | None = None,
    force: bool = False,
    progress: Progress | None = None,
) -> None:
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
        job_id=f"asmr_{project.source.sha256[:24]}",
        progress=progress,
        on_batch=checkpoint,
    )
    checkpoint()


def translate_project(
    project: DubProject,
    project_dir: Path,
    api_key: str | None = None,
    force: bool = False,
    progress: Progress | None = None,
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
        )
        metrics["translated_before"] = before
        metrics["translated_after"] = sum(bool(sentence.zh_text) for sentence in project.sentences)


def _synthesize_project_impl(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    sentence_ids: list[str] | None = None,
    progress: Progress | None = None,
) -> None:
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

    def checkpoint() -> None:
        save_project(project, project_dir)
        export_transcript(project, project_dir)

    failures = synthesize_sentences(
        project,
        project_dir,
        source,
        force=force,
        sentence_ids=sentence_ids,
        progress=progress,
        on_sentence=checkpoint,
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


def synthesize_project(
    project: DubProject,
    project_dir: Path,
    force: bool = False,
    sentence_ids: list[str] | None = None,
    progress: Progress | None = None,
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


def _mix_project_impl(
    project: DubProject,
    project_dir: Path,
    progress: Progress | None = None,
) -> Path:
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
        progress=progress,
    )
    if progress:
        progress("原轨 + 已校准中文轨直接相加（原轨不做任何处理）", 0, 1)
    mix_original_and_stem(source, stem, output, project.source, output_codec="pcm_s24le")
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
    save_project(project, project_dir)
    export_transcript(project, project_dir)
    if progress:
        progress("混音完成", 1, 1)
    return output


def mix_project(
    project: DubProject,
    project_dir: Path,
    progress: Progress | None = None,
) -> Path:
    with measure_stage(
        project_dir,
        "mix",
        audio_seconds=project.source.duration_seconds,
        sample_rate=project.source.sample_rate,
        channels=project.source.channels,
        retained_stem=project.settings.retain_chinese_stem,
    ) as metrics:
        output = _mix_project_impl(project, project_dir, progress=progress)
        metrics["output_bytes"] = output.stat().st_size
        return output


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_transcript(project: DubProject, project_dir: Path) -> None:
    exports = project_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    rows = []
    for sentence in project.sentences:
        rows.append(
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
        )
    (exports / "transcript.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (exports / "transcript.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    srt_blocks = []
    for index, sentence in enumerate(project.sentences, start=1):
        lines = [sentence.ja_text]
        if sentence.zh_text:
            lines.append(sentence.zh_text)
        srt_blocks.append(
            f"{index}\n{_srt_timestamp(sentence.start_seconds)} --> "
            f"{_srt_timestamp(sentence.end_seconds)}\n" + "\n".join(lines)
        )
    (exports / "bilingual.srt").write_text(
        "\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""),
        encoding="utf-8",
    )


def reload_project(path: str | Path) -> tuple[DubProject, Path]:
    require_supported_platform()
    return load_project(path)
