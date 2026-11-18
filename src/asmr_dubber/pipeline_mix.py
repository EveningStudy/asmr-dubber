"""Chinese voice stem construction, original/replacement mixing, and video muxing."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from .errors import ProjectError
from .lifecycle import project_operation
from .models import DubProject, Sentence
from .task_control import CancellationSignal

Progress = Callable[[str, int, int], None]


def _sentence_needs_dubbing(sentence: Sentence) -> bool:
    from . import pipeline

    if not sentence.enabled or not sentence.chinese_audio_enabled:
        return False
    if not sentence.zh_text:
        return sentence.status not in {"skipped_sound", "skipped_filler", "skipped_tts"}
    return pipeline.has_speakable_text(sentence.zh_text)


def _mix_project_impl(
    project: DubProject,
    project_dir: Path,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    output_variant: str | None = None,
    allow_missing: bool = False,
) -> Path:
    from . import pipeline

    pipeline.check_cancelled(cancel_event)
    pipeline.require_supported_platform()
    source = pipeline.verify_source(project_dir, project.source)
    replacing = project.settings.separation_mix_mode == "replace"
    effective_rtf = project.settings.spatial_rtf_enabled
    if effective_rtf and project.source.channels != 2:
        pipeline.logger.warning(
            "原音频为 %d 声道，已自动跳过双声道空间跟随（RTF）处理。",
            project.source.channels,
        )
        effective_rtf = False
    missing = [
        sentence.id
        for sentence in project.sentences
        if sentence.enabled
        and sentence.chinese_audio_enabled
        and sentence.zh_text
        and pipeline.has_speakable_text(sentence.zh_text)
        and (
            not sentence.tts_file
            or sentence.tts_cache_key != pipeline.tts_cache_key(project, sentence)
            or not pipeline.project_file_exists(
                project_dir,
                sentence.tts_file,
                f"句子 {sentence.id} 的中文音频",
            )
        )
    ]
    effective_sentences = project.sentences
    if missing and allow_missing:
        # Sentences without a usable dub are treated as "not dubbed": the original
        # voice stays in the bilingual and the fully dubbed version alike.  Use an
        # in-memory copy so the user's persisted tts_file references are preserved.
        lost = set(missing)
        effective_sentences = [
            sentence.model_copy(update={"tts_file": None}) if sentence.id in lost else sentence
            for sentence in project.sentences
        ]
        missing = []
    if missing:
        raise ProjectError(
            f"仍有 {len(missing)} 句的中文配音缺失或已过期：{', '.join(missing[:12])}"
        )
    mixing_settings = (
        project.settings
        if effective_rtf == project.settings.spatial_rtf_enabled
        else project.settings.model_copy(update={"spatial_rtf_enabled": effective_rtf})
    )
    mixing_project = (
        project
        if effective_sentences is project.sentences and mixing_settings is project.settings
        else project.model_copy(
            update={"sentences": effective_sentences, "settings": mixing_settings}
        )
    )
    mixing_source = source
    reference_source = source
    separated_vocals: Path | None = None
    if project.settings.separation_enabled:
        from .experimental_mix import compose_replacement_bed
        from .separation import ensure_separation

        vocals, background = ensure_separation(project, project_dir, source, progress)
        separated_vocals = vocals
        if replacing:
            reference_source = vocals
        if replacing and project.settings.replacement_scope == "dubbed_speech":
            from .experimental_mix import compose_speech_replacement

            mixing_source = compose_speech_replacement(mixing_project, project_dir, source, vocals)
        elif replacing or any(
            s.original_audio_enabled is not None or s.original_audio_gain_db != 0
            for s in effective_sentences
        ):
            mixing_source = compose_replacement_bed(mixing_project, project_dir, vocals, background)
    elif project.settings.separation_mix_mode == "replace":
        raise ProjectError("中文替换模式必须开启人声分离；不会静默改成原声叠加。")
    events = pipeline.sentence_events(
        project_dir,
        effective_sentences,
        (
            project.settings.replacement_dubbing_offset_ms
            if replacing
            else project.settings.chinese_dubbing_offset_ms
        ),
        (
            project.settings.replacement_max_auto_speed
            if replacing
            else project.settings.chinese_max_auto_speed
        ),
        (
            project.settings.replacement_timing_mode
            if replacing
            else project.settings.chinese_dubbing_timing_mode
        ),
        trim_silence=replacing,
    )
    enabled_sentences = [s for s in project.sentences if s.enabled]
    if not events and (
        not enabled_sentences or any(pipeline._sentence_needs_dubbing(s) for s in enabled_sentences)
    ):
        raise ProjectError("没有可混入的中文配音。")
    output_mode = project.settings.mix_output_mode
    keep_stem = output_mode in {"stem", "both"}
    from .stem_cache import build_cached_stem, variant_directory, write_compare_state

    output_dir = variant_directory(project_dir, output_variant)
    stem = (
        output_dir / pipeline.stem_output_filename(project, project_dir)
        if keep_stem
        else project_dir
        / "mix"
        / ("replace" if replacing else "bilingual")
        / "chinese_stem_float32.wav"
    )
    output = output_dir / pipeline.output_filename(project, project_dir)
    if replacing:
        if separated_vocals is not None and separated_vocals.is_file():
            # Full-bandwidth stereo vocals preserve high-frequency whisper energy and
            # equal-power single-ear levels without an extra 16 kHz FFmpeg downmix pass.
            loudness_reference = separated_vocals
        else:
            from .audio import sha256_file

            reference_key = sha256_file(reference_source)
            receipt = project_dir / "analysis" / "replacement_loudness_source.sha256"
            reference_audio = project_dir / "analysis" / "replacement_loudness_16k_mono.wav"
            if not receipt.is_file() or receipt.read_text(encoding="utf-8") != reference_key:
                reference_audio.unlink(missing_ok=True)
            loudness_reference = pipeline.make_analysis_copy(reference_source, reference_audio)
            pipeline.atomic_write_text(receipt, reference_key)
    else:
        loudness_reference = pipeline.make_analysis_copy(
            source, project_dir / "analysis" / "asr_16k_mono.wav"
        )
    if progress:
        progress("按对应源语言句子校准中文响度（不移动或修改原音频）", 0, max(1, len(events)))
    spatial_kwargs = {}
    if effective_rtf:
        from .experimental_mix import stereo_reference

        if separated_vocals is not None and project.settings.spatial_rtf_vocal_weighting:
            from .experimental_mix import vocal_weighted_reference

            spatial_reference = vocal_weighted_reference(source, separated_vocals, project_dir)
        else:
            spatial_reference = stereo_reference(source, project_dir)
        spatial_kwargs = {
            "spatial_reference_path": spatial_reference,
            "spatial_settings": mixing_settings,
        }
    stem_builder = (
        (lambda **kwargs: build_cached_stem(pipeline.build_chinese_stem, **kwargs))
        if effective_rtf
        else pipeline.build_chinese_stem
    )
    loudness_prefix = "replacement_" if replacing else ""

    def loudness(name):
        return getattr(project.settings, loudness_prefix + name)

    match_kwargs = {}
    if replacing and separated_vocals is not None:
        match_kwargs = {
            "voice_match_path": str(separated_vocals),
            "voice_match_strength": project.settings.replacement_voice_match_strength,
            "bandwidth_extension": project.settings.replacement_bandwidth_extension,
            "whisper_match": project.settings.replacement_whisper_match,
        }
    spatial_kwargs = dict(spatial_kwargs)
    if "spatial_settings" in spatial_kwargs:
        spatial_kwargs["spatial_settings"] = mixing_settings.model_copy(
            update={"chinese_line_peak_dbfs": loudness("chinese_line_peak_dbfs")}
        )
    stem_builder(
        destination=stem,
        events=events,
        source_info=project.source,
        chinese_gain_db=loudness("chinese_gain_db"),
        normalize_loudness=loudness("normalize_chinese_loudness"),
        source_reference_path=loudness_reference,
        match_source_loudness=loudness("match_source_loudness"),
        relative_loudness_db=loudness("chinese_relative_loudness_db"),
        minimum_active_rms_dbfs=loudness("chinese_min_active_rms_dbfs"),
        target_active_rms_dbfs=loudness("chinese_target_active_rms_dbfs"),
        max_loudness_boost_db=loudness("chinese_max_loudness_boost_db"),
        line_peak_dbfs=loudness("chinese_line_peak_dbfs"),
        stem_peak_dbfs=loudness("chinese_stem_peak_dbfs"),
        fade_ms=loudness("chinese_fade_ms"),
        channel_routing=project.settings.chinese_channel_routing,
        progress=progress,
        **spatial_kwargs,
        **match_kwargs,
    )
    pipeline.check_cancelled(cancel_event)
    mixed_output: Path | None = None
    if output_mode in {"mixed", "both"}:
        if progress:
            progress("正在把中文克隆音轨加入原音轨，并执行最终峰值保护", 0, 1)
        pipeline.mix_original_and_stem(
            mixing_source,
            stem,
            output,
            project.source,
            output_codec="pcm_s24le",
            peak_protection=project.settings.mix_peak_protection,
            peak_limit_dbfs=project.settings.mix_peak_limit_dbfs,
        )
        mixed_output = output
        if output_variant is None:
            variant_copy = (
                project_dir
                / "output"
                / ("replace" if replacing else "bilingual")
                / mixed_output.name
            )
            variant_copy.parent.mkdir(parents=True, exist_ok=True)
            variant_copy.unlink(missing_ok=True)
            try:
                os.link(mixed_output, variant_copy)
            except OSError:
                shutil.copyfile(mixed_output, variant_copy)
            write_compare_state(
                project,
                project_dir,
                "replacement" if replacing else "bilingual",
                mixed_output,
                variant_copy,
            )
        else:
            write_compare_state(
                project,
                project_dir,
                "replacement" if output_variant == "replace" else "bilingual",
                mixed_output,
            )
        pipeline.check_cancelled(cancel_event)
    if keep_stem:
        project.chinese_stem_file = str(stem.relative_to(project_dir))
    elif effective_rtf:
        # Each mix keeps a separate validated intermediate for its own loudness settings.
        project.chinese_stem_file = None
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
    pipeline.save_project(project, project_dir)
    if project.source.media_type == "video" and mixed_output is not None:
        if progress:
            progress("保留原画面并封装中文混合音轨", 0, 1)
        video_output = pipeline.mux_mixed_video(
            source,
            mixed_output,
            output_dir / pipeline.output_video_filename(project, project_dir),
        )
        project.output_video_file = video_output.relative_to(project_dir).as_posix()
    pipeline.save_project(project, project_dir)
    pipeline.export_transcript(project, project_dir)
    if progress:
        progress("中文配音输出完成", 1, 1)
    return mixed_output or stem


@project_operation
def mix_project(
    project: DubProject,
    project_dir: Path,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    output_variant: str | None = None,
    allow_missing: bool = False,
) -> Path:
    from . import pipeline

    with pipeline.measure_stage(
        project_dir,
        "mix",
        audio_seconds=project.source.duration_seconds,
        sample_rate=project.source.sample_rate,
        channels=project.source.channels,
        output_mode=project.settings.mix_output_mode,
    ) as metrics:
        output = pipeline._mix_project_impl(
            project,
            project_dir,
            progress=progress,
            cancel_event=cancel_event,
            output_variant=output_variant,
            allow_missing=allow_missing,
        )
        metrics["output_bytes"] = output.stat().st_size
        if project.chinese_stem_file:
            stem_output = pipeline.resolve_project_path(
                project_dir,
                project.chinese_stem_file,
                "中文克隆音轨",
            )
            if stem_output.is_file():
                metrics["stem_output_bytes"] = stem_output.stat().st_size
        if project.output_video_file:
            video_output = pipeline.resolve_project_path(
                project_dir,
                project.output_video_file,
                "混音视频",
            )
            if video_output.is_file():
                metrics["video_output_bytes"] = video_output.stat().st_size
        return output
