from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from .. import pipeline
from ..errors import ProjectError
from ..languages import SourceLanguage, SpeechSourceLanguage, source_language_label
from ..lifecycle import invalidate_outputs
from ..models import (
    ProjectSettings,
    save_project,
    settings_for_source_language,
)
from ..subtitles import SubtitleLanguage
from ..task_control import CancellationSignal
from ..user_settings import UserSettings, load_user_settings, resolve_api_key
from .project_records import ProjectView, apply_table, view

_ASR_AFFECTING_SETTINGS = frozenset(
    name
    for name in ProjectSettings.model_fields
    if name.startswith(("asr_", "translation_", "separation_"))
    or name in {"pause_split_seconds", "max_sentence_seconds", "skip_japanese_fillers"}
)


def create_project(
    source_media: Any,
    source_language: str = "ja",
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    source = Path(str(source_media or "")).expanduser().resolve()
    if not source.is_file():
        raise ProjectError("请先选择音频或视频。")
    if source_language not in {"ja", "en", "zh"}:
        raise ProjectError("新项目的音频语言必须是日语或英语。")
    defaults = load_user_settings()
    project, directory = pipeline.create_project(
        source,
        projects_root=defaults.projects_root or None,
        settings=defaults.to_project_settings(
            source_language=cast(SpeechSourceLanguage, source_language)
        ),
        source_language=cast(SourceLanguage, source_language),
        progress=progress,
        cancel_event=cancel_event,
    )
    return view(
        project,
        directory,
        f"项目已建立，音频语言为{source_language_label(source_language)}。可以开始识别。",
    )


def save_table(project_path: str, table: Any) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    changed = apply_table(project, table)
    save_project(project, directory)
    pipeline.export_transcript(project, directory)
    message = "句子表格已保存。" if changed else "句子表格没有变化，已确认磁盘版本。"
    return view(project, directory, message)


def import_transcript_data(
    project_path: str,
    transcript_file: Any,
    pasted_text: str,
    plain_timing: str,
    script_kind: str = "source",
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    if script_kind not in {"source", "zh"}:
        raise ProjectError("导入内容必须是原文台本或中文配音稿。")

    script_language = "zh" if script_kind == "zh" else project.source_language
    result = pipeline.import_project_transcript(
        project,
        directory,
        transcript_path=str(transcript_file) if transcript_file else None,
        pasted_text=str(pasted_text or ""),
        plain_timing=str(plain_timing or "estimate"),
        script_language=script_language,
        progress=progress,
        cancel_event=cancel_event,
    )
    chinese_script = result["language"] == "zh"
    if result.get("script_reconciled"):
        stage = "ASR、翻译和台本校对" if chinese_script else "ASR 和台本校对"
        message = (
            f"已完成 {stage}，保留识别得到的时间轴并校正了 {result['sentences']} 句文字。"
            "请抽查句子表后继续。"
        )
    elif result["timed"] and chinese_script:
        message = (
            f"已从 {result['format']} 导入 {result['sentences']} 句中文配音稿和时间轴。"
            "校对后可以直接生成配音。"
        )
    elif result["timed"]:
        message = (
            f"已从 {result['format']} 导入 {result['sentences']} 句原文和时间轴。"
            "检查内容后可以翻译为中文。"
        )
    elif plain_timing == "qwen":
        message = (
            f"已导入 {result['sentences']} 句纯台本，Qwen3 ForcedAligner 成功对齐 "
            f"{result['qwen_aligned_sentences']} 句。请先抽查时间轴。"
        )
    elif chinese_script:
        message = (
            f"已导入 {result['sentences']} 句中文配音稿，并按文字长度生成初始时间轴。"
            "请先校对时间轴，再生成配音。"
        )
    else:
        message = (
            f"已导入 {result['sentences']} 句纯台本，并按文字长度生成初始时间轴。"
            "时间仅供起步，请在表格中校对。"
        )
    return view(project, directory, message)


def analyze(
    project_path: str,
    table: Any,
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    if project.sentences:
        apply_table(project, table)
    pipeline.analyze_project(
        project,
        directory,
        force=bool(project.sentences),
        progress=progress,
        cancel_event=cancel_event,
    )
    return view(project, directory, f"ASR（语音识别）完成，共 {len(project.sentences)} 句。")


def translate(
    project_path: str,
    table: Any,
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    apply_table(project, table)
    key = resolve_api_key(project.settings.translation_provider)
    pipeline.translate_project(
        project,
        directory,
        api_key=key,
        progress=progress,
        cancel_event=cancel_event,
    )
    return view(project, directory, "翻译完成。请检查中文后保存表格。")


def synthesize(
    project_path: str,
    table: Any,
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    apply_table(project, table)
    pipeline.synthesize_project(
        project,
        directory,
        progress=progress,
        cancel_event=cancel_event,
    )
    return view(
        project,
        directory,
        "TTS（语音合成）完成。可以直接混音；之后调整混音设置不需要重做配音。",
    )


def mix(
    project_path: str,
    table: Any,
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    apply_table(project, table)
    pipeline.mix_project(project, directory, progress=progress, cancel_event=cancel_event)
    # Render the state that was actually persisted by the pipeline.  This also
    # makes an immediate result identical to reopening the same project.
    project, directory = pipeline.reload_project(project_path)
    mode_label = {
        "mixed": "混音成品",
        "stem": "中文克隆音轨",
        "both": "混音成品和中文克隆音轨",
    }[project.settings.mix_output_mode]
    return view(project, directory, f"混音完成：{mode_label}。本次没有重新运行 TTS。")


def subtitles(
    project_path: str,
    table: Any,
    language: str,
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    apply_table(project, table)
    if language == "ja":
        language = "source"
    if language not in {"bilingual", "zh", "source"}:
        raise ProjectError("字幕内容必须是双语、仅中文或仅源文。")
    pipeline.generate_subtitles(
        project,
        directory,
        language=cast(SubtitleLanguage, language),
        progress=progress,
        cancel_event=cancel_event,
    )
    count = sum(sentence.enabled for sentence in project.sentences)
    return view(project, directory, f"字幕生成完成：{count} 条有效字幕。")


def apply_global_settings(project_path: str, settings: UserSettings) -> ProjectView:
    settings.validate_mix_dependencies()
    project, directory = pipeline.reload_project(project_path)
    previous = project.settings.model_dump()
    project.settings = settings_for_source_language(
        settings.to_project_settings(
            project.settings,
            source_language=project.source_language,
        ),
        project.source_language,
    )
    current = project.settings.model_dump()
    if previous.get("tts_target_language", "zh") != current["tts_target_language"]:
        # Preserve the old table before changing the meaning of the legacy zh_text column.
        backup = (
            directory
            / f"translations-{previous.get('tts_target_language', 'zh')}-{project.revision}.json"
        )
        from ..storage import atomic_write_text

        atomic_write_text(backup, project.model_dump_json(indent=2))
        for sentence in project.sentences:
            sentence.zh_text = ""
            sentence.tts_file = None
            sentence.tts_cache_key = None
            sentence.tts_duration_seconds = None
            sentence.status = "pending"
        project.translation_language = project.settings.tts_target_language
    changed_fields = sorted(
        name for name in ProjectSettings.model_fields if previous.get(name) != current.get(name)
    )
    asr_changed = bool(_ASR_AFFECTING_SETTINGS.intersection(changed_fields))
    if changed_fields:
        affects_audio = any(
            name.startswith(("tts_", "chinese_", "mix_", "separation_", "spatial_", "replacement_"))
            or name
            in {
                "normalize_chinese_loudness",
                "match_source_loudness",
                "random_seed",
                "reference_padding_seconds",
            }
            for name in changed_fields
        )
        affects_subtitles = affects_audio or any(
            name.startswith("subtitle_") for name in changed_fields
        )
        invalidate_outputs(project, audio=affects_audio, subtitles=affects_subtitles)
    if asr_changed and project.sentences:
        # Keep the user's current table visible until they explicitly rerun
        # recognition, but never present it as matching the new configuration.
        project.asr_settings_dirty = True
    save_project(project, directory)
    review = "开启" if project.settings.asr_review_enabled else "关闭"
    timestamp = (
        "Qwen3 ForcedAligner"
        if project.settings.asr_forced_alignment_enabled
        or project.settings.asr_review_timestamp_priority_model.startswith("qwen_forced_aligner|")
        else "ASR 自带"
    )
    output_mode = {
        "mixed": "仅混音成品",
        "stem": "仅中文克隆音轨",
        "both": "混音成品+中文克隆音轨",
    }[project.settings.mix_output_mode]
    effective = (
        f"项目语言={source_language_label(project.source_language)}；"
        f"ASR={project.settings.asr_backend}；VAD={project.settings.asr_vad_mode}；"
        f"多模型交叉校对={review}；时间戳={timestamp}"
        f"；音频输出={output_mode}"
    )
    if not changed_fields:
        message = f"当前项目设置没有变化。生效配置：{effective}"
    elif asr_changed and project.sentences:
        message = f"设置已应用到当前项目；旧识别结果已标记为待更新。生效配置：{effective}"
    else:
        message = f"设置已应用到当前项目。生效配置：{effective}"
    return view(project, directory, message)
