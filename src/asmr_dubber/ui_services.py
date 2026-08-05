from __future__ import annotations

import os
import shutil
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from . import pipeline
from .audio import extract_reference, verify_source
from .errors import ProjectError
from .languages import SourceLanguage, SpeechSourceLanguage, source_language_label
from .models import (
    DubProject,
    ProjectSettings,
    Sentence,
    save_project,
    settings_for_source_language,
)
from .platforms import portable_home
from .subtitles import SubtitleLanguage
from .task_control import CancellationSignal
from .user_settings import UserSettings, load_user_settings, resolve_api_key
from .voice_reference import shared_reference_sentence

TABLE_HEADERS = [
    "句子 ID",
    "启用中文处理",
    "开始（秒）",
    "结束（秒）",
    "原文",
    "中文译文",
]
TABLE_TYPES = ["str", "bool", "number", "number", "str", "str"]

_ASR_AFFECTING_SETTINGS = frozenset(
    name
    for name in ProjectSettings.model_fields
    if name.startswith(("asr_", "translation_"))
    or name in {"pause_split_seconds", "max_sentence_seconds", "skip_japanese_fillers"}
)


@dataclass(frozen=True)
class ProjectView:
    manifest: str
    source_language: str
    rows: list[list[Any]]
    output_audio: str | None
    stem_audio: str | None
    output_video: str | None
    subtitle_files: list[str]
    subtitle_video: str | None
    diagnostics: str
    status: str


def _table_values(table: Any) -> list[list[Any]]:
    if table is None:
        return []
    if hasattr(table, "values"):
        table = table.values.tolist()
    elif hasattr(table, "to_list"):
        table = table.to_list()
    if not isinstance(table, (list, tuple)):
        raise ProjectError("句子表格格式无效。")
    return [list(row) for row in table]


def _text(value: Any, label: str, *, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ProjectError(f"{label}不能为空。")
    return result


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{label}必须是数字。") from exc
    if not (-1e9 < result < 1e9):
        raise ProjectError(f"{label}超出有效范围。")
    return result


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().casefold()
    if text in {"true", "yes", "1", "是"}:
        return True
    if text in {"false", "no", "0", "否", ""}:
        return False
    raise ProjectError(f"{label}必须是布尔值。")


def project_rows(project: DubProject) -> list[list[Any]]:
    return [
        [
            sentence.id,
            sentence.enabled,
            sentence.start_seconds,
            sentence.end_seconds,
            sentence.source_text,
            sentence.zh_text,
        ]
        for sentence in project.sentences
    ]


def apply_table(project: DubProject, table: Any) -> bool:
    """Apply only editable sentence fields; never import global settings."""

    rows = _table_values(table)
    previous = {sentence.id: sentence for sentence in project.sentences}
    parsed: list[Sentence] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if len(row) != len(TABLE_HEADERS):
            raise ProjectError(f"第 {index} 行列数错误，应为 {len(TABLE_HEADERS)} 列。")
        sentence_id = _text(row[0], f"第 {index} 行句子 ID", required=True)
        source_text = _text(row[4], f"{sentence_id} 源文")
        zh_text = _text(row[5], f"{sentence_id} 中文")
        if not source_text and not zh_text:
            # Clearing the only available text is how a user removes a sentence
            # from Gradio's fixed-column table.  Leave it out of the persisted
            # list so subtitles, TTS and mixing all observe the deletion.
            continue
        if sentence_id in seen:
            raise ProjectError(f"句子 ID 重复：{sentence_id}")
        seen.add(sentence_id)
        start = _number(row[2], f"{sentence_id} 开始时间")
        end = _number(row[3], f"{sentence_id} 结束时间")
        if start < 0 or end <= start or end > project.source.duration_seconds + 0.25:
            raise ProjectError(f"{sentence_id} 的时间范围无效。")
        old = previous.get(sentence_id)
        payload = {
            "id": sentence_id,
            "enabled": _boolean(row[1], f"{sentence_id} 启用状态"),
            "start_seconds": start,
            "end_seconds": end,
            "source_text": source_text,
            "zh_text": zh_text,
        }
        if old is None:
            parsed.append(Sentence(**payload))
            continue
        material_changed = any(
            getattr(old, field) != value
            for field, value in payload.items()
            if field not in {"id", "enabled"}
        )
        updated = old.model_copy(update=payload)
        if material_changed:
            updated.tts_file = None
            updated.tts_cache_key = None
            updated.tts_duration_seconds = None
            updated.reference_file = None
            updated.status = "translated" if updated.zh_text else "pending"
            updated.error = None
        parsed.append(updated)
    parsed.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.id))
    sentence_changed = [item.model_dump() for item in parsed] != [
        item.model_dump() for item in project.sentences
    ]
    remaining_ids = {item.id for item in parsed}
    reference_removed = bool(
        project.settings.tts_reference_sentence_id
        and project.settings.tts_reference_sentence_id not in remaining_ids
    )
    if reference_removed:
        project.settings.tts_reference_sentence_id = None
    changed = sentence_changed or reference_removed
    project.sentences = parsed
    if changed:
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_srt_file = None
        project.subtitle_lrc_file = None
        project.subtitle_video_file = None
    return changed


def ui_stage_directory() -> Path:
    """Return the only local directory exposed to Gradio as an output allowlist."""

    destination = portable_home() / "temp" / "ui"
    destination.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 24 * 3600
    for candidate in destination.rglob("*"):
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
        except OSError:
            pass
    return destination


def stage_for_ui(path: Path | None, *, category: str = "exports") -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    identity = sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[
        :20
    ]
    directory = ui_stage_directory() / category
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{identity}_{path.name}"
    if destination.is_file() and destination.stat().st_size == stat.st_size:
        return str(destination.resolve())
    try:
        os.link(path, destination)
    except FileExistsError:
        pass
    except OSError:
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copy2(path, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination.resolve())


def _project_asset(project_dir: Path, stored: str | None) -> Path | None:
    if not stored:
        return None
    root = project_dir.resolve()
    candidate = (root / stored).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def diagnostics(project: DubProject) -> str:
    counts = Counter(sentence.status for sentence in project.sentences)
    lines = [
        f"项目语言：{source_language_label(project.source_language)}",
        f"总句数：{len(project.sentences)}",
    ]
    if counts:
        lines.append("状态：" + "；".join(f"{key} {value}" for key, value in counts.items()))
    errors = [f"{item.id}：{item.error}" for item in project.sentences if item.error]
    if errors:
        lines.append("最近错误：\n" + "\n".join(errors[:8]))
    if project.migration_warnings:
        lines.append("项目迁移提示：\n" + "\n".join(project.migration_warnings))
    if project.asr_settings_dirty:
        lines.append("ASR（语音识别）设置已改变：当前表格是旧结果，请重新运行 ASR。")
    return "\n".join(lines)


def view(project: DubProject, project_dir: Path, status: str) -> ProjectView:
    subtitle_paths = [
        stage_for_ui(path)
        for path in (
            _project_asset(project_dir, project.subtitle_srt_file),
            _project_asset(project_dir, project.subtitle_lrc_file),
        )
    ]
    return ProjectView(
        manifest=str((project_dir / "project.json").resolve()),
        source_language=project.source_language,
        rows=project_rows(project),
        output_audio=stage_for_ui(_project_asset(project_dir, project.output_file)),
        stem_audio=stage_for_ui(_project_asset(project_dir, project.chinese_stem_file)),
        output_video=stage_for_ui(_project_asset(project_dir, project.output_video_file)),
        subtitle_files=[path for path in subtitle_paths if path],
        subtitle_video=stage_for_ui(_project_asset(project_dir, project.subtitle_video_file)),
        diagnostics=diagnostics(project),
        status=status,
    )


def load_view(project_path: str, status: str = "项目已加载。") -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    return view(project, directory, status)


def create_project(
    source_media: Any,
    source_language: str = "ja",
    progress: Any | None = None,
    cancel_event: CancellationSignal | None = None,
) -> ProjectView:
    source = Path(str(source_media or "")).expanduser().resolve()
    if not source.is_file():
        raise ProjectError("请先选择音频或视频。")
    if source_language not in {"ja", "en"}:
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
    if script_kind == "source" and project.source_language == "zh":
        raise ProjectError("当前项目使用中文配音稿；如需导入原文，请按正确的音频语言新建项目。")
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
    if result["timed"] and chinese_script:
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


def synthesize_and_mix(
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
    pipeline.mix_project(project, directory, progress=progress, cancel_event=cancel_event)
    mode_label = {
        "mixed": "混音成品",
        "stem": "中文克隆音轨",
        "both": "混音成品和中文克隆音轨",
    }[project.settings.mix_output_mode]
    return view(project, directory, f"TTS（语音合成）与输出完成：{mode_label}。")


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
    return view(project, directory, "字幕生成完成。")


def apply_global_settings(project_path: str, settings: UserSettings) -> ProjectView:
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
    changed_fields = sorted(
        name for name in ProjectSettings.model_fields if previous.get(name) != current.get(name)
    )
    asr_changed = bool(_ASR_AFFECTING_SETTINGS.intersection(changed_fields))
    if changed_fields:
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_video_file = None
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


def recent_projects(projects_root: str | None = None) -> list[tuple[str, str]]:
    root = Path(projects_root).expanduser() if projects_root else pipeline.default_projects_dir()
    if not root.is_dir():
        return []
    manifests: list[tuple[float, Path]] = []
    for manifest in root.glob("*/project.json"):
        try:
            manifests.append((manifest.stat().st_mtime, manifest))
        except OSError:
            continue
    manifests.sort(key=lambda item: item[0], reverse=True)
    return [
        (
            f"{manifest.parent.name} · {time.strftime('%Y-%m-%d %H:%M', time.localtime(modified))}",
            str(manifest.resolve()),
        )
        for modified, manifest in manifests[:100]
    ]


def reference_picker(project_path: str) -> tuple[list[tuple[str, str]], str | None, str | None]:
    """Return project sentence choices and a staged preview for the selected anchor."""

    project, directory = pipeline.reload_project(project_path)
    if not project.sentences:
        return [], None, None

    recommended_id: str | None = None
    with suppress(Exception):
        recommended_id = shared_reference_sentence(project).id
    selected = project.settings.tts_reference_sentence_id or recommended_id
    choices: list[tuple[str, str]] = []
    for sentence in project.sentences:
        duration = sentence.end_seconds - sentence.start_seconds
        prefix = "★ 推荐 · " if sentence.id == recommended_id else ""
        warning = "⚠ 过短 · " if duration < 1.5 else ""
        excerpt = " ".join((sentence.source_text or sentence.zh_text).split())[:42] or "（无文本）"
        choices.append(
            (
                f"{prefix}{warning}{sentence.id} · {duration:.1f}s · {excerpt}",
                sentence.id,
            )
        )
    valid_ids = {value for _, value in choices}
    if selected not in valid_ids:
        selected = choices[0][1]
    return choices, selected, reference_preview(project, directory, selected)


def reference_preview(
    project: DubProject,
    directory: Path,
    sentence_id: str | None,
) -> str | None:
    sentence = next((item for item in project.sentences if item.id == sentence_id), None)
    if sentence is None:
        return None
    source = verify_source(directory, project.source)
    preview_dir = ui_stage_directory() / "reference-previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    identity = sha256(
        (
            f"{project.source.sha256}|{sentence.id}|{sentence.start_seconds:.6f}|"
            f"{sentence.end_seconds:.6f}|{project.settings.reference_padding_seconds:.6f}"
        ).encode()
    ).hexdigest()[:20]
    destination = preview_dir / f"{sentence.id}_{identity}.wav"
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.name}.tmp.wav")
        try:
            extract_reference(
                source,
                temporary,
                sentence.start_seconds,
                sentence.end_seconds,
                project.settings.reference_padding_seconds,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination.resolve())


def preview_reference(project_path: str, sentence_id: str | None) -> str | None:
    project, directory = pipeline.reload_project(project_path)
    return reference_preview(project, directory, sentence_id)


def select_reference(project_path: str, sentence_id: str) -> tuple[str, str | None]:
    project, directory = pipeline.reload_project(project_path)
    if not any(item.id == sentence_id for item in project.sentences):
        raise ProjectError(f"项目中找不到参考句：{sentence_id}")
    project.settings.tts_reference_sentence_id = sentence_id
    save_project(project, directory)
    return (
        f"已把 {sentence_id} 设为项目统一音色参考。",
        reference_preview(project, directory, sentence_id),
    )
