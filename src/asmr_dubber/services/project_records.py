from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import pipeline
from ..errors import ProjectError
from ..languages import source_language_label
from ..models import (
    DubProject,
    Sentence,
)
from ..platforms import open_directory
from .project_audio import stage_for_ui

TABLE_HEADERS = [
    "句子 ID",
    "启用中文处理",
    "开始（秒）",
    "结束（秒）",
    "原文",
    "中文译文",
    "原声开关（仅分离时）",
    "原声音量微调（dB）",
    "播放中文配音",
    "中文音量微调（dB）",
]


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
    revision: int = 0
    media_duration: float = 0.0


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
            "default"
            if sentence.original_audio_enabled is None
            else "on"
            if sentence.original_audio_enabled
            else "off",
            sentence.original_audio_gain_db,
            sentence.chinese_audio_enabled,
            sentence.chinese_audio_gain_db,
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
        if len(row) not in {6, len(TABLE_HEADERS)}:
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
        if len(row) == len(TABLE_HEADERS):
            source_switch = str(row[6] or "default").strip().lower()
            if source_switch not in {"default", "on", "off"}:
                raise ProjectError(f"{sentence_id} 原声开关必须为 default/on/off。")
            source_gain = _number(row[7], f"{sentence_id} 原声音量")
            chinese_gain = _number(row[9], f"{sentence_id} 中文音量")
            if not -60 <= source_gain <= 12 or not -60 <= chinese_gain <= 12:
                raise ProjectError("逐句音量微调必须在 -60 到 +12 dB 之间。")
            payload.update(
                original_audio_enabled={"default": None, "on": True, "off": False}[source_switch],
                original_audio_gain_db=source_gain,
                chinese_audio_enabled=_boolean(row[8], f"{sentence_id} 中文播放"),
                chinese_audio_gain_db=chinese_gain,
            )
        if old is None:
            parsed.append(Sentence(**payload))
            continue
        material_changed = any(
            getattr(old, field) != value
            for field, value in payload.items()
            if field in {"start_seconds", "end_seconds", "source_text", "zh_text"}
        )
        updated = old.model_copy(update=payload)
        if material_changed:
            updated.review_locked = True
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
    review = [
        f"{item.id}：{item.script_review_note}"
        for item in project.sentences
        if item.script_review_note
    ]
    if review:
        lines.append(f"台本待人工校对（{len(review)} 句）：\n" + "\n".join(review))
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
        output_audio=stage_for_ui(
            _project_asset(project_dir, project.output_file), preserve_name=False
        ),
        stem_audio=stage_for_ui(
            _project_asset(project_dir, project.chinese_stem_file), preserve_name=False
        ),
        output_video=stage_for_ui(
            _project_asset(project_dir, project.output_video_file), preserve_name=False
        ),
        subtitle_files=[path for path in subtitle_paths if path],
        subtitle_video=stage_for_ui(
            _project_asset(project_dir, project.subtitle_video_file), preserve_name=False
        ),
        diagnostics=diagnostics(project),
        status=status,
        revision=project.revision,
        media_duration=project.source.duration_seconds,
    )


def load_view(project_path: str, status: str = "项目已加载。") -> ProjectView:
    project, directory = pipeline.reload_project(project_path)
    return view(project, directory, status)


def open_project_directory(project_path: str) -> str:
    if not str(project_path or "").strip():
        raise ProjectError("请先新建或打开项目。")
    _project, directory = pipeline.reload_project(project_path)
    opened = open_directory(directory)
    return f"已在文件管理器中打开项目目录：{opened}"


def open_project_output_directory(project_path: str) -> str:
    if not str(project_path or "").strip():
        raise ProjectError("请先新建或打开项目。")
    _project, directory = pipeline.reload_project(project_path)
    output_directory = directory / "output"
    output_directory.mkdir(parents=True, exist_ok=True)
    opened = open_directory(output_directory)
    return f"已在文件管理器中打开输出文件夹：{opened}"


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
