from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import pipeline
from .audio import extract_reference, verify_source
from .constants import (
    INDEXTTS_REQUIRED_DIRS as _INDEXTTS_REQUIRED_DIRS,
)
from .constants import (
    INDEXTTS_REQUIRED_FILES as _INDEXTTS_REQUIRED_FILES,
)
from .errors import AsmrDubberError, ProjectError, SynthesisError
from .filtering import is_japanese_filler_only
from .model_registry import ASR_BACKENDS, CLONE_MODE_LABELS, TTS_BACKENDS
from .models import DubProject, Sentence, save_project
from .platforms import portable_home, require_supported_platform, runtime_executable_candidates
from .runtime_manager import (
    available_asr_review_choices,
    available_backend_models_markdown,
    backend_catalog_rows,
    hardware_markdown,
    install_backend,
    installable_backend_ids,
    recommended_stack_markdown,
    refresh_hardware,
)
from .tts import shared_reference_sentence
from .user_settings import (
    PROVIDER_PRESETS,
    UserSettings,
    api_key_status,
    clear_api_key,
    clear_service_key,
    load_user_settings,
    resolve_api_key,
    save_api_key,
    save_service_key,
    save_user_settings,
    service_key_status,
    store_reference_audio,
)

TABLE_HEADERS = [
    "id",
    "启用",
    "start",
    "end",
    "日文",
    "中文",
    "本句秒数上限（留空 = 全局）",
    "状态",
    "错误",
]
TABLE_TYPES = [
    "str",
    "bool",
    "number",
    "number",
    "str",
    "str",
    "number",
    "str",
    "str",
]
_ALLOWED_STATUSES = {
    "pending",
    "translated",
    "synthesized",
    "skipped_filler",
    "skipped_hallucination",
    "error",
}
_SAFE_SENTENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

UIResult = tuple[str, list[list[Any]], str | None, str]
TTS_BACKEND_ORDER = (
    "indextts2",
    "voxcpm2",
    "qwen3_tts",
    "gpt_sovits",
    "cosyvoice",
    "f5_tts",
    "fish_speech",
    "xtts_v2",
)

APP_CSS = """
.asmr-brand { margin: 0 0 1rem; }
.asmr-brand__title {
    margin: 0;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 2rem;
    line-height: 1.2;
    font-weight: 750;
    letter-spacing: -0.025em;
}
.asmr-brand__subtitle {
    margin: .35rem 0 0;
    color: var(--body-text-color-subdued);
}
"""


class _StageProgress:
    """Map a pipeline stage's local counter onto one Gradio progress bar."""

    def __init__(
        self,
        progress: Callable[..., Any] | None,
        stage_index: int,
        stage_count: int,
    ) -> None:
        self.progress = progress
        self.stage_index = stage_index
        self.stage_count = max(1, stage_count)

    def __call__(self, message: str, current: int, total: int) -> None:
        if self.progress is None:
            return
        local = 0.0 if total <= 0 else min(1.0, max(0.0, current / total))
        overall = (self.stage_index + local) / self.stage_count
        # A tuple supplies an explicit denominator.  Gradio 6 otherwise renders
        # some long-running callbacks as an unhelpful 0/0 counter.
        self.progress((round(overall * 100), 100), desc=message)


def _progress_message(
    progress: Callable[..., Any] | None,
    fraction: float,
    message: str,
) -> None:
    if progress is not None:
        bounded = min(1.0, max(0.0, fraction))
        progress((round(bounded * 100), 100), desc=message)


def _uploaded_path(value: Any) -> Path:
    if value is None:
        raise ProjectError("请先选择日语音频。")
    candidate: Any = value
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
    elif not isinstance(value, (str, os.PathLike)):
        candidate = getattr(value, "path", None) or getattr(value, "name", None)
    if not candidate:
        raise ProjectError("无法取得上传音频的本地路径。")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise ProjectError(f"找不到输入音频：{path}")
    return path


def _projects_root(value: Any) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    root = Path(text).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ProjectError(f"项目输出位置不是目录：{root}")
    return root


def _finite_number(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ProjectError(f"{label}必须是有限数字。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{label}必须是有限数字。") from exc
    if not math.isfinite(number):
        raise ProjectError(f"{label}必须是有限数字。")
    return number


def indextts_installation_status(model_path: Any) -> str:
    text = str(model_path or "").strip()
    if not text:
        return "未填写 IndexTTS2 checkpoints 目录。"
    directory = Path(text).expanduser().resolve()
    executable = next(
        (
            candidate
            for candidate in runtime_executable_candidates(directory.parent, "indextts2")
            if candidate.is_file()
        ),
        None,
    )
    missing = sorted(
        [name for name in _INDEXTTS_REQUIRED_FILES if not (directory / name).is_file()]
        + [name + "/" for name in _INDEXTTS_REQUIRED_DIRS if not (directory / name).is_dir()]
    )
    if executable is None:
        expected = " 或 ".join(
            str(path) for path in runtime_executable_candidates(directory.parent, "indextts2")
        )
        return f"运行环境未安装：找不到 {expected}"
    if missing:
        preview = "、".join(missing[:5])
        suffix = f" 等 {len(missing)} 项" if len(missing) > 5 else ""
        return f"运行环境已安装，但模型不完整：缺少 {preview}{suffix}。"
    return f"IndexTTS2 已就绪：{executable}；模型目录：{directory}"


def _cache_component_defaults(app: Any, updates: Mapping[Any, Any]) -> None:
    """Update the config served to newly opened/refreshed browser sessions."""

    config = deepcopy(app.config)
    component_configs = {
        item.get("id"): item for item in config.get("components", []) if isinstance(item, dict)
    }

    def serialized_value(component: Any, value: Any) -> Any:
        try:
            processed = component.postprocess(value)
        except (AttributeError, TypeError, ValueError):
            processed = value
        if hasattr(processed, "model_dump"):
            return processed.model_dump(mode="json")
        return processed

    for component, update in updates.items():
        component_config = component_configs.get(getattr(component, "_id", None))
        if not component_config:
            continue
        props = component_config.setdefault("props", {})
        if isinstance(update, dict) and update.get("__type__") == "update":
            for key, value in update.items():
                if key == "__type__":
                    continue
                if key == "value":
                    props[key] = serialized_value(component, value)
                elif key == "choices":
                    props[key] = [
                        list(choice)
                        if isinstance(choice, (list, tuple)) and len(choice) == 2
                        else [choice, choice]
                        for choice in value
                    ]
                else:
                    props[key] = value
        else:
            props["value"] = serialized_value(component, update)
    app.config = config


def _project_rows(project: DubProject) -> list[list[Any]]:
    return [
        [
            sentence.id,
            sentence.enabled,
            sentence.start_seconds,
            sentence.end_seconds,
            sentence.ja_text,
            sentence.zh_text,
            sentence.overlap_seconds,
            sentence.status,
            sentence.error or "",
        ]
        for sentence in project.sentences
    ]


def _reference_choice_label(sentence: Sentence) -> str:
    duration = sentence.end_seconds - sentence.start_seconds
    content_length = sum(
        char.isalnum() or "\u3040" <= char <= "\u30ff" for char in sentence.ja_text
    )
    if is_japanese_filler_only(sentence.ja_text):
        quality = "⚠ 纯语气词"
    elif duration < 1.5:
        quality = "⚠ 过短"
    elif 4.0 <= duration <= 15.0 and content_length >= 8:
        quality = "★ 推荐范围"
    else:
        quality = "可试听"
    ja_text = sentence.ja_text.replace("\n", " ").strip()
    zh_text = sentence.zh_text.replace("\n", " ").strip()
    if len(ja_text) > 34:
        ja_text = ja_text[:33] + "…"
    if len(zh_text) > 30:
        zh_text = zh_text[:29] + "…"
    translation = f" · 中：{zh_text}" if zh_text else ""
    return f"{sentence.id} · {duration:.2f}s · {quality} · 日：{ja_text}{translation}"


def reference_preview_path(project_path: Any, sentence_id: Any) -> str | None:
    path_text = "" if project_path is None else str(project_path).strip()
    selected_id = "" if sentence_id is None else str(sentence_id).strip()
    if not path_text or not selected_id:
        return None
    project, directory = pipeline.reload_project(path_text)
    sentence = next((item for item in project.sentences if item.id == selected_id), None)
    if sentence is None:
        raise ProjectError(f"项目中找不到参考句：{selected_id}")
    source = verify_source(directory, project.source)
    fingerprint = hashlib.sha256(
        (
            f"{directory.resolve()}|{project.source.sha256}|{sentence.id}|"
            f"{sentence.start_seconds:.6f}|{sentence.end_seconds:.6f}"
        ).encode()
    ).hexdigest()[:16]
    # Gradio snapshots allowed_paths when the server starts. A user can change
    # the project output directory later without restarting, so returning a
    # preview from that newly selected directory is rejected by Gradio. UI
    # previews are disposable cache files; keep them in the always-allowed
    # portable temp tree while the actual project remains where the user chose.
    preview = portable_home() / "temp" / "reference-previews" / f"{sentence.id}_{fingerprint}.wav"
    if not preview.is_file():
        temporary = preview.with_name(f".{preview.stem}.tmp.wav")
        try:
            extract_reference(
                source=source,
                destination=temporary,
                start_seconds=sentence.start_seconds,
                end_seconds=sentence.end_seconds,
                padding_seconds=0.0,
            )
            temporary.replace(preview)
        finally:
            temporary.unlink(missing_ok=True)
    return str(preview.resolve())


def reference_picker_data(
    project_path: Any,
) -> tuple[list[tuple[str, str]], str | None, str | None]:
    path_text = "" if project_path is None else str(project_path).strip()
    if not path_text:
        return [], None, None
    project, _ = pipeline.reload_project(path_text)
    choices = [(_reference_choice_label(sentence), sentence.id) for sentence in project.sentences]
    ids = {sentence.id for sentence in project.sentences}
    selected = project.settings.tts_reference_sentence_id
    if selected not in ids:
        try:
            selected = shared_reference_sentence(project).id
        except SynthesisError:
            selected = project.sentences[0].id if project.sentences else None
    preview = reference_preview_path(path_text, selected) if selected else None
    return choices, selected, preview


def save_reference_sentence(project_path: Any, sentence_id: Any) -> tuple[str, str]:
    path_text = "" if project_path is None else str(project_path).strip()
    selected_id = "" if sentence_id is None else str(sentence_id).strip()
    if not path_text:
        raise ProjectError("请先新建或加载项目。")
    if not selected_id:
        raise ProjectError("请先选择并试听一个统一声纹参考句。")
    project, directory = pipeline.reload_project(path_text)
    sentence = next((item for item in project.sentences if item.id == selected_id), None)
    if sentence is None:
        raise ProjectError(f"项目中找不到参考句：{selected_id}")
    previous = project.settings.tts_reference_sentence_id
    project.settings.tts_reference_sentence_id = selected_id
    if previous != selected_id:
        project.chinese_stem_file = None
        project.output_file = None
    save_project(project, directory)
    pipeline.export_transcript(project, directory)
    preview = reference_preview_path(path_text, selected_id)
    duration = sentence.end_seconds - sentence.start_seconds
    status = (
        f"已将 {selected_id}（{duration:.2f} 秒）保存为统一声纹参考。"
        "下次执行“逐句克隆 + 混音”时，所有中文会使用该参考。"
    )
    return status, preview or ""


def _table_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "to_numpy"):
        value = value.to_numpy().tolist()
    elif hasattr(value, "rows") and callable(value.rows):
        value = value.rows()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProjectError("句子表格格式无效。")
    rows: list[list[Any]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ProjectError("句子表格包含无效行。")
        rows.append(list(row))
    return rows


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _text(value: Any, label: str, *, required: bool) -> str:
    if _blank(value):
        if required:
            raise ProjectError(f"{label}不能为空。")
        return ""
    if not isinstance(value, str):
        raise ProjectError(f"{label}必须是文本。")
    result = value.strip()
    if required and not result:
        raise ProjectError(f"{label}不能为空。")
    return result


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if type(value).__name__ == "bool_":  # numpy/pandas scalar without importing either here.
        return bool(value)
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "是"}:
            return True
        if normalized in {"false", "0", "否"}:
            return False
    raise ProjectError(f"{label}必须是布尔值（true/false）。")


def _optional_overlap(value: Any, label: str) -> float | None:
    if _blank(value):
        return None
    result = _finite_number(value, label)
    if result < -30.0 or result > 30.0:
        raise ProjectError(f"{label}必须在 -30 到 30 秒之间。")
    return result


def _parse_sentences(project: DubProject, table: Any) -> tuple[list[Sentence], int]:
    rows = _table_rows(table)
    if not rows:
        raise ProjectError("句子表格不能为空。")

    previous_by_id = {sentence.id: sentence for sentence in project.sentences}
    parsed: list[Sentence] = []
    ids: set[str] = set()
    stale_count = 0
    previous_start = -1.0

    for row_index, row in enumerate(rows, start=1):
        prefix = f"第 {row_index} 行"
        if len(row) != len(TABLE_HEADERS):
            raise ProjectError(
                f"{prefix}必须恰好有 {len(TABLE_HEADERS)} 列，实际为 {len(row)} 列。"
            )
        sentence_id = _text(row[0], f"{prefix} id", required=True)
        if not _SAFE_SENTENCE_ID.fullmatch(sentence_id):
            raise ProjectError(
                f"{prefix} id 不安全；仅允许 1–64 位英文字母、数字、点、下划线和连字符。"
            )
        if sentence_id in ids:
            raise ProjectError(f"{prefix} id 重复：{sentence_id}")
        ids.add(sentence_id)

        enabled = _boolean(row[1], f"{prefix} 启用")
        start = _finite_number(row[2], f"{prefix} start")
        end = _finite_number(row[3], f"{prefix} end")
        if start < 0:
            raise ProjectError(f"{prefix} start 不能小于 0。")
        if end <= start:
            raise ProjectError(f"{prefix} end 必须大于 start。")
        if end > project.source.duration_seconds + 1e-6:
            raise ProjectError(
                f"{prefix} end={end:.6f} 超过源音频时长 {project.source.duration_seconds:.6f} 秒。"
            )
        if start + 1e-9 < previous_start:
            raise ProjectError(f"{prefix} start 小于上一行；请按时间顺序排列句子。")
        previous_start = start

        ja_text = _text(row[4], f"{prefix} 日文", required=True)
        zh_text = _text(row[5], f"{prefix} 中文", required=False)
        overlap = _optional_overlap(row[6], f"{prefix} 本句秒数上限")
        status = _text(row[7], f"{prefix} 状态", required=True).lower()
        if status not in _ALLOWED_STATUSES:
            choices = ", ".join(sorted(_ALLOWED_STATUSES))
            raise ProjectError(f"{prefix} 状态只能是：{choices}。")
        error = _text(row[8], f"{prefix} 错误", required=False) or None

        old = previous_by_id.get(sentence_id)
        payload: dict[str, Any] = {
            "id": sentence_id,
            "enabled": enabled,
            "start_seconds": start,
            "end_seconds": end,
            "ja_text": ja_text,
            "zh_text": zh_text,
            "overlap_seconds": overlap,
            "status": status,
            "error": error,
        }
        if old is not None:
            payload.update(
                reference_file=old.reference_file,
                tts_file=old.tts_file,
                tts_duration_seconds=old.tts_duration_seconds,
                tts_cache_key=old.tts_cache_key,
            )

        clone_inputs_changed = old is None or (
            old.start_seconds != start
            or old.end_seconds != end
            or old.ja_text != ja_text
            or old.zh_text != zh_text
        )
        if clone_inputs_changed:
            stale_count += 1
            # Keep the old cache key/file. synthesize_sentences compares that key with
            # the newly computed key and regenerates only this sentence.
            payload["status"] = "translated" if zh_text else "pending"
            payload["error"] = None
        try:
            parsed.append(Sentence.model_validate(payload))
        except ValidationError as exc:
            raise ProjectError(f"{prefix} 校验失败：{exc}") from exc

    return parsed, stale_count


def _material_signature(sentences: list[Sentence]) -> list[tuple[Any, ...]]:
    return [
        (
            item.id,
            item.enabled,
            item.start_seconds,
            item.end_seconds,
            item.ja_text,
            item.zh_text,
            item.overlap_seconds,
        )
        for item in sentences
    ]


def _apply_table_and_settings(
    project: DubProject,
    table: Any,
) -> tuple[int, bool]:
    old_signature = _material_signature(project.sentences)
    old_settings = project.settings.model_dump()
    sentences, stale_count = _parse_sentences(project, table)
    settings = load_user_settings().to_project_settings(project.settings)
    material_changed = (
        old_signature != _material_signature(sentences) or old_settings != settings.model_dump()
    )
    project.sentences = sentences
    project.settings = settings
    if material_changed:
        project.chinese_stem_file = None
        project.output_file = None
    return stale_count, material_changed


def _manifest(project_dir: Path) -> str:
    return str((project_dir / "project.json").resolve())


def _output_audio(project: DubProject, project_dir: Path) -> str | None:
    if not project.output_file:
        return None
    root = project_dir.resolve()
    candidate = (root / project.output_file).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return str(candidate)


def _result(project: DubProject, project_dir: Path, status: str) -> UIResult:
    return (
        _manifest(project_dir),
        _project_rows(project),
        _output_audio(project, project_dir),
        status,
    )


def _partial_result(project_dir: Path, status: str) -> UIResult:
    project, directory = pipeline.reload_project(project_dir)
    return _result(project, directory, status)


def new_analyze_translate(
    source_audio: Any,
    progress: Callable[..., Any] | None = None,
) -> UIResult:
    """Create, transcribe and translate, stopping before voice cloning."""
    source = _uploaded_path(source_audio)
    user_settings = load_user_settings()
    key = resolve_api_key(user_settings.translation_provider)
    settings = user_settings.to_project_settings()
    _progress_message(progress, 0.0, "保存源音频并建立项目")
    project, directory = pipeline.create_project(
        source,
        projects_root=_projects_root(user_settings.projects_root),
        settings=settings,
        progress=_StageProgress(progress, 0, 3),
    )
    try:
        pipeline.analyze_project(
            project,
            directory,
            progress=_StageProgress(progress, 1, 3),
        )
        pipeline.translate_project(
            project,
            directory,
            api_key=key,
            progress=_StageProgress(progress, 2, 3),
        )
    except AsmrDubberError as exc:
        return _partial_result(directory, f"识别/翻译未完成：{exc}\n项目已保留，可稍后继续。")
    except Exception as exc:  # Model/runtime errors should not hide the resumable project.
        return _partial_result(
            directory,
            f"识别/翻译意外中断（{type(exc).__name__}）：{exc}\n项目已保留，可稍后继续。",
        )
    _progress_message(progress, 1.0, "识别与翻译完成")
    return _result(
        project,
        directory,
        f"识别与翻译完成，共 {len(project.sentences)} 句。请检查并保存表格。",
    )


def load_existing_project(project_path: Any) -> UIResult:
    text = "" if project_path is None else str(project_path).strip()
    if not text:
        raise ProjectError("请填写已有 project.json 或项目目录的本地路径。")
    project, directory = pipeline.reload_project(text)
    return _result(
        project,
        directory,
        f"已加载项目，共 {len(project.sentences)} 句。源文件：{project.source.path}",
    )


def save_project_table(
    project_path: Any,
    table: Any,
) -> UIResult:
    text = "" if project_path is None else str(project_path).strip()
    if not text:
        raise ProjectError("请先新建或加载项目。")
    project, directory = pipeline.reload_project(text)
    stale_count, material_changed = _apply_table_and_settings(project, table)
    manifest = save_project(project, directory)
    pipeline.export_transcript(project, directory)
    detail = ""
    if stale_count:
        detail = f"；{stale_count} 句的克隆输入有变化，下次只重做这些句子"
    elif material_changed:
        detail = "；配音模式、时间轴或混音设置有变化，下次会按需重做配音/混音"
    return _result(project, directory, f"已严格校验并保存：{manifest}{detail}")


def translate_missing(
    project_path: Any,
    table: Any,
    progress: Callable[..., Any] | None = None,
) -> UIResult:
    """Save edits and translate only enabled rows whose Chinese cell is blank."""
    text = "" if project_path is None else str(project_path).strip()
    if not text:
        raise ProjectError("请先新建或加载项目。")
    project, directory = pipeline.reload_project(text)
    _apply_table_and_settings(project, table)
    key = resolve_api_key(project.settings.translation_provider)
    save_project(project, directory)
    pipeline.export_transcript(project, directory)
    try:
        pipeline.translate_project(project, directory, api_key=key, progress=progress)
    except AsmrDubberError as exc:
        return _partial_result(directory, f"翻译未完成：{exc}\n已完成批次和表格修改均已保留。")
    except Exception as exc:
        return _partial_result(
            directory,
            f"翻译意外中断（{type(exc).__name__}）：{exc}\n项目已保留，可重试。",
        )
    return _result(project, directory, "空白中文已全部翻译；原有中文未被覆盖。")


def synthesize_and_mix(
    project_path: Any,
    table: Any,
    progress: Callable[..., Any] | None = None,
) -> UIResult:
    _progress_message(progress, 0.0, "准备逐句克隆，正在读取表格与缓存")
    text = "" if project_path is None else str(project_path).strip()
    if not text:
        raise ProjectError("请先新建或加载项目。")
    project, directory = pipeline.reload_project(text)
    _apply_table_and_settings(project, table)
    enabled = [sentence for sentence in project.sentences if sentence.enabled]
    if not enabled:
        raise ProjectError("没有启用的句子。")
    untranslated = [sentence.id for sentence in enabled if not sentence.zh_text]
    if untranslated:
        raise ProjectError("以下已启用句子没有中文，不能逐句克隆：" + ", ".join(untranslated[:20]))

    # This button promises a fresh mix. Do not expose an older mix if synthesis stops midway.
    project.chinese_stem_file = None
    project.output_file = None
    save_project(project, directory)
    pipeline.export_transcript(project, directory)
    try:
        pipeline.synthesize_project(
            project,
            directory,
            force=False,
            progress=_StageProgress(progress, 0, 2),
        )
    except SynthesisError as exc:
        return _partial_result(
            directory,
            f"逐句克隆未全部完成：{exc}\n成功句已缓存；请检查表格中的状态和错误后重试。",
        )
    except Exception as exc:
        return _partial_result(
            directory,
            f"逐句克隆意外中断（{type(exc).__name__}）：{exc}\n项目已保留，可重试。",
        )
    try:
        output = pipeline.mix_project(
            project,
            directory,
            progress=_StageProgress(progress, 1, 2),
        )
    except Exception as exc:
        return _partial_result(
            directory,
            f"混音未完成（{type(exc).__name__}）：{exc}\n逐句克隆缓存已保留。",
        )
    _progress_message(progress, 1.0, "逐句克隆与混音完成")
    return _result(project, directory, f"逐句克隆与混音完成。输出音频：{output}")


def translation_provider_fields(provider: Any) -> tuple[list[str], str, str, str, str]:
    provider_id = "" if provider is None else str(provider).strip()
    preset = PROVIDER_PRESETS.get(provider_id)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider_id}")
    return (
        list(preset["models"]),
        str(preset["default_model"]),
        str(preset["base_url"]),
        str(preset["help"]),
        api_key_status(provider_id),
    )


def asr_backend_fields(backend: Any) -> tuple[list[str], str, str, str, str]:
    backend_id = str(backend or "").strip()
    spec = ASR_BACKENDS.get(backend_id)
    if spec is None:
        raise ProjectError(f"未知 ASR 后端：{backend_id}")
    default_urls = {"openai_compatible_asr": "http://127.0.0.1:8080/v1"}
    return (
        list(spec.models),
        spec.default_model,
        f"{spec.help}\n\n**运行方式**：{spec.runtime}；{spec.execution_label}",
        f"安装/启动：{spec.setup}",
        default_urls.get(backend_id, ""),
    )


def tts_backend_fields(backend: Any) -> tuple[list[str], str, str, str, str]:
    backend_id = str(backend or "").strip()
    spec = TTS_BACKENDS.get(backend_id)
    if spec is None:
        raise ProjectError(f"未知 TTS 后端：{backend_id}")
    default_urls = {
        "gpt_sovits": "http://127.0.0.1:9880",
        "cosyvoice": "http://127.0.0.1:50000",
        "fish_speech": "http://127.0.0.1:8080",
    }
    reference = {
        "unused": "参考文本：该模型不使用",
        "optional": "参考文本：可选/由模式决定",
        "required": "参考文本：高质量克隆必填",
    }[spec.reference_text]
    return (
        list(spec.models),
        spec.default_model,
        f"{spec.help}\n\n**能力**：参考音频："
        f"{'需要' if spec.reference_audio else '不需要'}；{reference}；"
        f"{spec.execution_label}。",
        f"安装/启动：{spec.setup}",
        default_urls.get(backend_id, ""),
    )


def backend_device_choices(spec: Any, configured: str) -> tuple[list[str], str]:
    choices: list[str] = []
    for device in spec.devices:
        if device == "cuda":
            choices.extend(("cuda", "cuda:0"))
        elif device == "rocm":
            choices.append("cuda")
        else:
            choices.append(device)
    choices = list(dict.fromkeys(choices))
    configured = str(configured or "").strip()
    base = configured.partition(":")[0]
    supported = base in spec.devices or (base == "cuda" and "rocm" in spec.devices)
    value = configured if supported else choices[0]
    return choices, value


def save_settings_form(
    current_project_path: Any,
    projects_root: Any,
    huggingface_endpoint: Any,
    pypi_index_url: Any,
    global_overlap_seconds: Any,
    global_overlap_percentage: Any,
    chinese_gain_db: Any,
    match_source_loudness: Any,
    relative_loudness_db: Any,
    minimum_active_rms_dbfs: Any,
    maximum_active_rms_dbfs: Any,
    retain_chinese_stem: Any,
    asr_backend: Any,
    asr_model: Any,
    aligner_model: Any,
    asr_device: Any,
    asr_compute_type: Any,
    asr_batch_size: Any,
    asr_beam_size: Any,
    asr_vad_filter: Any,
    asr_vad_min_silence_ms: Any,
    asr_condition_on_previous_text: Any,
    asr_initial_prompt: Any,
    asr_api_base_url: Any,
    asr_timeout_seconds: Any,
    asr_funasr_vad_model: Any,
    asr_funasr_punc_model: Any,
    asr_parakeet_decoder: Any,
    asr_chunk_seconds: Any,
    asr_kotoba_chunk_seconds: Any,
    asr_review_enabled: Any,
    asr_review_models: Any,
    asr_review_background: Any,
    asr_review_prompt: Any,
    asr_review_max_drift_seconds: Any,
    asr_api_key: Any,
    translation_provider: Any,
    translation_model: Any,
    translation_base_url: Any,
    translation_api_key: Any,
    translation_temperature: Any,
    translation_top_p: Any,
    translation_max_output_tokens: Any,
    translation_prompt: Any,
    deepl_formality: Any,
    microsoft_region: Any,
    tts_backend: Any,
    tts_model: Any,
    tts_device: Any,
    clone_mode: Any,
    tts_reference_source: Any,
    tts_external_reference_audio: Any,
    tts_external_reference_text: Any,
    tts_api_base_url: Any,
    tts_timeout_seconds: Any,
    tts_model_path: Any,
    tts_config_path: Any,
    tts_executable: Any,
    tts_speed: Any,
    tts_temperature: Any,
    tts_top_p: Any,
    tts_api_key: Any,
    tts_qwen_x_vector_only: Any,
    tts_index_use_fp16: Any,
    tts_index_speaker_source: Any,
    tts_index_external_speaker_audio: Any,
    tts_index_emotion_source: Any,
    tts_index_external_emotion_audio: Any,
    tts_index_emo_alpha: Any,
    tts_index_emo_text: Any,
    tts_gpt_top_k: Any,
    tts_gpt_text_split_method: Any,
    tts_gpt_sample_steps: Any,
    tts_cosyvoice_mode: Any,
    tts_f5_nfe_steps: Any,
    tts_f5_cfg_strength: Any,
    tts_cfg_value: Any,
    tts_inference_timesteps: Any,
    tts_control_instruction: Any,
) -> tuple[str, str, str, str]:
    root = _projects_root(projects_root)
    asr_backend_id = str(asr_backend or "").strip()
    if asr_backend_id not in ASR_BACKENDS:
        raise ProjectError(f"未知 ASR 后端：{asr_backend_id}")
    asr_model_id = str(asr_model or "").strip()
    if not asr_model_id:
        raise ProjectError("ASR 模型 ID 不能为空。")
    provider = str(translation_provider or "").strip()
    if provider not in PROVIDER_PRESETS:
        raise ProjectError(f"未知翻译服务：{provider}")
    model = str(translation_model or "").strip()
    if not model:
        raise ProjectError("翻译模型 ID 不能为空。")
    base_url = str(translation_base_url or "").strip()
    if not base_url:
        raise ProjectError("翻译 API 地址不能为空。")
    prompt = str(translation_prompt or "").strip()
    if (
        provider in {"deepseek", "openai", "anthropic", "gemini", "openai_compatible"}
        and not prompt
    ):
        raise ProjectError("大模型翻译 Prompt 不能为空。")

    tts_backend_id = str(tts_backend or "").strip()
    tts_spec = TTS_BACKENDS.get(tts_backend_id)
    if tts_spec is None:
        raise ProjectError(f"未知 TTS 后端：{tts_backend_id}")
    tts_model_id = str(tts_model or "").strip()
    if not tts_model_id:
        raise ProjectError("TTS 模型 ID 不能为空。")
    clone_mode_id = str(clone_mode or "stable_reference").strip()
    if clone_mode_id not in tts_spec.clone_modes:
        raise ProjectError(f"{tts_spec.label} 不支持当前参考策略：{clone_mode_id}")
    reference_source = str(tts_reference_source or "project_sentence").strip()
    if reference_source not in {"project_sentence", "external"}:
        raise ProjectError("TTS 参考来源无效。")
    previous_settings = load_user_settings()
    uploaded_reference = str(tts_external_reference_audio or "").strip()
    uploaded_index_speaker = str(tts_index_external_speaker_audio or "").strip()
    persistent_reference = previous_settings.tts_external_reference_audio
    selected_speaker_upload = (
        uploaded_index_speaker if tts_backend_id == "indextts2" else uploaded_reference
    )
    if selected_speaker_upload:
        persistent_reference = str(store_reference_audio(selected_speaker_upload))
    uploaded_index_emotion = str(tts_index_external_emotion_audio or "").strip()
    persistent_index_emotion = previous_settings.tts_index_external_emotion_audio
    if uploaded_index_emotion:
        persistent_index_emotion = str(store_reference_audio(uploaded_index_emotion))
    external_text = str(tts_external_reference_text or "").strip()
    x_vector_only = _boolean(tts_qwen_x_vector_only, "Qwen 仅声纹向量")
    index_speaker_source = str(tts_index_speaker_source or "project_reference").strip()
    if index_speaker_source not in {
        "project_reference",
        "sentence_reference",
        "external",
    }:
        raise ProjectError("IndexTTS2 音色参考来源无效。")
    index_emotion_source = str(tts_index_emotion_source or "sentence_reference").strip()
    if index_emotion_source not in {
        "sentence_reference",
        "project_reference",
        "speaker_reference",
        "external",
        "text",
    }:
        raise ProjectError("IndexTTS2 情绪参考来源无效。")
    if (
        tts_backend_id == "indextts2"
        and index_speaker_source == "external"
        and (not persistent_reference or not Path(persistent_reference).is_file())
    ):
        raise ProjectError("选择外部音色参考后，必须上传一段可读取的音频。")
    if (
        tts_backend_id == "indextts2"
        and index_emotion_source == "external"
        and (not persistent_index_emotion or not Path(persistent_index_emotion).is_file())
    ):
        raise ProjectError("选择外部情绪参考后，必须上传一段可读取的音频。")
    if tts_backend_id != "indextts2" and reference_source == "external":
        if not persistent_reference or not Path(persistent_reference).is_file():
            raise ProjectError("选择外部参考后，必须上传一段可读取的参考音频。")
        text_required = tts_spec.reference_text == "required"
        if tts_backend_id == "qwen3_tts" and x_vector_only:
            text_required = False
        if tts_backend_id == "cosyvoice" and str(tts_cosyvoice_mode) == "cross_lingual":
            text_required = False
        if text_required and not external_text:
            raise ProjectError(f"{tts_spec.label} 使用外部参考时必须填写参考音频对应文本。")

    try:
        settings = UserSettings(
            projects_root=str(root or pipeline.default_projects_dir()),
            huggingface_endpoint=str(huggingface_endpoint or "").strip().rstrip("/"),
            pypi_index_url=str(pypi_index_url or "").strip().rstrip("/"),
            asr_backend=asr_backend_id,
            asr_model=asr_model_id,
            aligner_model=str(aligner_model or "").strip(),
            asr_device=str(asr_device or "cuda").strip(),
            asr_compute_type=str(asr_compute_type or "float16").strip(),
            asr_batch_size=int(_finite_number(asr_batch_size, "ASR 批大小")),
            asr_beam_size=int(_finite_number(asr_beam_size, "ASR Beam Size")),
            asr_vad_filter=_boolean(asr_vad_filter, "ASR VAD"),
            asr_vad_min_silence_ms=int(_finite_number(asr_vad_min_silence_ms, "VAD 最短静音毫秒")),
            asr_condition_on_previous_text=_boolean(asr_condition_on_previous_text, "ASR 上文条件"),
            asr_initial_prompt=str(asr_initial_prompt or "").strip(),
            asr_api_base_url=str(asr_api_base_url or "").strip(),
            asr_timeout_seconds=_finite_number(asr_timeout_seconds, "ASR 超时"),
            asr_funasr_vad_model=str(asr_funasr_vad_model or "fsmn-vad").strip(),
            asr_funasr_punc_model=str(asr_funasr_punc_model or "ct-punc").strip(),
            asr_parakeet_decoder=str(asr_parakeet_decoder or "tdt").strip(),
            asr_chunk_seconds=_finite_number(asr_chunk_seconds, "ASR 分块秒数"),
            asr_kotoba_chunk_seconds=_finite_number(
                asr_kotoba_chunk_seconds,
                "Kotoba 分块秒数",
            ),
            asr_review_enabled=_boolean(asr_review_enabled, "多 ASR 交叉校对"),
            asr_review_models=[
                str(value)
                for value in (
                    asr_review_models if isinstance(asr_review_models, (list, tuple)) else []
                )
            ],
            asr_review_background=str(asr_review_background or "").strip(),
            asr_review_prompt=str(asr_review_prompt or "").strip(),
            asr_review_max_drift_seconds=_finite_number(
                asr_review_max_drift_seconds,
                "ASR 校对最大漂移",
            ),
            global_overlap_seconds=_finite_number(global_overlap_seconds, "全局最长提前秒数"),
            global_overlap_percentage=_finite_number(
                global_overlap_percentage,
                "全局最长提前百分比",
            ),
            chinese_gain_db=_finite_number(chinese_gain_db, "中文增益"),
            tts_backend=tts_backend_id,
            tts_model=tts_model_id,
            tts_device=str(tts_device or "cuda").strip(),
            tts_clone_mode=clone_mode_id,
            tts_reference_source=reference_source,
            tts_external_reference_audio=persistent_reference,
            tts_external_reference_text=external_text,
            tts_api_base_url=str(tts_api_base_url or "").strip(),
            tts_timeout_seconds=_finite_number(tts_timeout_seconds, "TTS 超时"),
            tts_model_path=str(tts_model_path or "").strip(),
            tts_config_path=str(tts_config_path or "").strip(),
            tts_executable=str(tts_executable or "f5-tts_infer-cli").strip(),
            tts_speed=_finite_number(tts_speed, "TTS 语速"),
            tts_temperature=_finite_number(tts_temperature, "TTS Temperature"),
            tts_top_p=_finite_number(tts_top_p, "TTS Top P"),
            tts_qwen_x_vector_only=x_vector_only,
            tts_index_use_fp16=_boolean(tts_index_use_fp16, "IndexTTS FP16"),
            tts_index_emo_alpha=_finite_number(tts_index_emo_alpha, "IndexTTS 情绪权重"),
            tts_index_speaker_source=index_speaker_source,
            tts_index_emotion_source=index_emotion_source,
            tts_index_external_emotion_audio=persistent_index_emotion,
            tts_index_use_emo_text=index_emotion_source == "text",
            tts_index_emo_text=str(tts_index_emo_text or "").strip(),
            tts_gpt_top_k=int(_finite_number(tts_gpt_top_k, "GPT-SoVITS Top K")),
            tts_gpt_text_split_method=str(tts_gpt_text_split_method or "cut5").strip(),
            tts_gpt_sample_steps=int(_finite_number(tts_gpt_sample_steps, "GPT-SoVITS 采样步数")),
            tts_cosyvoice_mode=str(tts_cosyvoice_mode or "zero_shot").strip(),
            tts_f5_nfe_steps=int(_finite_number(tts_f5_nfe_steps, "F5 NFE 步数")),
            tts_f5_cfg_strength=_finite_number(tts_f5_cfg_strength, "F5 CFG"),
            match_source_loudness=_boolean(match_source_loudness, "跟随日语局部响度"),
            chinese_relative_loudness_db=_finite_number(relative_loudness_db, "相对日语响度"),
            chinese_min_active_rms_dbfs=_finite_number(minimum_active_rms_dbfs, "中文可听响度下限"),
            chinese_max_active_rms_dbfs=_finite_number(maximum_active_rms_dbfs, "中文响度上限"),
            retain_chinese_stem=_boolean(retain_chinese_stem, "保留中文中间轨"),
            tts_cfg_value=_finite_number(tts_cfg_value, "TTS CFG"),
            tts_inference_timesteps=int(_finite_number(tts_inference_timesteps, "TTS 推理步数")),
            tts_control_instruction=str(tts_control_instruction or "").strip(),
            translation_provider=provider,
            translation_model=model,
            translation_base_url=base_url,
            translation_temperature=_finite_number(translation_temperature, "Temperature"),
            translation_top_p=_finite_number(translation_top_p, "Top P"),
            translation_max_output_tokens=int(
                _finite_number(translation_max_output_tokens, "最大输出 tokens")
            ),
            translation_prompt=prompt,
            translation_deepl_formality=str(deepl_formality or "default").strip(),
            translation_microsoft_region=str(microsoft_region or "").strip(),
        )
        available_review_models = {value for _, value in available_asr_review_choices(settings)}
        settings.asr_review_models = [
            value for value in settings.asr_review_models if value in available_review_models
        ]
        # Also validates cross-field audio limits and the selected clone mode.
        settings.to_project_settings()
    except ValidationError as exc:
        raise ProjectError(f"设置校验失败：{exc}") from exc

    settings_path = save_user_settings(settings)
    key = str(translation_api_key or "").strip()
    if key:
        save_api_key(provider, key)
    asr_key = str(asr_api_key or "").strip()
    if asr_key:
        save_service_key(f"asr:{asr_backend_id}", asr_key)
    tts_key = str(tts_api_key or "").strip()
    if tts_key:
        save_service_key(f"tts:{tts_backend_id}", tts_key)

    project_message = ""
    path_text = str(current_project_path or "").strip()
    if path_text:
        project, directory = pipeline.reload_project(path_text)
        previous = project.settings.model_dump()
        project.settings = settings.to_project_settings(project.settings)
        if previous != project.settings.model_dump():
            project.chinese_stem_file = None
            project.output_file = None
        save_project(project, directory)
        pipeline.export_transcript(project, directory)
        project_message = "；当前项目已同步新设置"
    return (
        f"设置已保存到 {settings_path}{project_message}",
        service_key_status(f"asr:{asr_backend_id}", ASR_BACKENDS[asr_backend_id].api_key),
        api_key_status(provider),
        service_key_status(f"tts:{tts_backend_id}", tts_spec.api_key),
    )


def clear_saved_key(provider: Any) -> str:
    provider_id = str(provider or "").strip()
    if provider_id not in PROVIDER_PRESETS:
        raise ProjectError(f"未知翻译服务：{provider_id}")
    clear_api_key(provider_id)
    return api_key_status(provider_id)


def clear_saved_service_key(kind: str, backend: Any) -> str:
    backend_id = str(backend or "").strip()
    registry = ASR_BACKENDS if kind == "asr" else TTS_BACKENDS
    if backend_id not in registry:
        raise ProjectError(f"未知 {kind.upper()} 后端：{backend_id}")
    clear_service_key(f"{kind}:{backend_id}")
    return service_key_status(f"{kind}:{backend_id}", registry[backend_id].api_key)


def build_app() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("缺少 Gradio；请安装项目的 ui 可选依赖。") from exc

    def guarded(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    stored = load_user_settings()
    if not stored.projects_root:
        stored.projects_root = str(pipeline.default_projects_dir())
    preset = PROVIDER_PRESETS.get(stored.translation_provider, PROVIDER_PRESETS["deepseek"])
    initial_models = list(preset["models"])
    if stored.translation_model not in initial_models:
        initial_models.append(stored.translation_model)
    asr_spec = ASR_BACKENDS.get(stored.asr_backend, ASR_BACKENDS["parakeet_nemo"])
    initial_asr_models = list(asr_spec.models)
    if stored.asr_model not in initial_asr_models:
        initial_asr_models.append(stored.asr_model)
    tts_spec = TTS_BACKENDS.get(stored.tts_backend, TTS_BACKENDS["indextts2"])
    initial_tts_models = list(tts_spec.models)
    if stored.tts_model not in initial_tts_models:
        initial_tts_models.append(stored.tts_model)
    initial_asr_devices, initial_asr_device = backend_device_choices(asr_spec, stored.asr_device)
    initial_tts_devices, initial_tts_device = backend_device_choices(tts_spec, stored.tts_device)
    initial_clone_modes = list(tts_spec.clone_modes)
    if stored.tts_clone_mode not in initial_clone_modes:
        stored.tts_clone_mode = "stable_reference"
    llm_translation_providers = {
        "deepseek",
        "openai",
        "anthropic",
        "gemini",
        "openai_compatible",
    }
    local_asr_backends = {
        "parakeet_nemo",
        "kotoba_whisper",
        "qwen3_asr",
        "faster_whisper",
        "openai_whisper",
        "whisperx",
        "funasr",
    }
    local_tts_backends = {"voxcpm2", "qwen3_tts", "indextts2", "f5_tts", "xtts_v2"}
    installable_ids = installable_backend_ids()
    installable_asr_ids = [
        backend_id for backend_id in installable_ids if backend_id in ASR_BACKENDS
    ]
    installable_tts_ids = [
        backend_id for backend_id in installable_ids if backend_id in TTS_BACKENDS
    ]
    asr_review_choices = available_asr_review_choices(stored)
    available_review_values = {value for _, value in asr_review_choices}
    stored.asr_review_models = [
        value for value in stored.asr_review_models if value in available_review_values
    ]

    dataframe_count_options: dict[str, Any]
    if "column_count" in inspect.signature(gr.Dataframe).parameters:
        # Gradio 6.17 exposes row/column limit arguments but has not implemented
        # them yet.  Set the initial shape here; _apply_table() enforces exactly
        # nine columns server-side before any project data can be changed.
        dataframe_count_options = {
            "row_count": 1,
            "column_count": len(TABLE_HEADERS),
        }
    else:
        # Gradio 5.49/5.50 uses the tuple form.
        dataframe_count_options = {
            "row_count": (1, "dynamic"),
            "col_count": (len(TABLE_HEADERS), "fixed"),
        }

    with gr.Blocks(
        title="ASMR Dubber · 日语原声逐句中文复述",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as demo:
        gr.HTML(
            '<header class="asmr-brand">'
            '<h1 class="asmr-brand__title">ASMR Dubber</h1>'
            '<p class="asmr-brand__subtitle">日语原声逐句中文复述</p>'
            "</header>"
        )

        with gr.Tabs():
            with gr.Tab("项目", id="project"):
                source_audio = gr.File(
                    label="日语音频",
                    file_types=["audio"],
                    type="filepath",
                )
                with gr.Row():
                    new_button = gr.Button("① 新建并识别 + 翻译", variant="primary")
                    save_button = gr.Button("② 保存表格")
                    translate_button = gr.Button("补译空白中文")
                    synthesize_button = gr.Button("③ 逐句克隆 + 混音", variant="primary")

                with gr.Row():
                    project_path = gr.Textbox(
                        label="当前项目",
                        placeholder="填写已有 project.json 或项目目录的本地路径",
                    )
                    load_button = gr.Button("加载项目")

                sentence_table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=TABLE_TYPES,
                    type="array",
                    value=[],
                    interactive=True,
                    wrap=True,
                    label="句子时间轴",
                    column_widths=[90, 60, 90, 90, 360, 360, 150, 150, 280],
                    pinned_columns=2,
                    show_search="search",
                    **dataframe_count_options,
                )

                with gr.Accordion("统一声纹参考", open=True):
                    gr.Markdown(
                        "选择清晰、非纯气声、约 4–15 秒的日语句子并保存。"
                        "仅当“设置 → TTS 设置 → 参考来源”为项目内参考句时使用。"
                    )
                    with gr.Row():
                        reference_selector = gr.Dropdown(
                            label="参考句",
                            choices=[],
                            value=None,
                            interactive=True,
                            filterable=True,
                        )
                        refresh_reference_button = gr.Button("刷新")
                        save_reference_button = gr.Button("保存参考", variant="primary")
                    reference_audio = gr.Audio(
                        label="日语原句试听",
                        type="filepath",
                        interactive=False,
                    )

                with gr.Row():
                    output_audio = gr.Audio(
                        label="完成音频",
                        type="filepath",
                        interactive=False,
                    )
                    status = gr.Textbox(label="运行状态", lines=7, interactive=False)

            with gr.Tab("设置", id="settings"):
                with gr.Row():
                    save_settings_button = gr.Button("保存全部设置", variant="primary")
                    settings_status = gr.Textbox(label="设置状态", interactive=False)
                with gr.Tabs():
                    with gr.Tab("设备与模型", id="device_models"):
                        gr.Markdown(
                            f"**便携数据目录**：`{portable_home()}`  \n"
                            "运行时、模型、缓存、配置、项目和临时文件均位于程序目录内。"
                        )
                        with gr.Row():
                            refresh_system_button = gr.Button(
                                "重新检测设备与模型", variant="primary"
                            )
                        hardware_info = gr.Markdown(hardware_markdown())
                        device_recommendation = gr.Markdown(recommended_stack_markdown())
                        gr.Markdown(
                            "下表区分真实可用状态和支持等级。‘实验性’表示接口已经接入，"
                            "但尚未完成所有系统/硬件组合的真实测试。"
                        )
                        gr.Markdown("### ASR 后端")
                        asr_backend_catalog = gr.Dataframe(
                            headers=[
                                "后端",
                                "支持等级",
                                "本机适配",
                                "安装状态",
                                "详情",
                                "预计磁盘",
                            ],
                            datatype=["str"] * 6,
                            value=backend_catalog_rows(stored, "asr"),
                            interactive=False,
                            wrap=True,
                            label="ASR 后端兼容性与安装状态",
                        )
                        with gr.Accordion("安装或修复本地 ASR 后端", open=False):
                            gr.Markdown(
                                "只安装你选择的模型运行依赖，并下载、校验该后端的固定版本模型；"
                                "大型质量优先后端可能占用数 GB 到数十 GB。"
                            )
                            with gr.Row():
                                install_asr_backend_selector = gr.Dropdown(
                                    label="要安装的 ASR 后端",
                                    choices=[
                                        (ASR_BACKENDS[backend_id].label, backend_id)
                                        for backend_id in installable_asr_ids
                                    ],
                                    value=(
                                        "parakeet_nemo"
                                        if "parakeet_nemo" in installable_asr_ids
                                        else installable_asr_ids[0]
                                    ),
                                )
                                install_asr_backend_button = gr.Button(
                                    "安装/修复所选 ASR 后端", variant="primary"
                                )
                            install_asr_backend_status = gr.Textbox(
                                label="ASR 安装日志", lines=8, interactive=False
                            )

                        gr.Markdown("### TTS 后端")
                        tts_backend_catalog = gr.Dataframe(
                            headers=[
                                "后端",
                                "支持等级",
                                "本机适配",
                                "安装状态",
                                "详情",
                                "预计磁盘",
                            ],
                            datatype=["str"] * 6,
                            value=backend_catalog_rows(stored, "tts"),
                            interactive=False,
                            wrap=True,
                            label="TTS 后端兼容性与安装状态",
                        )
                        with gr.Accordion("安装或修复本地 TTS 后端", open=False):
                            gr.Markdown(
                                "只安装所选 TTS 后端的运行环境和固定模型；"
                                "外部 API 后端需要按照对应说明单独启动服务。"
                            )
                            with gr.Row():
                                install_tts_backend_selector = gr.Dropdown(
                                    label="要安装的 TTS 后端",
                                    choices=[
                                        (TTS_BACKENDS[backend_id].label, backend_id)
                                        for backend_id in installable_tts_ids
                                    ],
                                    value=(
                                        "indextts2"
                                        if "indextts2" in installable_tts_ids
                                        else installable_tts_ids[0]
                                    ),
                                )
                                install_tts_backend_button = gr.Button(
                                    "安装/修复所选 TTS 后端", variant="primary"
                                )
                            install_tts_backend_status = gr.Textbox(
                                label="TTS 安装日志", lines=8, interactive=False
                            )

                    with gr.Tab("下载与网络", id="download_network"):
                        reset_network_button = gr.Button("重置本页为默认值（需保存）")
                        gr.Markdown(
                            "留空时按项目根目录 `mirrors.json` 的顺序自动尝试国内镜像和官方源。"
                            "这里填写的地址会作为个人首选源，只用于下载依赖和模型。"
                        )
                        huggingface_endpoint = gr.Textbox(
                            label="Hugging Face Endpoint",
                            value=stored.huggingface_endpoint,
                            placeholder="可选：个人首选的 https://... 地址",
                            info="用于内置 Qwen/VoxCPM 模型下载，保存后下次安装立即生效。",
                        )
                        pypi_index_url = gr.Textbox(
                            label="Python 包索引",
                            value=stored.pypi_index_url,
                            placeholder="可选：个人首选的 PyPI /simple 地址",
                            info="用于界面内安装 ASR/TTS 后端依赖；失败后自动切换 mirrors.json。",
                        )
                        gr.Markdown("首次安装尚未进入界面时，直接编辑根目录 `mirrors.json`。")

                    with gr.Tab("总设置", id="general_settings"):
                        reset_general_button = gr.Button("重置本页为默认值（需保存）")
                        settings_projects_root = gr.Textbox(
                            label="项目输出目录",
                            value=stored.projects_root,
                            info="新项目及其源音频副本保存在此目录。",
                        )
                        gr.Markdown(
                            "#### 中文开始位置\n"
                            "正数提前量同时受秒数和句长百分比限制，实际使用两者中较小的值。"
                        )
                        with gr.Row():
                            global_overlap = gr.Number(
                                label="最长提前秒数",
                                value=stored.global_overlap_seconds,
                                minimum=-30.0,
                                maximum=30.0,
                                step=0.1,
                                info=(
                                    "正数表示句末前提前；0 从句末开始；"
                                    "负数表示句末后等待，负数不受百分比限制。"
                                ),
                            )
                            global_overlap_percentage = gr.Number(
                                label="最多提前到句长的百分比（%）",
                                value=stored.global_overlap_percentage,
                                minimum=0.0,
                                maximum=100.0,
                                step=1.0,
                                info=(
                                    "例如 50%：3 秒句最多提前 1.5 秒。"
                                    "逐句秒数可以覆盖左侧秒数，但仍受此全局百分比限制。"
                                ),
                            )
                        gr.Markdown(
                            "#### 中文响度\n自动匹配以对应日语片段为基准，并限制目标响度范围。"
                        )
                        match_source_loudness = gr.Checkbox(
                            label="自动跟随每句日语的局部响度（推荐）",
                            value=stored.match_source_loudness,
                        )
                        with gr.Row():
                            relative_loudness = gr.Number(
                                label="中文相对原句的响度（dB）",
                                value=stored.chinese_relative_loudness_db,
                                minimum=-24.0,
                                maximum=24.0,
                                step=0.5,
                                info="0 = 与原句相当；-3 = 中文稍轻；+3 = 中文稍响。",
                            )
                            minimum_loudness = gr.Number(
                                label="自动匹配的可听下限（dBFS）",
                                value=stored.chinese_min_active_rms_dbfs,
                                minimum=-60.0,
                                maximum=-20.0,
                                step=1.0,
                                info="原句极小时，中文不会低于此目标，避免完全听不清。",
                            )
                            maximum_loudness = gr.Number(
                                label="自动匹配的安全上限（dBFS）",
                                value=stored.chinese_max_active_rms_dbfs,
                                minimum=-50.0,
                                maximum=-16.0,
                                step=1.0,
                                info="原句很响时，中文目标不会超过此值，避免突然刺耳。",
                            )
                        with gr.Accordion("高级响度微调", open=False):
                            chinese_gain = gr.Number(
                                label="匹配完成后的最终微调增益（dB）",
                                value=stored.chinese_gain_db,
                                minimum=-40.0,
                                maximum=20.0,
                                step=0.5,
                                info=(
                                    "通常保持 0。它在上述自动匹配之后对整条中文轨再统一增减，"
                                    "不用于逐句大小声匹配。"
                                ),
                            )
                        retain_chinese_stem = gr.Checkbox(
                            label="保留完整中文中间轨（调试用）",
                            value=stored.retain_chinese_stem,
                            info=(
                                "默认关闭；最终音频生成后删除可重建的 float32 中文 stem，"
                                "长音频通常可节省数百 MB。逐句中文缓存仍会保留。"
                            ),
                        )

                    with gr.Tab("ASR 设置", id="asr_settings"):
                        with gr.Row():
                            reset_asr_button = gr.Button("重置本页为默认值（需保存）")
                        with gr.Accordion("本机可用的 ASR 后端与具体模型", open=True):
                            asr_availability = gr.Markdown(
                                available_backend_models_markdown("asr", stored)
                            )
                        asr_backend_selector = gr.Dropdown(
                            label="识别后端",
                            choices=[
                                (item.label, backend_id)
                                for backend_id, item in ASR_BACKENDS.items()
                            ],
                            value=stored.asr_backend,
                            interactive=True,
                        )
                        asr_help = gr.Markdown(asr_spec.help)
                        asr_setup = gr.Markdown(f"安装/启动：{asr_spec.setup}")
                        with gr.Row():
                            asr_model_selector = gr.Dropdown(
                                label="ASR 模型",
                                choices=initial_asr_models,
                                value=stored.asr_model,
                                allow_custom_value=True,
                                interactive=True,
                            )
                            asr_device = gr.Dropdown(
                                label="设备",
                                choices=initial_asr_devices,
                                value=initial_asr_device,
                                visible=stored.asr_backend in local_asr_backends,
                            )
                            asr_compute_type = gr.Dropdown(
                                label="计算精度",
                                choices=["float16", "bfloat16", "int8_float16", "int8", "float32"],
                                value=stored.asr_compute_type,
                                allow_custom_value=True,
                                visible=stored.asr_backend in {"faster_whisper", "whisperx"},
                            )
                        with gr.Row():
                            asr_batch_size = gr.Number(
                                label="批大小",
                                value=stored.asr_batch_size,
                                minimum=1,
                                maximum=32,
                                precision=0,
                                visible=stored.asr_backend
                                in {
                                    "qwen3_asr",
                                    "kotoba_whisper",
                                    "faster_whisper",
                                    "whisperx",
                                    "funasr",
                                },
                            )
                            asr_beam_size = gr.Number(
                                label="Beam Size",
                                value=stored.asr_beam_size,
                                minimum=1,
                                maximum=100,
                                precision=0,
                                visible=stored.asr_backend in {"faster_whisper", "openai_whisper"},
                            )
                            asr_timeout = gr.Number(
                                label="服务/命令超时（秒）",
                                value=stored.asr_timeout_seconds,
                                minimum=10,
                                maximum=7200,
                                visible=stored.asr_backend == "openai_compatible_asr",
                            )
                        asr_initial_prompt = gr.Textbox(
                            label="初始提示词 / 专有词汇",
                            value=stored.asr_initial_prompt,
                            info="Whisper 与兼容 API 使用；可填写角色名、作品名等。",
                            visible=stored.asr_backend
                            in {
                                "parakeet_nemo",
                                "faster_whisper",
                                "openai_whisper",
                                "openai_compatible_asr",
                            },
                        )
                        with gr.Row():
                            asr_vad_filter = gr.Checkbox(
                                label="启用 VAD",
                                value=stored.asr_vad_filter,
                                info="ASMR 默认关闭，避免过滤轻声、气声。",
                                visible=stored.asr_backend
                                in {"parakeet_nemo", "faster_whisper", "funasr"},
                            )
                            asr_vad_min_silence = gr.Number(
                                label="VAD 最短静音 ms",
                                value=stored.asr_vad_min_silence_ms,
                                minimum=50,
                                maximum=10000,
                                precision=0,
                                visible=stored.asr_backend in {"parakeet_nemo", "faster_whisper"},
                            )
                            asr_previous_context = gr.Checkbox(
                                label="使用上一段文本作上下文",
                                value=stored.asr_condition_on_previous_text,
                                visible=stored.asr_backend in {"faster_whisper", "openai_whisper"},
                            )
                        with gr.Group(visible=stored.asr_backend == "qwen3_asr") as asr_qwen_group:
                            aligner_model = gr.Textbox(
                                label="Qwen 强制对齐器",
                                value=stored.aligner_model,
                                info="默认已下载 Qwen3-ForcedAligner-0.6B。",
                            )
                        with (
                            gr.Group(visible=stored.asr_backend == "funasr") as asr_funasr_group,
                            gr.Row(),
                        ):
                            asr_funasr_vad = gr.Textbox(
                                label="FunASR VAD 模型", value=stored.asr_funasr_vad_model
                            )
                            asr_funasr_punc = gr.Textbox(
                                label="FunASR 标点模型", value=stored.asr_funasr_punc_model
                            )
                        with gr.Group(
                            visible=stored.asr_backend == "parakeet_nemo"
                        ) as asr_parakeet_group:
                            gr.Markdown("#### Parakeet / CrispASR 参数")
                            with gr.Row():
                                asr_parakeet_decoder = gr.Radio(
                                    label="0.6B 解码器",
                                    choices=[
                                        ("TDT（推荐，原生时长时间戳）", "tdt"),
                                        ("CTC（遇到 TDT 重复时尝试）", "ctc"),
                                    ],
                                    value=stored.asr_parakeet_decoder,
                                    info="1.1B 是纯 CTC，选择它时此项自动忽略。",
                                )
                                asr_chunk_seconds = gr.Number(
                                    label="分块上限（秒）",
                                    value=stored.asr_chunk_seconds,
                                    minimum=0,
                                    maximum=300,
                                    step=1,
                                    info=(
                                        "0 表示使用运行时推荐的自动流式编码；只有做对照实验"
                                        "时才手动覆盖，调得过大可能漏字或时间漂移。"
                                    ),
                                )
                        with gr.Group(
                            visible=stored.asr_backend == "kotoba_whisper"
                        ) as asr_kotoba_group:
                            asr_kotoba_chunk_seconds = gr.Number(
                                label="Kotoba 分块长度（秒）",
                                value=stored.asr_kotoba_chunk_seconds,
                                minimum=5,
                                maximum=30,
                                step=1,
                                info="官方示例使用 15 秒；ASMR 默认保持 15 秒。",
                            )
                        with gr.Group(
                            visible=stored.asr_backend == "openai_compatible_asr"
                        ) as asr_http_group:
                            asr_api_base_url = gr.Textbox(
                                label="ASR API 基础地址", value=stored.asr_api_base_url
                            )
                            with gr.Row():
                                asr_api_key = gr.Textbox(
                                    label="ASR API Key",
                                    type="password",
                                    placeholder="留空保留已保存密钥",
                                )
                                asr_key_status = gr.Textbox(
                                    label="ASR 密钥状态",
                                    value=service_key_status(
                                        f"asr:{stored.asr_backend}", asr_spec.api_key
                                    ),
                                    interactive=False,
                                )
                                clear_asr_key_button = gr.Button("清除 ASR 密钥")
                        with gr.Accordion("多 ASR + 大模型交叉校对（可选）", open=False):
                            asr_review_enabled = gr.Checkbox(
                                label="启用多模型交叉识别与上下文校对",
                                value=stored.asr_review_enabled,
                                info=(
                                    "各模型串行运行并卸载，随后使用当前“翻译设置”中的大模型"
                                    "校对文本。耗时与所选模型数近似成正比。"
                                ),
                            )
                            asr_review_models = gr.CheckboxGroup(
                                label="额外参与比对的 ASR",
                                choices=asr_review_choices,
                                value=stored.asr_review_models,
                                info=(
                                    "只显示本机已确认可用的模型；当前主 ASR 若也在列表中会自动"
                                    "去重。建议额外选择 2 个不同架构。"
                                ),
                            )
                            asr_review_background = gr.Textbox(
                                label="作品 / 人物 / 场景 / 专有词背景",
                                value=stored.asr_review_background,
                                lines=4,
                                placeholder="例如角色名、作品名、关系、场景；它只用于消歧，不能作为台词证据。",
                            )
                            asr_review_max_drift = gr.Number(
                                label="候选时间漂移容差（秒）",
                                value=stored.asr_review_max_drift_seconds,
                                minimum=0.1,
                                maximum=10,
                                step=0.1,
                                info="只用于把不同模型的候选归入同一窗口；最终边界来自被引用的真实证据。",
                            )
                            asr_review_prompt = gr.Textbox(
                                label="ASR 大模型校对 Prompt",
                                value=stored.asr_review_prompt,
                                lines=14,
                            )
                    with gr.Tab("翻译设置", id="translation_settings"):
                        reset_translation_button = gr.Button("重置本页为默认值（需保存）")
                        translation_provider = gr.Dropdown(
                            label="翻译供应商",
                            choices=[
                                (str(item["label"]), provider_id)
                                for provider_id, item in PROVIDER_PRESETS.items()
                            ],
                            value=stored.translation_provider,
                            interactive=True,
                        )
                        provider_help = gr.Markdown(str(preset["help"]))
                        with gr.Row():
                            translation_model = gr.Dropdown(
                                label="模型 / 翻译引擎",
                                choices=initial_models,
                                value=stored.translation_model,
                                allow_custom_value=True,
                                interactive=True,
                                info="可以直接输入账户可用的其他模型 ID。",
                            )
                            translation_base_url = gr.Textbox(
                                label="API 地址",
                                value=stored.translation_base_url or str(preset["base_url"]),
                            )
                        with gr.Row():
                            translation_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                placeholder="留空会保留并使用本机已保存的密钥",
                            )
                            saved_key_status = gr.Textbox(
                                label="密钥状态",
                                value=api_key_status(stored.translation_provider),
                                interactive=False,
                            )
                            clear_key_button = gr.Button("清除当前服务密钥")
                        gr.Markdown(
                            "密钥以明文保存在 `.asmr-dubber/config/secrets.json`，"
                            "不会写入源码、日志或音频项目。请勿共享该文件。"
                        )
                        with gr.Group(
                            visible=stored.translation_provider in llm_translation_providers
                        ) as translation_llm_group:
                            with gr.Row():
                                translation_temperature = gr.Number(
                                    label="Temperature",
                                    value=stored.translation_temperature,
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.1,
                                    info="LLM 采样随机度。翻译推荐较低值。",
                                )
                                translation_top_p = gr.Number(
                                    label="Top P",
                                    value=stored.translation_top_p,
                                    minimum=0.01,
                                    maximum=1.0,
                                    step=0.05,
                                )
                                translation_max_tokens = gr.Number(
                                    label="最大输出 tokens",
                                    value=stored.translation_max_output_tokens,
                                    minimum=1024,
                                    maximum=131072,
                                    step=1024,
                                    precision=0,
                                )
                            translation_prompt = gr.Textbox(
                                label="大模型翻译 Prompt",
                                value=stored.translation_prompt,
                                lines=14,
                            )
                        with gr.Group(
                            visible=stored.translation_provider == "deepl"
                        ) as translation_deepl_group:
                            deepl_formality = gr.Dropdown(
                                label="DeepL 语气正式度",
                                choices=["default", "prefer_less", "prefer_more", "less", "more"],
                                value=stored.translation_deepl_formality,
                            )
                        with gr.Group(
                            visible=stored.translation_provider == "microsoft_translate"
                        ) as translation_microsoft_group:
                            microsoft_region = gr.Textbox(
                                label="Microsoft Translator 区域",
                                value=stored.translation_microsoft_region,
                                placeholder="例如 eastasia；全局资源可留空",
                            )

                    with gr.Tab("TTS 设置", id="tts_settings"):
                        with gr.Row():
                            reset_tts_button = gr.Button("重置本页为默认值（需保存）")
                        with gr.Accordion("本机可用的 TTS 后端与具体模型", open=True):
                            tts_availability = gr.Markdown(
                                available_backend_models_markdown("tts", stored)
                            )
                        tts_backend_selector = gr.Dropdown(
                            label="配音后端",
                            choices=[
                                (TTS_BACKENDS[backend_id].label, backend_id)
                                for backend_id in TTS_BACKEND_ORDER
                            ],
                            value=stored.tts_backend,
                            interactive=True,
                        )
                        tts_help = gr.Markdown(tts_spec.help)
                        tts_setup = gr.Markdown(f"安装/启动：{tts_spec.setup}")
                        with gr.Row():
                            tts_model_selector = gr.Dropdown(
                                label="TTS 模型",
                                choices=initial_tts_models,
                                value=stored.tts_model,
                                allow_custom_value=True,
                                interactive=True,
                            )
                            tts_device = gr.Dropdown(
                                label="设备",
                                choices=initial_tts_devices,
                                value=initial_tts_device,
                                visible=stored.tts_backend in local_tts_backends,
                            )
                            tts_timeout = gr.Number(
                                label="服务/命令超时（秒）",
                                value=stored.tts_timeout_seconds,
                                minimum=10,
                                maximum=7200,
                                visible=stored.tts_backend
                                in {
                                    "indextts2",
                                    "gpt_sovits",
                                    "cosyvoice",
                                    "fish_speech",
                                    "f5_tts",
                                },
                            )
                        with gr.Group(
                            visible=stored.tts_backend != "indextts2"
                        ) as tts_generic_reference_group:
                            gr.Markdown("#### 音色参考来源")
                            tts_reference_source = gr.Radio(
                                label="参考来源",
                                choices=[
                                    ("项目内参考句（在项目页选择）", "project_sentence"),
                                    ("外部参考音频（所有项目复用）", "external"),
                                ],
                                value=stored.tts_reference_source,
                            )
                            with gr.Group(
                                visible=stored.tts_reference_source == "external"
                            ) as tts_external_group:
                                gr.Markdown(
                                    "外部参考将复制到便携配置目录。"
                                    "参考文本应与音频逐字对应；XTTS 不使用文本，"
                                    "CosyVoice 跨语言模式可不填。"
                                )
                                with gr.Row():
                                    tts_external_audio = gr.Audio(
                                        label="外部参考音频",
                                        value=(
                                            stored.tts_external_reference_audio
                                            if Path(stored.tts_external_reference_audio).is_file()
                                            else None
                                        ),
                                        type="filepath",
                                        sources=["upload"],
                                    )
                                    tts_external_text = gr.Textbox(
                                        label="外部参考文本",
                                        value=stored.tts_external_reference_text,
                                        lines=5,
                                        placeholder="准确填写参考音频中说出的日文/中文文本",
                                        visible=(
                                            tts_spec.reference_text != "unused"
                                            and not (
                                                stored.tts_backend == "qwen3_tts"
                                                and stored.tts_qwen_x_vector_only
                                            )
                                            and not (
                                                stored.tts_backend == "cosyvoice"
                                                and stored.tts_cosyvoice_mode == "cross_lingual"
                                            )
                                        ),
                                    )
                            clone_mode = gr.Radio(
                                choices=[
                                    (CLONE_MODE_LABELS[item], item) for item in initial_clone_modes
                                ],
                                value=stored.tts_clone_mode,
                                label="项目内参考策略",
                                info=(
                                    "选择外部参考时，此项不再改变音色来源；"
                                    "VoxCPM 逐句语气模式仍可读取原句韵律。"
                                ),
                            )
                        with gr.Row():
                            tts_speed = gr.Number(
                                label="语速",
                                value=stored.tts_speed,
                                minimum=0.25,
                                maximum=4,
                                step=0.05,
                                visible=stored.tts_backend in {"gpt_sovits", "f5_tts", "xtts_v2"},
                            )
                            tts_temperature = gr.Number(
                                label="Temperature",
                                value=stored.tts_temperature,
                                minimum=0,
                                maximum=2,
                                step=0.05,
                                visible=stored.tts_backend in {"qwen3_tts", "gpt_sovits"},
                            )
                            tts_top_p = gr.Number(
                                label="Top P",
                                value=stored.tts_top_p,
                                minimum=0.01,
                                maximum=1,
                                step=0.05,
                                visible=stored.tts_backend in {"qwen3_tts", "gpt_sovits"},
                            )
                        tts_instruction = gr.Textbox(
                            label="生成控制指令",
                            value=stored.tts_control_instruction,
                            placeholder="可选，例如：轻柔、贴近耳边的耳语",
                            visible=stored.tts_backend in {"voxcpm2", "qwen3_tts"},
                        )
                        with gr.Group(visible=stored.tts_backend == "voxcpm2") as tts_voxcpm_group:
                            gr.Markdown("#### VoxCPM2 参数")
                            with gr.Row():
                                tts_cfg = gr.Number(
                                    label="VoxCPM CFG",
                                    value=stored.tts_cfg_value,
                                    minimum=0.1,
                                    maximum=10,
                                    step=0.1,
                                )
                                tts_steps = gr.Number(
                                    label="推理步数",
                                    value=stored.tts_inference_timesteps,
                                    minimum=1,
                                    maximum=100,
                                    precision=0,
                                )
                        with gr.Group(visible=stored.tts_backend == "qwen3_tts") as tts_qwen_group:
                            tts_qwen_x_vector = gr.Checkbox(
                                label="仅使用声纹向量（无需参考文本，但质量可能下降）",
                                value=stored.tts_qwen_x_vector_only,
                            )
                        with gr.Group(visible=stored.tts_backend == "indextts2") as tts_index_group:
                            gr.Markdown("#### IndexTTS2 参数")
                            gr.Markdown(
                                "可在“设备与模型”页安装。Windows 也可运行 "
                                "`./scripts/windows/install-indextts2.ps1`，Linux 可运行 "
                                "`bash scripts/linux/install-indextts2.sh`。运行时与主程序隔离。"
                            )
                            with gr.Row():
                                tts_config_path = gr.Textbox(
                                    label="config.yaml 路径", value=stored.tts_config_path
                                )
                                tts_model_path = gr.Textbox(
                                    label="checkpoints 模型目录", value=stored.tts_model_path
                                )
                            with gr.Row():
                                tts_index_status = gr.Textbox(
                                    label="安装状态",
                                    value=indextts_installation_status(stored.tts_model_path),
                                    interactive=False,
                                )
                                check_tts_index_button = gr.Button("检查 IndexTTS2 安装")
                            with gr.Row():
                                tts_index_fp16 = gr.Checkbox(
                                    label="FP16", value=stored.tts_index_use_fp16
                                )
                            tts_index_speaker_source = gr.Radio(
                                label="音色参考来源",
                                choices=[
                                    ("项目参考句（推荐）", "project_reference"),
                                    ("每句话对应的日文原句", "sentence_reference"),
                                    ("外部音色参考音频", "external"),
                                ],
                                value=stored.tts_index_speaker_source,
                                info="决定中文配音使用谁的声纹和基础音色。",
                            )
                            with gr.Group(
                                visible=stored.tts_index_speaker_source == "external"
                            ) as tts_index_external_speaker_group:
                                tts_index_external_speaker_audio = gr.Audio(
                                    label="外部音色参考音频",
                                    value=(
                                        stored.tts_external_reference_audio
                                        if Path(stored.tts_external_reference_audio).is_file()
                                        else None
                                    ),
                                    type="filepath",
                                    sources=["upload"],
                                )
                            tts_index_emotion_source = gr.Radio(
                                label="情绪参考来源",
                                choices=[
                                    ("每句话对应的日文原句（推荐）", "sentence_reference"),
                                    ("项目参考句", "project_reference"),
                                    ("跟随音色参考", "speaker_reference"),
                                    ("外部情绪参考音频", "external"),
                                    ("文本情绪", "text"),
                                ],
                                value=stored.tts_index_emotion_source,
                                info="决定每句中文的语气、节奏和情绪倾向，不改变音色参考的选择。",
                            )
                            with gr.Group(
                                visible=stored.tts_index_emotion_source == "external"
                            ) as tts_index_external_emotion_group:
                                tts_index_external_emotion_audio = gr.Audio(
                                    label="外部情绪参考音频",
                                    value=(
                                        stored.tts_index_external_emotion_audio
                                        if Path(stored.tts_index_external_emotion_audio).is_file()
                                        else None
                                    ),
                                    type="filepath",
                                    sources=["upload"],
                                )
                            with gr.Row():
                                tts_index_emo_alpha = gr.Number(
                                    label="情绪参考强度",
                                    value=stored.tts_index_emo_alpha,
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    info="0 更接近基础音色，1 更充分采用所选情绪参考。",
                                )
                            with gr.Group(
                                visible=stored.tts_index_emotion_source == "text"
                            ) as tts_index_text_emotion_group:
                                tts_index_emo_text = gr.Textbox(
                                    label="情绪描述文本",
                                    value=stored.tts_index_emo_text,
                                    placeholder="留空时按当前句中文内容自动分析情绪",
                                )
                        with gr.Group(
                            visible=stored.tts_backend in {"gpt_sovits", "cosyvoice", "fish_speech"}
                        ) as tts_http_group:
                            tts_api_base_url = gr.Textbox(
                                label="TTS API 基础地址", value=stored.tts_api_base_url
                            )
                            with (
                                gr.Group(visible=tts_spec.api_key) as tts_key_group,
                                gr.Row(),
                            ):
                                tts_api_key = gr.Textbox(
                                    label="TTS API Key",
                                    type="password",
                                    placeholder="留空保留已保存密钥",
                                )
                                tts_key_status = gr.Textbox(
                                    label="TTS 密钥状态",
                                    value=service_key_status(
                                        f"tts:{stored.tts_backend}", tts_spec.api_key
                                    ),
                                    interactive=False,
                                )
                                clear_tts_key_button = gr.Button("清除 TTS 密钥")
                        with (
                            gr.Group(visible=stored.tts_backend == "gpt_sovits") as tts_gpt_group,
                            gr.Row(),
                        ):
                            tts_gpt_top_k = gr.Number(
                                label="Top K",
                                value=stored.tts_gpt_top_k,
                                minimum=1,
                                maximum=100,
                                precision=0,
                            )
                            tts_gpt_split = gr.Dropdown(
                                label="切句方法",
                                choices=["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"],
                                value=stored.tts_gpt_text_split_method,
                            )
                            tts_gpt_steps = gr.Number(
                                label="采样步数",
                                value=stored.tts_gpt_sample_steps,
                                minimum=1,
                                maximum=64,
                                precision=0,
                            )
                        with gr.Group(visible=stored.tts_backend == "cosyvoice") as tts_cosy_group:
                            tts_cosy_mode = gr.Radio(
                                label="CosyVoice 克隆模式",
                                choices=[
                                    ("Zero-shot（参考音频 + 文本）", "zero_shot"),
                                    ("Cross-lingual（仅参考音频）", "cross_lingual"),
                                ],
                                value=stored.tts_cosyvoice_mode,
                            )
                        with gr.Group(visible=stored.tts_backend == "f5_tts") as tts_f5_group:
                            tts_executable = gr.Textbox(
                                label="f5-tts_infer-cli 路径", value=stored.tts_executable
                            )
                            with gr.Row():
                                tts_f5_nfe = gr.Number(
                                    label="NFE 步数",
                                    value=stored.tts_f5_nfe_steps,
                                    minimum=4,
                                    maximum=128,
                                    precision=0,
                                )
                                tts_f5_cfg = gr.Number(
                                    label="CFG Strength",
                                    value=stored.tts_f5_cfg_strength,
                                    minimum=0,
                                    maximum=10,
                                    step=0.1,
                                )
                save_settings_bottom_button = gr.Button("保存全部设置", variant="primary")

        common_outputs = [project_path, sentence_table, output_audio, status]

        def render_result(result: UIResult) -> tuple[Any, ...]:
            manifest, rows, audio, message = result
            # Explicit component updates prevent Gradio's editable Dataframe
            # from preserving the browser's stale pre-translation value.
            return manifest, gr.update(value=rows), audio, message

        def new_callback(
            audio: Any,
            progress: Any = gr.Progress(),  # noqa: B008 - Gradio progress injection API.
        ) -> tuple[Any, ...]:
            return render_result(guarded(new_analyze_translate, audio, progress))

        def load_callback(path: Any) -> tuple[Any, ...]:
            return render_result(guarded(load_existing_project, path))

        def save_callback(path: Any, rows: Any) -> tuple[Any, ...]:
            return render_result(guarded(save_project_table, path, rows))

        def translate_callback(
            path: Any,
            rows: Any,
            progress: Any = gr.Progress(),  # noqa: B008 - Gradio progress injection API.
        ) -> tuple[Any, ...]:
            return render_result(guarded(translate_missing, path, rows, progress))

        def synthesize_callback(
            path: Any,
            rows: Any,
            progress: Any = gr.Progress(),  # noqa: B008 - Gradio progress injection API.
        ) -> tuple[Any, ...]:
            return render_result(guarded(synthesize_and_mix, path, rows, progress))

        def refresh_reference_callback(path: Any) -> tuple[Any, str | None]:
            choices, selected, preview = guarded(reference_picker_data, path)
            return gr.update(choices=choices, value=selected), preview

        def preview_reference_callback(path: Any, selected: Any) -> str | None:
            return guarded(reference_preview_path, path, selected)

        def save_reference_callback(path: Any, selected: Any) -> tuple[str, str]:
            return guarded(save_reference_sentence, path, selected)

        def provider_callback(provider: Any, current_model: Any) -> tuple[Any, ...]:
            models, model, base_url, help_text, key_text = guarded(
                translation_provider_fields, provider
            )
            provider_id = str(provider or "")
            selected_model = str(current_model or "").strip()
            if selected_model not in models:
                selected_model = model
            return (
                gr.update(choices=models, value=selected_model),
                base_url,
                help_text,
                key_text,
                gr.update(value=""),
                gr.update(visible=provider_id in llm_translation_providers),
                gr.update(visible=provider_id == "deepl"),
                gr.update(visible=provider_id == "microsoft_translate"),
            )

        def asr_backend_callback(
            backend: Any,
            current_device: Any,
            current_api_base_url: Any,
        ) -> tuple[Any, ...]:
            models, model, help_text, setup_text, base_url = guarded(asr_backend_fields, backend)
            backend_id = str(backend or "")
            spec = ASR_BACKENDS[backend_id]
            devices, device = backend_device_choices(spec, str(current_device or ""))
            return (
                gr.update(choices=models, value=model),
                help_text,
                setup_text,
                gr.update(value=base_url or str(current_api_base_url or "")),
                gr.update(visible=backend_id == "qwen3_asr"),
                gr.update(visible=backend_id == "funasr"),
                gr.update(visible=backend_id == "openai_compatible_asr"),
                service_key_status(f"asr:{backend_id}", spec.api_key),
                gr.update(value=""),
                gr.update(
                    choices=devices,
                    value=device,
                    visible=backend_id in local_asr_backends,
                ),
                gr.update(visible=backend_id in {"faster_whisper", "whisperx"}),
                gr.update(
                    visible=backend_id
                    in {
                        "qwen3_asr",
                        "kotoba_whisper",
                        "faster_whisper",
                        "whisperx",
                        "funasr",
                    }
                ),
                gr.update(visible=backend_id in {"faster_whisper", "openai_whisper"}),
                gr.update(visible=backend_id == "openai_compatible_asr"),
                gr.update(
                    visible=backend_id
                    in {
                        "parakeet_nemo",
                        "faster_whisper",
                        "openai_whisper",
                        "openai_compatible_asr",
                    }
                ),
                gr.update(visible=backend_id in {"parakeet_nemo", "faster_whisper", "funasr"}),
                gr.update(visible=backend_id in {"parakeet_nemo", "faster_whisper"}),
                gr.update(visible=backend_id in {"faster_whisper", "openai_whisper"}),
                gr.update(visible=backend_id == "parakeet_nemo"),
                gr.update(visible=backend_id == "kotoba_whisper"),
            )

        def tts_backend_callback(
            backend: Any,
            current_device: Any,
            current_clone_mode: Any,
            current_api_base_url: Any,
            qwen_x_vector_only: Any,
            cosyvoice_mode: Any,
        ) -> tuple[Any, ...]:
            models, model, help_text, setup_text, base_url = guarded(tts_backend_fields, backend)
            backend_id = str(backend or "")
            spec = TTS_BACKENDS[backend_id]
            devices, device = backend_device_choices(spec, str(current_device or ""))
            requested_clone_mode = str(current_clone_mode or "")
            clone_value = (
                requested_clone_mode
                if requested_clone_mode in spec.clone_modes
                else "stable_reference"
            )
            return (
                gr.update(choices=models, value=model),
                help_text,
                setup_text,
                gr.update(value=base_url or str(current_api_base_url or "")),
                gr.update(visible=backend_id == "voxcpm2"),
                gr.update(visible=backend_id == "qwen3_tts"),
                gr.update(visible=backend_id == "indextts2"),
                gr.update(visible=backend_id != "indextts2"),
                gr.update(visible=backend_id in {"gpt_sovits", "cosyvoice", "fish_speech"}),
                gr.update(visible=backend_id == "gpt_sovits"),
                gr.update(visible=backend_id == "cosyvoice"),
                gr.update(visible=backend_id == "f5_tts"),
                service_key_status(f"tts:{backend_id}", spec.api_key),
                gr.update(value=""),
                gr.update(
                    choices=[(CLONE_MODE_LABELS[item], item) for item in spec.clone_modes],
                    value=clone_value,
                ),
                gr.update(
                    choices=devices,
                    value=device,
                    visible=backend_id in local_tts_backends,
                ),
                gr.update(
                    visible=backend_id
                    in {
                        "indextts2",
                        "gpt_sovits",
                        "cosyvoice",
                        "fish_speech",
                        "f5_tts",
                    }
                ),
                gr.update(visible=backend_id in {"gpt_sovits", "f5_tts", "xtts_v2"}),
                gr.update(visible=backend_id in {"qwen3_tts", "gpt_sovits"}),
                gr.update(visible=backend_id in {"qwen3_tts", "gpt_sovits"}),
                gr.update(visible=backend_id in {"voxcpm2", "qwen3_tts"}),
                gr.update(visible=spec.api_key),
                gr.update(
                    visible=spec.reference_text != "unused"
                    and not (backend_id == "qwen3_tts" and bool(qwen_x_vector_only))
                    and not (
                        backend_id == "cosyvoice" and str(cosyvoice_mode or "") == "cross_lingual"
                    )
                ),
            )

        def tts_reference_source_callback(source: Any) -> Any:
            return gr.update(visible=str(source or "") == "external")

        def tts_index_reference_source_callback(
            speaker_source: Any,
            emotion_source: Any,
        ) -> tuple[Any, Any, Any]:
            return (
                gr.update(visible=str(speaker_source or "") == "external"),
                gr.update(visible=str(emotion_source or "") == "external"),
                gr.update(visible=str(emotion_source or "") == "text"),
            )

        def tts_reference_text_callback(
            source: Any,
            backend: Any,
            qwen_x_vector_only: Any,
            cosyvoice_mode: Any,
        ) -> Any:
            backend_id = str(backend or "")
            spec = TTS_BACKENDS[backend_id]
            visible = (
                str(source or "") == "external"
                and spec.reference_text != "unused"
                and not (backend_id == "qwen3_tts" and bool(qwen_x_vector_only))
                and not (backend_id == "cosyvoice" and str(cosyvoice_mode) == "cross_lingual")
            )
            return gr.update(visible=visible)

        def settings_callback(*values: Any) -> tuple[Any, ...]:
            settings_text, asr_key_text, translation_key_text, tts_key_text = guarded(
                save_settings_form, *values
            )
            _cache_component_defaults(
                demo,
                settings_reload_payload(load_user_settings()),
            )
            return (
                settings_text,
                asr_key_text,
                translation_key_text,
                tts_key_text,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            )

        def clear_key_callback(provider: Any) -> tuple[str, Any]:
            return guarded(clear_saved_key, provider), gr.update(value="")

        def clear_asr_key_callback(backend: Any) -> tuple[str, Any]:
            return guarded(clear_saved_service_key, "asr", backend), gr.update(value="")

        def clear_tts_key_callback(backend: Any) -> tuple[str, Any]:
            return guarded(clear_saved_service_key, "tts", backend), gr.update(value="")

        def _review_choices_update(
            current_values: Any,
            current: UserSettings,
        ) -> Any:
            choices = available_asr_review_choices(current)
            available_values = {value for _, value in choices}
            selected = [
                str(value)
                for value in (current_values if isinstance(current_values, (list, tuple)) else [])
                if str(value) in available_values
            ]
            return gr.update(choices=choices, value=selected)

        def refresh_system_callback(current_review_models: Any) -> tuple[Any, ...]:
            current = load_user_settings()
            profile = refresh_hardware()
            return (
                guarded(hardware_markdown, profile),
                guarded(recommended_stack_markdown, profile),
                guarded(backend_catalog_rows, current, "asr"),
                guarded(backend_catalog_rows, current, "tts"),
                guarded(available_backend_models_markdown, "asr", current),
                guarded(available_backend_models_markdown, "tts", current),
                _review_choices_update(current_review_models, current),
            )

        def install_asr_backend_callback(
            backend_id: Any,
            current_review_models: Any,
            progress: Any = gr.Progress(),  # noqa: B008 - Gradio progress injection API.
        ) -> tuple[Any, ...]:
            result = guarded(
                install_backend,
                str(backend_id or ""),
                progress=progress,
            )
            current = load_user_settings()
            return (
                result,
                guarded(backend_catalog_rows, current, "asr"),
                guarded(available_backend_models_markdown, "asr", current),
                _review_choices_update(current_review_models, current),
            )

        def install_tts_backend_callback(
            backend_id: Any,
            progress: Any = gr.Progress(),  # noqa: B008 - Gradio progress injection API.
        ) -> tuple[Any, ...]:
            result = guarded(
                install_backend,
                str(backend_id or ""),
                progress=progress,
            )
            current = load_user_settings()
            return (
                result,
                guarded(backend_catalog_rows, current, "tts"),
                guarded(available_backend_models_markdown, "tts", current),
            )

        click_parameters = inspect.signature(new_button.click).parameters
        private_event_options: dict[str, Any]
        if "api_visibility" in click_parameters:
            private_event_options = {"api_visibility": "private"}
        else:
            private_event_options = {"api_name": False}
        private_event_options.update(
            concurrency_id="asmr_dubber_pipeline",
            concurrency_limit=1,
        )

        refresh_system_button.click(
            refresh_system_callback,
            inputs=[asr_review_models],
            outputs=[
                hardware_info,
                device_recommendation,
                asr_backend_catalog,
                tts_backend_catalog,
                asr_availability,
                tts_availability,
                asr_review_models,
            ],
            **private_event_options,
        )
        install_asr_backend_button.click(
            install_asr_backend_callback,
            inputs=[install_asr_backend_selector, asr_review_models],
            outputs=[
                install_asr_backend_status,
                asr_backend_catalog,
                asr_availability,
                asr_review_models,
            ],
            **private_event_options,
        )
        install_tts_backend_button.click(
            install_tts_backend_callback,
            inputs=[install_tts_backend_selector],
            outputs=[
                install_tts_backend_status,
                tts_backend_catalog,
                tts_availability,
            ],
            **private_event_options,
        )

        project_path.change(
            refresh_reference_callback,
            inputs=[project_path],
            outputs=[reference_selector, reference_audio],
            **private_event_options,
        )
        refresh_reference_button.click(
            refresh_reference_callback,
            inputs=[project_path],
            outputs=[reference_selector, reference_audio],
            **private_event_options,
        )
        reference_selector.change(
            preview_reference_callback,
            inputs=[project_path, reference_selector],
            outputs=[reference_audio],
            **private_event_options,
        )
        save_reference_button.click(
            save_reference_callback,
            inputs=[project_path, reference_selector],
            outputs=[status, reference_audio],
            **private_event_options,
        )

        translation_provider.input(
            provider_callback,
            inputs=[translation_provider, translation_model],
            outputs=[
                translation_model,
                translation_base_url,
                provider_help,
                saved_key_status,
                translation_api_key,
                translation_llm_group,
                translation_deepl_group,
                translation_microsoft_group,
            ],
            **private_event_options,
        )

        asr_backend_selector.input(
            asr_backend_callback,
            inputs=[asr_backend_selector, asr_device, asr_api_base_url],
            outputs=[
                asr_model_selector,
                asr_help,
                asr_setup,
                asr_api_base_url,
                asr_qwen_group,
                asr_funasr_group,
                asr_http_group,
                asr_key_status,
                asr_api_key,
                asr_device,
                asr_compute_type,
                asr_batch_size,
                asr_beam_size,
                asr_timeout,
                asr_initial_prompt,
                asr_vad_filter,
                asr_vad_min_silence,
                asr_previous_context,
                asr_parakeet_group,
                asr_kotoba_group,
            ],
            **private_event_options,
        )

        tts_backend_selector.input(
            tts_backend_callback,
            inputs=[
                tts_backend_selector,
                tts_device,
                clone_mode,
                tts_api_base_url,
                tts_qwen_x_vector,
                tts_cosy_mode,
            ],
            outputs=[
                tts_model_selector,
                tts_help,
                tts_setup,
                tts_api_base_url,
                tts_voxcpm_group,
                tts_qwen_group,
                tts_index_group,
                tts_generic_reference_group,
                tts_http_group,
                tts_gpt_group,
                tts_cosy_group,
                tts_f5_group,
                tts_key_status,
                tts_api_key,
                clone_mode,
                tts_device,
                tts_timeout,
                tts_speed,
                tts_temperature,
                tts_top_p,
                tts_instruction,
                tts_key_group,
                tts_external_text,
            ],
            **private_event_options,
        )
        tts_reference_source.input(
            tts_reference_source_callback,
            inputs=[tts_reference_source],
            outputs=[tts_external_group],
            **private_event_options,
        )
        for index_reference_trigger in (
            tts_index_speaker_source,
            tts_index_emotion_source,
        ):
            index_reference_trigger.input(
                tts_index_reference_source_callback,
                inputs=[
                    tts_index_speaker_source,
                    tts_index_emotion_source,
                ],
                outputs=[
                    tts_index_external_speaker_group,
                    tts_index_external_emotion_group,
                    tts_index_text_emotion_group,
                ],
                **private_event_options,
            )
        for reference_text_trigger in (
            tts_reference_source,
            tts_backend_selector,
            tts_qwen_x_vector,
            tts_cosy_mode,
        ):
            reference_text_trigger.input(
                tts_reference_text_callback,
                inputs=[
                    tts_reference_source,
                    tts_backend_selector,
                    tts_qwen_x_vector,
                    tts_cosy_mode,
                ],
                outputs=[tts_external_text],
                **private_event_options,
            )
        check_tts_index_button.click(
            lambda path: guarded(indextts_installation_status, path),
            inputs=[tts_model_path],
            outputs=[tts_index_status],
            **private_event_options,
        )

        settings_inputs = [
            project_path,
            settings_projects_root,
            huggingface_endpoint,
            pypi_index_url,
            global_overlap,
            global_overlap_percentage,
            chinese_gain,
            match_source_loudness,
            relative_loudness,
            minimum_loudness,
            maximum_loudness,
            retain_chinese_stem,
            asr_backend_selector,
            asr_model_selector,
            aligner_model,
            asr_device,
            asr_compute_type,
            asr_batch_size,
            asr_beam_size,
            asr_vad_filter,
            asr_vad_min_silence,
            asr_previous_context,
            asr_initial_prompt,
            asr_api_base_url,
            asr_timeout,
            asr_funasr_vad,
            asr_funasr_punc,
            asr_parakeet_decoder,
            asr_chunk_seconds,
            asr_kotoba_chunk_seconds,
            asr_review_enabled,
            asr_review_models,
            asr_review_background,
            asr_review_prompt,
            asr_review_max_drift,
            asr_api_key,
            translation_provider,
            translation_model,
            translation_base_url,
            translation_api_key,
            translation_temperature,
            translation_top_p,
            translation_max_tokens,
            translation_prompt,
            deepl_formality,
            microsoft_region,
            tts_backend_selector,
            tts_model_selector,
            tts_device,
            clone_mode,
            tts_reference_source,
            tts_external_audio,
            tts_external_text,
            tts_api_base_url,
            tts_timeout,
            tts_model_path,
            tts_config_path,
            tts_executable,
            tts_speed,
            tts_temperature,
            tts_top_p,
            tts_api_key,
            tts_qwen_x_vector,
            tts_index_fp16,
            tts_index_speaker_source,
            tts_index_external_speaker_audio,
            tts_index_emotion_source,
            tts_index_external_emotion_audio,
            tts_index_emo_alpha,
            tts_index_emo_text,
            tts_gpt_top_k,
            tts_gpt_split,
            tts_gpt_steps,
            tts_cosy_mode,
            tts_f5_nfe,
            tts_f5_cfg,
            tts_cfg,
            tts_steps,
            tts_instruction,
        ]

        def settings_reload_payload(current: UserSettings) -> dict[Any, Any]:
            """Reload persisted settings for each browser session.

            Gradio serializes component defaults when the server starts. Without
            this load event, refreshing the browser restores those startup
            defaults even though the settings file was saved correctly.
            """
            current_asr_spec = ASR_BACKENDS.get(
                current.asr_backend,
                ASR_BACKENDS["parakeet_nemo"],
            )
            current_tts_spec = TTS_BACKENDS.get(
                current.tts_backend,
                TTS_BACKENDS["indextts2"],
            )
            current_preset = PROVIDER_PRESETS.get(
                current.translation_provider,
                PROVIDER_PRESETS["deepseek"],
            )
            asr_models = list(current_asr_spec.models)
            if current.asr_model not in asr_models:
                asr_models.append(current.asr_model)
            tts_models = list(current_tts_spec.models)
            if current.tts_model not in tts_models:
                tts_models.append(current.tts_model)
            translation_models = list(current_preset["models"])
            if current.translation_model not in translation_models:
                translation_models.append(current.translation_model)
            asr_devices, current_asr_device = backend_device_choices(
                current_asr_spec,
                current.asr_device,
            )
            tts_devices, current_tts_device = backend_device_choices(
                current_tts_spec,
                current.tts_device,
            )
            clone_modes = list(current_tts_spec.clone_modes)
            current_clone_mode = (
                current.tts_clone_mode
                if current.tts_clone_mode in clone_modes
                else "stable_reference"
            )
            asr_id = current_asr_spec.id
            tts_id = current_tts_spec.id
            provider_id = current.translation_provider
            external_reference = (
                current.tts_external_reference_audio
                if Path(current.tts_external_reference_audio).is_file()
                else None
            )
            external_index_emotion = (
                current.tts_index_external_emotion_audio
                if Path(current.tts_index_external_emotion_audio).is_file()
                else None
            )
            reference_text_visible = (
                current_tts_spec.reference_text != "unused"
                and not (tts_id == "qwen3_tts" and current.tts_qwen_x_vector_only)
                and not (tts_id == "cosyvoice" and current.tts_cosyvoice_mode == "cross_lingual")
            )
            review_choices = available_asr_review_choices(current)
            review_values = {value for _, value in review_choices}
            selected_review_models = [
                value for value in current.asr_review_models if value in review_values
            ]

            return {
                settings_projects_root: current.projects_root,
                huggingface_endpoint: current.huggingface_endpoint,
                pypi_index_url: current.pypi_index_url,
                global_overlap: current.global_overlap_seconds,
                global_overlap_percentage: current.global_overlap_percentage,
                chinese_gain: current.chinese_gain_db,
                match_source_loudness: current.match_source_loudness,
                relative_loudness: current.chinese_relative_loudness_db,
                minimum_loudness: current.chinese_min_active_rms_dbfs,
                maximum_loudness: current.chinese_max_active_rms_dbfs,
                retain_chinese_stem: current.retain_chinese_stem,
                asr_backend_selector: current.asr_backend,
                asr_model_selector: gr.update(
                    choices=asr_models,
                    value=current.asr_model,
                ),
                aligner_model: current.aligner_model,
                asr_device: gr.update(
                    choices=asr_devices,
                    value=current_asr_device,
                    visible=asr_id in local_asr_backends,
                ),
                asr_compute_type: gr.update(
                    value=current.asr_compute_type,
                    visible=asr_id in {"faster_whisper", "whisperx"},
                ),
                asr_batch_size: gr.update(
                    value=current.asr_batch_size,
                    visible=asr_id
                    in {
                        "qwen3_asr",
                        "kotoba_whisper",
                        "faster_whisper",
                        "whisperx",
                        "funasr",
                    },
                ),
                asr_beam_size: gr.update(
                    value=current.asr_beam_size,
                    visible=asr_id in {"faster_whisper", "openai_whisper"},
                ),
                asr_vad_filter: gr.update(
                    value=current.asr_vad_filter,
                    visible=asr_id in {"parakeet_nemo", "faster_whisper", "funasr"},
                ),
                asr_vad_min_silence: gr.update(
                    value=current.asr_vad_min_silence_ms,
                    visible=asr_id in {"parakeet_nemo", "faster_whisper"},
                ),
                asr_previous_context: gr.update(
                    value=current.asr_condition_on_previous_text,
                    visible=asr_id in {"faster_whisper", "openai_whisper"},
                ),
                asr_initial_prompt: gr.update(
                    value=current.asr_initial_prompt,
                    visible=asr_id
                    in {
                        "parakeet_nemo",
                        "faster_whisper",
                        "openai_whisper",
                        "openai_compatible_asr",
                    },
                ),
                asr_api_base_url: current.asr_api_base_url,
                asr_timeout: gr.update(
                    value=current.asr_timeout_seconds,
                    visible=asr_id == "openai_compatible_asr",
                ),
                asr_funasr_vad: current.asr_funasr_vad_model,
                asr_funasr_punc: current.asr_funasr_punc_model,
                asr_parakeet_decoder: current.asr_parakeet_decoder,
                asr_chunk_seconds: current.asr_chunk_seconds,
                asr_kotoba_chunk_seconds: current.asr_kotoba_chunk_seconds,
                asr_review_enabled: current.asr_review_enabled,
                asr_review_models: gr.update(
                    choices=review_choices,
                    value=selected_review_models,
                ),
                asr_review_background: current.asr_review_background,
                asr_review_prompt: current.asr_review_prompt,
                asr_review_max_drift: current.asr_review_max_drift_seconds,
                asr_api_key: gr.update(value=""),
                translation_provider: current.translation_provider,
                translation_model: gr.update(
                    choices=translation_models,
                    value=current.translation_model,
                ),
                translation_base_url: current.translation_base_url,
                translation_api_key: gr.update(value=""),
                translation_temperature: current.translation_temperature,
                translation_top_p: current.translation_top_p,
                translation_max_tokens: current.translation_max_output_tokens,
                translation_prompt: current.translation_prompt,
                deepl_formality: current.translation_deepl_formality,
                microsoft_region: current.translation_microsoft_region,
                tts_backend_selector: current.tts_backend,
                tts_model_selector: gr.update(
                    choices=tts_models,
                    value=current.tts_model,
                ),
                tts_device: gr.update(
                    choices=tts_devices,
                    value=current_tts_device,
                    visible=tts_id in local_tts_backends,
                ),
                clone_mode: gr.update(
                    choices=[(CLONE_MODE_LABELS[item], item) for item in clone_modes],
                    value=current_clone_mode,
                ),
                tts_reference_source: current.tts_reference_source,
                tts_external_audio: external_reference,
                tts_external_text: gr.update(
                    value=current.tts_external_reference_text,
                    visible=reference_text_visible,
                ),
                tts_api_base_url: current.tts_api_base_url,
                tts_timeout: gr.update(
                    value=current.tts_timeout_seconds,
                    visible=tts_id
                    in {
                        "indextts2",
                        "gpt_sovits",
                        "cosyvoice",
                        "fish_speech",
                        "f5_tts",
                    },
                ),
                tts_model_path: current.tts_model_path,
                tts_config_path: current.tts_config_path,
                tts_executable: current.tts_executable,
                tts_speed: gr.update(
                    value=current.tts_speed,
                    visible=tts_id in {"gpt_sovits", "f5_tts", "xtts_v2"},
                ),
                tts_temperature: gr.update(
                    value=current.tts_temperature,
                    visible=tts_id in {"qwen3_tts", "gpt_sovits"},
                ),
                tts_top_p: gr.update(
                    value=current.tts_top_p,
                    visible=tts_id in {"qwen3_tts", "gpt_sovits"},
                ),
                tts_api_key: gr.update(value=""),
                tts_qwen_x_vector: current.tts_qwen_x_vector_only,
                tts_index_fp16: current.tts_index_use_fp16,
                tts_index_speaker_source: current.tts_index_speaker_source,
                tts_index_external_speaker_audio: external_reference,
                tts_index_emotion_source: current.tts_index_emotion_source,
                tts_index_external_emotion_audio: external_index_emotion,
                tts_index_emo_alpha: current.tts_index_emo_alpha,
                tts_index_emo_text: current.tts_index_emo_text,
                tts_gpt_top_k: current.tts_gpt_top_k,
                tts_gpt_split: current.tts_gpt_text_split_method,
                tts_gpt_steps: current.tts_gpt_sample_steps,
                tts_cosy_mode: current.tts_cosyvoice_mode,
                tts_f5_nfe: current.tts_f5_nfe_steps,
                tts_f5_cfg: current.tts_f5_cfg_strength,
                tts_cfg: current.tts_cfg_value,
                tts_steps: current.tts_inference_timesteps,
                tts_instruction: gr.update(
                    value=current.tts_control_instruction,
                    visible=tts_id in {"voxcpm2", "qwen3_tts"},
                ),
                asr_help: current_asr_spec.help,
                asr_setup: f"安装/启动：{current_asr_spec.setup}",
                asr_qwen_group: gr.update(visible=asr_id == "qwen3_asr"),
                asr_funasr_group: gr.update(visible=asr_id == "funasr"),
                asr_parakeet_group: gr.update(visible=asr_id == "parakeet_nemo"),
                asr_kotoba_group: gr.update(visible=asr_id == "kotoba_whisper"),
                asr_http_group: gr.update(visible=asr_id == "openai_compatible_asr"),
                asr_key_status: service_key_status(
                    f"asr:{asr_id}",
                    current_asr_spec.api_key,
                ),
                provider_help: str(current_preset["help"]),
                saved_key_status: api_key_status(provider_id),
                translation_llm_group: gr.update(visible=provider_id in llm_translation_providers),
                translation_deepl_group: gr.update(visible=provider_id == "deepl"),
                translation_microsoft_group: gr.update(
                    visible=provider_id == "microsoft_translate"
                ),
                tts_help: current_tts_spec.help,
                tts_setup: f"安装/启动：{current_tts_spec.setup}",
                tts_voxcpm_group: gr.update(visible=tts_id == "voxcpm2"),
                tts_qwen_group: gr.update(visible=tts_id == "qwen3_tts"),
                tts_index_group: gr.update(visible=tts_id == "indextts2"),
                tts_generic_reference_group: gr.update(visible=tts_id != "indextts2"),
                tts_http_group: gr.update(
                    visible=tts_id in {"gpt_sovits", "cosyvoice", "fish_speech"}
                ),
                tts_gpt_group: gr.update(visible=tts_id == "gpt_sovits"),
                tts_cosy_group: gr.update(visible=tts_id == "cosyvoice"),
                tts_f5_group: gr.update(visible=tts_id == "f5_tts"),
                tts_key_group: gr.update(visible=current_tts_spec.api_key),
                tts_external_group: gr.update(visible=current.tts_reference_source == "external"),
                tts_index_external_speaker_group: gr.update(
                    visible=current.tts_index_speaker_source == "external"
                ),
                tts_index_external_emotion_group: gr.update(
                    visible=current.tts_index_emotion_source == "external"
                ),
                tts_index_text_emotion_group: gr.update(
                    visible=current.tts_index_emotion_source == "text"
                ),
                tts_key_status: service_key_status(
                    f"tts:{tts_id}",
                    current_tts_spec.api_key,
                ),
                tts_index_status: indextts_installation_status(current.tts_model_path),
                asr_availability: available_backend_models_markdown("asr", current),
                tts_availability: available_backend_models_markdown("tts", current),
                asr_backend_catalog: backend_catalog_rows(current, "asr"),
                tts_backend_catalog: backend_catalog_rows(current, "tts"),
            }

        network_page_components = [
            huggingface_endpoint,
            pypi_index_url,
        ]
        general_page_components = [
            settings_projects_root,
            global_overlap,
            global_overlap_percentage,
            match_source_loudness,
            relative_loudness,
            minimum_loudness,
            maximum_loudness,
            chinese_gain,
            retain_chinese_stem,
        ]
        asr_page_components = [
            asr_backend_selector,
            asr_model_selector,
            aligner_model,
            asr_device,
            asr_compute_type,
            asr_batch_size,
            asr_beam_size,
            asr_vad_filter,
            asr_vad_min_silence,
            asr_previous_context,
            asr_initial_prompt,
            asr_api_base_url,
            asr_timeout,
            asr_funasr_vad,
            asr_funasr_punc,
            asr_parakeet_decoder,
            asr_chunk_seconds,
            asr_kotoba_chunk_seconds,
            asr_review_enabled,
            asr_review_models,
            asr_review_background,
            asr_review_prompt,
            asr_review_max_drift,
            asr_api_key,
            asr_help,
            asr_setup,
            asr_qwen_group,
            asr_funasr_group,
            asr_parakeet_group,
            asr_kotoba_group,
            asr_http_group,
            asr_key_status,
            asr_availability,
        ]
        translation_page_components = [
            translation_provider,
            translation_model,
            translation_base_url,
            translation_api_key,
            translation_temperature,
            translation_top_p,
            translation_max_tokens,
            translation_prompt,
            deepl_formality,
            microsoft_region,
            provider_help,
            saved_key_status,
            translation_llm_group,
            translation_deepl_group,
            translation_microsoft_group,
        ]
        tts_page_components = [
            tts_backend_selector,
            tts_model_selector,
            tts_device,
            clone_mode,
            tts_reference_source,
            tts_external_audio,
            tts_external_text,
            tts_api_base_url,
            tts_timeout,
            tts_model_path,
            tts_config_path,
            tts_executable,
            tts_speed,
            tts_temperature,
            tts_top_p,
            tts_api_key,
            tts_qwen_x_vector,
            tts_index_fp16,
            tts_index_speaker_source,
            tts_index_external_speaker_audio,
            tts_index_emotion_source,
            tts_index_external_emotion_audio,
            tts_index_emo_alpha,
            tts_index_emo_text,
            tts_gpt_top_k,
            tts_gpt_split,
            tts_gpt_steps,
            tts_cosy_mode,
            tts_f5_nfe,
            tts_f5_cfg,
            tts_cfg,
            tts_steps,
            tts_instruction,
            tts_help,
            tts_setup,
            tts_voxcpm_group,
            tts_qwen_group,
            tts_index_group,
            tts_generic_reference_group,
            tts_http_group,
            tts_gpt_group,
            tts_cosy_group,
            tts_f5_group,
            tts_key_group,
            tts_external_group,
            tts_index_external_speaker_group,
            tts_index_external_emotion_group,
            tts_index_text_emotion_group,
            tts_key_status,
            tts_index_status,
            tts_availability,
        ]

        def reset_page_callback(components: list[Any]) -> tuple[Any, ...]:
            defaults = UserSettings()
            if not defaults.projects_root:
                defaults.projects_root = str(pipeline.default_projects_dir())
            payload = settings_reload_payload(defaults)
            return (
                *(payload[component] for component in components),
                "本页已恢复默认值；点击“保存全部设置”后生效。已保存的 API Key 和模型文件不会删除。",
            )

        reset_network_button.click(
            lambda: reset_page_callback(network_page_components),
            outputs=[*network_page_components, settings_status],
            **private_event_options,
        )
        reset_general_button.click(
            lambda: reset_page_callback(general_page_components),
            outputs=[*general_page_components, settings_status],
            **private_event_options,
        )
        reset_asr_button.click(
            lambda: reset_page_callback(asr_page_components),
            outputs=[*asr_page_components, settings_status],
            **private_event_options,
        )
        reset_translation_button.click(
            lambda: reset_page_callback(translation_page_components),
            outputs=[*translation_page_components, settings_status],
            **private_event_options,
        )
        reset_tts_button.click(
            lambda: reset_page_callback(tts_page_components),
            outputs=[*tts_page_components, settings_status],
            **private_event_options,
        )

        for settings_button in (save_settings_button, save_settings_bottom_button):
            settings_button.click(
                settings_callback,
                inputs=settings_inputs,
                outputs=[
                    settings_status,
                    asr_key_status,
                    saved_key_status,
                    tts_key_status,
                    asr_api_key,
                    translation_api_key,
                    tts_api_key,
                ],
                **private_event_options,
            )
        clear_key_button.click(
            clear_key_callback,
            inputs=[translation_provider],
            outputs=[saved_key_status, translation_api_key],
            **private_event_options,
        )
        clear_asr_key_button.click(
            clear_asr_key_callback,
            inputs=[asr_backend_selector],
            outputs=[asr_key_status, asr_api_key],
            **private_event_options,
        )
        clear_tts_key_button.click(
            clear_tts_key_callback,
            inputs=[tts_backend_selector],
            outputs=[tts_key_status, tts_api_key],
            **private_event_options,
        )

        new_button.click(
            new_callback,
            inputs=[source_audio],
            outputs=common_outputs,
            **private_event_options,
        )
        load_button.click(
            load_callback,
            inputs=[project_path],
            outputs=common_outputs,
            **private_event_options,
        )
        save_button.click(
            save_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            **private_event_options,
        )
        translate_button.click(
            translate_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            **private_event_options,
        )
        synthesize_button.click(
            synthesize_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            **private_event_options,
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    require_supported_platform()
    warnings.filterwarnings(
        "ignore",
        message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*",
        category=UserWarning,
        module=r"gradio\.routes",
    )
    app = build_app()
    home = portable_home().resolve()
    allowed_paths = {
        path
        for path in {
            (home / "temp").resolve(),
            (home / "config" / "references").resolve(),
            pipeline.default_projects_dir().resolve(),
        }
        if path.exists()
    }
    try:
        configured = load_user_settings().projects_root.strip()
        if configured:
            configured_path = Path(configured).expanduser().resolve()
            if configured_path.is_dir():
                allowed_paths.add(configured_path)
    except Exception:
        pass
    app.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=False,
        css=APP_CSS,
        allowed_paths=[str(path) for path in sorted(allowed_paths, key=str)],
    )


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
