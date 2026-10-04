from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..autoflow import engine
from ..autoflow.catalog import (
    AUDIO_EXTENSIONS,
    ScanResult,
)
from ..errors import ProjectError
from ..user_settings import UserSettings, load_user_settings
from .batch_catalog import (
    _background_options,
    _edition_choices,
    _relative_path,
    _scan,
    _sorted_editions,
    _sources_for_edition,
    _sources_from_payload,
    _track_view,
)
from .batch_configuration import _clean_folder, config_from_settings

SUBTITLE_LANGUAGE_LABELS = {
    "zh": "中文",
    "ja": "日语",
    "en": "英语",
}


@dataclass(frozen=True)
class AutoFlowEditView:
    folder: str
    edition_choices: list[tuple[str, str]]
    selected_edition: str
    include_bonus: bool
    background_choices: list[tuple[str, str]]
    selected_background: str
    selected_background_preview: str | None
    source_payloads: list[dict[str, Any]]
    track_items: list[dict[str, Any]]
    scan_summary: str
    selection_summary: str
    mode: str
    layout: str
    embed_subtitles: bool
    subtitles_only: bool
    source_subtitles_only: bool
    subtitle_language: str
    rebuild: bool
    plan_id: str


def _background_for_plan(
    scan: ScanResult,
    mode: str,
    value: Any,
) -> Path | None:
    if mode == engine.MODE_AUDIO:
        return None
    selected = str(value or "black")
    return engine._background_from_argument(scan, selected)


def serialize_plan(plan: engine.SmartTaskPlan) -> dict[str, Any]:
    return {
        "folder": str(plan.folder),
        "output_root": str(plan.output_root),
        "edition_label": plan.edition_label,
        "sources": [engine.serialize_audio_source(source) for source in plan.sources],
        "edition": dict(plan.edition),
        "mode": plan.mode,
        "layout": plan.layout,
        "background": str(plan.background) if plan.background is not None else None,
        "embed_subtitles": plan.embed_subtitles,
        "plan_id": plan.plan_id,
        "rebuild": plan.rebuild,
        "force": plan.force,
        "retry_of": plan.retry_of,
        "translate_work_title": plan.translate_work_title,
        "translate_track_titles": plan.translate_track_titles,
        "subtitles_only": plan.subtitles_only,
        "source_subtitles_only": plan.source_subtitles_only,
        "subtitle_language": plan.subtitle_language,
        "subtitle_naming": plan.subtitle_naming,
        "subtitle_custom_name": plan.subtitle_custom_name,
    }


def deserialize_plan(payload: Any) -> engine.SmartTaskPlan:
    if not isinstance(payload, dict):
        raise ProjectError("自动处理队列数据无效。")
    try:
        sources = tuple(engine.deserialize_audio_source(item) for item in payload["sources"])
        plan = engine.SmartTaskPlan(
            folder=Path(str(payload["folder"])).expanduser().resolve(),
            output_root=Path(str(payload["output_root"])).expanduser().resolve(),
            edition_label=str(payload["edition_label"]),
            sources=sources,
            edition=dict(payload["edition"]),
            mode=engine.normalize_mode(payload["mode"]),
            layout=engine.normalize_layout(payload["layout"]),
            background=(
                Path(str(payload["background"])).expanduser().resolve()
                if payload.get("background")
                else None
            ),
            embed_subtitles=bool(payload.get("embed_subtitles", True)),
            plan_id=str(payload["plan_id"]),
            rebuild=bool(payload.get("rebuild", False)),
            force=bool(payload.get("force", False)),
            retry_of=str(payload.get("retry_of") or "").strip() or None,
            translate_work_title=bool(payload.get("translate_work_title", True)),
            translate_track_titles=bool(payload.get("translate_track_titles", True)),
            subtitles_only=bool(payload.get("subtitles_only", False)),
            source_subtitles_only=bool(payload.get("source_subtitles_only", False)),
            subtitle_language=str(payload.get("subtitle_language", "source")),
            subtitle_naming=str(payload.get("subtitle_naming", "standard")),
            subtitle_custom_name=str(payload.get("subtitle_custom_name", "字幕")),
        )
    except (KeyError, TypeError, ValueError, OSError, engine.VideoPreparerError) as exc:
        raise ProjectError(f"自动处理队列数据无效：{exc}") from exc
    if not plan.sources:
        raise ProjectError("自动处理任务没有音轨。")
    if plan.subtitle_language not in {"source", "zh", "bilingual"} or plan.subtitle_naming not in {
        "original",
        "standard",
        "custom",
    }:
        raise ProjectError("字幕内容或命名方式无效。")
    return plan


def _guard_output_replacement(plan: engine.SmartTaskPlan) -> None:
    output_root = plan.output_root
    manifest = output_root / "处理清单.json"
    previous_plan = ""
    if manifest.is_file():
        try:
            previous_plan = str(
                json.loads(manifest.read_text(encoding="utf-8-sig")).get("plan_id") or ""
            )
        except (OSError, ValueError, json.JSONDecodeError):
            previous_plan = "invalid"
    generated_exists = any(
        (output_root / name).exists() for name in ("合并版", "分轨", ".autoflow", "处理清单.json")
    )
    if generated_exists and previous_plan != plan.plan_id and not plan.rebuild:
        raise ProjectError(
            f"输出目录已有其他选项生成的结果：{output_root}。"
            "如需替换，请先备份，再勾选按钮上方的“重做并替换本工具生成的旧结果”，重新加入队列。"
            "如需保留旧结果，请在“设置 → 自动处理”更改成品输出文件夹名称并保存后再加入。"
        )


def _validated_sources_for_plan(
    scan: ScanResult,
    source_payloads: Any,
) -> list[engine.AudioSource]:
    submitted = _sources_from_payload(source_payloads)
    candidates = {
        candidate.path.resolve(): candidate
        for edition in scan.editions
        for candidate in edition.all_tracks
    }
    transcripts = {item.path.resolve(): item for item in scan.transcripts}
    validated: list[engine.AudioSource] = []
    for index, source in enumerate(submitted, start=1):
        candidate = candidates.get(source.path.resolve())
        if candidate is None or candidate.path.suffix.casefold() not in AUDIO_EXTENSIONS:
            raise ProjectError(f"音轨已经不存在或不再属于当前作品：{source.path.name}")
        normalized = engine.source_from_candidate(index, candidate)
        if source.transcript_path is None:
            normalized = replace(
                normalized,
                transcript_path=None,
                transcript_language=None,
                transcript_timed=False,
                transcript_mode=engine.TRANSCRIPT_MODE_DIRECT,
            )
        else:
            transcript = transcripts.get(source.transcript_path.resolve())
            if transcript is None:
                raise ProjectError(f"台本或字幕已经不存在：{source.transcript_path.name}")
            language = str(source.transcript_language or transcript.language).casefold()
            if language not in SUBTITLE_LANGUAGE_LABELS:
                raise ProjectError(f"字幕语言无效：{source.transcript_path.name}")
            normalized = replace(
                normalized,
                transcript_path=transcript.path.resolve(),
                transcript_language=language,
                transcript_timed=transcript.timed,
                transcript_mode=(
                    engine.TRANSCRIPT_MODE_ASR_RECONCILE
                    if not transcript.timed
                    else source.transcript_mode
                    if source.transcript_mode in engine.TRANSCRIPT_MODES
                    else engine.TRANSCRIPT_MODE_DIRECT
                ),
            )
        validated.append(normalized)
    return validated


def build_plan_for_ui(
    folder_value: Any,
    edition_id: Any,
    source_payloads: Any,
    mode_value: Any,
    layout_value: Any,
    background_value: Any,
    embed_subtitles: bool,
    rebuild: bool,
    subtitles_only: bool = False,
    source_subtitles_only: bool = False,
    subtitle_language: str | None = None,
    *,
    settings: UserSettings | None = None,
    task_content: str | None = None,
) -> dict[str, Any]:
    folder = _clean_folder(folder_value)
    current = settings or load_user_settings()
    selected_subtitle_language = subtitle_language or current.autoflow_subtitle_language
    if selected_subtitle_language not in {"source", "zh", "bilingual"}:
        raise ProjectError("字幕内容只能选择双语、仅原文或仅译文。")
    custom_name = current.autoflow_subtitle_custom_name.strip()
    if current.autoflow_subtitle_naming == "custom" and (
        not custom_name
        or custom_name in {".", ".."}
        or custom_name.endswith((" ", "."))
        or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in custom_name)
    ):
        raise ProjectError("自定义字幕名称必须是有效文件名，不要填写路径或扩展名。")
    config = config_from_settings(current)
    scan = _scan(folder, config)
    label, _default_sources, edition = _sources_for_edition(
        scan,
        config,
        str(edition_id or ""),
        False,
    )
    sources = _validated_sources_for_plan(scan, source_payloads)
    edition = dict(edition)
    edition["included_optional"] = any(source.category != "main" for source in sources)
    from ..autoflow.output_policy import policy_for_settings

    content = task_content or (
        "source_subtitles"
        if source_subtitles_only
        else "subtitles"
        if subtitles_only
        else "dubbing"
    )
    edition["output_policy"] = policy_for_settings(current, content, selected_subtitle_language)
    mode = engine.MODE_AUDIO if source_subtitles_only else engine.normalize_mode(mode_value)
    subtitles_only = bool(subtitles_only or source_subtitles_only)
    layout = engine.normalize_layout(layout_value)
    background = _background_for_plan(scan, mode, background_value)
    output_root = (folder / config.output_folder_name).resolve()
    plan_id = engine.plan_identity(
        folder,
        mode=mode,
        layout=layout,
        edition=edition,
        sources=sources,
        output_root=output_root,
        background=background,
        embed_subtitles=bool(embed_subtitles) if mode != engine.MODE_AUDIO else False,
        translate_work_title=current.autoflow_translate_work_title and not source_subtitles_only,
        translate_track_titles=current.autoflow_translate_track_titles
        and not source_subtitles_only,
        subtitles_only=bool(subtitles_only),
        source_subtitles_only=bool(source_subtitles_only),
        subtitle_language=selected_subtitle_language,
        subtitle_naming=current.autoflow_subtitle_naming,
        subtitle_custom_name=custom_name,
    )
    plan = engine.SmartTaskPlan(
        folder=folder,
        output_root=output_root,
        edition_label=label,
        sources=tuple(sources),
        edition=edition,
        mode=mode,
        layout=layout,
        background=background,
        embed_subtitles=bool(embed_subtitles) if mode != engine.MODE_AUDIO else False,
        plan_id=plan_id,
        rebuild=bool(rebuild),
        force=bool(rebuild),
        translate_work_title=current.autoflow_translate_work_title and not source_subtitles_only,
        translate_track_titles=current.autoflow_translate_track_titles
        and not source_subtitles_only,
        subtitles_only=bool(subtitles_only),
        source_subtitles_only=bool(source_subtitles_only),
        subtitle_language=selected_subtitle_language,
        subtitle_naming=current.autoflow_subtitle_naming,
        subtitle_custom_name=custom_name,
    )
    _guard_output_replacement(plan)
    return serialize_plan(plan)


def edit_plan_for_ui(
    queue_payload: Any,
    plan_id: Any,
    *,
    settings: UserSettings | None = None,
) -> AutoFlowEditView:
    selected = str(plan_id or "")
    payload = next(
        (
            item
            for item in (queue_payload or [])
            if isinstance(item, dict) and str(item.get("plan_id") or "") == selected
        ),
        None,
    )
    if payload is None:
        raise ProjectError("要修改的队列任务已经不存在。")
    plan = deserialize_plan(payload)
    current = settings or load_user_settings()
    config = config_from_settings(current)
    scan = _scan(plan.folder, config)
    editions = _sorted_editions(scan, config)
    edition_id = str(plan.edition.get("edition_id") or "")
    if edition_id not in {item.id for item in editions}:
        raise ProjectError("这个任务的音频版本已经不存在，请重新扫描作品。")
    background_choices, _default_background, _default_preview = _background_options(
        scan,
        "auto",
    )
    if plan.background is None:
        selected_background = "black"
        background_preview = None
    else:
        selected_background = _relative_path(scan.root, plan.background)
        background_preview = str(plan.background)
        if selected_background not in {value for _label, value in background_choices}:
            background_choices.insert(0, (selected_background, selected_background))
    track_view = _track_view(
        scan,
        list(plan.sources),
        prefix=f"正在修改 {plan.folder.name}",
    )
    scan_summary = (
        f"**正在修改队列任务：** `{plan.folder}`  \n"
        f"发现 **{scan.audio_count}** 个音频、**{len(editions)}** 个可选版本、"
        f"**{len(scan.images)}** 张图片和 **{len(scan.transcripts)}** 份字幕。"
    )
    return AutoFlowEditView(
        folder=str(plan.folder),
        edition_choices=_edition_choices(editions),
        selected_edition=edition_id,
        include_bonus=bool(plan.edition.get("included_optional"))
        or any(source.category != "main" for source in plan.sources),
        background_choices=background_choices,
        selected_background=selected_background,
        selected_background_preview=background_preview,
        source_payloads=track_view.source_payloads,
        track_items=track_view.track_items,
        scan_summary=scan_summary,
        selection_summary=track_view.summary,
        mode=plan.mode,
        layout=plan.layout,
        embed_subtitles=plan.embed_subtitles,
        subtitles_only=plan.subtitles_only,
        source_subtitles_only=plan.source_subtitles_only,
        subtitle_language=plan.subtitle_language,
        rebuild=plan.rebuild,
        plan_id=plan.plan_id,
    )
