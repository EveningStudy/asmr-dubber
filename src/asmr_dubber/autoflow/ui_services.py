from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..errors import ProjectError
from ..platforms import open_directory, portable_home
from ..task_control import CancellationSignal, check_cancelled
from ..user_settings import UserSettings, load_user_settings
from . import engine
from .catalog import (
    AUDIO_EXTENSIONS,
    TIMED_TRANSCRIPT_EXTENSIONS,
    Edition,
    ScanResult,
    scan_work,
)

TRACK_HEADERS = ["序号", "音轨", "类型", "已有台本/字幕"]
QUEUE_HEADERS = ["序号", "作品", "音轨", "输出类型", "成品组织", "输出目录"]

MODE_LABELS = {
    engine.MODE_AUDIO: "纯音频",
    engine.MODE_VIDEO_NORMAL: "普通静态视频",
    engine.MODE_VIDEO_HARMONIZED: "和谐静态视频",
}
LAYOUT_LABELS = {
    engine.LAYOUT_MERGED: "合并成一部",
    engine.LAYOUT_SEPARATE: "每条音轨分别处理并输出（不合并）",
    engine.LAYOUT_BOTH: "分轨输出 + 合并版",
}
SUBTITLE_LANGUAGE_LABELS = {
    "zh": "中文",
    "ja": "日语",
    "en": "英语",
}


@dataclass(frozen=True)
class AutoFlowScanView:
    folder: str
    edition_choices: list[tuple[str, str]]
    selected_edition: str
    background_choices: list[tuple[str, str]]
    selected_background: str
    selected_background_preview: str | None
    source_payloads: list[dict[str, Any]]
    track_items: list[dict[str, Any]]
    summary: str


@dataclass(frozen=True)
class AutoFlowTrackView:
    source_payloads: list[dict[str, Any]]
    track_items: list[dict[str, Any]]
    summary: str


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
    rebuild: bool
    plan_id: str


def _clean_folder(value: Any) -> Path:
    text = str(value or "").strip(" \t\r\n\ufeff\u200b")
    quote_pairs = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’", "「": "」"}
    for _ in range(4):
        if len(text) >= 2 and text[0] in quote_pairs and text[-1] == quote_pairs[text[0]]:
            text = text[1:-1].strip()
            continue
        break
    if not text:
        raise ProjectError("请填写解压后的作品文件夹。")
    folder = Path(text).expanduser().resolve()
    if not folder.is_dir():
        raise ProjectError(f"作品文件夹不存在：{folder}")
    return folder


def _preferred_formats(value: str) -> tuple[str, ...]:
    formats: list[str] = []
    for raw in re.split(r"[,，;；\s]+", str(value or "")):
        item = raw.strip().casefold()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        if not re.fullmatch(r"\.[a-z0-9]{2,8}", item):
            raise ProjectError(f"音频格式写法无效：{raw}")
        if item not in formats:
            formats.append(item)
    if not formats:
        raise ProjectError("至少填写一种优先音频格式。")
    return tuple(formats)


def config_from_settings(settings: UserSettings | None = None) -> engine.AppConfig:
    current = settings or load_user_settings()
    output_name = current.autoflow_output_folder_name.strip()
    if (
        not output_name
        or output_name in {".", ".."}
        or output_name.rstrip(" .") != output_name
        or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in output_name)
    ):
        raise ProjectError("自动处理输出文件夹名称不符合 Windows 文件名规则。")
    return engine.AppConfig(
        asmr_root=portable_home().parent.resolve(),
        harmonized_volume_db=-abs(current.autoflow_harmonized_volume_reduction_db),
        harmonized_delay_seconds=round(current.autoflow_harmonized_delay_minutes * 60),
        timestamp_footer=current.autoflow_timestamp_footer.strip(),
        output_folder_name=output_name,
        default_output_layout=current.autoflow_default_layout,
        preferred_audio_formats=_preferred_formats(current.autoflow_preferred_audio_formats),
        bonus_policy="include" if current.autoflow_include_bonus else "exclude",
        background_policy=current.autoflow_background_policy,
        reference_wait_seconds=(
            current.autoflow_reference_wait_seconds
            if current.autoflow_reference_wait_enabled
            else 0
        ),
    )


def _scan(folder: Path, config: engine.AppConfig) -> ScanResult:
    return scan_work(
        folder,
        excluded_directories=(config.output_folder_name,),
    )


def _sorted_editions(scan: ScanResult, config: engine.AppConfig) -> list[Edition]:
    return engine._sorted_editions(scan, config)


def _edition_choices(editions: list[Edition]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for index, edition in enumerate(editions):
        optional = (
            f"，另有 {len(edition.optional_tracks)} 条附加音轨" if edition.optional_tracks else ""
        )
        recommended = "（推荐）" if index == 0 else ""
        choices.append(
            (f"{edition.label} · {len(edition.tracks)} 条{optional}{recommended}", edition.id)
        )
    return choices


def _source_id(source: engine.AudioSource) -> str:
    return source.relative_path or str(source.path.resolve())


def _normalize_source_order(sources: list[engine.AudioSource]) -> list[engine.AudioSource]:
    return [replace(source, order=index) for index, source in enumerate(sources, start=1)]


def _source_payloads(sources: list[engine.AudioSource]) -> list[dict[str, Any]]:
    return [engine.serialize_audio_source(source) for source in _normalize_source_order(sources)]


def _sources_from_payload(payload: Any) -> list[engine.AudioSource]:
    if not isinstance(payload, list):
        raise ProjectError("本作品的音轨列表无效，请重新扫描。")
    try:
        sources = [engine.deserialize_audio_source(item) for item in payload]
    except (TypeError, ValueError, OSError, engine.VideoPreparerError) as exc:
        raise ProjectError(f"本作品的音轨列表无效，请重新扫描：{exc}") from exc
    if not sources:
        raise ProjectError("本作品没有选择任何音轨。")
    identifiers = [_source_id(source) for source in sources]
    if len(identifiers) != len(set(identifiers)):
        raise ProjectError("本作品的音轨列表包含重复文件，请重新扫描。")
    return _normalize_source_order(sources)


def _relative_path(root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _background_options(
    scan: ScanResult,
    policy: str,
) -> tuple[list[tuple[str, str]], str, str | None]:
    choices = [
        (image.relative_to(scan.root).as_posix(), image.relative_to(scan.root).as_posix())
        for image in scan.images
    ]
    choices.append(("黑色背景", "black"))
    if policy == "black" or not scan.images:
        return choices, "black", None
    relative = scan.images[0].relative_to(scan.root).as_posix()
    return choices, relative, str(scan.images[0])


def _track_items(scan: ScanResult, sources: list[engine.AudioSource]) -> list[dict[str, Any]]:
    transcript_by_path = {item.path.resolve(): item for item in scan.transcripts}
    items: list[dict[str, Any]] = []
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        selected_path = source.transcript_path.resolve() if source.transcript_path else None
        transcript_choices: list[dict[str, Any]] = [
            {
                "value": "",
                "label": "不使用字幕",
                "language": "ignore",
                "selected": selected_path is None,
            }
        ]
        for transcript in scan.transcripts:
            label = SUBTITLE_LANGUAGE_LABELS.get(transcript.language, transcript.language)
            transcript_choices.append(
                {
                    "value": transcript.relative_path,
                    "label": f"{transcript.relative_path} · {label}",
                    "language": transcript.language,
                    "selected": selected_path == transcript.path.resolve(),
                }
            )
        selected = transcript_by_path.get(selected_path) if selected_path is not None else None
        selected_language = (
            source.transcript_language
            if source.transcript_language in SUBTITLE_LANGUAGE_LABELS
            else selected.language
            if selected is not None
            else "zh"
        )
        language_choices = [
            {
                "value": language,
                "label": label,
                "selected": selected_language == language,
            }
            for language, label in SUBTITLE_LANGUAGE_LABELS.items()
        ]
        items.append(
            {
                "id": _source_id(source),
                "position": index,
                "path": source.relative_path or source.path.name,
                "title": source.title_ja,
                "category": engine.category_label(source.category),
                "transcript_choices": transcript_choices,
                "language_choices": language_choices,
                "has_subtitle": selected_path is not None,
                "can_move_up": index > 1,
                "can_move_down": index < total,
            }
        )
    return items


def _track_view(
    scan: ScanResult,
    sources: list[engine.AudioSource],
    *,
    prefix: str,
) -> AutoFlowTrackView:
    normalized = _normalize_source_order(sources)
    matched = sum(source.transcript_path is not None for source in normalized)
    return AutoFlowTrackView(
        source_payloads=_source_payloads(normalized),
        track_items=_track_items(scan, normalized),
        summary=f"{prefix}：{len(normalized)} 条音轨，{matched} 条已选择字幕。",
    )


def _sources_for_edition(
    scan: ScanResult,
    config: engine.AppConfig,
    edition_id: str,
    include_bonus: bool,
) -> tuple[str, list[engine.AudioSource], dict[str, Any]]:
    return engine.choose_tracks(
        scan,
        replace(config, bonus_policy="exclude"),
        edition_argument=edition_id,
        include_bonus=include_bonus,
    )


def scan_for_ui(
    folder_value: Any,
    include_bonus: bool | None = None,
    *,
    settings: UserSettings | None = None,
) -> AutoFlowScanView:
    folder = _clean_folder(folder_value)
    current = settings or load_user_settings()
    config = config_from_settings(current)
    scan = _scan(folder, config)
    editions = _sorted_editions(scan, config)
    if not editions:
        raise ProjectError("这个文件夹里没有找到可处理的音频。")
    edition = editions[0]
    _label, sources, _metadata = _sources_for_edition(
        scan,
        config,
        edition.id,
        current.autoflow_include_bonus if include_bonus is None else bool(include_bonus),
    )
    background_choices, selected_background, selected_background_preview = _background_options(
        scan,
        current.autoflow_background_policy,
    )
    matched = sum(source.transcript_path is not None for source in sources)
    summary = (
        f"**扫描完成：** `{folder}`  \n"
        f"发现 **{scan.audio_count}** 个音频、**{len(editions)}** 个可选版本、"
        f"**{len(scan.images)}** 张图片和 **{len(scan.transcripts)}** 份字幕。  \n"
        f"当前版本将处理 **{len(sources)}** 条音轨，其中 **{matched}** 条已匹配字幕。"
    )
    return AutoFlowScanView(
        folder=str(folder),
        edition_choices=_edition_choices(editions),
        selected_edition=edition.id,
        background_choices=background_choices,
        selected_background=selected_background,
        selected_background_preview=selected_background_preview,
        source_payloads=_source_payloads(sources),
        track_items=_track_items(scan, sources),
        summary=summary,
    )


def preview_edition_for_ui(
    folder_value: Any,
    edition_id: Any,
    include_bonus: bool,
    *,
    settings: UserSettings | None = None,
) -> AutoFlowTrackView:
    folder = _clean_folder(folder_value)
    config = config_from_settings(settings)
    scan = _scan(folder, config)
    label, sources, _metadata = _sources_for_edition(
        scan,
        config,
        str(edition_id or ""),
        bool(include_bonus),
    )
    return _track_view(
        scan,
        sources,
        prefix=f"已选择 {label}",
    )


def track_view_from_payload(
    folder_value: Any,
    source_payloads: Any,
    *,
    settings: UserSettings | None = None,
    prefix: str = "当前选择",
) -> AutoFlowTrackView:
    folder = _clean_folder(folder_value)
    config = config_from_settings(settings)
    scan = _scan(folder, config)
    return _track_view(scan, _sources_from_payload(source_payloads), prefix=prefix)


def reorder_tracks_for_ui(
    folder_value: Any,
    source_payloads: Any,
    ordered_ids: Any,
    *,
    settings: UserSettings | None = None,
) -> AutoFlowTrackView:
    sources = _sources_from_payload(source_payloads)
    requested = [str(item) for item in (ordered_ids or [])]
    source_by_id = {_source_id(source): source for source in sources}
    if len(requested) != len(sources) or set(requested) != set(source_by_id):
        raise ProjectError("音轨顺序已经变化，请重新扫描后再试。")
    reordered = [source_by_id[item] for item in requested]
    return track_view_from_payload(
        folder_value,
        _source_payloads(reordered),
        settings=settings,
        prefix="已更新音轨顺序",
    )


def set_track_subtitle_for_ui(
    folder_value: Any,
    source_payloads: Any,
    track_id: Any,
    transcript_value: Any,
    language_value: Any,
    *,
    settings: UserSettings | None = None,
) -> AutoFlowTrackView:
    folder = _clean_folder(folder_value)
    config = config_from_settings(settings)
    scan = _scan(folder, config)
    sources = _sources_from_payload(source_payloads)
    selected_id = str(track_id or "")
    source_index = next(
        (index for index, source in enumerate(sources) if _source_id(source) == selected_id),
        None,
    )
    if source_index is None:
        raise ProjectError("找不到要修改的音轨，请重新扫描。")
    transcript_text = str(transcript_value or "").strip()
    if not transcript_text:
        sources[source_index] = replace(
            sources[source_index],
            transcript_path=None,
            transcript_language=None,
            transcript_timed=False,
        )
    else:
        transcript = next(
            (item for item in scan.transcripts if item.relative_path == transcript_text),
            None,
        )
        if transcript is None:
            raise ProjectError("所选字幕已经不存在，请重新扫描。")
        language = str(language_value or transcript.language).casefold()
        if language not in SUBTITLE_LANGUAGE_LABELS:
            raise ProjectError("字幕语言选择无效。")
        sources[source_index] = replace(
            sources[source_index],
            transcript_path=transcript.path.resolve(),
            transcript_language=language,
            transcript_timed=True,
        )
    return _track_view(scan, sources, prefix="已更新字幕选择")


def background_preview_path(
    folder_value: Any,
    background_value: Any,
) -> Path | None:
    folder = _clean_folder(folder_value)
    selected = str(background_value or "black").strip()
    if selected == "black":
        return None
    candidate = (folder / Path(selected)).resolve()
    try:
        candidate.relative_to(folder)
    except ValueError as exc:
        raise ProjectError("视频画面必须位于作品文件夹中。") from exc
    if not candidate.is_file() or candidate.suffix.casefold() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }:
        raise ProjectError("所选视频画面不存在或格式不支持。")
    return candidate


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
        )
    except (KeyError, TypeError, ValueError, OSError, engine.VideoPreparerError) as exc:
        raise ProjectError(f"自动处理队列数据无效：{exc}") from exc
    if not plan.sources:
        raise ProjectError("自动处理任务没有音轨。")
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
            "输出目录里已有另一套自动处理结果。需要替换时，请勾选“重做并替换本工具生成的旧结果”。"
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
            )
        else:
            transcript = transcripts.get(source.transcript_path.resolve())
            if (
                transcript is None
                or transcript.path.suffix.casefold() not in TIMED_TRANSCRIPT_EXTENSIONS
            ):
                raise ProjectError(f"字幕已经不存在或格式不支持：{source.transcript_path.name}")
            language = str(source.transcript_language or transcript.language).casefold()
            if language not in SUBTITLE_LANGUAGE_LABELS:
                raise ProjectError(f"字幕语言无效：{source.transcript_path.name}")
            normalized = replace(
                normalized,
                transcript_path=transcript.path.resolve(),
                transcript_language=language,
                transcript_timed=True,
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
    *,
    settings: UserSettings | None = None,
) -> dict[str, Any]:
    folder = _clean_folder(folder_value)
    current = settings or load_user_settings()
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
    mode = engine.normalize_mode(mode_value)
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
        translate_work_title=current.autoflow_translate_work_title,
        translate_track_titles=current.autoflow_translate_track_titles,
        subtitles_only=bool(subtitles_only),
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
        translate_work_title=current.autoflow_translate_work_title,
        translate_track_titles=current.autoflow_translate_track_titles,
        subtitles_only=bool(subtitles_only),
    )
    _guard_output_replacement(plan)
    return serialize_plan(plan)


def add_plan_to_queue(queue_payload: Any, plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    plan = deserialize_plan(plan_payload)
    for current_payload in queue:
        current = deserialize_plan(current_payload)
        if current.folder == plan.folder:
            raise ProjectError("这个作品已经在队列里。")
        if current.output_root == plan.output_root:
            raise ProjectError(f"队列中已有任务使用输出目录：{plan.output_root}")
    queue.append(plan_payload)
    return queue


def replace_plan_in_queue(
    queue_payload: Any,
    original_plan_id: Any,
    plan_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    selected = str(original_plan_id or "")
    index = next(
        (index for index, item in enumerate(queue) if str(item.get("plan_id") or "") == selected),
        None,
    )
    if index is None:
        raise ProjectError("要修改的队列任务已经不存在。")
    plan = deserialize_plan(plan_payload)
    for current_index, current_payload in enumerate(queue):
        if current_index == index:
            continue
        current = deserialize_plan(current_payload)
        if current.folder == plan.folder:
            raise ProjectError("队列中已有这个作品。")
        if current.output_root == plan.output_root:
            raise ProjectError(f"队列中已有任务使用输出目录：{plan.output_root}")
    queue[index] = plan_payload
    return queue


def remove_plan_from_queue(queue_payload: Any, plan_id: Any) -> list[dict[str, Any]]:
    selected = str(plan_id or "")
    return [
        dict(item)
        for item in (queue_payload or [])
        if isinstance(item, dict) and str(item.get("plan_id") or "") != selected
    ]


def reorder_queue_for_ui(queue_payload: Any, ordered_ids: Any) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    requested = [str(item) for item in (ordered_ids or [])]
    by_id = {str(item.get("plan_id") or ""): item for item in queue}
    if len(requested) != len(queue) or set(requested) != set(by_id):
        raise ProjectError("队列已经变化，请重试。")
    return [by_id[item] for item in requested]


def toggle_plan_rebuild(queue_payload: Any, plan_id: Any) -> list[dict[str, Any]]:
    queue = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
    selected = str(plan_id or "")
    found = False
    for item in queue:
        if str(item.get("plan_id") or "") != selected:
            continue
        enabled = not bool(item.get("rebuild", False))
        item["rebuild"] = enabled
        item["force"] = enabled
        found = True
        break
    if not found:
        raise ProjectError("要重新处理的队列任务已经不存在。")
    return queue


def queue_rows(queue_payload: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for index, payload in enumerate(queue_payload or [], start=1):
        plan = deserialize_plan(payload)
        rows.append(
            [
                index,
                plan.folder.name,
                len(plan.sources),
                f"{'仅字幕 · ' if plan.subtitles_only else ''}{MODE_LABELS[plan.mode]}",
                LAYOUT_LABELS[plan.layout],
                str(plan.output_root),
            ]
        )
    return rows


def queue_items_for_ui(
    queue_payload: Any,
    *,
    runtime: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    queue = list(queue_payload or [])
    total = len(queue)
    runtime_by_plan = runtime or {}
    for index, payload in enumerate(queue, start=1):
        plan = deserialize_plan(payload)
        task_runtime = runtime_by_plan.get(plan.plan_id, {})
        items.append(
            {
                "id": plan.plan_id,
                "position": index,
                "work": plan.folder.name,
                "tracks": len(plan.sources),
                "mode": f"{'仅字幕 · ' if plan.subtitles_only else ''}{MODE_LABELS[plan.mode]}",
                "layout": LAYOUT_LABELS[plan.layout],
                "output": plan.output_root.as_posix(),
                "titles": (
                    "翻译作品名和音轨标题"
                    if plan.translate_work_title and plan.translate_track_titles
                    else "只翻译作品名"
                    if plan.translate_work_title
                    else "只翻译音轨标题"
                    if plan.translate_track_titles
                    else "保留原标题"
                ),
                "rebuild": plan.rebuild,
                "rebuild_label": "取消重新处理" if plan.rebuild else "重新处理",
                "can_move_up": index > 1,
                "can_move_down": index < total,
                "reference_ready": bool(task_runtime.get("reference_ready")),
                "reference_request_id": str(task_runtime.get("request_id") or ""),
                "reference_status": str(task_runtime.get("status") or ""),
            }
        )
    return items


def queue_choices(queue_payload: Any) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for index, payload in enumerate(queue_payload or [], start=1):
        plan = deserialize_plan(payload)
        choices.append((f"{index}. {plan.folder.name}", plan.plan_id))
    return choices


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
        rebuild=plan.rebuild,
        plan_id=plan.plan_id,
    )


def run_queue(
    queue_payload: Any,
    *,
    cancel_event: CancellationSignal | None = None,
    reference_event_callback: engine.ReferenceEventCallback | None = None,
) -> tuple[int, list[str]]:
    plans = [deserialize_plan(payload) for payload in (queue_payload or [])]
    if not plans:
        raise ProjectError("队列还是空的，请先扫描作品并加入队列。")
    check_cancelled(cancel_event)
    config = config_from_settings()
    paths = engine.find_tool_paths(config)
    engine.validate_asmr_version()
    result = engine.execute_smart_queue(
        paths,
        config,
        plans,
        reference_event_callback=reference_event_callback,
    )
    check_cancelled(cancel_event)
    return result, [str(plan.output_root) for plan in plans]


def open_output_directory(path_value: Any) -> str:
    path = Path(str(path_value or "")).expanduser().resolve()
    opened = open_directory(path)
    return f"已打开输出目录：{opened}"


def recent_log_text(max_characters: int = 30_000) -> str:
    path = engine.LOG_FILE
    if not path.is_file():
        return "还没有自动处理日志。"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"无法读取自动处理日志：{exc}"
    return text[-max_characters:]
