from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..autoflow import engine
from ..autoflow.catalog import (
    Edition,
    ScanResult,
    scan_work,
)
from ..errors import ProjectError
from ..user_settings import UserSettings, load_user_settings
from .batch_configuration import _clean_folder, config_from_settings

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
                "label": "不使用台本或字幕",
                "language": "ignore",
                "timed": False,
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
                    "timed": transcript.timed,
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
        selected_timed = bool(selected is not None and selected.timed)
        selected_mode = (
            source.transcript_mode
            if source.transcript_mode in engine.TRANSCRIPT_MODES
            else engine.TRANSCRIPT_MODE_DIRECT
        )
        if selected is not None and not selected_timed:
            selected_mode = engine.TRANSCRIPT_MODE_ASR_RECONCILE
        timing_choices = [
            {
                "value": engine.TRANSCRIPT_MODE_DIRECT,
                "label": "完全采用字幕文字和时间轴（不运行 ASR）",
                "selected": selected_mode == engine.TRANSCRIPT_MODE_DIRECT,
                "disabled": not selected_timed,
            },
            {
                "value": engine.TRANSCRIPT_MODE_ASR_RECONCILE,
                "label": "运行 ASR 重新定时，只采用台本文字",
                "selected": selected_mode == engine.TRANSCRIPT_MODE_ASR_RECONCILE,
                "disabled": False,
            },
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
                "timing_choices": timing_choices,
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
    complete_direct = bool(normalized) and all(
        source.transcript_path is not None
        and source.transcript_timed
        and source.transcript_mode == engine.TRANSCRIPT_MODE_DIRECT
        for source in normalized
    )
    guidance = ""
    if complete_direct:
        guidance = " 所有音轨已选择时间轴字幕：完全按字幕制作，不运行 ASR。" + (
            "全部标为中文，不翻译字幕正文，直接配音。"
            if all(source.transcript_language == "zh" for source in normalized)
            else "中文部分直接配音；仅翻译标为日文/英文的字幕正文，请核对每轨语言。"
        )
    return AutoFlowTrackView(
        source_payloads=_source_payloads(normalized),
        track_items=_track_items(scan, normalized),
        summary=f"{prefix}：{len(normalized)} 条音轨，{matched} 条已选择台本或字幕。" + guidance,
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
        if scan.videos:
            raise ProjectError(
                "检测到视频文件。当前批量处理只扫描音频作品；单个视频请使用‘单个作品’。"
            )
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
        f"**{len(scan.images)}** 张图片和 **{len(scan.transcripts)}** 份台本或字幕。  \n"
        f"当前版本将处理 **{len(sources)}** 条音轨，其中 **{matched}** 条已匹配台本或字幕。"
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
    mode_value: Any = engine.TRANSCRIPT_MODE_DIRECT,
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
            transcript_mode=engine.TRANSCRIPT_MODE_DIRECT,
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
        mode = str(mode_value or engine.TRANSCRIPT_MODE_DIRECT).strip()
        if not transcript.timed:
            mode = engine.TRANSCRIPT_MODE_ASR_RECONCILE
        if mode not in engine.TRANSCRIPT_MODES:
            raise ProjectError("台本时间轴处理方式无效。")
        sources[source_index] = replace(
            sources[source_index],
            transcript_path=transcript.path.resolve(),
            transcript_language=language,
            transcript_timed=transcript.timed,
            transcript_mode=mode,
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
