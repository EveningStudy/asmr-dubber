"""Scanning work folders: audio, background and edition choices."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .catalog import (
    Edition,
    ScanResult,
    TrackCandidate,
    natural_key,
)
from .constants import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MODE_AUDIO,
    NUMBERED_NAME,
    TRANSCRIPT_MODE_ASR_RECONCILE,
)
from .domain import (
    TRANSCRIPT_MODE_DIRECT,
    AppConfig,
    AudioSource,
    SmartPlanDefaults,
    SmartPlanDraft,
)
from .errors import VideoPreparerError


def discover_audio(folder: Path) -> list[AudioSource]:
    matches: list[AudioSource] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.casefold() not in AUDIO_EXTENSIONS:
            continue
        match = NUMBERED_NAME.match(path.stem)
        if match is None:
            continue
        title = match.group(2).strip(" ._-、") or path.stem.strip()
        stat = path.stat()
        matches.append(
            AudioSource(
                order=int(match.group(1)),
                path=path.resolve(),
                title_ja=title,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )

    matches.sort(key=lambda item: (item.order, item.path.name.casefold()))
    if not matches:
        raise VideoPreparerError(
            "没有找到以数字开头的音频。示例：1 开场.mp3、2 催眠.flac、10 结束.m4a"
        )

    duplicates: dict[int, list[str]] = {}
    for item in matches:
        duplicates.setdefault(item.order, []).append(item.path.name)
    repeated = {number: names for number, names in duplicates.items() if len(names) > 1}
    if repeated:
        print("\n警告：以下编号重复；同编号将按文件名排序：")
        for number, names in repeated.items():
            print(f"  {number}: {', '.join(names)}")
    return matches


def discover_background(folder: Path) -> Path | None:
    candidates = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.stem.casefold() == "null"
        and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    if not candidates:
        return None
    preference = {extension: index for index, extension in enumerate(IMAGE_EXTENSIONS)}
    candidates.sort(key=lambda path: (preference[path.suffix.casefold()], path.name.casefold()))
    if len(candidates) > 1:
        print(f"警告：发现多张 null 图片，将使用：{candidates[0].name}")
    return candidates[0].resolve()


def source_from_candidate(index: int, candidate: TrackCandidate) -> AudioSource:
    stat = candidate.path.stat()
    transcript = candidate.transcript
    return AudioSource(
        order=index,
        path=candidate.path.resolve(),
        title_ja=candidate.title,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        relative_path=candidate.relative_path,
        category=candidate.category,
        transcript_path=transcript.path.resolve() if transcript else None,
        transcript_language=transcript.language if transcript else None,
        transcript_timed=transcript.timed if transcript else False,
        transcript_mode=(
            TRANSCRIPT_MODE_DIRECT
            if transcript is None or transcript.timed
            else TRANSCRIPT_MODE_ASR_RECONCILE
        ),
        source_language=candidate.language,
    )


def _edition_rank(edition: Edition, config: AppConfig) -> tuple[int, int, Any]:
    extension = edition.extension.casefold()
    try:
        format_rank = len(config.preferred_audio_formats) - config.preferred_audio_formats.index(
            extension
        )
    except ValueError:
        format_rank = 0
    return edition.score, format_rank, natural_key(edition.label)


def _all_scan_tracks(scan: ScanResult) -> list[TrackCandidate]:
    unique: dict[Path, TrackCandidate] = {}
    for edition in scan.editions:
        for track in edition.all_tracks:
            unique.setdefault(track.path.resolve(), track)
    return sorted(
        unique.values(),
        key=lambda item: (item.order_key, natural_key(item.relative_path)),
    )


def _parse_track_selection(value: str, maximum: int) -> list[int]:
    selected: set[int] = set()
    for raw_part in value.replace("，", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*[-~]\s*(\d+)", part)
        if match:
            start, end = map(int, match.groups())
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
            continue
        if not part.isdigit():
            raise VideoPreparerError(f"无法识别音轨选择：{part}")
        selected.add(int(part))
    invalid = sorted(index for index in selected if not 1 <= index <= maximum)
    if invalid:
        raise VideoPreparerError("音轨编号超出范围：" + "、".join(map(str, invalid)))
    return sorted(selected)


def category_label(value: str) -> str:
    return {
        "main": "正文",
        "bonus": "特典",
        "sample": "样本",
        "freetalk": "Free Talk",
        "alarm": "闹钟/提示音",
    }.get(value, value)


def _sorted_editions(scan: ScanResult, config: AppConfig) -> list[Edition]:
    return sorted(
        scan.editions,
        key=lambda item: _edition_rank(item, config),
        reverse=True,
    )


def default_edition_for_work(
    scan: ScanResult,
    config: AppConfig,
    defaults: SmartPlanDefaults,
) -> Edition:
    editions = _sorted_editions(scan, config)
    if not editions:
        raise VideoPreparerError("没有找到可处理的音频文件。")
    if not defaults.edition_extension:
        return editions[0]
    same_format = [
        item
        for item in editions
        if item.extension.casefold() == defaults.edition_extension.casefold()
    ]
    if not same_format:
        return editions[0]

    def preference_rank(item: Edition) -> tuple[int, int, int, Any]:
        matches = sum(
            (
                defaults.edition_language is not None
                and item.language == defaults.edition_language,
                defaults.edition_mix_variant is not None
                and item.mix_variant == defaults.edition_mix_variant,
                defaults.edition_orientation is not None
                and item.orientation == defaults.edition_orientation,
            )
        )
        score, format_rank, natural = _edition_rank(item, config)
        return matches, score, format_rank, natural

    return max(same_format, key=preference_rank)


def _background_from_argument(scan: ScanResult, value: str) -> Path | None:
    answer = value.strip().strip('"').strip("'")
    normalized = answer.casefold()
    if normalized in {"0", "black", "none"}:
        return None
    if normalized == "auto":
        return scan.images[0].resolve() if scan.images else None
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(scan.images):
            return scan.images[index - 1].resolve()
        raise VideoPreparerError(f"背景图片编号超出范围：{answer}")

    candidate = Path(answer).expanduser()
    if not candidate.is_absolute():
        candidate = scan.root / candidate
    candidate = candidate.resolve()
    available = {path.resolve() for path in scan.images}
    if candidate not in available:
        raise VideoPreparerError("指定的背景图片不在作品文件夹的图片列表中：" + str(candidate))
    return candidate


def smart_background(
    scan: ScanResult,
    config: AppConfig,
    argument: str | None = None,
) -> Path | None:
    if argument is not None:
        return _background_from_argument(scan, argument)
    if config.background_policy == "black" or not scan.images:
        if not scan.images:
            print("\n作品文件夹中没有找到图片，将使用黑色背景。")
        return None
    if config.background_policy == "auto":
        return scan.images[0].resolve()

    print("\n请选择视频背景图片：")
    for index, image in enumerate(scan.images, start=1):
        relative = image.relative_to(scan.root).as_posix()
        marker = "（推荐）" if index == 1 else ""
        print(f"  {index}. {relative}{marker}")
    print("  0. 使用黑色背景")
    while True:
        answer = input("输入图片编号；直接按 Enter 使用推荐图片 1：").strip()
        if not answer:
            return scan.images[0].resolve()
        try:
            return _background_from_argument(scan, answer)
        except VideoPreparerError as exc:
            print(f"输入无效：{exc}")


def resolve_inherited_background(
    scan: ScanResult,
    defaults: SmartPlanDefaults,
) -> tuple[Path | None, str, str | None]:
    if defaults.background_choice == "black":
        return None, "black", None
    if defaults.background_choice == "relative" and defaults.background_relative:
        candidate = (scan.root / defaults.background_relative).resolve()
        available = {path.resolve() for path in scan.images}
        if candidate in available:
            return candidate, "relative", defaults.background_relative
    return (
        scan.images[0].resolve() if scan.images else None,
        "auto",
        None,
    )


def background_selection_details(
    scan: ScanResult,
    selected: Path | None,
) -> tuple[Path | None, str, str | None]:
    if not scan.images:
        return None, "auto", None
    if selected is None:
        return None, "black", None
    resolved = selected.resolve()
    if resolved == scan.images[0].resolve():
        return resolved, "auto", None
    return resolved, "relative", resolved.relative_to(scan.root.resolve()).as_posix()


def background_draft_label(draft: SmartPlanDraft) -> str:
    if draft.mode == MODE_AUDIO:
        if draft.background_choice == "black":
            return "黑色背景（当前纯音频，不适用）"
        if draft.background is not None:
            relative = draft.background.relative_to(draft.scan.root.resolve()).as_posix()
            marker = "（推荐）" if draft.background_choice == "auto" else ""
            return f"{relative}{marker}（当前纯音频，不适用）"
        return "自动推荐图（当前纯音频，不适用；本作品未找到图片）"
    if draft.background is None:
        if draft.background_choice == "black":
            return "黑色背景"
        return "自动推荐图（本作品未找到图片，实际使用黑色）"
    relative = draft.background.relative_to(draft.scan.root.resolve()).as_posix()
    if draft.background_choice == "auto":
        return f"{relative}（推荐）"
    return relative
