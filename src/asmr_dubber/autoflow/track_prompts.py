"""Interactive track selection prompts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .catalog import (
    Edition,
    ScanResult,
    TrackCandidate,
    natural_key,
)
from .domain import (
    AppConfig,
    AudioSource,
    SmartPlanDraft,
)
from .errors import VideoPreparerError
from .prompts import ask_yes_no
from .source_scan import (
    _all_scan_tracks,
    _edition_rank,
    _parse_track_selection,
    _sorted_editions,
    category_label,
    source_from_candidate,
)


def choose_tracks(
    scan: ScanResult,
    config: AppConfig,
    *,
    edition_argument: str | None = None,
    include_bonus: bool = False,
) -> tuple[str, list[AudioSource], dict[str, Any]]:
    if not scan.editions:
        raise VideoPreparerError("没有找到可处理的音频文件。")

    editions = sorted(
        scan.editions,
        key=lambda item: _edition_rank(item, config),
        reverse=True,
    )
    recommended = editions[0]
    chosen: Edition | None = None

    if edition_argument:
        value = edition_argument.strip()
        if value.isdigit() and 1 <= int(value) <= len(editions):
            chosen = editions[int(value) - 1]
        else:
            chosen = next((item for item in editions if item.id == value), None)
        if chosen is None:
            raise VideoPreparerError(f"找不到指定版本：{edition_argument}")
    elif len(editions) == 1:
        chosen = editions[0]
        print(f"\n只发现一组可处理音轨：{chosen.label}")
    elif recommended.legacy_compatible:
        chosen = recommended
        print("\n检测到传统的根目录数字音轨，将直接沿用原来的处理顺序。")
    else:
        print("\n检测到多组音频，请选择要处理的版本：")
        for index, edition in enumerate(editions, start=1):
            marker = "（推荐）" if edition.id == recommended.id else ""
            optional_text = (
                f"，另有 {len(edition.optional_tracks)} 个特典/样本"
                if edition.optional_tracks
                else ""
            )
            print(f"  {index}. {edition.label}：{len(edition.tracks)} 轨{optional_text}{marker}")
        print("  M. 手动逐条选择")
        answer = input(f"输入编号；直接按 Enter 使用推荐的 {1}：").strip().casefold()
        if not answer:
            chosen = recommended
        elif answer == "m":
            all_tracks = _all_scan_tracks(scan)
            print("\n全部音频：")
            for index, track in enumerate(all_tracks, start=1):
                category = (
                    "" if track.category == "main" else f" [{category_label(track.category)}]"
                )
                print(f"  {index:>3}. {track.relative_path}{category}")
            raw = input("输入要处理的编号，例如 1,3-6：").strip()
            indexes = _parse_track_selection(raw, len(all_tracks))
            if not indexes:
                raise VideoPreparerError("没有选择任何音轨。")
            selected_candidates = [all_tracks[index - 1] for index in indexes]
            sources = [
                source_from_candidate(index, item)
                for index, item in enumerate(selected_candidates, start=1)
            ]
            return (
                "自定义音轨",
                sources,
                {
                    "edition_id": "custom",
                    "edition_label": "自定义音轨",
                    "manual": True,
                    "included_optional": any(item.is_optional for item in selected_candidates),
                },
            )
        elif answer.isdigit() and 1 <= int(answer) <= len(editions):
            chosen = editions[int(answer) - 1]
        else:
            raise VideoPreparerError("版本选择无效。")

    assert chosen is not None
    candidates = list(chosen.tracks)
    chosen_paths = {item.path.resolve() for item in candidates}
    # Optional material is often placed in a separate ``特典``/``Bonus``
    # directory, so it does not share the main edition's exact grouping key.
    # Offer compatible optional files from the whole scan as well, while
    # deduplicating WAV/MP3 mirrors of the same title.
    global_optional = [
        item
        for item in _all_scan_tracks(scan)
        if item.is_optional
        and item.path.resolve() not in chosen_paths
        and item.language == chosen.language
        and item.orientation == chosen.orientation
        and item.mix_variant in {chosen.mix_variant, "standard"}
    ]
    if chosen.extension != ".mixed":
        same_format = [item for item in global_optional if item.extension == chosen.extension]
        if same_format:
            global_optional = same_format
    optional_pool = [*chosen.optional_tracks, *global_optional]
    unique_optional: dict[Path, TrackCandidate] = {}
    for item in optional_pool:
        unique_optional.setdefault(item.path.resolve(), item)

    def optional_format_rank(candidate: TrackCandidate) -> tuple[int, str]:
        try:
            rank = config.preferred_audio_formats.index(candidate.extension)
        except ValueError:
            rank = len(config.preferred_audio_formats) + 1
        return rank, candidate.relative_path.casefold()

    optional_by_title: dict[tuple[str, int, int, str, str], TrackCandidate] = {}
    for item in unique_optional.values():
        section, number, suffix, _ = item.order_key
        key = (item.category, section, number, suffix, item.title.casefold().strip())
        current = optional_by_title.get(key)
        if current is None:
            optional_by_title[key] = item
            continue
        if optional_format_rank(item) < optional_format_rank(current):
            optional_by_title[key] = item
    optional_pool = list(optional_by_title.values())
    include_optional = include_bonus
    if optional_pool and not include_optional:
        if config.bonus_policy == "include":
            include_optional = True
        elif config.bonus_policy == "ask" and edition_argument is None:
            print("\n这一版本还包含：")
            counts: dict[str, int] = {}
            for item in optional_pool:
                counts[item.category] = counts.get(item.category, 0) + 1
            print(
                "  "
                + "，".join(f"{category_label(name)} {count} 轨" for name, count in counts.items())
            )
            include_optional = ask_yes_no("是否包含这些附加音轨？输入 Y 包含，输入 N 不包含：")
    if include_optional:
        candidates.extend(optional_pool)
        candidates.sort(key=lambda item: (item.order_key, natural_key(item.relative_path)))
    sources = [source_from_candidate(index, item) for index, item in enumerate(candidates, start=1)]
    return (
        chosen.label,
        sources,
        {
            "edition_id": chosen.id,
            "edition_label": chosen.label,
            "manual": False,
            "included_optional": include_optional,
            "directory": chosen.directory,
            "extension": chosen.extension,
            "language": chosen.language,
            "mix_variant": chosen.mix_variant,
            "orientation": chosen.orientation,
        },
    )


def tracks_for_edition(
    scan: ScanResult,
    config: AppConfig,
    edition: Edition,
    *,
    include_bonus: bool,
) -> tuple[str, list[AudioSource], dict[str, Any]]:
    return choose_tracks(
        scan,
        replace(config, bonus_policy="exclude"),
        edition_argument=edition.id,
        include_bonus=include_bonus,
    )


def ask_tracks_for_draft(
    scan: ScanResult,
    config: AppConfig,
    *,
    include_bonus: bool,
) -> tuple[str, list[AudioSource], dict[str, Any]]:
    editions = _sorted_editions(scan, config)
    if not editions:
        raise VideoPreparerError("没有找到可处理的音频文件。")
    print("\n请选择要处理的音频版本：")
    for index, edition in enumerate(editions, start=1):
        marker = "（推荐）" if index == 1 else ""
        optional_text = (
            f"，另有 {len(edition.optional_tracks)} 个特典/样本" if edition.optional_tracks else ""
        )
        print(f"  {index}. {edition.label}：{len(edition.tracks)} 轨{optional_text}{marker}")
    print("  M. 手动逐条选择")
    while True:
        answer = input("输入编号；直接按 Enter 使用推荐版本：").strip().casefold()
        if not answer:
            return tracks_for_edition(
                scan,
                config,
                editions[0],
                include_bonus=include_bonus,
            )
        if answer.isdigit() and 1 <= int(answer) <= len(editions):
            return tracks_for_edition(
                scan,
                config,
                editions[int(answer) - 1],
                include_bonus=include_bonus,
            )
        if answer != "m":
            print("输入无效，请输入版本编号或 M。")
            continue

        all_tracks = _all_scan_tracks(scan)
        print("\n全部音频：")
        for index, track in enumerate(all_tracks, start=1):
            category = "" if track.category == "main" else f" [{category_label(track.category)}]"
            print(f"  {index:>3}. {track.relative_path}{category}")
        while True:
            try:
                indexes = _parse_track_selection(
                    input("输入要处理的编号，例如 1,3-6：").strip(),
                    len(all_tracks),
                )
            except VideoPreparerError as exc:
                print(f"输入无效：{exc}")
                continue
            if not indexes:
                print("没有选择任何音轨，请重新输入。")
                continue
            break
        selected_candidates = [all_tracks[index - 1] for index in indexes]
        sources = [
            source_from_candidate(index, item)
            for index, item in enumerate(selected_candidates, start=1)
        ]
        return (
            "自定义音轨",
            sources,
            {
                "edition_id": "custom",
                "edition_label": "自定义音轨",
                "manual": True,
                "included_optional": any(item.is_optional for item in selected_candidates),
            },
        )


def update_draft_edition(draft: SmartPlanDraft, config: AppConfig) -> None:
    label, sources, edition = ask_tracks_for_draft(
        draft.scan,
        config,
        include_bonus=draft.include_bonus,
    )
    draft.edition_label = label
    draft.sources = sources
    draft.edition = edition
    if bool(edition.get("manual")):
        draft.include_bonus = bool(edition.get("included_optional"))


def update_draft_bonus(draft: SmartPlanDraft, config: AppConfig) -> None:
    if bool(draft.edition.get("manual")):
        print("自定义音轨中的特典由第 1 项决定；请重新选择音频版本或音轨。")
        return
    include_bonus = ask_yes_no("输入 Y 包含附加音轨，输入 N 不包含：")
    edition_id = str(draft.edition.get("edition_id") or "")
    label, sources, edition = choose_tracks(
        draft.scan,
        replace(config, bonus_policy="exclude"),
        edition_argument=edition_id,
        include_bonus=include_bonus,
    )
    draft.edition_label = label
    draft.sources = sources
    draft.edition = edition
    draft.include_bonus = include_bonus
