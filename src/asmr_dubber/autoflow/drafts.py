"""Building a plan draft from a scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import (
    scan_work,
)
from .constants import MODE_VIDEO_HARMONIZED, MODE_VIDEO_NORMAL
from .domain import (
    AppConfig,
    SmartPlanDefaults,
    SmartPlanDraft,
)
from .errors import VideoPreparerError
from .modes import normalize_layout, normalize_mode, parse_embed_subtitles_argument
from .source_scan import default_edition_for_work, resolve_inherited_background
from .track_prompts import tracks_for_edition
from .transcript_choices import default_transcript_choices


def prepare_smart_plan_draft(
    config: AppConfig,
    folder: Path,
    defaults: SmartPlanDefaults,
    args: argparse.Namespace,
) -> SmartPlanDraft:
    if not folder.is_dir():
        raise VideoPreparerError(f"文件夹不存在：{folder}")
    output_root = (folder / config.output_folder_name).resolve()
    scan = scan_work(
        folder,
        excluded_directories=(config.output_folder_name, output_root.name),
    )
    edition = default_edition_for_work(scan, config, defaults)
    include_bonus = bool(args.include_bonus or defaults.include_bonus)
    edition_label, sources, edition_metadata = tracks_for_edition(
        scan,
        config,
        edition,
        include_bonus=include_bonus,
    )
    mode = normalize_mode(args.mode) if args.mode else normalize_mode(defaults.mode)
    video_mode = (
        mode
        if mode in {MODE_VIDEO_NORMAL, MODE_VIDEO_HARMONIZED}
        else normalize_mode(defaults.video_mode)
    )
    layout = normalize_layout(args.layout) if args.layout else normalize_layout(defaults.layout)
    background, background_choice, background_relative = resolve_inherited_background(
        scan,
        defaults,
    )
    embed_subtitles = (
        parse_embed_subtitles_argument(args.embed_subtitles)
        if args.embed_subtitles is not None
        else defaults.embed_subtitles
    )
    return SmartPlanDraft(
        folder=folder.resolve(),
        output_root=output_root,
        scan=scan,
        edition_label=edition_label,
        sources=sources,
        edition=edition_metadata,
        include_bonus=include_bonus,
        mode=mode,
        video_mode=video_mode,
        layout=layout,
        background=background,
        background_choice=background_choice,
        background_relative=background_relative,
        embed_subtitles=embed_subtitles,
        transcript_choices=default_transcript_choices(scan),
    )
