"""Interactive plan configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .catalog import (
    scan_work,
)
from .constants import MODE_AUDIO
from .domain import (
    AppConfig,
    SmartPlanDefaults,
    SmartPlanDraft,
    SmartTaskPlan,
)
from .errors import VideoPreparerError
from .fileio import clean_user_path
from .modes import layout_label, mode_label, normalize_mode
from .plan_files import plan_identity
from .prompts import (
    ask_embed_subtitles,
    ask_media_type_for_draft,
    ask_mode,
    ask_output_layout,
    ask_video_branch_for_draft,
    ask_yes_no,
    configure_draft_transcripts,
    print_smart_plan_defaults_panel,
)
from .source_scan import background_selection_details, smart_background
from .track_prompts import choose_tracks, update_draft_bonus, update_draft_edition
from .transcript_choices import effective_sources_for_draft


def configure_smart_plan_draft(
    draft: SmartPlanDraft,
    config: AppConfig,
    *,
    index: int,
    rebuild: bool,
    force: bool,
) -> tuple[SmartTaskPlan, SmartPlanDefaults, bool]:
    """Return the plan, inherited defaults, and whether execution should start."""

    while True:
        print_smart_plan_defaults_panel(draft, config, index=index)
        answer = input("输入 1-8 修改；直接按 Enter 返回作品串；输入 0 开始：").strip()
        if answer in {"", "0"}:
            break
        if answer == "1":
            update_draft_edition(draft, config)
        elif answer == "2":
            update_draft_bonus(draft, config)
        elif answer == "3":
            ask_media_type_for_draft(draft)
        elif answer == "4":
            ask_video_branch_for_draft(draft, config)
        elif answer == "5":
            draft.layout = ask_output_layout(replace(config, default_output_layout="ask"))
        elif answer == "6":
            selected = smart_background(
                draft.scan,
                replace(config, background_policy="ask"),
            )
            (
                draft.background,
                draft.background_choice,
                draft.background_relative,
            ) = background_selection_details(draft.scan, selected)
        elif answer == "7":
            print("\n无论如何都会另外保留双语版.srt 和双语版.lrc。")
            draft.embed_subtitles = ask_yes_no("输入 Y 内嵌字幕，输入 N 仅保留外部字幕：")
        elif answer == "8":
            configure_draft_transcripts(draft)
        else:
            print("输入无效，请输入 1-8、0，或直接按 Enter。")

    mode = normalize_mode(draft.mode)
    plan_background = draft.background if mode != MODE_AUDIO else None
    plan_embed_subtitles = draft.embed_subtitles if mode != MODE_AUDIO else False
    effective_sources = effective_sources_for_draft(draft)
    plan_id = plan_identity(
        draft.folder,
        mode=mode,
        layout=draft.layout,
        edition=draft.edition,
        sources=effective_sources,
        output_root=draft.output_root,
        background=plan_background,
        embed_subtitles=plan_embed_subtitles,
    )
    plan = SmartTaskPlan(
        folder=draft.folder,
        output_root=draft.output_root,
        edition_label=draft.edition_label,
        sources=tuple(effective_sources),
        edition=draft.edition,
        mode=mode,
        layout=draft.layout,
        background=plan_background,
        embed_subtitles=plan_embed_subtitles,
        plan_id=plan_id,
        rebuild=rebuild,
        force=force,
    )
    edition_extension = None
    if not bool(draft.edition.get("manual")):
        edition_extension = str(draft.edition.get("extension") or "").strip() or None
    defaults = SmartPlanDefaults(
        mode=mode,
        video_mode=draft.video_mode,
        layout=draft.layout,
        include_bonus=draft.include_bonus,
        background_choice=draft.background_choice,
        background_relative=draft.background_relative,
        embed_subtitles=draft.embed_subtitles,
        edition_extension=edition_extension,
        edition_language=str(draft.edition.get("language") or "").strip() or None,
        edition_mix_variant=str(draft.edition.get("mix_variant") or "").strip() or None,
        edition_orientation=str(draft.edition.get("orientation") or "").strip() or None,
    )
    return plan, defaults, answer == "0"


def prepare_smart_plan(
    config: AppConfig,
    folder: Path,
    *,
    mode_argument: str | None,
    layout_argument: str | None,
    edition_argument: str | None,
    include_bonus: bool,
    output_root_argument: str | None,
    background_argument: str | None,
    embed_subtitles_argument: str | None,
    rebuild: bool,
    force: bool,
) -> SmartTaskPlan:
    """Scan and configure a DLsite task without running models or translating."""

    if not folder.is_dir():
        raise VideoPreparerError(f"文件夹不存在：{folder}")
    output_root = (
        clean_user_path(output_root_argument)
        if output_root_argument
        else (folder / config.output_folder_name).resolve()
    )
    if output_root == folder.resolve():
        raise VideoPreparerError("输出目录不能就是源作品文件夹；请使用 AutoFlow输出 子目录。")

    scan = scan_work(
        folder,
        excluded_directories=(config.output_folder_name, output_root.name),
    )
    edition_label, sources, edition = choose_tracks(
        scan,
        config,
        edition_argument=edition_argument,
        include_bonus=include_bonus,
    )
    if not sources:
        raise VideoPreparerError("选中的版本没有可处理的音轨。")
    mode = normalize_mode(mode_argument) if mode_argument else ask_mode(config)
    layout = ask_output_layout(config, layout_argument)
    background = smart_background(scan, config, background_argument) if mode != MODE_AUDIO else None
    embed_subtitles = ask_embed_subtitles(mode, embed_subtitles_argument)
    plan_id = plan_identity(
        folder,
        mode=mode,
        layout=layout,
        edition=edition,
        sources=sources,
        output_root=output_root,
        background=background,
        embed_subtitles=embed_subtitles,
    )
    return SmartTaskPlan(
        folder=folder.resolve(),
        output_root=output_root,
        edition_label=edition_label,
        sources=tuple(sources),
        edition=edition,
        mode=mode,
        layout=layout,
        background=background,
        embed_subtitles=embed_subtitles,
        plan_id=plan_id,
        rebuild=rebuild,
        force=force,
    )


def print_smart_plan_summary(plan: SmartTaskPlan, *, index: int | None = None) -> None:
    heading = f"作品 {index}" if index is not None else "本作品"
    print(f"\n{heading}已配置：{plan.folder.name}")
    print(f"  音频版本：{plan.edition_label}（{len(plan.sources)} 轨）")
    print(f"  处理类型：{mode_label(plan.mode)}")
    print(f"  处理内容：{'仅生成字幕' if plan.subtitles_only else '中文配音和字幕'}")
    print(f"  成品组织：{layout_label(plan.layout)}")
    if plan.mode != MODE_AUDIO:
        print(f"  背景图片：{plan.background.name if plan.background else '黑色背景'}")
        subtitle_mode = (
            "原声视频带字幕，同时保留 SRT/LRC"
            if plan.subtitles_only and plan.embed_subtitles
            else "原声视频不带字幕，仅保留 SRT/LRC"
            if plan.subtitles_only
            else "内嵌，同时保留 SRT/LRC"
            if plan.embed_subtitles
            else "不内嵌，仅保留 SRT/LRC"
        )
        print(f"  视频字幕：{subtitle_mode}")
    print(f"  输出目录：{plan.output_root}")
