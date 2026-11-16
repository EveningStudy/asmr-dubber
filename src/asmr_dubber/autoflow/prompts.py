"""Interactive terminal prompts for building plans."""

from __future__ import annotations

from pathlib import Path

from . import constants
from .constants import (
    LAYOUT_BOTH,
    LAYOUT_MERGED,
    LAYOUT_SEPARATE,
    MODE_AUDIO,
    MODE_VIDEO_HARMONIZED,
    MODE_VIDEO_NORMAL,
)
from .domain import (
    AppConfig,
    SmartPlanDraft,
)
from .modes import (
    layout_label,
    media_type_label,
    normalize_layout,
    normalize_mode,
    parse_embed_subtitles_argument,
    video_branch_label,
)
from .source_scan import (
    background_draft_label,
)
from .transcript_choices import transcript_choice_label, transcript_key, transcript_rows_for_draft
from .work_state import expected_output_paths


def print_header() -> None:
    print()
    print("=" * 68)
    print("  ASMR-Dubber AutoFlow · 音频 / 静态视频 / 双语制作")
    print("=" * 68)
    print(f"日志：{constants.LOG_FILE}")


def parse_yes_no(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized == "y":
        return True
    if normalized == "n":
        return False
    return None


def ask_yes_no(prompt: str) -> bool:
    """Ask an explicit question and accept only Y or N."""

    while True:
        answer = parse_yes_no(input(prompt))
        if answer is not None:
            return answer
        print("输入无效，请输入 Y 或 N。")


def ask_output_layout(config: AppConfig, argument: str | None = None) -> str:
    if argument:
        return normalize_layout(argument)
    if config.default_output_layout != "ask":
        return normalize_layout(config.default_output_layout)
    print("\n请选择成品组织方式：")
    print("  1. 合并成一部")
    print("  2. 每条音轨分别输出（不拼接）")
    print("  3. 分轨输出 + 合并版")
    while True:
        answer = input("输入 1、2 或 3：").strip()
        if answer == "1":
            return LAYOUT_MERGED
        if answer == "2":
            return LAYOUT_SEPARATE
        if answer == "3":
            return LAYOUT_BOTH
        print("输入无效，请重新选择。")


def ask_embed_subtitles(mode: str, argument: str | None = None) -> bool:
    if normalize_mode(mode) == MODE_AUDIO:
        return False
    if argument is not None:
        return parse_embed_subtitles_argument(argument)
    print("\n是否把字幕放进最终视频？")
    print("  无论如何都会另外保留双语版.srt 和双语版.lrc。")
    return ask_yes_no("输入 Y 内嵌字幕，输入 N 仅保留外部字幕：")


def ask_mode(config: AppConfig) -> str:
    while True:
        print("\n请选择处理类型：")
        print("  1. 纯音频模式（不制作视频，输出音频与字幕）")
        print("  2. 静态视频模式（下一步选择普通或和谐）")
        answer = input("输入 1 或 2：").strip()
        if answer == "1":
            return MODE_AUDIO
        if answer == "2":
            while True:
                print("\n请选择视频分支：")
                print("  1. 普通模式（静态背景 + 原音量）")
                print(
                    f"  2. 和谐模式（成品音量 {config.harmonized_volume_db:g} dB，"
                    f"视频与字幕延后 {config.harmonized_delay_seconds / 60:g} 分钟）"
                )
                video_answer = input("输入 1 或 2；输入 B 返回：").strip().casefold()
                if video_answer == "1":
                    return MODE_VIDEO_NORMAL
                if video_answer == "2":
                    return MODE_VIDEO_HARMONIZED
                if video_answer == "b":
                    break
                print("输入无效，请重新选择。")
            continue
        print("输入无效，请重新选择。")


def confirm_overwrite(folder: Path, mode: str, *, force: bool) -> None:
    existing = [path for path in expected_output_paths(folder, mode) if path.exists()]
    if not existing or force:
        return
    print("\n以下旧文件将在相应新文件完整生成后被替换：")
    for path in existing:
        print(f"  {path.name}")
    answer = input("继续吗？输入 Y 确认：").strip().casefold()
    if answer != "y":
        raise KeyboardInterrupt


def configure_draft_transcripts(draft: SmartPlanDraft) -> None:
    rows = transcript_rows_for_draft(draft)
    if not rows:
        print("\n没有发现可识别的台本或字幕文件。")
        return
    print("\n台本/字幕类型选择：")
    print("  默认：中文台本/字幕（LRC 通常是中文翻译）")
    print("  每份文件可选：中文、日文、英文、不使用、按文件名自动判断")
    while True:
        print()
        for index, (transcript, paired_tracks) in enumerate(rows, start=1):
            pairing = (
                "匹配：" + "、".join(paired_tracks) if paired_tracks else "未匹配到当前选中的音轨"
            )
            print(
                f"  {index}. {transcript.relative_path} · "
                f"{transcript_choice_label(draft, transcript)} · {pairing}"
            )
        print("  0. 返回作品设置")
        answer = input("输入编号修改这份台本，或输入 0 返回：").strip()
        if answer == "0":
            return
        if not answer.isdigit() or not 1 <= int(answer) <= len(rows):
            print("输入无效，请输入台本编号或 0。")
            continue
        transcript, _ = rows[int(answer) - 1]
        print(f"\n{transcript.relative_path} 当前：{transcript_choice_label(draft, transcript)}")
        print("  1. 中文台本/字幕（默认）")
        print("  2. 日文台本/字幕")
        print("  3. 英文台本/字幕")
        print("  4. 不使用这份台本/字幕")
        print("  5. 按文件名自动判断")
        choice = input("输入 1-5：").strip()
        language = {"1": "zh", "2": "ja", "3": "en", "4": "ignore", "5": "auto"}.get(choice)
        if language is None:
            print("输入无效，请重新选择。")
            continue
        draft.transcript_choices[transcript_key(transcript.path)] = language


def print_smart_plan_defaults_panel(
    draft: SmartPlanDraft,
    config: AppConfig,
    *,
    index: int,
) -> None:
    optional_count = sum(item.category != "main" for item in draft.sources)
    if bool(draft.edition.get("manual")):
        bonus_label = "由自定义音轨决定"
    else:
        bonus_label = "包含" if draft.include_bonus else "不包含"
        if optional_count:
            bonus_label += f"（当前 {optional_count} 轨）"
    video_suffix = "（当前不适用）" if draft.mode == MODE_AUDIO else ""
    print("\n" + "-" * 68)
    print(f"作品 {index}：{draft.folder.name}")
    print("当前默认选项（继承上一个作品，可输入编号修改）：")
    print(f"  1. 音频版本：{draft.edition_label}（共 {len(draft.sources)} 轨）")
    print(f"  2. 附加音轨：{bonus_label}")
    print(f"  3. 输出类型：{media_type_label(draft.mode)}")
    print(f"  4. 视频分支：{video_branch_label(draft.video_mode, config)}{video_suffix}")
    print(f"  5. 成品组织：{layout_label(draft.layout)}")
    print(f"  6. 视频画面：{background_draft_label(draft)}")
    subtitle_label = "内嵌，同时保留 SRT/LRC" if draft.embed_subtitles else "不内嵌，仅保留 SRT/LRC"
    print(f"  7. 视频字幕：{subtitle_label}{video_suffix}")
    transcript_count = len(draft.scan.transcripts)
    transcript_default = "默认中文；可逐份修改"
    print(f"  8. 台本/字幕：发现 {transcript_count} 份，{transcript_default}")
    print(f"  输出目录：{draft.output_root}")
    print("\n直接按 Enter：确认本作品并返回作品串")
    print("输入 0：确认本作品，并立即开始处理全部任务")


def ask_media_type_for_draft(draft: SmartPlanDraft) -> None:
    print("\n请选择输出类型：")
    print("  1. 纯音频")
    print("  2. 静态视频")
    while True:
        answer = input("输入 1 或 2：").strip()
        if answer == "1":
            draft.mode = MODE_AUDIO
            return
        if answer == "2":
            draft.mode = draft.video_mode
            return
        print("输入无效，请重新选择。")


def ask_video_branch_for_draft(draft: SmartPlanDraft, config: AppConfig) -> None:
    print("\n请选择视频分支：")
    print("  1. 普通（原音量，无前置空档）")
    print(
        f"  2. 和谐（{config.harmonized_volume_db:g} dB，"
        f"延后 {config.harmonized_delay_seconds / 60:g} 分钟）"
    )
    while True:
        answer = input("输入 1 或 2：").strip()
        if answer == "1":
            draft.video_mode = MODE_VIDEO_NORMAL
            break
        if answer == "2":
            draft.video_mode = MODE_VIDEO_HARMONIZED
            break
        print("输入无效，请重新选择。")
    if draft.mode != MODE_AUDIO:
        draft.mode = draft.video_mode
