"""Plan option helpers with non-interactive defaults and injectable input reader."""

from __future__ import annotations

from pathlib import Path

from . import constants, runlog
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
from .errors import VideoPreparerError
from .modes import (
    layout_label,
    media_type_label,
    normalize_layout,
    normalize_mode,
    parse_embed_subtitles_argument,
    video_branch_label,
)
from .runlog import InputReader
from .source_scan import (
    background_draft_label,
)
from .transcript_choices import transcript_choice_label, transcript_key, transcript_rows_for_draft
from .work_state import expected_output_paths


def print_header() -> None:
    runlog.emit("")
    runlog.emit("=" * 68)
    runlog.emit("  ASMR-Dubber AutoFlow · 音频 / 静态视频 / 双语制作")
    runlog.emit("=" * 68)
    runlog.emit(f"日志：{constants.LOG_FILE}")


def parse_yes_no(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized == "y":
        return True
    if normalized == "n":
        return False
    return None


def ask_yes_no(
    prompt: str,
    *,
    default: bool = False,
    input_reader: InputReader | None = None,
) -> bool:
    """Ask an explicit Y/N question via ``input_reader``, or return ``default``."""

    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return default
    while True:
        answer = parse_yes_no(reader(prompt))
        if answer is not None:
            return answer
        runlog.emit("输入无效，请输入 Y 或 N。")


def ask_output_layout(
    config: AppConfig,
    argument: str | None = None,
    *,
    input_reader: InputReader | None = None,
) -> str:
    if argument:
        return normalize_layout(argument)
    if config.default_output_layout != "ask":
        return normalize_layout(config.default_output_layout)
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return LAYOUT_MERGED
    runlog.emit("\n请选择成品组织方式：")
    runlog.emit("  1. 合并成一部")
    runlog.emit("  2. 每条音轨分别输出（不拼接）")
    runlog.emit("  3. 分轨输出 + 合并版")
    while True:
        answer = reader("输入 1、2 或 3：").strip()
        if answer == "1":
            return LAYOUT_MERGED
        if answer == "2":
            return LAYOUT_SEPARATE
        if answer == "3":
            return LAYOUT_BOTH
        runlog.emit("输入无效，请重新选择。")


def ask_embed_subtitles(
    mode: str,
    argument: str | None = None,
    *,
    input_reader: InputReader | None = None,
) -> bool:
    if normalize_mode(mode) == MODE_AUDIO:
        return False
    if argument is not None:
        return parse_embed_subtitles_argument(argument)
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return True
    runlog.emit("\n是否把字幕放进最终视频？")
    runlog.emit("  无论如何都会另外保留双语版.srt 和双语版.lrc。")
    return ask_yes_no("输入 Y 内嵌字幕，输入 N 仅保留外部字幕：", default=True, input_reader=reader)


def ask_mode(config: AppConfig, *, input_reader: InputReader | None = None) -> str:
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return MODE_VIDEO_NORMAL
    while True:
        runlog.emit("\n请选择处理类型：")
        runlog.emit("  1. 纯音频模式（不制作视频，输出音频与字幕）")
        runlog.emit("  2. 静态视频模式（下一步选择普通或和谐）")
        answer = reader("输入 1 或 2：").strip()
        if answer == "1":
            return MODE_AUDIO
        if answer == "2":
            while True:
                runlog.emit("\n请选择视频分支：")
                runlog.emit("  1. 普通模式（静态背景 + 原音量）")
                runlog.emit(
                    f"  2. 和谐模式（成品音量 {config.harmonized_volume_db:g} dB，"
                    f"视频与字幕延后 {config.harmonized_delay_seconds / 60:g} 分钟）"
                )
                video_answer = reader("输入 1 或 2；输入 B 返回：").strip().casefold()
                if video_answer == "1":
                    return MODE_VIDEO_NORMAL
                if video_answer == "2":
                    return MODE_VIDEO_HARMONIZED
                if video_answer == "b":
                    break
                runlog.emit("输入无效，请重新选择。")
            continue
        runlog.emit("输入无效，请重新选择。")


def confirm_overwrite(
    folder: Path,
    mode: str,
    *,
    force: bool,
    input_reader: InputReader | None = None,
) -> None:
    existing = [path for path in expected_output_paths(folder, mode) if path.exists()]
    if not existing or force:
        return
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        names = "、".join(path.name for path in existing)
        raise VideoPreparerError(
            f"目标目录已有成品文件（{names}）；如需替换已有成品，请传入 --force 参数。"
        )
    runlog.emit("\n以下旧文件将在相应新文件完整生成后被替换：")
    for path in existing:
        runlog.emit(f"  {path.name}")
    answer = reader("继续吗？输入 Y 确认：").strip().casefold()
    if answer != "y":
        raise KeyboardInterrupt


def configure_draft_transcripts(
    draft: SmartPlanDraft,
    *,
    input_reader: InputReader | None = None,
) -> None:
    rows = transcript_rows_for_draft(draft)
    if not rows:
        runlog.emit("\n没有发现可识别的台本或字幕文件。")
        return
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return
    runlog.emit("\n台本/字幕类型选择：")
    runlog.emit("  默认：中文台本/字幕（LRC 通常是中文翻译）")
    runlog.emit("  每份文件可选：中文、日文、英文、不使用、按文件名自动判断")
    while True:
        runlog.emit("")
        for index, (transcript, paired_tracks) in enumerate(rows, start=1):
            pairing = (
                "匹配：" + "、".join(paired_tracks) if paired_tracks else "未匹配到当前选中的音轨"
            )
            runlog.emit(
                f"  {index}. {transcript.relative_path} · "
                f"{transcript_choice_label(draft, transcript)} · {pairing}"
            )
        runlog.emit("  0. 返回作品设置")
        answer = reader("输入编号修改这份台本，或输入 0 返回：").strip()
        if answer == "0":
            return
        if not answer.isdigit() or not 1 <= int(answer) <= len(rows):
            runlog.emit("输入无效，请输入台本编号或 0。")
            continue
        transcript, _ = rows[int(answer) - 1]
        runlog.emit(
            f"\n{transcript.relative_path} 当前：{transcript_choice_label(draft, transcript)}"
        )
        runlog.emit("  1. 中文台本/字幕（默认）")
        runlog.emit("  2. 日文台本/字幕")
        runlog.emit("  3. 英文台本/字幕")
        runlog.emit("  4. 不使用这份台本/字幕")
        runlog.emit("  5. 按文件名自动判断")
        choice = reader("输入 1-5：").strip()
        language = {"1": "zh", "2": "ja", "3": "en", "4": "ignore", "5": "auto"}.get(choice)
        if language is None:
            runlog.emit("输入无效，请重新选择。")
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
    runlog.emit("\n" + "-" * 68)
    runlog.emit(f"作品 {index}：{draft.folder.name}")
    runlog.emit("当前默认选项（继承上一个作品，可输入编号修改）：")
    runlog.emit(f"  1. 音频版本：{draft.edition_label}（共 {len(draft.sources)} 轨）")
    runlog.emit(f"  2. 附加音轨：{bonus_label}")
    runlog.emit(f"  3. 输出类型：{media_type_label(draft.mode)}")
    runlog.emit(f"  4. 视频分支：{video_branch_label(draft.video_mode, config)}{video_suffix}")
    runlog.emit(f"  5. 成品组织：{layout_label(draft.layout)}")
    runlog.emit(f"  6. 视频画面：{background_draft_label(draft)}")
    subtitle_label = "内嵌，同时保留 SRT/LRC" if draft.embed_subtitles else "不内嵌，仅保留 SRT/LRC"
    runlog.emit(f"  7. 视频字幕：{subtitle_label}{video_suffix}")
    transcript_count = len(draft.scan.transcripts)
    transcript_default = "默认中文；可逐份修改"
    runlog.emit(f"  8. 台本/字幕：发现 {transcript_count} 份，{transcript_default}")
    runlog.emit(f"  输出目录：{draft.output_root}")
    runlog.emit("\n直接按 Enter：确认本作品并返回作品串")
    runlog.emit("输入 0：确认本作品，并立即开始处理全部任务")


def ask_media_type_for_draft(
    draft: SmartPlanDraft,
    *,
    input_reader: InputReader | None = None,
) -> None:
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return
    runlog.emit("\n请选择输出类型：")
    runlog.emit("  1. 纯音频")
    runlog.emit("  2. 静态视频")
    while True:
        answer = reader("输入 1 或 2：").strip()
        if answer == "1":
            draft.mode = MODE_AUDIO
            return
        if answer == "2":
            draft.mode = draft.video_mode
            return
        runlog.emit("输入无效，请重新选择。")


def ask_video_branch_for_draft(
    draft: SmartPlanDraft,
    config: AppConfig,
    *,
    input_reader: InputReader | None = None,
) -> None:
    reader = runlog.resolve_input_reader(input_reader)
    if reader is None:
        return
    runlog.emit("\n请选择视频分支：")
    runlog.emit("  1. 普通（原音量，无前置空档）")
    runlog.emit(
        f"  2. 和谐（{config.harmonized_volume_db:g} dB，"
        f"延后 {config.harmonized_delay_seconds / 60:g} 分钟）"
    )
    while True:
        answer = reader("输入 1 或 2：").strip()
        if answer == "1":
            draft.video_mode = MODE_VIDEO_NORMAL
            break
        if answer == "2":
            draft.video_mode = MODE_VIDEO_HARMONIZED
            break
        runlog.emit("输入无效，请重新选择。")
    if draft.mode != MODE_AUDIO:
        draft.mode = draft.video_mode
