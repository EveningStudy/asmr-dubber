"""Command line entry point for AutoFlow."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence

from . import app_settings, runlog, smart_queue
from .app_settings import load_app_config
from .errors import VideoPreparerError
from .fileio import clean_user_path
from .interactive_queue import collect_interactive_smart_plans
from .plan_run import execute_smart_plan
from .planned_run import prepare_or_resume
from .prompts import print_header
from .runlog import InputReader, append_log
from .self_test import self_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASMR-Dubber AutoFlow 音频与静态视频双语制作工作流"
    )
    parser.add_argument("folder", nargs="?", help="解压后的作品文件夹")
    parser.add_argument(
        "--scan",
        choices=("smart", "legacy"),
        default="smart",
        help="smart=递归识别 DLsite 作品目录；legacy=兼容旧版根目录数字音轨流程",
    )
    parser.add_argument(
        "--mode",
        choices=("audio", "video-normal", "video-harmonized", "normal", "harmonized"),
        help="audio=纯音频；video-normal=普通视频；video-harmonized=和谐视频",
    )
    parser.add_argument("--rebuild", action="store_true", help="丢弃此文件夹的旧任务状态并从头开始")
    parser.add_argument("--force", action="store_true", help="不询问是否替换已有成品")
    parser.add_argument(
        "--layout",
        choices=("merged", "separate", "both"),
        help="smart 扫描的成品组织：merged=合并、separate=分轨、both=分轨+合并",
    )
    parser.add_argument("--edition", help="smart 扫描时指定版本编号或处理清单中的版本 ID")
    parser.add_argument(
        "--include-bonus", action="store_true", help="smart 扫描时把特典/样本也加入任务"
    )
    parser.add_argument(
        "--output-root", help="smart 扫描的输出目录；默认是源文件夹下的 AutoFlow输出"
    )
    parser.add_argument(
        "--background",
        help=("视频背景：图片编号、作品目录内的相对/绝对路径、auto，或 black；未指定时按默认策略"),
    )
    parser.add_argument(
        "--embed-subtitles",
        choices=("yes", "no"),
        help="视频是否内嵌字幕；无论如何都会保留外部 SRT/LRC",
    )
    parser.add_argument("--self-test", action="store_true", help="只运行几秒钟的本地媒体自检")
    return parser


def validate_interactive_queue_arguments(args: argparse.Namespace) -> None:
    """Reject per-work overrides that cannot safely be shared by an ad-hoc queue."""

    unsupported: list[str] = []
    if args.edition:
        unsupported.append("--edition")
    if args.output_root:
        unsupported.append("--output-root")
    if args.background:
        unsupported.append("--background")
    if unsupported:
        joined = "、".join(unsupported)
        raise VideoPreparerError(
            f"多作品交互队列不支持 {joined}；这些参数只适合指定单个文件夹时使用。"
        )


def _write_stdout(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _write_stderr(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def main(argv: Sequence[str] | None = None, *, input_reader: InputReader | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reader = runlog.resolve_input_reader(input_reader)
    with runlog.bind_autoflow_callbacks(
        on_message=_write_stdout,
        on_error=_write_stderr,
        input_reader=reader,
    ):
        print_header()
        try:
            config = load_app_config()
            paths = app_settings.find_tool_paths(config)
            app_settings.validate_asmr_version()
            if args.self_test:
                self_test(paths)
                return 0
            if args.scan == "legacy":
                if args.folder:
                    folder = clean_user_path(args.folder)
                elif reader is not None:
                    folder = clean_user_path(reader("请粘贴解压后的作品文件夹路径："))
                else:
                    raise VideoPreparerError(
                        "未指定作品文件夹路径；请传入 folder 参数"
                        "（例如 asmr-dubber autoflow <文件夹路径>），"
                        "或使用桌面界面的批量处理。"
                    )
                if (
                    args.layout
                    or args.edition
                    or args.include_bonus
                    or args.output_root
                    or args.background
                ):
                    raise VideoPreparerError(
                        "legacy 扫描不支持 --layout、--edition、--include-bonus、"
                        "--output-root 或 --background。"
                    )
                prepare_or_resume(
                    paths,
                    config,
                    folder,
                    args.mode,
                    args.embed_subtitles,
                    rebuild=args.rebuild,
                    force=args.force,
                    input_reader=reader,
                )
            elif args.folder:
                folder = clean_user_path(args.folder)
                execute_smart_plan(
                    paths,
                    config,
                    folder,
                    mode_argument=args.mode,
                    layout_argument=args.layout,
                    edition_argument=args.edition,
                    include_bonus=args.include_bonus,
                    output_root_argument=args.output_root,
                    background_argument=args.background,
                    embed_subtitles_argument=args.embed_subtitles,
                    rebuild=args.rebuild,
                    force=args.force,
                    input_reader=reader,
                )
            else:
                validate_interactive_queue_arguments(args)
                plans = collect_interactive_smart_plans(config, args, input_reader=reader)
                return smart_queue.execute_smart_queue(paths, config, plans)
            return 0
        except KeyboardInterrupt:
            runlog.log_event("用户取消任务")
            runlog.emit("\n已取消。完整成品不会被半成品覆盖，已有任务状态会保留。")
            return 130
        except VideoPreparerError as exc:
            runlog.log_event(f"操作失败：{exc}")
            runlog.emit(f"\n操作失败：{exc}", error=True)
            return 1
        except Exception as exc:
            runlog.log_event(f"未预期错误：{type(exc).__name__}: {exc}")
            append_log(traceback.format_exc())
            runlog.emit(f"\n未预期错误：{type(exc).__name__}: {exc}", error=True)
            return 1
