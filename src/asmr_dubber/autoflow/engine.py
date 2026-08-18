from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

from ..errors import OperationCancelledError, ProjectError
from ..platforms import portable_home
from ..task_control import (
    check_cancelled,
    register_process,
    unregister_process,
)
from ..task_control import (
    terminate_process_tree as _terminate_process_tree,
)
from .catalog import (
    Edition,
    ScanResult,
    TrackCandidate,
    natural_key,
    scan_work,
)

# AutoFlow is part of the main application now.  Its resumable state remains
# below the portable application home, never beside package source files or in
# the user's original work directory.
TOOL_ROOT = portable_home() / "autoflow"
WORK_ROOT = TOOL_ROOT / ".work"
STATE_ROOT = TOOL_ROOT / ".state"
SETTINGS_FILE = TOOL_ROOT / "settings.txt"
LOG_FILE = STATE_ROOT / "autoflow.log"
LAST_OPTIONS_FILE = STATE_ROOT / "last-options.json"
FAILED_TASKS_FILE = STATE_ROOT / "failed-tasks.json"
LOG_MAX_BYTES = 8 * 1024 * 1024

DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB = 10.0
DEFAULT_HARMONIZED_DELAY_MINUTES = 20.0
SAMPLE_RATE = 48_000
VIDEO_SIZE = "1920x1080"
VIDEO_FILTER_SIZE = "1920:1080"
VIDEO_FPS = 5
KEYFRAME_INTERVAL_SECONDS = 10
REFERENCE_SELECTION_TIMEOUT_SECONDS = 5 * 60
TIMESTAMP_SCHEMA = 2
LAST_OPTIONS_SCHEMA = 1
FAILED_TASKS_SCHEMA = 1
PERIODIC_KEYFRAME_OPTIONS = (
    "-g",
    str(VIDEO_FPS * KEYFRAME_INTERVAL_SECONDS),
    "-force_key_frames",
    f"expr:gte(t,n_forced*{KEYFRAME_INTERVAL_SECONDS})",
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mka",
    ".m4b",
    ".ape",
}

ReferenceEventCallback = Callable[[dict[str, Any]], None]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
NUMBERED_NAME = re.compile(r"^\s*(\d+)(?:[\s._\-、]+)?(.*)$")

LAYOUT_MERGED = "merged"
LAYOUT_SEPARATE = "separate"
LAYOUT_BOTH = "both"
LAYOUT_ALIASES = {
    "merged": LAYOUT_MERGED,
    "merge": LAYOUT_MERGED,
    "separate": LAYOUT_SEPARATE,
    "split": LAYOUT_SEPARATE,
    "both": LAYOUT_BOTH,
}

MODE_AUDIO = "audio"
MODE_VIDEO_NORMAL = "video_normal"
MODE_VIDEO_HARMONIZED = "video_harmonized"
MODE_ALIASES = {
    "audio": MODE_AUDIO,
    "video-normal": MODE_VIDEO_NORMAL,
    "video-harmonized": MODE_VIDEO_HARMONIZED,
    MODE_VIDEO_NORMAL: MODE_VIDEO_NORMAL,
    MODE_VIDEO_HARMONIZED: MODE_VIDEO_HARMONIZED,
    # Compatibility with tasks and commands created by the personal version.
    "normal": MODE_VIDEO_NORMAL,
    "harmonized": MODE_VIDEO_HARMONIZED,
}

STATUS_ORDER = {
    "media_ready": 10,
    "project_created": 20,
    "analyzed": 30,
    "awaiting_reference": 40,
    "synthesized": 50,
    "mixed": 60,
    "subtitles_ready": 70,
    "outputs_ready": 80,
    "completed": 90,
}

TITLE_TRANSLATION_PROMPT = """你负责把日语作品文件夹名称和音频曲目标题翻译成自然、简洁的简体中文。
保留人物名、编号、括号、符号和作品专有名词，不要省略成人内容，不要解释。
如果输入只是 RJ、VJ、BJ 等作品编号，译文原样保留该编号。
每个输入 id 必须输出一项，顺序和 id 必须完全一致，译文不得为空。
只输出严格 JSON：
{"translations":[{"id":"title0001","zh":"中文标题"}]}"""

DEFAULT_TIMESTAMP_FOOTER = """双语音声制作器：BV1f43G6YEov
内嵌字幕和配音为本地AI生成。内容仅供参考。
仅供日语学习，有能力请购买正版支持。"""


class VideoPreparerError(ProjectError):
    pass


@dataclass(frozen=True)
class AppConfig:
    asmr_root: Path | None
    harmonized_volume_db: float
    harmonized_delay_seconds: int
    timestamp_footer: str
    output_folder_name: str = "AutoFlow输出"
    default_output_layout: str = "ask"
    preferred_audio_formats: tuple[str, ...] = (".wav", ".flac", ".ape", ".m4a", ".mp3")
    bonus_policy: str = "ask"
    background_policy: str = "ask"
    reference_wait_seconds: int = REFERENCE_SELECTION_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ToolPaths:
    asmr_root: Path
    asmr_home: Path
    python: Path
    cli_script: Path
    launcher: Path
    ffmpeg: Path
    ffprobe: Path
    powershell: str
    video_encoder_options: tuple[str, ...]


@dataclass(frozen=True)
class AudioSource:
    order: int
    path: Path
    title_ja: str
    size: int
    mtime_ns: int
    relative_path: str = ""
    category: str = "main"
    transcript_path: Path | None = None
    transcript_language: str | None = None
    transcript_timed: bool = False
    source_language: str = "ja"


@dataclass(frozen=True)
class SmartTaskPlan:
    """A fully configured smart-scan task that has not started processing yet."""

    folder: Path
    output_root: Path
    edition_label: str
    sources: tuple[AudioSource, ...]
    edition: dict[str, Any]
    mode: str
    layout: str
    background: Path | None
    embed_subtitles: bool
    plan_id: str
    rebuild: bool
    force: bool
    retry_of: str | None = None
    translate_work_title: bool = True
    translate_track_titles: bool = True


@dataclass(frozen=True)
class SmartPlanDefaults:
    """Reusable choices inherited by the next work and the next app launch."""

    mode: str
    video_mode: str
    layout: str
    include_bonus: bool
    background_choice: str
    background_relative: str | None
    embed_subtitles: bool
    edition_extension: str | None = None
    edition_language: str | None = None
    edition_mix_variant: str | None = None
    edition_orientation: str | None = None


@dataclass
class SmartPlanDraft:
    folder: Path
    output_root: Path
    scan: ScanResult
    edition_label: str
    sources: list[AudioSource]
    edition: dict[str, Any]
    include_bonus: bool
    mode: str
    video_mode: str
    layout: str
    background: Path | None
    background_choice: str
    background_relative: str | None
    embed_subtitles: bool
    transcript_choices: dict[str, str]


@dataclass
class QueuedSmartWork:
    draft: SmartPlanDraft
    plan: SmartTaskPlan
    defaults: SmartPlanDefaults


def print_header() -> None:
    print()
    print("=" * 68)
    print("  ASMR-Dubber AutoFlow · 音频 / 静态视频 / 双语制作")
    print("=" * 68)
    print(f"日志：{LOG_FILE}")


def append_log(text: str) -> None:
    """Append console output to a UTF-8 log without exposing API keys."""

    if not text:
        return
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.is_file() and LOG_FILE.stat().st_size >= LOG_MAX_BYTES:
            rotated = LOG_FILE.with_suffix(".log.1")
            rotated.unlink(missing_ok=True)
            os.replace(LOG_FILE, rotated)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except OSError:
        # Logging must never stop a media task.
        pass


def log_event(message: str) -> None:
    append_log(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}\n")


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


def normalize_mode(value: Any) -> str:
    normalized = MODE_ALIASES.get(str(value or "").strip().casefold())
    if normalized is None:
        raise VideoPreparerError(f"未知模式：{value}")
    return normalized


def normalize_layout(value: Any) -> str:
    normalized = LAYOUT_ALIASES.get(str(value or "").strip().casefold())
    if normalized is None:
        raise VideoPreparerError(f"未知成品组织方式：{value}")
    return normalized


def mode_label(mode: str) -> str:
    return {
        MODE_AUDIO: "纯音频模式",
        MODE_VIDEO_NORMAL: "视频模式 · 普通",
        MODE_VIDEO_HARMONIZED: "视频模式 · 和谐",
    }[normalize_mode(mode)]


def layout_label(layout: str) -> str:
    return {
        LAYOUT_MERGED: "合并成一部",
        LAYOUT_SEPARATE: "每条音轨分别输出",
        LAYOUT_BOTH: "分轨输出 + 合并版",
    }[normalize_layout(layout)]


def initial_smart_plan_defaults(config: AppConfig) -> SmartPlanDefaults:
    layout = (
        LAYOUT_MERGED
        if config.default_output_layout == "ask"
        else normalize_layout(config.default_output_layout)
    )
    return SmartPlanDefaults(
        mode=MODE_VIDEO_NORMAL,
        video_mode=MODE_VIDEO_NORMAL,
        layout=layout,
        include_bonus=config.bonus_policy == "include",
        background_choice=("black" if config.background_policy == "black" else "auto"),
        background_relative=None,
        embed_subtitles=True,
    )


def load_smart_plan_defaults(
    config: AppConfig,
    path: Path | None = None,
) -> SmartPlanDefaults:
    path = LAST_OPTIONS_FILE if path is None else path
    fallback = initial_smart_plan_defaults(config)
    if not path.is_file():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != LAST_OPTIONS_SCHEMA:
            return fallback
        mode = normalize_mode(payload.get("mode"))
        video_mode = normalize_mode(payload.get("video_mode"))
        if video_mode not in {MODE_VIDEO_NORMAL, MODE_VIDEO_HARMONIZED}:
            video_mode = MODE_VIDEO_NORMAL
        layout = normalize_layout(payload.get("layout"))
        background_choice = str(payload.get("background_choice") or "auto")
        if background_choice not in {"auto", "black", "relative"}:
            background_choice = "auto"
        background_relative = str(payload.get("background_relative") or "").strip()
        if background_choice != "relative" or not background_relative:
            background_relative = None
        return SmartPlanDefaults(
            mode=mode,
            video_mode=video_mode,
            layout=layout,
            include_bonus=bool(payload.get("include_bonus", False)),
            background_choice=background_choice,
            background_relative=background_relative,
            embed_subtitles=bool(payload.get("embed_subtitles", True)),
            edition_extension=str(payload.get("edition_extension") or "").strip() or None,
            edition_language=str(payload.get("edition_language") or "").strip() or None,
            edition_mix_variant=str(payload.get("edition_mix_variant") or "").strip() or None,
            edition_orientation=str(payload.get("edition_orientation") or "").strip() or None,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, VideoPreparerError):
        return fallback


def save_smart_plan_defaults(
    defaults: SmartPlanDefaults,
    path: Path | None = None,
) -> None:
    path = LAST_OPTIONS_FILE if path is None else path
    payload = {
        "schema": LAST_OPTIONS_SCHEMA,
        "mode": normalize_mode(defaults.mode),
        "video_mode": normalize_mode(defaults.video_mode),
        "layout": normalize_layout(defaults.layout),
        "include_bonus": bool(defaults.include_bonus),
        "background_choice": defaults.background_choice,
        "background_relative": defaults.background_relative,
        "embed_subtitles": bool(defaults.embed_subtitles),
        "edition_extension": defaults.edition_extension,
        "edition_language": defaults.edition_language,
        "edition_mix_variant": defaults.edition_mix_variant,
        "edition_orientation": defaults.edition_orientation,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_app_config(path: Path = SETTINGS_FILE) -> AppConfig:
    values: dict[str, str] = {}
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            raise VideoPreparerError(f"无法读取设置文件：{path}: {exc}") from exc
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if "=" not in line:
                raise VideoPreparerError(f"设置文件第 {line_number} 行缺少等号：{raw_line}")
            key, value = line.split("=", 1)
            key = key.strip().casefold()
            if key not in {
                "asmr_dubber_path",
                "harmonized_volume_reduction_db",
                "harmonized_delay_minutes",
                "output_folder_name",
                "default_output_layout",
                "preferred_audio_formats",
                "bonus_policy",
                "background_policy",
            } and not re.fullmatch(r"timestamp_footer_line_[1-5]", key):
                print(f"警告：忽略 settings.txt 中的未知设置：{key}")
                continue
            values[key] = value.strip().strip('"').strip("'")

    configured_root = values.get("asmr_dubber_path", "").strip()
    asmr_root: Path | None = None
    if configured_root:
        candidate = Path(configured_root).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        asmr_root = candidate.resolve()

    try:
        reduction = float(
            values.get(
                "harmonized_volume_reduction_db",
                str(DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB),
            )
        )
        delay_minutes = float(
            values.get(
                "harmonized_delay_minutes",
                str(DEFAULT_HARMONIZED_DELAY_MINUTES),
            )
        )
    except ValueError as exc:
        raise VideoPreparerError("settings.txt 中的音量或延后时间不是有效数字。") from exc
    if not 0 <= reduction <= 60:
        raise VideoPreparerError("harmonized_volume_reduction_db 必须在 0 到 60 之间。")
    if not 0 <= delay_minutes <= 24 * 60:
        raise VideoPreparerError("harmonized_delay_minutes 必须在 0 到 1440 之间。")
    default_footer_lines = DEFAULT_TIMESTAMP_FOOTER.splitlines()
    custom_footer = any(key.startswith("timestamp_footer_line_") for key in values)
    footer_lines = [
        values.get(
            f"timestamp_footer_line_{index}",
            (
                ""
                if custom_footer
                else default_footer_lines[index - 1]
                if index <= len(default_footer_lines)
                else ""
            ),
        ).strip()
        for index in range(1, 6)
    ]
    output_folder_name = values.get("output_folder_name", "AutoFlow输出").strip()
    if not output_folder_name or any(character in output_folder_name for character in '<>:"/\\|?*'):
        raise VideoPreparerError("output_folder_name 不是有效的 Windows 文件夹名称。")
    default_output_layout = values.get("default_output_layout", "ask").strip().casefold()
    if default_output_layout != "ask":
        default_output_layout = normalize_layout(default_output_layout)
    preferred_formats = tuple(
        item if item.startswith(".") else f".{item}"
        for item in (
            part.strip().casefold()
            for part in values.get("preferred_audio_formats", "wav,flac,ape,m4a,mp3").split(",")
        )
        if item
    )
    bonus_policy = values.get("bonus_policy", "ask").strip().casefold()
    if bonus_policy not in {"ask", "include", "exclude"}:
        raise VideoPreparerError("bonus_policy 必须是 ask、include 或 exclude。")
    background_policy = values.get("background_policy", "ask").strip().casefold()
    if background_policy not in {"ask", "auto", "black"}:
        raise VideoPreparerError("background_policy 必须是 ask、auto 或 black。")
    return AppConfig(
        asmr_root=asmr_root,
        harmonized_volume_db=-abs(reduction),
        harmonized_delay_seconds=round(delay_minutes * 60),
        timestamp_footer="\n".join(line for line in footer_lines if line),
        output_folder_name=output_folder_name,
        default_output_layout=default_output_layout,
        preferred_audio_formats=preferred_formats,
        bonus_policy=bonus_policy,
        background_policy=background_policy,
    )


def clean_user_path(value: str) -> Path:
    text = value.strip().strip('"').strip("'")
    if not text:
        raise VideoPreparerError("没有输入作品文件夹路径。")
    return Path(text).expanduser().resolve()


def find_tool_paths(config: AppConfig) -> ToolPaths:
    configured = (
        os.environ.get("ASMR_DUBBER_ROOT", "").strip()
        or os.environ.get("ASMR_NEXT_ROOT", "").strip()
    )
    candidates = []
    if config.asmr_root is not None:
        candidates.append(config.asmr_root)
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            portable_home().parent,
            TOOL_ROOT.parent / "ASMR-Dubber",
            TOOL_ROOT.parent / "asmr-next",
        )
    )

    asmr_root = next((path.resolve() for path in candidates if path.is_dir()), None)
    if asmr_root is None:
        raise VideoPreparerError(
            "找不到 ASMR Dubber 主程序目录；请确认 AutoFlow 从完整程序目录启动，"
            "或设置环境变量 ASMR_DUBBER_ROOT。"
        )

    asmr_home = asmr_root / ".asmr-dubber"
    python_candidates = (
        asmr_home / "venv" / "Scripts" / "python.exe",
        asmr_root / ".venv" / "Scripts" / "python.exe",
    )
    python = next((path for path in python_candidates if path.is_file()), None)
    if python is None:
        raise VideoPreparerError("ASMR Dubber 尚未安装完整运行环境，找不到便携 Python。")

    ffmpeg_root = asmr_home / "runtimes" / "ffmpeg-shared"
    ffmpeg = next(iter(sorted(ffmpeg_root.rglob("ffmpeg.exe"))), None)
    ffprobe = next(iter(sorted(ffmpeg_root.rglob("ffprobe.exe"))), None)
    if ffmpeg is None or ffprobe is None:
        raise VideoPreparerError("ASMR Dubber 的 FFmpeg/FFprobe 不完整，请先修复其安装。")

    # AutoFlow calls a few supported ASMR Dubber Python APIs directly (script
    # import and voice-reference extraction). The normal launcher sets these
    # variables in PowerShell; set the same portable paths here so a fresh
    # Windows machine does not need a system-wide FFmpeg installation.
    os.environ["ASMR_DUBBER_HOME"] = str(asmr_home)
    os.environ["ASMR_DUBBER_FFMPEG"] = str(ffmpeg)
    current_path = os.environ.get("PATH", "")
    ffmpeg_bin = str(ffmpeg.parent)
    if ffmpeg_bin.casefold() not in {
        item.casefold() for item in current_path.split(os.pathsep) if item
    }:
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + current_path

    cli_script = asmr_root / "scripts" / "windows" / "run-cli.ps1"
    launcher = asmr_root / "ASMR-Dubber.exe"
    if not cli_script.is_file() or not launcher.is_file():
        raise VideoPreparerError("ASMR Dubber 缺少命令行脚本或启动程序。")

    powershell_path = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell_path:
        raise VideoPreparerError("找不到 PowerShell。Windows 自带的 PowerShell 5.1 即可。")

    encoder_result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    encoder_text = encoder_result.stdout + encoder_result.stderr
    if re.search(r"\blibx264\b", encoder_text):
        video_encoder_options = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20")
    elif re.search(r"\blibopenh264\b", encoder_text):
        video_encoder_options = (
            "-c:v",
            "libopenh264",
            "-rc_mode",
            "quality",
            "-q:v",
            "20",
        )
    elif re.search(r"\bmpeg4\b", encoder_text):
        video_encoder_options = ("-c:v", "mpeg4", "-q:v", "2")
    else:
        raise VideoPreparerError("FFmpeg 没有可用的软件视频编码器。")
    video_encoder_options = (*video_encoder_options, *PERIODIC_KEYFRAME_OPTIONS)

    return ToolPaths(
        asmr_root=asmr_root,
        asmr_home=asmr_home,
        python=python,
        cli_script=cli_script,
        launcher=launcher,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        powershell=powershell_path,
        video_encoder_options=video_encoder_options,
    )


def validate_asmr_version() -> str:
    try:
        from asmr_dubber import __version__
    except Exception as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 版本：{exc}") from exc
    version = str(__version__)
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is not None and tuple(map(int, match.groups())) < (0, 7, 1):
        raise VideoPreparerError(
            f"ASMR Dubber {version} 太旧；AutoFlow 需要 0.7.1 或后续兼容版本。"
        )
    log_event(f"检测到 ASMR Dubber {version}")
    return version


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


def safe_filename_component(value: str, *, fallback: str = "未命名", limit: int = 80) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" ._")
    text = re.sub(r"\s+", " ", text)
    if not text:
        text = fallback
    return text[:limit].rstrip(" ._") or fallback


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


def parse_embed_subtitles_argument(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise VideoPreparerError(f"未知的字幕内嵌选项：{value}")


def ask_embed_subtitles(mode: str, argument: str | None = None) -> bool:
    if normalize_mode(mode) == MODE_AUDIO:
        return False
    if argument is not None:
        return parse_embed_subtitles_argument(argument)
    print("\n是否把字幕放进最终视频？")
    print("  无论如何都会另外保留双语版.srt 和双语版.lrc。")
    return ask_yes_no("输入 Y 内嵌字幕，输入 N 仅保留外部字幕：")


def file_stat_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def fingerprint(audio: Iterable[AudioSource], background: Path | None) -> dict[str, Any]:
    image_info: dict[str, Any] | None = None
    if background is not None:
        stat = background.stat()
        image_info = {
            "path": str(background),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {
        "audio": [
            {
                "order": item.order,
                "path": str(item.path),
                "relative_path": item.relative_path,
                "size": item.size,
                "mtime_ns": item.mtime_ns,
                "category": item.category,
                "transcript": file_stat_payload(item.transcript_path),
                "transcript_language": item.transcript_language,
                "transcript_timed": item.transcript_timed,
                "source_language": item.source_language,
            }
            for item in audio
        ],
        "background": image_info,
    }


def task_key(folder: Path) -> str:
    normalized = os.path.normcase(str(folder.resolve())).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:20]


def state_path(folder: Path) -> Path:
    return STATE_ROOT / f"{task_key(folder)}.json"


def workspace_path(folder: Path) -> Path:
    return WORK_ROOT / task_key(folder)


def planned_job_key(source_folder: Path, plan_id: str, job_id: str) -> str:
    raw = "\n".join(
        (
            os.path.normcase(str(source_folder.resolve())),
            plan_id,
            job_id,
        )
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def planned_state_path(source_folder: Path, plan_id: str, job_id: str) -> Path:
    return STATE_ROOT / f"smart-{planned_job_key(source_folder, plan_id, job_id)}.json"


def planned_workspace_path(source_folder: Path, plan_id: str, job_id: str) -> Path:
    return WORK_ROOT / f"smart-{planned_job_key(source_folder, plan_id, job_id)}"


def plan_identity(
    source_folder: Path,
    *,
    mode: str,
    layout: str,
    edition: dict[str, Any],
    sources: list[AudioSource],
    output_root: Path | None = None,
    background: Path | None = None,
    embed_subtitles: bool = True,
    translate_work_title: bool = True,
    translate_track_titles: bool = True,
) -> str:
    payload = {
        "source_folder": os.path.normcase(str(source_folder.resolve())),
        "mode": normalize_mode(mode),
        "layout": normalize_layout(layout),
        "edition": edition,
        "output_root": str(output_root.resolve()) if output_root else None,
        "background": file_stat_payload(background),
        "tracks": [
            {
                "path": os.path.normcase(str(item.path.resolve())),
                "size": item.size,
                "mtime_ns": item.mtime_ns,
                "transcript": file_stat_payload(item.transcript_path),
                **(
                    {
                        "transcript_language": item.transcript_language,
                        "transcript_timed": item.transcript_timed,
                    }
                    if item.transcript_path is not None
                    else {}
                ),
                "source_language": item.source_language,
            }
            for item in sources
        ],
    }
    # Keep the original identity for the historical/default behaviour so
    # existing completed and resumable plans remain usable after upgrading.
    if normalize_mode(mode) != MODE_AUDIO and not embed_subtitles:
        payload["embed_subtitles"] = False
    if not translate_work_title or not translate_track_titles:
        payload["title_translation"] = {
            "work": bool(translate_work_title),
            "tracks": bool(translate_track_titles),
        }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def plan_metadata_path(plan_id: str) -> Path:
    return STATE_ROOT / f"smart-plan-{plan_id}.json"


def load_plan_metadata(plan_id: str) -> dict[str, Any]:
    path = plan_metadata_path(plan_id)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if payload.get("schema") == 1 else {}


def save_plan_metadata(plan_id: str, payload: dict[str, Any]) -> None:
    stored = {"schema": 1, **payload}
    stored["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_text(
        plan_metadata_path(plan_id),
        json.dumps(stored, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_plan_manifest(
    destination: Path,
    *,
    source_folder: Path,
    output_folder: Path,
    mode: str,
    layout: str,
    edition: dict[str, Any],
    sources: list[AudioSource],
    background: Path | None,
    embed_subtitles: bool,
    plan_id: str | None = None,
    jobs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "plan_id": plan_id,
        "source_folder": str(source_folder),
        "output_folder": str(output_folder),
        "mode": normalize_mode(mode),
        "layout": normalize_layout(layout),
        "edition": edition,
        "background": str(background) if background else None,
        "embed_subtitles": bool(embed_subtitles),
        "tracks": [
            {
                "index": index,
                "path": str(item.path),
                "relative_path": item.relative_path or item.path.name,
                "title": item.title_ja,
                "category": item.category,
                "transcript": str(item.transcript_path) if item.transcript_path else None,
                "transcript_language": item.transcript_language,
                "transcript_timed": item.transcript_timed,
                "source_language": item.source_language,
            }
            for index, item in enumerate(sources, start=1)
        ],
        "jobs": list(jobs or ()),
    }
    atomic_write_text(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VideoPreparerError(f"任务状态文件损坏：{path}: {exc}") from exc
    if payload.get("schema") != 1:
        raise VideoPreparerError(f"不支持的任务状态版本：{path}")
    payload["mode"] = normalize_mode(payload.get("mode"))
    payload.setdefault("harmonized_volume_db", -DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB)
    payload.setdefault("harmonized_delay_seconds", round(DEFAULT_HARMONIZED_DELAY_MINUTES * 60))
    payload.setdefault("embed_subtitles", True)
    payload.setdefault("reference_wait_seconds", REFERENCE_SELECTION_TIMEOUT_SECONDS)
    return payload


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def status_at_least(state: dict[str, Any], status: str) -> bool:
    return STATUS_ORDER.get(str(state.get("status", "")), 0) >= STATUS_ORDER[status]


def legacy_chinese_timeline_needs_reimport(state: dict[str, Any]) -> bool:
    """Detect tasks created by the old ASR-plus-Chinese-overlay workflow."""

    transcript_import = state.get("transcript_import") or {}
    if str(transcript_import.get("kind") or "") != "zh_overlay":
        return False
    timeline = list(state.get("timeline") or [])
    return bool(timeline) and all(
        item.get("transcript")
        and str(item.get("transcript_language") or "") == "zh"
        and bool(item.get("transcript_timed"))
        for item in timeline
    )


def reset_legacy_chinese_timeline_state(state: dict[str, Any]) -> None:
    """Rewind generated stages so a complete Chinese timeline can be reimported."""

    state["status"] = "project_created"
    for key in (
        "chinese_transcript_overlay_done",
        "chinese_transcript_overlay_sentences",
        "outputs",
        "project_mixed_audio",
    ):
        state.pop(key, None)


def safe_reset_workspace(workspace: Path) -> None:
    root = WORK_ROOT.resolve()
    target = workspace.resolve()
    if target.parent != root:
        raise VideoPreparerError(f"拒绝清理非任务目录：{target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop the current FFmpeg/ASMR CLI command and its child processes."""

    _terminate_process_tree(process)


def run_process(arguments: list[str], *, cwd: Path | None = None) -> None:
    check_cancelled()
    log_event("运行命令：" + " ".join(str(item) for item in arguments))
    try:
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法启动命令：{arguments[0]}: {exc}") from exc
    signal = register_process(process)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="")
            append_log(line)
            check_cancelled(signal)
        result_code = process.wait()
        check_cancelled(signal)
    except (KeyboardInterrupt, OperationCancelledError):
        terminate_process_tree(process)
        raise
    finally:
        unregister_process(process, signal)
    if result_code != 0:
        raise VideoPreparerError(f"命令执行失败（退出码 {result_code}）：{Path(arguments[0]).name}")


def run_process_captured(arguments: list[str], *, cwd: Path | None = None) -> str:
    check_cancelled()
    environment = os.environ.copy()
    environment.update({"COLUMNS": "10000", "NO_COLOR": "1", "TERM": "dumb"})
    try:
        log_event("运行命令：" + " ".join(str(item) for item in arguments))
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法启动命令：{arguments[0]}: {exc}") from exc
    signal = register_process(process)
    try:
        stdout, stderr = process.communicate()
        check_cancelled(signal)
    except OperationCancelledError:
        terminate_process_tree(process)
        raise
    finally:
        unregister_process(process, signal)
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
        append_log(stdout)
    if stderr:
        print(
            stderr,
            end="" if stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
        append_log(stderr)
    if process.returncode != 0:
        raise VideoPreparerError(
            f"命令执行失败（退出码 {process.returncode}）：{Path(arguments[0]).name}"
        )
    return "\n".join(part for part in (stdout, stderr) if part)


def run_ffmpeg(paths: ToolPaths, arguments: list[str], *, cwd: Path | None = None) -> None:
    command = [
        str(paths.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-y",
        *arguments,
    ]
    run_process(command, cwd=cwd)


def ffprobe_json(paths: ToolPaths, media: Path, entries: str) -> dict[str, Any]:
    command = [
        str(paths.ffprobe),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        entries,
        "-of",
        "json",
        str(media),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法运行 FFprobe：{exc}") from exc
    if result.returncode != 0:
        raise VideoPreparerError(f"FFprobe 无法读取 {media.name}：{result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoPreparerError(f"FFprobe 返回了无效结果：{media}") from exc


def audio_duration_samples(paths: ToolPaths, media: Path) -> int:
    payload = ffprobe_json(
        paths,
        media,
        "stream=sample_rate,duration_ts,time_base:format=duration",
    )
    streams = payload.get("streams") or []
    if not streams:
        raise VideoPreparerError(f"文件没有可用音轨：{media}")
    stream = streams[0]
    try:
        sample_rate = int(stream["sample_rate"])
        duration_ts = int(stream["duration_ts"])
        time_base = Fraction(str(stream["time_base"]))
        duration = Fraction(duration_ts) * time_base
        samples = duration * SAMPLE_RATE
        if samples.denominator != 1:
            samples = Fraction(round(float(samples)), 1)
        if sample_rate <= 0 or samples <= 0:
            raise ValueError
        return int(samples)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        try:
            duration = Fraction(str(payload["format"]["duration"]))
            samples = round(float(duration) * SAMPLE_RATE)
            if samples <= 0:
                raise ValueError
            return samples
        except (KeyError, TypeError, ValueError) as fallback_exc:
            raise VideoPreparerError(f"无法取得准确音频时长：{media}") from fallback_exc


def normalize_and_concat(
    paths: ToolPaths,
    sources: list[AudioSource],
    workspace: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    segments_dir = workspace / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict[str, Any]] = []
    cumulative_samples = 0

    print("\n[1/5] 统一音频规格并计算实际时间轴")
    for index, source in enumerate(sources, start=1):
        output = segments_dir / f"seg_{index:06d}.flac"
        print(f"  [{index}/{len(sources)}] {source.path.name}")
        run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-i",
                str(source.path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                f"aresample={SAMPLE_RATE}:async=0",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-sample_fmt",
                "s16",
                "-c:a",
                "flac",
                "-compression_level",
                "0",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(output),
            ],
        )
        samples = audio_duration_samples(paths, output)
        timeline.append(
            {
                "order": source.order,
                "source": str(source.path),
                "filename": source.path.name,
                "relative_path": source.relative_path or source.path.name,
                "title_ja": source.title_ja,
                "category": source.category,
                "transcript": str(source.transcript_path) if source.transcript_path else None,
                "transcript_language": source.transcript_language,
                "transcript_timed": source.transcript_timed,
                "source_language": source.source_language,
                "normalized": str(output),
                "start_samples": cumulative_samples,
                "duration_samples": samples,
            }
        )
        cumulative_samples += samples

    concat_file = workspace / "concat.ffconcat"
    concat_lines = ["ffconcat version 1.0"]
    concat_lines.extend(
        f"file 'segments/seg_{index:06d}.flac'" for index in range(1, len(sources) + 1)
    )
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="ascii")

    master = workspace / "master.flac"
    print("  正在拼接无损母带……")
    run_ffmpeg(
        paths,
        [
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            concat_file.name,
            "-map",
            "0:a:0",
            "-af",
            "asetpts=N/SR/TB",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            str(master),
        ],
        cwd=workspace,
    )
    master_samples = audio_duration_samples(paths, master)
    if abs(master_samples - cumulative_samples) > 1:
        raise VideoPreparerError("拼接后的采样数与各小音频之和不一致，已停止，避免生成错误时间轴。")
    return master, timeline


def partial_output_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.stem}.partial.{uuid.uuid4().hex}{destination.suffix}"
    )


def background_input(background: Path | None) -> list[str]:
    if background is not None:
        return ["-i", str(background)]
    return [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={VIDEO_SIZE}:r=1:d=1",
    ]


def render_static_video(
    paths: ToolPaths,
    audio_source: Path,
    background: Path | None,
    destination: Path,
    *,
    lead_seconds: int = 0,
    volume_db: float = 0.0,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_output_path(destination)
    total_samples = audio_duration_samples(paths, audio_source) + lead_seconds * SAMPLE_RATE
    duration_text = f"{total_samples / SAMPLE_RATE:.6f}"
    audio_filters = [
        f"aresample={SAMPLE_RATE}:async=0",
        f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
    ]
    if volume_db:
        audio_filters.append(f"volume={volume_db:g}dB")
    if lead_seconds:
        audio_filters.append(f"adelay={lead_seconds * 1000}:all=1")
    audio_filters.append("asetpts=N/SR/TB")

    # Scale the source picture once, then loop that prepared 1080p frame in
    # memory. This avoids decoding an 8K JPEG again for every video frame.
    video_filter = (
        f"[0:v:0]scale={VIDEO_FILTER_SIZE}:force_original_aspect_ratio=decrease:"
        f"flags=lanczos,pad={VIDEO_FILTER_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,format=yuv420p,loop=loop=-1:size=1:start=0,"
        f"setpts=N/{VIDEO_FPS}/TB,trim=duration={duration_text}[v];"
        f"[1:a:0]{','.join(audio_filters)}[a]"
    )
    try:
        run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-stats",
                "-stats_period",
                "5",
                *background_input(background),
                "-i",
                str(audio_source),
                "-filter_complex",
                video_filter,
                "-map",
                "[v]",
                "-map",
                "[a]",
                *paths.video_encoder_options,
                "-r",
                str(VIDEO_FPS),
                "-fps_mode",
                "cfr",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-t",
                duration_text,
                "-shortest",
                "-movflags",
                "+faststart",
                "-metadata",
                f"title={destination.stem}",
                str(partial),
            ],
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def render_delayed_existing_video(
    paths: ToolPaths,
    source: Path,
    destination: Path,
    *,
    lead_seconds: int,
    subtitle_file: Path | None,
    volume_db: float = 0.0,
    audio_source: Path | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_output_path(destination)
    audio_filters = [
        f"aresample={SAMPLE_RATE}:async=0",
        f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
    ]
    if volume_db:
        audio_filters.append(f"volume={volume_db:g}dB")
    audio_filters.extend((f"adelay={lead_seconds * 1000}:all=1", "asetpts=N/SR/TB"))
    audio_input_index = 1 if audio_source is not None else 0
    filter_complex = (
        f"[0:v:0]tpad=start_mode=clone:start_duration={lead_seconds},"
        f"scale={VIDEO_FILTER_SIZE}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_FILTER_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={VIDEO_FPS},format=yuv420p,setpts=PTS-STARTPTS[v];"
        f"[{audio_input_index}:a:0]{','.join(audio_filters)}[a]"
    )
    input_arguments = ["-i", str(source)]
    if audio_source is not None:
        input_arguments.extend(("-i", str(audio_source)))
    subtitle_input_index: int | None = None
    if subtitle_file is not None:
        subtitle_input_index = 2 if audio_source is not None else 1
        input_arguments.extend(("-f", "srt", "-i", str(subtitle_file)))
    subtitle_arguments = (
        [
            "-map",
            f"{subtitle_input_index}:s:0",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=zho",
        ]
        if subtitle_input_index is not None
        else ["-sn"]
    )
    try:
        run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-stats",
                "-stats_period",
                "5",
                *input_arguments,
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[a]",
                *subtitle_arguments,
                *paths.video_encoder_options,
                "-r",
                str(VIDEO_FPS),
                "-fps_mode",
                "cfr",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-metadata",
                "title=双语版",
                str(partial),
            ],
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_output_path(destination)
    try:
        with source.open("rb") as reader, partial.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def run_asmr_cli(paths: ToolPaths, *arguments: str) -> None:
    command = [
        paths.powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(paths.cli_script),
        *arguments,
    ]
    run_process(command, cwd=paths.asmr_root)


def run_asmr_cli_captured(paths: ToolPaths, *arguments: str) -> str:
    command = [
        paths.powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(paths.cli_script),
        *arguments,
    ]
    return run_process_captured(command, cwd=paths.asmr_root)


def configured_projects_root(paths: ToolPaths) -> Path:
    try:
        from asmr_dubber.user_settings import load_user_settings

        value = str(load_user_settings().projects_root or "").strip()
    except Exception as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 用户设置：{exc}") from exc
    return Path(value).expanduser().resolve() if value else (paths.asmr_home / "projects")


def known_project_manifests(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.rglob("project.json") if path.is_file()}


def create_asmr_project(
    paths: ToolPaths,
    video: Path,
    *,
    source_language: str = "ja",
) -> Path:
    projects_root = configured_projects_root(paths)
    before = known_project_manifests(projects_root)
    started_ns = time.time_ns()
    output = run_asmr_cli_captured(
        paths,
        "create",
        str(video),
        "--projects-root",
        str(projects_root),
        "--source-language",
        "en" if source_language == "en" else "ja",
    )
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for raw_line in reversed(output.splitlines()):
        clean_line = ansi_escape.sub("", raw_line).strip().strip('"')
        if not clean_line.casefold().endswith("project.json"):
            continue
        candidate = Path(clean_line).expanduser().resolve()
        try:
            candidate.relative_to(projects_root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    after = known_project_manifests(projects_root)
    created = list(after - before)
    if not created:
        created = [path for path in after if path.stat().st_mtime_ns >= started_ns - 5_000_000_000]
    if not created:
        raise VideoPreparerError("ASMR Dubber 已返回成功，但没有找到新项目的 project.json。")
    created.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return created[0]


def read_project(project_json: Path) -> dict[str, Any]:
    try:
        return json.loads(project_json.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 项目：{project_json}: {exc}") from exc


def project_asset(project_json: Path, stored: Any) -> Path | None:
    text = str(stored or "").strip()
    if not text:
        return None
    candidate = (project_json.parent / text).resolve()
    try:
        candidate.relative_to(project_json.parent.resolve())
    except ValueError as exc:
        raise VideoPreparerError(f"项目输出路径越界：{text}") from exc
    return candidate if candidate.is_file() else None


def source_language_for_sources(sources: Sequence[AudioSource]) -> str:
    """Choose the only ASR-compatible source language represented by a job.

    ASMR Dubber currently accepts Japanese and English at project creation. A
    Chinese timed script is imported later and changes the project language,
    so Chinese-labelled files intentionally fall back to Japanese here.
    """

    languages = {str(item.source_language).casefold() for item in sources}
    return "en" if languages and languages == {"en"} else "ja"


def ensure_autoflow_project_outputs(project_json: Path) -> None:
    """Ensure an AutoFlow project can produce the promised bilingual mix."""

    try:
        from asmr_dubber.models import load_project, save_project

        project, project_dir = load_project(project_json)
        if project.settings.mix_output_mode == "stem":
            project.settings.mix_output_mode = "both"
            save_project(project, project_dir)
    except Exception as exc:
        raise VideoPreparerError(f"无法准备 ASMR Dubber 项目的混合输出设置：{exc}") from exc


def launch_asmr_ui(paths: ToolPaths, project_json: Path) -> None:
    with suppress(OSError):
        subprocess.run(
            [
                paths.powershell,
                "-NoProfile",
                "-Command",
                "Set-Clipboard -Value $args[0]",
                str(project_json),
            ],
            check=False,
            capture_output=True,
        )
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    try:
        subprocess.Popen(
            [str(paths.launcher)],
            cwd=paths.asmr_root,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法启动 ASMR Dubber 网页：{exc}") from exc


def project_reference_id(project_json: Path) -> str:
    project = read_project(project_json)
    settings = project.get("settings") or {}
    return str(settings.get("tts_reference_sentence_id") or "").strip()


def project_has_external_reference(project_json: Path) -> bool:
    """Return whether the current 0.7.x project already has a usable reference."""

    project = read_project(project_json)
    settings = project.get("settings") or {}
    source = str(settings.get("tts_reference_source") or "project_sentence").strip()
    if source != "external":
        return False
    audio = Path(str(settings.get("tts_external_reference_audio") or "")).expanduser()
    return audio.is_file()


def wait_for_reference(
    paths: ToolPaths,
    project_json: Path,
    *,
    timeout_seconds: int = REFERENCE_SELECTION_TIMEOUT_SECONDS,
    event_callback: ReferenceEventCallback | None = None,
    event_context: dict[str, Any] | None = None,
    launch_ui: bool = True,
) -> bool:
    if timeout_seconds <= 0:
        print("\n  未指定统一参考音频，将使用主程序自动推荐的清晰片段。")
        return True

    request_id = uuid.uuid4().hex
    deadline_epoch = time.time() + timeout_seconds

    def emit(kind: str, **details: Any) -> None:
        if event_callback is None:
            return
        payload = {
            **(event_context or {}),
            "kind": kind,
            "request_id": request_id,
            "project_json": str(project_json.resolve()),
            "timeout_seconds": timeout_seconds,
            "deadline_epoch": deadline_epoch,
            **details,
        }
        try:
            event_callback(payload)
        except Exception as exc:
            # Browser/UI notifications are optional. A broken or closed page
            # must never stop the actual AutoFlow task.
            with suppress(Exception):
                log_event(f"参考音频页面通知失败，任务继续：{type(exc).__name__}: {exc}")

    emit("ready")
    print("\n[4/5] 等你在网页中选择统一音色参考")
    print(f"  项目：{project_json}")
    if launch_ui:
        print("  项目路径已复制到剪贴板。网页会预选最近项目；请点击“打开项目”。")
        print("  选择清晰片段后，必须点击“设为项目音色参考”。")
        print("  如果还修改了表格，请同时点击“保存校对表格”。")
        launch_asmr_ui(paths, project_json)
    else:
        print("  可在批量处理页展开当前任务下方的参考音频选择器。")
    print(f"  {timeout_seconds} 秒内未选择则使用自动推荐的参考音频。")

    deadline = time.monotonic() + timeout_seconds
    next_notice = time.monotonic() + 60
    while True:
        reference_id = project_reference_id(project_json)
        if reference_id:
            print(f"已检测到统一参考：{reference_id}")
            emit("selected", source="project_sentence", sentence_id=reference_id)
            return True
        if project_has_external_reference(project_json):
            print("已检测到 ASMR Dubber 设置中的外部参考音频。")
            emit("selected", source="external")
            return True
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            print("等待时间结束，将使用 ASMR Dubber 自动推荐的参考音频。")
            emit("timeout", source="automatic")
            return True
        if now >= next_notice:
            print(f"  仍在等待参考音频，剩余约 {max(1, int(remaining // 60) + 1)} 分钟……")
            next_notice = now + 60
        check_cancelled()
        time.sleep(min(2.0, remaining))


def project_source_language(project_json: Path) -> str:
    project = read_project(project_json)
    return str(project.get("source_language") or "ja")


def extract_shared_reference(project_json: Path, destination: Path) -> dict[str, str]:
    """把首个分轨项目的统一参考固化为后续项目可复用的外部参考。"""

    try:
        from asmr_dubber.models import load_project, save_project
        from asmr_dubber.voice_reference import prepare_voice_reference, shared_reference_sentence

        project, project_dir = load_project(project_json)
        sentence = shared_reference_sentence(project)
        source = (project_dir / project.source.path).resolve()
        reference = prepare_voice_reference(project, project_dir, source, sentence)
        save_project(project, project_dir)
    except Exception as exc:
        raise VideoPreparerError(f"无法固化分轨共用参考音频：{exc}") from exc

    atomic_copy(reference.path, destination)
    return {
        "audio": str(destination.resolve()),
        "text": reference.text,
        "language": reference.language,
    }


def apply_shared_reference(project_json: Path, reference: dict[str, str]) -> None:
    audio = Path(str(reference.get("audio") or "")).expanduser().resolve()
    if not audio.is_file():
        raise VideoPreparerError(f"分轨共用参考音频已经不存在：{audio}")
    try:
        from asmr_dubber.models import load_project, save_project

        project, project_dir = load_project(project_json)
        project.settings.tts_reference_source = "external"
        project.settings.tts_external_reference_audio = str(audio)
        project.settings.tts_external_reference_text = str(reference.get("text") or "")
        language = str(reference.get("language") or "auto")
        project.settings.tts_external_reference_language = cast(
            Literal["auto", "ja", "en", "zh"],
            language if language in {"ja", "en", "zh"} else "auto",
        )
        # IndexTTS2 有独立的音色参考选择；情绪仍可跟随当前句。
        project.settings.tts_index_speaker_source = "external"
        save_project(project, project_dir)
    except Exception as exc:
        raise VideoPreparerError(f"无法给分轨项目设置共用参考音频：{exc}") from exc


def _qwen_script_alignment_available(paths: ToolPaths) -> bool:
    model_root = paths.asmr_home / "models"
    if not model_root.is_dir():
        return False
    return any(
        "qwen3-forcedaligner" in path.as_posix().casefold()
        for path in model_root.rglob("config.json")
    )


def _load_transcript_sentences(
    path: Path,
    *,
    language: Literal["ja", "en", "zh"],
    duration_seconds: float,
) -> Any:
    try:
        from asmr_dubber.transcript_import import parse_transcript

        return parse_transcript(
            duration_seconds=duration_seconds,
            path=path,
            language=language,
        )
    except Exception as exc:
        raise VideoPreparerError(f"无法解析台本/字幕 {path.name}：{exc}") from exc


def _replace_project_with_timed_transcripts(
    project_json: Path,
    timeline: list[dict[str, Any]],
    *,
    language: Literal["ja", "en", "zh"],
) -> int:
    """Build one project timeline directly from per-track timed subtitles."""

    try:
        from asmr_dubber.languages import source_language_label
        from asmr_dubber.models import load_project, save_project, settings_for_source_language
        from asmr_dubber.pipeline import export_transcript

        project, project_dir = load_project(project_json)
        combined = []
        for item in timeline:
            duration = int(item["duration_samples"]) / SAMPLE_RATE
            parsed = _load_transcript_sentences(
                Path(str(item["transcript"])),
                language=language,
                duration_seconds=duration,
            )
            offset = int(item["start_samples"]) / SAMPLE_RATE
            for sentence in parsed.sentences:
                copied = sentence.model_copy(deep=True)
                copied.id = f"s{len(combined) + 1:06d}"
                copied.start_seconds += offset
                copied.end_seconds += offset
                combined.append(copied)
        if not combined:
            raise VideoPreparerError("已有字幕中没有可导入的有效台词。")

        project.sentences = combined
        project.source_language = language
        project.settings = settings_for_source_language(project.settings, language)
        project.asr_language = f"自动导入的分轨{source_language_label(language)}时间轴字幕"
        project.asr_settings_dirty = False
        project.settings.tts_reference_sentence_id = None
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_srt_file = None
        project.subtitle_lrc_file = None
        project.subtitle_video_file = None
        save_project(project, project_dir)
        export_transcript(project, project_dir)
        return len(combined)
    except VideoPreparerError:
        raise
    except Exception as exc:
        raise VideoPreparerError(f"无法合并分轨字幕：{exc}") from exc


def import_available_source_transcript(
    paths: ToolPaths,
    project_json: Path,
    timeline: list[dict[str, Any]],
) -> dict[str, Any] | None:
    usable = [
        item
        for item in timeline
        if item.get("transcript")
        and str(item.get("transcript_language") or "") in {"ja", "en", "zh"}
        and Path(str(item["transcript"])).suffix.casefold() != ".pdf"
    ]
    if not usable:
        return None

    # 单轨可以直接使用 ASMR Dubber 的完整台本导入流程，包括纯文本对齐。
    if len(timeline) == 1 and len(usable) == 1:
        item = usable[0]
        transcript = Path(str(item["transcript"])).resolve()
        language = str(item["transcript_language"])
        plain_timing = (
            "qwen"
            if not bool(item.get("transcript_timed"))
            and language in {"ja", "en"}
            and _qwen_script_alignment_available(paths)
            else "estimate"
        )
        try:
            from asmr_dubber.models import load_project
            from asmr_dubber.pipeline import import_project_transcript

            project, project_dir = load_project(project_json)
            result = import_project_transcript(
                project,
                project_dir,
                transcript_path=transcript,
                plain_timing=plain_timing,
                script_language=language,
                progress=lambda message, current, total: print(f"  {message}"),
            )
        except Exception as exc:
            raise VideoPreparerError(f"自动导入台本/字幕失败：{exc}") from exc
        return {
            "kind": "direct",
            "language": language,
            "path": str(transcript),
            **result,
        }

    # 合并模式只有在每一轨都有同语言的时间轴字幕时才直接替代 ASR。
    if len(usable) != len(timeline):
        if any(
            str(item.get("transcript_language") or "") == "zh"
            and bool(item.get("transcript_timed"))
            for item in usable
        ):
            return {"kind": "zh_overlay_partial", "count": len(usable)}
        return {"kind": "partial", "count": len(usable)}
    languages = {str(item["transcript_language"]) for item in usable}
    if len(languages) != 1 or not all(bool(item.get("transcript_timed")) for item in usable):
        if any(
            str(item.get("transcript_language") or "") == "zh"
            and bool(item.get("transcript_timed"))
            for item in usable
        ):
            return {"kind": "zh_overlay_partial", "count": len(usable)}
        return {"kind": "partial", "count": len(usable)}
    language = cast(Literal["ja", "en", "zh"], languages.pop())
    sentence_count = _replace_project_with_timed_transcripts(
        project_json,
        timeline,
        language=language,
    )
    return {"kind": "direct", "language": language, "sentences": sentence_count}


def overlay_timed_chinese_transcripts(
    project_json: Path,
    timeline: list[dict[str, Any]],
) -> int:
    chinese_cues: list[Any] = []
    for item in timeline:
        if (
            str(item.get("transcript_language") or "") != "zh"
            or not item.get("transcript")
            or not bool(item.get("transcript_timed"))
        ):
            continue
        duration = int(item["duration_samples"]) / SAMPLE_RATE
        parsed = _load_transcript_sentences(
            Path(str(item["transcript"])),
            language="zh",
            duration_seconds=duration,
        )
        offset = int(item["start_samples"]) / SAMPLE_RATE
        for cue in parsed.sentences:
            copied = cue.model_copy(deep=True)
            copied.start_seconds += offset
            copied.end_seconds += offset
            chinese_cues.append(copied)
    if not chinese_cues:
        return 0

    try:
        from asmr_dubber.models import load_project, save_project
        from asmr_dubber.pipeline import export_transcript

        project, project_dir = load_project(project_json)
        assignments: dict[int, list[str]] = {}
        for cue in chinese_cues:
            best_index = -1
            best_overlap = 0.0
            cue_midpoint = (cue.start_seconds + cue.end_seconds) / 2
            best_distance = float("inf")
            for index, sentence in enumerate(project.sentences):
                overlap = max(
                    0.0,
                    min(cue.end_seconds, sentence.end_seconds)
                    - max(cue.start_seconds, sentence.start_seconds),
                )
                distance = abs(cue_midpoint - (sentence.start_seconds + sentence.end_seconds) / 2)
                if overlap > best_overlap or (
                    overlap == best_overlap and overlap > 0 and distance < best_distance
                ):
                    best_index = index
                    best_overlap = overlap
                    best_distance = distance
            if best_index < 0:
                nearest = min(
                    range(len(project.sentences)),
                    key=lambda index: abs(
                        cue_midpoint
                        - (
                            project.sentences[index].start_seconds
                            + project.sentences[index].end_seconds
                        )
                        / 2
                    ),
                    default=-1,
                )
                if nearest >= 0:
                    distance = abs(
                        cue_midpoint
                        - (
                            project.sentences[nearest].start_seconds
                            + project.sentences[nearest].end_seconds
                        )
                        / 2
                    )
                    if distance <= 3.0:
                        best_index = nearest
            if best_index >= 0:
                assignments.setdefault(best_index, []).append(cue.zh_text)

        applied = 0
        for index, texts in assignments.items():
            text = " ".join(part.strip() for part in texts if part.strip()).strip()
            if text:
                project.sentences[index].zh_text = text
                project.sentences[index].status = "translated"
                applied += 1
        if applied:
            save_project(project, project_dir)
            export_transcript(project, project_dir)
        return applied
    except Exception as exc:
        raise VideoPreparerError(f"无法合并已有中文字幕：{exc}") from exc


def shift_srt_text(text: str, offset_ms: int) -> str:
    timestamp = re.compile(r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})")

    def replace(match: re.Match[str]) -> str:
        hours, minutes, seconds, millis = map(int, match.groups())
        total = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis + offset_ms
        total = max(0, total)
        out_hours, remainder = divmod(total, 3_600_000)
        out_minutes, remainder = divmod(remainder, 60_000)
        out_seconds, out_millis = divmod(remainder, 1000)
        return f"{out_hours:02d}:{out_minutes:02d}:{out_seconds:02d},{out_millis:03d}"

    lines = [timestamp.sub(replace, line) if "-->" in line else line for line in text.splitlines()]
    return "\n".join(lines) + "\n"


def shift_lrc_text(text: str, offset_ms: int) -> str:
    timestamp = re.compile(r"\[(\d+):(\d{2})(?:[.](\d{2,3}))?\]")

    def replace(match: re.Match[str]) -> str:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction_text = match.group(3) or "00"
        fraction_ms = int(fraction_text.ljust(3, "0")[:3])
        total = (minutes * 60 + seconds) * 1000 + fraction_ms + offset_ms
        total = max(0, total)
        out_minutes, remainder = divmod(total, 60_000)
        out_seconds, out_millis = divmod(remainder, 1000)
        return f"[{out_minutes:02d}:{out_seconds:02d}.{out_millis // 10:02d}]"

    return timestamp.sub(replace, text).rstrip("\n") + "\n"


def atomic_write_text(destination: Path, text: str, *, encoding: str = "utf-8-sig") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def copy_subtitles(
    project_json: Path,
    folder: Path,
    mode: str,
    *,
    harmonized_delay_seconds: int,
) -> tuple[Path, Path]:
    project = read_project(project_json)
    srt_source = project_asset(project_json, project.get("subtitle_srt_file"))
    lrc_source = project_asset(project_json, project.get("subtitle_lrc_file"))
    if srt_source is None or lrc_source is None:
        raise VideoPreparerError("ASMR Dubber 没有生成完整的 SRT/LRC 字幕。")

    srt_destination = folder / "双语版.srt"
    lrc_destination = folder / "双语版.lrc"
    if normalize_mode(mode) == MODE_VIDEO_HARMONIZED:
        offset_ms = harmonized_delay_seconds * 1000
        srt_text = srt_source.read_text(encoding="utf-8-sig")
        lrc_text = lrc_source.read_text(encoding="utf-8-sig")
        atomic_write_text(srt_destination, shift_srt_text(srt_text, offset_ms))
        atomic_write_text(lrc_destination, shift_lrc_text(lrc_text, offset_ms))
    else:
        atomic_copy(srt_source, srt_destination)
        atomic_copy(lrc_source, lrc_destination)
    return srt_destination, lrc_destination


def remux_video_with_subtitle(
    paths: ToolPaths,
    source: Path,
    subtitle_file: Path,
    destination: Path,
) -> None:
    """Convert an MKV fallback to MP4 while keeping a selectable subtitle track."""
    attempts = (
        ["-c:v", "copy"],
        [*paths.video_encoder_options, "-pix_fmt", "yuv420p"],
    )
    failures: list[str] = []
    for video_options in attempts:
        partial = partial_output_path(destination)
        try:
            run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "warning",
                    "-i",
                    str(source),
                    "-f",
                    "srt",
                    "-i",
                    str(subtitle_file),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-map",
                    "1:s:0",
                    *video_options,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-c:s",
                    "mov_text",
                    "-metadata:s:s:0",
                    "language=zho",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
            )
            os.replace(partial, destination)
            return
        except VideoPreparerError as exc:
            failures.append(str(exc))
        finally:
            partial.unlink(missing_ok=True)
    raise VideoPreparerError("无法把 ASMR Dubber 的字幕视频转换成 MP4：" + "；".join(failures))


def remux_video_without_subtitles(
    paths: ToolPaths,
    source: Path,
    destination: Path,
) -> None:
    """Create an MP4 with video and audio only, stripping every subtitle stream."""

    attempts = (
        ["-c:v", "copy"],
        [*paths.video_encoder_options, "-pix_fmt", "yuv420p"],
    )
    failures: list[str] = []
    for video_options in attempts:
        partial = partial_output_path(destination)
        try:
            run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "warning",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-sn",
                    *video_options,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
            )
            os.replace(partial, destination)
            return
        except VideoPreparerError as exc:
            failures.append(str(exc))
        finally:
            partial.unlink(missing_ok=True)
    raise VideoPreparerError("无法生成不含字幕的 MP4：" + "；".join(failures))


def copy_final_outputs(
    paths: ToolPaths,
    project_json: Path,
    folder: Path,
    mode: str,
    *,
    harmonized_delay_seconds: int,
    harmonized_volume_db: float,
    embed_subtitles: bool,
) -> dict[str, str]:
    project = read_project(project_json)
    mode = normalize_mode(mode)
    if mode == MODE_AUDIO:
        audio_source = project_asset(project_json, project.get("output_file"))
        if audio_source is None:
            raise VideoPreparerError("ASMR Dubber 没有生成可用的双语音频。")
        srt, lrc = copy_subtitles(
            project_json,
            folder,
            mode,
            harmonized_delay_seconds=harmonized_delay_seconds,
        )
        audio_suffix = audio_source.suffix if audio_source.suffix else ".wav"
        audio_destination = folder / f"双语版{audio_suffix}"
        print("  正在把双语音频送回原文件夹……")
        atomic_copy(audio_source, audio_destination)
        return {
            "audio": str(audio_destination),
            "srt": str(srt),
            "lrc": str(lrc),
        }

    mixed_audio_source: Path | None = None
    if mode == MODE_VIDEO_HARMONIZED:
        # Do not freeze the first frame of a hard-subtitled video for 20 minutes.
        # Delay the clean mixed video, then attach the already shifted subtitle.
        video_source = project_asset(project_json, project.get("output_video_file"))
        if video_source is None and embed_subtitles:
            video_source = project_asset(project_json, project.get("subtitle_video_file"))
        # Use ASMR Dubber's lossless mixed WAV as the final audio source. The
        # delayed harmonious version is then encoded to AAC only once.
        mixed_audio_source = project_asset(project_json, project.get("output_file"))
    elif embed_subtitles:
        video_source = project_asset(project_json, project.get("subtitle_video_file"))
        if video_source is None:
            video_source = project_asset(project_json, project.get("output_video_file"))
    else:
        video_source = project_asset(project_json, project.get("output_video_file"))
    if video_source is None:
        if not embed_subtitles:
            raise VideoPreparerError("ASMR Dubber 没有生成可用的无字幕双语视频。")
        raise VideoPreparerError("ASMR Dubber 没有生成可用的双语视频。")

    srt, lrc = copy_subtitles(
        project_json,
        folder,
        mode,
        harmonized_delay_seconds=harmonized_delay_seconds,
    )
    video_destination = folder / "双语版.mp4"
    print("  正在把双语视频送回原文件夹……")
    if mode == MODE_VIDEO_HARMONIZED:
        render_delayed_existing_video(
            paths,
            video_source,
            video_destination,
            lead_seconds=harmonized_delay_seconds,
            subtitle_file=srt if embed_subtitles else None,
            volume_db=harmonized_volume_db,
            audio_source=mixed_audio_source,
        )
    else:
        if not embed_subtitles:
            remux_video_without_subtitles(paths, video_source, video_destination)
        elif video_source.suffix.casefold() == ".mp4":
            atomic_copy(video_source, video_destination)
        else:
            remux_video_with_subtitle(paths, video_source, srt, video_destination)
    return {
        "video": str(video_destination),
        "srt": str(srt),
        "lrc": str(lrc),
    }


def concat_audio_files(
    paths: ToolPaths,
    files: Sequence[Path],
    workspace: Path,
    *,
    expected_samples: Sequence[int] | None = None,
    name: str = "combined",
) -> tuple[Path, list[int]]:
    """Normalize and concatenate already-produced audio files.

    ASMR Dubber normally preserves the source duration, but a backend can
    differ by a few samples.  When the source timeline is available we pad or
    trim each part to that exact length before concatenation, so the resulting
    subtitle offsets remain deterministic.
    """

    if not files:
        raise VideoPreparerError("没有可合并的音频文件。")
    if expected_samples is not None and len(expected_samples) != len(files):
        raise VideoPreparerError("合并音频的时长清单与文件数量不一致。")
    workspace.mkdir(parents=True, exist_ok=True)
    segments = workspace / f"{name}-segments"
    segments.mkdir(parents=True, exist_ok=True)
    actual: list[int] = []
    for index, source in enumerate(files, start=1):
        source = source.resolve()
        if not source.is_file():
            raise VideoPreparerError(f"找不到要合并的音频：{source}")
        destination = segments / f"seg_{index:06d}.flac"
        filters = [
            f"aresample={SAMPLE_RATE}:async=0",
            f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
        ]
        target = int(expected_samples[index - 1]) if expected_samples is not None else 0
        if target > 0:
            filters.extend((f"apad=whole_len={target}", f"atrim=end_sample={target}"))
        filters.append("asetpts=N/SR/TB")
        run_ffmpeg(
            paths,
            [
                "-loglevel",
                "warning",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                ",".join(filters),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-sample_fmt",
                "s16",
                "-c:a",
                "flac",
                "-compression_level",
                "0",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(destination),
            ],
        )
        measured = audio_duration_samples(paths, destination)
        if target > 0 and abs(measured - target) > 1:
            raise VideoPreparerError(
                f"无法把第 {index} 段音频规整到原始时长（{measured}/{target} 采样）。"
            )
        actual.append(measured)

    concat_file = workspace / f"{name}.ffconcat"
    concat_file.write_text(
        "ffconcat version 1.0\n"
        + "\n".join(
            f"file '{segments.name}/seg_{index:06d}.flac'" for index in range(1, len(files) + 1)
        )
        + "\n",
        encoding="ascii",
    )
    destination = workspace / f"{name}.flac"
    run_ffmpeg(
        paths,
        [
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            concat_file.name,
            "-map",
            "0:a:0",
            "-af",
            "asetpts=N/SR/TB",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            str(destination),
        ],
        cwd=workspace,
    )
    measured_total = audio_duration_samples(paths, destination)
    expected_total = sum(actual)
    if abs(measured_total - expected_total) > 1:
        raise VideoPreparerError(
            f"合并音频采样数异常：得到 {measured_total}，应为 {expected_total}。"
        )
    return destination, actual


def _srt_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for raw in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [line.rstrip("\r") for line in raw.splitlines()]
        if any("-->" in line for line in lines):
            blocks.append(lines)
    return blocks


def _srt_timestamp_ms(value: str) -> int:
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if match is None:
        raise ValueError(value)
    hours, minutes, seconds, millis = map(int, match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _format_srt_timestamp(milliseconds: int) -> str:
    total = max(0, int(milliseconds))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def combine_srt_files(
    entries: Sequence[tuple[Path, int] | tuple[Path, int, int]],
    destination: Path,
    *,
    final_offset_ms: int = 0,
) -> Path:
    """Merge SRT files, shifting and re-numbering cues in source order."""

    output: list[str] = []
    sequence = 1
    timing_pattern = re.compile(
        r"^(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
        r"(\d{2,}:\d{2}:\d{2}[,.]\d{3})(.*)$"
    )
    for entry in entries:
        source, offset_ms = entry[0], entry[1]
        clip_end_ms = entry[2] if len(entry) == 3 else None
        if not source.is_file():
            raise VideoPreparerError(f"找不到字幕文件：{source}")
        text = source.read_text(encoding="utf-8-sig")
        for block in _srt_blocks(text):
            timing_index = next(
                (index for index, line in enumerate(block) if "-->" in line),
                None,
            )
            if timing_index is None:
                continue
            shifted = shift_srt_text(
                block[timing_index] + "\n",
                int(offset_ms) + int(final_offset_ms),
            ).strip("\r\n")
            timing_match = timing_pattern.match(shifted)
            if timing_match is not None and clip_end_ms is not None:
                cue_start = _srt_timestamp_ms(timing_match.group(1))
                cue_end = _srt_timestamp_ms(timing_match.group(2))
                absolute_limit = int(clip_end_ms) + int(final_offset_ms)
                if cue_start >= absolute_limit:
                    continue
                cue_end = min(cue_end, absolute_limit)
                if cue_end <= cue_start:
                    continue
                shifted = (
                    f"{_format_srt_timestamp(cue_start)} --> "
                    f"{_format_srt_timestamp(cue_end)}{timing_match.group(3)}"
                )
            payload = [str(sequence), shifted, *block[timing_index + 1 :]]
            output.extend(payload)
            output.append("")
            sequence += 1
    if not output:
        raise VideoPreparerError("字幕文件中没有可合并的 SRT 时间轴。")
    atomic_write_text(destination, "\n".join(output).rstrip() + "\n")
    return destination


def combine_lrc_files(
    entries: Sequence[tuple[Path, int]],
    destination: Path,
    *,
    final_offset_ms: int = 0,
) -> Path:
    """Merge LRC files while retaining metadata from the first file only."""

    timestamp_line = re.compile(r"\[\d+:\d{2}(?:[.]\d{1,3})?\]")
    headers: list[str] = []
    timed_lines: list[str] = []
    for entry_index, (source, offset_ms) in enumerate(entries):
        if not source.is_file():
            raise VideoPreparerError(f"找不到字幕文件：{source}")
        text = source.read_text(encoding="utf-8-sig")
        shifted = shift_lrc_text(text, int(offset_ms) + int(final_offset_ms))
        for line in shifted.splitlines():
            if timestamp_line.search(line):
                timed_lines.append(line)
            elif entry_index == 0 and line.strip():
                headers.append(line)
    if not timed_lines:
        raise VideoPreparerError("字幕文件中没有可合并的 LRC 时间轴。")
    atomic_write_text(destination, "\n".join((*headers, *timed_lines)) + "\n")
    return destination


def render_static_bilingual_video(
    paths: ToolPaths,
    audio_source: Path,
    background: Path | None,
    subtitle_file: Path,
    destination: Path,
    *,
    lead_seconds: int = 0,
    volume_db: float = 0.0,
) -> None:
    """Render a lightweight static video, preferring visible hard subtitles."""

    clean_video = destination.with_name(f".{destination.stem}.clean-{uuid.uuid4().hex}.mp4")
    render_dir: Path | None = None
    try:
        render_static_video(
            paths,
            audio_source,
            background,
            clean_video,
            lead_seconds=lead_seconds,
            volume_db=volume_db,
        )
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        render_dir = Path(tempfile.mkdtemp(prefix="subtitle-render-", dir=WORK_ROOT))
        local_subtitle = render_dir / "subtitle.srt"
        shutil.copy2(subtitle_file, local_subtitle)
        partial = partial_output_path(destination)
        try:
            run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "warning",
                    "-i",
                    str(clean_video),
                    "-vf",
                    "subtitles=filename=subtitle.srt:charenc=UTF-8",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    *paths.video_encoder_options,
                    "-r",
                    str(VIDEO_FPS),
                    "-fps_mode",
                    "cfr",
                    "-c:a",
                    "copy",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
                cwd=render_dir,
            )
            os.replace(partial, destination)
        except VideoPreparerError:
            partial.unlink(missing_ok=True)
            print("  当前 FFmpeg 无法烧录字幕，改为写入可选择的内嵌字幕轨。")
            remux_video_with_subtitle(paths, clean_video, subtitle_file, destination)
        finally:
            partial.unlink(missing_ok=True)
    finally:
        clean_video.unlink(missing_ok=True)
        if render_dir is not None:
            shutil.rmtree(render_dir, ignore_errors=True)


def identifier_only_name(value: str) -> bool:
    """Return whether a folder name is only a catalogue/product identifier."""

    return bool(
        re.fullmatch(
            r"(?i)(?:rj|vj|bj|cien|ci-en)?[\s._-]*\d+",
            value.strip(),
        )
    )


def translate_titles(
    state: dict[str, Any],
    paths: ToolPaths,
) -> dict[str, str]:
    try:
        from asmr_dubber.models import Sentence
        from asmr_dubber.translation import translate_sentences
        from asmr_dubber.user_settings import (
            PROVIDER_PRESETS,
            load_user_settings,
            resolve_api_key,
        )
    except Exception as exc:
        raise VideoPreparerError(f"无法加载 ASMR Dubber 的翻译工具：{exc}") from exc

    try:
        settings = load_user_settings()
        provider = settings.translation_provider
        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            raise VideoPreparerError(f"ASMR Dubber 中选择了未知翻译服务：{provider}")
        model = settings.translation_model or str(preset.get("default_model") or "")
        base_url = settings.translation_base_url.strip() or str(preset.get("base_url") or "")
        temperature = settings.translation_temperature
        top_p = settings.translation_top_p
        max_tokens = settings.translation_max_output_tokens
        api_key = resolve_api_key(provider)
    except Exception as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 的翻译设置或密钥：{exc}") from exc

    cached = {
        str(key): str(value)
        for key, value in (state.get("title_translations") or {}).items()
        if str(value).strip()
    }
    folder_name_original = str(
        state.get("folder_name_original") or Path(state["source_folder"]).name
    ).strip()
    existing_folder_translation = str(state.get("folder_name_translation") or "").strip()
    if identifier_only_name(folder_name_original) and not existing_folder_translation:
        existing_folder_translation = folder_name_original
    folder_sentence = Sentence(
        id="folder0000",
        start_seconds=0.0,
        end_seconds=1.0,
        source_text=folder_name_original,
        zh_text=existing_folder_translation,
    )
    sentences = [folder_sentence]
    for index, item in enumerate(state["timeline"], start=1):
        filename = str(item.get("relative_path") or item["filename"])
        sentences.append(
            Sentence(
                id=f"title{index:04d}",
                start_seconds=float(index),
                end_seconds=float(index + 1),
                source_text=str(item["title_ja"]),
                zh_text=cached.get(filename, ""),
            )
        )
    source_languages = {
        str(item.get("source_language") or "ja").casefold() for item in state["timeline"]
    }
    source_language = "en" if source_languages == {"en"} else "ja"

    try:
        translate_sentences(
            sentences,
            api_key=api_key,
            provider=provider,
            source_language=source_language,
            model=model,
            base_url=base_url,
            system_prompt=TITLE_TRANSLATION_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            deepl_formality=settings.translation_deepl_formality,
            microsoft_region=settings.translation_microsoft_region,
            send_context=True,
            context_sentences=100,
            memory_sentences=50,
            job_id=f"video_preparer_{task_key(Path(state['source_folder']))}",
            progress=lambda message, current, total: print(f"  {message}"),
        )
    except Exception as exc:
        if folder_sentence.zh_text.strip():
            state["folder_name_translation"] = folder_sentence.zh_text.strip()
        for item, sentence in zip(state["timeline"], sentences[1:], strict=True):
            if sentence.zh_text.strip():
                key = str(item.get("relative_path") or item["filename"])
                cached[key] = sentence.zh_text.strip()
        state["title_translations"] = cached
        raise VideoPreparerError(f"文件夹名称与短音频标题翻译失败：{exc}") from exc

    translated: dict[str, str] = {}
    missing: list[str] = []
    folder_name_translation = folder_sentence.zh_text.strip() or folder_name_original
    state["folder_name_original"] = folder_name_original
    state["folder_name_translation"] = folder_name_translation
    for item, sentence in zip(state["timeline"], sentences[1:], strict=True):
        title = sentence.zh_text.strip()
        key = str(item.get("relative_path") or item["filename"])
        if not title:
            missing.append(key)
            continue
        translated[key] = title
    state["title_translations"] = translated
    if missing:
        raise VideoPreparerError("以下名称或标题翻译为空：" + "、".join(missing))
    return translated


def translated_plan_titles(
    plan_id: str,
    source_folder: Path,
    sources: list[AudioSource],
    paths: ToolPaths,
    *,
    translate_work_title: bool = True,
    translate_track_titles: bool = True,
) -> tuple[str, dict[str, str]]:
    cached = load_plan_metadata(plan_id)
    expected_keys = [item.relative_path or item.path.name for item in sources]
    original_titles = {
        key: (source.title_ja.strip() or Path(key).stem)
        for key, source in zip(expected_keys, sources, strict=True)
    }
    policy = {
        "work": bool(translate_work_title),
        "tracks": bool(translate_track_titles),
    }
    cached_policy = cached.get("title_translation_policy")
    policy_matches = not isinstance(cached_policy, dict) or cached_policy == policy
    translations = (
        {
            str(key): str(value)
            for key, value in (cached.get("title_translations") or {}).items()
            if str(value).strip()
        }
        if policy_matches and translate_track_titles
        else dict(original_titles)
        if not translate_track_titles
        else {}
    )
    folder_translation = (
        str(cached.get("folder_name_translation") or "").strip()
        if policy_matches and translate_work_title
        else source_folder.name
        if not translate_work_title
        else ""
    )
    inferred_folder_translation = False
    if not folder_translation and identifier_only_name(source_folder.name):
        folder_translation = source_folder.name
        inferred_folder_translation = True
    if folder_translation and all(
        str(translations.get(key) or "").strip() for key in expected_keys
    ):
        if inferred_folder_translation or cached_policy != policy:
            updated_cache = dict(cached)
            updated_cache["folder_name_original"] = source_folder.name
            updated_cache["folder_name_translation"] = folder_translation
            updated_cache["title_translations"] = translations
            updated_cache["title_translation_policy"] = policy
            save_plan_metadata(plan_id, updated_cache)
        return folder_translation, translations

    state: dict[str, Any] = {
        "source_folder": str(source_folder),
        "folder_name_original": source_folder.name,
        "folder_name_translation": folder_translation,
        "title_translations": translations,
        "timeline": [
            {
                "filename": item.path.name,
                "relative_path": item.relative_path or item.path.name,
                "title_ja": item.title_ja,
                "source_language": item.source_language,
            }
            for item in sources
        ],
    }
    try:
        translations = translate_titles(state, paths)
    except VideoPreparerError:
        save_plan_metadata(
            plan_id,
            {
                "source_folder": str(source_folder),
                "folder_name_original": source_folder.name,
                "folder_name_translation": str(state.get("folder_name_translation") or ""),
                "title_translations": dict(state.get("title_translations") or {}),
                "title_translation_policy": policy,
            },
        )
        raise
    folder_translation = str(state["folder_name_translation"])
    save_plan_metadata(
        plan_id,
        {
            "source_folder": str(source_folder),
            "folder_name_original": source_folder.name,
            "folder_name_translation": folder_translation,
            "title_translations": translations,
            "title_translation_policy": policy,
        },
    )
    return folder_translation, translations


def bracketed(value: str) -> str:
    text = value.strip()
    if text.startswith("【") and text.endswith("】"):
        return text
    return f"【{text}】"


def format_timestamp_from_samples(samples: int) -> str:
    # 向下取整，保证显示时间不会晚于音频真正开始的时刻。
    seconds = max(0, samples // SAMPLE_RATE)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_timestamp_document(state: dict[str, Any], folder: Path) -> Path:
    translations = state.get("title_translations") or {}
    folder_name_original = str(
        state.get("folder_name_original") or Path(state["source_folder"]).name
    ).strip()
    folder_name_translation = str(state.get("folder_name_translation") or "").strip()
    if not folder_name_translation:
        raise VideoPreparerError("缺少文件夹名称的中文翻译。")
    offset_samples = (
        int(state.get("harmonized_delay_seconds") or 0) * SAMPLE_RATE
        if normalize_mode(state["mode"]) == MODE_VIDEO_HARMONIZED
        else 0
    )
    lines: list[str] = [
        f"中文名称：{folder_name_translation}",
        f"原始名称：{folder_name_original}",
        "",
    ]
    for item in state["timeline"]:
        filename = str(item.get("relative_path") or item["filename"])
        chinese = str(translations.get(filename, "")).strip()
        if not chinese:
            raise VideoPreparerError(f"缺少中文标题：{filename}")
        start = int(item["start_samples"]) + offset_samples
        lines.append(f"{format_timestamp_from_samples(start)} {bracketed(chinese)}")
        lines.append(bracketed(str(item["title_ja"])))
        lines.append("")
    stored_footer = state.get("timestamp_footer")
    footer = DEFAULT_TIMESTAMP_FOOTER if stored_footer is None else str(stored_footer).strip()
    if footer:
        lines.append(footer)
    destination = folder / "时间戳.txt"
    atomic_write_text(destination, "\n".join(lines).rstrip() + "\n")
    return destination


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


def expected_output_paths(folder: Path, mode: str) -> tuple[Path, ...]:
    mode = normalize_mode(mode)
    primary = (
        (folder / "原声.flac", folder / "双语版.wav", folder / "双语版.flac")
        if mode == MODE_AUDIO
        else (folder / "原声.mp4", folder / "双语版.mp4")
    )
    return (
        *primary,
        folder / "双语版.srt",
        folder / "双语版.lrc",
        folder / "时间戳.txt",
    )


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


def create_initial_state(
    folder: Path,
    mode: str,
    sources: list[AudioSource],
    background: Path | None,
    config: AppConfig,
    *,
    embed_subtitles: bool = True,
) -> dict[str, Any]:
    mode = normalize_mode(mode)
    return {
        "schema": 1,
        "source_folder": str(folder),
        "mode": mode,
        "harmonized_volume_db": config.harmonized_volume_db,
        "harmonized_delay_seconds": config.harmonized_delay_seconds,
        "timestamp_footer": config.timestamp_footer,
        "reference_wait_seconds": max(0, int(config.reference_wait_seconds)),
        "status": "",
        "fingerprint": fingerprint(sources, background),
        "background": str(background) if background else None,
        "embed_subtitles": bool(embed_subtitles),
        "workspace": str(workspace_path(folder)),
        "timeline": [],
        "title_translations": {},
        "folder_name_original": folder.name,
        "folder_name_translation": "",
        "timestamp_schema": 0,
        "outputs": {},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def create_planned_state(
    source_folder: Path,
    output_folder: Path,
    mode: str,
    sources: list[AudioSource],
    background: Path | None,
    config: AppConfig,
    *,
    plan_id: str,
    job_id: str,
    folder_translation: str,
    title_translations: dict[str, str],
    embed_subtitles: bool,
) -> dict[str, Any]:
    state = create_initial_state(
        output_folder,
        mode,
        sources,
        background,
        config,
        embed_subtitles=embed_subtitles,
    )
    state.update(
        {
            "source_folder": str(source_folder),
            "output_folder": str(output_folder),
            "workspace": str(planned_workspace_path(source_folder, plan_id, job_id)),
            "plan_id": plan_id,
            "job_id": job_id,
            "folder_name_original": source_folder.name,
            "folder_name_translation": folder_translation,
            "title_translations": {
                str(key): str(value) for key, value in title_translations.items()
            },
        }
    )
    return state


def execute_planned_job(
    paths: ToolPaths,
    config: AppConfig,
    *,
    source_folder: Path,
    output_folder: Path,
    mode: str,
    sources: list[AudioSource],
    background: Path | None,
    plan_id: str,
    job_id: str,
    folder_translation: str,
    title_translations: dict[str, str],
    embed_subtitles: bool,
    rebuild: bool,
    shared_reference: dict[str, str] | None = None,
    capture_reference_path: Path | None = None,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> dict[str, Any]:
    output_folder.mkdir(parents=True, exist_ok=True)
    state_file = planned_state_path(source_folder, plan_id, job_id)
    state = None if rebuild else load_state(state_file)
    current_fingerprint = fingerprint(sources, background)

    if state is not None and not rebuild:
        if state.get("fingerprint") != current_fingerprint:
            raise VideoPreparerError(
                f"{output_folder.name} 的源音频、字幕或背景发生变化；请使用 --rebuild 重做。"
            )
        if normalize_mode(mode) != MODE_AUDIO and bool(state.get("embed_subtitles", True)) != bool(
            embed_subtitles
        ):
            raise VideoPreparerError(
                f"{output_folder.name} 的字幕内嵌设置发生变化；请使用 --rebuild 重做。"
            )
        missing = missing_resume_artifacts(state, output_folder)
        if missing:
            timestamp_file = output_folder / "时间戳.txt"
            if missing == [timestamp_file] and status_at_least(state, "outputs_ready"):
                print(f"\n{output_folder.name} 只缺时间戳文档，正在直接补写。")
                state["status"] = "outputs_ready"
                save_state(state_file, state)
            else:
                print(f"\n{output_folder.name} 的旧任务缺少文件，将从头重做。")
                rebuild = True
        elif legacy_chinese_timeline_needs_reimport(state):
            print(f"\n{output_folder.name} 将按已有中文字幕重建时间轴，不会重跑 ASR 或翻译。")
            reset_legacy_chinese_timeline_state(state)
            save_state(state_file, state)
        elif status_at_least(state, "completed"):
            print(f"\n跳过已经完成的任务：{output_folder.name}")
            return state
        else:
            print(
                f"\n继续未完成任务：{output_folder.name} "
                f"（阶段：{state.get('status') or '尚未开始'}）"
            )

    if state is None or rebuild:
        workspace = planned_workspace_path(source_folder, plan_id, job_id)
        safe_reset_workspace(workspace)
        state = create_planned_state(
            source_folder,
            output_folder,
            mode,
            sources,
            background,
            config,
            plan_id=plan_id,
            job_id=job_id,
            folder_translation=folder_translation,
            title_translations=title_translations,
            embed_subtitles=embed_subtitles,
        )
        save_state(state_file, state)
    elif not status_at_least(state, "synthesized"):
        # Waiting is a current run preference rather than an immutable media
        # setting. This also lets older resumable jobs adopt the WebUI default.
        state["reference_wait_seconds"] = max(0, int(config.reference_wait_seconds))
        save_state(state_file, state)

    execute_task(
        paths,
        output_folder,
        state_file,
        state,
        sources,
        shared_reference=shared_reference,
        capture_reference_path=capture_reference_path,
        reference_event_callback=reference_event_callback,
    )
    return load_state(state_file) or state


def _single_track_source(source: AudioSource) -> AudioSource:
    return replace(source, order=1)


def _smart_track_folder_name(index: int, source: AudioSource) -> str:
    title = safe_filename_component(source.title_ja, fallback=f"音轨 {index}", limit=56)
    return f"{index:03d} {title}"


def clear_smart_outputs(output_root: Path) -> None:
    """Remove only files AutoFlow owns inside its output directory."""

    output_root = output_root.resolve()
    if output_root == output_root.parent:
        raise VideoPreparerError("拒绝清理文件系统根目录。")
    for name in ("合并版", "分轨", ".autoflow"):
        target = output_root / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for name in ("处理清单.json", "曲目清单.txt", "总时间戳.txt"):
        (output_root / name).unlink(missing_ok=True)


def cleanup_completed_workspace(state: dict[str, Any]) -> None:
    workspace = Path(str(state.get("workspace") or "")).expanduser()
    if not workspace.exists():
        return
    root = WORK_ROOT.resolve()
    target = workspace.resolve()
    if target.parent == root and status_at_least(state, "completed"):
        shutil.rmtree(target, ignore_errors=True)


def prepare_smart_output_root(
    output_root: Path,
    *,
    plan_id: str,
    force: bool,
    rebuild: bool,
) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise VideoPreparerError(f"输出路径已经被同名文件占用：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
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
    needs_clear = rebuild or (generated_exists and previous_plan != plan_id)
    if not needs_clear:
        return
    if not force and generated_exists:
        print("\nAutoFlow 输出目录中已有另一项任务的文件：")
        print(f"  {output_root}")
        answer = input("清理 AutoFlow 自己生成的旧内容并继续吗？输入 Y 确认：").strip().casefold()
        if answer != "y":
            raise KeyboardInterrupt
    clear_smart_outputs(output_root)


def smart_job_descriptors(
    output_root: Path,
    layout: str,
    sources: Sequence[AudioSource],
) -> list[dict[str, Any]]:
    layout = normalize_layout(layout)
    descriptors: list[dict[str, Any]] = []
    if layout in {LAYOUT_SEPARATE, LAYOUT_BOTH}:
        for index, source in enumerate(sources, start=1):
            name = _smart_track_folder_name(index, source)
            descriptors.append(
                {
                    "job_id": f"track-{index:04d}",
                    "kind": "track",
                    "index": index,
                    "source_indices": [index],
                    "output": str((output_root / "分轨" / name).resolve()),
                    "label": source.title_ja,
                }
            )
    if layout in {LAYOUT_MERGED, LAYOUT_BOTH}:
        descriptors.append(
            {
                "job_id": "merged",
                "kind": "merged",
                "index": 0,
                "source_indices": list(range(1, len(sources) + 1)),
                "output": str((output_root / "合并版").resolve()),
                "label": "合并版",
            }
        )
    return descriptors


def _state_master_audio(
    paths: ToolPaths,
    state: dict[str, Any],
    source: AudioSource,
    workspace: Path,
) -> Path:
    candidate = Path(str(state.get("master_audio") or "")).expanduser()
    if candidate.is_file():
        return candidate
    # A user may have removed .work after a successful run. Recreate only the
    # normalized single-track master needed for a later merge; ASR/TTS remain
    # untouched.
    rebuilt_workspace = workspace / f"rebuild-{source.order:04d}"
    rebuilt_workspace.mkdir(parents=True, exist_ok=True)
    rebuilt, _ = normalize_and_concat(paths, [_single_track_source(source)], rebuilt_workspace)
    return rebuilt


def _state_mixed_audio(
    state: dict[str, Any],
    project_json: Path | None = None,
) -> Path:
    candidate = Path(str(state.get("project_mixed_audio") or "")).expanduser()
    if candidate.is_file():
        return candidate
    if project_json is not None and project_json.is_file():
        project = read_project(project_json)
        candidate = project_asset(project_json, project.get("output_file"))
        if candidate is not None:
            return candidate
    outputs = state.get("outputs") or {}
    candidate = Path(str(outputs.get("audio") or "")).expanduser()
    if candidate.is_file():
        return candidate
    raise VideoPreparerError("ASMR Dubber 已完成，但找不到可合并的双语音频。")


def _timeline_from_states(states: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    offset = 0
    for state in states:
        for item in state.get("timeline") or []:
            copied = dict(item)
            copied["start_samples"] = int(item.get("start_samples") or 0) + offset
            relative = str(item.get("relative_path") or item.get("filename") or "")
            copied["relative_path"] = relative
            merged.append(copied)
        offset += sum(
            int(item.get("duration_samples") or 0) for item in state.get("timeline") or []
        )
    return merged


def _subtitle_entries_from_states(
    states: Sequence[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int]], int]:
    srt_entries: list[tuple[Path, int, int]] = []
    lrc_entries: list[tuple[Path, int]] = []
    offset_samples = 0
    delay_samples = (
        int(states[0].get("harmonized_delay_seconds") or 0) * SAMPLE_RATE
        if normalize_mode(mode) == MODE_VIDEO_HARMONIZED
        else 0
    )
    for state in states:
        outputs = state.get("outputs") or {}
        srt = Path(str(outputs.get("srt") or "")).expanduser()
        lrc = Path(str(outputs.get("lrc") or "")).expanduser()
        # Per-track harmonious outputs already contain their own delay. Remove
        # it here; the merged product receives exactly one delay at the end.
        per_track_delay = (
            delay_samples if normalize_mode(state["mode"]) == MODE_VIDEO_HARMONIZED else 0
        )
        shift = offset_samples - per_track_delay
        duration_samples = sum(
            int(item.get("duration_samples") or 0) for item in state.get("timeline") or []
        )
        srt_entries.append(
            (
                srt,
                shift * 1000 // SAMPLE_RATE,
                (offset_samples + duration_samples) * 1000 // SAMPLE_RATE,
            )
        )
        lrc_entries.append((lrc, shift * 1000 // SAMPLE_RATE))
        offset_samples += duration_samples
    return srt_entries, lrc_entries, delay_samples * 1000 // SAMPLE_RATE


def build_merged_outputs(
    paths: ToolPaths,
    config: AppConfig,
    *,
    source_folder: Path,
    output_folder: Path,
    mode: str,
    sources: Sequence[AudioSource],
    states: Sequence[dict[str, Any]],
    background: Path | None,
    folder_translation: str,
    title_translations: dict[str, str],
    embed_subtitles: bool,
) -> dict[str, str]:
    """Build the merged product from completed per-track projects."""

    mode = normalize_mode(mode)
    output_folder.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / f"merge-{task_key(output_folder)}"
    safe_reset_workspace(work)
    expected = [
        sum(int(item.get("duration_samples") or 0) for item in state.get("timeline") or [])
        for state in states
    ]
    masters = [
        _state_master_audio(paths, state, source, work)
        for source, state in zip(sources, states, strict=True)
    ]
    original_master, _ = concat_audio_files(
        paths,
        masters,
        work,
        expected_samples=expected,
        name="original",
    )
    mixed_files = [
        _state_mixed_audio(state, Path(str(state.get("project_json") or ""))) for state in states
    ]
    mixed_master, _ = concat_audio_files(
        paths,
        mixed_files,
        work,
        expected_samples=expected,
        name="mixed",
    )

    timeline = _timeline_from_states(states)
    merged_state: dict[str, Any] = {
        "source_folder": str(source_folder),
        "mode": mode,
        "harmonized_delay_seconds": int(config.harmonized_delay_seconds),
        "timestamp_footer": config.timestamp_footer,
        "timeline": timeline,
        "title_translations": dict(title_translations),
        "folder_name_original": source_folder.name,
        "folder_name_translation": folder_translation,
    }
    srt_entries, lrc_entries, final_offset_ms = _subtitle_entries_from_states(
        states,
        mode=mode,
    )
    srt = combine_srt_files(
        srt_entries,
        output_folder / "双语版.srt",
        final_offset_ms=final_offset_ms,
    )
    lrc = combine_lrc_files(
        lrc_entries,
        output_folder / "双语版.lrc",
        final_offset_ms=final_offset_ms,
    )

    if mode == MODE_AUDIO:
        original_destination = output_folder / "原声.flac"
        mixed_destination = output_folder / "双语版.flac"
        atomic_copy(original_master, original_destination)
        atomic_copy(mixed_master, mixed_destination)
        outputs = {
            "original": str(original_destination),
            "audio": str(mixed_destination),
            "srt": str(srt),
            "lrc": str(lrc),
        }
    else:
        original_destination = output_folder / "原声.mp4"
        bilingual_destination = output_folder / "双语版.mp4"
        lead = int(config.harmonized_delay_seconds) if mode == MODE_VIDEO_HARMONIZED else 0
        volume = config.harmonized_volume_db if mode == MODE_VIDEO_HARMONIZED else 0.0
        render_static_video(
            paths,
            original_master,
            background,
            original_destination,
            lead_seconds=lead,
            volume_db=volume,
        )
        if embed_subtitles:
            render_static_bilingual_video(
                paths,
                mixed_master,
                background,
                srt,
                bilingual_destination,
                lead_seconds=lead,
                volume_db=volume,
            )
        else:
            render_static_video(
                paths,
                mixed_master,
                background,
                bilingual_destination,
                lead_seconds=lead,
                volume_db=volume,
            )
        outputs = {
            "original": str(original_destination),
            "video": str(bilingual_destination),
            "srt": str(srt),
            "lrc": str(lrc),
        }
    timestamp = write_timestamp_document(merged_state, output_folder)
    outputs["timestamps"] = str(timestamp)
    shutil.rmtree(work, ignore_errors=True)
    return outputs


def write_smart_summary(
    output_root: Path,
    *,
    source_folder: Path,
    mode: str,
    layout: str,
    sources: Sequence[AudioSource],
    states: Sequence[dict[str, Any]],
    descriptors: Sequence[dict[str, Any]],
    folder_translation: str,
    title_translations: dict[str, str],
    footer: str,
    harmonized_delay_seconds: int,
) -> tuple[Path, Path]:
    """Write a human-readable track index and a timeline reference."""

    output_root.mkdir(parents=True, exist_ok=True)
    descriptor_by_id = {str(item["job_id"]): item for item in descriptors}
    lines = [
        f"中文名称：{folder_translation}",
        f"原始名称：{source_folder.name}",
        f"模式：{mode_label(mode)}",
        f"成品组织：{layout_label(layout)}",
        "",
        "曲目与输出：",
    ]
    timeline_lines = [
        f"中文名称：{folder_translation}",
        f"原始名称：{source_folder.name}",
        "",
    ]
    cumulative = 0
    timeline_offset = (
        harmonized_delay_seconds * SAMPLE_RATE
        if normalize_mode(mode) == MODE_VIDEO_HARMONIZED
        else 0
    )
    for index, (source, state) in enumerate(zip(sources, states, strict=True), start=1):
        key = source.relative_path or source.path.name
        translated = str(title_translations.get(key) or "未翻译").strip()
        descriptor = descriptor_by_id.get(f"track-{index:04d}")
        track_output = ""
        if descriptor is not None:
            try:
                track_output = Path(str(descriptor["output"])).relative_to(output_root).as_posix()
            except ValueError:
                track_output = str(descriptor["output"])
        duration = sum(
            int(item.get("duration_samples") or 0) for item in state.get("timeline") or []
        )
        lines.extend(
            (
                f"{index:03d}. {source.title_ja}",
                f"     中文：{translated}",
                f"     原文件：{source.relative_path or source.path.name}",
                f"     输出：{track_output or '仅合并版'}",
                "",
            )
        )
        timeline_lines.append(
            f"{format_timestamp_from_samples(cumulative + timeline_offset)} {bracketed(translated)}"
        )
        timeline_lines.extend((bracketed(source.title_ja), ""))
        cumulative += duration

    if layout in {LAYOUT_SEPARATE, LAYOUT_BOTH}:
        lines.extend(
            (
                "说明：分轨文件各自从 00:00 开始；上面的时间轴按原曲目顺序计算，",
                "仅用于查找，不代表把分轨文件直接无缝播放时的播放器时间。",
                "",
            )
        )
    if footer.strip():
        lines.append(footer.strip())
        timeline_lines.append(footer.strip())
    index_file = output_root / "曲目清单.txt"
    timeline_file = output_root / "总时间戳.txt"
    atomic_write_text(index_file, "\n".join(lines).rstrip() + "\n")
    atomic_write_text(timeline_file, "\n".join(timeline_lines).rstrip() + "\n")
    return index_file, timeline_file


def output_mapping_complete(outputs: Any, *, mode: str) -> bool:
    if not isinstance(outputs, dict):
        return False
    required = {"original", "srt", "lrc", "timestamps"}
    required.add("audio" if normalize_mode(mode) == MODE_AUDIO else "video")
    return all(Path(str(outputs.get(key) or "")).is_file() for key in required)


TRANSCRIPT_LANGUAGE_LABELS = {
    "zh": "中文台本/字幕",
    "ja": "日文台本/字幕",
    "en": "英文台本/字幕",
    "ignore": "不使用这份台本",
    "auto": "按文件名自动判断",
}


def transcript_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def default_transcript_language(path: Path) -> str:
    if path.suffix.casefold() == ".pdf":
        return "ignore"
    return "zh"


def default_transcript_choices(scan: ScanResult) -> dict[str, str]:
    return {
        transcript_key(item.path): default_transcript_language(item.path)
        for item in scan.transcripts
    }


def effective_sources_for_draft(draft: SmartPlanDraft) -> list[AudioSource]:
    effective: list[AudioSource] = []
    for source in draft.sources:
        if source.transcript_path is None:
            effective.append(source)
            continue
        choice = draft.transcript_choices.get(
            transcript_key(source.transcript_path),
            default_transcript_language(source.transcript_path),
        )
        if choice == "ignore":
            effective.append(
                replace(
                    source,
                    transcript_path=None,
                    transcript_language=None,
                    transcript_timed=False,
                )
            )
        elif choice == "auto":
            effective.append(source)
        elif choice in {"zh", "ja", "en"}:
            effective.append(replace(source, transcript_language=choice))
        else:
            raise VideoPreparerError(f"未知台本语言选择：{choice}")
    return effective


def transcript_rows_for_draft(
    draft: SmartPlanDraft,
) -> list[tuple[Any, list[str]]]:
    matched: dict[str, list[str]] = {}
    for source in draft.sources:
        if source.transcript_path is None:
            continue
        matched.setdefault(transcript_key(source.transcript_path), []).append(
            source.relative_path or source.path.name
        )
    return [
        (transcript, matched.get(transcript_key(transcript.path), []))
        for transcript in draft.scan.transcripts
    ]


def transcript_choice_label(
    draft: SmartPlanDraft,
    transcript: Any,
) -> str:
    choice = draft.transcript_choices.get(
        transcript_key(transcript.path),
        default_transcript_language(transcript.path),
    )
    if choice == "auto":
        detected = str(transcript.language or "ja").casefold()
        detected_label = TRANSCRIPT_LANGUAGE_LABELS.get(detected, detected)
        return f"{TRANSCRIPT_LANGUAGE_LABELS['auto']}（当前{detected_label}）"
    return TRANSCRIPT_LANGUAGE_LABELS.get(choice, choice)


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


def media_type_label(mode: str) -> str:
    return "纯音频" if normalize_mode(mode) == MODE_AUDIO else "静态视频"


def video_branch_label(mode: str, config: AppConfig) -> str:
    normalized = normalize_mode(mode)
    if normalized == MODE_VIDEO_HARMONIZED:
        return (
            f"和谐（{config.harmonized_volume_db:g} dB，"
            f"延后 {config.harmonized_delay_seconds / 60:g} 分钟）"
        )
    return "普通（原音量，无前置空档）"


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
    print(f"  成品组织：{layout_label(plan.layout)}")
    if plan.mode != MODE_AUDIO:
        print(f"  背景图片：{plan.background.name if plan.background else '黑色背景'}")
        subtitle_mode = (
            "内嵌，同时保留 SRT/LRC" if plan.embed_subtitles else "不内嵌，仅保留 SRT/LRC"
        )
        print(f"  视频字幕：{subtitle_mode}")
    print(f"  输出目录：{plan.output_root}")


def execute_prepared_smart_plan(
    paths: ToolPaths,
    config: AppConfig,
    plan: SmartTaskPlan,
    *,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> None:
    """Execute a previously configured smart task without asking plan questions."""

    folder = plan.folder
    output_root = plan.output_root
    edition_label = plan.edition_label
    sources = list(plan.sources)
    edition = plan.edition
    mode = plan.mode
    layout = plan.layout
    background = plan.background
    embed_subtitles = plan.embed_subtitles
    plan_id = plan.plan_id
    rebuild = plan.rebuild
    force = plan.force
    log_event(
        f"开始智能任务 plan={plan_id} source={folder} mode={mode} layout={layout} "
        f"tracks={len(sources)} embed_subtitles={embed_subtitles}"
    )
    prepare_smart_output_root(output_root, plan_id=plan_id, force=force, rebuild=rebuild)

    print(f"\n已选择：{edition_label}")
    print(f"扫描到 {len(sources)} 条音轨；输出目录：{output_root}")
    if mode != MODE_AUDIO:
        if background is not None:
            print(f"背景图片：{background.name}")
        else:
            print("背景图片：黑色背景")
        print(f"视频字幕：{'内嵌' if embed_subtitles else '不内嵌（外部 SRT/LRC 仍保留）'}")

    folder_translation, title_translations = translated_plan_titles(
        plan_id,
        folder,
        list(sources),
        paths,
        translate_work_title=plan.translate_work_title,
        translate_track_titles=plan.translate_track_titles,
    )
    descriptors = smart_job_descriptors(output_root, layout, sources)
    manifest = write_plan_manifest(
        output_root / "处理清单.json",
        source_folder=folder,
        output_folder=output_root,
        mode=mode,
        layout=layout,
        edition=edition,
        sources=list(sources),
        background=background,
        embed_subtitles=embed_subtitles,
        plan_id=plan_id,
        jobs=descriptors,
    )
    metadata = load_plan_metadata(plan_id)
    previous_jobs = list(metadata.get("jobs") or [])
    previous_by_id = {
        str(item.get("job_id")): item
        for item in previous_jobs
        if isinstance(item, dict) and item.get("job_id")
    }
    for descriptor in descriptors:
        previous = previous_by_id.get(str(descriptor["job_id"]))
        if previous is None:
            continue
        for key in ("status", "outputs", "project_json"):
            if key in previous:
                descriptor[key] = previous[key]
    metadata.update(
        {
            "source_folder": str(folder),
            "output_folder": str(output_root),
            "mode": mode,
            "layout": layout,
            "edition": edition,
            "background": str(background) if background else None,
            "embed_subtitles": embed_subtitles,
            "jobs": descriptors,
        }
    )
    save_plan_metadata(plan_id, metadata)

    states_by_index: dict[int, dict[str, Any]] = {}
    shared_reference = metadata.get("shared_reference")
    if (
        not isinstance(shared_reference, dict)
        or not Path(str(shared_reference.get("audio") or "")).is_file()
    ):
        shared_reference = None
    shared_path = output_root / ".autoflow" / "shared-reference.wav"

    if layout == LAYOUT_MERGED:
        descriptor = next(item for item in descriptors if item["kind"] == "merged")
        merged_folder = Path(str(descriptor["output"]))
        state = execute_planned_job(
            paths,
            config,
            source_folder=folder,
            output_folder=merged_folder,
            mode=mode,
            sources=list(sources),
            background=background,
            plan_id=plan_id,
            job_id="merged",
            folder_translation=folder_translation,
            title_translations=title_translations,
            embed_subtitles=embed_subtitles,
            rebuild=rebuild,
            reference_event_callback=reference_event_callback,
        )
        states_by_index = {index: state for index in range(1, len(sources) + 1)}
        states = [state]
    else:
        # Let the longest source establish the shared voice anchor. Processing
        # order is otherwise restored in the original track order for output.
        durations = [audio_duration_samples(paths, item.path) for item in sources]
        anchor_index = max(range(len(sources)), key=lambda index: (durations[index], -index))
        processing_order = [
            anchor_index,
            *[index for index in range(len(sources)) if index != anchor_index],
        ]
        print(f"\n分轨模式先处理最长音轨（第 {anchor_index + 1} 条）以建立共用参考。")
        for zero_index in processing_order:
            index = zero_index + 1
            descriptor = next(item for item in descriptors if item.get("index") == index)
            track_source = _single_track_source(sources[zero_index])
            capture = shared_path if shared_reference is None else None
            state = execute_planned_job(
                paths,
                config,
                source_folder=folder,
                output_folder=Path(str(descriptor["output"])),
                mode=mode,
                sources=[track_source],
                background=background,
                plan_id=plan_id,
                job_id=str(descriptor["job_id"]),
                folder_translation=folder_translation,
                title_translations=title_translations,
                embed_subtitles=embed_subtitles,
                rebuild=rebuild,
                shared_reference=shared_reference,
                capture_reference_path=capture,
                reference_event_callback=reference_event_callback,
            )
            states_by_index[index] = state
            if shared_reference is None:
                candidate = state.get("shared_reference")
                if (
                    isinstance(candidate, dict)
                    and Path(str(candidate.get("audio") or "")).is_file()
                ):
                    shared_reference = candidate
                else:
                    project_json = Path(str(state.get("project_json") or ""))
                    if project_json.is_file() and capture is not None:
                        shared_path.parent.mkdir(parents=True, exist_ok=True)
                        shared_reference = extract_shared_reference(project_json, shared_path)
                        state["shared_reference"] = shared_reference
                        save_state(
                            planned_state_path(folder, plan_id, str(descriptor["job_id"])), state
                        )
                if shared_reference is not None:
                    metadata["shared_reference"] = shared_reference
                    save_plan_metadata(plan_id, metadata)
        states = [states_by_index[index] for index in range(1, len(sources) + 1)]

    if layout == LAYOUT_BOTH:
        merged_descriptor = next(item for item in descriptors if item["kind"] == "merged")
        previous_merged = next(
            (item for item in previous_jobs if item.get("kind") == "merged"),
            {},
        )
        if (
            not rebuild
            and str(previous_merged.get("status") or "") == "completed"
            and output_mapping_complete(previous_merged.get("outputs"), mode=mode)
        ):
            print("\n合并成品已经完整，跳过重复生成。")
            merged_descriptor["outputs"] = dict(previous_merged["outputs"])
            merged_descriptor["status"] = "completed"
        else:
            print("\n分轨项目已完成，正在从分轨结果生成合并成品（不会再次 ASR/TTS）……")
            merged_outputs = build_merged_outputs(
                paths,
                config,
                source_folder=folder,
                output_folder=Path(str(merged_descriptor["output"])),
                mode=mode,
                sources=sources,
                states=states,
                background=background,
                folder_translation=folder_translation,
                title_translations=title_translations,
                embed_subtitles=embed_subtitles,
            )
            merged_descriptor["outputs"] = merged_outputs
            merged_descriptor["status"] = "completed"

    for descriptor in descriptors:
        state: dict[str, Any] | None = None
        if descriptor["kind"] == "track":
            state = states_by_index.get(int(descriptor["index"]))
        elif layout == LAYOUT_MERGED:
            state = states[0]
        if state is not None:
            descriptor["status"] = str(state.get("status") or "")
            descriptor["outputs"] = dict(state.get("outputs") or {})
            descriptor["project_json"] = str(state.get("project_json") or "")

    ordered_states = states if layout != LAYOUT_MERGED else [states[0]]
    if layout == LAYOUT_MERGED:
        # The merged job's timeline already contains the complete order.
        summary_states = [{"timeline": [{"duration_samples": 0}]} for _source in sources]
        # Use the actual merged timeline to derive per-track durations where
        # possible; this avoids re-opening every source solely for a summary.
        merged_timeline = list(states[0].get("timeline") or [])
        for index, _source in enumerate(sources):
            if index < len(merged_timeline):
                summary_states[index] = {
                    "timeline": [
                        {"duration_samples": merged_timeline[index].get("duration_samples", 0)}
                    ]
                }
        summary_states = summary_states
    else:
        summary_states = ordered_states
    write_smart_summary(
        output_root,
        source_folder=folder,
        mode=mode,
        layout=layout,
        sources=sources,
        states=summary_states,
        descriptors=descriptors,
        folder_translation=folder_translation,
        title_translations=title_translations,
        footer=config.timestamp_footer,
        harmonized_delay_seconds=config.harmonized_delay_seconds,
    )
    manifest["jobs"] = descriptors
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_text(
        output_root / "处理清单.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["jobs"] = descriptors
    metadata["completed_at"] = manifest["completed_at"]
    save_plan_metadata(plan_id, metadata)
    for state in {str(item.get("job_id")): item for item in states}.values():
        cleanup_completed_workspace(state)
    log_event(f"智能任务完成 plan={plan_id} output={output_root}")
    print("\n智能任务完成。源作品文件没有被修改。")
    print(f"所有成品位于：{output_root}")


def execute_smart_plan(
    paths: ToolPaths,
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
) -> None:
    """Compatibility wrapper for a single recursive DLsite task."""

    plan = prepare_smart_plan(
        config,
        folder,
        mode_argument=mode_argument,
        layout_argument=layout_argument,
        edition_argument=edition_argument,
        include_bonus=include_bonus,
        output_root_argument=output_root_argument,
        background_argument=background_argument,
        embed_subtitles_argument=embed_subtitles_argument,
        rebuild=rebuild,
        force=force,
    )
    print_smart_plan_summary(plan)
    try:
        execute_prepared_smart_plan(paths, config, plan)
    except (KeyboardInterrupt, OperationCancelledError):
        raise
    except VideoPreparerError as exc:
        save_failed_task(plan, error=str(exc))
        log_event(f"单作品任务失败 source={plan.folder} error={exc}")
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        save_failed_task(plan, error=message)
        log_event(f"单作品任务未预期失败 source={plan.folder} error={message}")
        append_log(traceback.format_exc())
        raise
    else:
        remove_failed_task(plan.plan_id)
        remove_failed_task(plan.retry_of)


def missing_resume_artifacts(state: dict[str, Any], folder: Path) -> list[Path]:
    missing: list[Path] = []

    def require_file(candidate: Path) -> None:
        if not candidate.is_file() and candidate not in missing:
            missing.append(candidate)

    if status_at_least(state, "media_ready"):
        default_original = (
            folder / "原声.flac"
            if normalize_mode(state["mode"]) == MODE_AUDIO
            else folder / "原声.mp4"
        )
        original = Path(
            str(state.get("original_media") or state.get("original_video") or default_original)
        )
        require_file(original)
    if status_at_least(state, "media_ready") and not status_at_least(state, "project_created"):
        dubbing_input = Path(str(state.get("dubbing_input") or ""))
        require_file(dubbing_input)
    if status_at_least(state, "project_created") and not status_at_least(state, "outputs_ready"):
        project_json = Path(str(state.get("project_json") or ""))
        require_file(project_json)
    if legacy_chinese_timeline_needs_reimport(state):
        project_json = Path(str(state.get("project_json") or ""))
        require_file(project_json)
    if status_at_least(state, "outputs_ready"):
        outputs = state.get("outputs") or {}
        primary_key = "audio" if normalize_mode(state["mode"]) == MODE_AUDIO else "video"
        default_primary = folder / ("双语版.wav" if primary_key == "audio" else "双语版.mp4")
        require_file(Path(str(outputs.get(primary_key) or default_primary)))
        for key, name in (("srt", "双语版.srt"), ("lrc", "双语版.lrc")):
            require_file(Path(str(outputs.get(key) or folder / name)))
    if status_at_least(state, "completed"):
        timestamp_file = folder / "时间戳.txt"
        require_file(timestamp_file)
    return missing


def prepare_media_phase(
    paths: ToolPaths,
    folder: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
) -> None:
    workspace = Path(state["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    background = Path(state["background"]) if state.get("background") else None
    master, timeline = normalize_and_concat(paths, sources, workspace)
    state["master_audio"] = str(master)
    state["timeline"] = timeline

    mode = normalize_mode(state["mode"])
    if mode == MODE_AUDIO:
        print("\n[2/5] 输出拼接后的无损原声")
        original = folder / "原声.flac"
        atomic_copy(master, original)
        dubbing_input = original
    elif mode == MODE_VIDEO_NORMAL:
        print("\n[2/5] 生成普通静态视频")
        original = folder / "原声.mp4"
        render_static_video(paths, master, background, original)
        dubbing_input = original
    else:
        print("\n[2/5] 生成和谐静态视频")
        original = folder / "原声.mp4"
        dubbing_input = workspace / "dubbing_input.mp4"
        print("  生成供 ASMR Dubber 使用的无前导正常音量版本……")
        render_static_video(paths, master, background, dubbing_input)
        harmonized_volume_db = float(state["harmonized_volume_db"])
        harmonized_delay_seconds = int(state["harmonized_delay_seconds"])
        print(
            f"  生成 {harmonized_volume_db:g} dB 且前置 "
            f"{harmonized_delay_seconds / 60:g} 分钟的原声版本……"
        )
        render_static_video(
            paths,
            master,
            background,
            original,
            lead_seconds=harmonized_delay_seconds,
            volume_db=harmonized_volume_db,
        )
    state["original_media"] = str(original)
    state["original_video"] = str(original) if mode != MODE_AUDIO else None
    state["dubbing_input"] = str(dubbing_input)
    state["status"] = "media_ready"


def execute_task(
    paths: ToolPaths,
    folder: Path,
    state_file: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
    *,
    shared_reference: dict[str, str] | None = None,
    capture_reference_path: Path | None = None,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> None:
    if not status_at_least(state, "media_ready"):
        prepare_media_phase(paths, folder, state, sources)
        save_state(state_file, state)

    if not status_at_least(state, "project_created"):
        print("\n[3/5] 创建 ASMR Dubber 项目")
        project_json = create_asmr_project(
            paths,
            Path(state["dubbing_input"]),
            source_language=source_language_for_sources(sources),
        )
        state["project_json"] = str(project_json)
        ensure_autoflow_project_outputs(project_json)
        state["status"] = "project_created"
        save_state(state_file, state)

    project_json = Path(state["project_json"])
    if not project_json.is_file() and not status_at_least(state, "outputs_ready"):
        raise VideoPreparerError(f"ASMR Dubber 项目已经不存在：{project_json}")

    if legacy_chinese_timeline_needs_reimport(state):
        reset_legacy_chinese_timeline_state(state)
        save_state(state_file, state)

    if not status_at_least(state, "analyzed"):
        transcript_result = import_available_source_transcript(
            paths,
            project_json,
            list(state.get("timeline") or []),
        )
        if transcript_result:
            state["transcript_import"] = transcript_result
            kind = str(transcript_result.get("kind") or "")
            if kind == "direct":
                language = str(transcript_result.get("language") or "ja")
                print(
                    f"\n  已自动导入{language.upper()}台本/字幕，不再运行不必要的 ASR（语音识别）。"
                )
                state["status"] = "awaiting_reference" if language == "zh" else "analyzed"
                save_state(state_file, state)
            elif kind in {"partial", "zh_overlay_partial"}:
                print(
                    f"\n  找到 {transcript_result.get('count', 0)} 份台本/字幕，"
                    "但不足以覆盖全部合并音轨；本次仍运行 ASR。"
                )

        if not status_at_least(state, "analyzed"):
            print("\n  运行 ASR（语音识别）……")
            run_asmr_cli(paths, "analyze", str(project_json))
            state["status"] = "analyzed"
            save_state(state_file, state)

    transcript_result = state.get("transcript_import") or {}
    if str(transcript_result.get("kind") or "") == "zh_overlay_partial" and not state.get(
        "chinese_transcript_overlay_done"
    ):
        applied = overlay_timed_chinese_transcripts(
            project_json,
            list(state.get("timeline") or []),
        )
        state["chinese_transcript_overlay_done"] = True
        state["chinese_transcript_overlay_sentences"] = applied
        print(f"  已把已有中文字幕匹配到 {applied} 条日文识别结果。")
        save_state(state_file, state)

    if not status_at_least(state, "awaiting_reference"):
        if project_source_language(project_json) == "zh":
            print("\n  当前是中文配音稿，跳过 ASR 与翻译。")
        else:
            print("\n  翻译日文……")
            run_asmr_cli(paths, "translate", str(project_json))
        state["status"] = "awaiting_reference"
        save_state(state_file, state)

    if not status_at_least(state, "synthesized"):
        if shared_reference is not None:
            apply_shared_reference(project_json, shared_reference)
            print("\n  已复用本作品的统一音色参考，不需要再次选择。")
        else:
            reference_id = project_reference_id(project_json)
            if reference_id:
                print(f"\n复用已保存的统一音色参考：{reference_id}")
            elif project_has_external_reference(project_json):
                print("\n复用当前项目已经保存的外部参考音频。")
            elif not wait_for_reference(
                paths,
                project_json,
                timeout_seconds=int(
                    state.get("reference_wait_seconds", REFERENCE_SELECTION_TIMEOUT_SECONDS)
                ),
                event_callback=reference_event_callback,
                event_context={
                    "plan_id": str(state.get("plan_id") or ""),
                    "job_id": str(state.get("job_id") or ""),
                    "work": Path(str(state.get("source_folder") or folder)).name,
                    "output_folder": str(folder.resolve()),
                },
                launch_ui=reference_event_callback is None,
            ):
                return
        if capture_reference_path is not None and not state.get("shared_reference"):
            state["shared_reference"] = extract_shared_reference(
                project_json,
                capture_reference_path,
            )
            save_state(state_file, state)
        print("\n[5/5] TTS（语音合成）、混音与字幕")
        run_asmr_cli(paths, "synthesize", str(project_json))
        state["status"] = "synthesized"
        save_state(state_file, state)

    if not status_at_least(state, "mixed"):
        run_asmr_cli(paths, "mix", str(project_json))
        project = read_project(project_json)
        mixed_audio = project_asset(project_json, project.get("output_file"))
        if mixed_audio is not None:
            state["project_mixed_audio"] = str(mixed_audio)
        state["status"] = "mixed"
        save_state(state_file, state)

    if not status_at_least(state, "subtitles_ready"):
        run_asmr_cli(paths, "subtitles", str(project_json), "--language", "bilingual")
        state["status"] = "subtitles_ready"
        save_state(state_file, state)

    if not status_at_least(state, "outputs_ready"):
        state["outputs"] = copy_final_outputs(
            paths,
            project_json,
            folder,
            str(state["mode"]),
            harmonized_delay_seconds=int(state["harmonized_delay_seconds"]),
            harmonized_volume_db=float(state["harmonized_volume_db"]),
            embed_subtitles=bool(state.get("embed_subtitles", True)),
        )
        state["status"] = "outputs_ready"
        save_state(state_file, state)

    if not status_at_least(state, "completed"):
        translations = state.get("title_translations") or {}
        missing_titles = [
            str(item.get("relative_path") or item["filename"])
            for item in state.get("timeline") or []
            if not str(
                translations.get(str(item.get("relative_path") or item["filename"]), "")
            ).strip()
        ]
        if missing_titles or not str(state.get("folder_name_translation") or "").strip():
            print("\n  使用 ASMR Dubber 当前翻译服务处理作品名和曲目标题……")
            try:
                state["title_translations"] = translate_titles(state, paths)
            except VideoPreparerError:
                save_state(state_file, state)
                raise
        save_state(state_file, state)
        timestamp_file = write_timestamp_document(state, folder)
        state.setdefault("outputs", {})["timestamps"] = str(timestamp_file)
        state["timestamp_schema"] = TIMESTAMP_SCHEMA
        state["status"] = "completed"
        save_state(state_file, state)

    print("\n全部完成。文件已放回：")
    print(f"  {folder}")
    if normalize_mode(state["mode"]) == MODE_AUDIO:
        print("  - 原声.flac")
        print(f"  - {Path(state['outputs']['audio']).name}")
    else:
        print("  - 原声.mp4")
        print("  - 双语版.mp4")
    print("  - 双语版.srt")
    print("  - 双语版.lrc")
    print("  - 时间戳.txt")
    if project_json.is_file():
        print(f"\nASMR Dubber 工作项目保留在：\n  {project_json.parent}")


def prepare_or_resume(
    paths: ToolPaths,
    config: AppConfig,
    folder: Path,
    mode_argument: str | None,
    embed_subtitles_argument: str | None,
    *,
    rebuild: bool,
    force: bool,
) -> None:
    if not folder.is_dir():
        raise VideoPreparerError(f"文件夹不存在：{folder}")
    sources = discover_audio(folder)
    state_file = state_path(folder)
    state = None if rebuild else load_state(state_file)
    requested_mode = normalize_mode(mode_argument) if mode_argument else None

    if state is not None and not rebuild:
        stored_embed_subtitles = bool(state.get("embed_subtitles", True))
        state_mode = normalize_mode(state["mode"])
        if requested_mode is not None and requested_mode != state_mode:
            raise VideoPreparerError("现有任务模式与 --mode 不一致；如需更换，请使用 --rebuild。")
        if embed_subtitles_argument is not None:
            requested_embed = parse_embed_subtitles_argument(embed_subtitles_argument)
            if state_mode != MODE_AUDIO and requested_embed != stored_embed_subtitles:
                raise VideoPreparerError(
                    "现有任务的字幕内嵌设置与 --embed-subtitles 不一致；"
                    "如需更换，请使用 --rebuild。"
                )
        background = discover_background(folder) if state_mode != MODE_AUDIO else None
        current_fingerprint = fingerprint(sources, background)
    else:
        background = None
        current_fingerprint = {}

    if state is not None and not rebuild:
        if state.get("fingerprint") != current_fingerprint:
            print("\n检测到小音频或视频背景在上次任务后发生变化。")
            answer = input("输入 R 按当前文件从头重做；直接按 Enter 退出：").strip().casefold()
            if answer != "r":
                return
            rebuild = True
        if not rebuild and legacy_chinese_timeline_needs_reimport(state):
            print("\n已有中文字幕将直接重建时间轴，不会重跑 ASR 或翻译。")
            reset_legacy_chinese_timeline_state(state)
            save_state(state_file, state)
        missing = missing_resume_artifacts(state, folder) if not rebuild else []
        if not rebuild and missing:
            print("\n旧任务记录依赖的文件已经不存在：")
            for path in missing:
                print(f"  {path}")
            answer = input("输入 R 从头重做；直接按 Enter 退出：").strip().casefold()
            if answer != "r":
                return
            rebuild = True
        if not rebuild and status_at_least(state, "completed"):
            if int(state.get("timestamp_schema") or 0) < TIMESTAMP_SCHEMA:
                print("\n正在补充文件夹名称的中英文信息；不会重跑识别、配音或混音。")
                state["status"] = "outputs_ready"
                save_state(state_file, state)
            else:
                print("\n这个文件夹的任务已经完成。")
                answer = input("输入 R 从头重做；直接按 Enter 退出：").strip().casefold()
                if answer != "r":
                    return
                rebuild = True
        elif not rebuild:
            print(f"\n发现未完成任务，当前阶段：{state.get('status') or '尚未开始'}")
            answer = input("输入 1 继续；输入 2 从头开始：").strip()
            if answer == "2":
                rebuild = True
            elif answer != "1":
                return

    if state is None or rebuild:
        mode = requested_mode or ask_mode(config)
        embed_subtitles = ask_embed_subtitles(mode, embed_subtitles_argument)
        background = discover_background(folder) if mode != MODE_AUDIO else None
        current_fingerprint = fingerprint(sources, background)
        confirm_overwrite(folder, mode, force=force)
        workspace = workspace_path(folder)
        safe_reset_workspace(workspace)
        state = create_initial_state(
            folder,
            mode,
            sources,
            background,
            config,
            embed_subtitles=embed_subtitles,
        )
        save_state(state_file, state)

    print("\n将按以下顺序处理：")
    for item in sources:
        print(f"  {item.order:>4}  {item.path.name}")
    if normalize_mode(state["mode"]) != MODE_AUDIO:
        print(f"背景：{background.name if background else '未找到 null 图片，使用黑色背景'}")
    print(f"模式：{mode_label(state['mode'])}")
    if normalize_mode(state["mode"]) != MODE_AUDIO:
        print(
            "视频字幕："
            + (
                "内嵌，同时保留 SRT/LRC"
                if state.get("embed_subtitles", True)
                else "不内嵌，仅保留 SRT/LRC"
            )
        )
    execute_task(paths, folder, state_file, state, sources)


def media_duration_seconds(paths: ToolPaths, media: Path) -> float:
    command = [
        str(paths.ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise VideoPreparerError(f"无法取得媒体时长：{media}")
    return float(result.stdout.strip())


def video_keyframe_times(paths: ToolPaths, media: Path) -> list[float]:
    command = [
        str(paths.ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,flags",
        "-of",
        "json",
        str(media),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise VideoPreparerError(f"无法检查视频关键帧：{media}")
    try:
        packets = json.loads(result.stdout).get("packets") or []
        return [
            float(packet["pts_time"]) for packet in packets if "K" in str(packet.get("flags") or "")
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoPreparerError(f"关键帧信息无效：{media}") from exc


def media_stream_types(paths: ToolPaths, media: Path) -> list[str]:
    command = [
        str(paths.ffprobe),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(media),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise VideoPreparerError(f"无法检查媒体流：{media}")
    try:
        streams = json.loads(result.stdout).get("streams") or []
        return [str(item["codec_type"]) for item in streams]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoPreparerError(f"媒体流信息无效：{media}") from exc


def self_test(paths: ToolPaths) -> None:
    print("运行轻量自检；不会调用 ASR、TTS 或网络……")
    test_harmonized_volume_db = -DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB
    with tempfile.TemporaryDirectory(prefix="video-preparer-test-") as temporary:
        root = Path(temporary)
        sources = (
            ("1 开场.wav", "sine=frequency=440:duration=0.60", "44100", "1"),
            ("2 囁き.flac", "sine=frequency=660:duration=0.70", "48000", "2"),
            ("10 終了.m4a", "sine=frequency=880:duration=0.80", "32000", "1"),
        )
        for name, generator, rate, channels in sources:
            run_ffmpeg(
                paths,
                [
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    generator,
                    "-ar",
                    rate,
                    "-ac",
                    channels,
                    str(root / name),
                ],
            )
        run_ffmpeg(
            paths,
            [
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:s=641x479",
                "-frames:v",
                "1",
                str(root / "null.png"),
            ],
        )

        discovered = discover_audio(root)
        if [item.order for item in discovered] != [1, 2, 10]:
            raise VideoPreparerError("自检失败：数字排序错误。")
        background = discover_background(root)
        if background is None:
            raise VideoPreparerError("自检失败：没有识别 null 图片。")
        smart_scan = scan_work(root, excluded_directories=("AutoFlow输出",))
        if smart_scan.audio_count != 3 or not smart_scan.editions:
            raise VideoPreparerError("自检失败：智能扫描没有识别传统数字音轨。")
        if (
            not identifier_only_name("RJ01563553")
            or not identifier_only_name("VJ_00123")
            or identifier_only_name("RJ01563553 作品标题")
        ):
            raise VideoPreparerError("自检失败：作品编号文件夹识别错误。")
        if (
            _background_from_argument(smart_scan, "auto") != smart_scan.images[0]
            or _background_from_argument(smart_scan, "1") != smart_scan.images[0]
            or _background_from_argument(smart_scan, "0") is not None
        ):
            raise VideoPreparerError("自检失败：推荐背景、编号选择或黑色背景错误。")
        if (
            parse_yes_no("Y") is not True
            or parse_yes_no("n") is not False
            or parse_yes_no("") is not None
            or parse_embed_subtitles_argument("yes") is not True
            or parse_embed_subtitles_argument("no") is not False
        ):
            raise VideoPreparerError("自检失败：Y/N 或字幕选项解析错误。")
        smart_sources = [
            source_from_candidate(index, item)
            for index, item in enumerate(smart_scan.editions[0].tracks, start=1)
        ]
        if [item.order for item in smart_sources] != [1, 2, 3]:
            raise VideoPreparerError("自检失败：智能扫描输出顺序错误。")
        if (
            len(smart_job_descriptors(root / "out", LAYOUT_MERGED, smart_sources)) != 1
            or len(smart_job_descriptors(root / "out", LAYOUT_SEPARATE, smart_sources)) != 3
            or len(smart_job_descriptors(root / "out", LAYOUT_BOTH, smart_sources)) != 4
        ):
            raise VideoPreparerError("自检失败：三种输出布局的任务规划错误。")
        queue_plan_base = {
            "output_root": root / "queue-output",
            "edition_label": "测试版本",
            "sources": tuple(smart_sources),
            "edition": {},
            "mode": MODE_AUDIO,
            "layout": LAYOUT_MERGED,
            "background": None,
            "embed_subtitles": False,
            "rebuild": False,
            "force": False,
        }
        queue_plans = [
            SmartTaskPlan(
                folder=root / "queue-first",
                plan_id="queue-first",
                **queue_plan_base,
            ),
            SmartTaskPlan(
                folder=root / "queue-second",
                plan_id="queue-second",
                **{**queue_plan_base, "output_root": root / "queue-output-2"},
            ),
        ]
        queue_calls: list[str] = []

        def fake_queue_executor(
            _paths: ToolPaths,
            _config: AppConfig,
            task: SmartTaskPlan,
        ) -> None:
            queue_calls.append(task.plan_id)
            if task.plan_id == "queue-first":
                raise VideoPreparerError("预期的队列测试失败")

        existing_failed_records = read_failed_task_records()
        try:
            with (
                tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sink,
                redirect_stdout(sink),
                redirect_stderr(sink),
            ):
                queue_result = execute_smart_queue(
                    paths,
                    AppConfig(
                        asmr_root=None,
                        harmonized_volume_db=-10.0,
                        harmonized_delay_seconds=1_200,
                        timestamp_footer="",
                    ),
                    queue_plans,
                    executor=fake_queue_executor,
                )
        finally:
            # The queue test deliberately fails one item. Do not leave its
            # temporary paths in the user's real retry list after self-test.
            write_failed_task_records(existing_failed_records)
        if queue_result != 1 or queue_calls != ["queue-first", "queue-second"]:
            raise VideoPreparerError("自检失败：队列没有在单项失败后继续处理。")
        workspace = root / "work"
        workspace.mkdir()
        master, timeline = normalize_and_concat(paths, discovered, workspace)
        total_samples = sum(int(item["duration_samples"]) for item in timeline)
        if audio_duration_samples(paths, master) != total_samples:
            raise VideoPreparerError("自检失败：母带采样数错误。")
        rebuilt_master, rebuilt_lengths = concat_audio_files(
            paths,
            [Path(str(item["normalized"])) for item in timeline],
            root / "exact-concat",
            expected_samples=[int(item["duration_samples"]) for item in timeline],
            name="self-test",
        )
        if rebuilt_lengths != [int(item["duration_samples"]) for item in timeline] or (
            audio_duration_samples(paths, rebuilt_master) != total_samples
        ):
            raise VideoPreparerError("自检失败：分轨成品的精确二次合并错误。")

        test_settings = root / "settings.txt"
        test_settings.write_text(
            "asmr_dubber_path=.\n"
            "harmonized_volume_reduction_db=12.5\n"
            "harmonized_delay_minutes=7.5\n"
            "timestamp_footer_line_1=测试页脚\n",
            encoding="utf-8",
        )
        parsed_config = load_app_config(test_settings)
        if (
            parsed_config.asmr_root != root.resolve()
            or parsed_config.harmonized_volume_db != -12.5
            or parsed_config.harmonized_delay_seconds != 450
            or parsed_config.timestamp_footer != "测试页脚"
        ):
            raise VideoPreparerError("自检失败：settings.txt 解析错误。")
        remembered_defaults = SmartPlanDefaults(
            mode=MODE_AUDIO,
            video_mode=MODE_VIDEO_HARMONIZED,
            layout=LAYOUT_BOTH,
            include_bonus=True,
            background_choice="relative",
            background_relative="cover.jpg",
            embed_subtitles=False,
            edition_extension=".wav",
            edition_language="ja",
            edition_mix_variant="standard",
            edition_orientation="standard",
        )
        remembered_path = root / "last-options.json"
        save_smart_plan_defaults(remembered_defaults, remembered_path)
        if load_smart_plan_defaults(parsed_config, remembered_path) != remembered_defaults:
            raise VideoPreparerError("自检失败：最后一次选项没有正确保存或恢复。")

        catalogue_root = root / "catalogue"
        wav_dir = catalogue_root / "WAV"
        mp3_dir = catalogue_root / "MP3"
        bonus_dir = catalogue_root / "特典"
        wav_dir.mkdir(parents=True)
        mp3_dir.mkdir(parents=True)
        bonus_dir.mkdir(parents=True)
        shutil.copy2(root / "1 开场.wav", wav_dir / "Track０１ 開場.wav")
        shutil.copy2(root / "1 开场.wav", mp3_dir / "Track01 開場.mp3")
        shutil.copy2(root / "2 囁き.flac", bonus_dir / "EX01 おまけ.wav")
        (wav_dir / "Track０１ 開場.wav.vtt").write_text(
            "WEBVTT\n\n00:00.000 --> 00:00.500\n開場\n",
            encoding="utf-8",
        )
        plain_lrc = catalogue_root / "字幕" / "01.lrc"
        plain_lrc.parent.mkdir(parents=True)
        plain_lrc.write_text("[00:00.00]中文台词\n", encoding="utf-8")
        japanese_lrc = catalogue_root / "日文" / "line.lrc"
        japanese_lrc.parent.mkdir(parents=True)
        japanese_lrc.write_text("[00:00.00]日本語\n", encoding="utf-8")
        catalogue = scan_work(catalogue_root)
        transcript_languages = {
            item.path.resolve(): item.language for item in catalogue.transcripts
        }
        if (
            transcript_languages.get(plain_lrc.resolve()) != "zh"
            or transcript_languages.get(japanese_lrc.resolve()) != "ja"
        ):
            raise VideoPreparerError("自检失败：普通 LRC 或明确日文目录的语言判断错误。")
        wav_edition = next(
            (
                item
                for item in catalogue.editions
                if item.extension == ".wav" and item.directory == "WAV"
            ),
            None,
        )
        if (
            wav_edition is None
            or wav_edition.tracks[0].order_key[1] != 1
            or wav_edition.tracks[0].transcript is None
        ):
            raise VideoPreparerError("自检失败：全角 Track 编号或 VTT 自动匹配错误。")
        _, selected_with_bonus, _ = choose_tracks(
            catalogue,
            parsed_config,
            edition_argument=wav_edition.id,
            include_bonus=True,
        )
        if len(selected_with_bonus) != 2 or selected_with_bonus[-1].category != "bonus":
            raise VideoPreparerError("自检失败：独立特典目录没有加入所选版本。")
        inherited_edition = default_edition_for_work(
            catalogue,
            parsed_config,
            remembered_defaults,
        )
        if inherited_edition.extension != ".wav":
            raise VideoPreparerError("自检失败：下一个作品没有沿用上次的音频格式。")
        queue_args = argparse.Namespace(
            mode=None,
            layout=None,
            edition=None,
            include_bonus=False,
            output_root=None,
            background=None,
            embed_subtitles=None,
            rebuild=False,
            force=False,
        )
        inherited_draft = prepare_smart_plan_draft(
            parsed_config,
            catalogue_root,
            remembered_defaults,
            queue_args,
        )
        if (
            inherited_draft.mode != MODE_AUDIO
            or inherited_draft.video_mode != MODE_VIDEO_HARMONIZED
            or inherited_draft.layout != LAYOUT_BOTH
            or not inherited_draft.include_bonus
            or inherited_draft.embed_subtitles
        ):
            raise VideoPreparerError("自检失败：作品选项没有从上一个项目继承。")
        with (
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sink,
            redirect_stdout(sink),
            redirect_stderr(sink),
            patch(
                "builtins.input",
                side_effect=("3", "2", "4", "1", "5", "1", "0"),
            ),
        ):
            inherited_plan, next_defaults, start_now = configure_smart_plan_draft(
                inherited_draft,
                parsed_config,
                index=1,
                rebuild=False,
                force=False,
            )
        if (
            not start_now
            or inherited_plan.mode != MODE_VIDEO_NORMAL
            or inherited_plan.layout != LAYOUT_MERGED
            or next_defaults.mode != MODE_VIDEO_NORMAL
            or next_defaults.video_mode != MODE_VIDEO_NORMAL
        ):
            raise VideoPreparerError("自检失败：默认选项面板没有正确修改或开始任务。")

        audio_output_folder = root / "audio-mode"
        audio_output_folder.mkdir()
        audio_state = create_initial_state(
            audio_output_folder,
            MODE_AUDIO,
            discovered,
            None,
            AppConfig(
                asmr_root=None,
                harmonized_volume_db=-10.0,
                harmonized_delay_seconds=1_200,
                timestamp_footer="",
            ),
        )
        audio_state["workspace"] = str(root / "audio-mode-work")
        prepare_media_phase(paths, audio_output_folder, audio_state, discovered)
        audio_original = audio_output_folder / "原声.flac"
        if (
            audio_state["status"] != "media_ready"
            or not audio_original.is_file()
            or audio_duration_samples(paths, audio_original) != total_samples
        ):
            raise VideoPreparerError("自检失败：纯音频模式输出错误。")

        fake_project = root / "audio-project"
        (fake_project / "output").mkdir(parents=True)
        (fake_project / "subtitles").mkdir()
        mixed_wav = fake_project / "output" / "mixed.wav"
        run_ffmpeg(
            paths,
            [
                "-loglevel",
                "error",
                "-i",
                str(audio_original),
                "-c:a",
                "pcm_s16le",
                str(mixed_wav),
            ],
        )
        (fake_project / "subtitles" / "bilingual.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n测试\n",
            encoding="utf-8",
        )
        (fake_project / "subtitles" / "bilingual.lrc").write_text(
            "[00:00.00]测试\n",
            encoding="utf-8",
        )
        fake_project_json = fake_project / "project.json"
        fake_project_json.write_text(
            json.dumps(
                {
                    "output_file": "output/mixed.wav",
                    "subtitle_srt_file": "subtitles/bilingual.srt",
                    "subtitle_lrc_file": "subtitles/bilingual.lrc",
                }
            ),
            encoding="utf-8",
        )
        audio_final_folder = root / "audio-final"
        audio_outputs = copy_final_outputs(
            paths,
            fake_project_json,
            audio_final_folder,
            MODE_AUDIO,
            harmonized_delay_seconds=1_200,
            harmonized_volume_db=-10.0,
            embed_subtitles=False,
        )
        if not all(Path(path).is_file() for path in audio_outputs.values()):
            raise VideoPreparerError("自检失败：纯音频模式最终文件回写错误。")

        normal = root / "normal.mp4"
        black = root / "black.mp4"
        harmony = root / "harmony.mp4"
        render_static_video(paths, master, background, normal)
        render_static_video(paths, master, None, black)
        render_static_video(
            paths,
            master,
            background,
            harmony,
            lead_seconds=KEYFRAME_INTERVAL_SECONDS,
            volume_db=test_harmonized_volume_db,
        )
        normal_duration = media_duration_seconds(paths, normal)
        black_duration = media_duration_seconds(paths, black)
        harmony_duration = media_duration_seconds(paths, harmony)
        if not (1.9 <= normal_duration <= 2.3):
            raise VideoPreparerError(f"自检失败：普通视频时长异常 {normal_duration:.3f}s")
        if not (1.9 <= black_duration <= 2.3):
            raise VideoPreparerError(f"自检失败：黑色背景视频时长异常 {black_duration:.3f}s")
        if not (
            normal_duration + KEYFRAME_INTERVAL_SECONDS - 0.2
            <= harmony_duration
            <= normal_duration + KEYFRAME_INTERVAL_SECONDS + 0.2
        ):
            raise VideoPreparerError(
                f"自检失败：前置静音时长异常 {harmony_duration - normal_duration:.3f}s"
            )
        keyframes = video_keyframe_times(paths, harmony)
        if len(keyframes) < 2 or not any(
            abs(timestamp - KEYFRAME_INTERVAL_SECONDS) <= 0.1 for timestamp in keyframes
        ):
            raise VideoPreparerError(f"自检失败：没有按 10 秒间隔写入关键帧：{keyframes}")

        sample_srt = "1\n00:00:01,250 --> 00:00:02,500\n测试\n"
        shifted = shift_srt_text(sample_srt, 1_200_000)
        if "00:20:01,250 --> 00:20:02,500" not in shifted:
            raise VideoPreparerError("自检失败：SRT 偏移错误。")
        shifted_file = root / "shifted.srt"
        shifted_file.write_text(shifted, encoding="utf-8")
        second_srt = root / "second.srt"
        second_srt.write_text(
            "99\n00:00:00,100 --> 00:00:00,300\n第二段\n",
            encoding="utf-8",
        )
        combined_srt = combine_srt_files(
            ((shifted_file, -1_200_000, 2_000), (second_srt, 2_000, 3_000)),
            root / "combined.srt",
        ).read_text(encoding="utf-8-sig")
        if (
            "1\n00:00:01,250 --> 00:00:02,000" not in combined_srt
            or "2\n00:00:02,100" not in combined_srt
        ):
            raise VideoPreparerError("自检失败：多文件 SRT 合并或重新编号错误。")
        delayed = root / "delayed.mp4"
        render_delayed_existing_video(
            paths,
            normal,
            delayed,
            lead_seconds=2,
            subtitle_file=shifted_file,
            volume_db=test_harmonized_volume_db,
            audio_source=master,
        )
        delayed_duration = media_duration_seconds(paths, delayed)
        if not (normal_duration + 1.8 <= delayed_duration <= normal_duration + 2.2):
            raise VideoPreparerError(
                f"自检失败：双语视频前导时长异常 {delayed_duration - normal_duration:.3f}s"
            )
        if "subtitle" not in media_stream_types(paths, delayed):
            raise VideoPreparerError("自检失败：开启字幕后视频中没有字幕流。")
        delayed_without_subtitle = root / "delayed-without-subtitle.mp4"
        render_delayed_existing_video(
            paths,
            normal,
            delayed_without_subtitle,
            lead_seconds=2,
            subtitle_file=None,
            volume_db=test_harmonized_volume_db,
            audio_source=master,
        )
        if "subtitle" in media_stream_types(paths, delayed_without_subtitle):
            raise VideoPreparerError("自检失败：关闭字幕后视频仍包含字幕流。")
        remuxed_without_subtitle = root / "remuxed-without-subtitle.mp4"
        remux_video_without_subtitles(paths, delayed, remuxed_without_subtitle)
        remuxed_streams = media_stream_types(paths, remuxed_without_subtitle)
        if "subtitle" in remuxed_streams or not {"video", "audio"}.issubset(remuxed_streams):
            raise VideoPreparerError("自检失败：无字幕重封装的媒体流错误。")
        sample_lrc = "[00:01.25]测试\n"
        if "[20:01.25]" not in shift_lrc_text(sample_lrc, 1_200_000):
            raise VideoPreparerError("自检失败：LRC 偏移错误。")
        first_lrc = root / "first.lrc"
        second_lrc = root / "second.lrc"
        first_lrc.write_text("[ar:测试]\n[00:01.25]第一段\n", encoding="utf-8")
        second_lrc.write_text("[ar:忽略]\n[00:00.10]第二段\n", encoding="utf-8")
        combined_lrc = combine_lrc_files(
            ((first_lrc, 0), (second_lrc, 2_000)),
            root / "combined.lrc",
        ).read_text(encoding="utf-8-sig")
        if "[ar:测试]" not in combined_lrc or "[00:02.10]第二段" not in combined_lrc:
            raise VideoPreparerError("自检失败：多文件 LRC 合并错误。")
        timestamp_state = {
            "source_folder": str(root / "日本語作品"),
            "mode": MODE_VIDEO_NORMAL,
            "timeline": timeline,
            "title_translations": {
                str(item["filename"]): f"中文曲目{index}"
                for index, item in enumerate(timeline, start=1)
            },
            "folder_name_original": "日本語作品",
            "folder_name_translation": "中文作品",
        }
        timestamp_text = write_timestamp_document(timestamp_state, root).read_text(
            encoding="utf-8-sig"
        )
        if (
            "中文名称：中文作品" not in timestamp_text
            or "原始名称：日本語作品" not in timestamp_text
        ):
            raise VideoPreparerError("自检失败：时间戳文档缺少文件夹中英文名称。")
    print("自检通过。")


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
        help=(
            "视频背景：图片编号、作品目录内的相对/绝对路径、auto，"
            "或 black；交互运行默认显示全部图片供选择"
        ),
    )
    parser.add_argument(
        "--embed-subtitles",
        choices=("yes", "no"),
        help="视频是否内嵌字幕；无论如何都会保留外部 SRT/LRC",
    )
    parser.add_argument("--self-test", action="store_true", help="只运行几秒钟的本地媒体自检")
    return parser


def serialize_audio_source(source: AudioSource) -> dict[str, Any]:
    return {
        "order": source.order,
        "path": str(source.path),
        "title_ja": source.title_ja,
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "relative_path": source.relative_path,
        "category": source.category,
        "transcript_path": str(source.transcript_path) if source.transcript_path else None,
        "transcript_language": source.transcript_language,
        "transcript_timed": source.transcript_timed,
        "source_language": source.source_language,
    }


def deserialize_audio_source(payload: Any) -> AudioSource:
    if not isinstance(payload, dict):
        raise VideoPreparerError("失败任务记录中的音轨信息无效。")
    transcript_text = str(payload.get("transcript_path") or "").strip()
    return AudioSource(
        order=int(payload.get("order") or 0),
        path=Path(str(payload.get("path") or "")).expanduser().resolve(),
        title_ja=str(payload.get("title_ja") or ""),
        size=int(payload.get("size") or 0),
        mtime_ns=int(payload.get("mtime_ns") or 0),
        relative_path=str(payload.get("relative_path") or ""),
        category=str(payload.get("category") or "main"),
        transcript_path=Path(transcript_text).expanduser().resolve() if transcript_text else None,
        transcript_language=str(payload.get("transcript_language") or "").strip() or None,
        transcript_timed=bool(payload.get("transcript_timed", False)),
        source_language=str(payload.get("source_language") or "ja"),
    )


def failed_task_payload(
    plan: SmartTaskPlan,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "plan_id": plan.plan_id,
        "retry_of": plan.retry_of,
        "folder": str(plan.folder),
        "output_root": str(plan.output_root),
        "edition_label": plan.edition_label,
        "edition": plan.edition,
        "mode": plan.mode,
        "layout": plan.layout,
        "background": str(plan.background) if plan.background else None,
        "embed_subtitles": plan.embed_subtitles,
        "translate_work_title": plan.translate_work_title,
        "translate_track_titles": plan.translate_track_titles,
        "sources": [serialize_audio_source(item) for item in plan.sources],
    }
    if error:
        payload["error"] = error
    payload["failed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return payload


def read_failed_task_records() -> list[dict[str, Any]]:
    if not FAILED_TASKS_FILE.is_file():
        return []
    try:
        payload = json.loads(FAILED_TASKS_FILE.read_text(encoding="utf-8"))
        if payload.get("schema") != FAILED_TASKS_SCHEMA:
            return []
        records = payload.get("tasks") or []
        return [item for item in records if isinstance(item, dict) and item.get("plan_id")]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def write_failed_task_records(records: Sequence[dict[str, Any]]) -> None:
    if not records:
        FAILED_TASKS_FILE.unlink(missing_ok=True)
        return
    atomic_write_text(
        FAILED_TASKS_FILE,
        json.dumps(
            {
                "schema": FAILED_TASKS_SCHEMA,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tasks": list(records),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_failed_task(plan: SmartTaskPlan, *, error: str | None = None) -> None:
    records = [
        item for item in read_failed_task_records() if str(item.get("plan_id")) != plan.plan_id
    ]
    records.append(failed_task_payload(plan, error=error))
    write_failed_task_records(records)


def remove_failed_task(plan_id: str | None) -> None:
    if not plan_id:
        return
    records = [
        item for item in read_failed_task_records() if str(item.get("plan_id")) != str(plan_id)
    ]
    write_failed_task_records(records)


def load_failed_task_plans() -> list[SmartTaskPlan]:
    plans: list[SmartTaskPlan] = []
    for record in read_failed_task_records():
        try:
            sources = [deserialize_audio_source(item) for item in record.get("sources") or []]
            if not sources:
                continue
            plans.append(
                SmartTaskPlan(
                    folder=Path(str(record.get("folder") or "")).expanduser().resolve(),
                    output_root=Path(str(record.get("output_root") or "")).expanduser().resolve(),
                    edition_label=str(record.get("edition_label") or "失败任务"),
                    sources=tuple(sources),
                    edition=dict(record.get("edition") or {}),
                    mode=normalize_mode(record.get("mode")),
                    layout=normalize_layout(record.get("layout")),
                    background=(
                        Path(str(record.get("background"))).expanduser().resolve()
                        if record.get("background")
                        else None
                    ),
                    embed_subtitles=bool(record.get("embed_subtitles", True)),
                    plan_id=str(record.get("plan_id")),
                    rebuild=False,
                    force=False,
                    retry_of=str(record.get("retry_of") or "").strip() or None,
                    translate_work_title=bool(record.get("translate_work_title", True)),
                    translate_track_titles=bool(record.get("translate_track_titles", True)),
                )
            )
        except (TypeError, ValueError, OSError, VideoPreparerError):
            continue
    return plans


def defaults_for_failed_plan(plan: SmartTaskPlan) -> SmartPlanDefaults:
    video_mode = plan.mode if plan.mode != MODE_AUDIO else MODE_VIDEO_NORMAL
    edition = plan.edition
    background_relative: str | None = None
    if plan.background is not None:
        try:
            background_relative = (
                plan.background.resolve().relative_to(plan.folder.resolve()).as_posix()
            )
        except ValueError:
            # A valid plan normally keeps its background inside the work folder,
            # but retaining the basename is safer than losing the remembered
            # choice if an older record used an external path.
            background_relative = plan.background.name
    return SmartPlanDefaults(
        mode=plan.mode,
        video_mode=video_mode,
        layout=plan.layout,
        include_bonus=bool(edition.get("included_optional")),
        background_choice="black" if plan.background is None else "relative",
        background_relative=background_relative,
        embed_subtitles=plan.embed_subtitles,
        edition_extension=str(edition.get("extension") or "").strip() or None,
        edition_language=str(edition.get("language") or "").strip() or None,
        edition_mix_variant=str(edition.get("mix_variant") or "").strip() or None,
        edition_orientation=str(edition.get("orientation") or "").strip() or None,
    )


def relative_background_for_plan(plan: SmartTaskPlan) -> str | None:
    """Return a failed plan's background relative to its work folder."""

    if plan.background is None:
        return None
    try:
        return plan.background.resolve().relative_to(plan.folder.resolve()).as_posix()
    except ValueError:
        return plan.background.name


def work_from_failed_plan(plan: SmartTaskPlan) -> QueuedSmartWork:
    if plan.folder.is_dir():
        try:
            scan = scan_work(
                plan.folder,
                excluded_directories=("AutoFlow输出", plan.output_root.name),
            )
        except Exception:
            scan = ScanResult(plan.folder, (), (), (), (), len(plan.sources))
    else:
        scan = ScanResult(plan.folder, (), (), (), (), len(plan.sources))
    choices = {
        transcript_key(source.transcript_path): str(source.transcript_language or "zh")
        for source in plan.sources
        if source.transcript_path is not None
    }
    draft = SmartPlanDraft(
        folder=plan.folder,
        output_root=plan.output_root,
        scan=scan,
        edition_label=plan.edition_label,
        sources=list(plan.sources),
        edition=dict(plan.edition),
        include_bonus=bool(plan.edition.get("included_optional")),
        mode=plan.mode,
        video_mode=plan.mode if plan.mode != MODE_AUDIO else MODE_VIDEO_NORMAL,
        layout=plan.layout,
        background=plan.background,
        background_choice="black" if plan.background is None else "relative",
        background_relative=relative_background_for_plan(plan),
        embed_subtitles=plan.embed_subtitles,
        transcript_choices=choices,
    )
    return QueuedSmartWork(
        draft=draft,
        plan=plan,
        defaults=defaults_for_failed_plan(plan),
    )


def print_smart_work_queue(
    queue: Sequence[QueuedSmartWork],
    config: AppConfig,
) -> None:
    print("\n" + "=" * 68)
    print(f"作品串（已加入 {len(queue)} 个作品）")
    print("=" * 68)
    for index, work in enumerate(queue, start=1):
        plan = work.plan
        transcript_count = len(work.draft.scan.transcripts)
        transcript_label = (
            f"台本/字幕 {transcript_count} 份，默认中文" if transcript_count else "没有台本/字幕"
        )
        if plan.background is None:
            background = "黑色背景"
        else:
            try:
                background = plan.background.resolve().relative_to(plan.folder.resolve()).as_posix()
            except ValueError:
                background = plan.background.name
        subtitle = "内嵌字幕" if plan.embed_subtitles else "外部字幕"
        detail = (
            f"{layout_label(plan.layout)} · 背景：{background} · {subtitle} · {transcript_label}"
        )
        print(
            f"  {index}. {plan.folder.name}\n"
            f"     {plan.edition_label} · {mode_label(plan.mode)} · "
            f"{video_branch_label(plan.mode, config) if plan.mode != MODE_AUDIO else '音频'}\n"
            f"     {detail}"
        )
    print("\n直接粘贴下一个作品路径：添加")
    print("输入作品编号：返回修改；输入 D+编号：删除；直接按 Enter：开始处理")


def _queue_index(value: str, maximum: int) -> int | None:
    raw = value.strip()
    if raw.casefold().startswith("d") or raw.casefold().startswith("e"):
        raw = raw[1:].strip()
    if not raw.isdigit():
        return None
    index = int(raw)
    return index if 1 <= index <= maximum else None


def _prefixed_queue_index(value: str, prefix: str, maximum: int) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}\s*(\d+)", value.strip(), re.IGNORECASE)
    if match is None:
        return None
    index = int(match.group(1))
    return index if 1 <= index <= maximum else None


def collect_interactive_smart_plans(
    config: AppConfig,
    args: argparse.Namespace,
) -> list[SmartTaskPlan]:
    """Configure a visible, editable queue before the first task starts."""

    queue: list[QueuedSmartWork] = []
    configured_folders: set[Path] = set()
    defaults = load_smart_plan_defaults(config)
    failed_available = load_failed_task_plans()
    if LAST_OPTIONS_FILE.is_file():
        print("已载入上次使用的默认选项；每个作品都可以单独修改。")

    while True:
        if queue:
            print_smart_work_queue(queue, config)
        if failed_available:
            print("\n上次失败、可重新开始的任务：")
            failed_records = {str(item.get("plan_id")): item for item in read_failed_task_records()}
            for index, plan in enumerate(failed_available, start=1):
                reason = str(failed_records.get(plan.plan_id, {}).get("error") or "").strip()
                suffix = f" · {reason}" if reason else ""
                print(f"  F{index}. {plan.folder} · {mode_label(plan.mode)}{suffix}")
        if queue:
            prompt = (
                "请粘贴下一个作品路径；输入编号修改、D+编号删除、"
                "F+编号重试、X+编号丢弃失败记录，或直接按 Enter 开始："
            )
        elif failed_available:
            prompt = "请粘贴新作品路径；输入 F+编号加入重试、X+编号丢弃失败记录："
        else:
            prompt = "请粘贴第一个作品文件夹路径："
        raw = input(prompt).strip()

        if not raw:
            if queue:
                break
            print("还没有添加作品，请先输入一个文件夹路径。")
            continue

        failed_index = _prefixed_queue_index(raw, "f", len(failed_available))
        if failed_index is not None:
            failed_plan = failed_available.pop(failed_index - 1)
            if failed_plan.folder.resolve() in configured_folders:
                print("这个作品已经在作品串中。")
                failed_available.insert(failed_index - 1, failed_plan)
                continue
            work = work_from_failed_plan(failed_plan)
            work.plan = replace(work.plan, retry_of=failed_plan.plan_id)
            queue.append(work)
            configured_folders.add(work.plan.folder.resolve())
            defaults = work.defaults
            continue

        discard_index = _prefixed_queue_index(raw, "x", len(failed_available))
        if discard_index is not None:
            failed_plan = failed_available.pop(discard_index - 1)
            if ask_yes_no(
                f"确定丢弃失败记录“{failed_plan.folder.name}”？输入 Y 丢弃，输入 N 保留："
            ):
                remove_failed_task(failed_plan.plan_id)
                print("已丢弃失败记录。")
            else:
                failed_available.insert(discard_index - 1, failed_plan)
            continue

        if queue:
            edit_index = _queue_index(raw, len(queue))
            command = raw.casefold()
            if command.startswith(("d", "e")) or raw.isdigit():
                if edit_index is None:
                    print("作品编号无效，请重新输入。")
                    continue
                if command.startswith("d"):
                    work_name = queue[edit_index - 1].plan.folder.name
                    if not ask_yes_no(
                        f"确定从作品串删除第 {edit_index} 项“{work_name}”？"
                        "输入 Y 删除，输入 N 保留："
                    ):
                        continue
                    queue.pop(edit_index - 1)
                    configured_folders = {work.plan.folder.resolve() for work in queue}
                    if queue:
                        defaults = queue[-1].defaults
                        save_smart_plan_defaults(defaults)
                    print("已从作品串删除。")
                    continue

                work = queue[edit_index - 1]
                old_plan_id = work.plan.plan_id
                plan, edited_defaults, start_now = configure_smart_plan_draft(
                    work.draft,
                    config,
                    index=edit_index,
                    rebuild=args.rebuild,
                    force=args.force,
                )
                failed_ids = {item.plan_id for item in load_failed_task_plans()}
                if old_plan_id in failed_ids:
                    retry_of = work.plan.retry_of or old_plan_id
                else:
                    retry_of = work.plan.retry_of
                work.plan = replace(plan, retry_of=retry_of)
                work.defaults = edited_defaults
                if edit_index == len(queue):
                    defaults = edited_defaults
                    save_smart_plan_defaults(defaults)
                if start_now:
                    break
                continue

        try:
            folder = clean_user_path(raw)
        except VideoPreparerError as exc:
            print(f"路径无效：{exc}")
            continue
        resolved = folder.resolve()
        if resolved in configured_folders:
            print("这个作品已经在作品串中，请换一个文件夹。")
            continue
        try:
            draft = prepare_smart_plan_draft(config, folder, defaults, args)
        except VideoPreparerError as exc:
            print(f"无法配置这个作品：{exc}")
            continue
        plan, work_defaults, start_now = configure_smart_plan_draft(
            draft,
            config,
            index=len(queue) + 1,
            rebuild=args.rebuild,
            force=args.force,
        )
        if any(existing.plan.output_root == plan.output_root for existing in queue):
            print(f"这个输出目录已经被队列中的其他作品使用：{plan.output_root}")
            continue
        queue.append(QueuedSmartWork(draft=draft, plan=plan, defaults=work_defaults))
        configured_folders.add(resolved)
        defaults = work_defaults
        save_smart_plan_defaults(defaults)
        print(f"\n已加入作品串：{plan.folder.name}")
        if start_now:
            break
    return [work.plan for work in queue]


def execute_smart_queue(
    paths: ToolPaths,
    config: AppConfig,
    plans: Sequence[SmartTaskPlan],
    *,
    executor: Callable[[ToolPaths, AppConfig, SmartTaskPlan], None] | None = None,
    reference_event_callback: ReferenceEventCallback | None = None,
) -> int:
    """Run configured works in order, retaining failures and continuing the queue."""

    print(f"\n全部 {len(plans)} 个作品已经配置完成，现在按顺序开始处理。")
    failures: list[tuple[SmartTaskPlan, str]] = []
    for index, plan in enumerate(plans, start=1):
        print("\n" + "=" * 68)
        print(f"  队列 {index}/{len(plans)} · {plan.folder.name}")
        print("=" * 68)
        try:
            if executor is None:
                execute_prepared_smart_plan(
                    paths,
                    config,
                    plan,
                    reference_event_callback=reference_event_callback,
                )
            else:
                executor(paths, config, plan)
            remove_failed_task(plan.plan_id)
            remove_failed_task(plan.retry_of)
        except (KeyboardInterrupt, OperationCancelledError):
            raise
        except VideoPreparerError as exc:
            message = str(exc)
            failures.append((plan, message))
            save_failed_task(plan, error=message)
            if plan.retry_of and plan.retry_of != plan.plan_id:
                remove_failed_task(plan.retry_of)
            log_event(f"队列任务失败 source={plan.folder} error={message}；继续下一项")
            print(f"\n这个作品处理失败：{message}", file=sys.stderr)
            print("已有状态和成品已保留，继续处理下一个作品。", file=sys.stderr)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((plan, message))
            save_failed_task(plan, error=message)
            if plan.retry_of and plan.retry_of != plan.plan_id:
                remove_failed_task(plan.retry_of)
            log_event(f"队列任务未预期失败 source={plan.folder} error={message}；继续下一项")
            append_log(traceback.format_exc())
            print(f"\n这个作品发生未预期错误：{message}", file=sys.stderr)
            print("已有状态和成品已保留，继续处理下一个作品。", file=sys.stderr)

    succeeded = len(plans) - len(failures)
    print("\n" + "=" * 68)
    print(f"队列结束：成功 {succeeded} 个，失败 {len(failures)} 个。")
    if failures:
        print("失败项目：", file=sys.stderr)
        for plan, message in failures:
            print(f"  - {plan.folder}: {message}", file=sys.stderr)
        print(f"详细日志：{LOG_FILE}", file=sys.stderr)
        return 1
    return 0


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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print_header()
    try:
        config = load_app_config()
        paths = find_tool_paths(config)
        validate_asmr_version()
        if args.self_test:
            self_test(paths)
            return 0
        if args.scan == "legacy":
            folder = (
                clean_user_path(args.folder)
                if args.folder
                else clean_user_path(input("请粘贴解压后的作品文件夹路径："))
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
            )
        else:
            validate_interactive_queue_arguments(args)
            plans = collect_interactive_smart_plans(config, args)
            return execute_smart_queue(paths, config, plans)
        return 0
    except KeyboardInterrupt:
        log_event("用户取消任务")
        print("\n已取消。完整成品不会被半成品覆盖，已有任务状态会保留。")
        return 130
    except VideoPreparerError as exc:
        log_event(f"操作失败：{exc}")
        print(f"\n操作失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        log_event(f"未预期错误：{type(exc).__name__}: {exc}")
        append_log(traceback.format_exc())
        print(f"\n未预期错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
