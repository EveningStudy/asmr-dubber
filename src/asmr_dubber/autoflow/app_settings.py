"""AutoFlow settings file, remembered options and tool discovery."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ..environment import ffmpeg_executable
from ..platforms import current_platform, portable_home
from . import runlog
from .constants import (
    DEFAULT_HARMONIZED_DELAY_MINUTES,
    DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB,
    DEFAULT_TIMESTAMP_FOOTER,
    LAST_OPTIONS_FILE,
    LAST_OPTIONS_SCHEMA,
    LAYOUT_MERGED,
    MODE_VIDEO_HARMONIZED,
    MODE_VIDEO_NORMAL,
    PERIODIC_KEYFRAME_OPTIONS,
    SETTINGS_FILE,
    TOOL_ROOT,
)
from .domain import (
    AppConfig,
    SmartPlanDefaults,
    ToolPaths,
)
from .errors import VideoPreparerError
from .fileio import atomic_write_text
from .modes import normalize_layout, normalize_mode


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
                "original_hard_subtitles",
                "timestamp_footer_position",
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
    footer_position = values.get("timestamp_footer_position", "after").casefold()
    if footer_position not in {"before", "after"}:
        raise VideoPreparerError("timestamp_footer_position 必须是 before 或 after。")
    hard_subtitles = values.get("original_hard_subtitles", "false").casefold()
    if hard_subtitles not in {"true", "false"}:
        raise VideoPreparerError("original_hard_subtitles 必须是 true 或 false。")
    return AppConfig(
        asmr_root=asmr_root,
        harmonized_volume_db=-abs(reduction),
        harmonized_delay_seconds=round(delay_minutes * 60),
        timestamp_footer="\n".join(line for line in footer_lines if line),
        timestamp_footer_position=footer_position,
        original_hard_subtitles=hard_subtitles == "true",
        output_folder_name=output_folder_name,
        default_output_layout=default_output_layout,
        preferred_audio_formats=preferred_formats,
        bonus_policy=bonus_policy,
        background_policy=background_policy,
    )


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

    # Keep the launcher's explicit home: changing it can silently load another
    # settings.json (including the default IndexTTS2 and empty references).
    asmr_home = portable_home()
    platform_info = current_platform()
    if platform_info.is_windows:
        python_candidates = (
            asmr_home / "venv" / "Scripts" / "python.exe",
            asmr_root / ".venv" / "Scripts" / "python.exe",
        )
    else:
        python_candidates = (
            asmr_home / "venv" / "bin" / "python",
            asmr_root / ".venv" / "bin" / "python",
        )
    python = next((path for path in python_candidates if path.is_file()), None)
    if (
        python is None
        and (asmr_root / "src" / "asmr_dubber").resolve() == Path(__file__).resolve().parents[1]
    ):
        # Editable installs may deliberately keep their venv outside the repository.
        python = Path(sys.executable)
    if python is None:
        raise VideoPreparerError("ASMR Dubber 尚未安装完整运行环境，找不到便携 Python。")

    try:
        ffmpeg = Path(ffmpeg_executable()).expanduser().resolve()
    except Exception as exc:
        raise VideoPreparerError(f"ASMR Dubber 的 FFmpeg 不可用，请先修复其安装：{exc}") from exc

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

    powershell_path = None
    if platform_info.is_windows:
        powershell_path = shutil.which("pwsh") or shutil.which("powershell")

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
        ffmpeg=ffmpeg,
        cli_command=(str(python), "-m", "asmr_dubber.cli"),
        ui_command=(str(python), "-m", "asmr_dubber.cli", "ui"),
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
    runlog.log_event(f"检测到 ASMR Dubber {version}")
    return version
