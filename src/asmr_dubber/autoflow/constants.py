"""Shared constants and state locations for AutoFlow."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..platforms import portable_home
from .domain import (
    TRANSCRIPT_MODE_DIRECT,
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


TRANSCRIPT_MODE_ASR_RECONCILE = "asr_reconcile"


TRANSCRIPT_MODES = {TRANSCRIPT_MODE_DIRECT, TRANSCRIPT_MODE_ASR_RECONCILE}


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


DEFAULT_TIMESTAMP_FOOTER = """双语音声制作器：BV1f43G6YEov
内嵌字幕和配音为本地AI生成。内容仅供参考。
仅供日语学习，有能力请购买正版支持。"""
