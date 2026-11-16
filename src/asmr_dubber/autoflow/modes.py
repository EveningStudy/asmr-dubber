"""Normalisation and labels for processing modes and output layouts."""

from __future__ import annotations

from typing import Any

from .constants import (
    LAYOUT_ALIASES,
    LAYOUT_BOTH,
    LAYOUT_MERGED,
    LAYOUT_SEPARATE,
    MODE_ALIASES,
    MODE_AUDIO,
    MODE_VIDEO_HARMONIZED,
    MODE_VIDEO_NORMAL,
)
from .domain import (
    AppConfig,
)
from .errors import VideoPreparerError


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


def parse_embed_subtitles_argument(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise VideoPreparerError(f"未知的字幕内嵌选项：{value}")


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
