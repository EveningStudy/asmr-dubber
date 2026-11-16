"""Preparing the media files of one work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import (
    media_audio,
    media_video,
)
from .constants import (
    MODE_AUDIO,
    MODE_VIDEO_NORMAL,
)
from .domain import (
    AudioSource,
    ToolPaths,
)
from .fileio import atomic_copy
from .modes import normalize_mode


def prepare_media_phase(
    paths: ToolPaths,
    folder: Path,
    state: dict[str, Any],
    sources: list[AudioSource],
) -> None:
    workspace = Path(state["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    background = Path(state["background"]) if state.get("background") else None
    master, timeline = media_audio.normalize_and_concat(paths, sources, workspace)
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
        media_video.render_static_video(paths, master, background, original)
        dubbing_input = original
    else:
        print("\n[2/5] 生成和谐静态视频")
        original = folder / "原声.mp4"
        dubbing_input = workspace / "dubbing_input.mp4"
        print("  生成供 ASMR Dubber 使用的无前导正常音量版本……")
        media_video.render_static_video(paths, master, background, dubbing_input)
        harmonized_volume_db = float(state["harmonized_volume_db"])
        harmonized_delay_seconds = int(state["harmonized_delay_seconds"])
        print(
            f"  生成 {harmonized_volume_db:g} dB 且前置 "
            f"{harmonized_delay_seconds / 60:g} 分钟的原声版本……"
        )
        media_video.render_static_video(
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
