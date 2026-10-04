from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..autoflow import engine
from ..errors import ProjectError
from ..platforms import portable_home
from ..user_settings import UserSettings, load_user_settings


def _clean_folder(value: Any) -> Path:
    text = str(value or "").strip(" \t\r\n\ufeff\u200b")
    quote_pairs = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’", "「": "」"}
    for _ in range(4):
        if len(text) >= 2 and text[0] in quote_pairs and text[-1] == quote_pairs[text[0]]:
            text = text[1:-1].strip()
            continue
        break
    if not text:
        raise ProjectError("请填写解压后的作品文件夹。")
    folder = Path(text).expanduser().resolve()
    if not folder.is_dir():
        raise ProjectError(f"作品文件夹不存在：{folder}")
    return folder


def _preferred_formats(value: str) -> tuple[str, ...]:
    formats: list[str] = []
    for raw in re.split(r"[,，;；\s]+", str(value or "")):
        item = raw.strip().casefold()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        if not re.fullmatch(r"\.[a-z0-9]{2,8}", item):
            raise ProjectError(f"音频格式写法无效：{raw}")
        if item not in formats:
            formats.append(item)
    if not formats:
        raise ProjectError("至少填写一种优先音频格式。")
    return tuple(formats)


def config_from_settings(settings: UserSettings | None = None) -> engine.AppConfig:
    current = settings or load_user_settings()
    output_name = current.autoflow_output_folder_name.strip()
    if (
        not output_name
        or output_name in {".", ".."}
        or output_name.rstrip(" .") != output_name
        or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in output_name)
    ):
        raise ProjectError("自动处理输出文件夹名称不符合 Windows 文件名规则。")
    return engine.AppConfig(
        asmr_root=next(
            (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
            portable_home().parent.resolve(),
        ),
        harmonized_volume_db=-abs(current.autoflow_harmonized_volume_reduction_db),
        harmonized_delay_seconds=round(current.autoflow_harmonized_delay_minutes * 60),
        timestamp_footer=current.autoflow_timestamp_footer.strip(),
        original_hard_subtitles=current.autoflow_original_hard_subtitles,
        timestamp_footer_position=current.autoflow_timestamp_footer_position,
        output_folder_name=output_name,
        default_output_layout=current.autoflow_default_layout,
        preferred_audio_formats=_preferred_formats(current.autoflow_preferred_audio_formats),
        bonus_policy="include" if current.autoflow_include_bonus else "exclude",
        background_policy=current.autoflow_background_policy,
        reference_wait_seconds=(
            current.autoflow_reference_wait_seconds
            if current.autoflow_reference_wait_enabled
            else 0
        ),
    )
