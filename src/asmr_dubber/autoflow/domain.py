"""Typed AutoFlow plans and settings; no filesystem or process side effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import ScanResult

REFERENCE_SELECTION_TIMEOUT_SECONDS = 5 * 60
TRANSCRIPT_MODE_DIRECT = "direct"


@dataclass(frozen=True)
class AppConfig:
    asmr_root: Path | None
    harmonized_volume_db: float
    harmonized_delay_seconds: int
    timestamp_footer: str
    original_hard_subtitles: bool = False
    timestamp_footer_position: str = "after"
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
    ffmpeg: Path
    cli_command: tuple[str, ...]
    ui_command: tuple[str, ...]
    powershell: str | None
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
    transcript_mode: str = TRANSCRIPT_MODE_DIRECT
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
    subtitles_only: bool = False
    source_subtitles_only: bool = False


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
