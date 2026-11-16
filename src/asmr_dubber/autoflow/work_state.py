"""Resumable per-work state files and workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import constants
from .constants import (
    DEFAULT_HARMONIZED_DELAY_MINUTES,
    DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB,
    MODE_AUDIO,
    STATUS_ORDER,
)
from .domain import (
    REFERENCE_SELECTION_TIMEOUT_SECONDS,
    TRANSCRIPT_MODE_DIRECT,
    AppConfig,
    AudioSource,
)
from .errors import VideoPreparerError
from .fileio import file_stat_payload
from .modes import normalize_mode


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
                "size": item.path.stat().st_size,
                "mtime_ns": item.path.stat().st_mtime_ns,
                "category": item.category,
                "transcript": file_stat_payload(item.transcript_path),
                "transcript_language": item.transcript_language,
                "transcript_timed": item.transcript_timed,
                "transcript_mode": item.transcript_mode,
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
    return constants.STATE_ROOT / f"{task_key(folder)}.json"


def workspace_path(folder: Path) -> Path:
    return constants.WORK_ROOT / task_key(folder)


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
    return constants.STATE_ROOT / f"smart-{planned_job_key(source_folder, plan_id, job_id)}.json"


def planned_workspace_path(source_folder: Path, plan_id: str, job_id: str) -> Path:
    return constants.WORK_ROOT / f"smart-{planned_job_key(source_folder, plan_id, job_id)}"


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
    payload.setdefault("subtitles_only", False)
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
        and str(item.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT) == TRANSCRIPT_MODE_DIRECT
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
    root = constants.WORK_ROOT.resolve()
    target = workspace.resolve()
    if target.parent != root:
        raise VideoPreparerError(f"拒绝清理非任务目录：{target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


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


def create_initial_state(
    folder: Path,
    mode: str,
    sources: list[AudioSource],
    background: Path | None,
    config: AppConfig,
    *,
    embed_subtitles: bool = True,
    subtitles_only: bool = False,
) -> dict[str, Any]:
    mode = normalize_mode(mode)
    return {
        "schema": 1,
        "source_folder": str(folder),
        "mode": mode,
        "harmonized_volume_db": config.harmonized_volume_db,
        "harmonized_delay_seconds": config.harmonized_delay_seconds,
        "timestamp_footer": config.timestamp_footer,
        "timestamp_footer_position": config.timestamp_footer_position,
        "original_hard_subtitles": config.original_hard_subtitles,
        "reference_wait_seconds": max(0, int(config.reference_wait_seconds)),
        "output_policy": config.output_policy,
        "status": "",
        "fingerprint": fingerprint(sources, background),
        "background": str(background) if background else None,
        "embed_subtitles": bool(embed_subtitles),
        "subtitles_only": bool(subtitles_only),
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
    subtitles_only: bool,
) -> dict[str, Any]:
    state = create_initial_state(
        output_folder,
        mode,
        sources,
        background,
        config,
        embed_subtitles=embed_subtitles,
        subtitles_only=subtitles_only,
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
        for key, value in outputs.items():
            if key.startswith(("bilingual_", "replace_")):
                require_file(Path(value))
        if state.get("original_hard_subtitles") and normalize_mode(state["mode"]) != MODE_AUDIO:
            require_file(
                Path(str(outputs.get("original_subtitle_video") or folder / "原声字幕版.mp4"))
            )
        subtitles_only = bool(state.get("subtitles_only", False))
        mode = normalize_mode(state["mode"])
        if not subtitles_only:
            primary_key = "audio" if mode == MODE_AUDIO else "video"
            default_primary = folder / ("双语版.wav" if primary_key == "audio" else "双语版.mp4")
            require_file(Path(str(outputs.get(primary_key) or default_primary)))
        elif mode != MODE_AUDIO and bool(state.get("embed_subtitles", True)):
            require_file(Path(str(outputs.get("video") or folder / "原声字幕版.mp4")))
        for key, name in (("srt", "双语版.srt"), ("lrc", "双语版.lrc")):
            require_file(Path(str(outputs.get(key) or folder / name)))
    if status_at_least(state, "completed"):
        timestamp_file = folder / "时间戳.txt"
        require_file(timestamp_file)
    return missing
