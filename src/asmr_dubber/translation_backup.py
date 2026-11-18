"""Safety copy of translations taken before a full re-translation.

A forced re-translation replaces text sentence by sentence.  The copy written here
lets the user return to the previous version, including hand-edited lines, when the
new result turns out worse or the run was interrupted.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ProjectError
from .models import DubProject
from .storage import atomic_write_text

_BACKUP_FIELDS = (
    "target_text",
    "status",
    "error_code",
    "error",
    "enabled",
    "chinese_audio_enabled",
    "review_locked",
    "translation_warning",
)
_BACKUP_PATTERN = re.compile(r"^translations-backup-r(\d+)\.json$")


def backup_path(project_dir: Path, revision: int) -> Path:
    return Path(project_dir) / f"translations-backup-r{revision}.json"


def write_translation_backup(project: DubProject, project_dir: Path) -> Path | None:
    """Save the current translations; returns None when there is nothing to keep."""

    rows = [
        {"id": sentence.id, **{name: getattr(sentence, name) for name in _BACKUP_FIELDS}}
        for sentence in project.sentences
        if sentence.zh_text or sentence.status == "skipped_sound"
    ]
    if not rows:
        return None
    path = backup_path(project_dir, project.revision)
    payload = {
        "revision": project.revision,
        "language": project.translation_language,
        "created_at": datetime.now(UTC).isoformat(),
        "sentences": rows,
    }
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def latest_backup(project_dir: Path) -> Path | None:
    candidates = [
        (int(match.group(1)), path)
        for path in Path(project_dir).glob("translations-backup-r*.json")
        if (match := _BACKUP_PATTERN.match(path.name))
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def backup_summary(project_dir: Path) -> dict[str, Any] | None:
    """Small description for the UI: which revision, and how many lines it holds."""

    path = latest_backup(project_dir)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"revision": int(payload["revision"]), "count": len(payload["sentences"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def restore_translation_backup(project: DubProject, project_dir: Path) -> int:
    """Put the saved translations back; returns how many sentences were restored."""

    path = latest_backup(project_dir)
    if path is None:
        raise ProjectError("没有可恢复的译文备份。")
    try:
        saved = {
            str(item["id"]): item
            for item in json.loads(path.read_text(encoding="utf-8"))["sentences"]
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProjectError(f"译文备份无法读取：{exc}") from exc
    restored = 0
    for sentence in project.sentences:
        item = saved.get(sentence.id)
        if item is None:
            continue
        saved_text = str(item.get("target_text", item.get("zh_text", "")))
        if sentence.target_text != saved_text or item.get("status") in {
            "skipped_sound",
            "skipped_filler",
        }:
            sentence.tts_file = None
            sentence.tts_cache_key = None
            sentence.tts_duration_seconds = None
            sentence.tts_candidates = []
            sentence.tts_warning = None
            sentence.tts_attempt = 0
            if "translation_warning" not in item:
                sentence.translation_warning = None
        sentence.target_text = saved_text
        for name in _BACKUP_FIELDS:
            if name != "target_text" and name in item:
                setattr(sentence, name, item[name])
        if not sentence.error:
            sentence.error_code = None
        restored += 1
    if restored:
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_srt_file = None
        project.subtitle_lrc_file = None
        project.subtitle_video_file = None
        if hasattr(project, "translation_proposals"):
            project.translation_proposals = []
    return restored


def restore_project_translations(project_path: str) -> int:
    """Reload the project from disk, restore the backup and save the result."""

    from . import pipeline
    from .models import save_project

    project, directory = pipeline.reload_project(project_path)
    restored = restore_translation_backup(project, directory)
    project.translation_proposals = []
    save_project(project, directory)
    pipeline.export_transcript(project, directory)
    return restored
