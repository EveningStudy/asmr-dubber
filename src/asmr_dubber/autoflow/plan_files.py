"""Plan identity, metadata, manifests and output roots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import constants
from .constants import LAYOUT_BOTH, LAYOUT_MERGED, LAYOUT_SEPARATE, MODE_AUDIO
from .domain import (
    AudioSource,
)
from .errors import VideoPreparerError
from .fileio import atomic_write_text, file_stat_payload, safe_filename_component
from .modes import normalize_layout, normalize_mode
from .work_state import status_at_least


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
    subtitles_only: bool = False,
    source_subtitles_only: bool = False,
    subtitle_language: str = "source",
    subtitle_naming: str = "standard",
    subtitle_custom_name: str = "字幕",
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
                        "transcript_mode": item.transcript_mode,
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
    if subtitles_only:
        payload["subtitles_only"] = True
    if source_subtitles_only:
        payload["source_subtitles_only"] = True
        if (subtitle_language, subtitle_naming, subtitle_custom_name) != (
            "source",
            "standard",
            "字幕",
        ):
            payload["subtitle_options"] = [subtitle_language, subtitle_naming, subtitle_custom_name]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def plan_metadata_path(plan_id: str) -> Path:
    return constants.STATE_ROOT / f"smart-plan-{plan_id}.json"


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
    subtitles_only: bool = False,
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
        "subtitles_only": bool(subtitles_only),
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
                "transcript_mode": item.transcript_mode,
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
    root = constants.WORK_ROOT.resolve()
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
        raise VideoPreparerError(
            f"输出目录已有旧选项生成的结果：{output_root}。"
            "旧文件已保留；请编辑该作品并勾选“重新处理”，或更换成品输出文件夹。"
        )
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


def output_mapping_complete(
    outputs: Any,
    *,
    mode: str,
    subtitles_only: bool = False,
    embed_subtitles: bool = True,
    original_hard_subtitles: bool = False,
) -> bool:
    if not isinstance(outputs, dict):
        return False
    required = {"original", "srt", "lrc", "timestamps"}
    if original_hard_subtitles and normalize_mode(mode) != MODE_AUDIO:
        required.add("original_subtitle_video")
    if not subtitles_only:
        required.add("audio" if normalize_mode(mode) == MODE_AUDIO else "video")
    elif normalize_mode(mode) != MODE_AUDIO and embed_subtitles:
        required.add("video")
    return all(Path(str(outputs.get(key) or "")).is_file() for key in required) and all(
        Path(value).is_file()
        for key, value in outputs.items()
        if key.startswith(("bilingual_", "replace_"))
    )
