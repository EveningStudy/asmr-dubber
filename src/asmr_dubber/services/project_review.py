"""What the sentence table shows about the full-replacement mix: report flags and compare tracks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..experimental_mix import sentence_is_replaced
from ..models import DubProject
from ..stem_cache import compare_sentence_signature

REPORT = Path("mix/replacement-report.json")
_VARIANTS = {"bilingual": "bilingual", "replace": "replacement"}


def replacement_flags(project: DubProject, directory: Path) -> dict[str, dict]:
    """Residual-voice figures from the last replacement mix, keyed by sentence id.

    The report is an energy estimate written by ``compose_speech_replacement``.  Entries for
    sentences that no longer exist, or whose original voice the user has since chosen to keep,
    are dropped so the table never flags something that has already been dealt with.
    """
    try:
        payload = json.loads((directory / REPORT).read_text(encoding="utf-8"))
        entries = payload["sentences"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    current = {sentence.id: sentence for sentence in project.sentences}
    flags: dict[str, dict] = {}
    for entry in entries if isinstance(entries, list) else []:
        sentence = current.get(str(entry.get("id"))) if isinstance(entry, dict) else None
        if sentence is None or not entry.get("exceeds") or not sentence_is_replaced(sentence):
            continue
        flags[sentence.id] = {
            "reasons": [str(reason) for reason in entry.get("reasons", [])],
            "residual_db": entry.get("residual_db"),
            "uncovered_before_seconds": entry.get("uncovered_before_seconds", 0),
            "uncovered_after_seconds": entry.get("uncovered_after_seconds", 0),
            "recognized_residual": str(entry.get("recognized_residual") or ""),
        }
    return flags


def _audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (
            path
            for path in folder.glob("*.wav")
            if path.is_file() and "chinese-voice-stem" not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return first.resolve() == second.resolve()


def _is_track_stale(
    project: DubProject,
    directory: Path,
    key: str,
    path: Path,
    current_output: Path | None,
) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return True
    mtime = stat.st_mtime
    root = directory.resolve()
    for sentence in project.sentences:
        if not sentence.tts_file:
            continue
        dub_path = (directory / sentence.tts_file).resolve()
        if (
            dub_path.is_relative_to(root)
            and dub_path.is_file()
            and dub_path.stat().st_mtime > mtime + 0.5
        ):
            return True
    try:
        updated_at_epoch = datetime.fromisoformat(project.updated_at).timestamp()
    except ValueError:
        updated_at_epoch = 0.0
    if current_output is None and (
        mtime + 2.0 < updated_at_epoch or (project.revision > 0 and mtime <= updated_at_epoch)
    ):
        return True
    try:
        saved = json.loads((directory / "mix" / f"compare-{key}.json").read_text(encoding="utf-8"))
        if stat.st_mtime_ns in saved.get("mtimes_ns", []) and stat.st_size == saved.get("size"):
            return saved.get("signature") != compare_sentence_signature(project)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    if current_output is not None and _same_file(path, current_output):
        return False
    return mtime + 2.0 < updated_at_epoch


def compare_tracks(
    project: DubProject, directory: Path, source: str | None, register: Callable[[Path], str | None]
) -> list[dict]:
    """The finished mixes that exist for this project, for side-by-side listening.

    Batch and CLI runs keep each version in ``output/bilingual`` and ``output/replace``.  A
    file directly in ``output`` belongs to whichever mix mode the project is set to.
    """
    tracks: list[dict] = []
    if source:
        tracks.append({"key": "source", "url": source})
    found: dict[str, Path] = {}
    for variant, key in _VARIANTS.items():
        files = _audio_files(directory / "output" / variant)
        if files:
            found[key] = files[0]
    root = directory.resolve()
    main = (directory / project.output_file).resolve() if project.output_file else None
    current_output = (
        main if main is not None and main.is_relative_to(root) and main.is_file() else None
    )
    if current_output is not None and current_output.parent.name not in _VARIANTS:
        # The project's own current output is the one that still matches its sentences.
        found[
            "replacement" if project.settings.separation_mix_mode == "replace" else "bilingual"
        ] = current_output
    for key in ("bilingual", "replacement"):
        if key not in found:
            continue
        url = register(found[key])
        if url:
            tracks.append(
                {
                    "key": key,
                    "url": url,
                    "modified": round(found[key].stat().st_mtime),
                    "stale": _is_track_stale(project, directory, key, found[key], current_output),
                }
            )
    return tracks
