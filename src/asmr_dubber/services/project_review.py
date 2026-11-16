"""What the sentence table shows about the full-replacement mix: report flags and compare tracks."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ..experimental_mix import sentence_is_replaced
from ..models import DubProject

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
    main = (directory / project.output_file).resolve() if project.output_file else None
    if (
        main is not None
        and main.is_relative_to(directory.resolve())
        and main.is_file()
        and main.parent.name not in _VARIANTS
    ):
        # The project's own current output is the one that still matches its sentences.
        found[
            "replacement" if project.settings.separation_mix_mode == "replace" else "bilingual"
        ] = main
    for key in ("bilingual", "replacement"):
        url = register(found[key]) if key in found else None
        if url:
            tracks.append({"key": key, "url": url, "modified": round(found[key].stat().st_mtime)})
    return tracks
