"""Content-addressed validation for the shared, rebuildable Chinese stem."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .audio import sha256_file
from .storage import atomic_write_text
from .task_control import check_cancelled


def build_cached_stem(builder, **kwargs):
    destination = Path(kwargs["destination"])
    receipt = destination.with_suffix(".cache.json")
    parameters = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "destination",
            "progress",
            "events",
            "source_info",
            "spatial_settings",
            "source_reference_path",
            "spatial_reference_path",
        }
    }
    parameters["source"] = kwargs["source_info"].model_dump(mode="json")
    parameters["implementation"] = 1
    parameters["events"] = []
    for event in kwargs["events"]:
        check_cancelled()
        parameters["events"].append({**asdict(event), "audio_path": sha256_file(event.audio_path)})
    settings = kwargs.get("spatial_settings")
    parameters["spatial"] = (
        {
            k: v
            for k, v in settings.model_dump(mode="json").items()
            if k.startswith("spatial_") or k == "chinese_line_peak_dbfs"
        }
        if settings
        else None
    )
    # References are derived from the verified source; include their actual content
    # so edits/replacement at the same path cannot return an obsolete stem.
    for key in ("source_reference_path", "spatial_reference_path"):
        path = kwargs.get(key)
        parameters[key] = sha256_file(path) if path else None
    fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            saved["key"] == fingerprint
            and destination.is_file()
            and sha256_file(destination) == saved["sha256"]
        ):
            if kwargs.get("progress"):
                kwargs["progress"]("复用已完成的中文 RTF 音轨", 1, 1)
            return destination
    except (OSError, ValueError, KeyError, TypeError):
        pass
    result = builder(**kwargs)
    check_cancelled()
    atomic_write_text(receipt, json.dumps({"key": fingerprint, "sha256": sha256_file(destination)}))
    return result


def variant_directory(project_dir: Path, variant: str | None) -> Path:
    if variant not in {None, "bilingual", "replace"}:
        raise ValueError("输出版本只能是 bilingual 或 replace。")
    return project_dir / "output" / variant if variant else project_dir / "output"
