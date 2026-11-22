"""Content-addressed validation for the shared, rebuildable Chinese stem."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .hashing import cached_sha256_file
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
            "voice_match_path",
        }
    }
    parameters["source"] = kwargs["source_info"].model_dump(mode="json")
    # 3: per-channel mean-power loudness reference (was summed L^2 + R^2, +3 dB).
    # 4: +2 dB high-band rebuild in the fused voice match.
    # 5: tempo changes back to FFmpeg atempo.
    # 6: gate replacement_whisper_match to genuinely whispered/breathy originals.
    parameters["implementation"] = 6
    parameters["events"] = []
    for event in kwargs["events"]:
        check_cancelled()
        parameters["events"].append(
            {**asdict(event), "audio_path": cached_sha256_file(event.audio_path)}
        )
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
    for key in ("source_reference_path", "spatial_reference_path", "voice_match_path"):
        path = kwargs.get(key)
        parameters[key] = cached_sha256_file(Path(path)) if path else None
    fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            saved["key"] == fingerprint
            and destination.is_file()
            and cached_sha256_file(destination) == saved["sha256"]
        ):
            if kwargs.get("progress"):
                kwargs["progress"]("复用已完成的中文 RTF 音轨", 1, 1)
            return destination
    except (OSError, ValueError, KeyError, TypeError):
        pass
    result = builder(**kwargs)
    check_cancelled()
    atomic_write_text(
        receipt, json.dumps({"key": fingerprint, "sha256": cached_sha256_file(destination)})
    )
    return result


def variant_directory(project_dir: Path, variant: str | None) -> Path:
    if variant not in {None, "bilingual", "replace"}:
        raise ValueError("输出版本只能是 bilingual 或 replace。")
    return project_dir / "output" / variant if variant else project_dir / "output"


def compare_sentence_signature(project) -> str:
    rows = [
        (
            sentence.id,
            sentence.enabled,
            sentence.start_seconds,
            sentence.end_seconds,
            sentence.zh_text,
            sentence.tts_file,
            sentence.tts_cache_key,
            sentence.original_audio_enabled,
            sentence.original_audio_gain_db,
            sentence.chinese_audio_enabled,
            sentence.chinese_audio_gain_db,
        )
        for sentence in project.sentences
    ]
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_compare_state(project, project_dir: Path, key: str, *paths: Path) -> None:
    existing = [path for path in paths if path is not None and path.is_file()]
    if not existing:
        return
    payload = {
        "signature": compare_sentence_signature(project),
        "mtimes_ns": sorted({path.stat().st_mtime_ns for path in existing}),
        "size": existing[0].stat().st_size,
    }
    atomic_write_text(
        project_dir / "mix" / f"compare-{key}.json",
        json.dumps(payload, ensure_ascii=False),
    )
