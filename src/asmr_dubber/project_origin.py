"""Readable batch project names and origin labels, including old checkpoints."""

import json
from pathlib import Path

from .storage import atomic_write_text


def batch_project_name(work, track, plan_id):
    from .autoflow.engine import safe_filename_component

    label = (
        f"批量_{safe_filename_component(str(work), limit=28)}_"
        f"{safe_filename_component(str(track), limit=24)}"
    )
    return f"{label}_{str(plan_id)[:8]}"


def record_batch_origin(manifest, *, source_folder, plan_id, job_id, track):
    origin = {
        "work": Path(source_folder).name,
        "source_folder": str(source_folder),
        "plan_id": str(plan_id),
        "job_id": str(job_id),
        "track": str(track),
    }
    atomic_write_text(
        Path(manifest).parent / "batch-origin.json",
        json.dumps(origin, ensure_ascii=False, indent=2),
    )


def origin_index():
    from .autoflow.engine import STATE_ROOT

    result = {}
    for path in STATE_ROOT.glob("smart-*.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                continue
            if state.get("project_json") and state.get("source_folder"):
                track = "、".join(
                    t.get("filename", "")
                    for t in (state.get("timeline") or [])
                    if isinstance(t, dict)
                )
                result[str(Path(state["project_json"]).parent.resolve())] = {
                    "work": Path(state["source_folder"]).name,
                    "plan_id": state.get("plan_id", ""),
                    "track": track or state.get("job_id", "合并版"),
                    "source_folder": state["source_folder"],
                }
        except (OSError, ValueError, TypeError):
            continue
    return result


def project_origin(directory, index=None):
    directory = Path(directory)
    try:
        origin = json.loads((directory / "batch-origin.json").read_text(encoding="utf-8"))
        if not isinstance(origin, dict):
            origin = {}
    except (OSError, ValueError):
        origin = (index or {}).get(str(directory.resolve()), {})
    label = " · ".join(str(origin.get(k) or "") for k in ("work", "track")).strip(" ·")
    if origin.get("plan_id"):
        label += f" [{str(origin['plan_id'])[:8]}]"
    return {"label": label or directory.name, "batch": origin, "path": str(directory)}


def project_inventory(root):
    root = Path(root)
    if not root.is_dir():
        return []
    index = origin_index()
    return [project_origin(p.parent, index) for p in sorted(root.glob("*/project.json"))]
