"""Conservative project cache inventory and explicit, snapshot-bound cleanup.

Only named derived artifacts are eligible. Never recurse-delete a project,
follow junctions, or trust a stale browser preview as permission for new files.
"""

import json
import re
import stat
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .audio import sha256_file
from .errors import ProjectError
from .models import load_project
from .platforms import portable_home
from .storage import exclusive_file_lock

_SMALL_STEM_VERIFY_BYTES = 4 * 1024 * 1024
_STEM_HASH_LOCK = threading.Lock()
_STEM_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@contextmanager
def _cleanup_lock(directory: Path):
    with ExitStack() as stack:
        stack.enter_context(exclusive_file_lock(directory / ".project.lock", timeout_seconds=0))
        separation = directory / "analysis/separation"
        if separation.is_dir() and plain_path(separation, directory):
            for folder in sorted(separation.iterdir()):
                if folder.is_dir() and plain_path(folder, directory):
                    stack.enter_context(exclusive_file_lock(folder / ".lock", timeout_seconds=0))
        yield


def plain_path(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        for item in (path, *path.parents):
            info = item.lstat()
            if item.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                return False
            if item == root:
                break
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _signature(path: Path) -> list[int | str]:
    s = path.stat()
    return [s.st_size, str(s.st_mtime_ns), str(s.st_ctime_ns), str(s.st_ino)]


def _referenced_files(value, directory: Path) -> set[Path]:
    """Protect even user-selected references that happen to live in cache folders."""
    if isinstance(value, dict):
        return set().union(*(_referenced_files(v, directory) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_referenced_files(v, directory) for v in value))
    if isinstance(value, str) and value and len(value) < 1024:
        try:
            path = Path(value)
            path = path if path.is_absolute() else directory / path
            if path.is_file():
                return {path.resolve()}
        except (OSError, ValueError):
            pass
    return set()


def _stems_complete(done: Path, stems: list[Path]) -> bool:
    try:
        hashes = json.loads(done.read_text(encoding="utf-8"))
        if not isinstance(hashes, dict):
            return False
        signatures = (
            hashes.get("signatures") if isinstance(hashes.get("signatures"), dict) else None
        )
        for path in stems:
            expected = hashes.get(path.name)
            if not isinstance(expected, str) or not expected:
                return False
            st = path.stat()
            if st.st_size <= 0:
                return False
            if signatures is not None:
                sig = signatures.get(path.name)
                if not (
                    isinstance(sig, list)
                    and len(sig) >= 2
                    and (st.st_size, st.st_mtime_ns) == (sig[0], sig[1])
                ):
                    return False
            elif st.st_size <= _SMALL_STEM_VERIFY_BYTES:
                cache_key = (str(path), st.st_size, st.st_mtime_ns)
                with _STEM_HASH_LOCK:
                    digest = _STEM_HASH_CACHE.get(cache_key)
                if digest is None:
                    digest = sha256_file(path)
                    with _STEM_HASH_LOCK:
                        if len(_STEM_HASH_CACHE) > 256:
                            _STEM_HASH_CACHE.clear()
                        _STEM_HASH_CACHE[cache_key] = digest
                if digest != expected:
                    return False
            elif len(expected) != 64:
                return False
        return True
    except (OSError, ValueError, AttributeError):
        return False


def _candidates(directory: Path, categories: list[str]) -> list[Path]:
    if not plain_path(directory / "project.json", directory):
        raise ProjectError("项目文件不是普通本地路径。")
    project, _ = load_project(directory / "project.json")
    protected = _referenced_files(project.model_dump(mode="json"), directory)
    files: list[Path] = []
    if "rebuild" in categories:
        files.extend(
            directory / name
            for name in (
                "analysis/asr_16k_mono.wav",
                "analysis/asr_16k_dominant.wav",
                "analysis/spatial-reference.wav",
                "analysis/spatial-reference-vocal.wav",
                "analysis/spatial-reference-vocal.key",
                "analysis/replacement_loudness_16k_mono.wav",
                "analysis/replacement_loudness_source.sha256",
                "mix/replacement-background.wav",
                "mix/chinese_stem_float32.wav",
                "mix/chinese_stem_float32.cache.json",
                "mix/bilingual/chinese_stem_float32.wav",
                "mix/bilingual/chinese_stem_float32.cache.json",
                "mix/replace/chinese_stem_float32.wav",
                "mix/replace/chinese_stem_float32.cache.json",
            )
        )
    if "separation" in categories:
        files.extend(
            directory / name
            for name in (
                "analysis/spatial-reference-vocal.wav",
                "analysis/spatial-reference-vocal.key",
            )
        )
    separation = directory / "analysis/separation"
    if separation.is_dir() and plain_path(separation, directory):
        for folder in separation.iterdir():
            if not folder.is_dir() or not plain_path(folder, directory):
                continue
            done = folder / "complete.json"
            stems = [folder / "vocals.wav", folder / "background.wav"]
            if not plain_path(done, directory) or not all(plain_path(p, directory) for p in stems):
                continue  # Incomplete/failed jobs retain every restart checkpoint.
            if not _stems_complete(done, stems):
                continue
            if "safe" in categories or "separation" in categories:
                files.append(folder / "source.wav")
                files.extend(
                    p
                    for p in folder.iterdir()
                    if re.fullmatch(r"chunk-\d{6}\.(wav|sha256)", p.name)
                )
            if "separation" in categories:
                # Do not leave a completion receipt after removing its stems.
                if any(p.resolve() in protected for p in stems):
                    continue
                files.extend(
                    [done, *stems, folder / "asr_16k_mono.wav", folder / "asr_16k_dominant.wav"]
                )
    return [
        p
        for p in dict.fromkeys(files)
        if plain_path(p, directory)
        and stat.S_ISREG(p.lstat().st_mode)
        and p.resolve() not in protected
    ]


def _resumable_workspaces(state_root: Path) -> set[Path]:
    active: set[Path] = set()
    if not state_root.is_dir():
        return active
    for state_file in state_root.glob("*.json"):
        if state_file.name.startswith("smart-plan-") or state_file.name in {
            "last-options.json",
            "failed-tasks.json",
        }:
            continue
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        workspace = str(payload.get("workspace") or "").strip()
        if workspace and str(payload.get("status") or "") != "completed":
            try:
                active.add(Path(workspace).expanduser().resolve())
            except OSError:
                continue
    return active


def _safe_app_cache_entries(home: Path) -> list[dict]:
    entries: list[dict] = []
    work_root = home / "autoflow" / ".work"
    if work_root.is_dir() and plain_path(work_root, home):
        resumable = _resumable_workspaces(home / "autoflow" / ".state")
        work_files: dict[str, list[int | str]] = {}
        for item in sorted(work_root.iterdir()):
            if not plain_path(item, work_root):
                continue
            try:
                if stat.S_ISREG(item.lstat().st_mode):
                    work_files[str(item.relative_to(work_root))] = _signature(item)
                elif item.is_dir() and item.resolve() not in resumable:
                    for child in sorted(item.rglob("*")):
                        if plain_path(child, work_root) and stat.S_ISREG(child.lstat().st_mode):
                            work_files[str(child.relative_to(work_root))] = _signature(child)
            except OSError:
                continue
        if work_files:
            entries.append(
                {
                    "path": str(work_root),
                    "label": "AutoFlow 临时工作区",
                    "kind": "app_cache",
                    "files": work_files,
                }
            )
    temp_root = home / "temp"
    if temp_root.is_dir() and plain_path(temp_root, home):
        diag_files: dict[str, list[int | str]] = {}
        for item in sorted(temp_root.glob("ASMR-Dubber-diagnostics-*.zip")):
            try:
                if plain_path(item, temp_root) and stat.S_ISREG(item.lstat().st_mode):
                    diag_files[item.name] = _signature(item)
            except OSError:
                continue
        if diag_files:
            entries.append(
                {
                    "path": str(temp_root),
                    "label": "诊断包临时文件",
                    "kind": "app_cache",
                    "files": diag_files,
                }
            )
    return entries


def scan_caches(root_text: str, categories: list[str], *, app_home: Path | None = None) -> dict:
    from .project_origin import origin_index, project_origin

    origins = origin_index()
    root = Path(root_text).expanduser().absolute()
    if not root.is_dir() or not plain_path(root, root):
        raise ProjectError("请选择实际项目目录；不接受符号链接或目录联接。")
    plan = {"root": str(root), "categories": list(categories), "projects": [], "skipped": []}
    directories = [root] if (root / "project.json").is_file() else sorted(root.iterdir())
    for directory in directories:
        if not directory.is_dir() or not plain_path(directory, root):
            continue
        if not (directory / "project.json").is_file():
            continue
        try:
            with _cleanup_lock(directory):
                files = _candidates(directory, categories)
                plan["projects"].append(
                    {
                        "path": str(directory),
                        **project_origin(directory, origins),
                        "files": {str(p.relative_to(directory)): _signature(p) for p in files},
                    }
                )
        except (OSError, ValueError, ProjectError) as exc:
            plan["skipped"].append(f"{directory.name}: {exc}")
    resolved_home: Path | None = None
    if app_home is not None:
        resolved_home = app_home.expanduser().resolve()
    else:
        default_home = portable_home().resolve()
        if root.resolve() in {default_home, (default_home / "projects").resolve()}:
            resolved_home = default_home
    if resolved_home is not None and "safe" in categories:
        plan["projects"].extend(_safe_app_cache_entries(resolved_home))
    return plan


def _clean_app_cache_entry(entry: dict, home: Path, skipped: list[str]) -> int:
    directory = Path(entry["path"])
    allowed = {home / "autoflow" / ".work", home / "temp"}
    try:
        resolved_dir = directory.resolve()
    except OSError:
        resolved_dir = directory
    if resolved_dir not in {p.resolve() for p in allowed} or not plain_path(directory, home):
        skipped.append(f"{directory.name}: 路径已改变")
        return 0
    removed = 0
    for relative, signature in entry["files"].items():
        path = directory / relative
        if not plain_path(path, directory):
            skipped.append(f"{directory.name}/{relative}: 已变化或受保护")
            continue
        try:
            if _signature(path) != signature:
                skipped.append(f"{directory.name}/{relative}: 扫描后已更新")
                continue
            path.unlink()
            removed += int(signature[0])
        except OSError as exc:
            skipped.append(f"{directory.name}/{relative}: {exc}")
    if resolved_dir == (home / "autoflow" / ".work").resolve():
        for child in sorted(directory.iterdir(), reverse=True):
            if child.is_dir() and plain_path(child, directory):
                try:
                    if not any(child.iterdir()):
                        child.rmdir()
                except OSError:
                    pass
    return removed


def clean_caches(plan: dict, selected: list[str], confirmed: bool) -> tuple[int, list[str]]:
    if not confirmed:
        raise ProjectError("请先确认清理范围及重建影响。")
    if not plan or not selected:
        raise ProjectError("请先扫描并选择项目。")
    removed = 0
    skipped = []
    root = Path(plan["root"])
    home = portable_home().resolve()
    for entry in plan["projects"]:
        if entry["path"] not in selected:
            continue
        if entry.get("kind") == "app_cache":
            removed += _clean_app_cache_entry(entry, home, skipped)
            continue
        directory = Path(entry["path"])
        if not plain_path(directory, root):
            skipped.append(f"{directory.name}: 路径已改变")
            continue
        try:
            with _cleanup_lock(directory):
                eligible = set(_candidates(directory, plan["categories"]))
                # Rebuild eligibility and file identity under the same operation lock.
                for relative, signature in entry["files"].items():
                    path = directory / relative
                    if path not in eligible or not plain_path(path, directory):
                        skipped.append(f"{directory.name}/{relative}: 已变化或受保护")
                        continue
                    try:
                        if _signature(path) != signature:
                            skipped.append(f"{directory.name}/{relative}: 扫描后已更新")
                            continue
                        path.unlink()
                        removed += int(signature[0])
                    except OSError as exc:
                        skipped.append(f"{directory.name}/{relative}: {exc}")
        except (OSError, ValueError, ProjectError) as exc:
            skipped.append(f"{directory.name}: {exc}")
    return removed, skipped
