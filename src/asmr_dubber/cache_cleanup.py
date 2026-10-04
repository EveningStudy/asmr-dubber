"""Conservative project cache inventory and explicit, snapshot-bound cleanup.

Only named derived artifacts are eligible. Never recurse-delete a project,
follow junctions, or trust a stale browser preview as permission for new files.
"""

import json
import re
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .audio import sha256_file
from .errors import ProjectError
from .models import load_project
from .storage import exclusive_file_lock


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


def _signature(path: Path) -> list[int]:
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]


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
                "analysis/spatial-reference.wav",
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
    separation = directory / "analysis/separation"
    if separation.is_dir() and plain_path(separation, directory):
        for folder in separation.iterdir():
            if not folder.is_dir() or not plain_path(folder, directory):
                continue
            done = folder / "complete.json"
            stems = [folder / "vocals.wav", folder / "background.wav"]
            if not plain_path(done, directory) or not all(plain_path(p, directory) for p in stems):
                continue  # Incomplete/failed jobs retain every restart checkpoint.
            try:
                hashes = json.loads(done.read_text(encoding="utf-8"))
                if not all(sha256_file(p) == hashes.get(p.name) for p in stems):
                    continue
            except (OSError, ValueError, AttributeError):
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
                files.extend([done, *stems, folder / "asr_16k_mono.wav"])
    return [
        p
        for p in dict.fromkeys(files)
        if plain_path(p, directory)
        and stat.S_ISREG(p.lstat().st_mode)
        and p.resolve() not in protected
    ]


def scan_caches(root_text: str, categories: list[str]) -> dict:
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
                        "files": {str(p.relative_to(directory)): _signature(p) for p in files},
                    }
                )
        except (OSError, ValueError, ProjectError) as exc:
            plan["skipped"].append(f"{directory.name}: {exc}")
    return plan


def clean_caches(plan: dict, selected: list[str], confirmed: bool) -> tuple[int, list[str]]:
    if not confirmed:
        raise ProjectError("请先确认清理范围及重建影响。")
    if not plan or not selected:
        raise ProjectError("请先扫描并选择项目。")
    removed = 0
    skipped = []
    root = Path(plan["root"])
    for entry in plan["projects"]:
        if entry["path"] not in selected:
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
                        removed += signature[0]
                    except OSError as exc:
                        skipped.append(f"{directory.name}/{relative}: {exc}")
        except (OSError, ValueError, ProjectError) as exc:
            skipped.append(f"{directory.name}: {exc}")
    return removed, skipped
