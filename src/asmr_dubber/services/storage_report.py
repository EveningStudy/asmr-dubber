"""Where the program's disk space goes, folder by folder, and a safe way to open each one.

Users mostly see "the program takes 80 GB" without knowing why: the backends'
own Python environments (PyTorch for IndexTTS, separation, MOSS) are often the
largest part, so they are counted and shown next to projects and models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..model_packs import model_pack_directory
from ..platforms import open_directory, portable_home
from . import models

# key -> (label, hint); the order is the order on the page.
FOLDERS: dict[str, tuple[str, str]] = {
    "projects": (
        "项目",
        "每个项目一个文件夹：源音频副本、人声分离结果、配音和导出的成品。"
        "不再需要的项目可以整个文件夹删除；下面的“清理缓存”只删能重新生成的中间文件。",
    ),
    "models": (
        "模型",
        "识别、配音和人声分离模型，含下载缓存和 IndexTTS 的模型文件。不用的模型请在“模型”页删除。",
    ),
    "runtimes": (
        "后端运行环境",
        "IndexTTS、人声分离、MOSS 等后端各自的 Python 环境（含 PyTorch），每个约 3–9 GB。"
        "请不要手动删除；在“模型”页删除 IndexTTS 时会连同它的环境一起删除。",
    ),
    "environment": (
        "程序主环境",
        "程序本身和识别模型使用的 Python 依赖（含 PyTorch）。请不要删除。",
    ),
    "cache": ("缓存", "临时文件和批量工作区，可以用下面的“清理缓存”安全清理。"),
    "model_packs": (
        "离线模型包",
        "手动放进来的离线模型包。导入完成后，模型已复制到程序里，这些压缩包可以删除。",
    ),
}


def folders(projects_root: str | None) -> dict[str, Path]:
    home = portable_home()
    paths = {
        "projects": Path(projects_root) if projects_root else home / "projects",
        "models": home / "models",
        "runtimes": home / "runtimes",
        "cache": home / "cache",
        "model_packs": model_pack_directory(),
    }
    # The running environment: <home>/venv in a portable install, .venv from source.
    if sys.prefix != sys.base_prefix:
        paths["environment"] = Path(sys.prefix)
    return {key: paths[key] for key in FOLDERS if key in paths}


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def report(settings: Any) -> dict[str, Any]:
    home = portable_home()
    paths = folders(settings.projects_root)
    model_sizes = models.model_storage(settings)
    model_roots: list[Path] = model_sizes["roots"]
    sizes: dict[str, int] = {}
    for key, path in paths.items():
        if key == "models":
            # Includes the download caches and IndexTTS checkpoints kept elsewhere.
            sizes[key] = model_sizes["total_bytes"]
            continue
        size = models.directory_bytes(path)
        # Model files living inside another folder are already counted as models.
        size -= sum(
            models.directory_bytes(root)
            for root in model_roots
            if _inside(root, path) and not _inside(path, root)
        )
        sizes[key] = max(0, size)
    temp = home / "temp"
    diagnostics = (
        sum(item.stat().st_size for item in temp.glob("ASMR-Dubber-diagnostics-*.zip"))
        if temp.is_dir()
        else 0
    )
    sizes["cache"] = (
        sizes.get("cache", 0) + models.directory_bytes(home / "autoflow" / ".work") + diagnostics
    )
    locations = [
        {
            "key": key,
            "label": FOLDERS[key][0],
            "hint": FOLDERS[key][1],
            "path": str(path),
            "bytes": sizes[key],
            "exists": path.is_dir(),
        }
        for key, path in paths.items()
        if key != "model_packs" or sizes[key] > 0
    ]
    return {"locations": locations, "projects_root": str(paths["projects"])}


def open_folder(key: str, projects_root: str | None) -> dict[str, str]:
    path = folders(projects_root).get(key)
    if path is None:
        raise ValueError("未知的存储位置。")
    if not path.is_dir():
        raise ValueError("这个文件夹还不存在。")
    return {"path": str(open_directory(path))}
