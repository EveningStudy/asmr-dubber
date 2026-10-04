"""Release check and in-place update of the portable program files."""

import contextlib
import hashlib
import os
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import httpx

from .. import __version__
from ..errors import OperationCancelledError
from ..platforms import portable_home

REPOSITORY = "EveningStudy/asmr-dubber"
RELEASE_PAGE = f"https://github.com/{REPOSITORY}/releases/latest"
RELEASE_API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
DOWNLOAD_PREFIX = f"https://github.com/{REPOSITORY}/releases/download/"
ASSET_NAME = re.compile(r"ASMR-Dubber-windows-portable-v\d+\.\d+\.\d+\.zip")
REQUIRED_FILES = ("ASMR-Dubber.exe", "pyproject.toml", "src/asmr_dubber/__init__.py")
DATA_DIRECTORY = ".asmr-dubber"
CACHE_SECONDS = 3600
WINDOWS = os.name == "nt"

_cached = (0.0, None)


def version_tuple(text):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", str(text).strip())
    return tuple(int(part) for part in match.groups()) if match else None


def installable_root():
    """The portable program folder, or None when files must not be replaced automatically."""
    home = portable_home()
    root = home.parent
    if (
        WINDOWS
        and home.name == DATA_DIRECTORY
        and (root / "ASMR-Dubber.exe").is_file()
        # A source checkout is updated with git, never by overwriting the working tree.
        and not (root / ".git").exists()
    ):
        return root
    return None


def check(force=False):
    global _cached
    if not force and _cached[1] is not None and time.monotonic() - _cached[0] < CACHE_SECONDS:
        return _cached[1]
    root = installable_root()
    if root is not None:
        with contextlib.suppress(OSError):
            (root / "ASMR-Dubber.exe.old").unlink(missing_ok=True)
    result = {
        "current": __version__,
        "latest": None,
        "newer": False,
        "automatic": False,
        "page": RELEASE_PAGE,
        "error": None,
    }
    try:
        response = httpx.get(
            RELEASE_API,
            timeout=8,
            follow_redirects=True,
            headers={"Accept": "application/vnd.github+json"},
        )
        response.raise_for_status()
        release = response.json()
        latest = version_tuple(release["tag_name"])
        if latest is None:
            raise ValueError("Unrecognised release tag")
    except Exception as exc:
        result["error"] = type(exc).__name__
        return result
    asset = next(
        (
            item
            for item in release.get("assets", [])
            if ASSET_NAME.fullmatch(item.get("name", ""))
            and str(item.get("browser_download_url", "")).startswith(DOWNLOAD_PREFIX)
        ),
        None,
    )
    result.update(
        latest=".".join(str(part) for part in latest),
        newer=latest > (version_tuple(__version__) or ()),
        page=release.get("html_url") or RELEASE_PAGE,
    )
    if asset:
        result["asset"] = {
            "name": asset["name"],
            "url": asset["browser_download_url"],
            "size": int(asset.get("size") or 0),
            "sha256": str(asset.get("digest") or "").removeprefix("sha256:"),
        }
        result["automatic"] = result["newer"] and installable_root() is not None
    _cached = (time.monotonic(), result)
    return result


def install(request, report, token):
    release = check(force=True)
    root = installable_root()
    if not release["newer"] or not release["automatic"] or root is None:
        raise ValueError("没有可以自动安装的新版本。")
    asset = release["asset"]
    workspace = portable_home() / "temp" / "update"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    archive = workspace / asset["name"]
    try:
        _download(asset, archive, report, token)
        report("正在安装新版本…", 1, 1)
        apply(archive, root, workspace / "staged")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return {
        "message": "已更新到 {version}。请关闭启动窗口，再重新运行 ASMR-Dubber.exe。".replace(
            "{version}", release["latest"]
        ),
        "version": release["latest"],
    }


def _download(asset, archive, report, token):
    digest = hashlib.sha256()
    done = 0
    with httpx.stream("GET", asset["url"], timeout=30, follow_redirects=True) as response:
        response.raise_for_status()
        total = asset["size"] or int(response.headers.get("Content-Length", 0))
        with archive.open("wb") as output:
            for block in response.iter_bytes(1024 * 1024):
                if token.is_set():
                    raise OperationCancelledError("更新已取消。")
                output.write(block)
                digest.update(block)
                done += len(block)
                report(f"正在下载新版本… {done / 1048576:.0f} MB", done, total)
    if asset["size"] and done != asset["size"]:
        raise ValueError("下载的更新包大小不对，请重试。")
    if asset["sha256"] and digest.hexdigest() != asset["sha256"]:
        raise ValueError("下载的更新包校验失败，请重试。")


def _members(archive):
    """Program files in the release archive, keyed by path relative to the program folder."""
    names = [info for info in archive.infolist() if not info.is_dir()]
    paths = {info: PurePosixPath(info.filename.replace("\\", "/")) for info in names}
    tops = {path.parts[0] for path in paths.values() if path.parts}
    prefix = 1 if len(tops) == 1 and all(len(path.parts) > 1 for path in paths.values()) else 0
    members = {}
    for info, path in paths.items():
        parts = path.parts[prefix:]
        if not parts or path.is_absolute() or ".." in parts or ":" in "".join(parts):
            raise ValueError("更新包里有不安全的路径。")
        if parts[0] == DATA_DIRECTORY:
            continue
        members["/".join(parts)] = info
    return members


def apply(archive_path, root, staging):
    """Replace program files from a release archive. User data under .asmr-dubber is untouched."""
    root = Path(root).resolve()
    staging = Path(staging)
    with zipfile.ZipFile(archive_path) as archive:
        members = _members(archive)
        if any(name not in members for name in REQUIRED_FILES):
            raise ValueError("更新包不完整。")
        # Extract everything before touching the installation so a bad archive changes nothing.
        for name, info in members.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    for name in members:
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError("更新包里有不安全的路径。")
        target.parent.mkdir(parents=True, exist_ok=True)
        _replace(staging / name, target)
    # Modules removed in the new version must not linger next to the new ones.
    source_root = root / "src"
    for file in source_root.rglob("*"):
        stale = file.relative_to(root).as_posix() not in members
        if file.is_file() and "__pycache__" not in file.parts and stale:
            file.unlink()
    for cache in source_root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _replace(source, target):
    try:
        os.replace(source, target)
    except PermissionError:
        # A running executable cannot be overwritten on Windows, but it can be renamed.
        retired = target.with_name(target.name + ".old")
        retired.unlink(missing_ok=True)
        os.replace(target, retired)
        os.replace(source, target)
