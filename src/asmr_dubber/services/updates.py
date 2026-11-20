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
from ..mirrors import github_url_candidates
from ..platforms import portable_home
from .settings import current

REPOSITORY = "EveningStudy/asmr-dubber"
RELEASE_PAGE = f"https://github.com/{REPOSITORY}/releases/latest"
RELEASE_API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
DOWNLOAD_PREFIX = f"https://github.com/{REPOSITORY}/releases/download/"
ASSET_NAME = re.compile(r"ASMR-Dubber-windows-portable-v\d+\.\d+\.\d+\.zip")
REQUIRED_FILES = ("ASMR-Dubber.exe", "pyproject.toml", "src/asmr_dubber/__init__.py")
DATA_DIRECTORY = ".asmr-dubber"
CACHE_SECONDS = 3600
WINDOWS = os.name == "nt"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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
        for retired in root.glob("ASMR-Dubber.exe.old*"):
            with contextlib.suppress(OSError):
                retired.unlink(missing_ok=True)
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
            raise ValueError("无法识别的版本号。")
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
        raw_digest = asset.get("digest")
        sha256 = ""
        digest_malformed = False
        if raw_digest is not None and str(raw_digest).strip():
            text = str(raw_digest).strip()
            if text.lower().startswith("sha256:"):
                text = text[7:].strip()
            if _SHA256_RE.fullmatch(text.lower()):
                sha256 = text.lower()
            else:
                digest_malformed = True
        sidecar_name = f"{asset['name']}.sha256"
        sidecar = next(
            (
                item
                for item in release.get("assets", [])
                if item.get("name") == sidecar_name
                and str(item.get("browser_download_url", "")).startswith(DOWNLOAD_PREFIX)
            ),
            None,
        )
        result["asset"] = {
            "name": asset["name"],
            "url": asset["browser_download_url"],
            "size": int(asset.get("size") or 0),
            "sha256": sha256,
            "digest_malformed": digest_malformed,
            "sha256_url": sidecar["browser_download_url"] if sidecar else "",
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


def _download_candidates(url):
    try:
        source = current().download_source
    except Exception:
        source = "modelscope"
    if source == "modelscope":
        candidates = github_url_candidates(url, allow_external=True)
        if candidates:
            return candidates
    return (url,)


def _resolve_expected_sha256(asset):
    if asset.get("digest_malformed"):
        raise ValueError("更新包的 SHA-256 摘要格式无效。")
    raw_sha256 = str(asset.get("sha256") or "").strip().lower()
    if raw_sha256:
        if not _SHA256_RE.fullmatch(raw_sha256):
            raise ValueError("更新包的 SHA-256 摘要格式无效。")
        return raw_sha256
    sidecar_url = str(asset.get("sha256_url") or "").strip()
    if not sidecar_url:
        return ""
    # The checksum is what makes a proxy download trustworthy, so it is only
    # read from GitHub itself; a proxy could otherwise serve a matching pair.
    try:
        response = httpx.get(sidecar_url, timeout=15, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        raise ValueError("无法下载更新包的 SHA-256 校验文件，请重试。") from exc
    match = re.match(r"^([0-9a-fA-F]{64})(?:\s|$)", response.text.strip())
    if not match:
        raise ValueError("更新包的 SHA-256 校验文件格式无效。")
    return match.group(1).lower()


def _download(asset, archive, report, token):
    expected_sha256 = _resolve_expected_sha256(asset)
    last_error = None
    # Third-party GitHub proxies are only used when the archive can be verified.
    candidates = _download_candidates(asset["url"]) if expected_sha256 else (asset["url"],)
    for candidate in candidates:
        digest = hashlib.sha256()
        done = 0
        try:
            with httpx.stream("GET", candidate, timeout=30, follow_redirects=True) as response:
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
            if expected_sha256 and digest.hexdigest() != expected_sha256:
                raise ValueError("下载的更新包校验失败，请重试。")
            return
        except OperationCancelledError:
            raise
        except Exception as exc:
            last_error = exc
            archive.unlink(missing_ok=True)
    if last_error is not None:
        raise last_error


def _members(archive):
    """Program files in the release archive, keyed by path relative to the program folder."""
    all_infos = archive.infolist()
    for info in all_infos:
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("更新包里有不安全的符号链接。")
    names = [info for info in all_infos if not info.is_dir()]
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
    except PermissionError as exc:
        # A running executable cannot be overwritten on Windows, but it can be renamed.
        retired = target.with_name(f"{target.name}.old")
        try:
            retired.unlink(missing_ok=True)
            os.replace(target, retired)
            os.replace(source, target)
            return
        except PermissionError:
            pass
        unique_retired = target.with_name(f"{target.name}.old.{os.getpid()}.{time.monotonic_ns()}")
        try:
            os.replace(target, unique_retired)
            os.replace(source, target)
        except PermissionError as inner_exc:
            raise ValueError(
                "无法替换正在运行的程序文件，请先关闭所有 ASMR Dubber 窗口后重试。"
            ) from (inner_exc or exc)
