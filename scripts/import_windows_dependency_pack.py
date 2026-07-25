from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
PACK_ID = "windows-recommended-dependencies"
PACK_VERSION = "1.0.0"
MANIFEST_NAME = "dependency-pack.json"
PAYLOAD_PREFIX = "payload/"
MAX_FILES = 200_000
MAX_UNCOMPRESSED_BYTES = 24 * 1024**3
ALLOWED_PREFIXES = (
    "payload/venv/",
    "payload/runtimes/python/cpython-3.11.13-windows-x86_64-none/",
    "payload/runtimes/index-tts/.venv/",
    "payload/runtimes/ffmpeg-shared/",
)
WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class DependencyPackError(RuntimeError):
    pass


def _safe_member(name: str, *, directory: bool) -> PurePosixPath:
    trimmed = name[:-1] if directory and name.endswith("/") else name
    if not trimmed or "\\" in trimmed or "\0" in trimmed or ":" in trimmed:
        raise DependencyPackError(f"unsafe ZIP path: {name!r}")
    if trimmed != unicodedata.normalize("NFC", trimmed):
        raise DependencyPackError(f"ZIP path is not NFC Unicode: {name!r}")
    path = PurePosixPath(trimmed)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DependencyPackError(f"unsafe ZIP path: {name!r}")
    for part in path.parts:
        if part.rstrip(" .") != part or part.split(".", 1)[0].casefold() in WINDOWS_RESERVED:
            raise DependencyPackError(f"Windows-incompatible ZIP path: {name!r}")
    return path


def _validate_manifest(handle: zipfile.ZipFile) -> dict[str, object]:
    try:
        raw = handle.read(MANIFEST_NAME)
        manifest = json.loads(raw.decode("utf-8"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise DependencyPackError(f"invalid {MANIFEST_NAME}: {exc}") from exc
    expected = {
        "schema_version",
        "pack_id",
        "pack_version",
        "platform",
        "architecture",
        "python",
        "indextts_revision",
        "components",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise DependencyPackError("dependency pack manifest fields are invalid")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["pack_id"] != PACK_ID:
        raise DependencyPackError("unsupported dependency pack type or schema")
    if manifest["pack_version"] != PACK_VERSION:
        raise DependencyPackError("dependency pack version does not match")
    if manifest["platform"] != "windows" or manifest["architecture"] != "x86_64":
        raise DependencyPackError("dependency pack is not for Windows x86_64")
    if manifest["python"] != ["3.12", "3.11.13"]:
        raise DependencyPackError("dependency pack Python versions do not match")
    if manifest["indextts_revision"] != "13495845e3028f0bb6ca1462ad22aa0e76349e40":
        raise DependencyPackError("dependency pack IndexTTS2 revision does not match")
    if manifest["components"] != ["application-ui", "indextts2", "ffmpeg-shared"]:
        raise DependencyPackError("dependency pack components do not match")
    return manifest


def _validate_members(handle: zipfile.ZipFile) -> None:
    infos = handle.infolist()
    if len(infos) > MAX_FILES:
        raise DependencyPackError(f"dependency pack contains more than {MAX_FILES} files")
    total = 0
    seen: set[str] = set()
    for info in infos:
        path = _safe_member(info.filename, directory=info.is_dir())
        folded = str(path).casefold()
        if folded in seen:
            raise DependencyPackError(f"duplicate ZIP path: {info.filename}")
        seen.add(folded)
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise DependencyPackError(f"symbolic links are not allowed: {info.filename}")
        if info.is_dir():
            continue
        if str(path) in {MANIFEST_NAME, "THIRD_PARTY_NOTICES.txt"}:
            continue
        if not any(info.filename.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            raise DependencyPackError(f"unexpected dependency pack path: {info.filename}")
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise DependencyPackError("dependency pack exceeds the uncompressed size limit")


def _extract(handle: zipfile.ZipFile, staging: Path) -> None:
    for info in handle.infolist():
        if info.is_dir() or not info.filename.startswith(PAYLOAD_PREFIX):
            continue
        relative = PurePosixPath(info.filename).relative_to(PAYLOAD_PREFIX.rstrip("/"))
        destination = staging.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(info, "r") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)


def _validate_staging(staging: Path) -> None:
    required = (
        "venv/Scripts/python.exe",
        "venv/Scripts/asmr-dubber.exe",
        "venv/Lib/site-packages/gradio",
        "runtimes/python/cpython-3.11.13-windows-x86_64-none/python.exe",
        "runtimes/index-tts/.venv/Scripts/python.exe",
        "runtimes/index-tts/.venv/Lib/site-packages/torch",
    )
    for relative in required:
        if not (staging / relative).exists():
            raise DependencyPackError(f"dependency pack is missing {relative}")
    ffmpeg = list((staging / "runtimes/ffmpeg-shared").rglob("ffmpeg.exe"))
    codecs = list((staging / "runtimes/ffmpeg-shared").rglob("avcodec-*.dll"))
    if not ffmpeg or not codecs:
        raise DependencyPackError("dependency pack is missing shared FFmpeg")


def _install_component(
    staging: Path,
    portable: Path,
    relative: str,
    backups: list[tuple[Path, Path]],
) -> None:
    source = staging / relative
    destination = portable / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.dependency-pack-backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
        backups.append((destination, backup))
    os.replace(source, destination)


def import_pack(archive: Path, portable: Path, sha256: str) -> None:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise DependencyPackError("dependency pack SHA-256 is invalid")
    portable.mkdir(parents=True, exist_ok=True)
    temp_root = portable / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="recommended-dependencies-", dir=temp_root))
    backups: list[tuple[Path, Path]] = []
    installed: list[str] = []
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as handle:
            manifest = _validate_manifest(handle)
            _validate_members(handle)
            _extract(handle, staging)
        _validate_staging(staging)
        components = (
            "venv",
            "runtimes/python/cpython-3.11.13-windows-x86_64-none",
            "runtimes/index-tts/.venv",
            "runtimes/ffmpeg-shared",
        )
        for relative in components:
            _install_component(staging, portable, relative, backups)
            installed.append(relative)
        marker = portable / "runtimes/windows-recommended-dependencies.json"
        marker.write_text(
            json.dumps(
                {
                    "pack_id": PACK_ID,
                    "pack_version": manifest["pack_version"],
                    "sha256": sha256,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        for relative in reversed(installed):
            destination = portable / relative
            if destination.exists():
                shutil.rmtree(destination)
        for original, backup in reversed(backups):
            if backup.exists() and not original.exists():
                os.replace(backup, original)
        raise
    else:
        for _original, backup in backups:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("portable", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    import_pack(args.archive.resolve(), args.portable.resolve(), args.sha256.lower())


if __name__ == "__main__":
    main()
