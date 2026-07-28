from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
PACK_ID = "windows-advanced-dependencies"
PACK_VERSION = "1.0.0"
MANIFEST_NAME = "dependency-pack.json"
PAYLOAD_PREFIX = "payload/"
ALLOWED_PREFIX = "payload/venv/"
MAX_FILES = 120_000
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
OPTIONAL_PAYLOAD_PREFIXES = ("payload/venv/Lib/site-packages/pkg_resources/tests/",)
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


def _windows_io_path(path: Path) -> Path:
    """Use Win32's extended path namespace without changing the physical location."""

    if os.name != "nt":
        return path
    value = os.path.abspath(os.fspath(path))
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def _remove_tree(path: Path, *, ignore_errors: bool = False) -> None:
    shutil.rmtree(_windows_io_path(path), ignore_errors=ignore_errors)


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


def _validate(handle: zipfile.ZipFile) -> dict[str, object]:
    try:
        manifest = json.loads(handle.read(MANIFEST_NAME).decode("utf-8"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise DependencyPackError(f"invalid {MANIFEST_NAME}: {exc}") from exc
    expected = {
        "schema_version",
        "pack_id",
        "pack_version",
        "platform",
        "architecture",
        "python",
        "components",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise DependencyPackError("dependency pack manifest fields are invalid")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["pack_id"] != PACK_ID:
        raise DependencyPackError("unsupported dependency pack type or schema")
    if manifest["pack_version"] != PACK_VERSION:
        raise DependencyPackError("dependency pack version does not match")
    if manifest["platform"] != "windows" or manifest["architecture"] != "x86_64":
        raise DependencyPackError("dependency pack platform does not match")
    if manifest["python"] != "3.12":
        raise DependencyPackError("dependency pack Python version does not match")
    if manifest["components"] != [
        "application-ui",
        "advanced-asr",
        "qwen-forced-aligner",
        "asmr-vad",
        "pytorch-cu130",
    ]:
        raise DependencyPackError("dependency pack components do not match the pinned artifact")

    infos = handle.infolist()
    if len(infos) > MAX_FILES:
        raise DependencyPackError("dependency pack contains too many files")
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
        if info.is_dir() or str(path) in {MANIFEST_NAME, "THIRD_PARTY_NOTICES.txt"}:
            continue
        if not info.filename.startswith(ALLOWED_PREFIX):
            raise DependencyPackError(f"unexpected dependency pack path: {info.filename}")
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise DependencyPackError("dependency pack exceeds the size limit")
    return manifest


def _extract(handle: zipfile.ZipFile, staging: Path) -> None:
    for info in handle.infolist():
        if info.is_dir() or not info.filename.startswith(PAYLOAD_PREFIX):
            continue
        if any(info.filename.startswith(prefix) for prefix in OPTIONAL_PAYLOAD_PREFIXES):
            continue
        relative = PurePosixPath(info.filename).relative_to(PAYLOAD_PREFIX.rstrip("/"))
        destination = staging.joinpath(*relative.parts)
        io_destination = _windows_io_path(destination)
        io_destination.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(info, "r") as source, io_destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)


def _validate_staging(staging: Path) -> None:
    required = (
        "venv/Scripts/python.exe",
        "venv/Lib/site-packages/gradio",
        "venv/Lib/site-packages/torch",
        "venv/Lib/site-packages/torchaudio",
        "venv/Lib/site-packages/transformers",
        "venv/Lib/site-packages/faster_whisper",
    )
    for relative in required:
        if not (staging / relative).exists():
            raise DependencyPackError(f"dependency pack is missing {relative}")


def _merge_file(source: Path, destination: Path) -> None:
    io_source = _windows_io_path(source)
    io_destination = _windows_io_path(destination)
    io_destination.parent.mkdir(parents=True, exist_ok=True)
    if io_destination.is_file() and filecmp.cmp(io_source, io_destination, shallow=False):
        return
    temporary = io_destination.with_name(f".{io_destination.name}.advanced-dependency-pack-new")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(io_source, temporary)
        os.replace(temporary, io_destination)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_venv(source: Path, destination: Path) -> None:
    """Overlay a prepared venv without renaming the running WebUI environment."""

    for current, directories, filenames in os.walk(_windows_io_path(source)):
        current_path = Path(current)
        relative = current_path.relative_to(_windows_io_path(source))
        target_directory = _windows_io_path(destination / relative)
        target_directory.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_directory / directory).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            _merge_file(current_path / filename, destination / relative / filename)


def import_pack(
    archive: Path,
    portable: Path,
    sha256: str,
    *,
    merge_existing: bool = False,
) -> None:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise DependencyPackError("dependency pack SHA-256 is invalid")
    staging_parent = portable / "t"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="a-", dir=staging_parent))
    destination = portable / "venv"
    backup = portable / ".venv.advanced-dependency-pack-backup"
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as handle:
            manifest = _validate(handle)
            _extract(handle, staging)
        _validate_staging(staging)
        if merge_existing and destination.exists():
            _merge_venv(staging / "venv", destination)
        else:
            _remove_tree(backup, ignore_errors=True)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(staging / "venv", destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
        marker = portable / "runtimes/windows-advanced-dependencies.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
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
        _remove_tree(backup, ignore_errors=True)
    finally:
        _remove_tree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("portable", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    import_pack(
        args.archive.resolve(),
        args.portable.resolve(),
        args.sha256.lower(),
        merge_existing=args.merge_existing,
    )


if __name__ == "__main__":
    main()
