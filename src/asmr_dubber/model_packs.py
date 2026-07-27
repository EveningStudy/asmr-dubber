from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import PROJECT_ROOT
from .errors import AsmrDubberError
from .platforms import current_platform, portable_home

MODEL_PACK_SCHEMA_VERSION = 1
MODEL_PACK_MANIFEST = "model-pack.json"
MODEL_PACK_PAYLOAD_PREFIX = "payload/"
MODEL_PACK_DIRECTORY_NAME = "model-packs"
WINDOWS_DEPENDENCY_PACK_PREFIXES = ("asmr-dubber-windows-recommended-dependencies-v",)
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_FILE_COUNT = 100_000
MAX_UNCOMPRESSED_BYTES = 100 * 1024**3

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int], None]

_PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_STORED_SUFFIXES = frozenset(
    {
        ".bin",
        ".gguf",
        ".model",
        ".pt",
        ".pth",
        ".safetensors",
    }
)


class ModelPackError(AsmrDubberError):
    """An offline model pack is malformed, incompatible, or incomplete."""


@dataclass(frozen=True)
class ModelPackFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelPackManifest:
    pack_id: str
    display_name: str
    pack_version: str
    platforms: tuple[str, ...]
    architectures: tuple[str, ...]
    files: tuple[ModelPackFile, ...]

    @property
    def uncompressed_bytes(self) -> int:
        return sum(file.size for file in self.files)


@dataclass(frozen=True)
class ModelPackInspection:
    archive: Path
    manifest: ModelPackManifest | None
    compatible: bool
    error: str = ""


@dataclass(frozen=True)
class ModelPackImportResult:
    archive: Path
    pack_id: str
    installed_files: int
    reused_files: int
    uncompressed_bytes: int
    already_installed: bool


@dataclass(frozen=True)
class ModelPackSource:
    source: Path
    target: str


def model_pack_directory(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / MODEL_PACK_DIRECTORY_NAME


def imported_hf_snapshot_path(
    model_id: str,
    revision: str,
    *,
    home: Path | None = None,
) -> Path:
    """Return the project-owned path used for an imported Hugging Face snapshot."""
    if not model_id or "/" not in model_id:
        raise ValueError("model_id must use the owner/repository form")
    repo_folder = "models--" + model_id.replace("/", "--")
    return (
        (home or portable_home()) / "models" / "huggingface" / repo_folder / "snapshots" / revision
    )


def _json_without_duplicate_keys(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ModelPackError(f"模型包 manifest 包含重复字段：{key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackError(f"模型包 manifest 不是有效的 UTF-8 JSON：{exc}") from exc


def _normalize_platform(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "win32": "windows",
        "win64": "windows",
        "linux2": "linux",
    }
    return aliases.get(normalized, normalized)


def _normalize_architecture(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    return aliases.get(normalized, normalized)


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ModelPackError("模型包文件路径不能为空。")
    if value != unicodedata.normalize("NFC", value):
        raise ModelPackError(f"模型包文件路径必须使用 NFC Unicode：{value!r}")
    if "\\" in value or "\x00" in value or ":" in value:
        raise ModelPackError(f"模型包文件路径不安全：{value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelPackError(f"模型包文件路径不安全：{value!r}")
    if not path.parts or path.parts[0] not in {"models", "runtimes"}:
        raise ModelPackError(f"模型包只能写入 models/ 或 runtimes/：{value!r}")
    for part in path.parts:
        if part.rstrip(" .") != part:
            raise ModelPackError(f"模型包文件路径包含 Windows 不兼容名称：{value!r}")
        stem = part.split(".", 1)[0].casefold()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise ModelPackError(f"模型包文件路径包含 Windows 保留名称：{value!r}")
    return path


def _manifest_from_mapping(value: Any) -> ModelPackManifest:
    if not isinstance(value, dict):
        raise ModelPackError("模型包 manifest 顶层必须是 JSON 对象。")
    allowed = {
        "schema_version",
        "pack_id",
        "display_name",
        "pack_version",
        "platforms",
        "architectures",
        "files",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ModelPackError("模型包 manifest 包含未知字段：" + "、".join(unknown))
    if value.get("schema_version") != MODEL_PACK_SCHEMA_VERSION:
        raise ModelPackError(f"不支持的模型包格式版本：{value.get('schema_version')!r}")
    pack_id = value.get("pack_id")
    if not isinstance(pack_id, str) or not _PACK_ID_RE.fullmatch(pack_id):
        raise ModelPackError("pack_id 只能包含小写字母、数字、点、下划线和连字符。")
    display_name = value.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ModelPackError("display_name 不能为空。")
    pack_version = value.get("pack_version")
    if not isinstance(pack_version, str) or not pack_version.strip():
        raise ModelPackError("pack_version 不能为空。")
    platforms = value.get("platforms")
    architectures = value.get("architectures")
    if (
        not isinstance(platforms, list)
        or not platforms
        or not all(isinstance(item, str) and item.strip() for item in platforms)
    ):
        raise ModelPackError("platforms 必须是非空字符串数组。")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(item, str) and item.strip() for item in architectures)
    ):
        raise ModelPackError("architectures 必须是非空字符串数组。")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ModelPackError("files 必须是非空数组。")
    if len(files) > MAX_FILE_COUNT:
        raise ModelPackError(f"模型包文件数超过上限 {MAX_FILE_COUNT}。")
    parsed_files: list[ModelPackFile] = []
    seen_paths: set[str] = set()
    total_size = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ModelPackError("每个 files 项必须且只能包含 path、size、sha256。")
        raw_path = item.get("path")
        if not isinstance(raw_path, str):
            raise ModelPackError("模型包文件路径必须是字符串。")
        path = str(_safe_relative_path(raw_path))
        folded = path.casefold()
        if folded in seen_paths:
            raise ModelPackError(f"模型包包含重复文件路径：{path}")
        seen_paths.add(folded)
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ModelPackError(f"模型包文件大小无效：{path}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
            raise ModelPackError(f"模型包文件 SHA-256 无效：{path}")
        total_size += size
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ModelPackError(
                f"模型包解压后超过安全上限 {MAX_UNCOMPRESSED_BYTES / 1024**3:g} GiB。"
            )
        parsed_files.append(ModelPackFile(path=path, size=size, sha256=digest.lower()))
    return ModelPackManifest(
        pack_id=pack_id,
        display_name=display_name.strip(),
        pack_version=pack_version.strip(),
        platforms=tuple(_normalize_platform(item) for item in platforms),
        architectures=tuple(_normalize_architecture(item) for item in architectures),
        files=tuple(parsed_files),
    )


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _validate_zip_member_name(name: str, *, directory: bool) -> None:
    trimmed = name[:-1] if directory and name.endswith("/") else name
    if not trimmed or "\\" in trimmed or "\x00" in trimmed or ":" in trimmed:
        raise ModelPackError(f"模型包 ZIP 路径不安全：{name!r}")
    path = PurePosixPath(trimmed)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelPackError(f"模型包 ZIP 路径不安全：{name!r}")
    if directory and path.parts[0] != MODEL_PACK_PAYLOAD_PREFIX.rstrip("/"):
        raise ModelPackError(f"模型包包含无关目录：{name}")


def _archive_manifest(archive: Path) -> tuple[ModelPackManifest, dict[str, zipfile.ZipInfo]]:
    try:
        handle = zipfile.ZipFile(archive, "r", allowZip64=True)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ModelPackError(f"无法打开模型包 {archive.name}：{exc}") from exc
    with handle:
        infos = handle.infolist()
        if len(infos) > MAX_FILE_COUNT + 10_000:
            raise ModelPackError("模型包 ZIP 项目数量异常。")
        by_name: dict[str, zipfile.ZipInfo] = {}
        seen_names: set[str] = set()
        for info in infos:
            name = unicodedata.normalize("NFC", info.filename)
            if name != info.filename:
                raise ModelPackError(f"模型包 ZIP 路径不安全：{info.filename!r}")
            _validate_zip_member_name(name, directory=info.is_dir())
            folded = name.casefold()
            if folded in seen_names:
                raise ModelPackError(f"模型包 ZIP 包含重复路径：{name}")
            seen_names.add(folded)
            if info.flag_bits & 0x1:
                raise ModelPackError(f"模型包不允许加密文件：{name}")
            if _zip_member_is_symlink(info):
                raise ModelPackError(f"模型包不允许符号链接：{name}")
            by_name[name] = info
        manifest_info = by_name.get(MODEL_PACK_MANIFEST)
        if manifest_info is None or manifest_info.is_dir():
            raise ModelPackError(f"模型包缺少 {MODEL_PACK_MANIFEST}。")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ModelPackError("模型包 manifest 过大。")
        manifest = _manifest_from_mapping(_json_without_duplicate_keys(handle.read(manifest_info)))
        expected_members = {MODEL_PACK_PAYLOAD_PREFIX + file.path: file for file in manifest.files}
        actual_members: dict[str, zipfile.ZipInfo] = {}
        for name, info in by_name.items():
            if name == MODEL_PACK_MANIFEST or info.is_dir():
                continue
            if name not in expected_members:
                raise ModelPackError(f"模型包包含 manifest 未声明的文件：{name}")
            declared = expected_members[name]
            if info.file_size != declared.size:
                raise ModelPackError(f"模型包文件大小与 manifest 不符：{declared.path}")
            actual_members[name] = info
        missing = sorted(set(expected_members) - set(actual_members))
        if missing:
            raise ModelPackError("模型包缺少 manifest 声明的文件：" + "、".join(missing[:5]))
        return manifest, actual_members


def is_manifest_compatible(manifest: ModelPackManifest) -> bool:
    platform_id = _normalize_platform(current_platform().id)
    architecture = _normalize_architecture(platform.machine())
    return ("any" in manifest.platforms or platform_id in manifest.platforms) and (
        "any" in manifest.architectures or architecture in manifest.architectures
    )


def inspect_model_pack(archive: Path) -> ModelPackInspection:
    archive = archive.expanduser().resolve()
    try:
        manifest, _ = _archive_manifest(archive)
        compatible = is_manifest_compatible(manifest)
        error = "" if compatible else "当前操作系统或 CPU 架构不兼容"
        return ModelPackInspection(archive, manifest, compatible, error)
    except (ModelPackError, OSError) as exc:
        return ModelPackInspection(archive, None, False, str(exc))


def discover_model_packs(directory: Path | None = None) -> list[ModelPackInspection]:
    root = (directory or model_pack_directory()).expanduser().resolve()
    if not root.is_dir():
        return []
    return [
        inspect_model_pack(archive)
        for archive in sorted(root.glob("*.zip"), key=lambda path: path.name.casefold())
        if archive.is_file()
        # This archive shares the user-facing inbox so Setup can install it
        # without another download, but it has a separate manifest and importer.
        # Do not misreport it as a corrupt offline model pack.
        and not archive.name.casefold().startswith(WINDOWS_DEPENDENCY_PACK_PREFIXES)
    ]


def _manifest_payload(manifest: ModelPackManifest) -> dict[str, Any]:
    return {
        "schema_version": MODEL_PACK_SCHEMA_VERSION,
        "pack_id": manifest.pack_id,
        "display_name": manifest.display_name,
        "pack_version": manifest.pack_version,
        "platforms": list(manifest.platforms),
        "architectures": list(manifest.architectures),
        "files": [
            {"path": file.path, "size": file.size, "sha256": file.sha256} for file in manifest.files
        ],
    }


def _manifest_digest(manifest: ModelPackManifest) -> str:
    encoded = json.dumps(
        _manifest_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _temporary_model_path(destination: Path) -> Path:
    """Use a short, deterministic leaf name to stay below legacy Win32 limits."""

    token = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:12]
    return destination.parent / f".mp-{os.getpid()}-{token}.tmp"


def _target_path(home: Path, relative: str) -> Path:
    safe = _safe_relative_path(relative)
    resolved_home = home.resolve()
    target = home.joinpath(*safe.parts)
    try:
        resolved_target = target.resolve(strict=False)
        resolved_target.relative_to(resolved_home)
    except (OSError, ValueError) as exc:
        raise ModelPackError(f"模型包目标路径逃逸便携目录：{relative}") from exc
    current = home
    for part in safe.parts:
        current /= part
        if current.is_symlink():
            raise ModelPackError(f"模型包目标路径包含符号链接：{relative}")
    return target


def _receipt_path(home: Path, manifest: ModelPackManifest) -> Path:
    return (
        home
        / MODEL_PACK_DIRECTORY_NAME
        / "installed"
        / f"{manifest.pack_id}-{_manifest_digest(manifest)}.json"
    )


def _receipt_is_current(
    receipt: Path,
    manifest: ModelPackManifest,
    *,
    home: Path,
) -> bool:
    if not receipt.is_file():
        return False
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        recorded = payload["files"]
        if payload["manifest_sha256"] != _manifest_digest(manifest):
            return False
        if not isinstance(recorded, dict) or len(recorded) != len(manifest.files):
            return False
        for file in manifest.files:
            target = _target_path(home, file.path)
            stat_result = target.stat()
            if not target.is_file() or stat_result.st_size != file.size:
                return False
            entry = recorded.get(file.path)
            if not isinstance(entry, dict):
                return False
            if (
                entry.get("size") != stat_result.st_size
                or entry.get("mtime_ns") != stat_result.st_mtime_ns
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_receipt(
    receipt: Path,
    manifest: ModelPackManifest,
    *,
    home: Path,
) -> None:
    files: dict[str, dict[str, int]] = {}
    for file in manifest.files:
        target = _target_path(home, file.path)
        stat_result = target.stat()
        files[file.path] = {
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }
    payload = {
        "schema_version": MODEL_PACK_SCHEMA_VERSION,
        "pack_id": manifest.pack_id,
        "pack_version": manifest.pack_version,
        "manifest_sha256": _manifest_digest(manifest),
        "files": files,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=receipt.parent,
        prefix=f".{receipt.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, receipt)


def import_model_pack(
    archive: Path,
    *,
    home: Path | None = None,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> ModelPackImportResult:
    archive = archive.expanduser().resolve()
    target_home = (home or portable_home()).expanduser().resolve()
    manifest, members = _archive_manifest(archive)
    if not is_manifest_compatible(manifest):
        raise ModelPackError(f"模型包 {manifest.display_name} 不支持当前系统或 CPU 架构。")
    target_home.mkdir(parents=True, exist_ok=True)
    receipt = _receipt_path(target_home, manifest)
    if _receipt_is_current(receipt, manifest, home=target_home):
        if log:
            log(f"已安装且文件未变化，直接复用：{manifest.display_name}")
        return ModelPackImportResult(
            archive=archive,
            pack_id=manifest.pack_id,
            installed_files=0,
            reused_files=len(manifest.files),
            uncompressed_bytes=manifest.uncompressed_bytes,
            already_installed=True,
        )

    total_steps = len(manifest.files) * 2
    if log:
        log(
            f"校验离线模型包：{manifest.display_name} "
            f"（{manifest.uncompressed_bytes / 1024**3:.2f} GiB）"
        )
    # Verify every byte before changing an installed model. Reading twice costs
    # time, but prevents a corrupt file near the end of a multi-gigabyte archive
    # from leaving a partially updated runtime.
    with zipfile.ZipFile(archive, "r", allowZip64=True) as handle:
        for index, file in enumerate(manifest.files, start=1):
            if progress:
                progress(f"校验 {file.path}", index - 1, total_steps)
            info = members[MODEL_PACK_PAYLOAD_PREFIX + file.path]
            digest = hashlib.sha256()
            observed = 0
            with handle.open(info, "r") as source:
                while block := source.read(8 * 1024 * 1024):
                    observed += len(block)
                    digest.update(block)
            if observed != file.size or digest.hexdigest() != file.sha256:
                raise ModelPackError(f"模型包文件校验失败：{file.path}")

    installed = 0
    reused = 0
    with zipfile.ZipFile(archive, "r", allowZip64=True) as handle:
        for index, file in enumerate(manifest.files, start=1):
            step = len(manifest.files) + index - 1
            if progress:
                progress(f"安装 {file.path}", step, total_steps)
            destination = _target_path(target_home, file.path)
            if (
                destination.is_file()
                and destination.stat().st_size == file.size
                and _hash_file(destination) == file.sha256
            ):
                reused += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            _target_path(target_home, file.path)
            temporary = _temporary_model_path(destination)
            temporary.unlink(missing_ok=True)
            try:
                info = members[MODEL_PACK_PAYLOAD_PREFIX + file.path]
                with handle.open(info, "r") as source, temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != file.size or _hash_file(temporary) != file.sha256:
                    raise ModelPackError(f"模型包文件落盘校验失败：{file.path}")
                os.replace(temporary, destination)
                installed += 1
            finally:
                temporary.unlink(missing_ok=True)
    _write_receipt(receipt, manifest, home=target_home)
    if progress:
        progress("离线模型包安装完成", total_steps, total_steps)
    if log:
        log(
            f"模型包安装完成：{manifest.display_name}；"
            f"新增/更新 {installed} 个文件，复用 {reused} 个文件。"
        )
    return ModelPackImportResult(
        archive=archive,
        pack_id=manifest.pack_id,
        installed_files=installed,
        reused_files=reused,
        uncompressed_bytes=manifest.uncompressed_bytes,
        already_installed=False,
    )


def import_discovered_model_packs(
    *,
    directory: Path | None = None,
    pack_ids: set[str] | None = None,
    home: Path | None = None,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> list[ModelPackImportResult]:
    results: list[ModelPackImportResult] = []
    for inspection in discover_model_packs(directory):
        if inspection.manifest is None:
            # A corrupt archive placed in the well-known inbox must never be
            # silently ignored and followed by an online download. Surface the
            # exact file so Setup/UI can tell the user what to replace.
            raise ModelPackError(f"无效的本地模型包 {inspection.archive.name}：{inspection.error}")
        if pack_ids is not None and inspection.manifest.pack_id not in pack_ids:
            continue
        if not inspection.compatible:
            if log:
                log(f"跳过不兼容模型包 {inspection.archive.name}：{inspection.error}")
            continue
        results.append(
            import_model_pack(
                inspection.archive,
                home=home,
                log=log,
                progress=progress,
            )
        )
    return results


def _source_files(sources: Sequence[ModelPackSource]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for source in sources:
        root = source.source.expanduser().resolve()
        target_root = str(_safe_relative_path(source.target)).rstrip("/")
        if root.is_symlink():
            raise ModelPackError(f"打包源不能是符号链接：{root}")
        if root.is_file():
            candidates: Iterable[Path] = (root,)
            source_base = root.parent
            target_base = PurePosixPath(target_root).parent
        elif root.is_dir():
            candidates = sorted(
                (path for path in root.rglob("*") if path.is_file()),
                key=lambda path: path.as_posix().casefold(),
            )
            source_base = root
            target_base = PurePosixPath(target_root)
        else:
            raise ModelPackError(f"打包源不存在：{root}")
        for candidate in candidates:
            if candidate.is_symlink():
                raise ModelPackError(f"打包源包含符号链接：{candidate}")
            relative = candidate.relative_to(source_base).as_posix()
            target = str(target_base / relative)
            target = str(_safe_relative_path(target))
            folded = target.casefold()
            if folded in seen:
                raise ModelPackError(f"多个打包源映射到同一目标：{target}")
            seen.add(folded)
            files.append((candidate, target))
    if not files:
        raise ModelPackError("模型包没有可打包文件。")
    return sorted(files, key=lambda item: item[1].casefold())


def build_model_pack(
    output: Path,
    *,
    pack_id: str,
    display_name: str,
    pack_version: str,
    platforms: Sequence[str],
    architectures: Sequence[str],
    sources: Sequence[ModelPackSource],
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> ModelPackManifest:
    if not _PACK_ID_RE.fullmatch(pack_id):
        raise ModelPackError(f"无效 pack_id：{pack_id}")
    source_files = _source_files(sources)
    manifest_files: list[ModelPackFile] = []
    total_steps = len(source_files) * 2
    for index, (source, target) in enumerate(source_files, start=1):
        if progress:
            progress(f"计算 SHA-256：{target}", index - 1, total_steps)
        manifest_files.append(
            ModelPackFile(
                path=target,
                size=source.stat().st_size,
                sha256=_hash_file(source),
            )
        )
    manifest = _manifest_from_mapping(
        {
            "schema_version": MODEL_PACK_SCHEMA_VERSION,
            "pack_id": pack_id,
            "display_name": display_name,
            "pack_version": pack_version,
            "platforms": list(platforms),
            "architectures": list(architectures),
            "files": [
                {"path": file.path, "size": file.size, "sha256": file.sha256}
                for file in manifest_files
            ],
        }
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as handle:
            manifest_bytes = (
                json.dumps(
                    _manifest_payload(manifest),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            handle.writestr(MODEL_PACK_MANIFEST, manifest_bytes)
            for index, ((source, target), declared) in enumerate(
                zip(source_files, manifest.files, strict=True),
                start=1,
            ):
                if progress:
                    progress(
                        f"写入模型包：{target}",
                        len(source_files) + index - 1,
                        total_steps,
                    )
                compression = (
                    zipfile.ZIP_STORED
                    if source.suffix.casefold() in _STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                info = zipfile.ZipInfo(
                    MODEL_PACK_PAYLOAD_PREFIX + declared.path,
                    date_time=(2020, 1, 1, 0, 0, 0),
                )
                info.compress_type = compression
                info.external_attr = 0o100644 << 16
                info.file_size = declared.size
                with (
                    source.open("rb") as input_handle,
                    handle.open(
                        info,
                        "w",
                        force_zip64=True,
                    ) as output_handle,
                ):
                    shutil.copyfileobj(
                        input_handle,
                        output_handle,
                        length=8 * 1024 * 1024,
                    )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{_hash_file(output)}  {output.name}\n", encoding="ascii")
    if progress:
        progress("模型包创建完成", total_steps, total_steps)
    if log:
        log(f"已创建 {output}（解压后 {manifest.uncompressed_bytes / 1024**3:.2f} GiB）")
    return manifest
