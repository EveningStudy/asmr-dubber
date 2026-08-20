from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/import_windows_dependency_pack.py"
SPEC = importlib.util.spec_from_file_location("dependency_pack_importer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ADVANCED_SCRIPT = ROOT / "scripts/import_windows_advanced_dependency_pack.py"
ADVANCED_SPEC = importlib.util.spec_from_file_location(
    "advanced_dependency_pack_importer", ADVANCED_SCRIPT
)
assert ADVANCED_SPEC is not None and ADVANCED_SPEC.loader is not None
ADVANCED_MODULE = importlib.util.module_from_spec(ADVANCED_SPEC)
ADVANCED_SPEC.loader.exec_module(ADVANCED_MODULE)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack_id": "windows-recommended-dependencies",
        "pack_version": "1.0.0",
        "platform": "windows",
        "architecture": "x86_64",
        "python": ["3.12", "3.11.13"],
        "indextts_revision": "13495845e3028f0bb6ca1462ad22aa0e76349e40",
        "components": ["application-ui", "indextts2", "ffmpeg-shared"],
    }


def _write_pack(path: Path, *, unsafe: bool = False) -> None:
    files = {
        "payload/venv/Scripts/python.exe": b"core-python",
        "payload/venv/Scripts/asmr-dubber.exe": b"launcher",
        "payload/venv/Lib/site-packages/gradio/__init__.py": b"",
        ("payload/runtimes/python/cpython-3.11.13-windows-x86_64-none/python.exe"): b"base-python",
        "payload/runtimes/index-tts/.venv/Scripts/python.exe": b"index-python",
        "payload/runtimes/index-tts/.venv/Lib/site-packages/torch/__init__.py": b"",
        (
            "payload/runtimes/python/cpython-3.11.13-windows-x86_64-none/Lib/"
            "site-packages/pkg_resources/tests/data/deep-fixture.txt"
        ): b"not needed at runtime",
        "payload/runtimes/ffmpeg-shared/build/bin/ffmpeg.exe": b"ffmpeg",
        "payload/runtimes/ffmpeg-shared/build/bin/avcodec-62.dll": b"codec",
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as handle:
        handle.writestr("dependency-pack.json", json.dumps(_manifest()))
        for name, content in files.items():
            handle.writestr(name, content)
        if unsafe:
            handle.writestr("payload/venv/../../escape.txt", b"escape")


def _append_deep_member(path: Path, payload_root: str = "venv") -> str:
    directories = "/".join(f"deep_dependency_segment_{index:02d}" for index in range(10))
    relative = f"{payload_root}/Lib/site-packages/{directories}/module.py"
    with zipfile.ZipFile(path, "a", allowZip64=True) as handle:
        handle.writestr(f"payload/{relative}", b"long-path-supported")
    return relative


def test_dependency_pack_imports_components_and_replaces_incomplete_runtime(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pack.zip"
    _write_pack(archive)
    portable = tmp_path / "portable"
    old_venv = portable / "venv"
    old_venv.mkdir(parents=True)
    (old_venv / "old.txt").write_text("old", encoding="utf-8")

    MODULE.import_pack(archive, portable, "a" * 64)

    assert (portable / "venv/Scripts/python.exe").read_bytes() == b"core-python"
    assert not (portable / "venv/old.txt").exists()
    assert (portable / "runtimes/index-tts/.venv/Scripts/python.exe").is_file()
    assert not (
        portable
        / "runtimes/python/cpython-3.11.13-windows-x86_64-none/Lib/"
        / "site-packages/pkg_resources/tests"
    ).exists()
    marker = json.loads(
        (portable / "runtimes/windows-recommended-dependencies.json").read_text(encoding="utf-8")
    )
    assert marker == {
        "pack_id": "windows-recommended-dependencies",
        "pack_version": "1.0.0",
        "sha256": "a" * 64,
    }


def test_dependency_pack_runtime_only_preserves_running_application(tmp_path: Path) -> None:
    archive = tmp_path / "pack.zip"
    _write_pack(archive)
    portable = tmp_path / "portable"
    application = portable / "venv"
    application.mkdir(parents=True)
    (application / "running.txt").write_text("keep", encoding="utf-8")

    MODULE.import_pack(archive, portable, "f" * 64, runtime_only=True)

    assert (application / "running.txt").read_text(encoding="utf-8") == "keep"
    assert not (application / "Scripts/python.exe").exists()
    assert (portable / "runtimes/index-tts/.venv/Scripts/python.exe").is_file()


def test_dependency_pack_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    _write_pack(archive, unsafe=True)

    with pytest.raises(MODULE.DependencyPackError, match="unsafe ZIP path"):
        MODULE.import_pack(archive, tmp_path / "portable", "b" * 64)

    assert not (tmp_path / "escape.txt").exists()


def test_dependency_pack_imports_long_windows_paths(tmp_path: Path) -> None:
    archive = tmp_path / "long-pack.zip"
    _write_pack(archive)
    relative = _append_deep_member(archive)
    portable = tmp_path / "portable"
    destination = portable / Path(relative)
    if os.name == "nt":
        assert len(str(destination)) >= 260

    try:
        MODULE.import_pack(archive, portable, "e" * 64)
        assert MODULE._windows_io_path(destination).read_bytes() == b"long-path-supported"
    finally:
        if portable.exists():
            MODULE._remove_tree(portable)


def _advanced_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack_id": "windows-advanced-dependencies",
        "pack_version": "1.0.0",
        "platform": "windows",
        "architecture": "x86_64",
        "python": "3.12",
        "components": [
            "application-ui",
            "advanced-asr",
            "qwen-forced-aligner",
            "asmr-vad",
            "pytorch-cu130",
        ],
    }


def _write_advanced_pack(path: Path, *, unsafe: bool = False) -> None:
    files = {
        "payload/venv/Scripts/python.exe": b"advanced-python",
        "payload/venv/Lib/site-packages/gradio/__init__.py": b"",
        "payload/venv/Lib/site-packages/torch/__init__.py": b"",
        "payload/venv/Lib/site-packages/torchaudio/__init__.py": b"",
        "payload/venv/Lib/site-packages/transformers/__init__.py": b"",
        "payload/venv/Lib/site-packages/faster_whisper/__init__.py": b"",
        (
            "payload/venv/Lib/site-packages/pkg_resources/tests/data/deep-fixture.txt"
        ): b"not needed at runtime",
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as handle:
        handle.writestr("dependency-pack.json", json.dumps(_advanced_manifest()))
        for name, content in files.items():
            handle.writestr(name, content)
        if unsafe:
            handle.writestr("payload/venv/../../escape.txt", b"escape")


def test_advanced_dependency_pack_imports_supported_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "advanced.zip"
    _write_advanced_pack(archive)
    portable = tmp_path / "portable"
    old_venv = portable / "venv"
    old_venv.mkdir(parents=True)
    (old_venv / "old.txt").write_text("old", encoding="utf-8")

    ADVANCED_MODULE.import_pack(archive, portable, "c" * 64)

    assert (portable / "venv/Scripts/python.exe").read_bytes() == b"advanced-python"
    assert not (portable / "venv/old.txt").exists()
    assert not (portable / "venv/Lib/site-packages/pkg_resources/tests").exists()
    marker = json.loads(
        (portable / "runtimes/windows-advanced-dependencies.json").read_text(encoding="utf-8")
    )
    assert marker == {
        "pack_id": "windows-advanced-dependencies",
        "pack_version": "1.0.0",
        "sha256": "c" * 64,
    }


def test_advanced_dependency_pack_can_merge_into_running_webui_venv(tmp_path: Path) -> None:
    archive = tmp_path / "advanced.zip"
    _write_advanced_pack(archive)
    portable = tmp_path / "portable"
    current_python = portable / "venv/Scripts/python.exe"
    current_python.parent.mkdir(parents=True)
    current_python.write_bytes(b"advanced-python")
    retained = portable / "venv/keep-recommended.txt"
    retained.write_text("keep", encoding="utf-8")

    ADVANCED_MODULE.import_pack(
        archive,
        portable,
        "e" * 64,
        merge_existing=True,
    )

    assert current_python.read_bytes() == b"advanced-python"
    assert retained.read_text(encoding="utf-8") == "keep"
    assert (portable / "venv/Lib/site-packages/faster_whisper").is_dir()
    assert (portable / "runtimes/windows-advanced-dependencies.json").is_file()


def test_advanced_dependency_pack_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe-advanced.zip"
    _write_advanced_pack(archive, unsafe=True)

    with pytest.raises(ADVANCED_MODULE.DependencyPackError, match="unsafe ZIP path"):
        ADVANCED_MODULE.import_pack(archive, tmp_path / "portable", "d" * 64)

    assert not (tmp_path / "escape.txt").exists()


def test_advanced_dependency_pack_imports_long_windows_paths(tmp_path: Path) -> None:
    archive = tmp_path / "long-advanced.zip"
    _write_advanced_pack(archive)
    relative = _append_deep_member(archive)
    portable = tmp_path / "portable"
    destination = portable / Path(relative)
    if os.name == "nt":
        assert len(str(destination)) >= 260

    try:
        ADVANCED_MODULE.import_pack(archive, portable, "f" * 64)
        assert ADVANCED_MODULE._windows_io_path(destination).read_bytes() == (
            b"long-path-supported"
        )
    finally:
        if portable.exists():
            ADVANCED_MODULE._remove_tree(portable)
