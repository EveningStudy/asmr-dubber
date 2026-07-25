from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/import_windows_dependency_pack.py"
SPEC = importlib.util.spec_from_file_location("dependency_pack_importer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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
        "payload/runtimes/ffmpeg-shared/build/bin/ffmpeg.exe": b"ffmpeg",
        "payload/runtimes/ffmpeg-shared/build/bin/avcodec-62.dll": b"codec",
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as handle:
        handle.writestr("dependency-pack.json", json.dumps(_manifest()))
        for name, content in files.items():
            handle.writestr(name, content)
        if unsafe:
            handle.writestr("payload/venv/../../escape.txt", b"escape")


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
    marker = json.loads(
        (portable / "runtimes/windows-recommended-dependencies.json").read_text(encoding="utf-8")
    )
    assert marker == {
        "pack_id": "windows-recommended-dependencies",
        "pack_version": "1.0.0",
        "sha256": "a" * 64,
    }


def test_dependency_pack_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    _write_pack(archive, unsafe=True)

    with pytest.raises(MODULE.DependencyPackError, match="unsafe ZIP path"):
        MODULE.import_pack(archive, tmp_path / "portable", "b" * 64)

    assert not (tmp_path / "escape.txt").exists()
