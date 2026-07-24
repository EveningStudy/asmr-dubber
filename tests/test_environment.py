import json
import tomllib
from pathlib import Path

import pytest

from asmr_dubber import __version__, environment
from asmr_dubber.constants import (
    DEFAULT_ASR_MODEL,
    MODEL_REQUIRED_FILES,
    OPTIONAL_ASR_MODEL_REVISIONS,
)
from asmr_dubber.errors import EnvironmentError


def test_partial_model_snapshot_is_not_usable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )
    assert environment.cached_model_path(DEFAULT_ASR_MODEL) is None


def test_complete_model_snapshot_is_usable(monkeypatch, tmp_path: Path) -> None:
    # Substitute tiny expected files so the test verifies completeness logic
    # without creating sparse multi-gigabyte fixtures.
    monkeypatch.setitem(MODEL_REQUIRED_FILES, DEFAULT_ASR_MODEL, {"weight.bin": 3})
    (tmp_path / "weight.bin").write_bytes(b"abc")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )
    assert environment.cached_model_path(DEFAULT_ASR_MODEL) == tmp_path.resolve()


def test_local_transformers_model_rejects_internal_kernel_field(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"nested": {"_attn_implementation_internal": "attacker/repo"}}),
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentError, match="不安全字段"):
        environment.resolve_transformers_model_source(str(tmp_path))


def test_unknown_remote_transformers_model_is_rejected() -> None:
    with pytest.raises(EnvironmentError, match="必须使用本地目录"):
        environment.resolve_transformers_model_source("unknown/model")


def test_known_transformers_model_uses_pinned_revision(monkeypatch, tmp_path: Path) -> None:
    model_id = "kotoba-tech/kotoba-whisper-v2.2"
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(environment, "cached_model_path", lambda _model_id: None)
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(tmp_path / "config.json")

    monkeypatch.setattr(environment, "hf_hub_download_with_fallback", fake_download)

    source, revision = environment.resolve_transformers_model_source(model_id)

    assert source == model_id
    assert revision == OPTIONAL_ASR_MODEL_REVISIONS[model_id]
    assert calls == [
        {
            "repo_id": model_id,
            "filename": "config.json",
            "revision": revision,
        }
    ]


def test_package_version_matches_project_metadata() -> None:
    root = Path(__file__).parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == metadata["project"]["version"]
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked_project = [item for item in lock["package"] if item["name"] == "asmr-dubber"]
    assert len(locked_project) == 1
    assert locked_project[0]["version"] == __version__


def test_release_files_are_present() -> None:
    root = Path(__file__).parents[1]
    required = {
        "ASMR-Dubber.exe",
        "ASMR-Dubber-Setup.exe",
        "mirrors.json",
        "README.md",
        "LICENSE",
        "docs/THIRD_PARTY_NOTICES.md",
        "docs/CHANGELOG.md",
        "scripts/linux/setup.sh",
        "scripts/windows/setup.ps1",
        "scripts/linux/run-ui.sh",
        "scripts/windows/run-ui.ps1",
        "launcher/windows/ASMRDubberLauncher.cs",
        "launcher/windows/ASMRDubberSetup.cs",
        "scripts/mirrors.ps1",
        "scripts/mirrors.sh",
        "scripts/portable-runtime.sh",
        "scripts/portable-runtime.ps1",
        ".github/workflows/ci.yml",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    assert not missing


@pytest.mark.parametrize("name", ["ASMR-Dubber.exe", "ASMR-Dubber-Setup.exe"])
def test_windows_launchers_are_console_executables(name: str) -> None:
    data = (Path(__file__).parents[1] / name).read_bytes()
    assert data[:2] == b"MZ"
    pe_offset = int.from_bytes(data[0x3C:0x40], "little")
    assert data[pe_offset : pe_offset + 4] == b"PE\0\0"
    subsystem = int.from_bytes(data[pe_offset + 92 : pe_offset + 94], "little")
    assert subsystem == 3  # IMAGE_SUBSYSTEM_WINDOWS_CUI


def test_launchers_use_project_portable_home() -> None:
    root = Path(__file__).parents[1]
    launchers = (
        "scripts/linux/setup.sh",
        "scripts/linux/run-cli.sh",
        "scripts/windows/setup.ps1",
        "scripts/windows/run-cli.ps1",
        "scripts/linux/install-indextts2.sh",
        "scripts/windows/install-indextts2.ps1",
        "scripts/portable-runtime.sh",
        "scripts/portable-runtime.ps1",
        "scripts/mirrors.sh",
        "scripts/mirrors.ps1",
    )
    combined = "\n".join((root / name).read_text(encoding="utf-8") for name in launchers)
    assert ".asmr-dubber" in combined
    assert "LOCALAPPDATA" not in combined
    assert "APPDATA" not in combined
    assert "XDG_DATA_HOME" not in combined
    assert "XDG_CONFIG_HOME" not in combined


def test_windows_portable_release_bundles_bootstrap_runtime() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    setup = (root / "scripts/windows/setup.ps1").read_text(encoding="utf-8")
    mirrors = json.loads((root / "mirrors.json").read_text(encoding="utf-8"))

    assert 'version: "0.11.30"' in workflow
    assert 'Join-Path $portable "bootstrap\\windows\\uv"' in workflow
    assert 'Join-Path $portable "runtimes\\python"' in workflow
    assert "uv python install 3.12 --managed-python --no-bin" in workflow
    assert "uv_archives_windows" in setup
    assert "be8d78c992312212e5cc05e9f9de3fa996db73b7c86a186dfb9231eb9f91d33e" in setup
    assert mirrors["uv_archives_windows"][0].startswith("https://releases.astral.sh/")
    assert mirrors["python_install_mirrors"][0].startswith("https://releases.astral.sh/")
