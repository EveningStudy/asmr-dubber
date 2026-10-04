import io
import threading
import zipfile

import pytest

from asmr_dubber.services import updates


def archive(path, files, prefix="ASMR-Dubber/"):
    with zipfile.ZipFile(path, "w") as output:
        for name, data in files.items():
            output.writestr(prefix + name, data)
    return path


RELEASE = {
    "ASMR-Dubber.exe": b"new-exe",
    "pyproject.toml": b"new-project",
    "src/asmr_dubber/__init__.py": b"new-init",
    "src/asmr_dubber/fresh.py": b"fresh",
    "scripts/run.ps1": b"script",
    ".asmr-dubber/venv/marker": b"must not be installed",
}


@pytest.fixture
def installed(tmp_path):
    root = tmp_path / "program"
    (root / "src/asmr_dubber").mkdir(parents=True)
    (root / ".asmr-dubber/projects").mkdir(parents=True)
    (root / "ASMR-Dubber.exe").write_bytes(b"old-exe")
    (root / "src/asmr_dubber/__init__.py").write_bytes(b"old-init")
    (root / "src/asmr_dubber/removed.py").write_bytes(b"gone in the new version")
    (root / ".asmr-dubber/projects/keep.json").write_bytes(b"user data")
    (root / "notes.txt").write_bytes(b"user file beside the program")
    return root


def test_version_comparison_uses_numbers_not_text():
    assert updates.version_tuple("v1.10.0") > updates.version_tuple("1.9.9")
    assert updates.version_tuple("nightly") is None


def test_apply_replaces_program_files_and_keeps_user_data(installed, tmp_path):
    updates.apply(archive(tmp_path / "release.zip", RELEASE), installed, tmp_path / "staged")
    assert (installed / "ASMR-Dubber.exe").read_bytes() == b"new-exe"
    assert (installed / "src/asmr_dubber/__init__.py").read_bytes() == b"new-init"
    assert (installed / "src/asmr_dubber/fresh.py").is_file()
    assert (installed / "scripts/run.ps1").is_file()
    assert not (installed / "src/asmr_dubber/removed.py").exists()
    assert (installed / ".asmr-dubber/projects/keep.json").read_bytes() == b"user data"
    assert not (installed / ".asmr-dubber/venv").exists()
    assert (installed / "notes.txt").is_file()


def test_incomplete_or_unsafe_archive_changes_nothing(installed, tmp_path):
    partial = {name: data for name, data in RELEASE.items() if name != "pyproject.toml"}
    with pytest.raises(ValueError):
        updates.apply(archive(tmp_path / "partial.zip", partial), installed, tmp_path / "a")
    unsafe = {**RELEASE, "../outside.txt": b"escape"}
    with pytest.raises(ValueError):
        updates.apply(archive(tmp_path / "unsafe.zip", unsafe), installed, tmp_path / "b")
    assert (installed / "ASMR-Dubber.exe").read_bytes() == b"old-exe"
    assert (installed / "src/asmr_dubber/removed.py").is_file()
    assert not (tmp_path / "outside.txt").exists()


def test_source_checkout_is_never_updated_automatically(installed, monkeypatch):
    monkeypatch.setattr(updates, "portable_home", lambda: installed / ".asmr-dubber")
    monkeypatch.setattr(updates.os, "name", "nt")
    assert updates.installable_root() == installed
    (installed / ".git").mkdir()
    assert updates.installable_root() is None


class Response:
    def __init__(self, payload=None, body=b""):
        self.payload, self.body, self.headers = payload, body, {}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_bytes(self, size):
        yield self.body

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False


def release(tag, body=b""):
    url = updates.DOWNLOAD_PREFIX + f"{tag}/ASMR-Dubber-windows-portable-{tag}.zip"
    asset = {"name": url.rsplit("/", 1)[1], "browser_download_url": url, "size": len(body)}
    return {"tag_name": tag, "html_url": "https://example.invalid/release", "assets": [asset]}


def test_check_reports_newer_release_and_network_failure(installed, monkeypatch):
    monkeypatch.setattr(updates, "portable_home", lambda: installed / ".asmr-dubber")
    monkeypatch.setattr(updates.os, "name", "nt")
    monkeypatch.setattr(updates.httpx, "get", lambda *a, **k: Response(release("v99.0.0")))
    found = updates.check(force=True)
    assert found["newer"] and found["automatic"] and found["latest"] == "99.0.0"
    monkeypatch.setattr(updates.httpx, "get", lambda *a, **k: Response(release("v0.0.1")))
    assert not updates.check(force=True)["newer"]

    def offline(*arguments, **keywords):
        raise updates.httpx.ConnectError("offline")

    monkeypatch.setattr(updates.httpx, "get", offline)
    failed = updates.check(force=True)
    assert failed["error"] and not failed["newer"] and failed["page"] == updates.RELEASE_PAGE


def test_install_downloads_verifies_and_applies(installed, tmp_path, monkeypatch):
    body = archive(io.BytesIO(), RELEASE).getvalue()
    monkeypatch.setattr(updates, "portable_home", lambda: installed / ".asmr-dubber")
    monkeypatch.setattr(updates.os, "name", "nt")
    monkeypatch.setattr(updates.httpx, "get", lambda *a, **k: Response(release("v99.0.0", body)))
    monkeypatch.setattr(updates.httpx, "stream", lambda *a, **k: Response(body=body))
    progress = []
    result = updates.install({}, lambda *a: progress.append(a), threading.Event())
    assert result["version"] == "99.0.0" and progress
    assert (installed / "ASMR-Dubber.exe").read_bytes() == b"new-exe"
    assert (installed / ".asmr-dubber/projects/keep.json").is_file()
    assert not (installed / ".asmr-dubber/temp/update").exists()

    monkeypatch.setattr(updates.httpx, "stream", lambda *a, **k: Response(body=body[:-10]))
    with pytest.raises(ValueError):
        updates.install({}, lambda *a: None, threading.Event())
