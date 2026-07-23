from __future__ import annotations

from pathlib import Path

from asmr_dubber import platforms


def test_linux_user_directories_are_portable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        lambda: platforms.PlatformInfo("Linux", "x86_64", False, True, False),
    )
    home = tmp_path / "project" / ".asmr-dubber"
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(home))
    monkeypatch.delenv("ASMR_DUBBER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("ASMR_DUBBER_DATA_DIR", raising=False)

    assert platforms.user_config_dir() == home.resolve() / "config"
    assert platforms.user_data_dir() == home.resolve()


def test_windows_user_directories_and_venv_layout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        lambda: platforms.PlatformInfo("Windows", "AMD64", True, False, False),
    )
    home = tmp_path / "project" / ".asmr-dubber"
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(home))
    monkeypatch.delenv("ASMR_DUBBER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("ASMR_DUBBER_DATA_DIR", raising=False)

    assert platforms.user_config_dir() == home.resolve() / "config"
    assert platforms.user_data_dir() == home.resolve()
    assert platforms.virtualenv_executable(tmp_path / ".venv", "python") == (
        tmp_path / ".venv" / "Scripts" / "python.exe"
    )


def test_isolated_windows_runtime_redirects_profile_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        lambda: platforms.PlatformInfo("Windows", "AMD64", True, False, False),
    )
    home = tmp_path / ".asmr-dubber"
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(home))
    monkeypatch.setenv("APPDATA", str(tmp_path / "real-roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "real-local"))

    env = platforms.isolated_runtime_environment("index-tts")

    state = home / "runtimes" / "index-tts" / "user-state"
    assert env["APPDATA"] == str(state / "Roaming")
    assert env["LOCALAPPDATA"] == str(state / "Local")
    assert env["XDG_CACHE_HOME"] == str(home / "cache" / "index-tts")
    assert Path(env["APPDATA"]).is_dir()
    assert Path(env["LOCALAPPDATA"]).is_dir()


def test_configure_windows_dll_directories(monkeypatch, tmp_path: Path) -> None:
    ffmpeg = tmp_path / "ffmpeg" / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.touch()
    cublas = tmp_path / "venv" / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
    cublas.mkdir(parents=True)
    added: list[str] = []

    monkeypatch.setattr(
        platforms,
        "current_platform",
        lambda: platforms.PlatformInfo("Windows", "AMD64", True, False, False),
    )
    monkeypatch.setenv("ASMR_DUBBER_FFMPEG", str(ffmpeg))
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    monkeypatch.setattr(platforms.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(
        platforms.os, "add_dll_directory", lambda path: added.append(path), raising=False
    )
    monkeypatch.setattr(platforms, "_WINDOWS_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(platforms, "_WINDOWS_DLL_DIRECTORIES", set())

    configured = platforms.configure_windows_dll_directories()

    assert configured == tuple(sorted((str(cublas), str(ffmpeg.parent))))
    assert added == [str(ffmpeg.parent), str(cublas)]
    assert platforms.os.environ["PATH"].split(";")[:2] == [str(cublas), str(ffmpeg.parent)]
