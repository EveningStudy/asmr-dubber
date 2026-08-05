from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .errors import EnvironmentError

_WINDOWS_DLL_DIRECTORY_HANDLES: list[object] = []
_WINDOWS_DLL_DIRECTORIES: set[str] = set()


def _absolute_user_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    # Avoid MSIX/AppContainer path virtualization on Windows.
    if current_platform().is_windows:
        return Path(os.path.abspath(candidate))
    return candidate.resolve()


@dataclass(frozen=True)
class PlatformInfo:
    system: str
    architecture: str
    is_windows: bool
    is_linux: bool
    is_wsl: bool

    @property
    def id(self) -> str:
        if self.is_windows:
            return "windows"
        if self.is_linux:
            return "linux"
        return self.system.lower()

    @property
    def label(self) -> str:
        if self.is_wsl:
            return "Windows WSL2"
        if self.is_windows:
            return "Windows"
        if self.is_linux:
            return "Linux"
        return self.system


def current_platform() -> PlatformInfo:
    system = platform.system()
    release = platform.release().lower()
    version = ""
    if system == "Linux":
        with suppress(OSError):
            version = Path("/proc/version").read_text(encoding="utf-8").lower()
    is_wsl = system == "Linux" and ("microsoft" in release or "microsoft" in version)
    return PlatformInfo(
        system=system,
        architecture=platform.machine() or "unknown",
        is_windows=system == "Windows",
        is_linux=system == "Linux",
        is_wsl=is_wsl,
    )


def require_supported_platform() -> None:
    info = current_platform()
    if not (info.is_windows or info.is_linux):
        raise EnvironmentError(f"当前系统 {info.system} 尚未支持；支持 Windows 和 Linux。")
    if not sys.maxsize > 2**32:
        raise EnvironmentError("本工具只支持 64 位 Windows/Linux。")


def open_directory(path: str | os.PathLike[str]) -> Path:
    """Open an existing local directory in the platform file manager."""

    directory = _absolute_user_path(path)
    if not directory.is_dir():
        raise EnvironmentError(f"目录不存在：{directory}")
    info = current_platform()
    try:
        if info.is_windows:
            startfile = getattr(os, "startfile", None)
            if startfile is None:
                raise EnvironmentError("当前 Python 无法调用 Windows 资源管理器。")
            startfile(str(directory))
        elif info.is_linux:
            opener = shutil.which("xdg-open")
            if not opener:
                raise EnvironmentError("未找到 xdg-open，无法调用系统文件管理器。")
            subprocess.Popen(
                [opener, str(directory)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            raise EnvironmentError(f"当前系统 {info.system} 不支持打开项目目录。")
    except OSError as exc:
        raise EnvironmentError(f"无法打开目录 {directory}：{exc}") from exc
    return directory


def portable_home() -> Path:
    """Return the project-owned home used for every persistent application file."""
    override = os.getenv("ASMR_DUBBER_HOME", "").strip()
    if override:
        return _absolute_user_path(override)

    # A launcher-created environment lives at <project>/.asmr-dubber/venv.
    prefix = _absolute_user_path(sys.prefix)
    if prefix.name.casefold() == "venv" and prefix.parent.name == ".asmr-dubber":
        return prefix.parent

    # Editable/source execution can locate the repository without relying on
    # the current working directory.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".asmr-dubber"
    return _absolute_user_path(Path.cwd() / ".asmr-dubber")


def isolated_runtime_environment(runtime_name: str) -> dict[str, str]:
    """Return a child-process environment whose writable state stays portable.

    Third-party command backends do not necessarily honor this application's
    cache/config variables.  In particular, some Windows CLIs write directly
    below APPDATA.  Override those locations only for the child process so the
    user's real profile and persistent environment remain untouched.
    """
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in runtime_name.strip()
    ).strip("._-")
    if not safe_name:
        raise ValueError("runtime_name must contain at least one safe character")

    env = os.environ.copy()
    state_root = portable_home() / "runtimes" / safe_name / "user-state"
    cache_root = portable_home() / "cache" / safe_name
    state_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    env["XDG_CONFIG_HOME"] = str(state_root / "config")
    env["XDG_DATA_HOME"] = str(state_root / "data")
    env["XDG_STATE_HOME"] = str(state_root / "state")
    env["XDG_CACHE_HOME"] = str(cache_root)
    # Batch TTS workers report one line per generated sentence.  Force Python
    # CLIs to flush those lines immediately so the UI can advance in real time.
    env["PYTHONUNBUFFERED"] = "1"
    if current_platform().is_windows:
        roaming = state_root / "Roaming"
        local = state_root / "Local"
        roaming.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        env["APPDATA"] = str(roaming)
        env["LOCALAPPDATA"] = str(local)
    return env


def user_config_dir() -> Path:
    override = os.getenv("ASMR_DUBBER_CONFIG_DIR", "").strip()
    if override:
        return _absolute_user_path(override)
    return portable_home() / "config"


def user_data_dir() -> Path:
    override = os.getenv("ASMR_DUBBER_DATA_DIR", "").strip()
    if override:
        return _absolute_user_path(override)
    return portable_home()


def virtualenv_executable(venv: Path, name: str) -> Path:
    """Return the native executable path for a venv on the current platform."""
    if current_platform().is_windows:
        suffix = "" if name.lower().endswith((".exe", ".cmd", ".bat")) else ".exe"
        return venv / "Scripts" / f"{name}{suffix}"
    return venv / "bin" / name


def runtime_executable_candidates(runtime_root: Path, name: str) -> tuple[Path, ...]:
    """Include both layouts so projects copied between WSL and Windows remain inspectable."""
    windows_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    return (
        runtime_root / ".venv" / "Scripts" / windows_name,
        runtime_root / ".venv" / "bin" / name,
    )


def configure_windows_dll_directories() -> tuple[str, ...]:
    """Expose private native runtimes to Python's secure Windows DLL loader."""
    if not current_platform().is_windows or not hasattr(os, "add_dll_directory"):
        return ()
    candidates: list[Path] = []
    ffmpeg = os.getenv("ASMR_DUBBER_FFMPEG", "").strip()
    if ffmpeg:
        candidates.append(_absolute_user_path(ffmpeg).parent)

    # CTranslate2's Windows GPU wheel loads CUDA 12 BLAS dynamically. PyTorch
    # can bundle a different CUDA major version, therefore Faster-Whisper's
    # optional extra installs NVIDIA's private CUDA 12 runtime under the venv.
    # Register those package-private DLL folders without modifying system PATH.
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    candidates.extend(
        (
            site_packages / "nvidia" / "cublas" / "bin",
            site_packages / "nvidia" / "cudnn" / "bin",
        )
    )

    for candidate in candidates:
        directory = str(candidate)
        if directory in _WINDOWS_DLL_DIRECTORIES or not candidate.is_dir():
            continue
        # Python 3.8+ requires an explicit directory for dependencies of
        # extension modules. Keep the handle alive for the process lifetime.
        handle = os.add_dll_directory(directory)
        _WINDOWS_DLL_DIRECTORY_HANDLES.append(handle)
        _WINDOWS_DLL_DIRECTORIES.add(directory)
        # CTranslate2 resolves CUDA libraries with its own native loader, which
        # does not consistently honor Python's add_dll_directory handles.
        # Change only this process environment; never persist a system PATH.
        path_entries = [item for item in os.environ.get("PATH", "").split(";") if item]
        if directory.casefold() not in {item.casefold() for item in path_entries}:
            os.environ["PATH"] = ";".join((directory, *path_entries))
    return tuple(sorted(_WINDOWS_DLL_DIRECTORIES))
