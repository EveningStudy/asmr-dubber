import os
from pathlib import Path

from asmr_dubber import separation


def test_shared_ffmpeg_stays_beside_its_dlls_and_shadows_stale_copy(tmp_path, monkeypatch):
    shared = tmp_path / "shared/bin"
    shared.mkdir(parents=True)
    executable = shared / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    executable.write_bytes(b"shared executable")
    (shared / "avcodec-62.dll").write_bytes(b"required sibling")
    stale = tmp_path / "runtimes/separation/bin/ffmpeg.exe"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"previous isolated copy")
    monkeypatch.setattr(separation, "portable_home", lambda: tmp_path)
    monkeypatch.setattr(separation, "ffmpeg_executable", lambda: str(executable))
    environment = separation.worker_environment()
    assert Path(environment["PATH"].split(os.pathsep)[0]) == shared
    assert (executable.parent / "avcodec-62.dll").is_file()


def test_static_imageio_binary_gets_the_name_required_by_separator(tmp_path, monkeypatch):
    source = tmp_path / "imageio/ffmpeg-version.exe"
    source.parent.mkdir()
    source.write_bytes(b"static executable")
    monkeypatch.setattr(separation, "portable_home", lambda: tmp_path)
    monkeypatch.setattr(separation, "ffmpeg_executable", lambda: str(source))
    environment = separation.worker_environment()
    folder = Path(environment["PATH"].split(os.pathsep)[0])
    assert (
        folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    ).read_bytes() == source.read_bytes()
