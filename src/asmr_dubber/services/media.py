"""Owned uploads and explicitly registered media files."""

import mimetypes
import secrets
import threading
from pathlib import Path

from ..platforms import portable_home


class MediaStore:
    def __init__(self):
        self._files = {}
        self._lock = threading.Lock()

    def register(self, path):
        if not path:
            return None
        file = Path(path).resolve()
        if not file.is_file():
            return None
        with self._lock:
            for token, registered in self._files.items():
                if registered == file:
                    return f"/media/{token}"
            token = secrets.token_urlsafe(24)
            self._files[token] = file
        return f"/media/{token}"

    def resolve(self, token):
        with self._lock:
            file = self._files.get(token)
        if file is None or not file.is_file():
            raise FileNotFoundError("Media not found")
        return file, mimetypes.guess_type(file.name)[0] or "application/octet-stream"

    def upload(self, stream, length, filename):
        name = Path(filename.replace("\\", "/")).name
        if not name or name in {".", ".."}:
            raise ValueError("Invalid filename")
        directory = portable_home() / "temp" / "uploads" / secrets.token_hex(12)
        directory.mkdir(parents=True)
        file = directory / name
        try:
            with file.open("wb") as output:
                remaining = length
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Upload ended before Content-Length")
                    output.write(block)
                    remaining -= len(block)
        except Exception:
            file.unlink(missing_ok=True)
            directory.rmdir()
            raise
        return {"path": str(file), "name": name, "url": self.register(file)}
