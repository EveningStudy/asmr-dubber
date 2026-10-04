"""Owned uploads and explicitly registered media files."""

import mimetypes
import secrets
import shutil
import threading
import time
from pathlib import Path

from ..platforms import portable_home

STALE_UPLOAD_SECONDS = 24 * 60 * 60


def uploads_root():
    return portable_home() / "temp" / "uploads"


def discard_upload(path):
    """Remove an upload once the service that consumed it has made its own copy."""
    if not path:
        return
    directory = Path(path).resolve().parent
    if directory.parent == uploads_root().resolve():
        shutil.rmtree(directory, ignore_errors=True)


class MediaStore:
    def __init__(self):
        self._files = {}
        self._tokens = {}
        self._lock = threading.Lock()
        # Uploads abandoned by a closed dialog or a crash are kept for one day so an
        # interrupted task can still be resumed, then removed.
        root = uploads_root()
        if root.is_dir():
            limit = time.time() - STALE_UPLOAD_SECONDS
            for directory in root.iterdir():
                if directory.is_dir() and directory.stat().st_mtime < limit:
                    shutil.rmtree(directory, ignore_errors=True)

    def register(self, path):
        if not path:
            return None
        file = Path(path).resolve()
        if not file.is_file():
            return None
        with self._lock:
            token = self._tokens.get(file)
            if token is None:
                token = self._tokens[file] = secrets.token_urlsafe(24)
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
        directory = uploads_root() / secrets.token_hex(12)
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
