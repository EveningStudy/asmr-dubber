"""Owned uploads and explicitly registered media files."""

import mimetypes
import secrets
import shutil
import threading
import time
import weakref
from pathlib import Path

from ..platforms import portable_home
from ..storage import require_disk_space

STALE_UPLOAD_SECONDS = 24 * 60 * 60
# Largest accepted upload (Content-Length); long multi-track recordings are big.
MAX_UPLOAD_BYTES = 20 * 1024**3
# A token that was neither registered nor requested for this long is dropped, and
# the table never keeps more than MAX_TOKENS entries (least recently used go first).
# Every Range request refreshes the token, so media that is playing stays valid.
TOKEN_IDLE_SECONDS = 6 * 60 * 60
MAX_TOKENS = 4096
_PURGE_INTERVAL_SECONDS = 60.0
_STORES = weakref.WeakSet()


def uploads_root():
    return portable_home() / "temp" / "uploads"


def discard_upload(path):
    """Remove an upload once the service that consumed it has made its own copy."""
    if not path:
        return
    directory = Path(path).resolve().parent
    if directory.parent == uploads_root().resolve():
        for store in list(_STORES):
            store.forget_under(directory)
        shutil.rmtree(directory, ignore_errors=True)


class MediaStore:
    def __init__(self):
        self._files = {}
        self._tokens = {}
        self._seen = {}
        self._purged_at = time.monotonic()
        self._lock = threading.Lock()
        _STORES.add(self)
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
            now = time.monotonic()
            self._purge(now)
            token = self._tokens.get(file)
            if token is None:
                token = self._tokens[file] = secrets.token_urlsafe(24)
                self._files[token] = file
            self._seen[token] = now
        return f"/media/{token}"

    def resolve(self, token):
        with self._lock:
            file = self._files.get(token)
            if file is not None:
                self._seen[token] = time.monotonic()
        if file is None or not file.is_file():
            raise FileNotFoundError("找不到这个媒体文件。")
        return file, mimetypes.guess_type(file.name)[0] or "application/octet-stream"

    def forget_under(self, directory):
        """Drop every token whose file lives below `directory` (closed or replaced)."""
        root = Path(directory).resolve()
        with self._lock:
            for file in [f for f in self._tokens if f == root or root in f.parents]:
                self._drop(file)

    def _drop(self, file):
        token = self._tokens.pop(file, None)
        if token is not None:
            self._files.pop(token, None)
            self._seen.pop(token, None)

    def _purge(self, now):
        """Reclaim expired tokens and tokens of vanished files; caller holds the lock."""
        if len(self._tokens) <= MAX_TOKENS and now - self._purged_at < _PURGE_INTERVAL_SECONDS:
            return
        self._purged_at = now
        for file, token in list(self._tokens.items()):
            if now - self._seen.get(token, now) > TOKEN_IDLE_SECONDS or not file.is_file():
                self._drop(file)
        excess = len(self._tokens) - MAX_TOKENS
        if excess > 0:
            oldest = sorted(self._tokens, key=lambda f: self._seen.get(self._tokens[f], 0.0))
            for file in oldest[:excess]:
                self._drop(file)

    def upload(self, stream, length, filename):
        name = Path(filename.replace("\\", "/")).name
        if not name or name in {".", ".."}:
            raise ValueError("文件名无效。")
        if length < 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"文件太大：上传上限为 {MAX_UPLOAD_BYTES / 1024**3:.0f} GiB，"
                f"这个文件有 {length / 1024**3:.2f} GiB。"
            )
        root = uploads_root()
        require_disk_space(root, length)
        directory = root / secrets.token_hex(12)
        directory.mkdir(parents=True)
        file = directory / name
        try:
            with file.open("wb") as output:
                remaining = length
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("上传提前结束，收到的数据少于声明的大小。")
                    output.write(block)
                    remaining -= len(block)
        except Exception:
            file.unlink(missing_ok=True)
            directory.rmdir()
            raise
        return {"path": str(file), "name": name, "url": self.register(file)}
