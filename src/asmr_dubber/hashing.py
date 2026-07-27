from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .audio import sha256_file


@lru_cache(maxsize=128)
def _cached_file_digest(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(Path(path_text))


def cached_sha256_file(path: Path) -> str:
    """Hash a stable file once per path/size/mtime tuple."""

    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return _cached_file_digest(str(resolved), stat.st_size, stat.st_mtime_ns)
