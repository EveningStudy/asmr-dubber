"""Download one pinned Hugging Face file into a portable model directory."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from asmr_dubber.mirrors import hf_hub_download_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--minimum-bytes", type=int, default=1)
    parser.add_argument("--sha256", default="")
    parser.add_argument("--endpoints", default="")
    args = parser.parse_args()

    destination = Path(args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = args.sha256.strip().lower()
    if expected_hash and (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError("--sha256 must be a lowercase or uppercase SHA-256 digest")
    if (
        destination.is_file()
        and destination.stat().st_size >= args.minimum_bytes
        and (not expected_hash or _sha256(destination) == expected_hash)
    ):
        print(f"{destination} ({destination.stat().st_size} bytes, cached)")
        return
    downloaded = Path(
        hf_hub_download_with_fallback(
            repo_id=args.repo,
            filename=args.filename,
            revision=args.revision,
            local_dir=destination.parent,
            preferred_endpoint=(
                args.endpoints.split(";", 1)[0].strip() if args.endpoints else None
            ),
        )
    ).resolve()
    if downloaded != destination:
        staging = destination.with_suffix(destination.suffix + ".partial")
        shutil.copy2(downloaded, staging)
        staging.replace(destination)
    size = destination.stat().st_size
    if size < args.minimum_bytes:
        raise RuntimeError(f"downloaded file is unexpectedly small: {destination} ({size} bytes)")
    if expected_hash and _sha256(destination) != expected_hash:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file failed SHA-256 verification: {destination}")
    print(f"{destination} ({size} bytes)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
