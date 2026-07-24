"""Download one pinned Hugging Face file into a portable model directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from asmr_dubber.mirrors import hf_hub_download_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--minimum-bytes", type=int, default=1)
    parser.add_argument("--endpoints", default="")
    args = parser.parse_args()

    destination = Path(args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        downloaded.replace(destination)
    size = destination.stat().st_size
    if size < args.minimum_bytes:
        raise RuntimeError(f"downloaded file is unexpectedly small: {destination} ({size} bytes)")
    print(f"{destination} ({size} bytes)")


if __name__ == "__main__":
    main()
