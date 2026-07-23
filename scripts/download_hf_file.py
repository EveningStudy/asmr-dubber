"""Download one pinned Hugging Face file into a portable model directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--minimum-bytes", type=int, default=1)
    args = parser.parse_args()

    destination = Path(args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            revision=args.revision,
            local_dir=destination.parent,
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
