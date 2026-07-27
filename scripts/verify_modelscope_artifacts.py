"""Validate the release-owned ModelScope artifact contract without downloading files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "modelscope-artifacts.lock.json"
MIRRORS_PATH = ROOT / "mirrors.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _configured_paths(mirrors: dict[str, object]) -> set[str]:
    raw = mirrors.get("modelscope_artifacts")
    if not isinstance(raw, dict):
        raise ValueError("mirrors.json: modelscope_artifacts must be an object")
    result: set[str] = set()
    for values in raw.values():
        if not isinstance(values, list):
            raise ValueError("mirrors.json: every modelscope_artifacts value must be a list")
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError("mirrors.json: ModelScope artifact entries must be strings")
            if value.startswith("https://"):
                parsed = urlparse(value)
                marker = "/resolve/master/"
                if marker not in parsed.path:
                    continue
                value = parsed.path.split(marker, 1)[1]
            result.add(unquote(value).lstrip("/"))
    return result


def validate_contract(local_root: Path | None = None, *, require_all: bool = False) -> list[str]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    mirrors = json.loads(MIRRORS_PATH.read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported ModelScope artifact lock schema")
    if lock.get("release") != project["project"]["version"]:
        raise ValueError("artifact lock release does not match pyproject version")
    modelscope = mirrors.get("modelscope")
    if not isinstance(modelscope, dict) or modelscope.get("repository") != lock.get("repository"):
        raise ValueError("artifact lock repository does not match mirrors.json")
    policy = mirrors.get("download_policy")
    if not isinstance(policy, dict) or policy.get("allow_external") is not False:
        raise ValueError("release mirrors.json must disable automatic external downloads")

    configured = _configured_paths(mirrors)
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact lock must contain artifacts")
    ids: set[str] = set()
    paths: set[str] = set()
    missing: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"id", "path", "size", "sha256"}:
            raise ValueError("artifact entry fields are invalid")
        artifact_id = artifact["id"]
        relative = artifact["path"]
        size = artifact["size"]
        digest = artifact["sha256"]
        if not isinstance(artifact_id, str) or artifact_id in ids:
            raise ValueError(f"duplicate or invalid artifact id: {artifact_id!r}")
        if not isinstance(relative, str) or relative in paths or relative.startswith(("/", "\\")):
            raise ValueError(f"duplicate or invalid artifact path: {relative!r}")
        if ".." in Path(relative).parts or relative not in configured:
            raise ValueError(f"artifact path is not configured in mirrors.json: {relative}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"invalid artifact size: {artifact_id}")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid artifact SHA-256: {artifact_id}")
        ids.add(artifact_id)
        paths.add(relative)
        if local_root is None:
            continue
        candidate = local_root / Path(relative)
        if not candidate.is_file():
            missing.append(relative)
            continue
        if candidate.stat().st_size != size or _sha256(candidate) != digest:
            raise ValueError(f"local artifact failed size/SHA-256 verification: {candidate}")

    sidecars = lock.get("sidecar_verified_archives")
    if not isinstance(sidecars, list) or not all(isinstance(item, str) for item in sidecars):
        raise ValueError("sidecar_verified_archives must be a string list")
    for relative in sidecars:
        if relative not in configured or f"{relative}.sha256" not in configured:
            raise ValueError(f"sidecar archive or checksum is not configured: {relative}")
    if require_all and missing:
        raise ValueError("local mirror is missing artifacts: " + ", ".join(missing))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    try:
        missing = validate_contract(args.local_root, require_all=args.require_all)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"ModelScope artifact contract FAILED: {exc}")
        return 1
    if args.local_root is not None and missing:
        print(
            f"ModelScope artifact contract OK; local mirror is missing {len(missing)} fixed files."
        )
    else:
        print("ModelScope artifact contract OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
