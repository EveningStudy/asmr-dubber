from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .constants import PROJECT_ROOT

MIRROR_CONFIG_PATH = PROJECT_ROOT / "mirrors.json"

_OFFICIAL_FALLBACKS = {
    "pypi_indexes": ("https://pypi.org/simple",),
    "huggingface_endpoints": ("https://huggingface.co",),
    "pytorch_indexes": ("https://download.pytorch.org/whl/cu130",),
    "github_proxy_prefixes": ("",),
}


def _valid_https_url(value: str, *, allow_empty: bool = False) -> bool:
    if allow_empty and not value:
        return True
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def load_mirror_config(path: Path = MIRROR_CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取镜像配置 {path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"镜像配置必须是 JSON 对象：{path}")
    return payload


def mirror_candidates(
    name: str,
    *,
    preferred: str | None = None,
    path: Path = MIRROR_CONFIG_PATH,
) -> tuple[str, ...]:
    payload = load_mirror_config(path)
    values: list[str] = []
    if preferred and _valid_https_url(preferred.strip()):
        values.append(preferred.strip().rstrip("/"))
    configured = payload.get(name, [])
    if not isinstance(configured, list):
        raise ValueError(f"mirrors.json 中的 {name} 必须是数组")
    allow_empty = name == "github_proxy_prefixes"
    for raw in [*configured, *_OFFICIAL_FALLBACKS.get(name, ())]:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if name != "github_proxy_prefixes":
            value = value.rstrip("/")
        if not _valid_https_url(value, allow_empty=allow_empty):
            continue
        if value not in values:
            values.append(value)
    return tuple(values)


def github_url_candidates(url: str) -> tuple[str, ...]:
    if not url.startswith("https://github.com/"):
        return (url,)
    candidates = [
        f"{prefix.rstrip('/')}/{url}" if prefix else url
        for prefix in mirror_candidates("github_proxy_prefixes")
    ]
    return tuple(dict.fromkeys(candidates))


def huggingface_endpoints(preferred: str | None = None) -> Iterable[str]:
    configured = os.getenv("ASMR_DUBBER_HF_ENDPOINTS", "")
    values = [item.strip().rstrip("/") for item in configured.split(";") if item.strip()]
    if preferred:
        values.insert(0, preferred.strip().rstrip("/"))
    values.extend(mirror_candidates("huggingface_endpoints"))
    return tuple(dict.fromkeys(value for value in values if _valid_https_url(value)))


def hf_hub_download_with_fallback(
    *,
    preferred_endpoint: str | None = None,
    **kwargs: Any,
) -> str:
    from huggingface_hub import hf_hub_download

    failures: list[str] = []
    for endpoint in huggingface_endpoints(preferred_endpoint):
        try:
            return hf_hub_download(endpoint=endpoint, **kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve provider failures for fallback
            failures.append(f"{endpoint}: {exc}")
    raise RuntimeError("所有 Hugging Face 下载源均失败：" + "；".join(failures))


def snapshot_download_with_fallback(
    *,
    preferred_endpoint: str | None = None,
    **kwargs: Any,
) -> str:
    from huggingface_hub import snapshot_download

    failures: list[str] = []
    for endpoint in huggingface_endpoints(preferred_endpoint):
        try:
            return snapshot_download(endpoint=endpoint, **kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve provider failures for fallback
            failures.append(f"{endpoint}: {exc}")
    raise RuntimeError("所有 Hugging Face 下载源均失败：" + "；".join(failures))
