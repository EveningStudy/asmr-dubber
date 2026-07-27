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

# Downloads are deliberately classified here instead of being classified in
# each installer.  This makes a new installer inherit the same policy and
# prevents an accidental ``github.com`` fallback from reappearing in a later
# release.  ModelScope and common mainland mirrors are allowed by default;
# overseas providers require an explicit opt-in.
_MODELSCOPE_HOSTS = frozenset({"modelscope.cn", "modelscope.ai"})
_EXTERNAL_HOSTS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "hf-mirror.com",
        "ghfast.top",
        "ghproxy.net",
        "download.pytorch.org",
        "pypi.org",
        "astral.sh",
        "releases.astral.sh",
        "python.org",
        "www.python.org",
    }
)


def _flag(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def external_downloads_allowed(path: Path = MIRROR_CONFIG_PATH) -> bool:
    """Return whether overseas download providers were explicitly enabled.

    The environment variable is intentionally an override so a user can
    temporarily recover an installation without editing a release file.  It
    is never set by the application itself.
    """

    raw = os.getenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS")
    if raw is not None:
        return _flag(raw)
    try:
        payload = load_mirror_config(path)
    except ValueError:
        return False
    policy = payload.get("download_policy")
    # A hand-written/legacy mirror file predates the policy block.  Keep its
    # historical behaviour; release mirrors.json carries an explicit false.
    if policy is None:
        return True
    return isinstance(policy, dict) and _flag(policy.get("allow_external"), False)


def _is_modelscope_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host in _MODELSCOPE_HOSTS or any(host.endswith("." + item) for item in _MODELSCOPE_HOSTS)


def _is_external_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host in _EXTERNAL_HOSTS or any(host.endswith("." + item) for item in _EXTERNAL_HOSTS)


def download_url_allowed(value: str, *, path: Path = MIRROR_CONFIG_PATH) -> bool:
    """Return whether a validated URL is permitted by the release policy."""

    return _valid_https_url(value) and (
        external_downloads_allowed(path) or not _is_external_url(value) or _is_modelscope_url(value)
    )


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


def modelscope_artifact_urls(
    name: str,
    *,
    path: Path = MIRROR_CONFIG_PATH,
) -> tuple[str, ...]:
    """Expand the release-owned ModelScope artifact aliases for ``name``.

    ``mirrors.json`` accepts either complete HTTPS URLs or paths relative to
    ``modelscope.base_url``.  Keeping the alias table in one file means a
    future release can change an artifact without changing every platform
    installer.
    """

    payload = load_mirror_config(path)
    configured = payload.get("modelscope_artifacts", {})
    if configured is None:
        return ()
    if not isinstance(configured, dict):
        raise ValueError("mirrors.json 中的 modelscope_artifacts 必须是对象")
    values = configured.get(name, [])
    if not isinstance(values, list):
        raise ValueError(f"mirrors.json 中的 modelscope_artifacts.{name} 必须是数组")
    modelscope = payload.get("modelscope", {})
    base = ""
    if isinstance(modelscope, dict):
        raw_base = modelscope.get("base_url", "")
        if isinstance(raw_base, str):
            base = raw_base.strip().rstrip("/")
    result: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if value and not value.startswith("https://"):
            if not base:
                raise ValueError(
                    f"mirrors.json 中的 modelscope_artifacts.{name} 使用相对路径，"
                    "但没有配置 modelscope.base_url"
                )
            value = f"{base}/{value.lstrip('/')}"
        if not _valid_https_url(value) or not _is_modelscope_url(value):
            raise ValueError(f"modelscope_artifacts.{name} 包含无效的 ModelScope URL")
        if value not in result:
            result.append(value)
    return tuple(result)


def download_candidates(
    name: str,
    *,
    preferred: str | None = None,
    path: Path = MIRROR_CONFIG_PATH,
    allow_external: bool | None = None,
) -> tuple[str, ...]:
    """Return policy-filtered sources with ModelScope candidates first.

    Regional mirrors remain available by default.  GitHub/Hugging Face/
    official PyPI and similar overseas hosts are omitted unless
    ``ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1`` (or the release configuration)
    explicitly opts in.
    """

    payload = load_mirror_config(path)
    configured = payload.get(name, [])
    if not isinstance(configured, list):
        raise ValueError(f"mirrors.json 中的 {name} 必须是数组")
    values: list[str] = [*modelscope_artifact_urls(name, path=path)]
    if preferred and _valid_https_url(preferred.strip()):
        values.insert(0, preferred.strip().rstrip("/"))
    values.extend(str(item).strip() for item in configured if isinstance(item, str))
    # Keep the official fallbacks for compatibility, but run them through the
    # same policy filter below.
    values.extend(_OFFICIAL_FALLBACKS.get(name, ()))
    if allow_external is None:
        allow_external = external_downloads_allowed(path)
    result: list[str] = []
    for value in values:
        if not value:
            if name == "github_proxy_prefixes" and allow_external:
                value = ""
            else:
                continue
        if name != "github_proxy_prefixes":
            value = value.rstrip("/")
        if not _valid_https_url(value, allow_empty=name == "github_proxy_prefixes"):
            continue
        if not allow_external and _is_external_url(value) and not _is_modelscope_url(value):
            continue
        if value not in result:
            result.append(value)
    return tuple(result)


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


def github_url_candidates(
    url: str,
    *,
    allow_external: bool | None = None,
) -> tuple[str, ...]:
    if not url.startswith("https://github.com/"):
        return (url,)
    if allow_external is None:
        # This helper historically returned GitHub proxy/direct candidates and
        # is also used by third-party scripts.  New installers pass the secure
        # policy explicitly through ``download_candidates``; preserve the
        # compatibility default here.
        allow_external = True
    if not allow_external:
        return ()
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
    elif os.getenv("HF_ENDPOINT", "").strip():
        values.insert(0, os.getenv("HF_ENDPOINT", "").strip().rstrip("/"))
    values.extend(mirror_candidates("huggingface_endpoints"))
    allowed = external_downloads_allowed()
    return tuple(
        dict.fromkeys(
            value
            for value in values
            if _valid_https_url(value)
            and (allowed or not _is_external_url(value) or _is_modelscope_url(value))
        )
    )


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
        except Exception as exc:
            failures.append(f"{endpoint}: {exc}")
    if not failures and not external_downloads_allowed():
        raise RuntimeError(
            "未启用海外模型源，且没有可用的本地/ModelScope 模型包；"
            "请先上传并配置 ModelScope 模型包，或显式设置 "
            "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1。"
        )
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
        except Exception as exc:
            failures.append(f"{endpoint}: {exc}")
    if not failures and not external_downloads_allowed():
        raise RuntimeError(
            "未启用海外模型源，且没有可用的本地/ModelScope 模型包；"
            "请先上传并配置 ModelScope 模型包，或显式设置 "
            "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1。"
        )
    raise RuntimeError("所有 Hugging Face 下载源均失败：" + "；".join(failures))
