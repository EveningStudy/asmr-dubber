from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class APIContractError(ValueError):
    """Raised when a configurable HTTP API does not follow its selected contract."""


def parse_json_object(value: str, *, label: str) -> dict[str, Any]:
    raw = value.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIContractError(
            f"{label}不是有效 JSON：第 {exc.lineno} 行第 {exc.colno} 列，{exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise APIContractError(f'{label}必须是一个 JSON 对象，例如 {{"temperature": 0}}。')
    return parsed


def merge_extra_body(
    payload: dict[str, Any],
    extra_body: str,
    *,
    label: str,
    reserved: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    extra = parse_json_object(extra_body, label=label)
    conflicts = sorted(reserved.intersection(extra))
    if conflicts:
        raise APIContractError(f"{label}不能覆盖程序必需字段：{', '.join(conflicts)}")
    return {**payload, **extra}


def endpoint_url(base_url: str, endpoint: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise APIContractError("API 基础地址不能为空。")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise APIContractError(f"API 基础地址无效：{base_url}")
    suffix = "/" + endpoint.strip().lstrip("/")
    if base.endswith(suffix):
        return base
    return base + suffix


def bearer_headers(api_key: str) -> dict[str, str]:
    key = api_key.strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _nested_value(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


_AUDIO_BASE64_PATHS = (
    "audio",
    "audio_base64",
    "data.audio",
    "data.audio_base64",
    "output.audio",
    "output.audio_base64",
)
_AUDIO_URL_PATHS = ("url", "audio_url", "data.url", "data.audio_url", "output.url")


def write_audio_response(
    response: httpx.Response,
    output: Path,
    *,
    client: httpx.Client,
) -> None:
    if response.is_error:
        raise APIContractError(f"HTTP {response.status_code}：{response.text[:500]}")
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type.startswith("audio/") or content_type == "application/octet-stream":
        if not response.content:
            raise APIContractError("API 返回了空音频。")
        output.write_bytes(response.content)
        return
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise APIContractError(
            f"API 没有返回音频或可识别的 JSON（Content-Type: {content_type or '未知'}）。"
        ) from exc
    if not isinstance(payload, Mapping):
        raise APIContractError("API JSON 响应必须是对象。")
    for path in _AUDIO_BASE64_PATHS:
        encoded = _nested_value(payload, path)
        if not isinstance(encoded, str) or not encoded.strip():
            continue
        raw = encoded.strip()
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            audio = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise APIContractError(f"API 响应字段 {path} 不是有效的 Base64 音频。") from exc
        if not audio:
            raise APIContractError(f"API 响应字段 {path} 是空音频。")
        output.write_bytes(audio)
        return
    for path in _AUDIO_URL_PATHS:
        url = _nested_value(payload, path)
        if not isinstance(url, str) or not url.strip():
            continue
        download = client.get(url.strip())
        if download.is_error or not download.content:
            raise APIContractError(f"下载 API 返回的音频地址失败（HTTP {download.status_code}）。")
        output.write_bytes(download.content)
        return
    raise APIContractError(
        "API JSON 响应中没有音频；支持 audio/audio_base64、data.audio、output.audio "
        "或 audio_url/url。"
    )
