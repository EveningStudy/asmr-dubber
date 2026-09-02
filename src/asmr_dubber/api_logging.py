from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

logger = logging.getLogger("asmr_dubber.api")


def safe_api_url(value: object) -> str:
    """Return a log-safe endpoint without credentials, query values, or fragments."""

    text = str(value or "").strip()
    if not text:
        return "未设置"
    try:
        parsed = urlsplit(text)
    except ValueError:
        return "无效地址"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "无效地址"
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        return "无效地址"
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/") or "/", "", ""))


def _request_hook(service: str) -> Callable[[httpx.Request], None]:
    def log_request(request: httpx.Request) -> None:
        request.extensions["asmr_dubber_started_at"] = time.perf_counter()
        logger.info(
            "API 请求开始：服务=%s 方法=%s 地址=%s",
            service,
            request.method,
            safe_api_url(request.url),
        )

    return log_request


def _response_hook(service: str) -> Callable[[httpx.Response], None]:
    def log_response(response: httpx.Response) -> None:
        started = response.request.extensions.get("asmr_dubber_started_at")
        elapsed_ms = (
            round((time.perf_counter() - float(started)) * 1000)
            if isinstance(started, int | float)
            else None
        )
        request_id = next(
            (
                response.headers.get(name, "").strip()
                for name in ("x-request-id", "request-id", "x-trace-id", "trace-id")
                if response.headers.get(name, "").strip()
            ),
            "无",
        )
        logger.info(
            "API 请求完成：服务=%s 状态=%s 耗时=%s ms 请求ID=%s 地址=%s",
            service,
            response.status_code,
            elapsed_ms if elapsed_ms is not None else "未知",
            request_id,
            safe_api_url(response.request.url),
        )

    return log_response


def logged_http_client(service: str, **kwargs: Any) -> httpx.Client:
    """Build an HTTP client that records safe request metadata in the application log."""

    hooks = dict(kwargs.pop("event_hooks", {}) or {})
    hooks["request"] = [*hooks.get("request", []), _request_hook(service)]
    hooks["response"] = [*hooks.get("response", []), _response_hook(service)]
    return httpx.Client(event_hooks=hooks, **kwargs)
