"""Standalone downloads shared by the isolated separation worker and tests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote, urlparse

DEFAULT_BASE = "https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Separation/resolve/master"
_MODELSCOPE_HOSTS = frozenset({"modelscope.cn", "modelscope.ai"})
_OVERSEAS_HOSTS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "hf-mirror.com",
        "dl.fbaipublicfiles.com",
    }
)


class _BadChecksumError(RuntimeError):
    pass


def _is_modelscope_host(host: str) -> bool:
    return host in _MODELSCOPE_HOSTS or any(host.endswith("." + h) for h in _MODELSCOPE_HOSTS)


def _fallback_url_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if _is_modelscope_host(host):
        return True
    try:
        from .mirrors import download_url_allowed, external_downloads_allowed

        if not download_url_allowed(url):
            return False
        return external_downloads_allowed() or not any(
            host == h or host.endswith("." + h) for h in _OVERSEAS_HOSTS
        )
    except ImportError:
        flag = os.getenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            return True
        return not any(host == h or host.endswith("." + h) for h in _OVERSEAS_HOSTS)


def _download_headers(url: str, offset: int = 0) -> dict[str, str]:
    headers = {"Accept-Encoding": "identity"}
    host = (urlparse(url).hostname or "").lower()
    if _is_modelscope_host(host):
        headers["User-Agent"] = "curl/8.0"
        headers["Referer"] = "https://modelscope.cn/"
        token = os.getenv("MODELSCOPE_API_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["Cookie"] = f"m_session_id={token}"
    else:
        headers["User-Agent"] = "ASMR-Dubber/0.4"
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    return headers


def source_manifest():
    return json.loads(
        Path(__file__).with_name("separation_sources.json").read_text(encoding="utf-8")
    )


def download_model_file(
    url, output_path, *, models, source="modelscope", base_url=DEFAULT_BASE, report=print, get=None
):
    if get is None:
        import requests

        get = requests.get
    target = Path(output_path)
    if not target.resolve().is_relative_to(Path(models).resolve()):
        raise RuntimeError("模型清单包含目录外路径。")
    record = source_manifest()["files"].get(target.name)
    expected = record["sha256"] if record else None
    if target.is_file():
        if not expected or _hash(target) == expected:
            report(f"已下载：{target.name}（本地文件校验通过）")
            return
        report(f"文件损坏，重新下载：{target.name}")
    candidates = []
    if record and source == "modelscope":
        candidates.append(base_url.rstrip("/") + "/" + quote(target.name))
    candidates.extend(
        extra
        for extra in ([record["upstream"]] if record else []) + [url]
        if source != "modelscope" or _fallback_url_allowed(extra)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    errors = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.startswith("https://"):
            raise RuntimeError("模型下载必须使用 HTTPS。")
        provider = (
            "ModelScope"
            if _is_modelscope_host((urlparse(candidate).hostname or "").lower())
            else "原始来源"
        )
        report(f"正在下载：{target.name} · {provider}")
        try:
            existing = partial.stat().st_size if partial.is_file() else 0
            with get(
                candidate,
                headers=_download_headers(candidate, existing),
                stream=True,
                timeout=60,
            ) as response:
                status = getattr(response, "status_code", 200)
                if status == 416 and existing > 0:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("断点续传范围无效")
                response.raise_for_status()
                resuming = status == 206 and existing > 0
                content_length = int(response.headers.get("Content-Length", 0))
                total = (
                    (existing + content_length) if (resuming and content_length) else content_length
                )
                done = existing if resuming else 0
                last = 0.0
                digest = hashlib.sha256()
                if resuming:
                    with partial.open("rb") as prev:
                        while chunk := prev.read(1024 * 1024):
                            digest.update(chunk)
                with partial.open("ab" if resuming else "wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        output.write(block)
                        digest.update(block)
                        done += len(block)
                        now = time.monotonic()
                        if now - last >= 1:
                            suffix = (
                                f" / {total / 1048576:.1f} MB"
                                f"（{min(done / total * 100, 100):.1f}%）"
                                if total
                                else "（总大小未知）"
                            )
                            report(f"下载进度：{target.name} · {done / 1048576:.1f} MB{suffix}")
                            last = now
                if total and done != total:
                    raise RuntimeError("下载长度不符")
                if expected and digest.hexdigest() != expected:
                    raise _BadChecksumError("SHA256 不符")
            os.replace(partial, target)
            report(f"下载完成：{target.name} · {done / 1048576:.1f} MB")
            return
        except _BadChecksumError as exc:
            partial.unlink(missing_ok=True)
            errors.append(f"{provider}：{type(exc).__name__}")
            report(f"{provider}校验失败，已清理损坏片段。")
        except Exception as exc:
            errors.append(f"{provider}：{type(exc).__name__}")
            report(f"{provider}下载失败，保留断点并尝试下一个来源。")
    raise RuntimeError(f"下载失败：{target.name}；" + "；".join(errors))


def _hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
