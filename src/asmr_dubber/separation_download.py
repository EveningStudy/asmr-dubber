"""Standalone downloads shared by the isolated separation worker and tests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote, urlparse

DEFAULT_BASE = "https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Separation/resolve/master"


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
    if record:
        candidates.append(record["upstream"])
    candidates.append(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    errors = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.startswith("https://"):
            raise RuntimeError("模型下载必须使用 HTTPS。")
        provider = (
            "ModelScope" if "modelscope." in (urlparse(candidate).hostname or "") else "原始来源"
        )
        report(f"正在下载：{target.name} · {provider}")
        try:
            with get(candidate, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                done, last = 0, 0.0
                digest = hashlib.sha256()
                with partial.open("wb") as output:
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
                    raise RuntimeError("SHA256 不符")
            os.replace(partial, target)
            report(f"下载完成：{target.name} · {done / 1048576:.1f} MB")
            return
        except Exception as exc:
            errors.append(f"{provider}：{type(exc).__name__}")
            report(f"{provider}下载失败，尝试下一个来源。")
        finally:
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"下载失败：{target.name}；" + "；".join(errors))


def _hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
