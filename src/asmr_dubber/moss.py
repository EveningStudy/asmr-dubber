"""Pinned MOSS assets and an isolated Transformers 5 runtime.

The mirror is tried first for every file. Only this model opts into upstream
fallback regardless of the application's older ModelScope-only policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from .errors import AsmrDubberError, InstallPausedError
from .mirrors import load_mirror_config
from .platforms import portable_home, virtualenv_executable

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def manifest() -> dict:
    return json.loads(Path(__file__).with_name("moss_manifest.json").read_text(encoding="utf-8"))


def runtime_root() -> Path:
    return portable_home() / "runtimes" / "moss-transcribe-diarize"


def runtime_python() -> Path:
    return virtualenv_executable(runtime_root() / ".venv", "python")


def model_directory() -> Path:
    return portable_home() / "models" / "moss-transcribe-diarize" / manifest()["revision"]


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def complete_model(*, verify: bool = False) -> bool:
    root = model_directory()
    for name, record in manifest()["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["size"]:
            return False
        if verify and file_digest(path) != record["sha256"]:
            return False
    return True


def download_urls(name: str) -> list[str]:
    revision = manifest()["revision"]
    config = load_mirror_config()
    base = config["modelscope"]["base_url"].rstrip("/")
    upstream = (
        f"https://raw.githubusercontent.com/OpenMOSS/MOSS-Transcribe-Diarize/{manifest()['source_revision']}/LICENSE"
        if name == "LICENSE"
        else f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}"
    )
    return [f"{base}/artifacts/models/moss-transcribe-diarize/{revision}/{name}", upstream]


def download_model(*, log: Callable = print, cancelled: Callable = lambda: False) -> Path:
    """Resume interrupted files, check every hash, and publish by atomic rename."""
    root = model_directory()
    root.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=30)) as client:
        for name, record in manifest()["files"].items():
            if cancelled():
                raise InstallPausedError("MOSS 下载已暂停，再次下载可继续。")
            path = root / name
            if (
                path.is_file()
                and path.stat().st_size == record["size"]
                and file_digest(path) == record["sha256"]
            ):
                continue
            partial = path.with_name(path.name + ".partial")
            errors = []
            for url in download_urls(name):
                for _attempt in range(2):
                    if cancelled():
                        raise InstallPausedError("MOSS 下载已暂停，再次下载可继续。")
                    provider = "ModelScope" if "modelscope" in url else "Upstream"
                    log(f"MOSS 下载 {name}：{provider}")
                    try:
                        offset = partial.stat().st_size if partial.exists() else 0
                        if offset >= record["size"]:
                            if (
                                offset == record["size"]
                                and file_digest(partial) == record["sha256"]
                            ):
                                os.replace(partial, path)
                                break
                            partial.unlink()
                            offset = 0
                        with client.stream(
                            "GET", url, headers={"Range": f"bytes={offset}-"}
                        ) as response:
                            response.raise_for_status()
                            resumed = response.status_code == 206 and offset > 0
                            if response.status_code == 206:
                                start = offset if resumed else 0
                                if not response.headers.get("content-range", "").startswith(
                                    f"bytes {start}-"
                                ):
                                    raise ValueError("下载服务器返回了错误的续传范围")
                            size = offset if resumed else 0
                            last_report = time.monotonic()
                            with partial.open("ab" if resumed else "wb") as handle:
                                for chunk in response.iter_bytes(1024 * 1024):
                                    if cancelled():
                                        raise InstallPausedError(
                                            "MOSS 下载已暂停，再次下载可继续。"
                                        )
                                    size += len(chunk)
                                    if size > record["size"]:
                                        raise ValueError("模型文件超过固定清单的大小")
                                    handle.write(chunk)
                                    if time.monotonic() - last_report > 2:
                                        log(f"MOSS 下载 {name} {100 * size / record['size']:.1f}%")
                                        last_report = time.monotonic()
                        if (
                            partial.stat().st_size != record["size"]
                            or file_digest(partial) != record["sha256"]
                        ):
                            partial.unlink()
                            raise ValueError("模型文件大小或 SHA-256 不匹配")
                        os.replace(partial, path)
                        break
                    except (httpx.HTTPError, OSError, ValueError) as exc:
                        errors.append(f"{url}: {type(exc).__name__}: {exc}")
                else:
                    continue
                break
            else:
                raise AsmrDubberError("MOSS 所有下载源均失败：" + "；".join(errors))
    return root


def install_runtime(
    uv: Path, run: Callable, *, env: dict, log: Callable, cancelled: Callable
) -> str:
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    commands = []
    if not runtime_python().is_file():
        commands.append([str(uv), "venv", str(root / ".venv"), "--python", sys.executable])
    commands.append(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(runtime_python()),
            "--torch-backend",
            "auto",
            "--index-strategy",
            "unsafe-best-match",
            "--index",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--default-index",
            "https://pypi.org/simple",
            "torch==2.11.0",
            "transformers==5.6.0",
            "numpy==2.3.5",
            "soundfile>=0.13,<1",
            "soxr>=0.5,<1",
            "librosa>=0.11,<1",
        ]
    )
    for command in commands:
        result = run(command, env=env)
        if result.returncode:
            raise AsmrDubberError("MOSS 独立环境安装失败：\n" + result.stdout[-4000:])
    check = run(
        [
            str(runtime_python()),
            "-c",
            (
                "import torch,transformers,librosa,soundfile,soxr; "
                "print(torch.__version__, transformers.__version__)"
            ),
        ],
        env=env,
    )
    if check.returncode:
        raise AsmrDubberError("MOSS 运行依赖检测失败：\n" + check.stdout[-4000:])
    download_model(log=log, cancelled=cancelled)
    (root / "ready.json").write_text(
        json.dumps({"revision": manifest()["revision"]}), encoding="utf-8"
    )
    return "MOSS 独立运行环境与固定模型快照已安装。"
