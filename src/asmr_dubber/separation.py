"""Opt-in separation, with bounded chunks, explicit downloads and per-project cache."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import soundfile as sf
import soxr

from .api_contracts import parse_json_object
from .audio import _run_ffmpeg, sha256_file
from .environment import ffmpeg_executable
from .errors import ProjectError
from .models import DubProject, ProjectSettings
from .platforms import isolated_runtime_environment, portable_home, virtualenv_executable
from .runtime_manager import _run_streaming_process
from .storage import atomic_write_text, exclusive_file_lock, require_disk_space
from .task_control import check_cancelled, current_cancellation, terminate_process_tree

SEPARATOR_VERSION = "0.47.0"
DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"
logger = logging.getLogger(__name__)


class SeparationSession:
    """One lazy worker per separation pass; each request retains its own timeout."""

    def __init__(self, root: Path):
        self.root = root
        self.process = None
        self.log = None
        self.temporary = None

    def __enter__(self):
        return self

    def separate(self, job: dict, timeout: float) -> None:
        check_cancelled()
        if self.process is None:
            self.temporary = tempfile.TemporaryDirectory(
                dir=self.root, prefix="session-", ignore_cleanup_errors=True
            )
            folder = Path(self.temporary.name)
            self.request = folder / "request.json"
            config = {**job, "action": "serve", "request": str(self.request)}
            config_path = folder / "job.json"
            atomic_write_text(config_path, json.dumps(config))
            self.log = (folder / "worker.log").open("w+b")
            self.process = subprocess.Popen(
                [
                    str(runtime_python()),
                    str(Path(__file__).with_name("separation_worker.py")),
                    str(config_path),
                ],
                cwd=portable_home(),
                env=worker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
        atomic_write_text(self.request, json.dumps(job))
        deadline = time.monotonic() + timeout
        while not Path(job["result"]).is_file():
            check_cancelled()
            if self.process.poll() is not None:
                assert self.log is not None
                self.log.seek(0, os.SEEK_END)
                self.log.seek(max(0, self.log.tell() - 6000))
                detail = self.log.read().decode("utf-8", errors="replace")
                raise ProjectError(f"分离子进程失败（退出码 {self.process.returncode}）：{detail}")
            if time.monotonic() >= deadline:
                raise ProjectError("分离片段处理超时；已完成的分块保留，可继续重试。")
            time.sleep(0.05)

    def __exit__(self, exc_type, *_):
        if self.process is not None:
            if exc_type is None and self.process.poll() is None:
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                    atomic_write_text(self.request, json.dumps({"stop": True}))
                    self.process.wait(timeout=5)
            terminate_process_tree(self.process)
        if self.log is not None:
            self.log.close()
        if self.temporary is not None:
            # Windows scanners/runtime teardown can briefly retain the log handle.
            # A disposable log must not turn successfully separated audio into failure.
            for _ in range(10):
                self.temporary.cleanup()
                if not Path(self.temporary.name).exists():
                    break
                time.sleep(0.1)
            else:
                logger.warning("分离日志暂时被占用，保留临时目录：%s", self.temporary.name)


def runtime_python() -> Path:
    return virtualenv_executable(portable_home() / "runtimes/separation/.venv", "python")


def model_directory() -> Path:
    return portable_home() / "models/separation"


def local_model_name(value: str) -> str:
    if not value or Path(value).name != value or "/" in value or "\\" in value or ":" in value:
        raise ProjectError("分离模型请选择文件名，不接受路径；模型放在 models/separation。")
    return value


def worker_environment() -> dict[str, str]:
    env = isolated_runtime_environment("separation")
    cache = portable_home() / "cache/separation"
    temp = portable_home() / "temp/separation"
    temp.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "TEMP": str(temp),
            "TMP": str(temp),
            "TMPDIR": str(temp),
            "UV_CACHE_DIR": str(portable_home() / "cache/uv"),
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
            "NUMBA_CACHE_DIR": str(cache / "numba"),
        }
    )
    env["PATH"] = str(Path(ffmpeg_executable()).parent) + os.pathsep + env.get("PATH", "")
    bin_dir = portable_home() / "runtimes/separation/bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not ffmpeg.is_file():
        shutil.copy2(ffmpeg_executable(), ffmpeg)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    return env


def _run(command: list[str], timeout: float, log=None) -> None:
    result = _run_streaming_process(
        command,
        cwd=portable_home(),
        env=worker_environment(),
        timeout_seconds=timeout,
        log_callback=log,
        cancel_event=current_cancellation(),
    )
    if result.returncode:
        raise ProjectError(f"分离子进程失败（退出码 {result.returncode}）：{result.stdout[-3000:]}")


def prepare_local_model(model: str = DEFAULT_MODEL, *, install: bool = False, log=None) -> str:
    """Explicit user action only. Never called automatically by inference."""
    model = local_model_name(model)
    with exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=1):
        python = runtime_python()
        if install:
            uv = shutil.which("uv") or str(portable_home() / "bootstrap/windows/uv/uv.exe")
            if not python.is_file():
                _run(
                    [
                        uv,
                        "venv",
                        "--python",
                        sys.executable,
                        str(python.parent.parent),
                    ],
                    300,
                    log,
                )
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    f"audio-separator[cpu]=={SEPARATOR_VERSION}",
                    "torch==2.11.0",
                    "torchaudio==2.11.0",
                    "librosa==0.11.0",
                    "audioread==3.0.1",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu130",
                    "--index-strategy",
                    "unsafe-best-match",
                ],
                7200,
                log,
            )
        if not python.is_file():
            raise ProjectError("独立分离环境未安装，请先点击安装环境。")
        with tempfile.TemporaryDirectory(
            dir=portable_home() / "temp", prefix="separation-install-"
        ) as tmp:
            job = Path(tmp) / "job.json"
            atomic_write_text(
                job,
                json.dumps(
                    {
                        "action": "download",
                        "models": str(model_directory()),
                        "model": model,
                        "output": tmp,
                        "device": "cpu",
                    }
                ),
            )
            _run(
                [str(python), str(Path(__file__).with_name("separation_worker.py")), str(job)],
                7200,
                log,
            )
        return f"已安装/校验 {model}；分离和 RTF 仍默认关闭。"


def verify_local_model(model: str) -> str:
    model = local_model_name(model)
    if not runtime_python().is_file():
        raise ProjectError("请先安装独立分离环境。")
    root = model_directory()
    manifest = root / f"{model}.integrity.json"
    if not manifest.is_file():
        raise ProjectError("分离模型未校验，请先在设置中下载/校验所选模型。")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if model not in records:
        raise ProjectError("分离模型校验清单不完整。")
    for name, digest in records.items():
        local_model_name(name)
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ProjectError(f"分离文件缺失或损坏：{name}，请重新下载。")
    return records[model]


def _result_field(payload: Any, field: str) -> Any:
    for part in field.split("."):
        if isinstance(payload, dict):
            payload = payload.get(part)
        elif isinstance(payload, list) and part.isdecimal() and int(part) < len(payload):
            payload = payload[int(part)]
        else:
            return None
    return payload


def _download_audio(url: str, destination: Path, maximum: int = 256 * 1024**2) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ProjectError("云端人声输出必须是无凭证的 HTTPS 下载地址。")
    # No API credentials or cookies are forwarded to returned URLs.
    with (
        httpx.Client(timeout=60, follow_redirects=False) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        total = 0
        with destination.open("wb") as out:
            for block in response.iter_bytes(1024 * 1024):
                check_cancelled()
                total += len(block)
                if total > maximum:
                    raise ProjectError("分离响应超过 256 MiB，已停止。")
                out.write(block)


def _cloud_separate(audio: Path, destination: Path, settings: ProjectSettings) -> Path:
    from .user_settings import saved_service_key

    if not settings.separation_cloud_consent:
        raise ProjectError("云端分离会上传音频并可能计费，需在设置中明确同意。")
    key = saved_service_key("separation:" + settings.separation_backend)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    params = parse_json_object(settings.separation_api_params, label="云端分离参数")
    field = settings.separation_api_audio_field.strip()
    if not field or field in params:
        raise ProjectError("音频字段为空或被附加参数覆盖。")
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        if settings.separation_backend == "replicate":
            if not key or not re.fullmatch(r"[0-9a-f]{64}", settings.separation_api_model):
                raise ProjectError("Replicate 需要 API Key 和固定 64 位版本 ID。")
            if audio.stat().st_size > 8 * 1024**2:
                raise ProjectError("Replicate 内联音频限制为 8 MiB，请将分块秒数降低到 30 或以下。")
            params[field] = "data:audio/wav;base64," + base64.b64encode(audio.read_bytes()).decode()
            base = "https://api.replicate.com/v1/predictions"
            response = client.post(
                base,
                headers=headers,
                json={"version": settings.separation_api_model, "input": params},
            )
            response.raise_for_status()
            record = response.json()
            prediction = str(record.get("id", ""))
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", prediction):
                raise ProjectError("Replicate 未返回有效任务 ID，不自动重发计费请求。")
            deadline = time.monotonic() + settings.separation_timeout_seconds
            try:
                while record.get("status") not in {"succeeded", "failed", "canceled"}:
                    check_cancelled()
                    if time.monotonic() > deadline:
                        raise ProjectError(
                            "Replicate 分离超时，已请求取消；请在服务商控制台确认计费状态。"
                        )
                    time.sleep(1)
                    response = client.get(f"{base}/{prediction}", headers=headers)
                    response.raise_for_status()
                    record = response.json()
            except BaseException:
                with contextlib.suppress(httpx.HTTPError):
                    client.post(f"{base}/{prediction}/cancel", headers=headers)
                raise
            if record.get("status") != "succeeded":
                raise ProjectError("Replicate 分离未成功，请在服务商控制台查看任务。")
            payload = record.get("output")
        else:
            url = settings.separation_api_url.strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
                raise ProjectError("请填写有效的自建分离 API 地址。")
            if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ProjectError("非本机分离接口必须使用 HTTPS。")
            if field in {"model", "parameters"}:
                raise ProjectError("上传文件字段不能是 model 或 parameters。")
            with (
                audio.open("rb") as handle,
                client.stream(
                    "POST",
                    url,
                    headers=headers,
                    files={field: ("audio.wav", handle, "audio/wav")},
                    data={"model": settings.separation_api_model, "parameters": json.dumps(params)},
                    timeout=settings.separation_timeout_seconds,
                ) as response,
            ):
                response.raise_for_status()
                content = bytearray()
                for block in response.iter_bytes(65536):
                    check_cancelled()
                    content.extend(block)
                    if len(content) > 2 * 1024**2:
                        raise ProjectError("分离 JSON 响应过大。")
                payload = json.loads(content)
    result = (
        _result_field(payload, settings.separation_api_output_field)
        if settings.separation_api_output_field
        else payload
    )
    if not isinstance(result, str):
        raise ProjectError("分离响应中找不到人声下载地址，请检查输出字段路径。")
    _download_audio(result, destination)
    return destination


def _ensure_separation_impl(
    project: DubProject, directory: Path, source: Path, progress=None
) -> tuple[Path, Path]:
    settings = project.settings
    if project.source.channels not in {1, 2}:
        raise ProjectError("实验分离仅支持单声道/双声道；多声道请先明确转换。")
    model_hash = (
        verify_local_model(settings.separation_model)
        if settings.separation_backend == "local"
        else "remote"
    )
    fields = {
        k: v
        for k, v in settings.model_dump().items()
        if k.startswith("separation_")
        and k
        not in {
            "separation_mix_mode",
            "separation_keep_original",
            "separation_keep_ids",
            "separation_keep_padding_ms",
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            {
                "source": project.source.sha256,
                "settings": fields,
                "model": model_hash,
                "implementation": "separation-v1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    root = directory / "analysis/separation" / digest
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(root / ".lock", timeout_seconds=1):
        vocals, residual = root / "vocals.wav", root / "background.wav"
        done = root / "complete.json"
        if done.is_file():
            hashes = json.loads(done.read_text(encoding="utf-8"))
            if all(
                p.is_file() and sha256_file(p) == hashes.get(p.name) for p in (vocals, residual)
            ):
                return vocals, residual
        rate, channels = 44100, project.source.channels
        require_disk_space(root, int(project.source.duration_seconds * rate * channels * 16))
        normalized = root / "source.wav"
        if not normalized.exists():
            normalized_partial = root / "source.partial.wav"
            _run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ar",
                    str(rate),
                    "-ac",
                    str(channels),
                    "-c:a",
                    "pcm_f32le",
                    "-rf64",
                    "auto",
                    str(normalized_partial),
                ]
            )
            os.replace(normalized_partial, normalized)
        block = round(rate * settings.separation_chunk_seconds)
        padding = round(rate * settings.separation_overlap_seconds)
        params = {
            name.removeprefix("separation_"): parse_json_object(getattr(settings, name), label=name)
            for name in (
                "separation_mdx_params",
                "separation_vr_params",
                "separation_demucs_params",
                "separation_mdxc_params",
            )
        }
        params.update(
            parse_json_object(settings.separation_common_params, label="分离通用高级参数")
        )
        vp, bp = root / "vocals.partial.wav", root / "background.partial.wav"
        with (
            SeparationSession(root) as session,
            sf.SoundFile(normalized) as original,
            sf.SoundFile(
                vp, "w", samplerate=rate, channels=channels, format="RF64", subtype="FLOAT"
            ) as voice,
            sf.SoundFile(
                bp, "w", samplerate=rate, channels=channels, format="RF64", subtype="FLOAT"
            ) as bed,
        ):
            total = (original.frames + block - 1) // block
            for index, start in enumerate(range(0, original.frames, block)):
                check_cancelled()
                end = min(original.frames, start + block)
                lo, hi = max(0, start - padding), min(original.frames, end + padding)
                result = root / f"chunk-{index:06d}.wav"
                receipt = result.with_suffix(".sha256")
                valid_cache = (
                    result.is_file()
                    and receipt.is_file()
                    and (sha256_file(result) == receipt.read_text(encoding="ascii").strip())
                )
                if not valid_cache:
                    if progress:
                        progress("实验性人声分离（不推荐）：处理片段", index, total)
                    with tempfile.TemporaryDirectory(dir=root, prefix="worker-") as temporary:
                        temp = Path(temporary)
                        original.seek(lo)
                        sf.write(
                            temp / "input.wav",
                            original.read(hi - lo, dtype="float32", always_2d=True),
                            rate,
                            subtype="PCM_16",
                        )
                        if settings.separation_backend == "local":
                            job = {
                                "action": "separate",
                                "models": str(model_directory()),
                                "model": settings.separation_model,
                                "output": str(temp),
                                "input": str(temp / "input.wav"),
                                "result": str(temp / "result.json"),
                                "device": settings.separation_device,
                                "autocast": settings.separation_use_autocast,
                                "normalization": settings.separation_normalization,
                                "vocal_stem": settings.separation_vocal_stem,
                                "params": params,
                            }
                            session.separate(job, settings.separation_timeout_seconds)
                            output = Path(
                                json.loads((temp / "result.json").read_text(encoding="utf-8"))[
                                    "vocals"
                                ]
                            )
                            if not output.resolve().is_relative_to(temp.resolve()):
                                raise ProjectError("分离器返回了临时目录外的文件。")
                        else:
                            output = _cloud_separate(
                                temp / "input.wav", temp / "download.wav", settings
                            )
                        audio, sample_rate = sf.read(output, dtype="float32", always_2d=True)
                        if sample_rate != rate:
                            audio = soxr.resample(audio, sample_rate, rate)
                        if channels == 1 and audio.shape[1] == 2:
                            audio = audio.mean(axis=1, keepdims=True)
                        if (
                            audio.shape[1] != channels
                            or abs(len(audio) - (hi - lo)) > rate * 0.1
                            or not np.isfinite(audio).all()
                        ):
                            raise ProjectError("分离输出的声道、时长或采样异常，停止混音。")
                        audio = np.pad(audio, ((0, max(0, hi - lo - len(audio))), (0, 0)))[
                            : hi - lo
                        ]
                        sf.write(temp / "validated.wav", audio, rate, subtype="FLOAT")
                        os.replace(temp / "validated.wav", result)
                        atomic_write_text(receipt, sha256_file(result), encoding="ascii")
                separated, _ = sf.read(result, dtype="float32", always_2d=True)
                if (
                    len(separated) != hi - lo
                    or separated.shape[1] != channels
                    or not np.isfinite(separated).all()
                ):
                    raise ProjectError("分离片段缓存损坏，请移除该项目的分离缓存后重试。")
                selected = separated[start - lo : end - lo]
                original.seek(start)
                raw = original.read(end - start, dtype="float32", always_2d=True)
                voice.write(selected)
                bed.write(
                    raw - selected
                )  # Exact complementary residual; never double the background.
        os.replace(vp, vocals)
        os.replace(bp, residual)
        atomic_write_text(done, json.dumps({p.name: sha256_file(p) for p in (vocals, residual)}))
        if progress:
            progress("实验分离完成；请试听检查日语泄漏与音效损伤", 1, 1)
        return vocals, residual


def ensure_separation(
    project: DubProject, directory: Path, source: Path, progress=None
) -> tuple[Path, Path]:
    with exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=30):
        return _ensure_separation_impl(project, directory, source, progress)
