from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .mirrors import (
    MIRROR_CONFIG_PATH,
    download_url_allowed,
    external_downloads_allowed,
    load_mirror_config,
)
from .model_packs import inspect_model_pack, model_pack_directory

_BLOCK_SIZE = 4 * 1024 * 1024
_PROBE_SIZE = 64 * 1024
_DEFAULT_SEGMENTS = 4
_DOWNLOAD_RETRIES = 4
_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_MODELSCOPE_HOSTS = ("modelscope.cn", "modelscope.ai")


class ModelPackDownloadError(RuntimeError):
    """A remote model pack could not be downloaded or verified."""


class ModelPackDownloadPaused(ModelPackDownloadError):
    """The caller requested a resumable model-pack download to pause."""


@dataclass(frozen=True)
class RemoteModelPack:
    pack_id: str
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceProbe:
    url: str
    seconds: float
    bytes_read: int
    supports_range: bool

    @property
    def bytes_per_second(self) -> float:
        return self.bytes_read / max(self.seconds, 0.001)


class _DownloadProgress:
    def __init__(self, total: int, present: int, log: LogCallback | None):
        self.total = total
        self.downloaded = present
        self.session_downloaded = 0
        self.log = log
        self.started = time.monotonic()
        self.last_report = 0.0
        self.lock = threading.Lock()

    def advance(self, count: int) -> None:
        if self.log is None:
            return
        with self.lock:
            self.downloaded += count
            self.session_downloaded += count
            now = time.monotonic()
            if now - self.last_report < 2.0 and self.downloaded < self.total:
                return
            elapsed = max(now - self.started, 0.001)
            speed = self.session_downloaded / elapsed / 1024**2
            percent = min(100.0, self.downloaded * 100 / self.total)
            self.log(f"模型包下载 {percent:.1f}%（{speed:.1f} MiB/s）")
            self.last_report = now


REMOTE_MODEL_PACKS: dict[str, RemoteModelPack] = {
    "parakeet-ja-windows": RemoteModelPack(
        pack_id="parakeet-ja-windows",
        filename="ASMR-Dubber-ModelPack-parakeet-ja-windows-v0.2.1.zip",
        size=4_070_471_378,
        sha256="3a9e95e02df01a40533d5f73893d62fe2bf0bb897b98d2b8e494faa2ed139790",
    ),
    "indextts2-checkpoints": RemoteModelPack(
        pack_id="indextts2-checkpoints",
        filename="ASMR-Dubber-ModelPack-indextts2-checkpoints-v0.2.1.zip",
        size=11_189_524_132,
        sha256="144aa91c4de24faf8d415df4fa4324b831609c4bbcef4406a5db4f2a952e108e",
    ),
    "kotoba-whisper-v2.2": RemoteModelPack(
        pack_id="kotoba-whisper-v2.2",
        filename="ASMR-Dubber-ModelPack-kotoba-whisper-v2.2-v1.0.0.zip",
        size=3_027_748_160,
        sha256="a5da2f63fd2c4972dad4cc53db89e0d0250af9d4431905b8c558d55169734c46",
    ),
    "faster-whisper-large-v2": RemoteModelPack(
        pack_id="faster-whisper-large-v2",
        filename="ASMR-Dubber-ModelPack-faster-whisper-large-v2-v1.0.0.zip",
        size=3_087_767_076,
        sha256="4a4a213561d327e82d5dc5a8e8c071313bd948ad90f7b4c51e650044fd3bc949",
    ),
    "qwen3-forced-aligner": RemoteModelPack(
        pack_id="qwen3-forced-aligner",
        filename="ASMR-Dubber-ModelPack-qwen3-forced-aligner-v1.0.0.zip",
        size=1_837_358_823,
        sha256="6697b80bfba3a182a86290ba0f7b8adc958d7112bfe6cc9caa73bc7207b74242",
    ),
    "whisper-vad-asmr-onnx": RemoteModelPack(
        pack_id="whisper-vad-asmr-onnx",
        filename="ASMR-Dubber-ModelPack-whisper-vad-asmr-onnx-v1.0.0.zip",
        size=54_692_316,
        sha256="f7d4c6ec7c9576d325685ffeaf7a39e5160fa1d3e6fe94ae60ed7dc866e5eaa9",
    ),
}

LogCallback = Callable[[str], None]
CancelledCallback = Callable[[], bool]


class _Headers(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class _Response(Protocol):
    status: int
    headers: _Headers

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def geturl(self) -> str: ...


OpenUrl = Callable[..., _Response]


def _valid_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _request(url: str, *, headers: dict[str, str] | None = None) -> Request:
    """Build a CDN-friendly request without ever putting secrets in logs."""

    request = Request(url, headers=headers or {})
    request.add_header("Accept-Encoding", "identity")
    host = (urlparse(url).hostname or "").lower()
    if any(host == suffix or host.endswith("." + suffix) for suffix in _MODELSCOPE_HOSTS):
        # ModelScope's object-storage CDN rejects urllib's default user agent
        # on some routes.  These headers match its resumable browser/curl path.
        request.add_header("User-Agent", "curl/8.0")
        request.add_header("Referer", "https://modelscope.cn/")
        token = os.getenv("MODELSCOPE_API_TOKEN", "").strip()
        if token:
            request.add_header("Authorization", f"Bearer {token}")
            request.add_header("Cookie", f"m_session_id={token}")
    else:
        request.add_header("User-Agent", "ASMR-Dubber/0.4")
    return request


def model_pack_sources(
    pack_id: str,
    *,
    path: Path = MIRROR_CONFIG_PATH,
) -> tuple[str, ...]:
    payload = load_mirror_config(path)
    configured = payload.get("model_pack_sources", {})
    if not isinstance(configured, dict):
        raise ValueError("mirrors.json 中的 model_pack_sources 必须是对象")
    values = configured.get(pack_id, [])
    if not isinstance(values, list):
        raise ValueError(f"model_pack_sources.{pack_id} 必须是数组")
    result: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or not _valid_https_url(raw.strip()):
            raise ValueError(f"model_pack_sources.{pack_id} 包含无效的 HTTPS URL")
        value = raw.strip()
        if not download_url_allowed(value, path=path):
            continue
        if value not in result:
            result.append(value)
    return tuple(result)


def _content_range_total(value: str | None) -> int | None:
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1]
    return int(total) if total.isdigit() else None


def _content_range_bounds(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    start, end, total = (int(item) for item in match.groups())
    return start, end, total


def _open_checked(opener: OpenUrl, request: Request, timeout: float) -> _Response:
    response = opener(request, timeout=timeout)
    if not _valid_https_url(response.geturl()):
        response.close()
        raise ModelPackDownloadError("下载源重定向到了非 HTTPS 地址。")
    status = int(getattr(response, "status", 200))
    if status >= 400:
        response.close()
        raise ModelPackDownloadError(f"下载源返回 HTTP {status}。")
    return response


def _probe_source(
    url: str,
    *,
    expected_size: int,
    opener: OpenUrl,
    timeout: float,
) -> SourceProbe:
    request = _request(url, headers={"Range": f"bytes=0-{_PROBE_SIZE - 1}"})
    started = time.monotonic()
    response = _open_checked(opener, request, timeout)
    try:
        status = int(getattr(response, "status", 200))
        total = _content_range_total(response.headers.get("Content-Range"))
        content_length = response.headers.get("Content-Length")
        if total is not None and total != expected_size:
            raise ModelPackDownloadError(
                f"下载源文件大小不符：应为 {expected_size}，实际为 {total}"
            )
        if status == 200 and content_length and int(content_length) != expected_size:
            raise ModelPackDownloadError(
                f"下载源文件大小不符：应为 {expected_size}，实际为 {content_length}"
            )
        data = response.read(_PROBE_SIZE)
        if not data:
            raise ModelPackDownloadError("下载源未返回数据。")
        return SourceProbe(
            url=url,
            seconds=time.monotonic() - started,
            bytes_read=len(data),
            supports_range=status == 206 and total == expected_size,
        )
    finally:
        response.close()


def probe_model_pack_sources(
    urls: Iterable[str],
    *,
    expected_size: int,
    opener: OpenUrl = urlopen,
    timeout: float = 12.0,
) -> tuple[SourceProbe, ...]:
    candidates = tuple(dict.fromkeys(urls))
    probes: list[SourceProbe] = []
    with ThreadPoolExecutor(max_workers=min(4, len(candidates) or 1)) as executor:
        futures = {
            executor.submit(
                _probe_source,
                url,
                expected_size=expected_size,
                opener=opener,
                timeout=timeout,
            ): url
            for url in candidates
        }
        for future in as_completed(futures):
            try:
                probes.append(future.result())
            except Exception:
                continue
    return tuple(
        sorted(
            probes,
            key=lambda item: (
                not _is_modelscope_source(item.url),
                not item.supports_range,
                -item.bytes_per_second,
                item.seconds,
            ),
        )
    )


def _is_modelscope_source(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in _MODELSCOPE_HOSTS)


def _raise_if_cancelled(cancelled: CancelledCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise ModelPackDownloadPaused("模型包下载已暂停；再次安装时会继续。")


def _copy_response(
    response: _Response,
    handle: BinaryIO,
    *,
    remaining: int,
    cancelled: CancelledCallback | None,
    progress: _DownloadProgress,
) -> None:
    while remaining:
        _raise_if_cancelled(cancelled)
        block = response.read(min(_BLOCK_SIZE, remaining))
        if not block:
            raise ModelPackDownloadError("下载连接提前结束。")
        handle.write(block)
        remaining -= len(block)
        progress.advance(len(block))


def _download_range_part(
    url: str,
    part: Path,
    *,
    start: int,
    end: int,
    opener: OpenUrl,
    timeout: float,
    cancelled: CancelledCallback | None,
    progress: _DownloadProgress,
) -> None:
    expected = end - start + 1
    present = part.stat().st_size if part.is_file() else 0
    if present > expected:
        part.unlink()
        present = 0
    if present == expected:
        return
    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_RETRIES):
        _raise_if_cancelled(cancelled)
        present = part.stat().st_size if part.is_file() else 0
        if present > expected:
            part.unlink()
            present = 0
        if present == expected:
            return
        request = _request(
            url,
            headers={"Range": f"bytes={start + present}-{end}"},
        )
        response: _Response | None = None
        try:
            response = _open_checked(opener, request, timeout)
            if int(getattr(response, "status", 200)) != 206:
                raise ModelPackDownloadError("下载源未按请求返回分段数据。")
            bounds = _content_range_bounds(response.headers.get("Content-Range"))
            requested_start = start + present
            if (
                bounds is None
                or bounds[:2] != (requested_start, end)
                or bounds[2] != progress.total
            ):
                raise ModelPackDownloadError("下载源返回的 Content-Range 与请求不一致。")
            mode = "ab" if present else "wb"
            with part.open(mode) as handle:
                _copy_response(
                    response,
                    handle,
                    remaining=expected - present,
                    cancelled=cancelled,
                    progress=progress,
                )
            return
        except ModelPackDownloadPaused:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= _DOWNLOAD_RETRIES:
                raise
            time.sleep(_RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)])
        finally:
            if response is not None:
                response.close()
    if last_error is not None:
        raise last_error


def _download_segmented(
    url: str,
    staging: Path,
    *,
    expected_size: int,
    opener: OpenUrl,
    timeout: float,
    cancelled: CancelledCallback | None,
    segments: int,
    log: LogCallback | None,
) -> None:
    part_dir = staging.with_suffix(staging.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = (expected_size + segments - 1) // segments
    work: list[tuple[Path, int, int]] = []
    for index in range(segments):
        start = index * chunk_size
        end = min(expected_size - 1, start + chunk_size - 1)
        if start <= end:
            work.append((part_dir / f"{index:02d}.part", start, end))
    metadata_path = part_dir / "download.json"
    metadata = {
        "schema_version": 1,
        "expected_size": expected_size,
        "segments": [[start, end] for _part, start, end in work],
    }
    if metadata_path.is_file():
        try:
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            recorded = None
        if recorded is not None and recorded != metadata:
            for old_part in part_dir.glob("*.part"):
                old_part.unlink(missing_ok=True)
    metadata_staging = metadata_path.with_suffix(".json.new")
    metadata_staging.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_staging, metadata_path)
    present = sum(
        min(part.stat().st_size, end - start + 1) for part, start, end in work if part.is_file()
    )
    progress = _DownloadProgress(expected_size, present, log)
    with ThreadPoolExecutor(max_workers=len(work)) as executor:
        futures = [
            executor.submit(
                _download_range_part,
                url,
                part,
                start=start,
                end=end,
                opener=opener,
                timeout=timeout,
                cancelled=cancelled,
                progress=progress,
            )
            for part, start, end in work
        ]
        for future in as_completed(futures):
            future.result()
    _raise_if_cancelled(cancelled)
    with staging.open("wb") as destination:
        for part, start, end in work:
            if part.stat().st_size != end - start + 1:
                raise ModelPackDownloadError("下载分段大小不完整。")
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=_BLOCK_SIZE)
            part.unlink()
    metadata_path.unlink(missing_ok=True)
    part_dir.rmdir()


def _download_single(
    url: str,
    staging: Path,
    *,
    expected_size: int,
    opener: OpenUrl,
    timeout: float,
    cancelled: CancelledCallback | None,
    log: LogCallback | None,
) -> None:
    present = staging.stat().st_size if staging.is_file() else 0
    if present > expected_size:
        staging.unlink()
        present = 0
    if present == expected_size:
        return
    progress = _DownloadProgress(expected_size, present, log)
    for attempt in range(_DOWNLOAD_RETRIES):
        _raise_if_cancelled(cancelled)
        present = staging.stat().st_size if staging.is_file() else 0
        if present > expected_size:
            staging.unlink()
            present = 0
        if present == expected_size:
            return
        headers = {"Range": f"bytes={present}-"} if present else {}
        response: _Response | None = None
        try:
            response = _open_checked(opener, _request(url, headers=headers), timeout)
            status = int(getattr(response, "status", 200))
            if present and status != 206:
                present = 0
            elif present:
                bounds = _content_range_bounds(response.headers.get("Content-Range"))
                if bounds != (present, expected_size - 1, expected_size):
                    raise ModelPackDownloadError("续传响应的 Content-Range 与本地断点不一致。")
            content_length = response.headers.get("Content-Length")
            if (
                not present
                and status == 200
                and content_length
                and int(content_length) != expected_size
            ):
                raise ModelPackDownloadError("单连接下载源返回的文件大小不符。")
            mode = "ab" if present else "wb"
            with staging.open(mode) as handle:
                _copy_response(
                    response,
                    handle,
                    remaining=expected_size - present,
                    cancelled=cancelled,
                    progress=progress,
                )
            return
        except ModelPackDownloadPaused:
            raise
        except Exception:
            if attempt + 1 >= _DOWNLOAD_RETRIES:
                raise
            time.sleep(_RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)])
        finally:
            if response is not None:
                response.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _archive_is_ready(path: Path, spec: RemoteModelPack) -> bool:
    if not path.is_file() or path.stat().st_size != spec.size:
        return False
    if _sha256(path) != spec.sha256:
        return False
    inspection = inspect_model_pack(path)
    return inspection.manifest is not None and inspection.manifest.pack_id == spec.pack_id


def _discard_partial_state(staging: Path) -> None:
    staging.unlink(missing_ok=True)
    part_dir = staging.with_suffix(staging.suffix + ".parts")
    if not part_dir.is_dir():
        return
    for child in part_dir.iterdir():
        if child.is_file() and (child.suffix == ".part" or child.name == "download.json"):
            child.unlink(missing_ok=True)
    with suppress(OSError):
        part_dir.rmdir()


def _local_cached_archive(spec: RemoteModelPack) -> Path | None:
    configured = os.getenv("ASMR_DUBBER_LOCAL_CACHE_ROOTS", "")
    for raw_root in configured.split(os.pathsep):
        if not raw_root.strip():
            continue
        root = Path(raw_root.strip()).expanduser()
        candidates = (
            root / spec.filename,
            root / "model-packs" / spec.filename,
            root / ".asmr-dubber" / "cache" / "downloads" / spec.filename,
        )
        for candidate in candidates:
            try:
                if _archive_is_ready(candidate, spec):
                    return candidate.resolve()
            except OSError:
                continue
    return None


def prepare_remote_model_pack(
    pack_id: str,
    *,
    directory: Path | None = None,
    mirror_path: Path = MIRROR_CONFIG_PATH,
    log: LogCallback | None = None,
    cancelled: CancelledCallback | None = None,
    opener: OpenUrl = urlopen,
    timeout: float = 60.0,
    segments: int = _DEFAULT_SEGMENTS,
) -> Path | None:
    """Download a verified model pack into the local inbox, if one is configured."""
    spec = REMOTE_MODEL_PACKS.get(pack_id)
    if spec is None:
        return None
    destination_dir = directory or model_pack_directory()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / spec.filename
    if _archive_is_ready(destination, spec):
        if log:
            log(f"复用本地模型包：{spec.filename}")
        return destination
    if destination.exists():
        destination.unlink()
    local_archive = _local_cached_archive(spec)
    if local_archive is not None:
        staging = destination.with_suffix(destination.suffix + ".partial")
        if local_archive != destination.resolve():
            shutil.copy2(local_archive, staging)
            os.replace(staging, destination)
        if log:
            log(f"复用只读本地缓存模型包：{local_archive}")
        return destination
    urls = model_pack_sources(pack_id, path=mirror_path)
    if not urls:
        if external_downloads_allowed(mirror_path):
            return None
        raise ModelPackDownloadError(
            "没有允许的 ModelScope 模型包来源；海外源默认关闭。"
            "请补齐 mirrors.json/ModelScope 仓库，或显式设置 "
            "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1。"
        )
    _raise_if_cancelled(cancelled)
    if log:
        log(f"正在测速 {len(urls)} 个模型包下载源：{pack_id}")
    probes = probe_model_pack_sources(
        urls,
        expected_size=spec.size,
        opener=opener,
        timeout=min(timeout, 12.0),
    )
    if not probes:
        raise ModelPackDownloadError("所有模型包下载源均不可用。")
    staging = destination.with_suffix(destination.suffix + ".partial")
    failures: list[str] = []
    for probe in probes:
        _raise_if_cancelled(cancelled)
        if log:
            speed = probe.bytes_per_second / 1024**2
            mode = "分段续传" if probe.supports_range else "单连接"
            log(f"使用下载源（{speed:.1f} MiB/s 测速，{mode}）：{probe.url}")
        try:
            if probe.supports_range:
                _download_segmented(
                    probe.url,
                    staging,
                    expected_size=spec.size,
                    opener=opener,
                    timeout=timeout,
                    cancelled=cancelled,
                    segments=max(1, min(8, segments)),
                    log=log,
                )
            else:
                _download_single(
                    probe.url,
                    staging,
                    expected_size=spec.size,
                    opener=opener,
                    timeout=timeout,
                    cancelled=cancelled,
                    log=log,
                )
            if staging.stat().st_size != spec.size or _sha256(staging) != spec.sha256:
                _discard_partial_state(staging)
                raise ModelPackDownloadError("模型包大小或 SHA-256 校验失败。")
            inspection = inspect_model_pack(staging)
            if inspection.manifest is None or inspection.manifest.pack_id != pack_id:
                _discard_partial_state(staging)
                raise ModelPackDownloadError("模型包 manifest 校验失败。")
            os.replace(staging, destination)
            _discard_partial_state(staging)
            if log:
                log(f"模型包下载并校验完成：{spec.filename}")
            return destination
        except ModelPackDownloadPaused:
            raise
        except Exception as exc:
            failures.append(f"{probe.url}: {exc}")
            if log:
                log(f"当前模型包下载源失败，保留断点并尝试下一个允许的来源：{exc}")
    raise ModelPackDownloadError(
        "所有允许的模型包下载源均失败；断点文件已保留：" + "；".join(failures)
    )
