from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .mirrors import MIRROR_CONFIG_PATH, load_mirror_config
from .model_packs import inspect_model_pack, model_pack_directory

_BLOCK_SIZE = 4 * 1024 * 1024
_PROBE_SIZE = 64 * 1024
_DEFAULT_SEGMENTS = 4


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
        self.log = log
        self.started = time.monotonic()
        self.last_report = 0.0
        self.lock = threading.Lock()

    def advance(self, count: int) -> None:
        if self.log is None:
            return
        with self.lock:
            self.downloaded += count
            now = time.monotonic()
            if now - self.last_report < 2.0 and self.downloaded < self.total:
                return
            elapsed = max(now - self.started, 0.001)
            speed = max(0, self.downloaded) / elapsed / 1024**2
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
}

LogCallback = Callable[[str], None]
CancelledCallback = Callable[[], bool]


class _Response(Protocol):
    status: int
    headers: object

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def geturl(self) -> str: ...


OpenUrl = Callable[..., _Response]


def _valid_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


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
        if value not in result:
            result.append(value)
    return tuple(result)


def _content_range_total(value: str | None) -> int | None:
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1]
    return int(total) if total.isdigit() else None


def _open_checked(opener: OpenUrl, request: Request, timeout: float) -> _Response:
    response = opener(request, timeout=timeout)
    if not _valid_https_url(response.geturl()):
        response.close()
        raise ModelPackDownloadError("下载源重定向到了非 HTTPS 地址。")
    return response


def _probe_source(
    url: str,
    *,
    expected_size: int,
    opener: OpenUrl,
    timeout: float,
) -> SourceProbe:
    request = Request(url, headers={"Range": f"bytes=0-{_PROBE_SIZE - 1}"})
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
            key=lambda item: (not item.supports_range, -item.bytes_per_second, item.seconds),
        )
    )


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
    request = Request(url, headers={"Range": f"bytes={start + present}-{end}"})
    response = _open_checked(opener, request, timeout)
    try:
        if int(getattr(response, "status", 200)) != 206:
            raise ModelPackDownloadError("下载源未按请求返回分段数据。")
        mode = "ab" if present else "wb"
        with part.open(mode) as handle:
            _copy_response(
                response,
                handle,
                remaining=expected - present,
                cancelled=cancelled,
                progress=progress,
            )
    finally:
        response.close()


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
    present = sum(
        min(part.stat().st_size, end - start + 1)
        for part, start, end in work
        if part.is_file()
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
    progress = _DownloadProgress(expected_size, present, log)
    headers = {"Range": f"bytes={present}-"} if present else {}
    response = _open_checked(opener, Request(url, headers=headers), timeout)
    try:
        status = int(getattr(response, "status", 200))
        if present and status != 206:
            present = 0
        mode = "ab" if present else "wb"
        with staging.open(mode) as handle:
            _copy_response(
                response,
                handle,
                remaining=expected_size - present,
                cancelled=cancelled,
                progress=progress,
            )
    finally:
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
    urls = model_pack_sources(pack_id, path=mirror_path)
    if not urls:
        return None
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
                if staging.exists():
                    staging.unlink()
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
                raise ModelPackDownloadError("模型包大小或 SHA-256 校验失败。")
            inspection = inspect_model_pack(staging)
            if inspection.manifest is None or inspection.manifest.pack_id != pack_id:
                raise ModelPackDownloadError("模型包 manifest 校验失败。")
            os.replace(staging, destination)
            if log:
                log(f"模型包下载并校验完成：{spec.filename}")
            return destination
        except ModelPackDownloadPaused:
            raise
        except Exception as exc:  # noqa: BLE001 - try the next configured source
            failures.append(f"{probe.url}: {exc}")
            if log:
                log(f"当前模型包下载源失败，自动切换：{exc}")
    if staging.exists():
        staging.unlink()
    raise ModelPackDownloadError("所有模型包下载源均失败：" + "；".join(failures))
