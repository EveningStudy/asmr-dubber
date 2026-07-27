from __future__ import annotations

import hashlib
import json
from email.message import Message
from pathlib import Path
from urllib.request import Request

import pytest

from asmr_dubber import model_pack_download as download_module
from asmr_dubber.model_pack_download import (
    REMOTE_MODEL_PACKS,
    ModelPackDownloadError,
    RemoteModelPack,
    SourceProbe,
    model_pack_sources,
    prepare_remote_model_pack,
    probe_model_pack_sources,
)
from asmr_dubber.model_packs import ModelPackSource, build_model_pack
from asmr_dubber.platforms import current_platform


class FakeResponse:
    def __init__(self, data: bytes, *, status: int, url: str, headers: dict[str, str]):
        self._data = data
        self._offset = 0
        self.status = status
        self._url = url
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._offset
        result = self._data[self._offset : self._offset + size]
        self._offset += len(result)
        return result

    def close(self) -> None:
        return None

    def geturl(self) -> str:
        return self._url


class BytesOpener:
    def __init__(self, payload: bytes, *, range_supported: bool = True):
        self.payload = payload
        self.range_supported = range_supported
        self.requests: list[str] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        del timeout
        value = request.headers.get("Range")
        self.requests.append(value or "")
        if value and self.range_supported:
            start_text, end_text = value.removeprefix("bytes=").split("-", 1)
            start = int(start_text)
            end = int(end_text) if end_text else len(self.payload) - 1
            data = self.payload[start : end + 1]
            return FakeResponse(
                data,
                status=206,
                url=request.full_url,
                headers={
                    "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
                },
            )
        return FakeResponse(
            self.payload,
            status=200,
            url=request.full_url,
            headers={"Content-Length": str(len(self.payload))},
        )


class FlakyBytesOpener(BytesOpener):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.interrupted = False

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        value = request.headers.get("Range") or ""
        if value and not value.endswith("-65535") and not self.interrupted:
            del timeout
            self.requests.append(value)
            start_text, end_text = value.removeprefix("bytes=").split("-", 1)
            start, end = int(start_text), int(end_text)
            data = self.payload[start : min(end + 1, start + 17)]
            self.interrupted = True
            return FakeResponse(
                data,
                status=206,
                url=request.full_url,
                headers={
                    "Content-Length": str(end - start + 1),
                    "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
                },
            )
        return super().__call__(request, timeout=timeout)


def _test_pack(tmp_path: Path, pack_id: str) -> tuple[Path, bytes, RemoteModelPack]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"test weights" * 100)
    archive = tmp_path / "source.zip"
    build_model_pack(
        archive,
        pack_id=pack_id,
        display_name="Remote Test Pack",
        pack_version="1",
        platforms=(current_platform().id,),
        architectures=("any",),
        sources=(ModelPackSource(source, "models/test"),),
    )
    payload = archive.read_bytes()
    spec = RemoteModelPack(
        pack_id=pack_id,
        filename=f"{pack_id}.zip",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return archive, payload, spec


def _mirror_config(tmp_path: Path, pack_id: str, urls: list[str]) -> Path:
    path = tmp_path / "mirrors.json"
    path.write_text(json.dumps({"model_pack_sources": {pack_id: urls}}), encoding="utf-8")
    return path


def test_model_pack_source_urls_must_be_https(tmp_path: Path) -> None:
    config = _mirror_config(tmp_path, "pack", ["http://unsafe.example/pack.zip"])
    with pytest.raises(ValueError, match="无效的 HTTPS URL"):
        model_pack_sources("pack", path=config)


def test_probe_prefers_range_then_speed(monkeypatch) -> None:
    probes = {
        "https://slow.example/pack.zip": SourceProbe(
            "https://slow.example/pack.zip", 1.0, 10, True
        ),
        "https://fast.example/pack.zip": SourceProbe(
            "https://fast.example/pack.zip", 0.1, 10, True
        ),
        "https://single.example/pack.zip": SourceProbe(
            "https://single.example/pack.zip", 0.01, 10, False
        ),
    }
    monkeypatch.setattr(
        "asmr_dubber.model_pack_download._probe_source",
        lambda url, **_kwargs: probes[url],
    )
    result = probe_model_pack_sources(probes, expected_size=10)
    assert [item.url for item in result] == [
        "https://fast.example/pack.zip",
        "https://slow.example/pack.zip",
        "https://single.example/pack.zip",
    ]


def test_complete_local_pack_never_opens_network(tmp_path: Path, monkeypatch) -> None:
    _archive, payload, spec = _test_pack(tmp_path, "local-pack")
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, spec)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    destination = inbox / spec.filename
    destination.write_bytes(payload)
    config = _mirror_config(tmp_path, spec.pack_id, ["https://unused.example/pack.zip"])

    def fail_open(*_args, **_kwargs):
        raise AssertionError("network must not be used")

    assert (
        prepare_remote_model_pack(
            spec.pack_id,
            directory=inbox,
            mirror_path=config,
            opener=fail_open,
        )
        == destination
    )


def test_range_download_resumes_existing_part(tmp_path: Path, monkeypatch) -> None:
    _archive, payload, spec = _test_pack(tmp_path, "range-pack")
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, spec)
    inbox = tmp_path / "inbox"
    config = _mirror_config(tmp_path, spec.pack_id, ["https://mirror.example/pack.zip"])
    staging = inbox / f"{spec.filename}.partial"
    part_dir = staging.with_suffix(staging.suffix + ".parts")
    part_dir.mkdir(parents=True)
    first_segment_size = (len(payload) + 1) // 2
    resumed_bytes = 17
    (part_dir / "00.part").write_bytes(payload[:resumed_bytes])
    opener = BytesOpener(payload)

    destination = prepare_remote_model_pack(
        spec.pack_id,
        directory=inbox,
        mirror_path=config,
        opener=opener,
        segments=2,
    )

    assert destination is not None and destination.read_bytes() == payload
    assert f"bytes={resumed_bytes}-{first_segment_size - 1}" in opener.requests


def test_non_range_source_uses_single_stream(tmp_path: Path, monkeypatch) -> None:
    _archive, payload, spec = _test_pack(tmp_path, "single-pack")
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, spec)
    inbox = tmp_path / "inbox"
    config = _mirror_config(tmp_path, spec.pack_id, ["https://mirror.example/pack.zip"])
    opener = BytesOpener(payload, range_supported=False)

    destination = prepare_remote_model_pack(
        spec.pack_id,
        directory=inbox,
        mirror_path=config,
        opener=opener,
    )

    assert destination is not None and destination.read_bytes() == payload


def test_complete_single_stream_partial_is_verified_without_redownload(
    tmp_path: Path, monkeypatch
) -> None:
    _archive, payload, spec = _test_pack(tmp_path, "complete-partial-pack")
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, spec)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    staging = inbox / f"{spec.filename}.partial"
    staging.write_bytes(payload)
    config = _mirror_config(tmp_path, spec.pack_id, ["https://mirror.example/pack.zip"])
    opener = BytesOpener(payload, range_supported=False)

    destination = prepare_remote_model_pack(
        spec.pack_id,
        directory=inbox,
        mirror_path=config,
        opener=opener,
    )

    assert destination is not None and destination.read_bytes() == payload
    assert opener.requests == [f"bytes=0-{download_module._PROBE_SIZE - 1}"]


def test_range_connection_interruption_retries_from_new_part_offset(
    tmp_path: Path, monkeypatch
) -> None:
    _archive, payload, spec = _test_pack(tmp_path, "retry-pack")
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, spec)
    inbox = tmp_path / "inbox"
    config = _mirror_config(tmp_path, spec.pack_id, ["https://modelscope.cn/retry.zip"])
    opener = FlakyBytesOpener(payload)

    destination = prepare_remote_model_pack(
        spec.pack_id,
        directory=inbox,
        mirror_path=config,
        opener=opener,
        segments=1,
    )

    assert destination is not None and destination.read_bytes() == payload
    assert any(request.startswith("bytes=17-") for request in opener.requests)


def test_hash_mismatch_never_leaves_final_archive(tmp_path: Path, monkeypatch) -> None:
    _archive, payload, spec = _test_pack(tmp_path, "bad-pack")
    wrong = RemoteModelPack(
        pack_id=spec.pack_id,
        filename=spec.filename,
        size=spec.size,
        sha256="0" * 64,
    )
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, wrong)
    inbox = tmp_path / "inbox"
    config = _mirror_config(tmp_path, spec.pack_id, ["https://mirror.example/pack.zip"])

    with pytest.raises(ModelPackDownloadError, match="均失败"):
        prepare_remote_model_pack(
            spec.pack_id,
            directory=inbox,
            mirror_path=config,
            opener=BytesOpener(payload),
            segments=2,
        )

    assert not (inbox / spec.filename).exists()
    assert not (inbox / f"{spec.filename}.partial").exists()


def test_modelscope_requests_use_cdn_headers_without_logging_token(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "secret-test-token")

    request = download_module._request(
        "https://modelscope.cn/models/example/repo/resolve/master/file.zip",
        headers={"Range": "bytes=0-9"},
    )
    headers = {name.lower(): value for name, value in request.header_items()}

    assert headers["user-agent"] == "curl/8.0"
    assert headers["accept-encoding"] == "identity"
    assert headers["referer"] == "https://modelscope.cn/"
    assert headers["authorization"] == "Bearer secret-test-token"
    assert headers["cookie"] == "m_session_id=secret-test-token"
    assert "secret-test-token" not in request.full_url


def test_release_policy_filters_external_model_pack_sources(tmp_path: Path) -> None:
    config = tmp_path / "mirrors.json"
    config.write_text(
        json.dumps(
            {
                "download_policy": {"allow_external": False},
                "model_pack_sources": {
                    "pack": [
                        "https://modelscope.cn/models/example/repo/resolve/master/pack.zip",
                        "https://huggingface.co/example/repo/resolve/main/pack.zip",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert model_pack_sources("pack", path=config) == (
        "https://modelscope.cn/models/example/repo/resolve/master/pack.zip",
    )


def test_network_failure_preserves_segment_parts(tmp_path: Path, monkeypatch) -> None:
    _archive, _payload, spec = _test_pack(tmp_path, "interrupted-pack")
    monkeypatch.setitem(REMOTE_MODEL_PACKS, spec.pack_id, spec)
    inbox = tmp_path / "inbox"
    config = _mirror_config(tmp_path, spec.pack_id, ["https://modelscope.cn/pack.zip"])
    monkeypatch.setattr(
        download_module,
        "probe_model_pack_sources",
        lambda *_args, **_kwargs: (SourceProbe("https://modelscope.cn/pack.zip", 0.1, 1024, True),),
    )

    def fail_segmented(_url: str, staging: Path, **_kwargs) -> None:
        part_dir = staging.with_suffix(staging.suffix + ".parts")
        part_dir.mkdir(parents=True)
        (part_dir / "00.part").write_bytes(b"resumable")
        raise OSError("connection reset")

    monkeypatch.setattr(download_module, "_download_segmented", fail_segmented)

    with pytest.raises(ModelPackDownloadError, match="断点文件已保留"):
        prepare_remote_model_pack(
            spec.pack_id,
            directory=inbox,
            mirror_path=config,
        )

    assert (inbox / f"{spec.filename}.partial.parts" / "00.part").read_bytes() == b"resumable"
