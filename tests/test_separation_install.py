import hashlib
import json

import pytest

from asmr_dubber import separation_download as downloads


class Response:
    def __init__(self, content):
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        yield self.content


def record(monkeypatch, content=b"weights"):
    monkeypatch.setattr(
        downloads,
        "source_manifest",
        lambda: {
            "files": {
                "model.ckpt": {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "upstream": "https://upstream.example/model.ckpt",
                }
            }
        },
    )


def test_modelscope_failure_falls_back_and_repairs_corrupt_file(tmp_path, monkeypatch):
    record(monkeypatch)
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"broken")
    calls, logs = [], []

    def get(url, **_):
        calls.append(url)
        if "modelscope" in url:
            raise OSError("offline")
        return Response(b"weights")

    downloads.download_model_file(
        "https://other.example/model.ckpt", target, models=tmp_path, get=get, report=logs.append
    )
    assert "modelscope.cn" in calls[0]
    assert calls[1] == "https://upstream.example/model.ckpt"
    assert target.read_bytes() == b"weights"
    assert any("下载进度" in line for line in logs)
    assert not target.with_name("model.ckpt.partial").exists()


def test_bad_hash_never_becomes_a_ready_model(tmp_path, monkeypatch):
    record(monkeypatch)
    target = tmp_path / "model.ckpt"
    with pytest.raises(RuntimeError, match="下载失败"):
        downloads.download_model_file(
            "https://upstream.example/model.ckpt",
            target,
            models=tmp_path,
            get=lambda *a, **k: Response(b"bad"),
        )
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_original_source_and_existing_verified_file(tmp_path, monkeypatch):
    record(monkeypatch)
    target = tmp_path / "model.ckpt"
    calls = []

    def get(url, **_):
        calls.append(url)
        return Response(b"weights")

    for _ in range(2):
        downloads.download_model_file(
            "https://upstream.example/model.ckpt",
            target,
            models=tmp_path,
            source="original",
            get=get,
        )
    assert calls == ["https://upstream.example/model.ckpt"]


def test_missing_environment_and_selected_model_status(tmp_path, monkeypatch):
    from asmr_dubber import separation

    monkeypatch.setattr(separation, "model_directory", lambda: tmp_path)
    monkeypatch.setattr(separation, "runtime_python", lambda: tmp_path / "python.exe")
    status = separation.local_install_status("m.ckpt")
    assert "未安装" in status and "未下载" in status
    (tmp_path / "m.ckpt").write_bytes(b"weights")
    assert "尚未校验" in separation.local_install_status("m.ckpt")
    (tmp_path / "m.ckpt.integrity.json").write_text(
        json.dumps({"m.ckpt": "hash", "m.yaml": "hash"})
    )
    assert "文件不完整" in separation.local_install_status("m.ckpt")
