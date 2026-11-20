from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import threading
import zipfile
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import httpx
import pytest

from asmr_dubber import model_pack_download as download_module
from asmr_dubber import ui
from asmr_dubber.constants import ASMR_VAD_MODEL, OPTIONAL_ASR_MODEL_REVISIONS
from asmr_dubber.http_server import Server
from asmr_dubber.model_pack_download import ModelPackDownloadError, RemoteModelPack
from asmr_dubber.model_packs import discover_model_packs
from asmr_dubber.services import models as models_service
from asmr_dubber.services import updates
from asmr_dubber.services.application import Application
from asmr_dubber.user_settings import (
    _MANAGED_ENV_VARS,
    UserSettings,
    load_user_settings,
    save_user_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ui_main_parses_host_and_port_and_calls_launch(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        ui, "launch", lambda host="127.0.0.1", port=7860: calls.append((host, port))
    )

    ui.main([])
    ui.main(["--host", "0.0.0.0", "--port", "8899"])

    assert calls == [("127.0.0.1", 7860), ("0.0.0.0", 8899)]


def test_http_server_remote_blocks_open_allows_https_origin_and_static_charset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASMR_DUBBER_UI_USERNAME", "asmr")
    monkeypatch.setenv("ASMR_DUBBER_UI_PASSWORD", "test-pass")
    app = Application(tmp_path / "tasks.json")
    server = Server(("127.0.0.1", 0), app)
    server.auth = ("asmr", "test-pass")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(trust_env=False) as client:
            for route in ("batch/open", "projects/open", "files/folder", "app/shutdown"):
                resp = client.post(
                    f"{url}/api/{route}",
                    json={"path": "x"},
                    headers={
                        "Host": "proxy.example.com",
                        "Origin": "https://proxy.example.com",
                        "X-ASMR-Token": server.token,
                    },
                    auth=("asmr", "test-pass"),
                )
                assert resp.status_code == 403
                assert resp.json()["error"] == "Local host operation forbidden"

            # HTTPS reverse-proxy Origin matching Host is accepted on normal API calls.
            ok = client.post(
                f"{url}/api/settings/update",
                json={"changes": {}},
                headers={
                    "Host": "proxy.example.com",
                    "Origin": "https://proxy.example.com",
                    "X-ASMR-Token": server.token,
                },
                auth=("asmr", "test-pass"),
            )
            assert ok.status_code == 200

            # Text static files get charset=utf-8; binary assets do not.
            js_resp = client.get(
                f"{url}/app.js",
                headers={"Host": "proxy.example.com"},
                auth=("asmr", "test-pass"),
            )
            assert js_resp.status_code == 200
            assert "charset=utf-8" in js_resp.headers["content-type"].lower()

            ico_resp = client.get(
                f"{url}/favicon.ico",
                headers={"Host": "proxy.example.com"},
                auth=("asmr", "test-pass"),
            )
            if ico_resp.status_code == 200:
                assert "charset=" not in ico_resp.headers["content-type"].lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_http_server_static_charset_excludes_binary_extensions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    app = Application(tmp_path / "tasks.json")
    server = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(trust_env=False) as client:
            for ext in ("png", "wasm", "ico"):
                # Inspect mimetype logic via fake resource inside frontend if absent.
                res = client.get(f"{url}/icon.{ext}")
                if res.status_code == 200:
                    assert "charset=" not in res.headers["content-type"].lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_http_server_ipv6_loopback_host_allowlist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    try:
        server = Server(("::1", 0), Application(tmp_path / "tasks.json"))
    except OSError:
        pytest.skip("IPv6 loopback unavailable in this environment")
    assert server.auth is None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://[::1]:{server.server_port}"
        with httpx.Client(trust_env=False) as client:
            resp = client.get(f"{url}/app.js", headers={"Host": f"[::1]:{server.server_port}"})
            assert resp.status_code == 200
            bad = client.get(f"{url}/app.js", headers={"Host": "attacker.invalid"})
            assert bad.status_code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_discover_model_packs_excludes_advanced_and_wheelhouse_archives(tmp_path: Path) -> None:
    inbox = tmp_path / "model-packs"
    inbox.mkdir()
    for name in (
        "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip",
        "ASMR-Dubber-Windows-Advanced-Dependencies-v1.0.0.zip",
        "ASMR-Dubber-Windows-Wheelhouse-v0.4.0.zip",
        "ASMR-Dubber-Windows-CUDA130-Wheelhouse-v0.4.0.zip",
        "ASMR-Dubber-IndexTTS2-Wheelhouse-v0.4.0.zip",
        "ASMR-Dubber-IndexTTS25-Wheelhouse-v1.0.0.zip",
        "wheelhouse-extra-v1.zip",
    ):
        (inbox / name).write_bytes(b"not-a-model-pack")

    assert discover_model_packs(inbox) == []


def test_model_pack_download_strips_auth_and_cookie_on_cross_origin_redirect(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "top-secret-token")
    seen_headers: list[tuple[str, dict[str, str]]] = []

    class RedirectServer(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return None

        def do_GET(self):
            seen_headers.append(("source", {k.lower(): v for k, v in self.headers.items()}))
            self.send_response(302)
            self.send_header("Location", f"http://{target_host}/file.bin")
            self.end_headers()

    class TargetServer(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return None

        def do_GET(self):
            seen_headers.append(("target", {k.lower(): v for k, v in self.headers.items()}))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    target_srv = HTTPServer(("127.0.0.1", 0), TargetServer)
    source_srv = HTTPServer(("127.0.0.1", 0), RedirectServer)
    target_host = f"127.0.0.1:{target_srv.server_port}"
    t1 = threading.Thread(target=target_srv.serve_forever, daemon=True)
    t2 = threading.Thread(target=source_srv.serve_forever, daemon=True)
    t1.start()
    t2.start()
    try:
        req = Request(f"http://127.0.0.1:{source_srv.server_port}/start")
        req.add_unredirected_header("Authorization", "Bearer top-secret-token")
        req.add_unredirected_header("Cookie", "m_session_id=top-secret-token")
        with download_module._SAFE_OPENER.open(req, timeout=5) as resp:
            assert resp.read() == b"ok"
    finally:
        source_srv.shutdown()
        target_srv.shutdown()
        source_srv.server_close()
        target_srv.server_close()
        t1.join(3)
        t2.join(3)

    assert len(seen_headers) == 2
    assert seen_headers[0][1].get("authorization") == "Bearer top-secret-token"
    assert seen_headers[0][1].get("cookie") == "m_session_id=top-secret-token"
    assert "authorization" not in seen_headers[1][1]
    assert "cookie" not in seen_headers[1][1]

    # Also verify _SafeRedirectHandler preserves auth on same-origin redirects
    req_same = Request("https://modelscope.cn/a")
    req_same.add_unredirected_header("Authorization", "Bearer top-secret-token")
    req_same.add_unredirected_header("Cookie", "m_session_id=top-secret-token")
    handler = download_module._SafeRedirectHandler()
    same_redirect = handler.redirect_request(
        req_same, io.BytesIO(), 302, "Found", Message(), "https://modelscope.cn/b"
    )
    assert same_redirect is not None
    same_hdrs = {k.lower(): v for k, v in same_redirect.header_items()}
    assert same_hdrs["authorization"] == "Bearer top-secret-token"
    assert same_hdrs["cookie"] == "m_session_id=top-secret-token"


def test_model_pack_download_checks_disk_space_before_download(tmp_path: Path, monkeypatch) -> None:
    spec = RemoteModelPack(
        pack_id="big-pack",
        filename="big-pack.zip",
        size=10 * 1024**3,
        sha256="a" * 64,
    )
    monkeypatch.setitem(download_module.REMOTE_MODEL_PACKS, spec.pack_id, spec)
    config = tmp_path / "mirrors.json"
    config.write_text(
        json.dumps({"model_pack_sources": {spec.pack_id: ["https://modelscope.cn/big.zip"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        download_module,
        "require_disk_space",
        lambda _dir, _req: (_ for _ in ()).throw(
            download_module.ProjectError("磁盘空间不足：需至少 10.06 GiB，可用 1.00 GiB。")
        ),
    )

    with pytest.raises(ModelPackDownloadError, match="磁盘空间不足"):
        download_module.prepare_remote_model_pack(
            spec.pack_id,
            directory=tmp_path / "inbox",
            mirror_path=config,
        )


def test_services_models_remove_parakeet_and_faster_whisper(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / ".asmr-dubber"
    parakeet_dir = home / "models" / "parakeet"
    parakeet_dir.mkdir(parents=True)
    (parakeet_dir / "parakeet-ctc-1.1b-ja-f16.gguf").write_bytes(b"gguf")

    fw_snapshot = (
        home / "models" / "huggingface" / "models--Systran--faster-whisper-large-v2" / "snapshots"
    ) / "rev1"
    fw_snapshot.mkdir(parents=True)
    (fw_snapshot / "model.bin").write_bytes(b"ct2")
    fw_local = home / "models" / "faster-whisper-large-v3"
    fw_local.mkdir(parents=True)
    (fw_local / "model.bin").write_bytes(b"ct2-v3")

    monkeypatch.setattr(models_service, "portable_home", lambda: home)
    monkeypatch.setattr(models_service, "current", UserSettings)
    monkeypatch.setattr(
        models_service,
        "cached_model_path",
        lambda model_id: fw_snapshot if model_id == "Systran/faster-whisper-large-v2" else None,
    )

    assert models_service.remove("parakeet_nemo") == {"removed": "parakeet_nemo"}
    assert not parakeet_dir.exists()

    assert models_service.remove("faster_whisper") == {"removed": "faster_whisper"}
    assert not fw_snapshot.exists()
    assert not fw_local.exists()


def test_removing_indextts_also_removes_its_own_environment(tmp_path: Path, monkeypatch) -> None:
    import os
    import stat

    home = tmp_path / ".asmr-dubber"
    runtimes = home / "runtimes"
    for name in ("index-tts", "index-tts-2.5"):
        (runtimes / name / "checkpoints").mkdir(parents=True)
        (runtimes / name / "checkpoints" / "gpt.pth").write_bytes(b"w")
        (runtimes / name / ".venv" / "Lib").mkdir(parents=True)
        (runtimes / name / "user-state" / "Local").mkdir(parents=True)
    locked = runtimes / "index-tts" / ".venv" / "Lib" / "torch.dll"
    locked.write_bytes(b"dll")
    os.chmod(locked, stat.S_IREAD)  # installed packages are often read-only on Windows
    shared_python = runtimes / "python" / "cpython-3.11"
    shared_python.mkdir(parents=True)

    monkeypatch.setattr(models_service, "portable_home", lambda: home)
    monkeypatch.setattr(
        models_service,
        "current",
        lambda: UserSettings(
            tts_model_path=str(runtimes / "index-tts" / "checkpoints"),
            tts_index25_model_path=str(runtimes / "index-tts-2.5" / "checkpoints"),
        ),
    )

    assert models_service.remove("indextts2") == {"removed": "indextts2"}
    assert not (runtimes / "index-tts").exists()
    assert (runtimes / "index-tts-2.5" / ".venv").is_dir()  # the other backend stays
    assert shared_python.is_dir()  # the shared base Python stays

    assert models_service.remove("indextts2_5") == {"removed": "indextts2_5"}
    assert not (runtimes / "index-tts-2.5").exists()
    assert shared_python.is_dir()


def test_optional_model_download_passes_pinned_revision_and_only_needed_extra(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(models_service, "current", UserSettings)
    monkeypatch.setattr(models_service.importlib.util, "find_spec", lambda _name: None)
    uv_commands: list[list[str]] = []
    monkeypatch.setattr(models_service.runtime, "_uv_executable", lambda: Path("uv"))
    monkeypatch.setattr(
        models_service.runtime,
        "download_candidates",
        lambda *_a, **_k: ("https://pypi.tuna.tsinghua.edu.cn/simple",),
    )

    def fake_run(command, **_kwargs):
        uv_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(models_service.runtime, "_run_streaming_process", fake_run)
    monkeypatch.setattr(
        "asmr_dubber.model_pack_download.prepare_remote_model_pack",
        lambda *_a, **_k: None,
    )
    snapshots: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "asmr_dubber.mirrors.snapshot_download_with_fallback",
        lambda **kw: snapshots.append((kw["repo_id"], kw.get("revision"))) or "ok",
    )

    class Token:
        def is_set(self) -> bool:
            return False

    models_service._download({"model": "asmr_vad"}, lambda *_a, **_k: None, Token())

    assert len(uv_commands) == 1
    assert any(arg.endswith("[asr-asmr-vad]") for arg in uv_commands[0])
    assert snapshots == [(ASMR_VAD_MODEL, OPTIONAL_ASR_MODEL_REVISIONS[ASMR_VAD_MODEL])]


def test_updates_rejects_symlinks_verifies_sidecar_and_handles_old_lock(
    tmp_path: Path, monkeypatch
) -> None:
    # 1. Symlink entry in ZIP is rejected.
    sym_zip = tmp_path / "sym.zip"
    with zipfile.ZipFile(sym_zip, "w") as zf:
        info = zipfile.ZipInfo("ASMR-Dubber/src/evil")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "target")
        zf.writestr("ASMR-Dubber/ASMR-Dubber.exe", b"exe")
        zf.writestr("ASMR-Dubber/pyproject.toml", b"proj")
        zf.writestr("ASMR-Dubber/src/asmr_dubber/__init__.py", b"init")
    with pytest.raises(ValueError, match="符号链接"):
        updates.apply(sym_zip, tmp_path / "root", tmp_path / "stage")

    # 2. Malformed API digest is rejected instead of skipped.
    with pytest.raises(ValueError, match="SHA-256"):
        updates._download(
            {
                "name": "a.zip",
                "url": "https://github.com/EveningStudy/asmr-dubber/releases/download/v1.0.0/a.zip",
                "size": 2,
                "sha256": "",
                "digest_malformed": True,
            },
            tmp_path / "a.zip",
            lambda *_: None,
            threading.Event(),
        )

    # 3. Sidecar .sha256 is verified when API digest is absent, and proxy prefixes are tried.
    payload = b"hello-update"
    good_hash = hashlib.sha256(payload).hexdigest()
    requested_urls: list[str] = []

    monkeypatch.setattr(updates, "current", lambda: UserSettings(download_source="modelscope"))
    monkeypatch.setattr(
        updates,
        "github_url_candidates",
        lambda url, allow_external=True: (f"https://ghfast.top/{url}", url),
    )
    monkeypatch.setattr(
        updates.httpx,
        "get",
        lambda url, **_k: (
            requested_urls.append(url)
            or SimpleNamespace(text=f"{good_hash}  a.zip\n", raise_for_status=lambda: None)
        ),
    )

    class StreamResp:
        def __init__(self, data: bytes):
            self.data = data
            self.headers = {"Content-Length": str(len(data))}

        def raise_for_status(self):
            return None

        def iter_bytes(self, _n):
            yield self.data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        updates.httpx,
        "stream",
        lambda _m, url, **_k: requested_urls.append(url) or StreamResp(b"tampered" * 2),
    )
    with pytest.raises(ValueError, match="校验失败"):
        updates._download(
            {
                "name": "a.zip",
                "url": "https://github.com/EveningStudy/asmr-dubber/releases/download/v1.0.0/a.zip",
                "size": len(b"tampered" * 2),
                "sha256": "",
                "digest_malformed": False,
                "sha256_url": (
                    "https://github.com/EveningStudy/asmr-dubber/releases/download/v1.0.0/a.zip.sha256"
                ),
            },
            tmp_path / "out.zip",
            lambda *_: None,
            threading.Event(),
        )
    # The checksum itself only comes from GitHub; the archive may use a proxy.
    assert requested_urls[0].endswith("/a.zip.sha256")
    assert requested_urls[0].startswith("https://github.com/")
    assert requested_urls[1].startswith("https://ghfast.top/")

    # 3b. Without any trusted checksum the archive is never fetched through a proxy.
    requested_urls.clear()
    updates._download(
        {
            "name": "a.zip",
            "url": "https://github.com/EveningStudy/asmr-dubber/releases/download/v1.0.0/a.zip",
            "size": len(b"tampered" * 2),
            "sha256": "",
            "digest_malformed": False,
        },
        tmp_path / "direct.zip",
        lambda *_: None,
        threading.Event(),
    )
    assert requested_urls == [
        "https://github.com/EveningStudy/asmr-dubber/releases/download/v1.0.0/a.zip"
    ]

    # 4. _replace falls back to unique .old.<pid> if .old cannot be unlinked.
    src = tmp_path / "new.exe"
    dst = tmp_path / "ASMR-Dubber.exe"
    old = tmp_path / "ASMR-Dubber.exe.old"
    src.write_bytes(b"new")
    dst.write_bytes(b"cur")
    old.write_bytes(b"locked-old")
    real_replace = os.replace

    def flaky_replace(a, b):
        if Path(b) == dst:
            raise PermissionError("locked exe")
        if Path(b) == old:
            raise PermissionError("locked old")
        return real_replace(a, b)

    call_count = 0

    def allow_after_retire(a, b):
        nonlocal call_count
        call_count += 1
        if call_count == 1 and Path(b) == dst:
            raise PermissionError("running exe")
        if Path(b) == old:
            raise PermissionError("locked old")
        return real_replace(a, b)

    monkeypatch.setattr(updates.os, "replace", allow_after_retire)
    updates._replace(src, dst)
    assert dst.read_bytes() == b"new"
    assert list(tmp_path.glob("ASMR-Dubber.exe.old.*"))


def test_user_settings_clears_managed_env_vars_without_wiping_external(
    tmp_path: Path, monkeypatch
) -> None:
    _MANAGED_ENV_VARS.clear()
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://external.pypi.test/simple")

    # 1. Clearing when externally set does not wipe external value.
    save_user_settings(UserSettings(huggingface_endpoint="", pypi_index_url=""))
    load_user_settings()
    assert "HF_ENDPOINT" not in os.environ
    assert os.environ["UV_DEFAULT_INDEX"] == "https://external.pypi.test/simple"

    # 2. Setting and then clearing restores previous values (None / external).
    save_user_settings(
        UserSettings(
            huggingface_endpoint="https://hf.user.test",
            pypi_index_url="https://pypi.user.test/simple",
        )
    )
    load_user_settings()
    assert os.environ["HF_ENDPOINT"] == "https://hf.user.test"
    assert os.environ["UV_DEFAULT_INDEX"] == "https://pypi.user.test/simple"

    save_user_settings(UserSettings(huggingface_endpoint="", pypi_index_url=""))
    load_user_settings()
    assert "HF_ENDPOINT" not in os.environ
    assert os.environ["UV_DEFAULT_INDEX"] == "https://external.pypi.test/simple"


def test_mirror_scripts_do_not_pass_tokens_on_curl_cli_and_ps1_keeps_bom() -> None:
    mirrors_ps1 = (ROOT / "scripts/mirrors.ps1").read_text(encoding="utf-8-sig")
    mirrors_sh = (ROOT / "scripts/mirrors.sh").read_text(encoding="utf-8")

    assert "Authorization: Bearer $($env:MODELSCOPE_API_TOKEN)" not in mirrors_ps1
    assert "Cookie: m_session_id=$($env:MODELSCOPE_API_TOKEN)" not in mirrors_ps1
    assert '"-K", $ConfigFile' in mirrors_ps1

    assert '-H "Authorization: Bearer $MODELSCOPE_API_TOKEN"' not in mirrors_sh
    assert '-H "Cookie: m_session_id=$MODELSCOPE_API_TOKEN"' not in mirrors_sh
    assert '-K "$cfg"' in mirrors_sh

    for ps1 in (ROOT / "scripts").rglob("*.ps1"):
        assert ps1.read_bytes().startswith(b"\xef\xbb\xbf"), ps1
