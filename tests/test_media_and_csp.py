import io
import threading

import httpx
import pytest

from asmr_dubber.errors import ProjectError
from asmr_dubber.http_server import CONTENT_SECURITY_POLICY, Server
from asmr_dubber.services import media as media_module
from asmr_dubber.services.application import Application
from asmr_dubber.services.media import MediaStore, discard_upload


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return tmp_path


def test_every_response_carries_a_same_origin_content_security_policy(home):
    server = Server(("127.0.0.1", 0), Application(home / "tasks.json"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(trust_env=False) as client:
            for path in ("/", "/app.js", "/missing.js"):
                policy = client.get(url + path).headers["content-security-policy"]
                assert policy == CONTENT_SECURITY_POLICY
        directives = dict(item.split(" ", 1) for item in policy.split("; "))
        assert directives["script-src"] == "'self'"
        assert directives["media-src"] == "'self'"
        assert directives["default-src"] == "'none'"
        assert "'unsafe-inline'" not in directives["script-src"]
        assert directives["frame-ancestors"] == "'none'"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_upload_rejects_oversized_requests_before_writing(home, monkeypatch):
    store = MediaStore()
    monkeypatch.setattr(media_module, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="文件太大"):
        store.upload(io.BytesIO(b"x" * 11), 11, "a.wav")
    assert not list(media_module.uploads_root().glob("*"))
    assert store.upload(io.BytesIO(b"x" * 10), 10, "a.wav")["name"] == "a.wav"


def test_upload_checks_disk_space_before_writing(home, monkeypatch):
    checked = []

    def refuse(directory, required):
        checked.append((directory, required))
        raise ProjectError("磁盘空间不足")

    monkeypatch.setattr(media_module, "require_disk_space", refuse)
    store = MediaStore()
    with pytest.raises(ProjectError):
        store.upload(io.BytesIO(b"abc"), 3, "a.wav")
    assert checked and checked[0][1] == 3
    assert not list(media_module.uploads_root().glob("*"))


def _file(directory, name):
    path = directory / name
    path.write_bytes(b"data")
    return path


def test_idle_tokens_are_reclaimed_but_requested_ones_survive(home, monkeypatch):
    store = MediaStore()
    clock = [1000.0]
    monkeypatch.setattr(media_module.time, "monotonic", lambda: clock[0])
    playing = store.register(_file(home, "playing.wav")).removeprefix("/media/")
    idle = store.register(_file(home, "idle.wav")).removeprefix("/media/")
    for _ in range(3):  # Range requests keep refreshing the playing token
        clock[0] += media_module.TOKEN_IDLE_SECONDS * 0.6
        store.resolve(playing)
    store.register(_file(home, "other.wav"))
    clock[0] += media_module._PURGE_INTERVAL_SECONDS + 1
    store.register(_file(home, "other.wav"))
    assert store.resolve(playing)[0].name == "playing.wav"
    with pytest.raises(FileNotFoundError):
        store.resolve(idle)


def test_missing_files_and_token_cap(home, monkeypatch):
    monkeypatch.setattr(media_module, "MAX_TOKENS", 3)
    store = MediaStore()
    gone = _file(home, "gone.wav")
    store.register(gone)
    gone.unlink()
    tokens = [store.register(_file(home, f"f{i}.wav")) for i in range(6)]
    store.register(_file(home, "last.wav"))
    assert len(store._files) <= 4
    assert len(store._tokens) == len(store._files) == len(store._seen)
    assert gone.resolve() not in store._tokens
    assert tokens[-1].removeprefix("/media/") in store._files or len(store._files) <= 4


def test_discarding_an_upload_releases_its_token(home):
    store = MediaStore()
    info = store.upload(io.BytesIO(b"abc"), 3, "voice.wav")
    token = info["url"].removeprefix("/media/")
    assert store.resolve(token)[0].read_bytes() == b"abc"
    discard_upload(info["path"])
    assert token not in store._files and not store._tokens
    with pytest.raises(FileNotFoundError):
        store.resolve(token)


def test_register_is_stable_for_the_same_file(home):
    store = MediaStore()
    path = _file(home, "a.wav")
    assert store.register(path) == store.register(path)
    assert len(store._files) == 1


def test_service_errors_are_chinese_with_english_catalog_entries():
    import ast
    import re
    from pathlib import Path

    from asmr_dubber.localization import catalog

    english = catalog("en")
    normalized_templates = {re.sub(r"\{\w+\}", "{*}", key) for key in english if "{" in key}
    services = Path(media_module.__file__).parent
    for path in sorted(services.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args
            ):
                continue
            arg = node.exc.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                text = arg.value
                assert re.search(r"[一-鿿]", text), f"{path.name}:{node.lineno}: {text}"
                if not text.endswith((": ", "：")):
                    assert text in english, f"{path.name}:{node.lineno}: missing English for {text}"
            elif isinstance(arg, ast.JoinedStr):
                skeleton = "".join(
                    part.value if isinstance(part, ast.Constant) else "{*}" for part in arg.values
                )
                assert re.search(r"[一-鿿]", skeleton), f"{path.name}:{node.lineno}: {skeleton}"
                assert skeleton in normalized_templates, (
                    f"{path.name}:{node.lineno}: missing English template for {skeleton}"
                )
