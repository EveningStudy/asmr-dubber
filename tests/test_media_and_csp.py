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
    clock = [1000.0]
    monkeypatch.setattr(media_module.time, "monotonic", lambda: clock[0])
    # Create the store on the fake clock: its first purge time comes from the clock, and
    # a real uptime above ~11 h would otherwise keep every purge in the "future".
    store = MediaStore()
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


def test_frontend_save_recovery_compare_stale_and_busy_pagination_contracts():
    import base64
    import json
    import shutil
    import subprocess
    from importlib.resources import files

    from asmr_dubber.localization import catalog

    root = files("asmr_dubber").joinpath("frontend")
    app_js = root.joinpath("app.js").read_text(encoding="utf-8")
    projects_js = root.joinpath("projects.js").read_text(encoding="utf-8")

    # pollTasks adds seenTasks only after project refresh succeeds, and syncs all busy controls.
    seen_pos = app_js.index("state.seenTasks.add(key)")
    refresh_pos = app_js.index("api('projects/get'")
    assert seen_pos > refresh_pos
    assert "syncProjectControls()" in app_js

    for selector in (
        "#reviseTranslations",
        "#restoreBackup",
        "#restoreTranslations",
        "#compressTooLong",
        "#disableFailed",
        "#importTranscript",
        "#changeReference",
        "#addSentence",
        "#deleteSentence",
        '#rows input[type="checkbox"]',
        "#rows select",
        "#rows [data-candidates]",
        "#rows [data-take]",
        "contenteditable",
    ):
        assert selector in projects_js, selector
    assert "filteredSentences()" in projects_js
    assert "renderVirtualRows(sentences, selected)" in projects_js

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to run frontend ESM behavior checks")

    session_uri = (
        "data:text/javascript;base64,"
        + base64.b64encode(root.joinpath("session.js").read_bytes()).decode()
    )
    compare_source = (
        root.joinpath("compare_player.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
    )
    compare_uri = (
        "data:text/javascript;base64," + base64.b64encode(compare_source.encode("utf-8")).decode()
    )
    script = f"""
const nodes = new Map();
function makeNode(id) {{
  return {{
    id, hidden: false, textContent: '', innerHTML: '', value: 0, max: 0, checked: false,
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    querySelector: () => ({{ onclick: null }}),
    querySelectorAll: () => [],
  }};
}}
globalThis.document = {{
  documentElement: {{ lang: 'zh-CN' }},
  querySelector: (sel) => {{
    if (!nodes.has(sel)) nodes.set(sel, makeNode(sel));
    return nodes.get(sel);
  }},
  querySelectorAll: () => [],
}};
globalThis.Audio = class {{
  constructor() {{ this.preload = ''; this.muted = false; this.src = ''; this.currentTime = 0; }}
  pause() {{}}
  removeAttribute() {{}}
}};
const s = await import({json.dumps(session_uri)});
s.state.boot = {{ locales: {json.dumps(catalog(), ensure_ascii=False)}, labels: {{}} }};
let directErr = '';
try {{
  await s.enqueueSave(async () => {{ throw new Error('conflict'); }});
}} catch (err) {{
  directErr = err.message;
}}
let firstFlush = '';
try {{
  await s.flushSaves();
}} catch (err) {{
  firstFlush = err.message;
}}
// Subsequent flushSaves and ignoreError navigations must never deadlock!
await s.flushSaves();
let ignored = false;
s.enqueueSave(async () => {{ throw new Error('second'); }}).catch(() => {{}});
await s.flushSaves({{ ignoreError: true }});
ignored = s.state.saveError === null;

const c = await import({json.dumps(compare_uri)});
const project = {{
  source: {{ duration_seconds: 12 }},
  compare_tracks: [
    {{ key: 'source', url: '/m/s', stale: false }},
    {{ key: 'bilingual', url: '/m/b', stale: true, modified: 10 }},
    {{ key: 'replacement', url: '/m/r', stale: false, modified: 20 }},
  ],
}};
s.state.language = 'zh';
c.renderCompare(project);
const zhHtml = nodes.get('#compareTracks').innerHTML;
s.state.language = 'en';
c.renderCompare(project);
const enHtml = nodes.get('#compareTracks').innerHTML;
console.log(JSON.stringify({{ directErr, firstFlush, ignored, zhHtml, enHtml }}));
"""
    proc = subprocess.run(
        [node, "--input-type=module"],
        input=script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    assert out["directErr"] == "conflict"
    assert out["firstFlush"] == "conflict"
    assert out["ignored"] is True
    assert "双语版（旧）" in out["zhHtml"] and 'class="stale"' in out["zhHtml"]
    assert "完全配音版（旧）" not in out["zhHtml"]
    assert "Bilingual mix (outdated)" in out["enHtml"]
