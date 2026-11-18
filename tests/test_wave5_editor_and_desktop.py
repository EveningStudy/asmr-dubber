from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
from importlib.resources import files
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import copy_source_verbatim, extract_waveform_peaks
from asmr_dubber.http_server import Server
from asmr_dubber.models import Sentence, save_project
from asmr_dubber.pipeline import create_project
from asmr_dubber.services.application import Application


def _make_wav(path: Path, seconds: float = 4.0, rate: int = 16_000) -> Path:
    t = np.arange(round(seconds * rate), dtype=np.float64) / rate
    sig = (0.35 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    sf.write(path, np.column_stack([sig, sig]), rate)
    return path


def test_copy_source_verbatim_uses_same_volume_hardlink_and_waveform_peaks(
    tmp_path: Path,
) -> None:
    src = _make_wav(tmp_path / "original.wav", seconds=3.0)
    proj_dir = tmp_path / "proj"
    dst, info = copy_source_verbatim(src, proj_dir)
    assert dst.is_file()
    assert info.duration_seconds == pytest.approx(3.0, abs=0.05)
    # Same-volume NTFS/POSIX hardlink shares inode (st_nlink >= 2).
    assert os.stat(src).st_ino == os.stat(dst).st_ino

    peaks = extract_waveform_peaks(dst, 0.2, 2.2, bins=80)
    assert len(peaks) == 80
    assert max(peaks) == pytest.approx(1.0, abs=1e-3)


def test_incremental_task_poll_waveform_single_row_save_and_loopback_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    src = _make_wav(tmp_path / "clip.wav", seconds=4.0)
    project, proj_dir = create_project(src, tmp_path / "projects")
    project.sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.2,
            end_seconds=1.5,
            source_text="おはよう",
            target_text="早上好",
            status="translated",
        ),
        Sentence(
            id="s000002",
            start_seconds=1.7,
            end_seconds=3.1,
            source_text="おやすみ",
            target_text="晚安",
            status="translated",
        ),
    ]
    manifest = save_project(project, proj_dir)

    app = Application(tmp_path / "tasks.json")
    srv = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{srv.server_port}"
    headers = {"X-ASMR-Token": srv.token}
    try:
        with httpx.Client(trust_env=False) as client:
            p1 = client.post(url + "/api/tasks/poll", json={"since": 0}, headers=headers).json()
            assert p1["changed"] is True
            p2 = client.post(
                url + "/api/tasks/poll", json={"since": p1["version"]}, headers=headers
            ).json()
            assert p2["changed"] is False
            assert p2["tasks"] == []

            wf = client.post(
                url + "/api/projects/waveform",
                json={"project": str(manifest), "start": 0.0, "end": 2.0, "bins": 64},
                headers=headers,
            )
            assert wf.status_code == 200
            assert len(wf.json()["peaks"]) == 64

            # 11-column single-row save with speaker_id
            loaded = client.post(
                url + "/api/projects/get", json={"project": str(manifest)}, headers=headers
            ).json()
            row_11 = [
                "s000001",
                True,
                0.25,
                1.55,
                "おはよう",
                "早安呀",
                "default",
                0.0,
                True,
                0.0,
                "S02",
            ]
            saved = client.post(
                url + "/api/projects/sentence",
                json={
                    "project": str(manifest),
                    "revision": loaded["revision"],
                    "row": row_11,
                },
                headers=headers,
            )
            assert saved.status_code == 200
            s1 = saved.json()["sentences"][0]
            assert s1["target_text"] == "早安呀"
            assert s1["zh_text"] == "早安呀"
            assert s1["speaker_id"] == "S02"

            # Remote mode blocks local file picker & shutdown
            srv.auth = ("asmr", "secret-pass")
            basic = base64.b64encode(b"asmr:secret-pass").decode()
            remote_headers = {
                "X-ASMR-Token": srv.token,
                "Authorization": f"Basic {basic}",
            }
            for blocked_route in ("/api/files/media", "/api/files/folder", "/api/app/shutdown"):
                resp = client.post(url + blocked_route, json={}, headers=remote_headers)
                assert resp.status_code == 403

            # Loopback clean shutdown succeeds
            srv.auth = None
            shut = client.post(url + "/api/app/shutdown", json={}, headers=headers)
            assert shut.status_code == 200
            assert shut.json()["ok"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(3)


def test_virtual_scrolling_1200_rows_and_undo_redo_in_node_esm() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    root = files("asmr_dubber").joinpath("frontend")

    def data_uri(source: str) -> str:
        return "data:text/javascript;base64," + base64.b64encode(source.encode("utf-8")).decode()

    session_uri = data_uri(root.joinpath("session.js").read_text(encoding="utf-8"))
    review_src = (
        root.joinpath("project_review.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
    )
    review_uri = data_uri(review_src)
    wave_src = (
        root.joinpath("waveform_bar.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
    )
    wave_uri = data_uri(wave_src)
    editor_src = (
        root.joinpath("sentence_editor.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
        .replace("'./project_review.js'", json.dumps(review_uri))
        .replace("'./waveform_bar.js'", json.dumps(wave_uri))
    )
    editor_uri = data_uri(editor_src)

    script = f"""
const dom = new Map();
function el(id) {{
  if (!dom.has(id)) {{
    dom.set(id, {{
      id, innerHTML: '', textContent: '', disabled: false, scrollTop: 24000, clientHeight: 480,
      addEventListener() {{}}, classList: {{ contains: () => false }},
    }});
  }}
  return dom.get(id);
}}
globalThis.window = {{ devicePixelRatio: 1, getSelection: () => null }};
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.document = {{
  documentElement: {{ lang: 'zh-CN' }},
  querySelector: (sel) => el(sel),
  querySelectorAll: () => [],
  getElementById: (id) => el('#' + id),
  addEventListener() {{}},
}};
const {{ state }} = await import({json.dumps(session_uri)});
const editor = await import({json.dumps(editor_uri)});
state.boot = {{ labels: {{}}, locales: {{}} }};
state.tasks = [];
const sentences = Array.from({{ length: 1200 }}, (_, i) => ({{
  id: 's' + String(i + 1).padStart(6, '0'),
  enabled: true,
  start_seconds: i * 1.5,
  end_seconds: i * 1.5 + 1.0,
  source_text: 'テスト' + i,
  target_text: '测试' + i,
  zh_text: '测试' + i,
  original_audio_enabled: null,
  original_audio_gain_db: 0,
  chinese_audio_enabled: true,
  chinese_audio_gain_db: 0,
}}));
state.project = {{ source_language: 'ja', source: {{ duration_seconds: 2000 }}, sentences }};

let savedIds = [];
editor.initializeSentenceEditor({{
  filteredSentences: () => state.project.sentences,
  getSelected: () => 's000500',
  setSelected: () => {{}},
  saveRows: async (id) => {{ savedIds.push(id || 'all'); }},
  renderProject: () => {{}},
  syncBusy: () => {{}},
  getListenMode: () => 'both',
}});

editor.renderVirtualRows(sentences, 's000500');
const html = el('#rows').innerHTML;
const renderedDataRows = (html.match(/data-sentence=/g) || []).length;
if (renderedDataRows > 65 || renderedDataRows < 20) {{
  throw new Error('Expected virtualized window of ~35-55 rows, got ' + renderedDataRows);
}}
if (!html.includes('vs-spacer') || !html.includes('data-play-source=')) {{
  throw new Error('Missing virtual spacers or source audition buttons');
}}

editor.recordFieldUndo('s000001', {{ zh_text: '测试0' }}, {{ zh_text: '修改后' }});
sentences[0].zh_text = '修改后';
await editor.undoEdit();
if (sentences[0].zh_text !== '测试0') throw new Error('Undo failed: ' + sentences[0].zh_text);
await editor.redoEdit();
if (sentences[0].zh_text !== '修改后') throw new Error('Redo failed: ' + sentences[0].zh_text);
"""
    res = subprocess.run(
        [node, "--input-type=module", "-"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res.returncode == 0, res.stderr
