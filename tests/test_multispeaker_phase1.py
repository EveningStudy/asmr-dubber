from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import threading
from importlib.resources import files
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber.diarization import diarize_sentences, stereo_pan_features
from asmr_dubber.http_server import Server
from asmr_dubber.localization import catalog
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    SpeakerReference,
    load_project,
    save_project,
)
from asmr_dubber.services import parameters
from asmr_dubber.services.application import Application
from asmr_dubber.services.project_records import apply_single_row, apply_table
from asmr_dubber.tts import tts_cache_key
from asmr_dubber.voice_reference import (
    effective_project_reference_sentence,
    prepare_index_speaker_reference,
    prepare_voice_reference,
    speaker_reference_sentence,
)


def _synth_voice(
    rate: int,
    seconds: float,
    *,
    f0: float,
    formants: tuple[float, float, float],
    seed: int = 0,
) -> np.ndarray:
    n = round(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    glottal = np.zeros(n, dtype=np.float64)
    max_k = int((rate * 0.45) // f0)
    for k in range(1, max(2, min(max_k, 40))):
        freq = k * f0
        gain = sum(
            math.exp(-0.5 * ((freq - fc) / (0.14 * fc)) ** 2) for fc in formants
        ) / math.sqrt(k)
        glottal += gain * np.sin(2.0 * math.pi * freq * t + 0.1 * k)
    rng = np.random.default_rng(seed)
    glottal += rng.standard_normal(n) * 0.004
    glottal /= max(float(np.max(np.abs(glottal))), 1e-9)
    return (glottal * 0.35).astype(np.float32)


def _pan_stereo(mono: np.ndarray, *, left_gain: float, right_gain: float) -> np.ndarray:
    return np.column_stack([mono * left_gain, mono * right_gain]).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16_000) -> Path:
    sf.write(path, samples, rate)
    return path


def test_stereo_pan_features_and_binaural_diarization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate = 16_000
    tone = _synth_voice(rate, 1.5, f0=235.0, formants=(650.0, 1950.0, 3000.0))
    left_feat = stereo_pan_features(_pan_stereo(tone, left_gain=1.0, right_gain=0.05), rate)
    right_feat = stereo_pan_features(_pan_stereo(tone, left_gain=0.05, right_gain=1.0), rate)
    center_feat = stereo_pan_features(_pan_stereo(tone, left_gain=0.7, right_gain=0.7), rate)
    assert left_feat[0] < -0.9 and left_feat[1] < -0.9
    assert right_feat[0] > 0.9 and right_feat[1] > 0.9
    assert abs(center_feat[0]) < 1e-3

    # Test with pure acoustic embeddings so the threshold stays deterministic across setups.
    monkeypatch.setattr("asmr_dubber.diarization.extract_campplus_embeddings", lambda _clips: None)

    # Close female timbres in opposite ears should separate cleanly.
    gap = np.zeros((round(rate * 0.25), 2), dtype=np.float32)
    turns = ["A", "B", "A", "B", "A", "B"]
    pieces: list[np.ndarray] = [gap]
    sentences: list[Sentence] = []
    cursor = len(gap) / rate
    for i, spk in enumerate(turns, start=1):
        if spk == "A":
            mono = _synth_voice(rate, 1.6, f0=242.0, formants=(660.0, 1960.0, 3020.0), seed=i)
            clip = _pan_stereo(mono, left_gain=1.0, right_gain=0.06)
        else:
            mono = _synth_voice(rate, 1.6, f0=218.0, formants=(590.0, 1820.0, 2860.0), seed=20 + i)
            clip = _pan_stereo(mono, left_gain=0.06, right_gain=1.0)
        start = round(cursor, 3)
        end = round(cursor + len(clip) / rate, 3)
        sentences.append(
            Sentence(id=f"s{i:06d}", start_seconds=start, end_seconds=end, source_text=f"台詞{i}")
        )
        pieces.extend([clip, gap])
        cursor += (len(clip) + len(gap)) / rate

    duet_wav = _write_wav(tmp_path / "binaural_duet.wav", np.vstack(pieces), rate)
    auto_out = diarize_sentences(sentences, duet_wav)
    explicit_out = diarize_sentences(sentences, duet_wav, num_speakers=2)
    expected = ["S01" if spk == "A" else "S02" for spk in turns]
    assert [s.speaker_id for s in auto_out] == expected
    assert [s.speaker_id for s in explicit_out] == expected

    # A solo voice moving between Left and Right ears stays unified in auto mode.
    solo_pieces: list[np.ndarray] = [gap]
    solo_sents: list[Sentence] = []
    cursor = len(gap) / rate
    for i in range(1, 7):
        mono = _synth_voice(rate, 1.6, f0=235.0, formants=(640.0, 1940.0, 3000.0), seed=i)
        clip = (
            _pan_stereo(mono, left_gain=1.0, right_gain=0.05)
            if i % 2 == 1
            else _pan_stereo(mono, left_gain=0.05, right_gain=1.0)
        )
        solo_sents.append(
            Sentence(
                id=f"s{i:06d}",
                start_seconds=round(cursor, 3),
                end_seconds=round(cursor + len(clip) / rate, 3),
                source_text=f"ひとり{i}",
            )
        )
        solo_pieces.extend([clip, gap])
        cursor += (len(clip) + len(gap)) / rate
    solo_wav = _write_wav(tmp_path / "binaural_solo.wav", np.vstack(solo_pieces), rate)
    solo_out = diarize_sentences(solo_sents, solo_wav)
    assert {s.speaker_id for s in solo_out} == {"S01"}


def test_speaker_reference_overrides_cache_keys_and_cleanup(tmp_path: Path) -> None:
    rate = 16_000
    source = _write_wav(
        tmp_path / "source.wav",
        _synth_voice(rate, 12.0, f0=210.0, formants=(600.0, 1800.0, 2800.0)),
        rate,
    )
    ext_a = _write_wav(
        tmp_path / "ext_a.wav",
        _synth_voice(rate, 2.0, f0=260.0, formants=(700.0, 2100.0, 3200.0), seed=1),
        rate,
    )
    ext_b = _write_wav(
        tmp_path / "ext_b.wav",
        _synth_voice(rate, 2.0, f0=125.0, formants=(420.0, 1250.0, 2300.0), seed=2),
        rate,
    )
    (tmp_path / "references").mkdir(exist_ok=True)
    proj = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="b" * 64,
            duration_seconds=12.0,
            sample_rate=rate,
            channels=1,
        ),
        settings=ProjectSettings(
            tts_backend="indextts2",
            tts_model="IndexTTS2",
            tts_index_speaker_source="sentence_reference",
        ),
        speaker_references={
            "S01": SpeakerReference(source="external", audio_path=str(ext_a), audio_label="A.wav"),
            "S02": SpeakerReference(source="project_sentence", sentence_id="s000003"),
        },
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.5,
                end_seconds=3.0,
                speaker_id="S01",
                source_text="おはよう。",
                zh_text="早安。",
            ),
            Sentence(
                id="s000002",
                start_seconds=3.5,
                end_seconds=6.5,
                speaker_id="S02",
                source_text="おはよう、よく眠れた？",
                zh_text="早安，睡得好吗？",
            ),
            Sentence(
                id="s000003",
                start_seconds=7.0,
                end_seconds=10.5,
                speaker_id="S02",
                source_text="とてもよく眠れたよ。",
                zh_text="睡得特别好哦。",
            ),
        ],
    )

    ref_s1 = prepare_index_speaker_reference(proj, tmp_path, source, proj.sentences[0])
    ref_s2 = prepare_index_speaker_reference(proj, tmp_path, source, proj.sentences[1])
    assert ref_s1.path == ext_a.resolve()
    assert "index_speaker_anchor" in ref_s2.path.name
    assert effective_project_reference_sentence(proj, proj.sentences[1])[0].id == "s000003"
    assert speaker_reference_sentence(proj, proj.sentences[1])[0].id == "s000003"
    assert (
        speaker_reference_sentence(proj, proj.sentences[1], ignore_override=True)[0].id == "s000003"
    )

    # Generic cloning backend uses distinct per-speaker overrides too.
    vr1 = prepare_voice_reference(proj, tmp_path, source, proj.sentences[0])
    vr2 = prepare_voice_reference(proj, tmp_path, source, proj.sentences[1])
    assert vr1.path == ext_a.resolve()
    assert "speaker_S02" in vr2.path.name

    key_before = tts_cache_key(proj, proj.sentences[0])
    proj.speaker_references["S01"] = SpeakerReference(
        source="external", audio_path=str(ext_b), audio_label="B.wav"
    )
    assert tts_cache_key(proj, proj.sentences[0]) != key_before

    # Generic TTS backend cache key also reflects per-speaker overrides.
    proj.settings.tts_backend = "gpt_sovits"
    proj.settings.tts_model = "v2"
    g_before = tts_cache_key(proj, proj.sentences[0])
    proj.speaker_references["S01"] = SpeakerReference(
        source="external", audio_path=str(ext_a), audio_label="A.wav"
    )
    assert tts_cache_key(proj, proj.sentences[0]) != g_before

    # Deleting s000003 prunes S02's project_sentence override, keeping S01 external intact.
    row3_del = [
        "s000003",
        True,
        7.0,
        10.5,
        "",
        "",
        "default",
        0.0,
        True,
        0.0,
        "S02",
    ]
    assert apply_single_row(proj, row3_del) is True
    assert "S02" not in proj.speaker_references
    assert "S01" in proj.speaker_references

    # Full-table deletion prunes remaining stale project_sentence binding.
    proj.speaker_references["S02"] = SpeakerReference(
        source="project_sentence", sentence_id="s000002"
    )
    row1_keep = [
        "s000001",
        True,
        0.5,
        3.0,
        "おはよう。",
        "早安。",
        "default",
        0.0,
        True,
        0.0,
        "S01",
    ]
    assert apply_table(proj, [row1_keep]) is True
    assert "S02" not in proj.speaker_references
    assert "S01" in proj.speaker_references


def test_analyze_project_prunes_sentence_overrides_but_keeps_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_wav(tmp_path / "clip.wav", np.zeros(32_000, dtype=np.float32), 16_000)
    proj, proj_dir = pipeline.create_project(source, tmp_path / "projects")
    ext = _write_wav(tmp_path / "ext.wav", np.ones(16_000, dtype=np.float32) * 0.1, 16_000)
    proj.speaker_references = {
        "S01": SpeakerReference(source="external", audio_path=str(ext), audio_label="ext.wav"),
        "S02": SpeakerReference(source="project_sentence", sentence_id="s000001"),
    }
    save_project(proj, proj_dir)

    monkeypatch.setattr(pipeline, "make_asr_analysis_copy", lambda src, _dir: src)
    monkeypatch.setattr(
        pipeline,
        "transcribe_source",
        lambda *_a, **_k: (
            [Sentence(id="s1", start_seconds=0.1, end_seconds=1.5, source_text="こんにちは。")],
            "ja",
        ),
    )
    monkeypatch.setattr("asmr_dubber.dub_qc.snap_sentences_to_silence", lambda sents, _audio: sents)
    monkeypatch.setattr(pipeline, "save_project", save_project)

    pipeline.analyze_project(proj, proj_dir)
    assert "S01" in proj.speaker_references
    assert "S02" not in proj.speaker_references


def test_http_api_speaker_profiles_targeted_invalidation_and_parameter_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    by_key = {item["key"]: item for item in parameters.catalog()}
    assert by_key["asr_speaker_count"]["basic"] is True
    assert by_key["asr_speaker_count"]["visible_when"] == {"asr_diarize_speakers": [True]}
    assert parameters.visible(by_key["asr_speaker_count"], {"asr_diarize_speakers": True})

    rate = 16_000
    src = _write_wav(
        tmp_path / "proj_src.wav",
        _synth_voice(rate, 8.0, f0=220.0, formants=(600.0, 1800.0, 2800.0)),
        rate,
    )
    ext_wav = _write_wav(
        tmp_path / "custom_voice.wav",
        _synth_voice(rate, 2.0, f0=140.0, formants=(450.0, 1300.0, 2300.0)),
        rate,
    )
    proj, proj_dir = pipeline.create_project(src, tmp_path / "projects")
    (proj_dir / "chinese").mkdir(parents=True, exist_ok=True)
    (proj_dir / "chinese" / "s1.wav").write_bytes(b"dub1")
    (proj_dir / "chinese" / "s2.wav").write_bytes(b"dub2")
    proj.sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.2,
            end_seconds=2.8,
            speaker_id="S01",
            source_text="おはよう。",
            zh_text="早安。",
            status="synthesized",
            tts_file="chinese/s1.wav",
        ),
        Sentence(
            id="s000002",
            start_seconds=3.2,
            end_seconds=6.5,
            speaker_id="S02",
            source_text="おやすみ。",
            zh_text="晚安。",
            status="synthesized",
            tts_file="chinese/s2.wav",
        ),
    ]
    manifest = save_project(proj, proj_dir)

    app = Application(tmp_path / "tasks.json")
    srv = Server(("127.0.0.1", 0), app)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{srv.server_port}"
    headers = {"X-ASMR-Token": srv.token}
    try:
        with httpx.Client(trust_env=False) as client:
            snap = client.post(
                url + "/api/projects/get", json={"project": str(manifest)}, headers=headers
            ).json()
            assert [p["speaker_id"] for p in snap["speaker_profiles"]] == ["S01", "S02"]
            assert snap["speaker_profiles"][0]["auto_sentence_id"] == "s000001"
            assert snap["speaker_profiles"][1]["auto_sentence_id"] == "s000002"

            up = client.post(
                url + "/api/uploads",
                content=ext_wav.read_bytes(),
                headers={"X-ASMR-Token": srv.token, "X-Filename": "custom_voice.wav"},
            )
            assert up.status_code == 200
            uploaded_path = up.json()["path"]

            res = client.post(
                url + "/api/projects/speaker-reference",
                json={
                    "project": str(manifest),
                    "revision": snap["revision"],
                    "speaker_id": "S02",
                    "mode": "external",
                    "external": uploaded_path,
                    "external_name": "custom_voice.wav",
                    "text": "你好呀",
                    "language": "zh",
                },
                headers=headers,
            )
            assert res.status_code == 200
            updated = res.json()["project"]
            s2_profile = next(p for p in updated["speaker_profiles"] if p["speaker_id"] == "S02")
            assert s2_profile["mode"] == "external"
            assert s2_profile["audio_name"] == "custom_voice.wav"
            assert s2_profile["audio_url"]
            # S02 dubbing cache cleared, S01 preserved intact.
            assert updated["sentences"][0]["tts_file"] == "chinese/s1.wav"
            assert updated["sentences"][1]["tts_file"] is None

            # Reset S02 back to auto.
            res2 = client.post(
                url + "/api/projects/speaker-reference",
                json={
                    "project": str(manifest),
                    "revision": updated["revision"],
                    "speaker_id": "S02",
                    "mode": "auto",
                },
                headers=headers,
            )
            assert res2.status_code == 200
            reloaded, _ = load_project(manifest)
            assert "S02" not in reloaded.speaker_references
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(3)


def test_frontend_multispeaker_columns_filter_step3_and_translations() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    frontend = files("asmr_dubber").joinpath("frontend")
    html = frontend.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="speakerFilter"' in html
    assert 'id="batchSpeakerBtn"' in html
    assert 'id="speakerVoicesSection"' in html
    thead = html.split("<thead>", 1)[1].split("</thead>", 1)[0]
    assert thead.index("说话人") < thead.index("原文")

    def data_uri(source: str) -> str:
        return "data:text/javascript;base64," + base64.b64encode(source.encode("utf-8")).decode()

    session_uri = data_uri(frontend.joinpath("session.js").read_text(encoding="utf-8"))
    review_uri = data_uri(
        frontend.joinpath("project_review.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
    )
    wave_uri = data_uri(
        frontend.joinpath("waveform_bar.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
    )
    revise_uri = data_uri(
        frontend.joinpath("project_revise.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
        .replace("'./project_review.js'", json.dumps(review_uri))
    )
    editor_uri = data_uri(
        frontend.joinpath("sentence_editor.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
        .replace("'./project_review.js'", json.dumps(review_uri))
        .replace("'./waveform_bar.js'", json.dumps(wave_uri))
    )
    voices_uri = data_uri(
        frontend.joinpath("speaker_voices.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
        .replace("'./sentence_editor.js'", json.dumps(editor_uri))
    )

    locales = catalog()
    script = f"""
const dom = new Map();
function el(id) {{
  if (!dom.has(id)) {{
    dom.set(id, {{
      id, innerHTML: '', textContent: '', hidden: false, disabled: false, value: 'all',
      scrollTop: 0, clientHeight: 480,
      addEventListener() {{}}, classList: {{ contains: () => false }},
    }});
  }}
  return dom.get(id);
}}
el('#search').value = '';
globalThis.window = {{ devicePixelRatio: 1, getSelection: () => null }};
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.document = {{
  documentElement: {{ lang: 'zh-CN' }},
  querySelector: (sel) => el(sel),
  querySelectorAll: () => [],
  getElementById: (id) => el('#' + id),
  addEventListener() {{}},
}};
const {{ state, t }} = await import({json.dumps(session_uri)});
const {{ matchesRowFilter }} = await import({json.dumps(revise_uri)});
const editor = await import({json.dumps(editor_uri)});
const voices = await import({json.dumps(voices_uri)});

state.boot = {{
  labels: {{}},
  locales: {json.dumps(locales, ensure_ascii=False)},
  choices: {{ tts: {{ indextts2: {{ reference: true }}, edge_tts: {{ reference: false }} }} }},
}};
const sentences = [
  {{
    id: 's000001', enabled: true, speaker_id: 'S01', start_seconds: 0, end_seconds: 1,
    source_text: 'A', target_text: '甲', zh_text: '甲', original_audio_enabled: null,
    original_audio_gain_db: 0, chinese_audio_enabled: true, chinese_audio_gain_db: 0,
  }},
  {{
    id: 's000002', enabled: true, speaker_id: 'S02', start_seconds: 1, end_seconds: 2,
    source_text: 'B', target_text: '乙', zh_text: '乙', original_audio_enabled: null,
    original_audio_gain_db: 0, chinese_audio_enabled: true, chinese_audio_gain_db: 0,
  }},
  {{
    id: 's000003', enabled: true, speaker_id: null, start_seconds: 2, end_seconds: 3,
    source_text: 'C', target_text: '丙', zh_text: '丙', original_audio_enabled: null,
    original_audio_gain_db: 0, chinese_audio_enabled: true, chinese_audio_gain_db: 0,
  }},
];
state.project = {{
  source_language: 'ja',
  source: {{ duration_seconds: 5 }},
  settings: {{ tts_backend: 'indextts2', asr_diarize_speakers: true }},
  speaker_references: {{}},
  speaker_profiles: [
    {{
      speaker_id: 'S01', sentence_count: 1, mode: 'auto',
      sentence_id: 's000001', auto_sentence_id: 's000001', audio_url: null,
    }},
    {{
      speaker_id: 'S02', sentence_count: 1, mode: 'external',
      audio_name: 's2.wav', audio_url: '/media/abc',
    }},
  ],
  sentences,
}};
editor.renderVirtualRows(sentences, 's000001', {{ force: true }});
const rowsHtml = el('#rows').innerHTML;
if (
  !rowsHtml.includes('data-row-field="speaker_id"') ||
  !rowsHtml.includes('value="S02" selected')
) {{
  throw new Error('Missing inline speaker selector in table rows: ' + rowsHtml);
}}

state.speakerFilter = 'S02';
if (
  matchesRowFilter(sentences[0]) ||
  !matchesRowFilter(sentences[1]) ||
  matchesRowFilter(sentences[2])
) {{
  throw new Error('Speaker filter S02 failed');
}}
state.speakerFilter = '__none__';
if (matchesRowFilter(sentences[0]) || !matchesRowFilter(sentences[2])) {{
  throw new Error('Speaker filter __none__ failed');
}}
state.speakerFilter = 'all';

voices.renderSpeakerFilter(state.project);
if (el('#speakerFilter').hidden || el('#batchSpeakerBtn').hidden) {{
  throw new Error('Speaker toolbar controls should be visible');
}}
voices.renderSpeakerVoices(state.project);
const spkHtml = el('#speakerVoicesSection').innerHTML;
if (!spkHtml.includes('data-spk-edit="S01"') || !spkHtml.includes('data-spk-reset="S02"')) {{
  throw new Error('Step 3 speaker panel missing expected buttons: ' + spkHtml);
}}

state.language = 'en';
const dialogTitle = t('S01 的参考音色');
if (dialogTitle !== 'Voice reference for S01') {{
  throw new Error('Expected English template translation, got: ' + dialogTitle);
}}
"""
    res = subprocess.run(
        [node, "--input-type=module", "-"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res.returncode == 0, res.stderr

    # Verify focusout ignores <select data-row-field="speaker_id"> so blurring between
    # speaker dropdowns never reads option concatenation textContent ("—S01S02+ 新增…").
    projects_js = frontend.joinpath("projects.js").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "if (!key || !['source_text', 'zh_text'].includes(key)) return;" in projects_js
    assert "if (!key || ['source_text', 'zh_text'].includes(key)) return;" in projects_js
    focusout_block = projects_js.split("  document.addEventListener(\n    'focusout',", 1)[1].split(
        "\n  document.addEventListener(\n    'change',", 1
    )[0]
    focusout_check = f"""
const commits = [];
const state = {{
  project: {{
    sentences: [{{ id: 's1', speaker_id: 'S01', source_text: 'A', tts_candidates: [] }}],
  }},
}};
const guard = (fn) => fn;
const projectBusy = () => false;
const commitSentenceEdit = async (...args) => commits.push(args);
let focusHandler;
const document = {{ addEventListener: (_name, fn) => {{ focusHandler = fn; }} }};
document.addEventListener('focusout', {focusout_block}
await focusHandler({{
  target: {{
    dataset: {{ rowField: 'speaker_id' }},
    textContent: '—S01S02+ 新增…',
    closest: () => ({{ dataset: {{ sentence: 's1' }} }}),
  }},
}});
if (commits.length !== 0 || state.project.sentences[0].speaker_id !== 'S01') {{
  throw new Error('focusout must ignore speaker_id select element');
}}
await focusHandler({{
  target: {{
    dataset: {{ rowField: 'source_text' }},
    textContent: 'B',
    closest: () => ({{ dataset: {{ sentence: 's1' }} }}),
  }},
}});
if (commits.length !== 1) {{
  throw new Error('focusout should commit editable source_text cell');
}}
"""
    res2 = subprocess.run(
        [node, "--input-type=module", "-"],
        input=focusout_check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res2.returncode == 0, res2.stderr
