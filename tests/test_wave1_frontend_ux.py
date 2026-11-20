from __future__ import annotations

import base64
import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.models import DubProject, Sentence, load_project, save_project
from asmr_dubber.pipeline import create_project
from asmr_dubber.services import (
    parameters,
    presets,
    project_audio,
    project_operations,
    project_review,
    settings,
)
from asmr_dubber.services.application import Application


def _make_wav(path: Path, seconds: float = 4.0, rate: int = 16_000) -> Path:
    t = np.arange(round(seconds * rate), dtype=np.float64) / rate
    sig = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(path, np.column_stack([sig, sig]), rate)
    return path


def _make_test_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DubProject, Path, Path]:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    src = _make_wav(tmp_path / "clip.wav", seconds=4.0)
    project, proj_dir = create_project(src, projects_root=tmp_path / "projects")
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
    return project, proj_dir, manifest


def test_media_path_traversal_and_non_audio_external_reference_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, proj_dir, manifest = _make_test_project(tmp_path, monkeypatch)
    outside_audio = _make_wav(tmp_path / "outside.wav", seconds=1.0)
    secrets_file = tmp_path / "home" / "config" / "secrets.json"
    secrets_file.parent.mkdir(parents=True, exist_ok=True)
    secrets_file.write_text('{"api_key": "top-secret"}', encoding="utf-8")

    dub_inside = proj_dir / "tts" / "s000002.wav"
    dub_inside.parent.mkdir(parents=True, exist_ok=True)
    _make_wav(dub_inside, seconds=1.0)

    project.sentences[0].tts_file = "../outside.wav"
    project.sentences[0].tts_candidates = [
        {"attempt": 1, "file": "../outside.wav", "duration": 1.0, "warning": None},
        {"attempt": 2, "file": "tts/s000002.wav", "duration": 1.0, "warning": None},
    ]
    project.sentences[1].tts_file = "tts/s000002.wav"
    project.settings.tts_external_reference_audio = str(secrets_file)
    save_project(project, proj_dir)

    app = Application(tmp_path / "tasks.json")
    snapshot = app.projects.get(str(manifest))

    assert snapshot["sentences"][0]["tts_url"] is None
    assert snapshot["sentences"][0]["tts_candidates"][0]["url"] is None
    assert snapshot["sentences"][0]["tts_candidates"][1]["url"].startswith("/media/")
    assert snapshot["sentences"][1]["tts_url"].startswith("/media/")
    assert snapshot["external_reference_url"] is None

    # Valid audio file external reference IS allowed and registered.
    project.settings.tts_external_reference_audio = str(outside_audio)
    save_project(project, proj_dir)
    snapshot2 = app.projects.get(str(manifest))
    assert snapshot2["external_reference_url"].startswith("/media/")


def test_fast_sentence_save_skips_reference_picker_and_reports_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _proj_dir, manifest = _make_test_project(tmp_path, monkeypatch)
    app = Application(tmp_path / "tasks.json")

    def boom(*args, **kwargs):
        raise AssertionError("Projects.sentence() must not invoke reference_picker()")

    monkeypatch.setattr(project_audio, "reference_picker", boom)
    row = ["s000001", True, 0.2, 1.5, "おはよう", "早安", "default", 0.0, True, 0.0]
    res = app.save_sentence(str(manifest), row, project.revision)
    assert res["deleted"] is False
    assert res["row"]["id"] == "s000001"
    assert res["row"]["target_text"] == "早安"
    assert res["rows"][0][5] == "早安"

    empty_row = ["s000001", True, 0.2, 1.5, "", "", "default", 0.0, True, 0.0]
    deleted_res = app.save_sentence(str(manifest), empty_row, res["revision"])
    assert deleted_res["deleted"] is True
    assert deleted_res["row"] is None
    assert len(deleted_res["sentences"]) == 1


def test_recommended_reference_id_cache_ignores_translation_and_tts_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, proj_dir, _manifest = _make_test_project(tmp_path, monkeypatch)
    project_audio._scored_reference_id.cache_clear()
    calls = 0

    def fake_pick(proj, source):
        nonlocal calls
        calls += 1
        return proj.sentences[0] if proj.sentences else None

    import asmr_dubber.voice_reference as voice_reference

    monkeypatch.setattr(voice_reference, "pick_reference_sentence", fake_pick)
    first = project_audio.recommended_reference_id(project, proj_dir)
    assert first == "s000001"
    assert calls == 1

    # Editing translation text, status, tts_file, or gains must reuse the LRU entry.
    project.sentences[0].target_text = "改了译文"
    project.sentences[0].status = "synthesized"
    project.sentences[0].tts_file = "tts/s000001.wav"
    project.sentences[0].chinese_audio_gain_db = -3.0
    second = project_audio.recommended_reference_id(project, proj_dir)
    assert second == "s000001"
    assert calls == 1

    # Editing source_text invalidates the cache key.
    project.sentences[0].source_text = "こんにちは"
    project_audio.recommended_reference_id(project, proj_dir)
    assert calls == 2


def test_select_autoflow_project_reference_preserves_dub_when_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, proj_dir, manifest = _make_test_project(tmp_path, monkeypatch)
    monkeypatch.setattr(project_audio, "reference_preview", lambda *args: None)
    project.sentences[0].tts_file = "tts/s000001.wav"
    project.sentences[0].tts_cache_key = "abc123"
    project.output_file = "output/out.wav"
    save_project(project, proj_dir)

    project_audio.select_autoflow_project_reference(
        str(manifest),
        "s000001",
        project.sentences[0].start_seconds,
        project.sentences[0].end_seconds,
        project.sentences[0].source_text,
    )
    reloaded, _ = load_project(manifest)
    assert reloaded.sentences[0].tts_file == "tts/s000001.wav"
    assert reloaded.sentences[0].tts_cache_key == "abc123"
    assert reloaded.output_file == "output/out.wav"

    # End tolerance +0.25s works, and external reference invalidates outputs.
    project_audio.select_autoflow_project_reference(
        str(manifest), "s000002", 1.7, project.source.duration_seconds + 0.1, "おやすみ"
    )
    ext = _make_wav(tmp_path / "ext.wav", seconds=1.0)
    project_audio.select_autoflow_external_reference(str(manifest), ext, text="hi", language="ja")
    after_ext, _ = load_project(manifest)
    assert after_ext.output_file is None


def test_separation_mix_mode_replace_auto_enables_separation_and_corrupt_interface_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _proj_dir, manifest = _make_test_project(tmp_path, monkeypatch)
    interface_path = tmp_path / "home" / "config" / "interface.json"
    interface_path.parent.mkdir(parents=True, exist_ok=True)
    interface_path.write_text("{corrupt json", encoding="utf-8")
    assert settings.current().ui_language == "zh"

    # Global scope auto-enables separation_enabled
    updated_global = settings.update({"separation_mix_mode": "replace"})
    assert updated_global["separation_enabled"] is True

    # Project scope auto-enables separation_enabled
    settings.update({"separation_mix_mode": "replace"}, str(manifest), project.revision)
    active, _ = load_project(manifest)
    assert active.settings.separation_mix_mode == "replace"
    assert active.settings.separation_enabled is True

    # The bilingual preset leaves vocal separation as the user set it
    bilingual = next(p for p in presets.catalog() if p["id"] == "bilingual")
    assert "separation_enabled" not in bilingual["changes"]
    assert bilingual["changes"]["separation_mix_mode"] == "bilingual"

    # Explicit separation_enabled=False is still rejected
    with pytest.raises(Exception, match="人声分离"):
        settings.update(
            {"separation_mix_mode": "replace", "separation_enabled": False},
            str(manifest),
            active.revision,
        )


def test_project_review_passes_recognized_residual(tmp_path: Path) -> None:
    src = _make_wav(tmp_path / "source.wav", seconds=2.0)
    from asmr_dubber.audio import probe_audio

    project = DubProject(
        source=probe_audio(src),
        sentences=[
            Sentence(
                id="s1",
                start_seconds=0.0,
                end_seconds=1.0,
                source_text="a",
                target_text="啊",
                tts_file="tts/s1.wav",
            )
        ],
    )
    report = tmp_path / "mix" / "replacement-report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "sentences": [
                    {
                        "id": "s1",
                        "exceeds": True,
                        "reasons": ["residual_speech"],
                        "residual_db": -12.0,
                        "recognized_residual": "もしもし",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    flags = project_review.replacement_flags(project, tmp_path)
    assert flags["s1"]["recognized_residual"] == "もしもし"


def test_parameter_layout_visibility_and_hidden_rules() -> None:
    by_key = {item["key"]: item for item in parameters.catalog()}
    for key in (
        "asr_vad_filter",
        "asr_review_text_priority_model",
        "asr_review_timestamp_priority_model",
        "asr_review_background",
        "asr_review_prompt",
    ):
        assert by_key[key].get("hidden") is True

    for key in (
        "asr_asmr_vad_threshold",
        "asr_asmr_vad_min_speech_ms",
        "asr_asmr_vad_min_silence_ms",
        "asr_asmr_vad_speech_pad_ms",
    ):
        assert parameters.visible(
            by_key[key], {"asr_vad_mode": "off", "asr_backend": "anime_whisper"}
        )
        assert parameters.visible(by_key[key], {"asr_vad_mode": "off", "asr_backend": "qwen3_asr"})
        assert parameters.visible(
            by_key[key], {"asr_vad_mode": "asmr", "asr_backend": "faster_whisper"}
        )
        assert not parameters.visible(
            by_key[key], {"asr_vad_mode": "off", "asr_backend": "faster_whisper"}
        )

    for key in (
        "replacement_chinese_gain_db",
        "replacement_normalize_chinese_loudness",
        "replacement_chinese_relative_loudness_db",
    ):
        assert not parameters.visible(by_key[key], {"separation_mix_mode": "bilingual"})
        assert parameters.visible(by_key[key], {"separation_mix_mode": "replace"})


def test_project_operations_same_language_and_target_language_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, proj_dir, manifest = _make_test_project(tmp_path, monkeypatch)
    # _ASR_AFFECTING_SETTINGS excludes separation_* and translation_*
    assert "separation_mix_mode" not in project_operations._ASR_AFFECTING_SETTINGS
    assert "translation_temperature" not in project_operations._ASR_AFFECTING_SETTINGS
    assert "asr_backend" in project_operations._ASR_AFFECTING_SETTINGS

    # Same-language project skips resolve_api_key
    project.source_language = "zh"
    project.settings.tts_target_language = "zh"
    project.sentences[0].status = "skipped_sound"
    project.sentences[1].tts_candidates = [{"attempt": 1, "file": "tts/s2.wav"}]
    project.sentences[1].review_locked = True
    project.sentences[1].translation_warning = "warn"
    project.translation_proposals = [{"id": "s000002", "old_zh": "a", "new_zh": "b"}]
    save_project(project, proj_dir)
    assert project_operations._translation_api_key(project) == ""

    # Changing target language clears candidates/review/warnings/proposals, keeps skipped_sound
    app = Application(tmp_path / "tasks.json")
    app.update_settings({"tts_target_language": "en"}, str(manifest), project.revision)
    reloaded, _ = load_project(manifest)
    assert reloaded.translation_proposals == []
    assert reloaded.sentences[0].status == "skipped_sound"
    assert reloaded.sentences[1].status == "pending"
    assert reloaded.sentences[1].tts_candidates == []
    assert reloaded.sentences[1].review_locked is False
    assert reloaded.sentences[1].translation_warning is None


def test_frontend_js_syntax_and_behavior_in_node() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    frontend = files("asmr_dubber").joinpath("frontend")
    for js_file in sorted(Path(str(frontend)).glob("*.js")):
        res = subprocess.run(
            [node, "--check", str(js_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert res.returncode == 0, f"{js_file.name}: {res.stderr}"

    def data_uri(source: str) -> str:
        return "data:text/javascript;base64," + base64.b64encode(source.encode("utf-8")).decode()

    session_uri = data_uri(frontend.joinpath("session.js").read_text(encoding="utf-8"))
    review_uri = data_uri(
        frontend.joinpath("project_review.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
    )
    revise_uri = data_uri(
        frontend.joinpath("project_revise.js")
        .read_text(encoding="utf-8")
        .replace("'./session.js'", json.dumps(session_uri))
        .replace("'./project_review.js'", json.dumps(review_uri))
    )
    script = f"""
const els = new Map([['#search', {{ value: 's02' }}]]);
globalThis.document = {{
  documentElement: {{ lang: 'zh-CN' }},
  querySelector: (s) => els.get(s) || {{ value: '' }},
  querySelectorAll: () => [],
  addEventListener() {{}},
}};
const {{ state }} = await import({json.dumps(session_uri)});
const {{ matchesRowFilter }} = await import({json.dumps(revise_uri)});
const {{ flagsCell }} = await import({json.dumps(review_uri)});
state.boot = {{ labels: {{}}, locales: {{}} }};
state.tasks = [];
state.filter = 'all';
const s1 = {{ id: 's1', speaker_id: 'S02', enabled: true, source_text: 'hi', target_text: '' }};
const s2 = {{ id: 's2', speaker_id: 'S01', enabled: false, source_text: 'bye', target_text: '' }};
if (!matchesRowFilter(s1)) throw new Error('Expected search to match speaker_id');
els.set('#search', {{ value: '' }});
state.filter = 'translation';
if (!matchesRowFilter(s1) || matchesRowFilter(s2)) {{
  throw new Error('Disabled row leaked into translation filter');
}}
state.project = {{
  replacement_report: {{
    s1: {{ reasons: ['residual_speech'], recognized_residual: 'こんにちは' }},
  }},
}};
const html = flagsCell(s1);
if (!html.includes('こんにちは') || html.includes('0.0')) {{
  throw new Error('Bad residual_speech HTML: ' + html);
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
