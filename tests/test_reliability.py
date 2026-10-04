from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from asmr_dubber.api_contracts import APIContractError, write_audio_response
from asmr_dubber.errors import ProjectConflictError, SynthesisError
from asmr_dubber.lifecycle import browser_revision_scope
from asmr_dubber.models import AudioInfo, DubProject, Sentence, load_project, save_project
from asmr_dubber.services import project_audio, project_operations, project_records
from asmr_dubber.tts import tts_cache_key, valid_tts_cache
from asmr_dubber.voice_reference import VoiceReference, reference_selection_scope


def project() -> DubProject:
    return DubProject(
        source=AudioInfo(
            path="source.wav", sha256="a" * 64, duration_seconds=12, sample_rate=16000, channels=1
        ),
        sentences=[
            Sentence(
                id="s1",
                start_seconds=0.2,
                end_seconds=1.8,
                source_text="始めましょう。",
                zh_text="开始吧。",
            ),
            Sentence(
                id="s2",
                start_seconds=2,
                end_seconds=8,
                source_text="参考文章です。",
                zh_text="参考句。",
            ),
        ],
    )


def test_audio_download_has_no_service_credentials(tmp_path):
    observed = []

    def handler(request):
        observed.append(request)
        return httpx.Response(200, content=b"audio")

    with httpx.Client(
        headers={"Authorization": "Bearer FAKE", "api-key": "FAKE"},
        cookies={"session": "FAKE"},
        auth=("user", "password"),
        transport=httpx.MockTransport(handler),
    ) as client:
        write_audio_response(
            httpx.Response(200, json={"audio_url": "https://cdn.example/a"}),
            tmp_path / "audio",
            client=client,
        )
    assert len(observed) == 1
    assert not set(observed[0].headers).intersection({"authorization", "api-key", "cookie"})


@pytest.mark.parametrize("url", ["file:///secret", "https://user:pass@example.com/audio"])
def test_download_rejects_unsafe_url(tmp_path, url):
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("network"))) as client,
        pytest.raises(APIContractError),
    ):
        write_audio_response(
            httpx.Response(200, json={"url": url}), tmp_path / "audio", client=client
        )


def test_stale_browser_revision_preserves_saved_table(tmp_path, monkeypatch):
    p = project()
    save_project(p, tmp_path)
    revision = p.revision
    stale = project_records.project_rows(p)
    edited = project_records.project_rows(p)
    edited[0][5] = "已校对"
    monkeypatch.setattr(project_operations, "view", lambda p, *a: p)
    with browser_revision_scope(str(tmp_path), revision):
        project_operations.save_table(str(tmp_path), edited)
    with pytest.raises(ProjectConflictError), browser_revision_scope(str(tmp_path), revision):
        project_operations.save_table(str(tmp_path), stale)
    assert load_project(tmp_path)[0].sentences[0].zh_text == "已校对"


def test_stale_execution_rejected_before_audio_changes(tmp_path, monkeypatch):
    from asmr_dubber.pipeline import _synthesize_project_impl

    p = project()
    save_project(p, tmp_path)
    stale = p.model_copy(deep=True)
    save_project(p, tmp_path)
    monkeypatch.setattr(
        "asmr_dubber.pipeline.verify_source", lambda *a: pytest.fail("started stale task")
    )
    with pytest.raises(ProjectConflictError):
        _synthesize_project_impl(stale, tmp_path)


@pytest.mark.parametrize("kind", ["fallback", "gpt_seed", "index_vector"])
def test_cache_covers_actual_synthesis_inputs(kind):
    p = project()
    if kind == "fallback":
        p.settings.tts_index_speaker_source = "sentence_reference"
        p.settings.tts_reference_sentence_id = "s2"
    elif kind == "gpt_seed":
        p.settings.tts_backend = "gpt_sovits"
        p.settings.tts_clone_mode = "reference_only"
    else:
        p.settings.tts_index_emotion_source = "vector"
    before = tts_cache_key(p, p.sentences[0])
    if kind == "fallback":
        p.sentences[1].start_seconds = 4
    elif kind == "gpt_seed":
        p.settings.random_seed += 1
    else:
        p.settings.tts_index25_emotion_vector = [0.5] + [0.0] * 7
    assert before != tts_cache_key(p, p.sentences[0])


def test_corrupt_tts_cache_is_not_reused(tmp_path):
    p = project()
    s = p.sentences[0]
    (tmp_path / "bad.wav").write_bytes(b"bad")
    s.tts_file = "bad.wav"
    s.tts_cache_key = tts_cache_key(p, s)
    assert not valid_tts_cache(p, s, tmp_path)


def test_recent_stage_of_old_media_survives_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    source = tmp_path / "old.wav"
    source.write_bytes(b"audio")
    old = time.time() - 48 * 3600
    os.utime(source, (old, old))
    staged = Path(project_audio.stage_for_ui(source))
    project_audio.ui_stage_directory()
    assert staged.is_file()
    assert source.stat().st_mtime == old


def test_subtitle_setting_change_invalidates_subtitles(tmp_path, monkeypatch):
    from asmr_dubber.user_settings import UserSettings

    p = project()
    p.subtitle_srt_file = "subtitles/old.srt"
    save_project(p, tmp_path)
    settings = UserSettings.model_validate(p.settings.model_dump())
    settings.subtitle_timeline = "dubbing" if p.settings.subtitle_timeline == "source" else "source"
    monkeypatch.setattr(project_operations, "view", lambda p, *a: p)
    project_operations.apply_global_settings(str(tmp_path), settings)
    assert load_project(tmp_path)[0].subtitle_srt_file is None


def test_duplicate_id_casefold_rejected():
    p = project()
    p.sentences[1].id = "S1"
    with pytest.raises(ValidationError):
        DubProject.model_validate_json(p.model_dump_json())


def test_performance_failure_does_not_mask_original(tmp_path, monkeypatch):
    from asmr_dubber.performance import measure_stage

    def fail(*a):
        raise PermissionError("log unavailable")

    monkeypatch.setattr("asmr_dubber.performance._append_event", fail)
    with pytest.raises(SynthesisError, match="original"), measure_stage(tmp_path, "tts"):
        raise SynthesisError("original")


def test_reference_index_matches_uncached_selection():
    from asmr_dubber.voice_reference import fallback_reference_sentence

    p = project()
    before = [tts_cache_key(p, s) for s in p.sentences]
    with reference_selection_scope(p):
        assert [tts_cache_key(p, s) for s in p.sentences] == before
        assert fallback_reference_sentence(p, p.sentences[0]) is p.sentences[1]


def test_force_translation_preserves_disabled_text(tmp_path, monkeypatch):
    from asmr_dubber.pipeline import _translate_project_impl

    p = project()
    p.sentences[1].enabled = False
    save_project(p, tmp_path)
    monkeypatch.setattr("asmr_dubber.pipeline.resolve_api_key", lambda *a: "FAKE")

    def translate(sentences, **kw):
        sentences[0].zh_text = "新译文"
        kw["on_batch"]()

    monkeypatch.setattr("asmr_dubber.pipeline.translate_sentences", translate)
    _translate_project_impl(p, tmp_path, force=True)
    assert load_project(tmp_path)[0].sentences[1].zh_text == "参考句。"


def test_queued_source_fingerprint_reads_live_metadata(tmp_path):
    from asmr_dubber.autoflow.engine import AudioSource, fingerprint

    source = tmp_path / "audio.wav"
    source.write_bytes(b"old")
    item = AudioSource(
        order=1,
        path=source,
        relative_path="audio.wav",
        title_ja="audio",
        size=3,
        mtime_ns=source.stat().st_mtime_ns,
    )
    before = fingerprint([item], None)
    source.write_bytes(b"new and longer")
    assert fingerprint([item], None) != before


def test_worker_seed_stable_across_retry_subsets(tmp_path, monkeypatch):
    import json

    from asmr_dubber import indextts25_worker as worker

    (tmp_path / "config.yaml").write_text("{}")
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"fake")
    seeds = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.gpt = SimpleNamespace(inference_speech=lambda *a, **kw: None)

        def infer(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"audio")
            return True

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "indextts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", SimpleNamespace(IndexTTS2=FakeModel))
    monkeypatch.setattr(worker, "_seed_everything", seeds.append)
    tasks = [
        dict(
            id=f"s{i}",
            text="测试",
            voice=str(reference),
            output=str(tmp_path / f"s{i}.wav"),
            seed=42,
        )
        for i in range(1, 3)
    ]
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text("\n".join(json.dumps(t) for t in tasks), encoding="utf-8")
    args = ["--batch-file", str(manifest), "--model-dir", str(tmp_path), "--device", "cpu"]
    assert worker.main(args) == 0
    manifest.write_text(json.dumps(tasks[1]), encoding="utf-8")
    assert worker.main(args) == 0
    assert seeds[1] == seeds[2]


def test_timeout_preserves_completed_clip(tmp_path, monkeypatch):
    from asmr_dubber.tts_backends import _synthesize_indextts25_batch

    p = project()
    p.settings.tts_backend = "indextts2_5"
    p.settings.tts_timeout_seconds = 0.7
    p.settings.tts_index25_model_path = str(tmp_path / "models")
    p.settings.tts_index25_config_path = str(tmp_path / "models/config.yaml")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/config.yaml").write_text("{}")
    (tmp_path / "chinese").mkdir()
    reference = VoiceReference(tmp_path / "ref.wav", "reference", "test")
    sf.write(reference.path, np.ones(1600, dtype=np.float32) * 0.1, 16000)
    code = (
        "import sys,json,shutil,time; from pathlib import Path; "
        "t=json.loads(Path(sys.argv[sys.argv.index('--batch-file')+1])"
        ".read_text(encoding='utf-8').splitlines()[0]); "
        "shutil.copyfile(t['voice'],t['output']); "
        "print('Generated: '+t['id'],flush=True); time.sleep(60)"
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends._indextts25_command", lambda p: [sys.executable, "-c", code]
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_index_speaker_reference", lambda *a: reference
    )
    monkeypatch.setattr("asmr_dubber.tts_backends.prepare_index_emotion_reference", lambda *a: None)
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.isolated_runtime_environment", lambda *a: os.environ.copy()
    )
    saved = []
    with pytest.raises(SynthesisError, match="超过"):
        _synthesize_indextts25_batch(
            p, tmp_path, tmp_path / "source.wav", p.sentences, None, lambda: saved.append(1), None
        )
    assert saved == [1]
    assert valid_tts_cache(p, p.sentences[0], tmp_path)
    assert p.sentences[1].tts_file is None


def test_alignment_reads_only_requested_window(tmp_path):
    from asmr_dubber.forced_alignment import _read_alignment_window

    path = tmp_path / "audio.wav"
    sf.write(path, np.arange(1000, dtype=np.float32) / 1000, 1000, subtype="FLOAT")
    window = _read_alignment_window(path, 100, 200)
    assert window.shape == (100,)
    assert window[0] == pytest.approx(0.1)
    assert window[-1] == pytest.approx(0.199)


def test_disk_preflight_stops_before_writing(tmp_path, monkeypatch):
    from asmr_dubber.errors import ProjectError
    from asmr_dubber.storage import require_disk_space

    monkeypatch.setattr("asmr_dubber.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=10))
    with pytest.raises(ProjectError, match="磁盘空间不足"):
        require_disk_space(tmp_path, 100)


def test_audio_download_limit_and_redirect_policy(tmp_path, monkeypatch):
    monkeypatch.setattr("asmr_dubber.api_contracts.MAX_AUDIO_RESPONSE_BYTES", 100)
    response = httpx.Response(200, json={"url": "https://cdn.example/audio"})
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"a" * 101))
        ) as client,
        pytest.raises(APIContractError, match="上限"),
    ):
        write_audio_response(response, tmp_path / "audio", client=client)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(302, headers={"location": "http://other.example/audio"})
            )
        ) as client,
        pytest.raises(APIContractError, match="重定向"),
    ):
        write_audio_response(response, tmp_path / "audio", client=client)


def test_readable_cache_rechecks_after_file_change(tmp_path):
    p = project()
    sentence = p.sentences[0]
    path = tmp_path / "clip.wav"
    sf.write(path, np.ones(800, dtype=np.float32), 8000)
    sentence.tts_file = path.name
    sentence.tts_cache_key = tts_cache_key(p, sentence)
    assert valid_tts_cache(p, sentence, tmp_path)
    path.write_bytes(b"corrupted")
    assert not valid_tts_cache(p, sentence, tmp_path)
