from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline, separation
from asmr_dubber.audio import StemEvent, build_chinese_stem, probe_audio
from asmr_dubber.errors import ProjectError
from asmr_dubber.experimental_mix import compose_replacement_bed, spatialize_clip
from asmr_dubber.models import DubProject, ProjectSettings, Sentence
from asmr_dubber.spatial_cues import rms
from asmr_dubber.spatial_rtf import render_rtf


def test_experiments_default_off_and_validate_parameters():
    s = ProjectSettings()
    assert not s.separation_enabled and not s.spatial_rtf_enabled
    assert s.separation_mix_mode == "bilingual"
    assert s.separation_keep_original == "none"
    with pytest.raises(ValueError):
        ProjectSettings(separation_mdxc_params="[]")


@pytest.mark.parametrize(
    "start,end,silent", [(0, 0.04, False), (0.99, 1.02, False), (2, 3, False), (0, 0.8, True)]
)
def test_rtf_short_or_silent_reference_keeps_chinese(tmp_path, caplog, start, end, silent):
    rate = 16000
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 0.01, rate).astype(np.float32)
    source = tmp_path / "ref.wav"
    sf.write(
        source, np.column_stack((ref, ref * 0.5)) * (0 if silent else 1), rate, subtype="FLOAT"
    )
    mono = rng.normal(0, 0.02, 8000).astype(np.float32)
    result = spatialize_clip(
        mono, rate, StemEvent("s000002", 0, source, start, end), source, ProjectSettings()
    )
    assert result.shape == (len(mono), 2)
    assert np.isfinite(result).all() and rms(result) > 0
    assert "RTF 降级" in caplog.text
    if silent or start >= 1:
        np.testing.assert_allclose(result[:, 0], result[:, 1])
    else:
        assert rms(result[:, 0]) > rms(result[:, 1]) * 1.5


def test_rtf_nan_audio_still_rejected(tmp_path):
    with pytest.raises(ProjectError, match="非有限"):
        spatialize_clip(
            np.array([np.nan]), 16000, StemEvent("s1", 0, tmp_path), tmp_path, ProjectSettings()
        )


def test_mix_dependency_conflict_and_independent_rtf(tmp_path, monkeypatch):
    from asmr_dubber.user_settings import UserSettings, save_user_settings

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    with pytest.raises(ProjectError, match="中文替换需要"):
        save_user_settings(UserSettings(separation_mix_mode="replace"))
    assert not (tmp_path / "config/settings.json").exists()
    ProjectSettings(spatial_rtf_enabled=True).validate_mix_dependencies()
    ProjectSettings(
        separation_enabled=True, separation_mix_mode="replace"
    ).validate_mix_dependencies()


def test_rtf_direction_and_swap_match_lab():
    rng = np.random.default_rng(11)
    signal = rng.normal(0, 0.03, 24000)
    reference = np.column_stack((signal, np.pad(signal, (8, 0))[: len(signal)] * 0.5))
    mono = rng.normal(0, 0.03, len(signal))
    result, _ = render_rtf(mono, reference, 24000)
    swapped, _ = render_rtf(mono, reference[:, ::-1], 24000)
    assert 5 < 20 * np.log10(rms(result[:, 0]) / rms(result[:, 1])) < 7
    np.testing.assert_allclose(result, swapped[:, ::-1], atol=1e-5)


def test_rtf_block_processing_and_stem(tmp_path):
    rate = 16000
    rng = np.random.default_rng(5)
    mono = rng.normal(0, 0.015, rate * 3).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, np.column_stack((mono, mono * 0.5)), rate, subtype="FLOAT")
    clip = tmp_path / "tts.wav"
    sf.write(clip, mono, rate, subtype="FLOAT")
    settings = ProjectSettings(spatial_rtf_enabled=True, spatial_rtf_block_seconds=1)
    event = StemEvent("s1", 0, clip, 0, 3)
    result = spatialize_clip(mono, rate, event, source, settings)
    assert result.shape == (len(mono), 2) and np.isfinite(result).all()
    assert rms(result[:, 0]) > rms(result[:, 1]) * 1.5
    destination = tmp_path / "stem.wav"
    build_chinese_stem(
        destination,
        [event],
        probe_audio(source),
        0,
        normalize_loudness=False,
        spatial_reference_path=source,
        spatial_settings=settings,
    )
    rendered, _ = sf.read(destination, always_2d=True)
    assert rms(rendered[:, 0]) > rms(rendered[:, 1]) * 1.5


@pytest.mark.parametrize("policy,ids", [("none", ""), ("manual", "s1,s2"), ("unvoiced", "")])
def test_replacement_never_doubles_background(tmp_path, policy, ids):
    rate = 16000
    voice = np.full((rate * 3, 2), 0.05, np.float32)
    background = np.full_like(voice, 0.02)
    source = tmp_path / "source.wav"
    vocals = tmp_path / "vocals.wav"
    bed = tmp_path / "background.wav"
    sf.write(source, voice + background, rate, subtype="FLOAT")
    sf.write(vocals, voice, rate, subtype="FLOAT")
    sf.write(bed, background, rate, subtype="FLOAT")
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(
            separation_enabled=True,
            separation_mix_mode="replace",
            separation_keep_original=policy,
            separation_keep_ids=ids,
            separation_keep_padding_ms=0,
        ),
        sentences=[
            Sentence(id="s1", start_seconds=0.2, end_seconds=1.6, source_text="a", zh_text=""),
            Sentence(id="s2", start_seconds=1.2, end_seconds=2.7, source_text="b", zh_text="中文"),
        ],
    )
    result, _ = sf.read(compose_replacement_bed(project, tmp_path, vocals, bed), always_2d=True)
    expected = 0.02 if policy == "none" else 0.07
    np.testing.assert_allclose(result[rate], expected, atol=1e-6)
    assert np.max(result) < 0.071
    np.testing.assert_allclose(result[-rate // 10 :], 0.02, atol=1e-6)


def test_unknown_manual_ids_rejected(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((16000, 2)), 16000)
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(separation_keep_original="manual", separation_keep_ids="bad"),
    )
    with pytest.raises(ProjectError, match="不存在"):
        compose_replacement_bed(project, tmp_path, source, source)


def test_separation_chunks_reconstruct_and_cache(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    rng = np.random.default_rng(8)
    audio = rng.normal(0, 0.01, (44100 * 12, 2)).astype(np.float32)
    sf.write(source, audio, 44100, subtype="FLOAT")
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(separation_enabled=True, separation_chunk_seconds=5),
    )
    monkeypatch.setattr(separation, "verify_local_model", lambda _: "hash")

    def convert(args):
        import shutil

        shutil.copyfile(source, args[-1])

    monkeypatch.setattr(separation, "_run_ffmpeg", convert)
    calls = []

    def fake_run(self, job, timeout):
        calls.append(job)
        data, rate = sf.read(job["input"], always_2d=True)
        out = Path(job["output"]) / "out.wav"
        sf.write(out, data * 0.6, rate, subtype="FLOAT")
        Path(job["result"]).write_text(json.dumps({"vocals": str(out)}))

    monkeypatch.setattr(separation.SeparationSession, "separate", fake_run)
    vocals, background = separation.ensure_separation(project, tmp_path, source)
    assert len(calls) == 3
    v, _ = sf.read(vocals, always_2d=True)
    b, _ = sf.read(background, always_2d=True)
    np.testing.assert_allclose(v + b, audio, atol=1e-8)
    separation.ensure_separation(project, tmp_path, source)
    assert len(calls) == 3
    # Corrupt final output: rebuild safely from checked per-chunk cache.
    vocals.write_bytes(b"broken")
    separation.ensure_separation(project, tmp_path, source)
    assert len(calls) == 3


def test_cloud_requires_opt_in_and_protects_credentials(tmp_path):
    settings = ProjectSettings(separation_backend="replicate")
    with pytest.raises(ProjectError, match="同意"):
        separation._cloud_separate(tmp_path / "audio", tmp_path / "out", settings)
    with pytest.raises(ProjectError):
        separation._download_audio("http://example.com/file", tmp_path / "out")
    with pytest.raises(ProjectError):
        separation.local_model_name("../model.ckpt")


def test_cloud_replicate_success_never_forwards_key_to_output(tmp_path, monkeypatch):
    import httpx

    import asmr_dubber.user_settings as user_settings

    calls = []
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF-test")
    settings = ProjectSettings(
        separation_backend="replicate", separation_cloud_consent=True, separation_api_model="a" * 64
    )
    monkeypatch.setattr(user_settings, "saved_service_key", lambda service: "private-test")

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload["input"]["audio"].startswith("data:audio/wav;base64,")
            assert request.headers["authorization"] == "Bearer private-test"
            return httpx.Response(
                200,
                json={
                    "id": "job-1",
                    "status": "succeeded",
                    "output": {"vocals": "https://files.example/v.wav"},
                },
            )
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"wav-result")

    client_type = httpx.Client
    monkeypatch.setattr(
        separation.httpx,
        "Client",
        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw),
    )
    out = separation._cloud_separate(audio, tmp_path / "out.wav", settings)
    assert out.read_bytes() == b"wav-result" and len(calls) == 2


def test_failed_cloud_submission_not_retried(tmp_path, monkeypatch):
    import httpx

    import asmr_dubber.user_settings as user_settings

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF-test")
    monkeypatch.setattr(user_settings, "saved_service_key", lambda service: "test")
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("timeout")

    client_type = httpx.Client
    monkeypatch.setattr(
        separation.httpx,
        "Client",
        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw),
    )
    with pytest.raises(httpx.ReadTimeout):
        separation._cloud_separate(
            audio,
            tmp_path / "out.wav",
            ProjectSettings(
                separation_backend="replicate",
                separation_cloud_consent=True,
                separation_api_model="a" * 64,
            ),
        )
    assert len(calls) == 1


def test_http_cloud_contract(tmp_path, monkeypatch):
    import httpx

    import asmr_dubber.user_settings as user_settings

    monkeypatch.setattr(user_settings, "saved_service_key", lambda _: "test-token")
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF-test")
    settings = ProjectSettings(
        separation_backend="http",
        separation_cloud_consent=True,
        separation_api_url="https://service.example/separate",
        separation_api_params='{"quality":"high"}',
    )

    def handler(request):
        if request.method == "POST":
            assert b'name="audio"' in request.content
            assert b'"quality": "high"' in request.content
            return httpx.Response(200, json={"vocals": "https://files.example/v.wav"})
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"result")

    client_type = httpx.Client
    monkeypatch.setattr(
        separation.httpx,
        "Client",
        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw),
    )
    assert (
        separation._cloud_separate(audio, tmp_path / "result.wav", settings).read_bytes()
        == b"result"
    )


def test_ui_experimental_controls_and_hydration(tmp_path, monkeypatch):
    import asmr_dubber.ui as ui
    from asmr_dubber.user_settings import UserSettings

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    settings = UserSettings()
    monkeypatch.setattr(ui, "load_user_settings", lambda: settings)
    app = ui.build_app()
    tabs = [c.label for c in app.blocks.values() if type(c).__name__ == "Tab"]
    assert tabs.index("常规") < tabs.index("人声分离（实验性）") < tabs.index("ASR（语音识别）")
    fn = next(f for f in app.fns.values() if f.name == "refresh_settings_form_callback")
    result = fn.fn()
    values = {getattr(c, "label", ""): v for c, v in zip(fn.outputs, result, strict=True)}
    assert values["启用人声分离（实验性，不推荐）"]["value"] is False
    assert values["启用 RTF（原声空间线索迁移）"]["value"] is False


def test_rtf_settings_do_not_invalidate_tts_cache(tmp_path):
    from asmr_dubber.tts import tts_cache_key

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((16000, 2)), 16000)
    project = DubProject(source=probe_audio(source))
    sentence = Sentence(
        id="s1", start_seconds=0, end_seconds=0.8, source_text="こんにちは", zh_text="你好"
    )
    project.sentences = [sentence]
    before = tts_cache_key(project, sentence)
    project.settings.spatial_rtf_enabled = True
    project.settings.separation_mix_mode = "replace"
    assert tts_cache_key(project, sentence) == before


def test_asr_uses_vocals_only_when_enabled(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((16000, 2)), 16000)
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(
            separation_enabled=True, asr_forced_alignment_enabled=False, asr_review_enabled=False
        ),
    )
    vocal = tmp_path / "separated/vocals.wav"
    vocal.parent.mkdir()
    import shutil

    shutil.copyfile(source, vocal)
    monkeypatch.setattr(pipeline, "verify_source", lambda *a: source)
    monkeypatch.setattr(separation, "ensure_separation", lambda *a: (vocal, source))
    seen = []
    monkeypatch.setattr(pipeline, "make_analysis_copy", lambda s, d: seen.append(s) or d)
    monkeypatch.setattr(
        pipeline,
        "transcribe_source",
        lambda *a, **k: (
            [Sentence(id="s1", start_seconds=0, end_seconds=0.8, source_text="こんにちは")],
            "ja",
        ),
    )
    monkeypatch.setattr(pipeline, "save_project", lambda *a: None)
    monkeypatch.setattr(pipeline, "export_transcript", lambda *a: None)
    pipeline._analyze_project_impl(project, tmp_path)
    assert seen == [vocal]
