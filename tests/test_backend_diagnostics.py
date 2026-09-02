from pathlib import Path

import numpy as np
import soundfile as sf

import asmr_dubber.backend_diagnostics as diagnostics
from asmr_dubber.audio import sha256_file
from asmr_dubber.models import AudioInfo, DubProject, Sentence, save_project
from asmr_dubber.user_settings import UserSettings


def test_asr_api_probe_uses_form_key_and_cleans_test_audio(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_transcribe(path: Path, settings, _progress, **kwargs):
        captured["exists"] = path.is_file()
        captured["backend"] = settings.asr_backend
        captured.update(kwargs)
        return [], "ja"

    monkeypatch.setattr(diagnostics, "portable_home", lambda: tmp_path / "portable")
    monkeypatch.setattr(diagnostics, "_transcribe_generic_api", fake_transcribe)
    settings = UserSettings(
        asr_backend="generic_asr_api",
        asr_model="remote-asr",
        asr_api_base_url="https://example.test/v1",
    )

    result = diagnostics.test_asr_api(settings, "form-secret")

    assert "测试通过" in result
    assert captured["exists"] is True
    assert captured["backend"] == "generic_asr_api"
    assert captured["api_key"] == "form-secret"
    assert captured["probe_only"] is True
    assert not list((tmp_path / "portable" / "cache" / "api-tests").glob("asr-*"))


def test_translation_api_uses_current_form_values(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_translate(sentences, **kwargs) -> None:
        captured.update(kwargs)
        sentences[0].zh_text = "你好。"

    monkeypatch.setattr(diagnostics, "translate_sentences", fake_translate)
    monkeypatch.setattr(diagnostics, "resolve_api_key", lambda provider, key: key)
    settings = UserSettings(
        default_source_language="en",
        translation_provider="openai_compatible",
        translation_model="translator-test",
        translation_base_url="https://example.test/v1",
    )

    result = diagnostics.test_translation_api(settings, "form-secret")

    assert "Hello. → 你好。" in result
    assert captured["api_key"] == "form-secret"
    assert captured["model"] == "translator-test"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["send_context"] is False


def test_tts_api_uses_project_without_saving_and_cleans_output(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source = project_dir / "source.wav"
    sf.write(source, np.zeros(32_000, dtype=np.float32), 16_000, subtype="FLOAT")
    project = DubProject(
        source=AudioInfo(
            path=source.name,
            sha256=sha256_file(source),
            duration_seconds=2.0,
            sample_rate=16_000,
            channels=1,
        ),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.0,
                end_seconds=1.0,
                source_text="テスト。",
                zh_text="测试。",
            )
        ],
    )
    manifest = save_project(project, project_dir)
    original = manifest.read_bytes()
    captured: dict[str, object] = {}

    def fake_runner(project_value, api_key):
        captured["backend"] = project_value.settings.tts_backend
        captured["api_key"] = api_key

        def run(sentence, _reference, output: Path) -> None:
            captured["text"] = sentence.zh_text
            sf.write(output, np.zeros(8_000, dtype=np.float32), 16_000, subtype="FLOAT")

        return run, lambda: captured.setdefault("cleaned", True)

    monkeypatch.setattr(diagnostics, "portable_home", lambda: tmp_path / "portable")
    monkeypatch.setattr(diagnostics, "_runner", fake_runner)
    settings = UserSettings(
        tts_backend="generic_tts_api",
        tts_model="tts-test",
        tts_api_base_url="https://example.test/v1",
    )

    result = diagnostics.test_tts_api(str(manifest), settings, "form-secret")

    assert "测试通过" in result
    assert captured == {
        "backend": "generic_tts_api",
        "api_key": "form-secret",
        "text": "这是一次语音接口测试。",
        "cleaned": True,
    }
    assert manifest.read_bytes() == original
    assert not list((tmp_path / "portable" / "cache" / "api-tests").glob("tts-*"))
