from pathlib import Path

import pytest

from asmr_dubber import pipeline
from asmr_dubber.autoflow import engine
from asmr_dubber.autoflow.output_policy import (
    configure_project,
    policy_for_settings,
    render_variants,
)
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
    settings_for_source_language,
)
from asmr_dubber.translation import default_translation_prompt
from asmr_dubber.tts import tts_cache_key
from asmr_dubber.user_settings import UserSettings


def project_at(tmp_path, **settings):
    project = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="a" * 64, duration_seconds=2, sample_rate=48000, channels=2
        ),
        settings=ProjectSettings(**settings),
        sentences=[
            Sentence(
                id="s000001", start_seconds=0, end_seconds=1, source_text="Hello.", zh_text="你好。"
            )
        ],
    )
    save_project(project, tmp_path)
    return project, tmp_path / "project.json"


@pytest.mark.parametrize("language", ["ja", "en", "zh"])
def test_source_language_and_prompt(language):
    settings = settings_for_source_language(ProjectSettings(), language)
    if language != "ja":
        assert settings.asr_backend == "faster_whisper"
    prompt = default_translation_prompt(language, "en")
    assert {"ja": "日语原文", "en": "英语原文", "zh": "中文原文"}[language] in prompt
    assert "翻译成英语配音稿" in prompt
    assert UserSettings(default_source_language=language).default_source_language == language


def test_chinese_asr_rejects_english_only_model():
    settings = settings_for_source_language(
        ProjectSettings(asr_backend="faster_whisper", asr_model="small.en"), "zh"
    )
    assert settings.asr_model == "large-v2"


def test_same_language_skips_key_and_preserves_old_translation(tmp_path, monkeypatch):
    project, _ = project_at(tmp_path, tts_target_language="en")
    project.source_language = "en"
    monkeypatch.setattr(pipeline, "resolve_api_key", lambda *args: pytest.fail("no API needed"))
    pipeline._translate_project_impl(project, tmp_path)
    assert project.sentences[0].zh_text == "Hello."
    assert project.translation_language == "en"
    assert list(tmp_path.glob("translations-zh-*.json"))


def test_target_language_changes_tts_cache_but_mix_variant_does_not(tmp_path):
    project, _ = project_at(tmp_path)
    initial = tts_cache_key(project, project.sentences[0])
    project.settings.tts_target_language = "en"
    english = tts_cache_key(project, project.sentences[0])
    assert initial != english
    project.settings.separation_enabled = True
    project.settings.separation_mix_mode = "replace"
    assert english == tts_cache_key(project, project.sentences[0])


@pytest.mark.parametrize("content", ["replacement", "both"])
def test_replacement_preflight(content):
    with pytest.raises(ValueError, match="人声分离"):
        policy_for_settings(ProjectSettings(), content, "bilingual")


def test_subtitles_do_not_run_separation_or_rtf(tmp_path):
    _, path = project_at(tmp_path)
    policy = policy_for_settings(
        ProjectSettings(
            separation_enabled=True, spatial_rtf_enabled=True, tts_target_language="en"
        ),
        "source_subtitles",
        "bilingual",
    )
    configure_project(path, policy, subtitles_only=True)
    project, _ = load_project(path)
    assert not project.settings.separation_enabled
    assert not project.settings.spatial_rtf_enabled
    assert project.settings.tts_target_language == "en"


def test_two_variants_reuse_synthesis_and_preserve_outputs(tmp_path, monkeypatch):
    _, path = project_at(tmp_path, separation_enabled=True, spatial_rtf_enabled=True)
    policy = policy_for_settings(
        ProjectSettings(separation_enabled=True, spatial_rtf_enabled=True), "both", "zh"
    )
    state = {
        "output_policy": policy,
        "mode": "audio",
        "harmonized_delay_seconds": 0,
        "harmonized_volume_db": 0,
    }
    calls = []

    def run(_paths, command, project_path, *args):
        calls.append(command)
        project, directory = load_project(project_path)
        assert project.settings.spatial_rtf_enabled
        variant = project.settings.separation_mix_mode
        output_dir = directory / "output" / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        if command == "mix":
            assert args == ("--output-variant", variant)
            output = output_dir / "mixed.wav"
            output.write_bytes(project.settings.separation_mix_mode.encode())
            project.output_file = output.relative_to(directory).as_posix()
        elif command == "subtitles":
            assert args == ("--language", "zh", "--output-variant", variant)
            for ext in ("srt", "lrc"):
                file = output_dir / f"subtitle.{ext}"
                file.write_text(variant, encoding="utf-8")
                setattr(project, f"subtitle_{ext}_file", file.relative_to(directory).as_posix())
        else:
            pytest.fail(f"unexpected inference: {command}")
        save_project(project, directory)

    monkeypatch.setattr(engine, "run_asmr_cli", run)
    render_variants(None, path, tmp_path / "products", state)
    assert calls == ["mix", "subtitles", "mix", "subtitles"]
    assert Path(state["mix_variants"]["bilingual"]["audio"]).read_bytes() == b"bilingual"
    assert Path(state["mix_variants"]["replace"]["audio"]).read_bytes() == b"replace"
    assert all(Path(value).is_file() for value in state["variant_outputs"].values())
    restored, _ = load_project(path)
    assert restored.settings.separation_mix_mode == "bilingual"
    assert Path(restored.output_file).read_bytes() == b"bilingual"
    assert (tmp_path / restored.subtitle_srt_file).read_text() == "bilingual"
    assert not list((tmp_path / "output").glob("batch-*"))


def test_mix_failure_restores_settings_and_keeps_tts(tmp_path, monkeypatch):
    project, path = project_at(tmp_path, separation_enabled=True)
    project.sentences[0].tts_file = "voice.wav"
    save_project(project, tmp_path)
    policy = policy_for_settings(project.settings, "both", "source")
    state = {"output_policy": policy}

    def fail(*args):
        raise RuntimeError("mix failed")

    monkeypatch.setattr(engine, "run_asmr_cli", fail)
    with pytest.raises(RuntimeError, match="mix failed"):
        render_variants(None, path, tmp_path, state)
    restored, _ = load_project(path)
    assert restored.settings.separation_mix_mode == "bilingual"
    assert restored.sentences[0].tts_file == "voice.wav"
