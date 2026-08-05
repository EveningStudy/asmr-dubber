import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import asmr_dubber.pipeline as pipeline
from asmr_dubber.audio import probe_audio
from asmr_dubber.languages import SourceLanguage
from asmr_dubber.models import DubProject, ProjectSettings, Sentence


@pytest.mark.parametrize("source_language", ["ja", "en"])
def test_single_asr_can_run_qwen_forced_alignment_without_multi_asr(
    tmp_path: Path,
    monkeypatch,
    source_language: SourceLanguage,
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(160_000, dtype=np.float32), 16_000, subtype="FLOAT")
    project = DubProject(
        source=probe_audio(source),
        source_language=source_language,
        settings=ProjectSettings(
            asr_backend="faster_whisper",
            asr_model="large-v2",
            asr_review_enabled=False,
            asr_forced_alignment_enabled=True,
        ),
        asr_settings_dirty=True,
    )
    aligned = []

    def fake_transcribe(_audio, _settings, source_language="ja", progress=None):
        assert source_language == project.source_language
        return (
            [
                Sentence(
                    id="s000001",
                    start_seconds=2.0,
                    end_seconds=3.0,
                    ja_text="最初の文章です。",
                ),
                Sentence(
                    id="s000002",
                    start_seconds=4.0,
                    end_seconds=5.0,
                    ja_text="次の文章です。",
                ),
            ],
            "Japanese",
        )

    def fake_align(audio, sentences, settings, progress=None, *, source_language="ja"):
        assert settings.asr_review_enabled is False
        assert source_language == project.source_language
        aligned.append(audio)
        sentences[0].start_seconds = 5.0
        sentences[0].end_seconds = 6.0
        sentences[1].start_seconds = 1.0
        sentences[1].end_seconds = 2.0
        return [{"sentence_id": sentence.id, "fallback": False} for sentence in sentences]

    monkeypatch.setattr(pipeline, "transcribe_source", fake_transcribe)
    monkeypatch.setattr(pipeline, "align_sentences_with_qwen", fake_align)

    pipeline._analyze_project_impl(project, tmp_path)

    assert aligned
    assert project.asr_settings_dirty is False
    assert [sentence.start_seconds for sentence in project.sentences] == [1.0, 5.0]
    assert [sentence.id for sentence in project.sentences] == ["s000001", "s000002"]
    report = json.loads(
        (tmp_path / "analysis" / "asr_forced_alignment.json").read_text(encoding="utf-8")
    )
    assert report["source"] == "faster_whisper|large-v2"
    assert len(report["sentences"]) == 2
