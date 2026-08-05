from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import sha256_file
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence, load_project
from asmr_dubber.pipeline import mix_project, output_filename, stem_output_filename


def _project_with_cached_voice(project_dir: Path, mode: str) -> DubProject:
    rate = 16_000
    source = project_dir / "source.wav"
    time = np.arange(rate * 2, dtype=np.float32) / rate
    original = (0.03 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    sf.write(source, np.column_stack([original, original]), rate, subtype="FLOAT")

    chinese_dir = project_dir / "chinese"
    chinese_dir.mkdir()
    voice = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)).astype(np.float32)
    sf.write(chinese_dir / "s000001.wav", voice, rate, subtype="FLOAT")

    sentence = Sentence(
        id="s000001",
        start_seconds=0.0,
        end_seconds=1.5,
        source_text="Hello.",
        zh_text="你好。",
        tts_file="chinese/s000001.wav",
        tts_duration_seconds=0.5,
        tts_cache_key="valid-cache",
        status="synthesized",
    )
    settings = ProjectSettings(
        tts_backend="gpt_sovits",
        tts_model="GPT-SoVITS-v4",
        tts_clone_mode="reference_only",
        mix_output_mode=mode,
        normalize_chinese_loudness=False,
        match_source_loudness=False,
        chinese_gain_db=0.0,
        chinese_stem_peak_dbfs=-0.1,
        chinese_dubbing_offset_ms=500,
        mix_peak_protection=False,
    )
    return DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256=sha256_file(source),
            duration_seconds=2.0,
            sample_rate=rate,
            channels=2,
            channel_layout="stereo",
            codec="pcm_f32le",
        ),
        source_language="en",
        settings=settings,
        sentences=[sentence],
    )


@pytest.mark.parametrize(
    ("mode", "has_mixed", "has_stem"),
    [("mixed", True, False), ("stem", False, True), ("both", True, True)],
)
def test_mix_output_modes_create_only_requested_artifacts(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    has_mixed: bool,
    has_stem: bool,
) -> None:
    project = _project_with_cached_voice(tmp_path, mode)
    monkeypatch.setattr("asmr_dubber.pipeline.tts_cache_key", lambda *_args: "valid-cache")

    primary = mix_project(project, tmp_path)

    mixed = tmp_path / "output" / output_filename(project, tmp_path)
    stem = tmp_path / "output" / stem_output_filename(project, tmp_path)
    assert mixed.is_file() is has_mixed
    assert stem.is_file() is has_stem
    assert primary == (mixed if has_mixed else stem)
    assert bool(project.output_file) is has_mixed
    assert bool(project.chinese_stem_file) is has_stem
    loaded, _ = load_project(tmp_path)
    assert bool(loaded.output_file) is has_mixed
    assert bool(loaded.chinese_stem_file) is has_stem


def test_stem_can_be_mixed_back_into_original_without_regenerating_tts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project_with_cached_voice(tmp_path, "stem")
    cache_checks = 0

    def valid_cache(*_args) -> str:
        nonlocal cache_checks
        cache_checks += 1
        return "valid-cache"

    monkeypatch.setattr("asmr_dubber.pipeline.tts_cache_key", valid_cache)
    mix_project(project, tmp_path)
    stem_path = tmp_path / str(project.chinese_stem_file)
    assert stem_path.is_file()
    original_cache = project.sentences[0].tts_cache_key

    project.settings.mix_output_mode = "both"
    mix_project(project, tmp_path)

    assert project.sentences[0].tts_cache_key == original_cache
    assert cache_checks >= 2
    mixed_path = tmp_path / str(project.output_file)
    assert mixed_path.is_file()
    assert (tmp_path / str(project.chinese_stem_file)).is_file()

    original, rate = sf.read(tmp_path / "source.wav", dtype="float32", always_2d=True)
    mixed, mixed_rate = sf.read(mixed_path, dtype="float32", always_2d=True)
    assert mixed_rate == rate
    before_voice = slice(0, int(0.4 * rate))
    during_voice = slice(int(0.6 * rate), int(0.9 * rate))
    np.testing.assert_allclose(mixed[before_voice], original[before_voice], atol=2e-5, rtol=0)
    assert float(np.max(np.abs(mixed[during_voice] - original[during_voice]))) > 0.05
