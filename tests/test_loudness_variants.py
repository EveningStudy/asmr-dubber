import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber.audio import active_rms_dbfs, probe_audio
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, save_project
from asmr_dubber.tts import tts_cache_key


def sentence(**kwargs):
    return Sentence(id="s000001", start_seconds=0, end_seconds=1, source_text="原文", **kwargs)


def test_real_loudness_remix_preserves_tts_and_first_output(tmp_path, monkeypatch):
    rate = 16000
    wave = (np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate) * 0.08).astype("float32")
    source = tmp_path / "source.wav"
    sf.write(source, np.column_stack((wave, wave)), rate, subtype="FLOAT")
    vocals, background = tmp_path / "vocals.wav", tmp_path / "background.wav"
    for path in (vocals, background):
        sf.write(path, np.column_stack((wave * 0.5, wave * 0.5)), rate, subtype="FLOAT")
    clip = tmp_path / "tts.wav"
    sf.write(clip, wave[:rate], rate, subtype="FLOAT")
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(
            tts_backend="edge_tts",
            separation_enabled=True,
            spatial_rtf_enabled=False,
            chinese_target_active_rms_dbfs=-16,
            replacement_chinese_target_active_rms_dbfs=-16,
            chinese_line_peak_dbfs=-1,
            replacement_chinese_line_peak_dbfs=-1,
        ),
        sentences=[sentence(zh_text="你好", tts_file="tts.wav", tts_duration_seconds=1)],
    )
    key = tts_cache_key(project, project.sentences[0])
    project.sentences[0].tts_cache_key = key
    save_project(project, tmp_path)
    monkeypatch.setattr("asmr_dubber.separation.ensure_separation", lambda *a: (vocals, background))
    first = pipeline.mix_project(project, tmp_path, output_variant="bilingual")
    first_bytes = first.read_bytes()
    first_stem = tmp_path / project.chinese_stem_file
    first_stem_bytes = first_stem.read_bytes()
    project.settings.separation_mix_mode = "replace"
    save_project(project, tmp_path)
    pipeline.mix_project(project, tmp_path, output_variant="replace")
    replacement_stem = tmp_path / project.chinese_stem_file
    bil, _ = sf.read(first_stem, dtype="float32")
    repl, _ = sf.read(replacement_stem, dtype="float32")
    difference = active_rms_dbfs(repl[:, 0], rate) - active_rms_dbfs(bil[:, 0], rate)
    assert difference == pytest.approx(8 - 20 * np.log10(2), abs=0.15)
    project.settings.replacement_chinese_relative_loudness_db = -6
    save_project(project, tmp_path)
    pipeline.mix_project(project, tmp_path, output_variant="replace")
    quieter, _ = sf.read(replacement_stem, dtype="float32")
    assert active_rms_dbfs(quieter[:, 0], rate) - active_rms_dbfs(
        repl[:, 0], rate
    ) == pytest.approx(-6, abs=0.15)
    assert first.read_bytes() == first_bytes
    assert first_stem.read_bytes() == first_stem_bytes
    assert tts_cache_key(project, project.sentences[0]) == key
