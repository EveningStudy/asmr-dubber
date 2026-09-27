from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import StemEvent, build_chinese_stem, probe_audio, sentence_events
from asmr_dubber.errors import ProjectError
from asmr_dubber.experimental_mix import compose_replacement_bed
from asmr_dubber.models import DubProject, ProjectSettings, Sentence
from asmr_dubber.tts import tts_cache_key
from asmr_dubber.ui_services import apply_table, project_rows


def project_at(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.full((16000 * 2, 2), 0.01), 16000, subtype="FLOAT")
    return DubProject(
        source=probe_audio(path),
        sentences=[
            Sentence(
                id="s1",
                start_seconds=0.2,
                end_seconds=1.5,
                source_text="原文",
                zh_text="中文",
                tts_file="tts.wav",
                tts_cache_key="keep",
                tts_duration_seconds=0.5,
            )
        ],
    )


def test_table_mix_edits_keep_tts_and_do_not_lock_text(tmp_path):
    p = project_at(tmp_path)
    key = tts_cache_key(p, p.sentences[0])
    rows = project_rows(p)
    rows[0][6:] = ["off", -6, False, -3]
    apply_table(p, rows)
    row = p.sentences[0]
    assert row.original_audio_enabled is False and not row.chinese_audio_enabled
    assert row.tts_file == "tts.wav" and row.tts_cache_key == "keep"
    assert not row.review_locked
    assert tts_cache_key(p, row) == key
    apply_table(p, [project_rows(p)[0][:6]])
    assert p.sentences[0].original_audio_enabled is False


def test_bad_gain_rejected_without_modifying_project(tmp_path):
    p = project_at(tmp_path)
    rows = project_rows(p)
    rows[0][9] = 100
    with pytest.raises(ProjectError):
        apply_table(p, rows)
    assert p.sentences[0].chinese_audio_gain_db == 0


def test_chinese_gain_after_defaults_and_mute(tmp_path):
    p = project_at(tmp_path)
    clip = tmp_path / "tts.wav"
    sf.write(clip, np.full(8000, 0.02), 16000, subtype="FLOAT")
    event = StemEvent("s1", 0.2, clip)
    plain = tmp_path / "plain.wav"
    quieter = tmp_path / "quieter.wav"
    for path, e in ((plain, event), (quieter, replace(event, gain_db=-6))):
        build_chinese_stem(path, [e], p.source, 0, normalize_loudness=False, stem_peak_dbfs=None)
    x, _ = sf.read(plain)
    y, _ = sf.read(quieter)
    np.testing.assert_allclose(y, x * 10 ** (-6 / 20), atol=1e-8)
    p.sentences[0].chinese_audio_enabled = False
    assert sentence_events(tmp_path, p.sentences) == []


@pytest.mark.parametrize("mode,expected", [("bilingual", 0.04), ("replace", 0.04)])
def test_original_sentence_override_controls_voice_only(tmp_path, mode, expected):
    p = project_at(tmp_path)
    p.settings = ProjectSettings(separation_enabled=True, separation_mix_mode=mode)
    vocals, background = tmp_path / "vocals.wav", tmp_path / "background.wav"
    sf.write(vocals, np.full((32000, 2), 0.04), 16000, subtype="FLOAT")
    sf.write(background, np.full((32000, 2), 0.02), 16000, subtype="FLOAT")
    s = p.sentences[0]
    s.original_audio_enabled = True
    s.original_audio_gain_db = 20 * np.log10(0.5)
    data, _ = sf.read(compose_replacement_bed(p, tmp_path, vocals, background))
    np.testing.assert_allclose(data[16000], expected, atol=1e-6)
    s.original_audio_enabled = False
    data, _ = sf.read(compose_replacement_bed(p, tmp_path, vocals, background))
    np.testing.assert_allclose(data[16000], 0.02, atol=1e-6)
