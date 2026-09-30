"""Opt-in smoke check using an already installed model and a short local excerpt."""

import os

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import separation
from asmr_dubber.audio import _run_ffmpeg, probe_audio
from asmr_dubber.models import DubProject, ProjectSettings


@pytest.mark.skipif(not os.environ.get("ASMR_TEST_SOURCE"), reason="local model smoke is opt-in")
def test_installed_separator_reuses_process(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    _run_ffmpeg(
        [
            "-y",
            "-ss",
            "60",
            "-i",
            os.environ["ASMR_TEST_SOURCE"],
            "-t",
            "11",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_f32le",
            str(source),
        ]
    )
    project = DubProject(
        source=probe_audio(source),
        settings=ProjectSettings(
            separation_enabled=True, separation_chunk_seconds=5, separation_device="cuda"
        ),
    )
    original = separation.SeparationSession.separate
    process_ids = []

    def traced(self, job, timeout):
        original(self, job, timeout)
        process_ids.append(self.process.pid)

    monkeypatch.setattr(separation.SeparationSession, "separate", traced)
    vocals, background = separation.ensure_separation(project, tmp_path, source)
    assert len(process_ids) == 3 and len(set(process_ids)) == 1
    v, rate = sf.read(vocals, always_2d=True)
    b, _ = sf.read(background, always_2d=True)
    raw, _ = sf.read(source, always_2d=True)
    np.testing.assert_allclose(v + b, raw, atol=1e-7)
    separation.ensure_separation(project, tmp_path, source)
    assert len(process_ids) == 3
    print(f"Real model: 3 chunks, 1 process, {len(v) / rate:.1f}s audio, cache hit verified")
