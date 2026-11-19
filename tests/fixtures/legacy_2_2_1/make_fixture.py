"""Generate genuine ASMR Dubber 2.2.1 projects with the 2.2.1 code itself.

Usage: python make_fixture.py <v2.2.1 src dir> <output dir>
Writes source.wav, chinese/*.wav and a project.json saved by 2.2.1's save_project,
with tts_cache_key values computed by 2.2.1, for three common TTS setups.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(src_dir))

from asmr_dubber import __version__  # noqa: E402
from asmr_dubber.audio import probe_audio  # noqa: E402
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, save_project  # noqa: E402
from asmr_dubber.tts import tts_cache_key  # noqa: E402

assert __version__ == "2.2.1", __version__
RATE = 8_000
SETUPS = {
    "indextts2": {"tts_backend": "indextts2", "tts_model": "IndexTTS2"},
    "edge_tts": {"tts_backend": "edge_tts", "tts_model": "zh-CN-XiaoxiaoNeural"},
    "indextts2_replace": {
        "tts_backend": "indextts2",
        "tts_model": "IndexTTS2",
        "separation_enabled": True,
        "separation_mix_mode": "replace",
    },
}


def tone(seconds: float, freq: float, level: float) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))).astype(np.float32)


def main() -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    for name, overrides in SETUPS.items():
        project_dir = out_dir / name
        (project_dir / "chinese").mkdir(parents=True)
        left = np.concatenate([tone(1.0, 220, 0.2), tone(1.0, 330, 0.05), tone(1.0, 260, 0.2)])
        right = left * 0.5
        source = project_dir / "source.wav"
        sf.write(source, np.column_stack([left, right]), RATE, subtype="PCM_16")
        info = probe_audio(source)
        info = info.model_copy(update={"path": "source.wav"})
        sentences = [
            Sentence(
                id="s000001",
                start_seconds=0.1,
                end_seconds=0.9,
                source_text="おはよう",
                zh_text="早上好",
            ),
            Sentence(
                id="s000002",
                start_seconds=1.1,
                end_seconds=1.9,
                source_text="はぁ……",
                zh_text="",
                chinese_audio_enabled=False,
                status="skipped_sound",
            ),
            Sentence(
                id="s000003",
                start_seconds=2.1,
                end_seconds=2.9,
                source_text="大好きだよ",
                zh_text="最喜欢你了",
                chinese_audio_gain_db=-2.0,
                original_audio_gain_db=-3.0,
            ),
        ]
        project = DubProject(
            source=info, settings=ProjectSettings(**overrides), sentences=sentences
        )
        for sentence in project.sentences:
            if not sentence.zh_text:
                continue
            wav = project_dir / "chinese" / f"{sentence.id}.wav"
            t = np.arange(round(0.6 * 16_000)) / 16_000
            line = (0.25 * np.sin(2 * np.pi * 440 * t) * np.hanning(len(t))).astype(np.float32)
            sf.write(wav, line, 16_000, subtype="PCM_16")
            sentence.tts_file = f"chinese/{sentence.id}.wav"
            sentence.tts_duration_seconds = 0.6
            sentence.tts_cache_key = tts_cache_key(project, sentence)
            sentence.status = "synthesized"
        save_project(project, project_dir)
        digest = hashlib.sha256((project_dir / "project.json").read_bytes()).hexdigest()[:12]
        print(name, "saved", digest)


if __name__ == "__main__":
    main()
