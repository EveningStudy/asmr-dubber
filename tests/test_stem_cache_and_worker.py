import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import separation
from asmr_dubber.audio import StemEvent, build_chinese_stem, probe_audio
from asmr_dubber.errors import OperationCancelledError, ProjectError
from asmr_dubber.models import ProjectSettings
from asmr_dubber.stem_cache import build_cached_stem, variant_directory
from asmr_dubber.task_control import cancellation_scope


def test_real_rtf_cache_reuse_and_invalidation(tmp_path):
    audio = np.random.default_rng(17).normal(0, 0.01, 8000).astype(np.float32)
    source, clip = tmp_path / "source.wav", tmp_path / "clip.wav"
    sf.write(source, np.column_stack((audio, audio * 0.5)), 16000, subtype="FLOAT")
    sf.write(clip, audio, 16000, subtype="FLOAT")
    settings = ProjectSettings(spatial_rtf_enabled=True)
    arguments = dict(
        destination=tmp_path / "stem.wav",
        events=[StemEvent("s1", 0, clip, 0, 0.5)],
        source_info=probe_audio(source),
        chinese_gain_db=0,
        spatial_reference_path=source,
        spatial_settings=settings,
    )
    calls = []

    def build(**kwargs):
        calls.append(1)
        return build_chinese_stem(**kwargs)

    build_cached_stem(build, **arguments)
    initial = arguments["destination"].read_bytes()
    settings.separation_mix_mode = "replace"
    build_cached_stem(build, **arguments)
    assert len(calls) == 1 and arguments["destination"].read_bytes() == initial
    settings.spatial_rtf_strength = 0.3
    build_cached_stem(build, **arguments)
    assert len(calls) == 2
    arguments["events"] = [replace(arguments["events"][0], gain_db=-3)]
    build_cached_stem(build, **arguments)
    assert len(calls) == 3
    sf.write(clip, audio * 0.8, 16000, subtype="FLOAT")
    build_cached_stem(build, **arguments)
    assert len(calls) == 4
    arguments["destination"].write_bytes(b"corrupt")
    build_cached_stem(build, **arguments)
    assert len(calls) == 5
    arguments["events"] = [replace(arguments["events"][0], source_end_seconds=0.4)]
    build_cached_stem(build, **arguments)
    assert len(calls) == 6


def test_variant_paths_cannot_escape(tmp_path):
    assert variant_directory(tmp_path, "replace") == tmp_path / "output/replace"
    with pytest.raises(ValueError):
        variant_directory(tmp_path, "../../outside")


def test_real_pipeline_variants_and_subtitles_share_rtf(tmp_path, monkeypatch):
    from asmr_dubber import pipeline
    from asmr_dubber.audio import _run_ffmpeg
    from asmr_dubber.models import Sentence, save_project
    from asmr_dubber.tts import tts_cache_key

    source = tmp_path / "original.wav"
    wave = np.random.default_rng(31).normal(0, 0.02, (32000, 2)).astype(np.float32)
    sf.write(source, wave, 16000, subtype="FLOAT")
    video = tmp_path / "original.mp4"
    _run_ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=10:d=2",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
    )
    project, directory = pipeline.create_project(
        video,
        projects_root=tmp_path / "projects",
        settings=ProjectSettings(
            tts_backend="edge_tts",
            spatial_rtf_enabled=True,
            separation_enabled=True,
            mix_output_mode="mixed",
        ),
    )
    clip = directory / "voice.wav"
    sf.write(clip, wave[:8000, 0], 16000, subtype="FLOAT")
    sentence = Sentence(
        id="s1",
        start_seconds=0.1,
        end_seconds=1,
        source_text="Hello",
        zh_text="你好",
        tts_file="voice.wav",
        tts_duration_seconds=0.5,
    )
    project.sentences = [sentence]
    sentence.tts_cache_key = tts_cache_key(project, sentence)
    save_project(project, directory)
    vocals, background = directory / "vocals.wav", directory / "background.wav"
    sf.write(vocals, wave * 0.6, 16000, subtype="FLOAT")
    sf.write(background, wave * 0.4, 16000, subtype="FLOAT")
    monkeypatch.setattr(separation, "ensure_separation", lambda *args: (vocals, background))
    builder = pipeline.build_chinese_stem
    calls = []

    def traced(**kwargs):
        calls.append(1)
        return builder(**kwargs)

    monkeypatch.setattr(pipeline, "build_chinese_stem", traced)
    first_outputs = {}
    for variant in ("bilingual", "replace"):
        project.settings.separation_mix_mode = variant
        save_project(project, directory)
        path = pipeline.mix_project(project, directory, output_variant=variant)
        srt, lrc, subtitle_video = pipeline.generate_subtitles(
            project, directory, output_variant=variant
        )
        assert path.parent == directory / "output" / variant
        assert (directory / project.output_video_file).parent == path.parent
        assert subtitle_video.parent == path.parent
        assert srt.parent == directory / "subtitles" / variant
        for output in (path, srt, lrc, subtitle_video, directory / project.output_video_file):
            assert output.is_file()
        if variant == "bilingual":
            first_outputs = {p: p.read_bytes() for p in (path, srt, lrc, subtitle_video)}
    assert len(calls) == 1
    assert all(p.read_bytes() == contents for p, contents in first_outputs.items())
    assert not list((directory / "output").glob("*.wav"))


@pytest.fixture
def worker_setup(tmp_path, monkeypatch):
    package = tmp_path / "audio_separator"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "separator.py").write_text("""
import os, time
from pathlib import Path
import soundfile as sf
class Separator:
    def __init__(self, **kwargs):
        self.output_dir = kwargs['output_dir']
        self.model_instance = self
    def load_model(self, name):
        with open(os.environ['TEST_LOADS'], 'a') as f: f.write('load\\n')
    def separate(self, path):
        if 'slow' in path: time.sleep(60)
        if 'fail' in path: raise RuntimeError('test worker failure')
        data, rate = sf.read(path)
        out = Path(self.output_dir) / 'audio_(Vocals).wav'
        sf.write(out, data * .5, rate, subtype='FLOAT')
        return [str(out)]
""")
    monkeypatch.setattr(separation, "runtime_python", lambda: Path(sys.executable))
    monkeypatch.setattr(separation, "portable_home", lambda: tmp_path)
    monkeypatch.setattr(
        separation,
        "worker_environment",
        lambda: {
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "TEST_LOADS": str(tmp_path / "loads.txt"),
        },
    )

    def job(name):
        folder = tmp_path / name
        folder.mkdir()
        source = folder / "input.wav"
        sf.write(source, np.zeros((1600, 2)), 16000)
        return dict(
            input=str(source),
            output=str(folder),
            result=str(folder / "result.json"),
            device="auto",
            models=str(tmp_path / "models"),
            model="fake",
            params={},
        )

    return job


def test_persistent_worker_loads_once_for_multiple_chunks(tmp_path, worker_setup):
    with separation.SeparationSession(tmp_path) as session:
        for name in ("one", "two"):
            job = worker_setup(name)
            session.separate(job, 15)
            assert Path(json.loads(Path(job["result"]).read_text())["vocals"]).is_file()
        process = session.process
    assert process.poll() is not None
    assert (tmp_path / "loads.txt").read_text().splitlines() == ["load"]
    assert not list(tmp_path.glob("session-*"))


def test_worker_timeout_and_failure_stop_process(tmp_path, worker_setup):
    with (
        pytest.raises(ProjectError, match="超时"),
        separation.SeparationSession(tmp_path) as session,
    ):
        session.separate(worker_setup("slow"), 0.4)
    assert session.process.poll() is not None
    with (
        pytest.raises(ProjectError, match="test worker failure"),
        separation.SeparationSession(tmp_path) as session,
    ):
        session.separate(worker_setup("fail"), 15)
    assert session.process.poll() is not None


def test_worker_cancel_keeps_completed_results(tmp_path, worker_setup):
    event = threading.Event()
    with (
        cancellation_scope(event),
        pytest.raises(OperationCancelledError),
        separation.SeparationSession(tmp_path) as session,
    ):
        completed = worker_setup("completed")
        session.separate(completed, 15)
        timer = threading.Timer(0.3, event.set)
        timer.start()
        try:
            session.separate(worker_setup("slow"), 15)
        finally:
            timer.cancel()
    assert Path(completed["result"]).is_file()
    assert session.process.poll() is not None
