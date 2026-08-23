"""Small end-to-end checks for the workflows users actually run.

These tests deliberately create real media files and inspect the files written
by the pipeline.  They do not call cloud services or load a large model; the
sentence TTS cache is prepared with a short deterministic WAV so failures in
project import, mixing, subtitle generation, or the CLI still surface.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from asmr_dubber.cli import app
from asmr_dubber.models import load_project, save_project
from asmr_dubber.pipeline import create_project
from asmr_dubber.tts import tts_cache_key

runner = CliRunner()


def _source(path: Path, seconds: float = 1.5) -> None:
    sample_rate = 16_000
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    sf.write(path, (0.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)


@pytest.mark.integration
def test_cli_workflow_writes_real_mix_and_subtitles(tmp_path: Path, monkeypatch) -> None:
    """create → import Chinese timed subtitles → mix → subtitles."""

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "app-data"))
    source = tmp_path / "中文源音频.wav"
    _source(source)
    _project, project_dir = create_project(source, projects_root=tmp_path / "projects")

    transcript = tmp_path / "字幕.srt"
    transcript.write_text(
        "1\n00:00:00,100 --> 00:00:00,700\n你好。\n\n"
        "2\n00:00:00,800 --> 00:00:01,400\n欢迎使用。\n",
        encoding="utf-8",
    )
    import_result = runner.invoke(
        app,
        [
            "import-transcript",
            str(project_dir / "project.json"),
            str(transcript),
            "--kind",
            "zh",
        ],
    )
    assert import_result.exit_code == 0, import_result.stdout
    project, _manifest = load_project(project_dir / "project.json")
    assert project.source_language == "zh"
    assert len(project.sentences) == 2

    project.settings.tts_backend = "edge_tts"
    for sentence in project.sentences:
        sentence_path = project_dir / "chinese" / f"{sentence.id}.wav"
        sentence_path.parent.mkdir(parents=True, exist_ok=True)
        _source(sentence_path, seconds=0.25)
        sentence.tts_file = sentence_path.relative_to(project_dir).as_posix()
        sentence.tts_cache_key = tts_cache_key(project, sentence)
        sentence.tts_duration_seconds = 0.25
    save_project(project, project_dir)

    run_result = runner.invoke(
        app,
        [
            "run",
            str(project_dir / "project.json"),
            "--start",
            "mix",
            "--stop",
            "subtitles",
            "--subtitle-language",
            "bilingual",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout

    output = project_dir / "output"
    subtitles = project_dir / "subtitles"
    assert any(path.suffix == ".wav" for path in output.iterdir())
    assert (subtitles / "subtitles_bilingual.srt").is_file()
    assert (subtitles / "subtitles_bilingual.lrc").is_file()
    mixed, _ = load_project(project_dir / "project.json")
    assert mixed.output_file
    assert mixed.subtitle_srt_file


@pytest.mark.integration
def test_cli_settings_round_trip_is_local_and_machine_readable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "app-data"))

    shown = runner.invoke(app, ["settings", "show"])
    assert shown.exit_code == 0, shown.stdout
    payload = json.loads(shown.stdout)
    assert "tts_backend" in payload

    changed = runner.invoke(app, ["settings", "set", "tts_backend", "edge_tts"])
    assert changed.exit_code == 0, changed.stdout
    assert "tts_backend" in changed.stdout

    shown_again = runner.invoke(app, ["settings", "show"])
    assert json.loads(shown_again.stdout)["tts_backend"] == "edge_tts"
    assert not list((tmp_path / "app-data").parent.glob("secrets.json"))


@pytest.mark.linux
def test_linux_scripts_are_syntactically_valid() -> None:
    if os.name == "nt":
        # bash is still available in WSL on the development machine, but this
        # check should remain portable for Windows contributors without WSL.
        bash = subprocess.run(["where", "bash"], capture_output=True, text=True)
        if bash.returncode != 0:
            return
        command = ["bash"]
    else:
        command = ["bash"]
    root = Path(__file__).parents[1]
    scripts = sorted((root / "scripts" / "linux").glob("*.sh"))
    assert scripts
    for script in scripts:
        result = subprocess.run(
            [*command, "-n", script.relative_to(root).as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


@pytest.mark.integration
@pytest.mark.slow
def test_autoflow_self_test_is_a_real_media_smoke(monkeypatch) -> None:
    """Run the local AutoFlow media pipeline without network or models."""

    from asmr_dubber.autoflow.engine import main

    monkeypatch.setenv("ASMR_DUBBER_ROOT", str(Path(__file__).parents[1]))
    assert main(["--self-test"]) == 0
