"""Projects written by the released 2.2.1 code open in 3.0 without losing work (W27)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber.models import PROJECT_SCHEMA_VERSION, load_project, save_project
from asmr_dubber.tts import tts_cache_key

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_2_2_1"
CASES = ("indextts2", "edge_tts", "indextts2_replace")


def _copy(case: str, tmp_path: Path) -> Path:
    target = tmp_path / case
    shutil.copytree(FIXTURES / case, target)
    return target


@pytest.mark.parametrize("case", CASES)
def test_2_2_1_project_keeps_text_mix_choices_and_dubs(case: str, tmp_path: Path) -> None:
    directory = _copy(case, tmp_path)
    before = json.loads((directory / "project.json").read_text(encoding="utf-8"))
    assert (before["schema_version"], before["app_version"]) == (3, "2.2.1")

    project, project_dir = load_project(directory)

    for old, new in zip(before["sentences"], project.sentences, strict=True):
        assert (new.id, new.source_text, new.zh_text) == (
            old["id"],
            old["source_text"],
            old["zh_text"],
        )
        assert new.chinese_audio_enabled == old.get("chinese_audio_enabled", True)
        assert new.chinese_audio_gain_db == old.get("chinese_audio_gain_db", 0)
        assert new.original_audio_gain_db == old.get("original_audio_gain_db", 0)
        assert new.status == old["status"]
        assert new.tts_file == old.get("tts_file")
    # No dub the user already has may be reported as outdated.
    for sentence in project.sentences:
        if sentence.tts_file:
            assert tts_cache_key(project, sentence) == sentence.tts_cache_key, sentence.id
    for key, value in before["settings"].items():
        if key.endswith("_params"):
            assert json.loads(getattr(project.settings, key)) == json.loads(value), key
        else:
            assert getattr(project.settings, key) == value, key

    save_project(project, project_dir)
    after = json.loads((directory / "project.json").read_text(encoding="utf-8"))
    assert after["schema_version"] == PROJECT_SCHEMA_VERSION
    backups = list((directory / "backups").glob("project-schema-v3-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == before


def test_2_2_1_bilingual_project_mixes_without_resynthesis(tmp_path: Path) -> None:
    directory = _copy("indextts2", tmp_path)
    project, project_dir = load_project(directory)
    dubs = {p.name: p.read_bytes() for p in (directory / "chinese").glob("*.wav")}

    output = pipeline.mix_project(project, project_dir)

    assert output.is_file()
    mixed, _ = sf.read(output, dtype="float32", always_2d=True)
    assert mixed.shape[1] == 2 and float(abs(mixed).max()) > 0.01
    assert {p.name: p.read_bytes() for p in (directory / "chinese").glob("*.wav")} == dubs
