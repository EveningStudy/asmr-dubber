"""Schema v4 serves sentences with ``target_text``; the page must read that field."""

from __future__ import annotations

import re
from importlib.resources import files

from asmr_dubber.models import AudioInfo, DubProject, Sentence


def test_served_sentences_carry_target_text_only() -> None:
    project = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="0" * 64, duration_seconds=2, sample_rate=16000, channels=1
        ),
        sentences=[
            Sentence(
                id="s000001", start_seconds=0, end_seconds=1, source_text="原文", zh_text="译文"
            )
        ],
    )
    served = project.model_dump(mode="json")["sentences"][0]
    assert served["target_text"] == "译文"
    assert "zh_text" not in served


def test_frontend_never_reads_zh_text_from_a_served_sentence() -> None:
    """``zh_text`` survives only as a table column / edit-patch key, never as a field read."""
    offenders = []
    for path in files("asmr_dubber").joinpath("frontend").iterdir():
        if not path.name.endswith(".js"):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in re.finditer(r"(\w+)\.zh_text\b", line):
                fallback = re.search(r"target_text\s*\?\?\s*" + re.escape(match.group(0)), line)
                if match.group(1) != "patch" and not fallback:
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
