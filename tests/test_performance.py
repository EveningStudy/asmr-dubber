import json
from pathlib import Path

import pytest

from asmr_dubber.performance import measure_stage


def test_measure_stage_records_timing_without_sensitive_details(tmp_path: Path) -> None:
    with measure_stage(
        tmp_path,
        "tts",
        backend="gpt_sovits",
        api_key="must-not-be-written",
        prompt="must-not-be-written",
    ) as details:
        details["sentences"] = 2

    events = json.loads((tmp_path / "performance.json").read_text(encoding="utf-8"))
    assert events[-1]["stage"] == "tts"
    assert events[-1]["status"] == "completed"
    assert events[-1]["elapsed_seconds"] >= 0
    assert events[-1]["details"] == {
        "backend": "gpt_sovits",
        "sentences": 2,
    }


def test_measure_stage_records_error_type_and_reraises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="failure"), measure_stage(tmp_path, "asr"):
        raise RuntimeError("failure")

    events = json.loads((tmp_path / "performance.json").read_text(encoding="utf-8"))
    assert events[-1]["status"] == "error"
    assert events[-1]["error_type"] == "RuntimeError"
