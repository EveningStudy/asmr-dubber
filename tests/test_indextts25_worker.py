from __future__ import annotations

import json
from pathlib import Path

import pytest

from asmr_dubber.indextts25_worker import _infer_kwargs, _load_tasks, _require_optional_module


def test_indextts25_task_mapping_preserves_supported_controls(tmp_path: Path) -> None:
    task = {
        "id": "s000001",
        "text": "今天是[2026]年。",
        "voice": str(tmp_path / "voice.wav"),
        "output": str(tmp_path / "output.wav"),
        "language": "zh",
        "emotion_text": "平静、轻柔",
        "emotion_weight": 0.6,
        "duration_factor": 0.85,
        "text_normalization": False,
        "do_sample": False,
        "top_p": 0.7,
        "top_k": 20,
        "temperature": 0.65,
        "num_beams": 2,
        "repetition_penalty": 8.0,
        "length_penalty": 0.2,
        "max_mel_tokens": 1200,
    }

    kwargs = _infer_kwargs(task)

    assert kwargs["lang"] == "zh"
    assert kwargs["use_emo_text"] is True
    assert kwargs["emo_text"] == "平静、轻柔"
    assert kwargs["emo_alpha"] == 0.6
    assert kwargs["duration_factor"] == 0.85
    assert kwargs["text_normalization"] is False
    assert kwargs["do_sample"] is False
    assert kwargs["top_p"] == 0.7
    assert kwargs["top_k"] == 20
    assert kwargs["temperature"] == 0.65
    assert kwargs["num_beams"] == 2
    assert kwargs["repetition_penalty"] == 8.0
    assert kwargs["length_penalty"] == 0.2
    assert kwargs["max_mel_tokens"] == 1200


def test_indextts25_batch_manifest_requires_complete_tasks(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        json.dumps({"id": "s1", "text": "测试", "voice": "voice.wav"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output is required"):
        _load_tasks(manifest)


def test_indextts25_emotion_vector_must_have_eight_bounded_values() -> None:
    base = {"text": "测试", "voice": "voice.wav", "output": "output.wav"}

    with pytest.raises(ValueError, match="exactly 8"):
        _infer_kwargs({**base, "emotion_vector": [0.0] * 7})
    with pytest.raises(ValueError, match="between 0 and 1"):
        _infer_kwargs({**base, "emotion_vector": [0.0] * 7 + [1.1]})


def test_indextts25_optional_acceleration_fails_explicitly(monkeypatch) -> None:
    def unavailable(_module: str):
        raise ImportError("missing")

    monkeypatch.setattr("asmr_dubber.indextts25_worker.importlib.import_module", unavailable)

    with pytest.raises(RuntimeError, match="DeepSpeed requires"):
        _require_optional_module(True, "deepspeed", "DeepSpeed")
    _require_optional_module(False, "deepspeed", "DeepSpeed")


def test_missing_acceleration_dependency_is_skipped_with_a_warning(capsys):
    from asmr_dubber.indextts25_worker import _optional_acceleration

    assert _optional_acceleration(True, "module_that_does_not_exist", "DeepSpeed") is False
    assert "continuing without it" in capsys.readouterr().err
    assert _optional_acceleration(True, "json", "Present") is True
    assert _optional_acceleration(False, "module_that_does_not_exist", "Off") is False
