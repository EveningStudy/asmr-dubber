import pytest

from asmr_dubber.autoflow.output_policy import policy_for_settings
from asmr_dubber.models import ProjectSettings
from asmr_dubber.services import models
from asmr_dubber.user_settings import UserSettings


def test_independent_loudness_roundtrip_and_batch_policy(monkeypatch):
    names = [
        "normalize_chinese_loudness",
        "chinese_target_active_rms_dbfs",
        "replacement_normalize_chinese_loudness",
        "replacement_chinese_target_active_rms_dbfs",
        "replacement_chinese_relative_loudness_db",
    ]
    from asmr_dubber.services import parameters

    values = dict(zip(names, ["uniform", -35, "source", -19, -2], strict=True))
    settings = UserSettings.model_validate(parameters.expand_changes(values))
    assert not settings.match_source_loudness
    assert settings.replacement_match_source_loudness
    assert settings.chinese_target_active_rms_dbfs == -35
    assert settings.replacement_chinese_target_active_rms_dbfs == -19
    fields = {item["key"]: item for item in parameters.catalog()}
    assert [parameters.display_value(fields[key], settings.model_dump()) for key in names] == [
        "uniform",
        -35,
        "source",
        -19,
        -2,
    ]
    settings.separation_enabled = True
    policy = policy_for_settings(settings, "both", "bilingual")
    assert policy["settings"]["replacement_chinese_relative_loudness_db"] == -2
    with pytest.raises(ValueError, match="floor"):
        ProjectSettings(
            replacement_chinese_min_active_rms_dbfs=-20,
            replacement_chinese_target_active_rms_dbfs=-30,
        )


def test_streamed_installer_requires_consent_and_reports_logs(tmp_path, monkeypatch):
    from asmr_dubber.services import settings
    from asmr_dubber.services.application import Application
    from tests.test_native_services import completed

    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    calls = []

    def prepare(model, **kwargs):
        calls.append(kwargs)
        kwargs["log"]("下载进度：25%")
        return "准备完成"

    monkeypatch.setattr(models, "prepare_local_model", prepare)
    app = Application(tmp_path / "tasks.json")
    assert app.tasks.list() == []
    assert not calls
    settings.update({"download_source": "original"})
    task = app.start({"kind": "download", "model": "separation_roformer"})
    result = completed(app.tasks, task["id"])
    assert any("25%" in event for event in result["logs"])
    assert result["result"]["message"] == "准备完成"
    assert calls[0]["install"] and calls[0]["source"] == "original"
