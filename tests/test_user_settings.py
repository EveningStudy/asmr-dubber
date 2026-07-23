import os
import stat
from pathlib import Path

from asmr_dubber.user_settings import (
    UserSettings,
    api_key_status,
    clear_api_key,
    load_user_settings,
    save_api_key,
    save_user_settings,
    saved_api_key,
    store_reference_audio,
)


def test_settings_and_provider_keys_are_private_and_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    settings = UserSettings(
        projects_root=str(Path.home() / "projects"),
        translation_provider="openai",
        translation_model="gpt-5.2",
        translation_base_url="https://api.openai.com/v1",
        huggingface_endpoint="https://models.example.test",
        pypi_index_url="https://packages.example.test/simple",
    )
    settings_path = save_user_settings(settings)
    secrets_path = save_api_key("openai", "private-test-key")

    assert load_user_settings().translation_model == "gpt-5.2"
    assert load_user_settings().huggingface_endpoint == "https://models.example.test"
    assert os.environ["HF_ENDPOINT"] == "https://models.example.test"
    assert saved_api_key("openai") == "private-test-key"
    assert "private-test-key" not in settings_path.read_text(encoding="utf-8")
    # POSIX bits do not represent Windows ACLs, including a Windows-mounted
    # directory reached through WSL. Only assert modes when this filesystem
    # reports chmod changes.
    settings_path.chmod(0o600)
    reports_posix_modes = stat.S_IMODE(settings_path.stat().st_mode) == 0o600
    if os.name != "nt" and reports_posix_modes:
        assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(secrets_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(secrets_path.parent.stat().st_mode) == 0o700
    assert "已在本地配置中保存" in api_key_status("openai")

    clear_api_key("openai")
    assert saved_api_key("openai") == ""


def test_portable_paths_are_saved_relative_to_application_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "app" / ".asmr-dubber"
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(home))
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(home / "config"))
    settings = UserSettings(
        projects_root=str(home / "projects"),
        tts_model_path=str(home / "runtimes" / "index-tts" / "checkpoints"),
    )

    path = save_user_settings(settings)
    payload = path.read_text(encoding="utf-8")

    assert "${ASMR_DUBBER_HOME}/projects" in payload
    assert str(home) not in payload
    assert load_user_settings().projects_root == str(home / "projects")


def test_user_settings_copy_all_material_options_to_project() -> None:
    settings = UserSettings(
        global_overlap_seconds=0.4,
        global_overlap_percentage=35.0,
        chinese_gain_db=-2.0,
        match_source_loudness=True,
        chinese_relative_loudness_db=-1.5,
        chinese_min_active_rms_dbfs=-44.0,
        chinese_max_active_rms_dbfs=-31.0,
        translation_provider="anthropic",
        translation_model="claude-sonnet-5",
        translation_base_url="https://api.anthropic.com",
    )
    project = settings.to_project_settings()
    assert project.global_overlap_seconds == 0.4
    assert project.global_overlap_percentage == 35.0
    assert project.chinese_relative_loudness_db == -1.5
    assert project.chinese_min_active_rms_dbfs == -44.0
    assert project.chinese_target_active_rms_dbfs == -31.0
    assert project.translation_provider == "anthropic"
    assert project.translation_model == "claude-sonnet-5"


def test_model_backends_and_external_reference_are_copied_to_project(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np
    import soundfile as sf

    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "config"))
    upload = tmp_path / "voice.wav"
    sf.write(upload, np.zeros(16_000, dtype=np.float32), 16_000, subtype="FLOAT")
    stored_reference = store_reference_audio(upload)
    settings = UserSettings(
        asr_backend="faster_whisper",
        asr_model="large-v3",
        tts_backend="qwen3_tts",
        tts_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        tts_reference_source="external",
        tts_external_reference_audio=str(stored_reference),
        tts_external_reference_text="参考テキストです。",
    )
    project = settings.to_project_settings()
    assert project.asr_backend == "faster_whisper"
    assert project.tts_backend == "qwen3_tts"
    assert project.tts_reference_source == "external"
    assert Path(project.tts_external_reference_audio).is_file()
    assert stored_reference.parent == tmp_path / "config" / "references"
