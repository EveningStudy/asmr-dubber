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
    assert "便携式明文保存在程序目录" in api_key_status("openai")

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
        tts_index_external_emotion_audio=str(home / "config" / "references" / "emotion.wav"),
    )

    path = save_user_settings(settings)
    payload = path.read_text(encoding="utf-8")

    assert "${ASMR_DUBBER_HOME}/projects" in payload
    assert "${ASMR_DUBBER_HOME}/config/references/emotion.wav" in payload
    assert str(home) not in payload
    assert load_user_settings().projects_root == str(home / "projects")


def test_user_settings_copy_all_material_options_to_project() -> None:
    settings = UserSettings(
        chinese_dubbing_offset_ms=-400,
        chinese_max_auto_speed=1.35,
        chinese_gain_db=-2.0,
        match_source_loudness=True,
        chinese_relative_loudness_db=-1.5,
        chinese_min_active_rms_dbfs=-44.0,
        chinese_max_active_rms_dbfs=-31.0,
        translation_provider="anthropic",
        translation_model="claude-sonnet-5",
        translation_base_url="https://api.anthropic.com",
        tts_index_speaker_source="sentence_reference",
        tts_index_emotion_source="text",
        tts_index_emo_text="轻柔",
    )
    project = settings.to_project_settings()
    assert project.chinese_dubbing_offset_ms == -400
    assert project.chinese_max_auto_speed == 1.35
    assert project.chinese_relative_loudness_db == -1.5
    assert project.chinese_min_active_rms_dbfs == -44.0
    assert project.chinese_target_active_rms_dbfs == -31.0
    assert project.translation_provider == "anthropic"
    assert project.translation_model == "claude-sonnet-5"
    assert project.tts_index_speaker_source == "sentence_reference"
    assert project.tts_index_emotion_source == "text"


def test_source_language_defaults_keep_separate_translation_prompts() -> None:
    settings = UserSettings(
        default_source_language="en",
        translation_prompt_ja="日语自定义",
        translation_prompt_en="English custom",
    )

    assert settings.to_project_settings(source_language="ja").translation_prompt == "日语自定义"
    assert settings.to_project_settings(source_language="en").translation_prompt == "English custom"


def test_legacy_global_translation_prompt_is_preserved_for_both_languages() -> None:
    settings = UserSettings.model_validate({"translation_prompt": "旧版自定义"})

    assert settings.translation_prompt == ""
    assert settings.translation_prompt_ja == "旧版自定义"
    assert settings.translation_prompt_en == "旧版自定义"


def test_legacy_packaged_prompt_is_not_migrated_as_a_custom_english_prompt() -> None:
    legacy = (
        "你是日语音声、广播剧和 ASMR 的简体中文配音翻译。\n"
        "规则内容。\n"
        '{"translations":[{"id":"s000001","zh":"中文台词；无实义时为空字符串"}]}'
    )
    settings = UserSettings.model_validate({"translation_prompt": legacy})

    assert settings.translation_prompt_ja == ""
    assert settings.translation_prompt_en == ""


def test_default_relative_chinese_loudness_matches_saved_project_default() -> None:
    settings = UserSettings()

    assert settings.chinese_relative_loudness_db == -8.0
    assert settings.to_project_settings().chinese_relative_loudness_db == -8.0


def test_legacy_user_setting_migrates_index_text_emotion() -> None:
    settings = UserSettings.model_validate({"tts_index_use_emo_text": True})

    assert settings.tts_index_emotion_source == "text"


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
        tts_backend="gpt_sovits",
        tts_model="GPT-SoVITS-v4",
        tts_reference_source="external",
        tts_external_reference_audio=str(stored_reference),
        tts_external_reference_text="参考テキストです。",
    )
    project = settings.to_project_settings()
    assert project.asr_backend == "faster_whisper"
    assert project.tts_backend == "gpt_sovits"
    assert project.tts_reference_source == "external"
    assert Path(project.tts_external_reference_audio).is_file()
    assert stored_reference.parent == tmp_path / "config" / "references"


def test_applying_global_defaults_preserves_project_reference_sentence() -> None:
    from asmr_dubber.models import ProjectSettings

    current = ProjectSettings(tts_reference_sentence_id="s000042")
    updated = UserSettings(chinese_dubbing_offset_ms=250).to_project_settings(current)

    assert updated.tts_reference_sentence_id == "s000042"
    assert updated.chinese_dubbing_offset_ms == 250
