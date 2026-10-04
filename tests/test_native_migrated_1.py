import time
from importlib.resources import files
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.api import API
from asmr_dubber.api_contract import TaskRequest
from asmr_dubber.api_logging import safe_api_url
from asmr_dubber.audio import sha256_file
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.services import parameters, settings
from asmr_dubber.services.application import Application
from asmr_dubber.services.project_operations import (
    analyze,
    apply_global_settings,
    import_transcript_data,
)
from asmr_dubber.services.project_operations import (
    mix as mix_service,
)
from asmr_dubber.services.project_operations import (
    synthesize as synthesize_service,
)
from asmr_dubber.translation import default_translation_prompt
from asmr_dubber.user_settings import UserSettings

FRONTEND = files("asmr_dubber").joinpath("frontend")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return Application(tmp_path / "tasks.json")


def wait_task(app, identifier):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = app.tasks.get(identifier)
        if task["status"] not in {"queued", "running", "cancelling"}:
            return task
        time.sleep(0.01)
    pytest.fail("Task did not finish")


def _project(tmp_path: Path) -> tuple[DubProject, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 10, dtype=np.float32), 16_000, subtype="FLOAT")
    project = DubProject(
        source=AudioInfo(
            path=source.name,
            sha256=sha256_file(source),
            duration_seconds=10.0,
            sample_rate=16_000,
            channels=1,
        ),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.0,
                end_seconds=0.5,
                ja_text="あ、",
                zh_text="啊，",
            ),
            Sentence(
                id="s000002",
                start_seconds=1.0,
                end_seconds=7.0,
                ja_text="これは十分に長くて明瞭な参考文章です。",
                zh_text="这是一句足够长而清晰的参考句。",
            ),
        ],
    )
    save_project(project, tmp_path)
    return load_project(tmp_path / "project.json")[0], tmp_path / "project.json"


def test_prompt_editor_switches_languages_without_losing_the_other_draft(app):
    japanese = default_translation_prompt("ja") + "\n日语自定义。"
    settings.update({"translation_prompt_ja": japanese})
    settings.update({"translation_prompt_en": ""})
    assert settings.current().translation_prompt_ja == japanese
    assert settings.current().translation_prompt_en == ""
    assert parameters.choices()["translation_prompts"]["en"] == default_translation_prompt("en")


def test_chinese_transcript_mode_keeps_only_supported_timing_choices(app):
    assert parameters.choices()["transcript_timing"]["zh"] == ["estimate", "script_review"]
    assert parameters.choices()["transcript_timing"]["source"] == [
        "estimate",
        "qwen",
        "script_review",
    ]
    for kind in ["zh", "source"]:
        TaskRequest(
            kind="import", project="project.json", revision=0, script_kind=kind, timing="estimate"
        )
    with pytest.raises(ValueError):
        TaskRequest(
            kind="import", project="project.json", revision=0, script_kind="zh", timing="qwen"
        )
    for removed in ("qwen3_asr", "qwen3_tts", "voxcpm2", "whisperx", "funasr", "f5_tts", "xtts_v2"):
        assert (
            removed not in parameters.choices()["asr"]
            and removed not in parameters.choices()["tts"]
        )


def test_original_transcript_follows_current_project_language(tmp_path: Path, monkeypatch) -> None:
    project, manifest = _project(tmp_path / "project")
    project.source_language = "en"
    save_project(project, manifest.parent)
    received: list[str] = []

    def fake_import(current, _directory, **kwargs):
        received.append(kwargs["script_language"])
        return {
            "language": kwargs["script_language"],
            "format": "SRT",
            "sentences": 1,
            "timed": True,
            "qwen_aligned_sentences": 0,
        }

    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.import_project_transcript",
        fake_import,
    )

    import_transcript_data(str(manifest), None, "Hello", "estimate", "source")
    import_transcript_data(str(manifest), None, "你好", "estimate", "zh")

    assert received == ["en", "zh"]


def test_project_action_error_preserves_values_and_updates_status(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    before = manifest.read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("failed operation")

    app.tasks.execute = fail
    task = app.start({"kind": "subtitles", "project": str(manifest), "revision": project.revision})
    result = wait_task(app, task["id"])
    assert result["status"] == "failed"
    assert result["error"] == "failed operation"
    assert manifest.read_bytes() == before
    assert len(app.projects.get(str(manifest))["rows"]) == 2


def test_changed_asr_settings_are_used_by_the_next_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project, manifest = _project(tmp_path / "project")
    project.settings.asr_review_enabled = True
    save_project(project, manifest.parent)

    settings = UserSettings.model_validate(project.settings.model_dump())
    settings.asr_review_enabled = False
    applied = apply_global_settings(str(manifest), settings)

    persisted, _ = load_project(manifest)
    assert persisted.settings.asr_review_enabled is False
    assert persisted.asr_settings_dirty is True
    assert "多模型交叉校对=关闭" in applied.status
    assert "请重新运行 ASR" in applied.diagnostics

    received: dict[str, object] = {}

    def fake_analyze_project(
        current,
        _directory,
        *,
        force=False,
        progress=None,
        cancel_event=None,
    ):
        received["review_enabled"] = current.settings.asr_review_enabled
        received["force"] = force
        received["cancel_event"] = cancel_event

    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.analyze_project",
        fake_analyze_project,
    )

    analyze(str(manifest), applied.rows)

    assert received == {
        "review_enabled": False,
        "force": True,
        "cancel_event": None,
    }


def test_tts_and_mix_services_are_independent(tmp_path: Path, monkeypatch) -> None:
    _project_value, manifest = _project(tmp_path / "project")
    calls: list[str] = []

    monkeypatch.setattr("asmr_dubber.services.project_operations.apply_table", lambda *_args: False)
    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.synthesize_project",
        lambda *_args, **_kwargs: calls.append("tts"),
    )
    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.mix_project",
        lambda *_args, **_kwargs: calls.append("mix"),
    )

    synthesize_service(str(manifest), [])
    assert calls == ["tts"]

    calls.clear()
    mix_service(str(manifest), [])
    assert calls == ["mix"]


def test_mix_service_renders_the_persisted_output_state(tmp_path: Path, monkeypatch) -> None:
    project, manifest = _project(tmp_path / "project")
    persisted = project.model_copy(deep=True)
    output_dir = manifest.parent / "output"
    output_dir.mkdir()
    mixed = output_dir / "mixed.wav"
    stem = output_dir / "stem.wav"
    mixed.write_bytes(b"mixed")
    stem.write_bytes(b"stem")
    persisted.settings.mix_output_mode = "both"
    persisted.output_file = "output/mixed.wav"
    persisted.chinese_stem_file = "output/stem.wav"
    reloads = iter(((project, manifest.parent), (persisted, manifest.parent)))

    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.reload_project",
        lambda _path: next(reloads),
    )
    monkeypatch.setattr("asmr_dubber.services.project_operations.apply_table", lambda *_args: False)
    monkeypatch.setattr(
        "asmr_dubber.services.project_operations.pipeline.mix_project",
        lambda *_args, **_kwargs: None,
    )

    result = mix_service(str(manifest), [])

    assert result.output_audio is not None
    assert result.stem_audio is not None
    assert "混音成品和中文克隆音轨" in result.status


def test_apply_settings_button_saves_defaults_and_updates_both_pages(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    settings.update({"asr_review_enabled": False, "autoflow_default_mode": "audio"})
    assert settings.current().asr_review_enabled is False
    assert settings.current().autoflow_default_mode == "audio"
    assert load_project(manifest)[0].settings.model_dump() == project.settings.model_dump()
    settings.update({"asr_review_enabled": False}, str(manifest), project.revision)
    assert app.projects.get(str(manifest))["settings"]["asr_review_enabled"] is False
    assert app.bootstrap()["settings"]["asr_review_enabled"] is False


def test_tts_settings_tab_reloads_the_current_project_backend(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    project.settings.tts_backend = "indextts2_5"
    project.settings.tts_model = "IndexTTS-2.5"
    save_project(project, manifest.parent)
    result = app.projects.get(str(manifest))
    assert result["settings"]["tts_backend"] == "indextts2_5"
    assert result["settings"]["tts_model"] == "IndexTTS-2.5"
    assert result["manifest"] == str(manifest)


def test_tts_hydration_reads_current_disk_not_startup_values(app, tmp_path):
    settings.update({"tts_backend": "indextts2_5"})
    assert app.bootstrap()["settings"]["tts_backend"] == "indextts2_5"
    settings.update({"tts_backend": "indextts2"})
    assert app.bootstrap()["settings"]["tts_backend"] == "indextts2"
    assert app.bootstrap()["settings"]["tts_model"] == "IndexTTS2"


def test_tts_programmatic_updates_do_not_replace_saved_values(app):
    settings.update({"tts_backend": "indextts2", "tts_speed": 1.3})
    before = settings.current().model_dump()
    parameters.catalog()
    parameters.choices()
    app.bootstrap()
    assert settings.current().model_dump() == before


def test_local_tts_save_uses_selected_backend_even_if_model_event_is_pending(app):
    settings.update({"tts_backend": "indextts2", "tts_model": "IndexTTS-2.5"})
    current = settings.current()
    assert current.tts_backend == "indextts2" and current.tts_model == "IndexTTS2"


def test_settings_default_scope_preserves_current_project(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    before = manifest.read_bytes()
    settings.update({"tts_speed": 1.3})
    assert manifest.read_bytes() == before
    assert settings.current().tts_speed == 1.3
    assert load_project(manifest)[0].settings.tts_speed == project.settings.tts_speed


def test_workflow_controls_follow_prerequisites(app):
    for kind in ("analyze", "translate", "synthesize", "mix", "export"):
        with pytest.raises(ValueError):
            API(app).call("tasks/start", {"kind": kind})
    assert app.tasks.list() == []
    assert "if (!state.project)" in FRONTEND.joinpath("session.js").read_text(encoding="utf-8")


def test_sentence_table_uses_bounded_native_editor(app):
    script = FRONTEND.joinpath("projects.js").read_text(encoding="utf-8")
    assert "slice(rowPage * 50, (rowPage + 1) * 50)" in script
    assert "contenteditable" in script
    assert 'data-row-field="enabled"' in script
    assert "table-layout: fixed" in FRONTEND.joinpath("styles.css").read_text(encoding="utf-8")


def test_sentence_table_does_not_send_whole_table_on_cell_edit(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    rows = app.projects.get(str(manifest))["rows"]
    row = list(rows[0])
    row[5] = "改后译文"
    saved = API(app).call(
        "projects/sentence", {"project": str(manifest), "revision": project.revision, "row": row}
    )
    assert saved["rows"][0] == row and saved["rows"][1] == rows[1]
    script = FRONTEND.joinpath("projects.js").read_text(encoding="utf-8")
    assert "only ? 'projects/sentence' : 'projects/table'" in script


def test_review_feature_and_result_panel_have_explicit_experimental_warning(app):
    html = FRONTEND.joinpath("index.html").read_text(encoding="utf-8")
    script = FRONTEND.joinpath("projects.js").read_text(encoding="utf-8")
    assert "实验性，效果可能不如单模型" in html
    assert "实验性，效果可能不如单模型" in script
    assert "普通单模型识别不需要使用" in script


def test_backend_usage_distinguishes_current_project_from_pending_form(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    project.settings.tts_backend = "indextts2"
    project.settings.tts_model = "IndexTTS2"
    save_project(project, manifest.parent)
    settings.update({"tts_backend": "generic_tts_api", "tts_model": "tts-1"})
    assert app.projects.get(str(manifest))["settings"]["tts_backend"] == "indextts2"
    assert app.bootstrap()["settings"]["tts_backend"] == "generic_tts_api"
    url = safe_api_url("https://user:secret@example.test/v1?token=hidden")
    assert url == "https://example.test/v1"
    assert "secret" not in url and "token=" not in url


def test_installation_and_inference_share_one_runtime_queue(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    release = Event()

    def execute(*args):
        release.wait(2)
        return {}

    app.tasks.execute = execute
    task = app.start({"kind": "analyze", "project": str(manifest), "revision": project.revision})
    for kind in ("analyze", "translate", "synthesize", "mix", "batch", "download"):
        with pytest.raises(ValueError):
            app.start({"kind": kind, "project": str(manifest), "revision": project.revision})
    release.set()
    assert wait_task(app, task["id"])["status"] == "completed"


def test_ui_uses_accessible_fonts_focus_and_chinese_profiles(app):
    css = FRONTEND.joinpath("styles.css").read_text(encoding="utf-8")
    assert '"Segoe UI"' in css and '"Microsoft YaHei UI"' in css
    assert ":focus-visible" in css
    assert "@media (max-width: 760px)" in css
    assert ".work" in css and ".sidebar" in css
    html = FRONTEND.joinpath("index.html").read_text(encoding="utf-8")
    assert 'data-language="zh"' in html and 'data-language="en"' in html


def test_vad_choices_hide_uninstalled_or_backend_irrelevant_modes(monkeypatch):
    descriptor = next(item for item in parameters.catalog() if item["key"] == "asr_vad_mode")

    def allowed(backend, language, ready):
        values = {"asr_backend": backend, "source_language": language, "asmr_vad_ready": ready}
        return [
            v
            for _, v in descriptor["options"]
            if all(
                values.get(k) in choices
                for k, choices in descriptor.get("options_when", {}).get(v, {}).items()
            )
        ]

    assert allowed("kotoba_whisper", "ja", False) == ["off"]
    assert allowed("parakeet_nemo", "ja", False) == ["off", "backend"]
    assert allowed("kotoba_whisper", "ja", True) == ["off", "asmr"]
    assert allowed("faster_whisper", "en", True) == ["off", "backend"]


def test_english_language_setting_filters_japanese_asr_backends(app):
    descriptor = next(item for item in parameters.catalog() if item["key"] == "asr_backend")
    allowed = [
        value
        for _, value in descriptor["options"]
        if "source_language" not in descriptor["options_when"].get(value, {})
    ]
    assert allowed == ["faster_whisper", "generic_asr_api"]
    assert (
        "kotoba-tech/kotoba-whisper-v2.0-faster"
        not in parameters.choices()["asr"]["faster_whisper"]["models_by_language"]["en"]
    )


def test_translation_provider_hides_unrelated_provider_settings(app):
    by_key = {item["key"]: item for item in parameters.catalog()}
    assert parameters.visible(
        by_key["translation_extra_body"], {"translation_provider": "deepseek"}
    )
    assert not parameters.visible(
        by_key["translation_deepl_formality"], {"translation_provider": "deepseek"}
    )
    assert not parameters.visible(
        by_key["translation_extra_body"], {"translation_provider": "deepl"}
    )
    assert parameters.visible(
        by_key["translation_deepl_formality"], {"translation_provider": "deepl"}
    )
    assert parameters.visible(
        by_key["translation_microsoft_region"], {"translation_provider": "microsoft_translate"}
    )


def test_asr_backend_hides_parameters_owned_by_other_backends(app):
    by_key = {item["key"]: item for item in parameters.catalog()}
    fields = [
        "asr_compute_type",
        "asr_beam_size",
        "asr_batch_size",
        "asr_condition_on_previous_text",
        "asr_initial_prompt",
        "asr_parakeet_decoder",
        "asr_chunk_seconds",
        "asr_kotoba_chunk_seconds",
    ]

    def shown(backend):
        return [parameters.visible(by_key[key], {"asr_backend": backend}) for key in fields]

    assert shown("parakeet_nemo") == [False, False, True, False, False, True, True, False]
    assert shown("kotoba_whisper") == [True, False, True, False, False, False, False, True]
    assert shown("faster_whisper") == [True, True, True, True, True, False, False, False]
