import sys
import time
from importlib.resources import files
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.api import API
from asmr_dubber.api_contract import TableRequest, TaskRequest
from asmr_dubber.audio import sha256_file
from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES
from asmr_dubber.model_registry import TTS_BACKENDS
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.services import batch_queue, parameters, settings
from asmr_dubber.services.application import Application
from asmr_dubber.services.model_status import (
    indextts_installation_status,
)
from asmr_dubber.services.project_audio import (
    preview_edge_tts_voice,
    reference_picker,
    select_autoflow_external_reference,
    select_autoflow_project_reference,
    select_reference,
    stage_for_ui,
)
from asmr_dubber.services.project_records import (
    apply_table,
    open_project_directory,
    open_project_output_directory,
)
from asmr_dubber.translation import SYSTEM_PROMPT, default_translation_prompt
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


def test_project_actions_prompt_before_pipeline_when_manifest_is_empty(app):
    with pytest.raises(ValueError, match="project and revision"):
        API(app).call("tasks/start", {"kind": "analyze", "project": ""})
    assert app.tasks.list() == []


def test_indextts_status_checks_runtime_and_all_resources(tmp_path: Path) -> None:
    model_dir = tmp_path / "index-tts" / "checkpoints"
    model_dir.mkdir(parents=True)
    assert "运行环境未安装" in indextts_installation_status(model_dir)

    executable = model_dir.parent / ".venv" / "bin" / "indextts2"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert "模型不完整" in indextts_installation_status(model_dir)

    for relative in INDEXTTS_REQUIRED_FILES:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for relative in INDEXTTS_REQUIRED_DIRS:
        (model_dir / relative).mkdir(parents=True, exist_ok=True)
    assert "IndexTTS2 已就绪" in indextts_installation_status(model_dir)


def test_sentence_table_sorts_rows_and_parses_false_string(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    rows = [
        ["s000002", "false", 5.0, 6.0, "後です。", "在后面。"],
        ["s000001", True, 1.0, 2.0, "先です。", "在前面。"],
    ]

    assert apply_table(project, rows) is True
    assert [item.id for item in project.sentences] == ["s000001", "s000002"]
    assert project.sentences[1].enabled is False


def test_sentence_table_accepts_chinese_only_rows_and_deletes_empty_rows(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    chinese_only = [
        ["s000001", True, 1.0, 2.0, "", "直接配音。"],
        ["s000002", True, 2.0, 7.0, "", "这是第二句。"],
    ]

    assert apply_table(project, chinese_only) is True
    assert [item.ja_text for item in project.sentences] == ["", ""]
    assert [item.zh_text for item in project.sentences] == ["直接配音。", "这是第二句。"]

    project.settings.tts_reference_sentence_id = "s000001"
    chinese_only[0][5] = ""
    assert apply_table(project, chinese_only) is True
    assert [item.id for item in project.sentences] == ["s000002"]
    assert project.sentences[0].zh_text == "这是第二句。"
    assert project.settings.tts_reference_sentence_id is None


def test_ui_staging_is_deterministic_and_below_one_allowlist(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "portable"
    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: home)
    output = tmp_path / "outside" / "finished.wav"
    output.parent.mkdir()
    output.write_bytes(b"audio")

    first = Path(stage_for_ui(output))
    second = Path(stage_for_ui(output))

    assert first == second
    assert first.is_relative_to(home / "temp" / "ui")
    assert first.read_bytes() == b"audio"


def test_media_preview_staging_uses_browser_safe_filename(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "portable"
    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: home)
    output = tmp_path / "outside" / "#1.中文试听.wav"
    output.parent.mkdir()
    output.write_bytes(b"audio")

    staged = Path(stage_for_ui(output, preserve_name=False))

    assert staged.name.endswith(".wav")
    assert staged.stem.isascii()
    assert all(character.isalnum() for character in staged.stem)
    assert staged.read_bytes() == b"audio"


def test_output_media_reuses_portable_cache_and_native_audio_player(app, tmp_path):
    project, manifest = _project(tmp_path / "project")
    output = manifest.parent / "output.wav"
    sf.write(output, np.zeros(16000), 16000)
    project.output_file = output.name
    project.chinese_stem_file = output.name
    save_project(project, manifest.parent)
    result = app.projects.get(str(manifest))
    assert result["outputs"]["output_file"]["url"] == result["outputs"]["chinese_stem_file"]["url"]
    assert app.media.resolve(result["outputs"]["output_file"]["url"].split("/")[-1])[0] == output
    assert "<audio" in FRONTEND.joinpath("projects.js").read_text(encoding="utf-8")
    assert "waveform" not in FRONTEND.joinpath("projects.js").read_text(encoding="utf-8")


def test_edge_tts_voice_preview_is_cached_and_browser_safe(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "portable"
    calls: list[tuple[str, str]] = []

    class FakeCommunicate:
        def __init__(self, text: str, *, voice: str):
            calls.append((text, voice))

        async def save(self, path: str) -> None:
            Path(path).write_bytes(b"edge-preview")

    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: home)
    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate))

    first = Path(preview_edge_tts_voice("zh-CN-XiaoxiaoNeural"))
    second = Path(preview_edge_tts_voice("zh-CN-XiaoxiaoNeural"))

    assert first == second
    assert first.name.endswith(".mp3")
    assert first.stem.isascii()
    assert first.read_bytes() == b"edge-preview"
    assert calls == [("你好，欢迎使用 ASMR Dubber。", "zh-CN-XiaoxiaoNeural")]


def test_open_project_directory_uses_loaded_project_path(tmp_path: Path, monkeypatch) -> None:
    _project_value, manifest = _project(tmp_path / "project")
    opened: list[Path] = []
    monkeypatch.setattr(
        "asmr_dubber.services.project_records.open_directory",
        lambda path: opened.append(Path(path).resolve()) or Path(path).resolve(),
    )

    message = open_project_directory(str(manifest))

    assert opened == [manifest.parent.resolve()]
    assert str(manifest.parent.resolve()) in message


def test_open_project_output_directory_uses_project_output_path(
    tmp_path: Path, monkeypatch
) -> None:
    _project_value, manifest = _project(tmp_path / "project")
    opened: list[Path] = []
    monkeypatch.setattr(
        "asmr_dubber.services.project_records.open_directory",
        lambda path: opened.append(Path(path).resolve()) or Path(path).resolve(),
    )

    message = open_project_output_directory(str(manifest))

    expected = (manifest.parent / "output").resolve()
    assert expected.is_dir()
    assert opened == [expected]
    assert str(expected) in message


def test_reference_picker_previews_and_persists_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "portable"
    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: home)
    _project_value, manifest = _project(tmp_path / "project")

    def fake_extract(source, destination, *_args, **_kwargs):
        assert source.is_file()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")

    monkeypatch.setattr("asmr_dubber.services.project_audio.extract_reference", fake_extract)
    choices, selected, preview = reference_picker(str(manifest))

    assert len(choices) == 2
    assert selected == "s000002"
    assert "★ 推荐" in choices[1][0]
    assert "⚠ 过短" in choices[0][0]
    assert Path(preview).is_relative_to(home / "temp" / "ui")

    message, _preview = select_reference(str(manifest), "s000001")
    loaded, _ = load_project(manifest)
    assert loaded.settings.tts_reference_sentence_id == "s000001"
    assert "s000001" in message


def test_reference_picker_uses_chinese_text_for_chinese_only_script(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "portable"
    monkeypatch.setattr("asmr_dubber.services.project_audio.portable_home", lambda: home)
    project, manifest = _project(tmp_path / "project")
    for sentence in project.sentences:
        sentence.ja_text = ""
    save_project(project, manifest.parent)

    def fake_extract(_source, destination, *_args, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")

    monkeypatch.setattr("asmr_dubber.services.project_audio.extract_reference", fake_extract)
    choices, selected, _preview = reference_picker(str(manifest))

    assert selected == "s000002"
    assert "这是一句足够长而清晰的参考句" in choices[1][0]


def test_ui_exposes_clear_five_step_workflow_and_only_supported_backends(app):
    catalog = parameters.catalog()
    keys = {item["key"] for item in catalog}
    assert set(UserSettings.model_fields) <= keys
    assert set(ProjectSettings.model_fields) <= keys
    assert set(parameters.choices()["asr"]) == {
        "parakeet_nemo",
        "kotoba_whisper",
        "faster_whisper",
        "generic_asr_api",
    }
    assert set(parameters.choices()["tts"]) == set(TTS_BACKENDS)
    html = FRONTEND.joinpath("index.html").read_text(encoding="utf-8")
    assert all(f'data-step="{step}"' in html for step in range(1, 5))
    assert all(
        f'id="{name}"' in html
        for name in (
            "importTranscript",
            "showReview",
            "changeReference",
            "exportProject",
            "openProjectFolder",
        )
    )
    routes = API(app).routes
    assert all(
        name in routes
        for name in (
            "review/get",
            "projects/reference",
            "settings/key",
            "batch/subtitle",
            "batch/tracks",
            "batch/edit",
            "batch/remove",
            "batch/reorder",
        )
    )
    assert "revision" in TableRequest.model_fields
    assert {"file", "text", "timing", "script_kind", "revision"} <= set(TaskRequest.model_fields)
    assert "settings/update" in routes


def test_workspace_nests_both_work_modes_and_separates_autoflow_scopes(app):
    html = FRONTEND.joinpath("index.html").read_text(encoding="utf-8")
    assert all(f'id="{page}"' in html for page in ("project", "batch", "settings"))
    assert 'data-scope="proj"' in html and 'data-scope="def"' in html
    script = FRONTEND.joinpath("batch.js").read_text(encoding="utf-8")
    assert all(
        name in script
        for name in (
            "batch/tracks",
            "batch/subtitle",
            "batch/reorder",
            "batch/edit",
            "batch/remove",
        )
    )
    assert "referenceDialog(request.project_json,true)" in script
    assert "source_subtitles" in script and "subtitles" in script
    assert "background_choices" in script


def test_autoflow_reference_events_are_streamed_without_blocking_ui(app, monkeypatch):
    release = Event()

    def run_queue(payload, *, cancel_event, reference_event_callback):
        assert cancel_event is not None
        reference_event_callback(
            {"kind": "ready", "project_json": "project.json", "request_id": "request-1"}
        )
        assert release.wait(2)
        reference_event_callback({"kind": "timeout", "request_id": "request-1"})
        return 0, []

    monkeypatch.setattr(batch_queue, "run_queue", run_queue)
    task = app.start({"kind": "batch"})
    deadline = time.monotonic() + 2
    while not app.tasks.get(task["id"])["reference"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert app.tasks.get(task["id"])["reference"]["kind"] == "ready"
    assert app.tasks.get(task["id"])["status"] == "running"
    release.set()
    assert wait_task(app, task["id"])["reference"] is None


def test_autoflow_can_select_project_or_external_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, manifest = _project(tmp_path / "project")
    project.settings.tts_reference_source = "external"
    project.settings.tts_index_speaker_source = "external"
    save_project(project, manifest.parent)
    monkeypatch.setattr(
        "asmr_dubber.services.project_audio.reference_preview",
        lambda _project, _directory, sentence_id: f"preview-{sentence_id}.wav",
    )

    message, preview = select_autoflow_project_reference(str(manifest), "s000002")
    loaded, _ = load_project(manifest)
    assert "s000002" in message
    assert preview == "preview-s000002.wav"
    assert loaded.settings.tts_reference_source == "project_sentence"
    assert loaded.settings.tts_index_speaker_source == "project_reference"
    assert loaded.settings.tts_reference_sentence_id == "s000002"

    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "config"))
    upload = tmp_path / "external.wav"
    sf.write(upload, np.zeros(16_000, dtype=np.float32), 16_000, subtype="FLOAT")
    message, stored = select_autoflow_external_reference(
        str(manifest),
        upload,
        text="参考音频",
        language="zh",
    )
    loaded, _ = load_project(manifest)
    assert "external.wav" not in message
    assert Path(stored).is_file()
    assert loaded.settings.tts_reference_source == "external"
    assert loaded.settings.tts_index_speaker_source == "external"
    assert loaded.settings.tts_external_reference_text == "参考音频"
    assert loaded.settings.tts_external_reference_language == "zh"


def test_loudness_modes_map_to_existing_project_fields(monkeypatch):
    def configured(mode, target):
        values = parameters.expand_changes(
            {
                "normalize_chinese_loudness": mode,
                "chinese_target_active_rms_dbfs": target,
                "chinese_gain_db": 3,
            }
        )
        return UserSettings.model_validate(values)

    source = configured("source", -31)
    uniform = configured("uniform", -28)
    raw = configured("raw", -28)
    assert source.normalize_chinese_loudness is True and source.match_source_loudness is True
    assert source.chinese_target_active_rms_dbfs == -31
    assert uniform.normalize_chinese_loudness is True and uniform.match_source_loudness is False
    assert uniform.chinese_target_active_rms_dbfs == -28
    assert raw.normalize_chinese_loudness is False and raw.chinese_gain_db == 3
    descriptor = next(
        item for item in parameters.catalog() if item["key"] == "normalize_chinese_loudness"
    )
    assert parameters.display_value(descriptor, source.model_dump()) == "source"
    assert parameters.display_value(descriptor, uniform.model_dump()) == "uniform"
    assert parameters.display_value(descriptor, raw.model_dump()) == "raw"


def test_builtin_translation_prompt_is_visible_but_not_frozen_in_settings(app):
    assert app.bootstrap()["choices"]["translation_prompts"]["ja"] == SYSTEM_PROMPT
    assert app.bootstrap()["choices"]["translation_prompts"]["en"] == default_translation_prompt(
        "en"
    )
    settings.update({"translation_prompt_ja": SYSTEM_PROMPT})
    assert settings.current().translation_prompt_ja == ""
    settings.update({"translation_prompt_ja": SYSTEM_PROMPT + "\n请使用更口语的表达。"})
    assert settings.current().translation_prompt_ja.endswith("请使用更口语的表达。")
    assert settings.current().translation_prompt_en == ""
