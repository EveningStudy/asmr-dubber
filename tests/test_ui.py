from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import asmr_dubber.ui as ui_module
from asmr_dubber.audio import sha256_file
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import AudioInfo, DubProject, Sentence, load_project, save_project
from asmr_dubber.ui import (
    _INDEXTTS_REQUIRED_DIRS,
    _INDEXTTS_REQUIRED_FILES,
    _cache_component_defaults,
    _output_audio,
    _projects_root,
    indextts_installation_status,
    reference_picker_data,
    save_reference_sentence,
)


def test_projects_root_accepts_local_directory() -> None:
    candidate = Path.home() / "asmr-projects"
    assert _projects_root(str(candidate)) == candidate.resolve()


def test_projects_root_rejects_existing_file(tmp_path: Path) -> None:
    candidate = tmp_path / "not-a-directory"
    candidate.write_text("x", encoding="utf-8")
    with pytest.raises(ProjectError, match="不是目录"):
        _projects_root(candidate)


def test_indextts_installation_status_checks_runtime_and_all_resources(tmp_path: Path) -> None:
    model_dir = tmp_path / "index-tts" / "checkpoints"
    model_dir.mkdir(parents=True)
    assert "运行环境未安装" in indextts_installation_status(model_dir)

    executable = model_dir.parent / ".venv" / "bin" / "indextts2"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert "模型不完整" in indextts_installation_status(model_dir)

    for relative in _INDEXTTS_REQUIRED_FILES:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for relative in _INDEXTTS_REQUIRED_DIRS:
        (model_dir / relative).mkdir(parents=True, exist_ok=True)
    assert "IndexTTS2 已就绪" in indextts_installation_status(model_dir)


def test_reference_picker_previews_and_persists_selected_sentence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portable = tmp_path / "portable"
    monkeypatch.setattr("asmr_dubber.ui.portable_home", lambda: portable)
    rate = 16_000
    project_directory = tmp_path / "projects" / "outside-launch-allowlist"
    project_directory.mkdir(parents=True)
    source = project_directory / "source.wav"
    sf.write(source, np.zeros(rate * 8, dtype=np.float32), rate, subtype="FLOAT")
    project = DubProject(
        source=AudioInfo(
            path=source.name,
            sha256=sha256_file(source),
            duration_seconds=8.0,
            sample_rate=rate,
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
    manifest = save_project(project, project_directory)

    choices, selected, preview = reference_picker_data(manifest)
    assert len(choices) == 2
    assert selected == "s000002"
    assert preview is not None and Path(preview).is_file()
    assert Path(preview).is_relative_to(portable / "temp" / "reference-previews")
    assert "⚠ 纯语气词" in choices[0][0]
    assert "★ 推荐范围" in choices[1][0]

    status, selected_preview = save_reference_sentence(manifest, "s000001")
    loaded, _ = load_project(manifest)
    assert loaded.settings.tts_reference_sentence_id == "s000001"
    assert "s000001" in status
    assert Path(selected_preview).is_file()


def test_saved_settings_update_defaults_for_new_browser_sessions() -> None:
    pytest.importorskip("gradio")
    app = ui_module.build_app()
    asr_model = next(
        component
        for component in app.blocks.values()
        if getattr(component, "label", None) == "ASR 模型"
    )
    _cache_component_defaults(
        app,
        {
            asr_model: {
                "choices": ["model-a", "model-b"],
                "value": "model-b",
                "__type__": "update",
            }
        },
    )
    component_config = next(
        item for item in app.config["components"] if item["id"] == asr_model._id
    )

    assert component_config["props"]["value"] == "model-b"
    assert component_config["props"]["choices"] == [
        ["model-a", "model-a"],
        ["model-b", "model-b"],
    ]


def test_completed_audio_uses_the_original_simple_player_and_output_path(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    output = tmp_path / "output" / "finished.wav"
    output.parent.mkdir()
    sf.write(output, np.zeros(8_000, dtype=np.float32), 8_000, subtype="PCM_24")
    project = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="0" * 64,
            duration_seconds=1.0,
            sample_rate=8_000,
            channels=1,
        ),
        output_file="output/finished.wav",
    )

    assert _output_audio(project, tmp_path) == str(output.resolve())

    app = ui_module.build_app()
    player = next(
        component
        for component in app.blocks.values()
        if getattr(component, "label", None) == "完成音频"
    )
    labels = {getattr(component, "label", None) for component in app.blocks.values()}

    assert player.type == "filepath"
    assert player.interactive is False
    assert player.elem_id is None
    assert "预处理" not in labels
