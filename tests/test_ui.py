from pathlib import Path
from threading import Event

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
    APP_CSS,
    DownloadController,
    _cache_component_defaults,
    _install_backend_log_events,
    _output_audio,
    _projects_root,
    indextts_installation_status,
    offline_model_pack_markdown,
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


def test_settings_show_split_backend_tables_and_page_reset_buttons() -> None:
    pytest.importorskip("gradio")
    app = ui_module.build_app()
    labels = [getattr(component, "label", None) for component in app.blocks.values()]
    values = [getattr(component, "value", None) for component in app.blocks.values()]

    assert "ASR 后端兼容性与安装状态" in labels
    assert "TTS 后端兼容性与安装状态" in labels
    assert "离线模型包导入日志" in labels
    assert values.count("重置本页为默认值（需保存）") == 5
    assert "暂停当前下载" in values


def test_ui_uses_consistent_latin_and_chinese_fonts() -> None:
    assert '"Segoe UI"' in APP_CSS
    assert '"Microsoft YaHei UI"' in APP_CSS


def test_download_controller_pauses_only_active_download() -> None:
    controller = DownloadController()

    assert controller.pause() == "当前没有下载任务。"
    controller.begin("kotoba_whisper")
    assert controller.cancel_event.is_set() is False
    assert "kotoba_whisper" in controller.pause()
    assert controller.cancel_event.is_set() is True
    controller.finish("kotoba_whisper")
    assert controller.pause() == "当前没有下载任务。"


def test_offline_model_pack_status_names_the_well_known_inbox(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "model-packs"
    monkeypatch.setattr("asmr_dubber.ui.model_pack_directory", lambda: inbox)

    text = offline_model_pack_markdown()

    assert str(inbox) in text
    assert "未发现 ZIP" in text


def test_backend_install_log_stream_yields_output_and_heartbeats() -> None:
    release = Event()

    def fake_installer(_backend_id: str, *, log_callback) -> str:
        log_callback("下载第一部分")
        assert release.wait(timeout=2)
        log_callback("下载第二部分")
        return "安装完成"

    events = _install_backend_log_events(
        "test_backend",
        installer=fake_installer,
        heartbeat_seconds=0.05,
    )

    initial = next(events)
    first_log = next(events)
    heartbeat = next(events)
    release.set()
    remaining = list(events)

    assert initial[1:] == (False, True)
    assert "下载第一部分" in first_log[0]
    assert "仍在安装中" in heartbeat[0]
    assert remaining[-1][1:] == (True, True)
    assert "下载第二部分" in remaining[-1][0]
    assert "安装完成" in remaining[-1][0]


def test_backend_install_log_stream_finishes_on_error() -> None:
    def failing_installer(_backend_id: str, *, log_callback) -> str:
        log_callback("已经开始")
        raise RuntimeError("模拟安装失败")

    events = list(
        _install_backend_log_events(
            "test_backend",
            installer=failing_installer,
            heartbeat_seconds=0.05,
        )
    )

    assert events[-1][1:] == (True, False)
    assert "安装失败" in events[-1][0]
    assert "模拟安装失败" in events[-1][0]


def test_backend_install_events_are_streamed_and_use_an_independent_queue() -> None:
    pytest.importorskip("gradio")
    app = ui_module.build_app()
    functions = {function.name: function for function in app.fns.values()}

    for name in (
        "install_asr_backend_callback",
        "install_tts_backend_callback",
        "import_offline_packs_callback",
    ):
        function = functions[name]
        dependency = next(item for item in app.config["dependencies"] if item["api_name"] == name)
        assert function.concurrency_id == "backend_install"
        assert function.concurrency_limit == 1
        assert dependency["types"]["generator"] is True
        assert dependency["show_progress"] == "minimal"

    assert functions["new_callback"].concurrency_id == "asmr_dubber_pipeline"
    component_ids = {
        getattr(component, "label", None): component._id for component in app.blocks.values()
    }
    asr_dependency = next(
        item
        for item in app.config["dependencies"]
        if item["api_name"] == "install_asr_backend_callback"
    )
    tts_dependency = next(
        item
        for item in app.config["dependencies"]
        if item["api_name"] == "install_tts_backend_callback"
    )
    assert component_ids["ASR 后端兼容性与安装状态"] in asr_dependency["outputs"]
    assert component_ids["TTS 后端兼容性与安装状态"] in tts_dependency["outputs"]


def test_install_profile_help_matches_setup_profiles() -> None:
    source = Path(ui_module.__file__).read_text(encoding="utf-8")

    assert "Recommended | Parakeet 1.1B/0.6B + IndexTTS2" in source
    assert "Advanced | Recommended + Kotoba v2.2 + Faster-Whisper large-v2" in source
    assert "Full | Advanced + 其余已集成且可自动安装的本地后端" in source
    assert "bash scripts/linux/setup.sh <档位>" in source
