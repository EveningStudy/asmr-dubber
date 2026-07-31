from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf

import asmr_dubber.ui as ui_module
from asmr_dubber.audio import sha256_file
from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES
from asmr_dubber.models import AudioInfo, DubProject, Sentence, load_project, save_project
from asmr_dubber.runtime_manager import BackendStatus
from asmr_dubber.ui import (
    APP_CSS,
    DownloadController,
    _install_backend_log_events,
    _provider_update,
    _remote_auth,
    _transcript_language_update,
    asr_vad_choices,
    indextts_installation_status,
    offline_model_pack_markdown,
)
from asmr_dubber.ui_services import (
    analyze,
    apply_global_settings,
    apply_table,
    reference_picker,
    select_reference,
    stage_for_ui,
)
from asmr_dubber.user_settings import UserSettings


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("gradio")
    return ui_module.build_app()


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
    return project, tmp_path / "project.json"


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
    monkeypatch.setattr("asmr_dubber.ui_services.portable_home", lambda: home)
    output = tmp_path / "outside" / "finished.wav"
    output.parent.mkdir()
    output.write_bytes(b"audio")

    first = Path(stage_for_ui(output))
    second = Path(stage_for_ui(output))

    assert first == second
    assert first.is_relative_to(home / "temp" / "ui")
    assert first.read_bytes() == b"audio"


def test_reference_picker_previews_and_persists_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "portable"
    monkeypatch.setattr("asmr_dubber.ui_services.portable_home", lambda: home)
    _project_value, manifest = _project(tmp_path / "project")

    def fake_extract(source, destination, *_args, **_kwargs):
        assert source.is_file()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")

    monkeypatch.setattr("asmr_dubber.ui_services.extract_reference", fake_extract)
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
    monkeypatch.setattr("asmr_dubber.ui_services.portable_home", lambda: home)
    project, manifest = _project(tmp_path / "project")
    for sentence in project.sentences:
        sentence.ja_text = ""
    save_project(project, manifest.parent)

    def fake_extract(_source, destination, *_args, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")

    monkeypatch.setattr("asmr_dubber.ui_services.extract_reference", fake_extract)
    choices, selected, _preview = reference_picker(str(manifest))

    assert selected == "s000002"
    assert "这是一句足够长而清晰的参考句" in choices[1][0]


def test_ui_exposes_clear_four_step_workflow_and_only_supported_backends(app) -> None:
    components = list(app.blocks.values())
    labels = {getattr(component, "label", None) for component in components}
    values = [getattr(component, "value", None) for component in components]

    assert "日语音频或视频" in labels
    assert "ASR（语音识别）后端" in labels
    assert "TTS（语音合成）后端" in labels
    assert "台本语言" in labels
    assert "束搜索宽度（Beam Size）" in labels
    assert "随机度（Temperature）" in labels
    assert "核采样概率（Top P）" in labels
    assert "使用半精度计算（FP16）" in labels
    assert "中文配音整体偏移（毫秒）" in labels
    assert "冲突时最大自动加速倍速" in labels
    assert "中文最多提前秒数" not in labels
    assert "提前量最多占日语句长百分比" not in labels
    assert "1 · 运行 ASR（语音识别）" in values
    assert "4 · TTS（语音合成）并混音" in values
    assert "仅保存为以后新项目默认值" in values
    assert "保存并应用到当前项目" in values

    import_callback = next(
        function for function in app.fns.values() if function.name == "import_transcript_callback"
    )
    assert [component.label for component in import_callback.inputs] == [
        "当前项目文件",
        "台本或字幕文件",
        "也可以直接粘贴纯台本",
        "纯文本台本如何生成时间轴",
        "台本语言",
    ]


def test_chinese_transcript_mode_removes_qwen_timing_choice() -> None:
    chinese = _transcript_language_update("zh")
    japanese = _transcript_language_update("ja")

    assert chinese["value"] == "estimate"
    assert [value for _label, value in chinese["choices"]] == ["estimate"]
    assert [value for _label, value in japanese["choices"]] == ["estimate", "qwen"]

    source = Path(ui_module.__file__).read_text(encoding="utf-8").casefold()
    for removed in (
        "qwen3_asr",
        "qwen3_tts",
        "voxcpm2",
        "whisperx",
        "funasr",
        "f5_tts",
        "xtts_v2",
    ):
        assert removed not in source


def test_project_action_error_preserves_values_and_updates_status(app) -> None:
    callback = next(
        function.fn for function in app.fns.values() if function.name == "subtitle_callback"
    )

    result = callback("", [], "zh")

    assert len(result) == 10
    assert all(value == {"__type__": "update"} for value in result[:7])
    assert "当前项目、表格和已有输出均已保留" in result[7]
    assert result[8] == {"__type__": "update"}
    assert result[9] == {"__type__": "update"}


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
        "asmr_dubber.ui_services.pipeline.analyze_project",
        fake_analyze_project,
    )

    analyze(str(manifest), applied.rows)

    assert received == {
        "review_enabled": False,
        "force": True,
        "cancel_event": None,
    }


def test_apply_settings_button_saves_defaults_and_updates_both_pages(app, monkeypatch) -> None:
    function = next(
        function for function in app.fns.values() if function.name == "apply_settings_callback"
    )
    assert len(function.outputs) == 11
    assert function.outputs[0].label == "设置状态"

    values = [getattr(component, "value", None) for component in function.inputs]
    values[0] = r"D:\projects\current\project.json"
    review_index = next(
        index
        for index, component in enumerate(function.inputs)
        if component.label == "启用多 ASR（语音识别）+ 大模型交叉校对"
    )
    values[review_index] = False

    captured: dict[str, object] = {}
    monkeypatch.setattr(ui_module, "load_user_settings", UserSettings)

    def fake_save(settings: UserSettings) -> Path:
        captured["saved"] = settings.asr_review_enabled
        return Path(r"D:\portable\.asmr-dubber\config\settings.json")

    def fake_apply(manifest: str, settings: UserSettings):
        captured["manifest"] = manifest
        captured["applied"] = settings.asr_review_enabled
        return ui_module.ProjectView(
            manifest=manifest,
            rows=[],
            output_audio=None,
            output_video=None,
            subtitle_files=[],
            subtitle_video=None,
            diagnostics="ASR（语音识别）设置已改变，请重新运行 ASR。",
            status="设置已应用到当前项目。多模型交叉校对=关闭",
        )

    monkeypatch.setattr(ui_module, "save_user_settings", fake_save)
    monkeypatch.setattr(ui_module, "apply_global_settings", fake_apply)

    result = function.fn(*values)

    assert captured == {
        "saved": False,
        "manifest": r"D:\projects\current\project.json",
        "applied": False,
    }
    assert len(result) == 11
    assert "新项目默认值已保存" in result[0]
    assert "多模型交叉校对=关闭" in result[0]
    assert "多模型交叉校对=关闭" in result[8]

    captured.clear()
    values[0] = ""
    without_project = function.fn(*values)

    assert captured == {"saved": False}
    assert "当前没有打开项目" in without_project[0]
    assert "默认设置已保存；当前没有打开项目" in without_project[8]


def test_installation_and_inference_share_one_runtime_queue(app) -> None:
    functions = {function.name: function for function in app.fns.values()}

    for name in ("asr_callback", "translate_callback", "synthesize_callback", "install_callback"):
        assert functions[name].concurrency_id == "runtime_mutation"
        assert functions[name].concurrency_limit == 1


def test_ui_uses_accessible_fonts_focus_and_chinese_profiles() -> None:
    assert '"Segoe UI"' in APP_CSS
    assert '"Microsoft YaHei UI"' in APP_CSS
    assert ":focus-visible" in APP_CSS
    assert "width: 100% !important" in APP_CSS
    assert '.gradio-container [role="tablist"]' in APP_CSS
    assert ".backend-table table { min-width: 900px; }" in APP_CSS
    assert ".mobile-stack { flex-direction: column !important; }" in APP_CSS
    assert "推荐" in ui_module.PROFILE_MARKDOWN
    assert "进阶" in ui_module.PROFILE_MARKDOWN
    assert "| Full |" not in ui_module.PROFILE_MARKDOWN


def test_vad_choices_hide_uninstalled_or_backend_irrelevant_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_module,
        "asmr_vad_status",
        lambda: BackendStatus("missing", "模型未下载"),
    )
    assert [value for _, value in asr_vad_choices("kotoba_whisper")] == ["off"]
    assert [value for _, value in asr_vad_choices("parakeet_nemo")] == [
        "off",
        "backend",
    ]

    monkeypatch.setattr(
        ui_module,
        "asmr_vad_status",
        lambda: BackendStatus("ready", "可用"),
    )
    assert [value for _, value in asr_vad_choices("kotoba_whisper")] == ["off", "asmr"]


def test_translation_provider_hides_unrelated_provider_settings() -> None:
    deepseek = _provider_update("deepseek")
    deepl = _provider_update("deepl")
    microsoft = _provider_update("microsoft_translate")

    assert deepseek[4]["visible"] is True
    assert deepseek[5]["visible"] is False
    assert deepl[4]["visible"] is False
    assert deepl[5]["visible"] is True
    assert microsoft[6]["visible"] is True


def test_asr_backend_hides_parameters_owned_by_other_backends(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_module,
        "asmr_vad_status",
        lambda: BackendStatus("missing", "模型未下载"),
    )

    parakeet = ui_module._asr_backend_update("parakeet_nemo")
    kotoba = ui_module._asr_backend_update("kotoba_whisper")
    faster = ui_module._asr_backend_update("faster_whisper")

    assert [parakeet[index]["visible"] for index in range(2, 10)] == [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert [kotoba[index]["visible"] for index in range(2, 10)] == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    assert [faster[index]["visible"] for index in range(2, 10)] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_tts_detail_visibility_tracks_active_reference_mode() -> None:
    gpt_external = ui_module._tts_detail_visibility(
        "gpt_sovits", "external", "project_reference", "sentence_reference", "zero_shot"
    )
    cosy_cross_lingual = ui_module._tts_detail_visibility(
        "cosyvoice", "external", "project_reference", "sentence_reference", "cross_lingual"
    )
    index_external = ui_module._tts_detail_visibility(
        "indextts2", "project_sentence", "external", "external", "zero_shot"
    )
    index_text = ui_module._tts_detail_visibility(
        "indextts2", "project_sentence", "project_reference", "text", "zero_shot"
    )

    assert [update["visible"] for update in gpt_external] == [True, True, False, False]
    assert [update["visible"] for update in cosy_cross_lingual] == [
        True,
        False,
        False,
        False,
    ]
    assert [update["visible"] for update in index_external] == [True, False, True, False]
    assert [update["visible"] for update in index_text] == [False, False, False, True]


def test_download_controller_pauses_only_active_download() -> None:
    controller = DownloadController()

    assert controller.pause() == "当前没有下载任务。"
    controller.begin("kotoba_whisper")
    assert controller.cancel_event.is_set() is False
    assert "kotoba_whisper" in controller.pause()
    assert controller.cancel_event.is_set() is True
    controller.finish("kotoba_whisper")
    assert controller.pause() == "当前没有下载任务。"


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
    assert "仍在处理中" in heartbeat[0]
    assert remaining[-1][1:] == (True, True)


def test_non_loopback_ui_always_requires_authentication(monkeypatch) -> None:
    monkeypatch.delenv("ASMR_DUBBER_UI_PASSWORD", raising=False)
    assert _remote_auth("127.0.0.1") is None
    username, password = _remote_auth("0.0.0.0")
    assert username == "asmr"
    assert len(password) >= 20


def test_launch_exposes_only_ui_stage_directory(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeApp:
        def queue(self, **_kwargs):
            return self

        def launch(
            self,
            server_name=None,
            server_port=None,
            share=None,
            allowed_paths=None,
            max_file_size=None,
            footer_links=None,
            show_error=None,
            enable_monitoring=None,
            strict_cors=None,
            auth=None,
            auth_message=None,
            css=None,
            theme=None,
        ):
            captured.update(locals())

    stage = tmp_path / "portable" / "temp" / "ui"
    monkeypatch.setattr(ui_module, "require_supported_platform", lambda: None)
    monkeypatch.setattr(ui_module, "build_app", FakeApp)
    monkeypatch.setattr(ui_module, "ui_stage_directory", lambda: stage)

    ui_module.launch("127.0.0.1", 9999)

    assert captured["allowed_paths"] == [str(stage.resolve())]
    assert captured["auth"] is None
    assert captured["max_file_size"] == "20gb"
    assert captured["footer_links"] == []


def test_offline_model_pack_status_names_inbox(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "model-packs"
    monkeypatch.setattr("asmr_dubber.ui.model_pack_directory", lambda: inbox)

    text = offline_model_pack_markdown()

    assert str(inbox) in text
    assert "未发现 ZIP" in text
