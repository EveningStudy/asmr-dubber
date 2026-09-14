from types import SimpleNamespace

import pytest

from asmr_dubber import ui
from asmr_dubber.autoflow import engine
from asmr_dubber.autoflow.ui_services import config_from_settings
from asmr_dubber.user_settings import UserSettings


def test_entire_form_reload_reads_disk_each_time(monkeypatch, tmp_path):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path))
    current = UserSettings(asr_review_enabled=True, translation_model="custom-first")
    monkeypatch.setattr(ui, "load_user_settings", lambda: current)
    app = ui.build_app()
    callback = next(f for f in app.fns.values() if f.name == "refresh_settings_form_callback")
    current.asr_review_enabled = False
    current.translation_model = "custom-after"
    current.autoflow_original_hard_subtitles = True
    current.autoflow_timestamp_footer_position = "before"
    result = callback.fn()
    labels = {c.label: r for c, r in zip(callback.outputs, result, strict=True)}
    assert labels["启用统一音频片段复核"]["value"] is False
    assert labels["翻译模型"]["value"] == "custom-after"
    assert labels["原声视频也编码硬字幕"]["value"] is True
    assert labels["时间戳文档附加文字位置"]["value"] == "before"
    assert len(result) == len(callback.outputs)


def test_programmatic_backend_hydration_preserves_values():
    updates = ui._visibility_only(ui._asr_backend_update("faster_whisper"))
    assert all("value" not in item for item in updates if isinstance(item, dict))


@pytest.mark.parametrize("position", ["before", "after"])
def test_timestamp_position(tmp_path, position):
    state = {
        "source_folder": str(tmp_path),
        "folder_name_translation": "作品",
        "mode": engine.MODE_AUDIO,
        "timestamp_footer": "附加说明",
        "timestamp_footer_position": position,
        "title_translations": {"01.wav": "标题"},
        "timeline": [{"filename": "01.wav", "title_ja": "原名", "start_samples": 0}],
    }
    text = engine.write_timestamp_document(state, tmp_path).read_text(encoding="utf-8")
    assert (text.index("附加说明") < text.index("00:00:00")) == (position == "before")


def test_options_defaults_and_config():
    defaults = UserSettings()
    assert not config_from_settings(defaults).original_hard_subtitles
    assert config_from_settings(defaults).timestamp_footer_position == "after"
    settings = UserSettings(
        autoflow_original_hard_subtitles=True, autoflow_timestamp_footer_position="before"
    )
    assert config_from_settings(settings).original_hard_subtitles
    assert config_from_settings(settings).timestamp_footer_position == "before"


@pytest.mark.parametrize("position", ["before", "after"])
def test_summary_position(tmp_path, position):
    source = SimpleNamespace(relative_path="01.wav", path=tmp_path / "01.wav", title_ja="原文")
    _, timeline = engine.write_smart_summary(
        tmp_path,
        source_folder=tmp_path,
        mode=engine.MODE_VIDEO_NORMAL,
        layout=engine.LAYOUT_MERGED,
        sources=[source],
        states=[{"timeline": []}],
        descriptors=[],
        folder_translation="作品",
        title_translations={"01.wav": "中文"},
        footer="附加说明",
        footer_position=position,
        harmonized_delay_seconds=0,
    )
    text = timeline.read_text(encoding="utf-8")
    assert (text.index("附加说明") < text.index("00:00:00")) == (position == "before")


@pytest.mark.integration
def test_real_original_hardsub_video(tmp_path, monkeypatch):
    import av
    import imageio_ffmpeg
    import numpy as np
    import soundfile as sf

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(32000, dtype=np.float32), 16000)
    subtitle = tmp_path / "source.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,500\nHard subtitle test\n", encoding="utf-8")
    output = tmp_path / "original-subtitle.mp4"
    monkeypatch.setattr(engine, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(engine, "VIDEO_FILTER_SIZE", "640:360")
    paths = SimpleNamespace(
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(),
        video_encoder_options=("-c:v", "libx264", "-preset", "ultrafast"),
    )
    engine.render_static_bilingual_video(
        paths, source, None, subtitle, output, require_hard_subtitles=True
    )
    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert len(container.streams.subtitles) == 0
        frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
        assert frame.max() > 100  # White subtitle pixels in otherwise black frame.


def test_hard_subtitle_failure_does_not_fallback(tmp_path, monkeypatch):
    subtitle = tmp_path / "source.srt"
    subtitle.write_text("example")
    monkeypatch.setattr(engine, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(engine, "render_static_video", lambda *a, **k: None)

    def fail(*a, **k):
        raise engine.VideoPreparerError("libass unavailable")

    monkeypatch.setattr(engine, "run_ffmpeg", fail)
    monkeypatch.setattr(
        engine, "remux_video_with_subtitle", lambda *a: pytest.fail("must not use soft subtitles")
    )
    with pytest.raises(engine.VideoPreparerError, match="libass"):
        engine.render_static_bilingual_video(
            SimpleNamespace(video_encoder_options=()),
            tmp_path / "audio.wav",
            None,
            subtitle,
            tmp_path / "output.mp4",
            require_hard_subtitles=True,
        )


@pytest.mark.parametrize(
    "mode", [engine.MODE_AUDIO, engine.MODE_VIDEO_NORMAL, engine.MODE_VIDEO_HARMONIZED]
)
@pytest.mark.parametrize("subtitles_only", [True, False])
def test_original_hard_subtitle_execution(tmp_path, monkeypatch, mode, subtitles_only):
    master = tmp_path / "master.wav"
    master.write_bytes(b"original")
    project = tmp_path / "project.json"
    project.write_text("{}")
    state = {
        "status": "subtitles_ready",
        "mode": mode,
        "subtitles_only": subtitles_only,
        "original_hard_subtitles": True,
        "embed_subtitles": False,
        "master_audio": str(master),
        "project_json": str(project),
        "original_media": str(master),
        "harmonized_delay_seconds": 120,
        "harmonized_volume_db": -10,
        "timeline": [],
        "title_translations": {},
        "folder_name_translation": "作品",
        "source_folder": str(tmp_path),
    }

    def copied(*a, **k):
        return {"srt": str(tmp_path / "out.srt"), "audio": str(master)}

    monkeypatch.setattr(engine, "copy_final_outputs", copied)
    monkeypatch.setattr(engine, "copy_subtitle_only_outputs", copied)
    rendered = []
    monkeypatch.setattr(
        engine, "render_static_bilingual_video", lambda *a, **k: rendered.append((a, k))
    )
    monkeypatch.setattr(engine, "write_timestamp_document", lambda *a: tmp_path / "timestamp.txt")
    engine.execute_task(SimpleNamespace(), tmp_path, tmp_path / "state.json", state, [])
    assert len(rendered) == (0 if mode == engine.MODE_AUDIO else 1)
    if rendered:
        assert rendered[0][0][1] == master
        assert rendered[0][1]["require_hard_subtitles"] is True
        assert rendered[0][1]["lead_seconds"] == (
            120 if mode == engine.MODE_VIDEO_HARMONIZED else 0
        )
        assert "original_subtitle_video" in state["outputs"]
