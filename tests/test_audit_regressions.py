import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from asmr_dubber import asr, pipeline
from asmr_dubber.audio import probe_audio
from asmr_dubber.autoflow.catalog import scan_work
from asmr_dubber.cli import app
from asmr_dubber.errors import AsmrDubberError, ProjectError
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, load_project, save_project
from asmr_dubber.subtitles import write_subtitle_files


def sentence(**kwargs):
    return Sentence(id="s000001", start_seconds=0, end_seconds=1, source_text="原文", **kwargs)


def test_digital_silence_preserves_saved_sentences_without_calling_model(tmp_path, monkeypatch):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(32000), 16000, subtype="FLOAT")
    project = DubProject(source=probe_audio(source), sentences=[sentence()])
    save_project(project, tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("Digital silence must not reach an ASR model")

    monkeypatch.setattr(asr, "_transcribe_faster_whisper", unexpected)
    project.settings.asr_backend = "faster_whisper"
    with pytest.raises(AsmrDubberError, match="数字静音"):
        pipeline.analyze_project(project, tmp_path, force=True)
    restored, _ = load_project(tmp_path)
    assert restored.sentences[0].source_text == "原文"


def test_very_quiet_signal_is_not_filtered(tmp_path, monkeypatch):
    audio = tmp_path / "quiet.wav"
    sf.write(audio, np.full(16000, 1e-9), 16000, subtype="FLOAT")
    monkeypatch.setattr(asr, "_transcribe_faster_whisper", lambda *a, **kw: ([sentence()], "ja"))
    rows, _ = asr.transcribe_source(audio, ProjectSettings(asr_backend="faster_whisper"))
    assert len(rows) == 1


@pytest.mark.parametrize("text,expected", [("", "未检测到"), ("文字", "时间戳")])
def test_no_text_and_missing_timestamps_have_distinct_errors(text, expected):
    with pytest.raises(AsmrDubberError, match=expected):
        asr._finish_tokens([], text, "ja", ProjectSettings())


def test_readability_padding_bounded_without_erasing_real_overlap(tmp_path):
    rows = [
        Sentence(id="s1", start_seconds=0, end_seconds=0.4, source_text="一"),
        Sentence(id="s2", start_seconds=0.5, end_seconds=0.9, source_text="二"),
        Sentence(id="s3", start_seconds=2.7, end_seconds=3, source_text="三"),
    ]
    warnings = []
    srt, _ = write_subtitle_files(
        rows, tmp_path, "source", media_duration_seconds=3, warnings=warnings
    )
    text = srt.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,500" in text
    assert "00:00:02,700 --> 00:00:03,000" in text
    assert len(warnings) == 2
    rows[0].end_seconds = 0.7
    srt, _ = write_subtitle_files(rows, tmp_path, "source", media_duration_seconds=3)
    assert "00:00:00,000 --> 00:00:00,700" in srt.read_text(encoding="utf-8")


def test_dubbing_subtitle_boundary_uses_actual_extended_output(tmp_path):
    source = tmp_path / "source.wav"
    mixed = tmp_path / "mixed.wav"
    sf.write(source, np.ones(32000) * 0.001, 16000)
    sf.write(mixed, np.ones(64000) * 0.001, 16000)
    project = DubProject(
        source=probe_audio(source),
        output_file="mixed.wav",
        settings=ProjectSettings(subtitle_timeline="dubbing", chinese_dubbing_offset_ms=0),
        sentences=[
            Sentence(
                id="s1",
                start_seconds=1.7,
                end_seconds=2,
                source_text="原文",
                zh_text="中文",
                tts_duration_seconds=1.0,
            )
        ],
    )
    save_project(project, tmp_path)
    srt, _, _ = pipeline.generate_subtitles(project, tmp_path, language="zh")
    assert "00:00:01,700 --> 00:00:02,700" in srt.read_text(encoding="utf-8")


def test_missing_source_rejected_before_overwriting_subtitles(tmp_path):
    old = tmp_path / "subtitles_source.srt"
    old.write_text("previous subtitles", encoding="utf-8")
    rows = [
        sentence(),
        Sentence(id="s2", start_seconds=1, end_seconds=2, source_text="", zh_text="只有中文"),
    ]
    with pytest.raises(ProjectError, match="没有原文"):
        write_subtitle_files(rows, tmp_path, "source")
    assert old.read_text(encoding="utf-8") == "previous subtitles"
    assert not (tmp_path / "subtitles_source.lrc").exists()


def test_language_suffix_pairing_and_ambiguity(tmp_path):
    (tmp_path / "01 track.wav").touch()
    for language in ("ja", "zh", "en"):
        (tmp_path / f"01 track.{language}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext")
    track = scan_work(tmp_path).editions[0].all_tracks[0]
    assert track.transcript.path.name == "01 track.ja.srt"
    (tmp_path / "01 track.ja-JP.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext")
    assert scan_work(tmp_path).editions[0].all_tracks[0].transcript is None


@pytest.mark.parametrize("timeline,cleared", [("source", False), ("dubbing", True)])
def test_cli_timing_invalidates_affected_subtitles(tmp_path, timeline, cleared):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.ones(16000) * 0.001, 16000)
    project = DubProject(
        source=probe_audio(audio),
        sentences=[sentence()],
        settings=ProjectSettings(subtitle_timeline=timeline),
        subtitle_srt_file="subtitles/old.srt",
        subtitle_lrc_file="subtitles/old.lrc",
        output_file="output/old.wav",
    )
    save_project(project, tmp_path)
    result = CliRunner().invoke(app, ["set-timing", str(tmp_path), "--offset-ms", "1500"])
    assert result.exit_code == 0, result.output
    restored, _ = load_project(tmp_path)
    assert restored.output_file is None
    assert (restored.subtitle_srt_file is None) == cleared
    assert (restored.subtitle_lrc_file is None) == cleared
