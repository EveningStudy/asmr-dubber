import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import (
    StemEvent,
    active_rms_dbfs,
    build_chinese_stem,
    copy_source_verbatim,
    extract_reference,
    make_analysis_copy,
    mix_original_and_stem,
    mux_mixed_video,
    probe_audio,
    project_file_exists,
    sentence_events,
    sha256_file,
    verify_source,
)
from asmr_dubber.environment import ffmpeg_executable
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import Sentence


def test_model_copies_are_16khz_float_without_touching_source(tmp_path: Path) -> None:
    rate = 48_000
    source = tmp_path / "quiet_source.wav"
    wave = np.full((rate, 2), 1e-5, dtype=np.float32)
    sf.write(source, wave, rate, subtype="FLOAT")
    original_hash = sha256_file(source)

    reference = extract_reference(source, tmp_path / "reference.wav", 0.1, 0.9)
    analysis = make_analysis_copy(source, tmp_path / "analysis.wav")

    for derived in (reference, analysis):
        info = sf.info(derived)
        assert info.samplerate == 16_000
        assert info.channels == 1
        assert info.subtype == "FLOAT"
    assert sha256_file(source) == original_hash


def test_mix_adds_stem_without_normalizing_original(tmp_path: Path) -> None:
    rate = 48_000
    frames = rate
    time = np.arange(frames, dtype=np.float32) / rate
    original = np.column_stack(
        [0.4 * np.sin(2 * np.pi * 220 * time), 0.3 * np.sin(2 * np.pi * 330 * time)]
    ).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, original, rate, subtype="FLOAT")
    original_hash = sha256_file(source)

    chinese = tmp_path / "line.wav"
    sf.write(chinese, np.full(rate // 4, 0.1, dtype=np.float32), rate, subtype="FLOAT")
    info = probe_audio(source)
    stem = tmp_path / "stem.wav"
    build_chinese_stem(
        stem,
        [StemEvent("s000001", 0.5, chinese)],
        info,
        0.0,
        normalize_loudness=False,
        stem_peak_dbfs=None,
    )

    output = tmp_path / "output.wav"
    mix_original_and_stem(source, stem, output, info)
    mixed, _ = sf.read(output, dtype="float32", always_2d=True)
    stem_data, _ = sf.read(stem, dtype="float32", always_2d=True)
    original_decoded, _ = sf.read(source, dtype="float32", always_2d=True)

    assert sha256_file(source) == original_hash
    np.testing.assert_allclose(mixed, original_decoded + stem_data, atol=2e-6, rtol=0)
    np.testing.assert_allclose(mixed[: rate // 2], original_decoded[: rate // 2], atol=2e-6)
    assert sf.info(output).subtype == "PCM_24"


def test_float_mix_option_preserves_samples_above_full_scale(tmp_path: Path) -> None:
    rate = 16_000
    source = tmp_path / "source.wav"
    stem = tmp_path / "stem.wav"
    output = tmp_path / "float-master.wav"
    sf.write(source, np.full(rate, 0.75, dtype=np.float32), rate, subtype="FLOAT")
    sf.write(stem, np.full(rate, 0.5, dtype=np.float32), rate, subtype="FLOAT")

    mix_original_and_stem(
        source,
        stem,
        output,
        probe_audio(source),
        output_codec="pcm_f32le",
        peak_protection=False,
    )

    data, _ = sf.read(output, dtype="float32")
    assert sf.info(output).subtype == "FLOAT"
    assert float(np.max(data)) > 1.0


def test_source_copy_hashes_while_copying_and_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    sf.write(source, np.zeros(8_000, dtype=np.float32), 8_000, subtype="PCM_16")
    project_dir = tmp_path / "project"
    progress: list[tuple[int, int]] = []

    copied, info = copy_source_verbatim(
        source,
        project_dir,
        progress=lambda _message, current, total: progress.append((current, total)),
    )

    assert copied.read_bytes() == source.read_bytes()
    assert info.sha256 == sha256_file(source)
    assert progress[-1][0] == progress[-1][1] == source.stat().st_size
    assert verify_source(project_dir, info) == copied
    info.path = "../input.wav"
    with pytest.raises(ProjectError, match="超出项目目录"):
        verify_source(project_dir, info)


def test_video_input_is_detected_and_mixed_audio_is_muxed_without_reencoding_video(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    subprocess.run(
        [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )
    info = probe_audio(source)
    assert info.media_type == "video"
    assert info.video_width == 160
    assert info.video_height == 90
    assert info.video_codec == "mpeg4"

    mixed = tmp_path / "mixed.wav"
    sf.write(mixed, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_24")
    output = mux_mixed_video(source, mixed, tmp_path / "output.mp4")

    output_info = probe_audio(output)
    assert output.is_file()
    assert output_info.media_type == "video"
    assert output_info.video_codec == info.video_codec


def test_sentence_audio_path_cannot_escape_project(tmp_path: Path) -> None:
    external = tmp_path / "outside.wav"
    sf.write(external, np.zeros(100, dtype=np.float32), 8_000, subtype="FLOAT")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    sentence = Sentence(
        id="s000001",
        start_seconds=0,
        end_seconds=1,
        ja_text="テスト",
        zh_text="测试",
        tts_file="../outside.wav",
    )

    with pytest.raises(ProjectError, match="超出项目目录"):
        sentence_events(project_dir, [sentence])

    with pytest.raises(ProjectError, match="超出项目目录"):
        project_file_exists(project_dir, "../outside.wav", "中文音频")


def test_absolute_manifest_audio_path_is_rejected(tmp_path: Path) -> None:
    external = tmp_path / "outside.wav"
    sf.write(external, np.zeros(100, dtype=np.float32), 8_000, subtype="FLOAT")
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(ProjectError, match="必须位于项目目录内"):
        project_file_exists(project_dir, str(external), "中文音频")


def test_sentence_events_plan_from_actual_audio_durations(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    chinese_dir = project_dir / "chinese"
    chinese_dir.mkdir(parents=True)
    rate = 8_000
    sf.write(chinese_dir / "s1.wav", np.zeros(round(rate * 2.2)), rate, subtype="FLOAT")
    sf.write(chinese_dir / "s2.wav", np.zeros(rate), rate, subtype="FLOAT")
    sentences = [
        Sentence(
            id="s1",
            start_seconds=1.0,
            end_seconds=2.0,
            ja_text="一。",
            zh_text="一。",
            tts_file="chinese/s1.wav",
        ),
        Sentence(
            id="s2",
            start_seconds=3.0,
            end_seconds=4.0,
            ja_text="二。",
            zh_text="二。",
            tts_file="chinese/s2.wav",
        ),
    ]

    events = sentence_events(project_dir, sentences, 500, 1.2)

    assert [event.start_seconds for event in events] == [1.5, 3.5]
    assert events[0].speed_factor == pytest.approx(1.1)
    assert events[0].effective_duration_seconds == pytest.approx(2.0)

    sequential = sentence_events(project_dir, sentences, 500, 4.0, "sequential")

    assert [event.start_seconds for event in sequential] == pytest.approx([1.5, 3.7])
    assert [event.speed_factor for event in sequential] == [1.0, 1.0]


def test_chinese_lines_get_consistent_active_level_without_touching_raw_files(
    tmp_path: Path,
) -> None:
    rate = 48_000
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((rate, 2), dtype=np.float32), rate, subtype="FLOAT")
    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    sf.write(quiet, np.full(rate // 5, 0.01, dtype=np.float32), rate, subtype="FLOAT")
    sf.write(loud, np.full(rate // 5, 0.5, dtype=np.float32), rate, subtype="FLOAT")
    quiet_hash = sha256_file(quiet)
    loud_hash = sha256_file(loud)

    stem = tmp_path / "normalized_stem.wav"
    build_chinese_stem(
        stem,
        [StemEvent("quiet", 0.1, quiet), StemEvent("loud", 0.6, loud)],
        probe_audio(source),
        0.0,
        target_active_rms_dbfs=-30.0,
        stem_peak_dbfs=-3.0,
    )
    data, _ = sf.read(stem, dtype="float32", always_2d=True)
    quiet_level = active_rms_dbfs(data[int(0.1 * rate) : int(0.3 * rate), 0], rate)
    loud_level = active_rms_dbfs(data[int(0.6 * rate) : int(0.8 * rate), 0], rate)

    assert quiet_level == pytest.approx(-30.0, abs=0.5)
    assert loud_level == pytest.approx(-30.0, abs=0.5)
    assert sha256_file(quiet) == quiet_hash
    assert sha256_file(loud) == loud_hash


def test_chinese_overlap_is_peak_protected(tmp_path: Path) -> None:
    rate = 48_000
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((rate, 2), dtype=np.float32), rate, subtype="FLOAT")
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, np.full(rate // 4, 0.5, dtype=np.float32), rate, subtype="FLOAT")
    sf.write(second, np.full(rate // 4, 0.5, dtype=np.float32), rate, subtype="FLOAT")

    stem = tmp_path / "protected_stem.wav"
    build_chinese_stem(
        stem,
        [StemEvent("first", 0.25, first), StemEvent("second", 0.25, second)],
        probe_audio(source),
        0.0,
        normalize_loudness=False,
        stem_peak_dbfs=-3.0,
    )
    data, _ = sf.read(stem, dtype="float32")
    assert float(np.max(np.abs(data))) <= 10.0 ** (-3.0 / 20.0) + 1e-6


def test_chinese_loudness_follows_japanese_with_audible_floor(tmp_path: Path) -> None:
    rate = 48_000
    source = tmp_path / "source.wav"
    source_wave = np.concatenate(
        [
            np.full(rate, 10.0 ** (-36.0 / 20.0), dtype=np.float32),
            np.full(rate, 10.0 ** (-55.0 / 20.0), dtype=np.float32),
        ]
    )
    sf.write(source, source_wave, rate, subtype="FLOAT")
    chinese = tmp_path / "chinese.wav"
    sf.write(chinese, np.full(rate // 4, 0.3, dtype=np.float32), rate, subtype="FLOAT")

    stem = tmp_path / "matched.wav"
    build_chinese_stem(
        stem,
        [
            StemEvent("normal-quiet", 0.1, chinese, 0.0, 1.0),
            StemEvent("extremely-quiet", 1.1, chinese, 1.0, 2.0),
        ],
        probe_audio(source),
        0.0,
        source_reference_path=source,
        match_source_loudness=True,
        relative_loudness_db=0.0,
        minimum_active_rms_dbfs=-42.0,
        target_active_rms_dbfs=-30.0,
    )
    data, _ = sf.read(stem, dtype="float32", always_2d=True)
    first = active_rms_dbfs(data[int(0.1 * rate) : int(0.35 * rate), 0], rate)
    second = active_rms_dbfs(data[int(1.1 * rate) : int(1.35 * rate), 0], rate)

    assert first == pytest.approx(-36.0, abs=0.5)
    assert second == pytest.approx(-42.0, abs=0.5)


@pytest.mark.parametrize("speed_factor", [1.25, 3.0])
def test_mix_time_stretch_shortens_clip_without_changing_pitch_or_raw_cache(
    tmp_path: Path,
    speed_factor: float,
) -> None:
    rate = 16_000
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(rate * 3, dtype=np.float32), rate, subtype="FLOAT")
    time = np.arange(rate * 2, dtype=np.float32) / rate
    chinese = tmp_path / "line.wav"
    sf.write(
        chinese,
        (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32),
        rate,
        subtype="FLOAT",
    )
    original_hash = sha256_file(chinese)

    stem = tmp_path / "sped-up.wav"
    build_chinese_stem(
        stem,
        [StemEvent("s000001", 0.0, chinese, speed_factor=speed_factor)],
        probe_audio(source),
        0.0,
        normalize_loudness=False,
        stem_peak_dbfs=None,
    )

    data, _ = sf.read(stem, dtype="float32")
    active = np.flatnonzero(np.abs(data) > 1e-4)
    expected_duration = 2.0 / speed_factor
    assert active[-1] / rate == pytest.approx(expected_duration, abs=0.08)
    analysis = data[int(0.08 * rate) : int((expected_duration - 0.08) * rate)]
    spectrum = np.abs(np.fft.rfft(analysis * np.hanning(len(analysis))))
    frequencies = np.fft.rfftfreq(len(analysis), 1.0 / rate)
    dominant = frequencies[int(np.argmax(spectrum))]
    assert dominant == pytest.approx(440.0, abs=3.0)
    assert sha256_file(chinese) == original_hash
    assert not list(tmp_path.glob("*.tempo.wav"))
