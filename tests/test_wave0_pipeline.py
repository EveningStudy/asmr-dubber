"""Regression tests for Wave 0 pipeline, subtitles, TTS skipping, compare tracks and caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline, separation, stage_result
from asmr_dubber.audio import (
    StemEvent,
    build_chinese_stem,
    probe_audio,
    sentence_events,
    sha256_file,
)
from asmr_dubber.autoflow.output_policy import policy_for_settings
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import (
    AudioInfo,
    DubProject,
    ProjectSettings,
    Sentence,
    load_project,
    save_project,
)
from asmr_dubber.services import presets, project_records, project_review
from asmr_dubber.stem_cache import build_cached_stem
from asmr_dubber.subtitles import write_subtitle_files
from asmr_dubber.translation import default_translation_prompt
from asmr_dubber.tts import tts_cache_key
from asmr_dubber.tts_backends import _selected_sentences, synthesize_with_selected_backend

RATE = 16_000


def _write_wave(path: Path, seconds: float = 3.0, *, channels: int = 2, hz: float = 220.0) -> Path:
    t = np.arange(int(RATE * seconds)) / RATE
    mono = (0.04 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    data = mono if channels == 1 else np.column_stack([mono, mono * 0.7])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, RATE, subtype="FLOAT")
    return path


def _make_project(tmp_path: Path, *, channels: int = 2, **settings_kwargs) -> DubProject:
    source = _write_wave(tmp_path / "source.wav", seconds=4.0, channels=channels)
    info = probe_audio(source, sha256=sha256_file(source))
    return DubProject(
        source=info,
        settings=ProjectSettings(
            tts_backend="edge_tts",
            normalize_chinese_loudness=False,
            match_source_loudness=False,
            mix_peak_protection=False,
            **settings_kwargs,
        ),
    )


# --- 1. Subtitle fallback on single-sentence translation failure ---


def test_subtitles_fallback_to_source_when_some_sentences_lack_translation(tmp_path: Path) -> None:
    untranslated_only = [
        Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="おはよう。"),
        Sentence(id="s2", start_seconds=1.2, end_seconds=2.2, source_text="おやすみ。"),
    ]
    for lang in ("zh", "bilingual"):
        with pytest.raises(ProjectError, match="没有中文"):
            write_subtitle_files(untranslated_only, tmp_path / "fail", lang)

    partial_sentences = [
        Sentence(
            id="s1",
            start_seconds=0.0,
            end_seconds=1.2,
            source_text="おはよう。",
            zh_text="早上好。",
            status="translated",
        ),
        Sentence(
            id="s2",
            start_seconds=1.4,
            end_seconds=2.6,
            source_text="そのまま聞いててね。",
            zh_text="",
            status="translation_failed",
        ),
        Sentence(
            id="s3",
            start_seconds=2.8,
            end_seconds=3.6,
            source_text="",
            zh_text="",
            status="translation_failed",
        ),
    ]
    bilingual_warnings: list[str] = []
    srt_bi, lrc_bi = write_subtitle_files(
        partial_sentences, tmp_path / "bi", "bilingual", warnings=bilingual_warnings
    )
    bi_text = srt_bi.read_text(encoding="utf-8")
    assert "おはよう。\n早上好。" in bi_text
    assert "そのまま聞いててね。" in bi_text
    assert "[00:01.40]そのまま聞いててね。" in lrc_bi.read_text(encoding="utf-8")
    assert any("s2" in item and "回退为原文" in item for item in bilingual_warnings)
    assert any("s3" in item and "跳过字幕" in item for item in bilingual_warnings)

    zh_warnings: list[str] = []
    srt_zh, _ = write_subtitle_files(partial_sentences, tmp_path / "zh", "zh", warnings=zh_warnings)
    zh_text = srt_zh.read_text(encoding="utf-8")
    assert "早上好。" in zh_text and "おはよう。" not in zh_text
    assert "そのまま聞いててね。" in zh_text
    assert any("s2" in item for item in zh_warnings)


# --- 2. chinese_audio_enabled=False skipped across TTS, stage_result & mix ---


def test_chinese_audio_disabled_skips_synthesis_and_preserves_sound_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path, tts_quality_check=False)
    project.sentences = [
        Sentence(
            id="s1",
            start_seconds=0.1,
            end_seconds=1.2,
            source_text="こんにちは",
            zh_text="你好",
            status="translated",
        ),
        Sentence(
            id="s2",
            start_seconds=1.5,
            end_seconds=2.5,
            source_text="ちゅっ、ちゅぱ",
            zh_text="chu, chupa",
            status="skipped_sound",
            chinese_audio_enabled=False,
        ),
    ]
    save_project(project, tmp_path)
    assert [s.id for s in _selected_sentences(project, None)] == ["s1"]

    synthesized_ids: list[str] = []

    def fake_runner(_proj):
        def run(sentence: Sentence, _ref, output: Path) -> None:
            synthesized_ids.append(sentence.id)
            _write_wave(output, seconds=0.5, channels=1, hz=440.0)

        return run, lambda: None

    monkeypatch.setattr("asmr_dubber.tts_backends._runner", fake_runner)
    pipeline.synthesize_project(project, tmp_path)
    assert synthesized_ids == ["s1"]
    assert project.sentences[1].status == "skipped_sound"
    assert project.sentences[1].tts_file is None

    total, failed = stage_result.collect_failures("synthesize", project)
    assert total == 1 and failed == []

    # Cached rerun should see needs_generation == False and keep output_file intact.
    project.output_file = "output/keep.wav"
    save_project(project, tmp_path)
    pipeline.synthesize_project(project, tmp_path)
    assert synthesized_ids == ["s1"]
    assert project.output_file == "output/keep.wav"


def test_all_muted_sentences_pass_synthesis_and_mix_while_all_missing_dubs_fail(
    tmp_path: Path,
) -> None:
    muted_dir = tmp_path / "muted"
    muted_project = _make_project(muted_dir)
    muted_project.sentences = [
        Sentence(
            id="s1",
            start_seconds=0.2,
            end_seconds=1.2,
            source_text="ちゅっ",
            zh_text="chu",
            status="skipped_sound",
            chinese_audio_enabled=False,
        )
    ]
    save_project(muted_project, muted_dir)
    assert (
        synthesize_with_selected_backend(muted_project, muted_dir, muted_dir / "source.wav") == []
    )
    mixed = pipeline.mix_project(muted_project, muted_dir)
    assert mixed.is_file()

    # If a sentence actually needs dubbing and all dubs are missing, allow_missing still fails.
    broken_dir = tmp_path / "broken"
    broken_project = _make_project(broken_dir)
    broken_project.sentences = [
        Sentence(
            id="s1",
            start_seconds=0.2,
            end_seconds=1.2,
            source_text="こんにちは",
            zh_text="你好",
            status="error",
            chinese_audio_enabled=True,
        ),
        Sentence(
            id="s2",
            start_seconds=1.5,
            end_seconds=2.5,
            source_text="ちゅっ",
            zh_text="chu",
            status="skipped_sound",
            chinese_audio_enabled=False,
        ),
    ]
    save_project(broken_project, broken_dir)
    with pytest.raises(ProjectError, match="没有可混入的中文配音"):
        pipeline.mix_project(broken_project, broken_dir, allow_missing=True)

    # An untranslated project must also refuse to mix empty events.
    raw_dir = tmp_path / "raw"
    raw_project = _make_project(raw_dir)
    raw_project.sentences = [
        Sentence(id="s1", start_seconds=0.2, end_seconds=1.2, source_text="未翻訳", zh_text="")
    ]
    save_project(raw_project, raw_dir)
    with pytest.raises(ProjectError, match="没有可混入的中文配音"):
        pipeline.mix_project(raw_project, raw_dir)


# --- 3. Single-project bilingual & replacement coexistence + stale tracking ---


def test_single_project_keeps_both_variants_for_compare_and_tracks_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path, mix_output_mode="mixed")
    dub = _write_wave(tmp_path / "chinese" / "s1.wav", seconds=0.6, channels=1, hz=520.0)
    sentence = Sentence(
        id="s1",
        start_seconds=0.3,
        end_seconds=1.3,
        source_text="おはよう",
        zh_text="早上好",
        tts_file=dub.relative_to(tmp_path).as_posix(),
        tts_duration_seconds=0.6,
        status="synthesized",
    )
    project.sentences = [sentence]
    sentence.tts_cache_key = tts_cache_key(project, sentence)
    save_project(project, tmp_path)

    vocals = _write_wave(tmp_path / "sep" / "vocals.wav", seconds=4.0, channels=2, hz=220.0)
    bg = _write_wave(tmp_path / "sep" / "bg.wav", seconds=4.0, channels=2, hz=110.0)
    monkeypatch.setattr(separation, "ensure_separation", lambda *a, **kw: (vocals, bg))
    register = lambda p: f"/media/{p.relative_to(tmp_path).as_posix()}"  # noqa: E731

    out_bi = pipeline.mix_project(project, tmp_path)
    assert out_bi.parent == tmp_path / "output"
    assert (tmp_path / "output" / "bilingual" / out_bi.name).is_file()
    bi_bytes = (tmp_path / "output" / "bilingual" / out_bi.name).read_bytes()

    project.settings.separation_enabled = True
    project.settings.separation_mix_mode = "replace"
    save_project(project, tmp_path)
    out_rep = pipeline.mix_project(project, tmp_path)
    assert out_rep == out_bi
    assert (tmp_path / "output" / "replace" / out_rep.name).is_file()
    assert (tmp_path / "output" / "bilingual" / out_bi.name).read_bytes() == bi_bytes

    tracks = {
        t["key"]: t
        for t in project_review.compare_tracks(project, tmp_path, "/media/source", register)
    }
    assert list(tracks) == ["source", "bilingual", "replacement"]
    assert tracks["bilingual"]["stale"] is False
    assert tracks["replacement"]["stale"] is False

    # Editing a sentence invalidates outputs -> both existing tracks become stale.
    rows = project_records.project_rows(project)
    rows[0][9] = -3.0
    assert project_records.apply_table(project, rows) is True
    save_project(project, tmp_path)
    after_edit = {
        t["key"]: t
        for t in project_review.compare_tracks(project, tmp_path, "/media/source", register)
    }
    assert after_edit["bilingual"]["stale"] is True
    assert after_edit["replacement"]["stale"] is True

    # Re-mixing only replacement refreshes replacement while bilingual remains stale.
    pipeline.mix_project(project, tmp_path)
    remixed = {
        t["key"]: t
        for t in project_review.compare_tracks(project, tmp_path, "/media/source", register)
    }
    assert remixed["replacement"]["stale"] is False
    assert remixed["bilingual"]["stale"] is True


# --- 4. Replacement dubbing-timeline subtitles & export_transcript alignment ---


def test_replacement_dubbing_timeline_aligns_subtitles_and_transcript_with_trimmed_events(
    tmp_path: Path,
) -> None:
    project = _make_project(
        tmp_path,
        separation_enabled=True,
        separation_mix_mode="replace",
        subtitle_timeline="dubbing",
        subtitle_min_duration_seconds=0.2,
        subtitle_max_cps=40.0,
        chinese_dubbing_offset_ms=0,
        chinese_max_auto_speed=1.0,
        chinese_dubbing_timing_mode="fit_window",
        replacement_dubbing_offset_ms=300,
        replacement_max_auto_speed=1.5,
        replacement_timing_mode="fit_then_shift",
    )
    # 2.0s WAV with only 0.6s of audible speech surrounded by leading/trailing silence.
    padded = np.zeros(RATE * 2, dtype=np.float32)
    active_samples = int(RATE * 0.6)
    padded[int(RATE * 0.5) : int(RATE * 0.5) + active_samples] = 0.15 * np.sin(
        2 * np.pi * 330 * np.arange(active_samples) / RATE
    ).astype(np.float32)
    clip_path = tmp_path / "chinese" / "s1.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(clip_path, padded, RATE, subtype="FLOAT")

    s1 = Sentence(
        id="s1",
        start_seconds=1.0,
        end_seconds=1.8,
        source_text="一",
        zh_text="一",
        tts_file="chinese/s1.wav",
        tts_duration_seconds=2.0,
        status="synthesized",
    )
    s2 = Sentence(
        id="s2",
        start_seconds=1.7,
        end_seconds=2.5,
        source_text="二",
        zh_text="二",
        tts_file="chinese/s1.wav",
        tts_duration_seconds=2.0,
        status="synthesized",
    )
    project.sentences = [s1, s2]
    for s in project.sentences:
        s.tts_cache_key = tts_cache_key(project, s)
    save_project(project, tmp_path)

    events = sentence_events(
        tmp_path,
        project.sentences,
        project.settings.replacement_dubbing_offset_ms,
        project.settings.replacement_max_auto_speed,
        project.settings.replacement_timing_mode,
        trim_silence=True,
    )
    pipeline.export_transcript(project, tmp_path)
    transcript = json.loads((tmp_path / "exports" / "transcript.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in transcript}
    for ev in events:
        assert by_id[ev.sentence_id]["zh_start_seconds"] == pytest.approx(ev.start_seconds)
        assert by_id[ev.sentence_id]["auto_speed_factor"] == pytest.approx(ev.speed_factor)
        assert by_id[ev.sentence_id]["zh_effective_duration_seconds"] == pytest.approx(
            ev.effective_duration_seconds
        )
    assert by_id["s1"]["zh_effective_duration_seconds"] < 1.0  # trimmed from 2.0s!

    srt, _, _ = pipeline.generate_subtitles(project, tmp_path, language="zh")
    srt_text = srt.read_text(encoding="utf-8")
    assert "00:00:01,300 -->" in srt_text


# --- 5. Presets / batch policy spatial_rtf_enabled=True + mono auto-downgrade ---


def test_presets_and_batch_enable_rtf_and_mono_source_downgrades_cleanly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert presets.REPLACEMENT["spatial_rtf_enabled"] is True
    for content in ("replacement", "both"):
        pol = policy_for_settings(ProjectSettings(), content, "bilingual")
        assert pol["settings"]["spatial_rtf_enabled"] is True

    project = _make_project(
        tmp_path,
        channels=1,
        separation_enabled=True,
        separation_mix_mode="replace",
        spatial_rtf_enabled=True,
        mix_output_mode="mixed",
    )
    _write_wave(tmp_path / "chinese" / "s1.wav", seconds=0.5, channels=1, hz=440.0)
    s1 = Sentence(
        id="s1",
        start_seconds=0.2,
        end_seconds=1.2,
        source_text="テスト",
        zh_text="测试",
        tts_file="chinese/s1.wav",
        tts_duration_seconds=0.5,
        status="synthesized",
    )
    project.sentences = [s1]
    s1.tts_cache_key = tts_cache_key(project, s1)
    save_project(project, tmp_path)

    vocals = _write_wave(tmp_path / "sep" / "vocals.wav", seconds=4.0, channels=1, hz=220.0)
    bg = _write_wave(tmp_path / "sep" / "bg.wav", seconds=4.0, channels=1, hz=110.0)
    monkeypatch.setattr(separation, "ensure_separation", lambda *a, **kw: (vocals, bg))

    out = pipeline.mix_project(project, tmp_path)
    assert out.is_file() and sf.info(out).channels == 1
    assert "自动跳过双声道空间跟随" in caplog.text
    reloaded, _ = load_project(tmp_path)
    assert reloaded.settings.spatial_rtf_enabled is True


# --- 6. stem_cache voice_match_path digest & legacy translation_prompt migration ---


def test_stem_cache_hashes_voice_match_file_and_migrates_legacy_prompts(tmp_path: Path) -> None:
    source = _write_wave(tmp_path / "source.wav", seconds=1.0, channels=2, hz=220.0)
    clip = _write_wave(tmp_path / "clip.wav", seconds=0.5, channels=1, hz=440.0)
    vocals = _write_wave(tmp_path / "vocals.wav", seconds=1.0, channels=2, hz=300.0)
    calls: list[int] = []

    def build(**kwargs):
        calls.append(1)
        return build_chinese_stem(**kwargs)

    kwargs = dict(
        destination=tmp_path / "stem.wav",
        events=[StemEvent("s1", 0.0, clip, 0.0, 0.5)],
        source_info=probe_audio(source),
        chinese_gain_db=0.0,
        normalize_loudness=False,
        spatial_reference_path=source,
        spatial_settings=ProjectSettings(spatial_rtf_enabled=True),
        voice_match_path=str(vocals),
        voice_match_strength=0.8,
    )
    build_cached_stem(build, **kwargs)
    build_cached_stem(build, **kwargs)
    assert len(calls) == 1
    # Overwriting vocals at the exact same path must invalidate the cached stem.
    _write_wave(vocals, seconds=1.0, channels=2, hz=900.0)
    build_cached_stem(build, **kwargs)
    assert len(calls) == 2

    # Legacy and packaged builtin prompts are cleared on project load; custom prompts stay.
    base_payload = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="0" * 64, duration_seconds=1.0, sample_rate=RATE, channels=2
        )
    ).model_dump(mode="json")
    legacy_prompt = (
        "你是日语音声、广播剧和 ASMR 的简体中文配音翻译。\n"
        '{"translations":[{"id":"s000001","zh":"中文台词；无实义时为空字符串"}]}'
    )
    for raw_prompt, expected in [
        (legacy_prompt, ""),
        (default_translation_prompt("ja", "zh", "skip"), ""),
        ("请使用古风口吻翻译所有台词。", "请使用古风口吻翻译所有台词。"),
    ]:
        manifest_dir = tmp_path / f"proj_{len(expected)}_{abs(hash(raw_prompt)) % 10000}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            **base_payload,
            "settings": {**base_payload["settings"], "translation_prompt": raw_prompt},
        }
        (manifest_dir / "project.json").write_text(json.dumps(payload), encoding="utf-8")
        loaded, _ = load_project(manifest_dir)
        assert loaded.settings.translation_prompt == expected
