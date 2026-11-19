from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.audio import probe_audio, sha256_file
from asmr_dubber.errors import TranslationError
from asmr_dubber.experimental_spatial import _suppress_vocal_residual
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, load_project
from asmr_dubber.pipeline import (
    export_transcript,
    generate_subtitles,
    mix_project,
    synthesize_project,
    translate_project,
)
from asmr_dubber.translation_machine import DeepLTranslator
from asmr_dubber.translation_schema import TranslationChunk, TranslationEntry
from asmr_dubber.tts import spoken_text, tts_cache_key, valid_tts_cache
from asmr_dubber.zh_phonetics import (
    apply_spoken_normalization,
    int_to_spoken_zh,
    normalize_chinese_spoken_numbers,
    parse_pronunciation_glossary,
)


def _write_wav(path: Path, seconds: float = 4.0, rate: int = 16_000, mono: bool = False) -> Path:
    t = np.arange(round(seconds * rate), dtype=np.float64) / rate
    sig = (0.25 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
    sf.write(path, sig if mono else np.column_stack([sig, sig]), rate)
    return path


def test_w15_pronunciation_glossary_and_chinese_spoken_numbers() -> None:
    rules = parse_pronunciation_glossary("# 专名读音\n行长=航长\n银行行长 -> 银行航长\nASMR：音声")
    assert rules[0] == ("银行行长", "银行航长")
    assert (
        apply_spoken_normalization(
            "这位银行行长喜欢听ASMR", glossary="行长=航长\n银行行长 -> 银行航长\nASMR：音声"
        )
        == "这位银行航长喜欢听音声"
    )

    assert int_to_spoken_zh(0) == "零"
    assert int_to_spoken_zh(12) == "十二"
    assert int_to_spoken_zh(105) == "一百零五"
    assert int_to_spoken_zh(10_020) == "一万零二十"
    assert normalize_chinese_spoken_numbers("2026年完成度100%，误差3.14%，共25句，编号007") == (
        "二零二六年完成度百分之百，误差百分之三点一四，共二十五句，编号零零七"
    )

    # Selective cache invalidation: sentences without glossary matches keep identical cache key!
    src_info = {
        "path": "source.wav",
        "sha256": "a" * 64,
        "duration_seconds": 6.0,
        "sample_rate": 16_000,
        "channels": 2,
    }
    s_plain = Sentence(
        id="s000001", start_seconds=0.2, end_seconds=1.5, source_text="a", target_text="普通台词"
    )
    s_poly = Sentence(
        id="s000002", start_seconds=1.8, end_seconds=3.2, source_text="b", target_text="请重来一次"
    )
    p_base = DubProject(source=src_info, settings=ProjectSettings(), sentences=[s_plain, s_poly])
    k_plain_before = tts_cache_key(p_base, s_plain)
    k_poly_before = tts_cache_key(p_base, s_poly)

    p_gloss = DubProject(
        source=src_info,
        settings=ProjectSettings(tts_pronunciation_glossary="重来=虫来"),
        sentences=[s_plain, s_poly],
    )
    assert tts_cache_key(p_gloss, s_plain) == k_plain_before
    assert tts_cache_key(p_gloss, s_poly) != k_poly_before
    assert spoken_text(p_gloss.settings, "请重来一次") == "请虫来一次"


def test_w27_hand_written_old_manifest_opens_translates_mixes_without_forced_resynthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    proj_dir = tmp_path / "legacy_221_project"
    proj_dir.mkdir(parents=True)
    src_wav = _write_wav(proj_dir / "source.wav", seconds=5.0)
    src_hash = sha256_file(src_wav)
    info = probe_audio(src_wav, sha256=src_hash)
    info.path = "source.wav"

    # Cache key the current code gives s000001 under pre-2.3 settings (edge_tts).
    legacy_equiv = DubProject(
        source=info,
        settings=ProjectSettings(
            tts_backend="edge_tts",
            tts_model="edge-tts",
            tts_voice="zh-CN-XiaoxiaoNeural",
            separation_enabled=False,
            spatial_rtf_enabled=False,
            reference_audio_mode="legacy",
            tts_normalize_pauses=False,
            tts_fit_source_duration=False,
            tts_dual_voice_anchor=False,
            tts_spoken_normalization=False,
        ),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.2,
                end_seconds=1.8,
                source_text="おはよう……第1話",
                target_text="早上好……第1话",
                status="synthesized",
            )
        ],
    )
    s1_key = tts_cache_key(legacy_equiv, legacy_equiv.sentences[0])
    dub_dir = proj_dir / "chinese"
    dub_dir.mkdir(parents=True)
    s1_wav = _write_wav(dub_dir / f"s000001-{s1_key}.wav", seconds=0.9, mono=True)

    # Hand-written older manifest (ja_text, retain_chinese_stem); projects written by the
    # real 2.2.1 code are covered by tests/test_legacy_2_2_1_projects.py.
    v221_manifest = {
        "schema_version": 2,
        "app_version": "2.2.1",
        "revision": 4,
        "source": info.model_dump(),
        "subtitle_language": "ja",
        "settings": {
            "tts_backend": "edge_tts",
            "tts_model": "edge-tts",
            "tts_voice": "zh-CN-XiaoxiaoNeural",
            "tts_quality_check": False,
            "separation_enabled": False,
            "spatial_rtf_enabled": False,
            "retain_chinese_stem": True,
            "translation_provider": "openai_compatible",
            "translation_base_url": "http://127.0.0.1:9999/v1",
        },
        "sentences": [
            {
                "id": "s000001",
                "start_seconds": 0.2,
                "end_seconds": 1.8,
                "ja_text": "おはよう……第1話",
                "zh_text": "早上好……第1话",
                "status": "synthesized",
                "tts_file": f"chinese/{s1_wav.name}",
                "tts_duration_seconds": 0.9,
                "tts_cache_key": s1_key,
            },
            {
                "id": "s000002",
                "start_seconds": 2.2,
                "end_seconds": 3.8,
                "ja_text": "おやすみなさい",
                "zh_text": "",
                "status": "pending",
            },
        ],
    }
    manifest_path = proj_dir / "project.json"
    manifest_path.write_text(json.dumps(v221_manifest, ensure_ascii=False), encoding="utf-8")

    # 1. Open in 3.0 -> migrates to v4 without invalidating s000001's 2.2.1 dub cache!
    project, directory = load_project(manifest_path)
    assert project.schema_version == 4
    assert project.subtitle_language == "source"
    assert project.settings.mix_output_mode == "both"
    assert valid_tts_cache(project, project.sentences[0], directory) is True

    # 2. Continue translating remaining sentence s000002!
    with patch(
        "asmr_dubber.translation.LLMTranslator.translate_chunk",
        side_effect=lambda chunk, *_a, **_k: {
            s.id: TranslationEntry(zh="晚安好梦", kind="speech") for s in chunk.sentences
        },
    ):
        translate_project(project, directory, api_key="local")

    assert project.sentences[0].target_text == "早上好……第1话"
    assert project.sentences[1].target_text == "晚安好梦"

    # 3. Synthesize -> ONLY s000002 is synthesized; s000001 is reused from 2.2.1 cache!
    synth_ids: list[str] = []
    with patch(
        "asmr_dubber.tts_backends._runner",
        side_effect=lambda _p: (
            lambda s, _ref, tmp: (synth_ids.append(s.id), _write_wav(tmp, 0.8, mono=True))[1],
            lambda: None,
        ),
    ):
        synthesize_project(project, directory)

    assert synth_ids == ["s000002"]

    # 4. Mix & export subtitles & transcript cleanly!
    mixed = mix_project(project, directory)
    assert mixed.is_file()
    srt, lrc, _ = generate_subtitles(project, directory, language="bilingual")
    assert srt.is_file() and lrc.is_file()
    export_transcript(project, directory)
    assert len(list((directory / "backups").glob("project-schema-v2-*.json"))) == 1


def test_w25_spatial_wiener_suppression_and_machine_translator_errors() -> None:
    rate = 48_000
    t = np.arange(rate, dtype=np.float64) / rate
    vocal = np.column_stack(
        [0.4 * np.sin(2 * np.pi * 16_000 * t), 0.4 * np.sin(2 * np.pi * 16_000 * t)]
    ).astype(np.float32)
    residual = (0.15 * vocal).astype(np.float32)
    removal = np.ones(rate, dtype=np.float32)
    suppressed = _suppress_vocal_residual(residual, vocal, removal, rate, strength=0.6)
    assert suppressed.shape == residual.shape
    assert float(np.sqrt(np.mean(suppressed**2))) < float(np.sqrt(np.mean(residual**2)))

    # Mono branch & zero-strength no-op
    mono_out = _suppress_vocal_residual(residual[:, :1], vocal[:, :1], removal, rate, strength=0.5)
    assert mono_out.shape == (rate, 1)
    assert _suppress_vocal_residual(residual, vocal, removal, rate, strength=0.0) is residual

    # Machine translator error branches
    chunk = TranslationChunk(
        [Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="a")]
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(503, text="busy"))
    ) as client:
        tr = DeepLTranslator(
            api_key="k", model="deepl", base_url="https://api.deepl.com", client=client
        )
        with pytest.raises(TranslationError, match="HTTP 503"):
            tr.translate_chunk(chunk, "[]", "[]", "job")

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"translations": [{"text": "   "}]})
        )
    ) as client:
        tr2 = DeepLTranslator(
            api_key="k", model="deepl", base_url="https://api.deepl.com", client=client
        )
        entries = tr2.translate_chunk(chunk, "[]", "[]", "job")
        assert entries == {"s1": ""}

    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"translations": []}))
    ) as client:
        tr3 = DeepLTranslator(
            api_key="k", model="deepl", base_url="https://api.deepl.com", client=client
        )
        with pytest.raises(TranslationError, match="数量无效"):
            tr3.translate_chunk(chunk, "[]", "[]", "job")
