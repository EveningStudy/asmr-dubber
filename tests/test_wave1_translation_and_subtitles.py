from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import pipeline
from asmr_dubber import translation as translation_module
from asmr_dubber.audio import probe_audio, sha256_file
from asmr_dubber.errors import OperationCancelledError, ProjectError, TranslationError
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, save_project
from asmr_dubber.pipeline_subtitles import _SPEECH_DURATION_CACHE, _replacement_dubbing_durations
from asmr_dubber.subtitles import write_subtitle_files
from asmr_dubber.translation import (
    _bounded_translation_memory,
    _context_for_chunk,
    check_translations,
    translate_sentences,
    validate_translation_entries,
)
from asmr_dubber.translation_backup import (
    latest_backup,
    restore_translation_backup,
    write_translation_backup,
)
from asmr_dubber.translation_llm import LLMTranslator
from asmr_dubber.translation_revise_core import _revision_messages, validate_revisions
from asmr_dubber.translation_revision import revise_project_translations
from asmr_dubber.translation_schema import TranslationChunk
from asmr_dubber.tts import tts_cache_key


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translation_module.time, "sleep", lambda _seconds: None)


def _make_wav(path: Path, seconds: float = 4.0, channels: int = 1, rate: int = 16_000) -> Path:
    t = np.arange(round(rate * seconds)) / rate
    wave = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.column_stack([wave] * channels) if channels > 1 else wave
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, rate, subtype="FLOAT")
    return path


def _make_project(
    tmp_path: Path,
    sentences: list[Sentence],
    *,
    channels: int = 1,
    duration: float = 10.0,
    **settings_kwargs: Any,
) -> tuple[DubProject, Path]:
    source = _make_wav(tmp_path / "source.wav", seconds=duration, channels=channels)
    info = probe_audio(source, sha256=sha256_file(source))
    project = DubProject(
        source=info,
        settings=ProjectSettings(**settings_kwargs),
        sentences=sentences,
    )
    save_project(project, tmp_path)
    return project, tmp_path


# --- P0-2, P0-11, P1 in pipeline.py -----------------------------------------


def test_retranslate_selected_ids_on_fully_translated_project_creates_backup_and_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.5,
            source_text="おはよう。",
            target_text="早安。",
            status="translated",
            review_locked=True,
        ),
        Sentence(
            id="s000002",
            start_seconds=2.0,
            end_seconds=3.5,
            source_text="おやすみ。",
            target_text="晚安。",
            status="translated",
        ),
    ]
    project, directory = _make_project(
        tmp_path,
        sentences,
        translation_provider="deepseek",
        translation_model="deepseek-chat",
    )
    project.output_file = "output/old.wav"
    project.subtitle_srt_file = "subtitles/old.srt"
    save_project(project, directory)
    expected_backup_rev = project.revision

    seen_api_keys: list[str] = []
    seen_ids: list[list[str]] = []

    def fake_translate(sents, **kwargs):
        seen_api_keys.append(kwargs["api_key"])
        seen_ids.append(list(kwargs["sentence_ids"]))
        for s in sents:
            if s.id in kwargs["sentence_ids"]:
                s.zh_text = "早上好。"
        return translation_module.TranslationReport(translated=1)

    monkeypatch.setattr(pipeline, "translate_sentences", fake_translate)
    monkeypatch.setattr(pipeline, "resolve_api_key", lambda _p, k: k or "resolved-secret-key")

    pipeline.translate_project(
        project,
        directory,
        sentence_ids=["s000001"],
        overwrite_edited=True,
    )

    assert seen_api_keys == ["resolved-secret-key"]
    assert seen_ids == [["s000001"]]
    assert project.sentences[0].zh_text == "早上好。"
    assert project.sentences[1].zh_text == "晚安。"
    assert project.output_file is None
    assert project.subtitle_srt_file is None
    assert (
        latest_backup(directory) == directory / f"translations-backup-r{expected_backup_rev}.json"
    )


def test_analyze_preserves_skipped_sound_without_poisoning_previous_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, directory = _make_project(tmp_path, [], separation_enabled=False)

    emitted = [
        Sentence(
            id="s1",
            start_seconds=0.1,
            end_seconds=0.9,
            source_text="（吐息）",
            status="skipped_sound",
            enabled=False,
        ),
        Sentence(
            id="s2",
            start_seconds=1.0,
            end_seconds=1.5,
            source_text="ちゅっ",
            status="pending",
            enabled=True,
        ),
    ]
    monkeypatch.setattr(
        pipeline,
        "make_asr_analysis_copy",
        lambda _src, dst: _make_wav(dst, seconds=2.0),
    )
    monkeypatch.setattr(pipeline, "transcribe_source", lambda *a, **k: (emitted, "ja"))
    monkeypatch.setattr(
        "asmr_dubber.dub_qc.snap_sentences_to_silence",
        lambda rows, _a: rows,
    )
    monkeypatch.setattr(
        "asmr_dubber.voice_reference.annotate_sentence_voice_modes",
        lambda *_a, **_k: None,
    )

    pipeline.analyze_project(project, directory, force=True)

    assert project.sentences[0].status == "skipped_sound"
    assert project.sentences[0].source_text == "（吐息）"
    assert project.sentences[0].enabled is False
    assert project.sentences[0].error is None


def test_target_language_switch_and_force_analyze_clear_stale_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.5,
            source_text="こんにちは。",
            target_text="你好。",
            status="synthesized",
            tts_file="tts/s1.wav",
            tts_cache_key="old-key",
            tts_duration_seconds=1.2,
            tts_candidates=[{"attempt": 1, "file": "tts/s1.wav"}],
            tts_warning="old warning",
            tts_attempt=2,
            translation_warning="old trans warning",
            review_locked=True,
            error="old err",
            error_code="tts_failed",
        ),
        Sentence(
            id="s000002",
            start_seconds=2.0,
            end_seconds=2.5,
            source_text="き",
            target_text="ki",
            status="skipped_sound",
            enabled=True,
            chinese_audio_enabled=False,
        ),
        Sentence(
            id="s000003",
            start_seconds=3.0,
            end_seconds=3.5,
            source_text="ご視聴ありがとうございました",
            status="skipped_hallucination",
            enabled=False,
            error="hallucination",
        ),
    ]
    project, directory = _make_project(
        tmp_path,
        sentences,
        tts_target_language="en",
        tts_reference_sentence_id="s000001",
        tts_whisper_reference_sentence_id="s000001",
    )
    project.translation_language = "zh"
    project.translation_proposals = [{"id": "s000001", "old_zh": "你好。", "new_zh": "Hi"}]
    save_project(project, directory)

    observed_statuses: list[str] = []

    def fake_translate(sents, **_kwargs):
        observed_statuses.extend(s.status for s in sents)
        assert sents[0].tts_duration_seconds is None
        assert sents[0].tts_candidates == []
        assert sents[0].tts_warning is None
        assert sents[0].tts_attempt == 0
        assert sents[0].translation_warning is None
        assert sents[0].review_locked is False
        assert sents[0].error is None and sents[0].error_code is None
        assert project.translation_proposals == []
        sents[0].zh_text = "Hello."
        sents[0].status = "translated"
        return translation_module.TranslationReport(translated=1)

    monkeypatch.setattr(pipeline, "translate_sentences", fake_translate)
    monkeypatch.setattr(pipeline, "resolve_api_key", lambda *_a: "key")
    pipeline.translate_project(project, directory)

    assert observed_statuses == ["pending", "skipped_sound", "skipped_hallucination"]
    assert project.sentences[2].error == "hallucination"

    # Now verify force=True analyze clears proposals and reference sentence IDs
    project.translation_proposals = [{"id": "s000001"}]
    project.settings.tts_reference_sentence_id = "s000001"
    project.settings.tts_whisper_reference_sentence_id = "s000001"
    project.sentences[0].review_locked = False
    save_project(project, directory)

    monkeypatch.setattr(
        pipeline,
        "make_asr_analysis_copy",
        lambda _src, dst: _make_wav(dst, seconds=2.0),
    )
    monkeypatch.setattr(
        pipeline,
        "transcribe_source",
        lambda *a, **k: (
            [Sentence(id="s1", start_seconds=0.1, end_seconds=1.0, source_text="テスト")],
            "ja",
        ),
    )
    monkeypatch.setattr("asmr_dubber.dub_qc.snap_sentences_to_silence", lambda rows, _a: rows)
    monkeypatch.setattr(
        "asmr_dubber.voice_reference.annotate_sentence_voice_modes",
        lambda *_a, **_k: None,
    )
    pipeline.analyze_project(project, directory, force=True)
    assert project.translation_proposals == []
    assert project.settings.tts_reference_sentence_id is None
    assert project.settings.tts_whisper_reference_sentence_id is None


def test_partial_compress_invalidates_outputs_on_checkpoint_and_runs_qa_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.0,
            source_text="原文1",
            target_text="这是一段特别长的需要压缩的初版中文翻译内容",
            status="synthesized",
        ),
        Sentence(
            id="s000002",
            start_seconds=2.0,
            end_seconds=3.0,
            source_text="原文2",
            target_text="这也是一段特别长的需要压缩的初版中文翻译内容",
            status="translated",
        ),
    ]
    project, directory = _make_project(tmp_path, sentences)
    project.output_file = "output/mixed.wav"
    project.subtitle_srt_file = "subtitles/sub.srt"
    save_project(project, directory)

    def partial_compress(sents, **kwargs):
        sents[0].zh_text = "短ありがとう"  # leftover kana triggers QA check in finally!
        sents[0].status = "translated"
        kwargs["on_batch"]()
        assert project.output_file is None
        assert project.subtitle_srt_file is None
        raise OperationCancelledError("cancelled")

    monkeypatch.setattr(pipeline, "translate_sentences", partial_compress)
    monkeypatch.setattr(pipeline, "resolve_api_key", lambda *_a: "key")

    with pytest.raises(OperationCancelledError):
        pipeline.translate_project(project, directory, compress=True)

    assert project.sentences[0].translation_warning is not None
    assert "假名" in project.sentences[0].translation_warning


# --- P0-3: translation.py & translation_checks.py ---------------------------


def test_compress_includes_synthesized_and_error_and_mt_reports_unsupported() -> None:
    lines = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.0,
            source_text="原文1",
            target_text="这是一段特别长的需要压缩的已合成翻译句子内容",
            status="synthesized",
        ),
        Sentence(
            id="s000002",
            start_seconds=2.0,
            end_seconds=3.0,
            source_text="原文2",
            target_text="这是一段特别长的需要压缩的错误状态翻译句子内容",
            status="error",
            error="tts failed",
        ),
    ]
    mt_messages: list[str] = []
    translate_sentences(
        lines,
        api_key="k",
        model="quality_optimized",
        provider="deepl",
        length_budget_rate=4.5,
        compress=True,
        progress=lambda msg, _c, _t: mt_messages.append(msg),
    )
    assert mt_messages == ["当前机器翻译服务不支持自动压缩译文"]

    seen_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][-1]["content"]
        start = content.rindex("[{", 0, content.index('"source"'))
        end = content.index("}]", start) + 2
        items = json.loads(content[start:end])
        seen_ids.extend(item["id"] for item in items)
        rows = [{"id": item["id"], "zh": "简短", "kind": "speech"} for item in items]
        body = json.dumps({"translations": rows})
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines,
            api_key="k",
            model="m",
            client=client,
            length_budget_rate=4.5,
            compress=True,
        )
    assert seen_ids == ["s000001", "s000002"]
    assert report.translated == 2
    assert lines[1].error is None and lines[1].status == "translated"


def test_tighten_all_propagates_fatal_errors_and_trips_circuit_breaker() -> None:
    lines = [
        Sentence(
            id=f"s{i:06d}",
            start_seconds=float(i * 2),
            end_seconds=float(i * 2 + 1),
            source_text=f"原文{i}",
            target_text="这是一段特别长的需要压缩的超预算中文台词内容",
            status="translated",
        )
        for i in range(1, 21)
    ]
    calls = 0

    def fatal_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    with (
        httpx.Client(transport=httpx.MockTransport(fatal_handler)) as client,
        pytest.raises(TranslationError, match="401"),
    ):
        translate_sentences(
            lines,
            api_key="k",
            model="m",
            client=client,
            length_budget_rate=4.5,
            compress=True,
        )
    assert calls == 1

    many_lines = [
        Sentence(
            id=f"s{i:06d}",
            start_seconds=float(i * 2),
            end_seconds=float(i * 2 + 1),
            source_text=f"原文{i}",
            target_text="这是一段特别长的需要压缩的超预算中文台词内容",
            status="translated",
        )
        for i in range(1, 51)
    ]
    seen_500 = 0

    def breaker_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal seen_500
        seen_500 += 1
        return httpx.Response(500, text="server error")

    with (
        httpx.Client(transport=httpx.MockTransport(breaker_handler)) as client,
        pytest.raises(TranslationError, match="连续 3 批压缩失败"),
    ):
        translate_sentences(
            many_lines,
            api_key="k",
            model="m",
            client=client,
            length_budget_rate=4.5,
            compress=True,
        )
    assert seen_500 > 0


def test_checks_cover_synthesized_error_residual_cjk_and_word_tokens() -> None:
    sentences = [
        Sentence(
            id="s1",
            start_seconds=0.0,
            end_seconds=1.0,
            source_text="你好",
            target_text="Hello 你好",
            status="synthesized",
        ),
        Sentence(
            id="s2",
            start_seconds=2.0,
            end_seconds=3.0,
            source_text="一二三",
            target_text="Hello , world ! ... —",
            status="error",
        ),
    ]
    flags = check_translations(
        sentences,
        budgets={"s1": 5, "s2": 2},
        budget_key="max_words",
        source_language="zh",
        target_language="en",
    )
    codes = {(f.sentence_id, f.code) for f in flags}
    assert ("s1", "cjk_left") in codes
    assert ("s2", "too_long") not in codes


# --- P0-16: translation_prompts.py ------------------------------------------


def test_non_contiguous_context_for_chunk_stays_bounded_and_memory_prefers_nearest() -> None:
    sentences = [
        Sentence(
            id=f"s{i:03d}",
            start_seconds=float(i),
            end_seconds=float(i + 1),
            source_text=f"src{i}",
            target_text=f"zh{i}" if i % 2 == 0 else "",
            status="skipped_sound" if i == 4 else ("translated" if i % 2 == 0 else "pending"),
        )
        for i in range(100)
    ]
    chunk = TranslationChunk([sentences[5], sentences[95]])
    ctx = json.loads(_context_for_chunk(sentences, chunk, maximum_sentences=6))
    ctx_ids = [item["id"] for item in ctx]
    assert len(ctx_ids) <= 6
    assert len(ctx_ids) == len(set(ctx_ids))
    assert ctx_ids == sorted(ctx_ids)
    assert "s050" not in ctx_ids
    assert {"s005", "s095"}.issubset(ctx_ids)

    mem_chunk = TranslationChunk([sentences[10]])
    mem = json.loads(_bounded_translation_memory(sentences, mem_chunk, maximum_sentences=4))
    mem_ids = [item["id"] for item in mem]
    assert "s004" not in mem_ids
    assert mem_ids == ["s000", "s002", "s006", "s008"]


def test_romaji_explicit_empty_sound_respected() -> None:
    sentences = [
        Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="はぁっ"),
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        body = json.dumps({"translations": [{"id": "s1", "zh": None, "kind": "sound"}]})
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            sentences, api_key="k", model="m", client=client, sound_policy="romaji"
        )
    assert report.skipped == 1
    assert sentences[0].status == "skipped_sound"
    assert sentences[0].zh_text == ""
    assert sentences[0].enabled is False


# --- P0-17: fatal errors, <think>, providers, machine translation -----------


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_fatal_http_statuses_stop_without_bisection(status_code: int) -> None:
    sentences = [
        Sentence(id=f"s{i}", start_seconds=float(i), end_seconds=float(i + 1), source_text=f"文{i}")
        for i in range(8)
    ]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, text="bad config")

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(TranslationError, match=f"HTTP {status_code}"),
    ):
        translate_sentences(sentences, api_key="k", model="m", client=client)
    assert calls == 1


def test_invalid_extra_body_stops_without_retries_or_bisection() -> None:
    sentences = [
        Sentence(id=f"s{i}", start_seconds=float(i), end_seconds=float(i + 1), source_text=f"文{i}")
        for i in range(4)
    ]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(TranslationError),
    ):
        translate_sentences(
            sentences,
            api_key="k",
            model="m",
            provider="openai",
            base_url="https://example.invalid/v1",
            extra_body="{not-json",
            client=client,
        )
    assert calls == 0


def test_think_blocks_stripped_and_null_zh_coerced() -> None:
    content = (
        '<THINK>\nLet me plan {"draft": 123} carefully\n</think>\n'
        '```json\n{"translations":[{"id":"s1","zh":null,"kind":"sound"},'
        '{"id":"s2","zh":"你好","kind":"speech"}]}\n```'
    )
    entries = validate_translation_entries(content, ["s1", "s2"])
    assert entries["s1"].zh == "" and entries["s1"].kind == "sound"
    assert entries["s2"].zh == "你好" and entries["s2"].kind == "speech"


def test_openai_reasoning_models_and_response_format_downgrade_remembered() -> None:
    seen_payloads: list[dict] = []

    def reasoning_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_payloads.append(payload)
        body = json.dumps({"translations": [{"id": "s1", "zh": "Hi"}]})
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    for model in ("o3-mini", "o1", "o4-mini", "gpt-5-pro"):
        seen_payloads.clear()
        s = [Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="おはよう")]
        with httpx.Client(transport=httpx.MockTransport(reasoning_handler)) as client:
            translate_sentences(
                s,
                api_key="k",
                model=model,
                provider="openai",
                base_url="https://api.openai.com/v1",
                client=client,
            )
        assert "temperature" not in seen_payloads[0]
        assert "top_p" not in seen_payloads[0]

    compat_payloads: list[dict] = []

    def compat_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        compat_payloads.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, text="response_format unsupported")
        items_str = payload["messages"][-1]["content"].split("：\n", 1)[1].split("\n每项", 1)[0]
        ids = [item["id"] for item in json.loads(items_str)]
        body = json.dumps({"translations": [{"id": i, "zh": "好"} for i in ids]})
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(compat_handler)) as client:
        translator = LLMTranslator(
            provider="openai_compatible",
            api_key="",
            model="local",
            base_url="http://127.0.0.1:11434/v1",
            temperature=0.1,
            top_p=1.0,
            max_output_tokens=2048,
            client=client,
        )
        c1 = TranslationChunk([Sentence(id="s1", start_seconds=0, end_seconds=1, source_text="a")])
        c2 = TranslationChunk([Sentence(id="s2", start_seconds=1, end_seconds=2, source_text="b")])
        translator.translate_chunk(c1, "[]", "[]", "job")
        translator.translate_chunk(c2, "[]", "[]", "job")

    assert len(compat_payloads) == 3
    assert "response_format" in compat_payloads[0]
    assert "response_format" not in compat_payloads[1]
    assert "response_format" not in compat_payloads[2]


def test_anthropic_url_temp_top_p_and_extra_body() -> None:
    seen_requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"translations":[{"id":"s1","zh":"好"}]}'}],
            },
        )

    s = [Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="はい")]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            s,
            api_key="k",
            model="claude-sonnet-4-5",
            provider="anthropic",
            base_url="https://proxy.example/v1",
            extra_body='{"metadata":{"user_id":"u1"}}',
            client=client,
        )
    url, payload = seen_requests[0]
    assert url == "https://proxy.example/v1/messages"
    assert "temperature" in payload and "top_p" not in payload
    assert payload["metadata"] == {"user_id": "u1"}


def test_gemini_thought_filtering_extra_body_and_safety_block() -> None:
    seen_payloads: list[dict] = []

    def ok_handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": '{"wrong": "thought"}', "thought": True},
                                {"text": '{"translations":[{"id":"s1","zh":"好"}]}'},
                            ]
                        },
                    }
                ]
            },
        )

    s = [Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="はい")]
    with httpx.Client(transport=httpx.MockTransport(ok_handler)) as client:
        translate_sentences(
            s,
            api_key="k",
            model="gemini-2.5-flash",
            provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            extra_body='{"safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"}]}',
            client=client,
        )
    assert s[0].zh_text == "好"
    assert "safetySettings" in seen_payloads[0]

    calls = 0

    def blocked_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})

    s_multi = [
        Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="a"),
        Sentence(id="s2", start_seconds=1.0, end_seconds=2.0, source_text="b"),
    ]
    with (
        httpx.Client(transport=httpx.MockTransport(blocked_handler)) as client,
        pytest.raises(TranslationError, match="PROHIBITED_CONTENT"),
    ):
        translate_sentences(
            s_multi,
            api_key="k",
            model="gemini-2.5-flash",
            provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            client=client,
        )
    assert calls == 1


def test_machine_translation_retries_429_and_passes_empty_items_to_settle() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited")
        texts = json.loads(request.content)["text"]
        return httpx.Response(
            200,
            json={
                "translations": [{"text": "" if t == "き" else "你好。"} for t in texts],
            },
        )

    lines = [
        Sentence(id="s1", start_seconds=0.0, end_seconds=0.5, source_text="き"),
        Sentence(id="s2", start_seconds=1.0, end_seconds=2.0, source_text="こんにちは。"),
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines,
            api_key="k",
            model="quality_optimized",
            provider="deepl",
            base_url="https://api.deepl.com",
            client=client,
        )
    # 1st attempt 429 -> 2nd attempt returns ["", "你好。"] -> single-line retry on "き" -> 3 calls
    assert attempts == 3
    assert report.translated == 1 and report.skipped == 1
    assert lines[0].status == "skipped_sound"
    assert lines[1].zh_text == "你好。"


# --- P1: backup, revision, models, subtitles, pipeline_mix ------------------


def test_backup_rejects_mismatched_target_language(tmp_path: Path) -> None:
    sentences = [
        Sentence(id="s1", start_seconds=0.0, end_seconds=1.0, source_text="a", target_text="你好")
    ]
    project, directory = _make_project(tmp_path, sentences, tts_target_language="zh")
    project.translation_language = "zh"
    write_translation_backup(project, directory)

    project.settings.tts_target_language = "en"
    with pytest.raises(ProjectError, match="不一致"):
        restore_translation_backup(project, directory)


def test_revision_speaker_case_insensitive_kind_and_noop_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsed = validate_revisions(
        '{"revisions":[{"id":"s1","zh":"改后","kind":"Speech"}]}',
        ["s1"],
    )
    assert parsed["s1"].kind == "speech"

    sentences = [
        Sentence(
            id="s1",
            start_seconds=0.0,
            end_seconds=1.0,
            speaker_id="S1",
            source_text="原文1",
            target_text="译文1",
            status="translated",
        ),
        Sentence(
            id="s2",
            start_seconds=2.0,
            end_seconds=3.0,
            speaker_id="S2",
            source_text="原文2",
            target_text="译文2",
            status="translated",
        ),
    ]
    msgs = _revision_messages("sys", TranslationChunk(sentences), "[]", "改", "skip", 1)
    assert '"speaker":"S1"' in msgs[-1]["content"]

    project, directory = _make_project(tmp_path, sentences)
    save_calls = 0
    orig_save = pipeline.save_project

    def counting_save(proj, pdir):
        nonlocal save_calls
        save_calls += 1
        return orig_save(proj, pdir)

    monkeypatch.setattr("asmr_dubber.translation_revision.save_project", counting_save)
    monkeypatch.setattr("asmr_dubber.translation_revision.resolve_api_key", lambda *_a: "k")

    def handler(_request: httpx.Request) -> httpx.Response:
        body = '{"revisions":[]}'
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        revise_project_translations(project, directory, ["s1", "s2"], "保持不变", client=client)
    assert save_calls == 0


def test_sentence_model_copy_clears_error_code_and_mix_uses_posix_and_stereo_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Sentence(
        id="s1",
        start_seconds=0.0,
        end_seconds=1.0,
        source_text="a",
        target_text="你好",
        status="error",
        error="boom",
        error_code="tts_failed",
    )
    copied = s.model_copy(update={"error": None})
    assert copied.error is None
    assert copied.error_code is None

    _make_wav(tmp_path / "chinese" / "s1.wav", seconds=0.5, channels=1)
    project, directory = _make_project(
        tmp_path,
        [
            Sentence(
                id="s1",
                start_seconds=0.0,
                end_seconds=1.0,
                source_text="a",
                target_text="你好",
                tts_file="chinese/s1.wav",
                tts_duration_seconds=0.5,
                status="synthesized",
            )
        ],
        channels=2,
        duration=2.0,
        spatial_rtf_enabled=False,
    )
    project.sentences[0].tts_cache_key = tts_cache_key(project, project.sentences[0])
    save_project(project, directory)

    called_stereo_ref = False

    def fake_stereo_ref(_src: Path, pdir: Path) -> Path:
        nonlocal called_stereo_ref
        called_stereo_ref = True
        return _make_wav(pdir / "analysis" / "spatial-reference.wav", seconds=2.0, channels=2)

    monkeypatch.setattr("asmr_dubber.experimental_mix.stereo_reference", fake_stereo_ref)
    out = pipeline.mix_project(project, directory)
    assert called_stereo_ref is True
    assert out.is_file()
    assert project.output_file is not None and "\\" not in project.output_file
    assert project.chinese_stem_file is not None and "\\" not in project.chinese_stem_file


def test_subtitles_dubbing_past_source_english_wrap_cps_escape_warnings_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s_long = Sentence(
        id="s000001",
        start_seconds=1.5,
        end_seconds=2.0,
        speaker_id="S1",
        source_text="息<breath>",
        target_text="This is a reasonably wide English subtitle line that fits forty chars",
        tts_duration_seconds=2.0,
        status="synthesized",
    )
    srt, lrc = write_subtitle_files(
        [s_long],
        tmp_path / "sub_en",
        "bilingual",
        timeline="dubbing",
        maximum_chars=22,
        target_language="en",
        media_duration_seconds=2.0,
        chinese_dubbing_offset_ms=500,
    )
    srt_text = srt.read_text(encoding="utf-8")
    lrc_text = lrc.read_text(encoding="utf-8")
    assert "&lt;breath&gt;" in srt_text and "<breath>" not in srt_text
    zh_lines = [line for line in srt_text.splitlines() if "reasonably" in line or "forty" in line]
    assert len(zh_lines[0]) <= 44 and len(zh_lines) == 2
    assert len(lrc_text.splitlines()) == 2

    s_cps = Sentence(
        id="s1",
        start_seconds=0.0,
        end_seconds=1.0,
        source_text="一二三四五六七八九十",
        target_text="甲乙丙丁戊己庚辛壬癸",
    )
    srt_cps, _ = write_subtitle_files(
        [s_cps],
        tmp_path / "cps",
        "bilingual",
        minimum_duration=0.5,
        maximum_cps=10.0,
    )
    assert "00:00:00,000 --> 00:00:01,000" in srt_cps.read_text(encoding="utf-8")

    _SPEECH_DURATION_CACHE.clear()
    _make_wav(tmp_path / "proj" / "chinese" / "s1.wav", seconds=1.0)
    row = Sentence(
        id="s000001",
        start_seconds=0.0,
        end_seconds=1.0,
        speaker_id="SPK1",
        source_text="原文",
        target_text="译文",
        tts_file="chinese/s1.wav",
        tts_duration_seconds=1.0,
        status="synthesized",
    )
    row_missing = Sentence(
        id="s000002",
        start_seconds=1.5,
        end_seconds=2.5,
        speaker_id="SPK2",
        source_text="原文二",
        target_text="",
        status="pending",
    )
    project, directory = _make_project(
        tmp_path / "proj",
        [row, row_missing],
        separation_enabled=True,
        separation_mix_mode="replace",
        subtitle_timeline="source",
    )
    row.tts_cache_key = "stale"
    assert _replacement_dubbing_durations(project, directory) == {}

    row.tts_cache_key = tts_cache_key(project, row)
    save_project(project, directory)

    bounds_calls = 0
    orig_bounds = pipeline.speech_bounds

    def counting_bounds(p: Path) -> tuple[float, float]:
        nonlocal bounds_calls
        bounds_calls += 1
        return orig_bounds(p)

    monkeypatch.setattr(pipeline, "speech_bounds", counting_bounds)
    pipeline.export_transcript(project, directory)
    pipeline.export_transcript(project, directory)
    assert bounds_calls == 1

    exported = json.loads((directory / "exports" / "transcript.json").read_text(encoding="utf-8"))
    assert exported[0]["speaker_id"] == "SPK1"
    with (directory / "exports" / "transcript.csv").open(encoding="utf-8-sig") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["speaker_id"] == "SPK1"

    progress_msgs: list[str] = []
    pipeline.generate_subtitles(
        project,
        directory,
        language="bilingual",
        progress=lambda msg, _c, _t: progress_msgs.append(msg),
    )
    assert "字幕生成完成" in progress_msgs[-1]
    assert "回退原文" in progress_msgs[-1]
