"""One bad sentence must never stop the translation of the others (roadmap A1-A4)."""

import json
from types import SimpleNamespace

import httpx
import pytest

from asmr_dubber import translation as translation_module
from asmr_dubber.errors import TranslationError
from asmr_dubber.filtering import is_negligible_sound_segment, kana_to_romaji
from asmr_dubber.models import Sentence
from asmr_dubber.translation import (
    default_translation_prompt,
    translate_sentences,
    validate_translation_entries,
)
from asmr_dubber.translation_backup import restore_translation_backup, write_translation_backup

LONG_LINE = "これは大切な長い文章なので最後まで翻訳してください。"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(translation_module.time, "sleep", lambda _seconds: None)


def _sentences(count: int, special: dict[int, str] | None = None) -> list[Sentence]:
    special = special or {}
    return [
        Sentence(
            id=f"s{index:06d}",
            start_seconds=float(index),
            end_seconds=float(index) + 1.0,
            source_text=special.get(index, f"台詞{index}です。"),
        )
        for index in range(1, count + 1)
    ]


def _targets(request: httpx.Request) -> list[dict]:
    payload = json.loads(request.content)
    text = payload["messages"][-1]["content"].split("目标项：", maxsplit=1)[1]
    return json.loads(text.strip())


def _ok(rows: list[dict]) -> httpx.Response:
    content = json.dumps({"translations": rows}, ensure_ascii=False)
    return httpx.Response(
        200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    )


def _run(sentences, handler, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return translate_sentences(
            sentences, api_key="test-key", model="deepseek-v4-flash", client=client, **kwargs
        )


def _echo(items: list[dict], blank: set[str] = frozenset(), kind: str | None = None):
    rows = []
    for item in items:
        row = {"id": item["id"], "zh": "" if item["id"] in blank else f"译文{item['id']}"}
        if kind and item["id"] in blank:
            row["kind"] = kind
        rows.append(row)
    return _ok(rows)


def test_single_kana_sound_does_not_stop_the_other_fifty() -> None:
    sentences = _sentences(51, {12: "き"})
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        calls.append([item["id"] for item in items])
        return _echo(items, blank={"s000012"})

    report = _run(sentences, handler)

    translated = [s for s in sentences if s.zh_text]
    assert len(translated) == 50
    sound = sentences[11]
    assert sound.id == "s000012"
    assert sound.status == "skipped_sound" and sound.enabled is False and not sound.zh_text
    assert report.failed == [] and report.translated == 50 and report.skipped_ids == ["s000012"]


@pytest.mark.parametrize(
    ("policy", "expected_text", "expected_enabled"),
    [("skip", "", False), ("romaji", "ki", True), ("keep_source", "き", True)],
)
def test_sound_policy_decides_what_a_lone_kana_becomes(
    policy: str, expected_text: str, expected_enabled: bool
) -> None:
    sentences = _sentences(3, {2: "き"})

    def handler(request: httpx.Request) -> httpx.Response:
        return _echo(_targets(request), blank={"s000002"})

    report = _run(sentences, handler, sound_policy=policy)

    sound = sentences[1]
    assert report.failed == []
    assert sound.status == "skipped_sound"
    assert sound.zh_text == expected_text
    assert sound.enabled is expected_enabled
    assert sound.chinese_audio_enabled is (not expected_text)  # never voiced when it has text
    assert sentences[0].zh_text and sentences[2].zh_text


def test_model_marked_sound_follows_policy_even_for_long_sources() -> None:
    sentences = _sentences(2, {1: "ちゅっ、ちゅぱっ"})

    def handler(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        return _ok(
            [
                {"id": items[0]["id"], "zh": "chu, chupa", "kind": "sound"},
                {"id": items[1]["id"], "zh": "你好。", "kind": "speech"},
            ]
        )

    _run(sentences, handler, sound_policy="romaji")
    assert sentences[0].zh_text == "chu, chupa" and sentences[0].status == "skipped_sound"
    assert sentences[0].chinese_audio_enabled is False
    assert sentences[1].zh_text == "你好。" and sentences[1].status == "translated"


def test_blank_real_line_is_a_failure_that_does_not_stop_the_batch() -> None:
    sentences = _sentences(4, {2: LONG_LINE})

    def handler(request: httpx.Request) -> httpx.Response:
        return _echo(_targets(request), blank={"s000002"})

    report = _run(sentences, handler)

    assert report.failed_ids == ["s000002"]
    assert sentences[1].status == "translation_failed" and sentences[1].enabled
    assert sentences[1].error
    assert all(s.zh_text for s in sentences if s.id != "s000002")
    # A later plain run retries exactly this sentence.
    seen: list[list[str]] = []

    def second(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        seen.append([item["id"] for item in items])
        return _echo(items)

    assert _run(sentences, second).failed == []
    assert seen == [["s000002"]]
    assert sentences[1].status == "translated" and sentences[1].error is None


def test_failed_batch_is_split_in_half_until_the_bad_sentences_are_isolated() -> None:
    sentences = _sentences(16)
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        sizes.append(len(items))
        if len(items) > 4 or any(item["id"] == "s000003" for item in items):
            return httpx.Response(500, text="boom")
        return _echo(items)

    report = _run(sentences, handler)

    assert report.failed_ids == ["s000003"]
    assert [s.id for s in sentences if not s.zh_text] == ["s000003"]
    assert 16 in sizes and 8 in sizes and 1 in sizes


def test_middle_batch_failing_does_not_block_batches_one_three_and_four() -> None:
    sentences = _sentences(64)
    bad = {f"s{index:06d}" for index in range(17, 33)}

    def handler(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        if any(item["id"] in bad for item in items):
            return httpx.Response(500, text="boom")
        return _echo(items)

    report = _run(sentences, handler)

    assert set(report.failed_ids) == bad
    assert all(s.zh_text for s in sentences if s.id not in bad)
    assert all(not s.zh_text and s.status == "translation_failed" for s in sentences[16:32])


def test_three_consecutive_failed_batches_stop_and_keep_finished_work() -> None:
    sentences = _sentences(96)
    checkpoints: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        if items[0]["id"] >= "s000017":
            return httpx.Response(500, text="down")
        return _echo(items)

    with pytest.raises(TranslationError, match="连续 3 批"):
        _run(
            sentences,
            handler,
            on_batch=lambda: checkpoints.append(sum(bool(s.zh_text) for s in sentences)),
        )

    assert all(s.zh_text for s in sentences[:16])
    assert not any(s.zh_text for s in sentences[16:])
    # The fifth and sixth batches were never attempted.
    assert all(s.status == "pending" for s in sentences[64:])
    assert checkpoints and max(checkpoints) == 16


def test_authentication_error_stops_immediately_after_saving_earlier_batches() -> None:
    sentences = _sentences(40)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        items = _targets(request)
        if items[0]["id"] == "s000017":
            return httpx.Response(401, text="invalid key")
        return _echo(items)

    with pytest.raises(TranslationError, match="HTTP 401"):
        _run(sentences, handler)

    assert calls == 2
    assert all(s.zh_text for s in sentences[:16])
    assert not any(s.zh_text for s in sentences[16:])


def test_retranslation_failure_keeps_every_old_translation() -> None:
    sentences = _sentences(20)
    for sentence in sentences:
        sentence.zh_text = f"旧译文{sentence.id}"
        sentence.status = "translated"

    def handler(request: httpx.Request) -> httpx.Response:
        items = _targets(request)
        if items[0]["id"] == "s000001":
            return _echo(items)
        return httpx.Response(500, text="down")

    report = _run(sentences, handler, retranslate=True)

    assert all(s.zh_text == f"译文{s.id}" for s in sentences[:16])
    assert all(s.zh_text == f"旧译文{s.id}" for s in sentences[16:])
    assert set(report.failed_ids) == {s.id for s in sentences[16:]}
    # The old text is still there and the sentence can be retried on its own.
    assert all(s.status == "translation_failed" for s in sentences[16:])


def test_retranslation_skips_hand_edited_sentences_and_hides_old_text_from_the_model() -> None:
    sentences = _sentences(3)
    for sentence in sentences:
        sentence.zh_text = f"旧译文{sentence.id}"
        sentence.status = "translated"
    sentences[1].review_locked = True
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(request.content.decode("utf-8"))
        return _echo(_targets(request))

    report = _run(sentences, handler, retranslate=True)

    assert sentences[1].zh_text == "旧译文s000002"
    assert sentences[0].zh_text == "译文s000001" and sentences[2].zh_text == "译文s000003"
    assert report.translated == 2
    assert "旧译文s000001" not in prompts[0] and "旧译文s000003" not in prompts[0]

    _run(sentences, handler, retranslate=True, overwrite_edited=True)
    assert sentences[1].zh_text == "译文s000002"


def test_backup_round_trip_restores_text_and_flags() -> None:
    sentences = _sentences(2)
    sentences[0].zh_text, sentences[0].review_locked = "手改的译文", True
    project = SimpleNamespace(
        sentences=sentences,
        revision=7,
        translation_language="zh",
        chinese_stem_file=None,
        output_file=None,
        output_video_file=None,
        subtitle_srt_file=None,
        subtitle_lrc_file=None,
        subtitle_video_file=None,
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = write_translation_backup(project, Path(directory))
        assert path is not None and path.name == "translations-backup-r7.json"
        sentences[0].zh_text, sentences[0].review_locked = "新译文", False
        assert restore_translation_backup(project, Path(directory)) == 1
    assert sentences[0].zh_text == "手改的译文" and sentences[0].review_locked is True


def test_kind_is_optional_and_unknown_values_are_ignored() -> None:
    content = json.dumps(
        {
            "translations": [
                {"id": "a", "zh": "好。", "kind": "speech"},
                {"id": "b", "zh": "", "kind": "SOUND"},
                {"id": "c", "zh": "嗯", "kind": "whatever"},
                {"id": "d", "zh": "是。"},
            ]
        },
        ensure_ascii=False,
    )
    entries = validate_translation_entries(content, ["a", "b", "c", "d"])
    assert [entries[key].kind for key in "abcd"] == ["speech", "sound", None, None]


def test_prompt_and_validation_agree_on_each_policy() -> None:
    skip = default_translation_prompt("ja", sound_policy="skip")
    romaji = default_translation_prompt("ja", sound_policy="romaji")
    keep = default_translation_prompt("ja", sound_policy="keep_source")
    assert "必须是空字符串" in skip and "罗马字" in romaji and "原样抄写" in keep
    assert all('"kind"' in prompt or "`kind`" in prompt for prompt in (skip, romaji, keep))
    assert "无实义时为空字符串" not in skip


def test_tiny_fragments_are_negligible_and_kana_converts_to_hepburn() -> None:
    assert is_negligible_sound_segment("き", 0.08)
    assert not is_negligible_sound_segment("き", 0.4)
    assert not is_negligible_sound_segment("ありがとう", 0.1)
    assert kana_to_romaji("き") == "ki"
    assert kana_to_romaji("ちょっと") == "chotto"
    assert kana_to_romaji("ラーメン") == "raamen"
