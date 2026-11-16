"""Response tolerance (D3, D4) and the post-translation self-check (D5)."""

import json

import httpx
import pytest

from asmr_dubber import translation as translation_module
from asmr_dubber.errors import TranslationError
from asmr_dubber.models import Sentence
from asmr_dubber.translation import (
    _extract_json,
    check_translations,
    translate_sentences,
    validate_translation_entries,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(translation_module.time, "sleep", lambda _seconds: None)


def _sentence(index: int, source: str, zh: str = "", status: str = "translated") -> Sentence:
    return Sentence(
        id=f"s{index:06d}",
        start_seconds=float(index),
        end_seconds=float(index) + 1.0,
        source_text=source,
        zh_text=zh,
        status=status,
    )


def _body(rows: list[dict]) -> str:
    return json.dumps({"translations": rows}, ensure_ascii=False)


# D3 -------------------------------------------------------------------------


def test_same_ids_in_another_order_are_matched_by_id() -> None:
    content = _body([{"id": "b", "zh": "乙"}, {"id": "a", "zh": "甲"}, {"id": "c", "zh": "丙"}])
    entries = validate_translation_entries(content, ["a", "b", "c"])
    assert {key: entry.zh for key, entry in entries.items()} == {"a": "甲", "b": "乙", "c": "丙"}


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "a", "zh": "甲"}],
        [{"id": "a", "zh": "甲"}, {"id": "b", "zh": "乙"}, {"id": "x", "zh": "多"}],
        [{"id": "a", "zh": "甲"}, {"id": "a", "zh": "甲"}],
        [{"id": "a", "zh": "甲"}, {"id": "x", "zh": "乙"}],
    ],
)
def test_missing_extra_or_duplicate_ids_are_still_rejected(rows) -> None:
    with pytest.raises(TranslationError, match="id 不一致"):
        validate_translation_entries(_body(rows), ["a", "b"])


def test_reordered_response_does_not_trigger_a_retry() -> None:
    sentences = [_sentence(index, f"台詞{index}です。", status="pending") for index in range(1, 4)]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        text = json.loads(request.content)["messages"][-1]["content"].split("目标项：")[1]
        ids = [item["id"] for item in json.loads(text.strip())]
        rows = [{"id": item, "zh": f"译文{item}"} for item in reversed(ids)]
        content = _body(rows)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            sentences, api_key="k", model="deepseek-v4-flash", client=client
        )
    assert calls == 1
    assert report.translated == 3
    assert [s.zh_text for s in sentences] == [f"译文{s.id}" for s in sentences]


# D4 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        '好的，结果如下：\n{"translations": [{"id": "a", "zh": "甲"}]}\n希望有帮助。',
        '{"translations": [{"id": "a", "zh": "甲"},],}',
        '```json\n{"translations": [{"id": "a", "zh": "甲",}]}\n```',
        '说明 {"translations": [{"id": "a", "zh": "甲"},]} 完',
    ],
)
def test_extract_json_repairs_wrapping_and_trailing_commas(content: str) -> None:
    entries = validate_translation_entries(content, ["a"])
    assert entries["a"].zh == "甲"


def test_trailing_comma_inside_a_string_is_kept() -> None:
    parsed = _extract_json('x {"translations": [{"id": "a", "zh": "好,]",},]} y')
    assert parsed["translations"][0]["zh"] == "好,]"


@pytest.mark.parametrize("content", ["没有对象", '{"translations": [', "[1, 2]", ""])
def test_unrepairable_content_still_fails(content: str) -> None:
    with pytest.raises(TranslationError):
        _extract_json(content)


def test_repair_does_not_accept_missing_or_extra_ids() -> None:
    content = '前言 {"translations": [{"id": "a", "zh": "甲"},]}'
    with pytest.raises(TranslationError, match="id 不一致"):
        validate_translation_entries(content, ["a", "b"])


# D5 -------------------------------------------------------------------------


def _codes(flags) -> set[tuple[str, str]]:
    return {(flag.sentence_id, flag.code) for flag in flags}


def test_leftover_kana_is_flagged_but_prolonged_mark_and_dot_are_not() -> None:
    sentences = [
        _sentence(1, "ありがとう", "谢谢ありがとう"),
        _sentence(2, "うん", "嗯ー・好"),
    ]
    assert _codes(check_translations(sentences)) == {("s000001", "kana_left")}


def test_length_over_budget_by_more_than_twenty_percent_is_flagged() -> None:
    sentences = [
        _sentence(1, "原文一", "一二三四五六七八九十"),
        _sentence(2, "原文二", "一二三四五六七八九十一二"),
        _sentence(3, "原文三", "一二三四五六七八九十一二三"),
    ]
    budgets = {"s000001": 10, "s000002": 10, "s000003": 10}
    assert _codes(check_translations(sentences, budgets)) == {("s000003", "too_long")}
    assert check_translations(sentences) == []


def test_english_budget_counts_words() -> None:
    sentences = [_sentence(1, "原文", "one two three four five six seven")]
    flags = check_translations(sentences, {"s000001": 5}, "max_words")
    assert _codes(flags) == {("s000001", "too_long")}


def test_same_as_source_and_repeat_of_previous_line() -> None:
    sentences = [
        _sentence(1, "Hello there", "Hello there"),
        _sentence(2, "今日は寒いね", "今天很冷"),
        _sentence(3, "外は雪だよ", "今天很冷"),
        _sentence(4, "外は雪だよ", "今天很冷"),
    ]
    assert _codes(check_translations(sentences)) == {
        ("s000001", "same_as_source"),
        ("s000003", "repeats_previous"),
    }


def test_repeated_source_with_repeated_translation_is_ordinary_speech() -> None:
    sentences = [_sentence(1, "すきすき", "喜欢喜欢"), _sentence(2, "すきすき", "喜欢喜欢")]
    assert check_translations(sentences) == []


def test_only_requested_translated_enabled_rows_are_checked() -> None:
    sentences = [
        _sentence(1, "a", "かな"),
        _sentence(2, "b", "かな", status="skipped_sound"),
        _sentence(3, "c", "かな"),
    ]
    sentences[2].enabled = False
    flags = check_translations(sentences, only_ids=["s000001", "s000002", "s000003"])
    assert _codes(flags) == {("s000001", "kana_left")}
    assert check_translations(sentences, only_ids=[]) == []


def test_translate_sentences_reports_flags_and_can_redo_only_those() -> None:
    sentences = [_sentence(index, f"台詞{index}です。", status="pending") for index in range(1, 4)]
    answers = {"s000001": "你好", "s000002": "还有ありがとう", "s000003": "再见"}
    requested: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][-1]["content"].split("目标项：")[1]
        ids = [item["id"] for item in json.loads(text.strip())]
        requested.append(ids)
        rows = [{"id": item, "zh": answers[item]} for item in ids]
        content = _body(rows)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            sentences, api_key="k", model="deepseek-v4-flash", client=client
        )
        assert report.flagged_ids == ["s000002"]
        assert [(f.sentence_id, f.code) for f in report.flagged] == [("s000002", "kana_left")]
        # Sentences are not touched by the check.
        assert sentences[1].error is None and sentences[1].script_review_note is None

        answers["s000002"] = "还有谢谢"
        again = translate_sentences(
            sentences,
            api_key="k",
            model="deepseek-v4-flash",
            client=client,
            sentence_ids=report.flagged_ids,
        )
    assert requested[-1] == ["s000002"]
    assert again.translated == 1 and again.flagged == []
    assert [s.zh_text for s in sentences] == ["你好", "还有谢谢", "再见"]
