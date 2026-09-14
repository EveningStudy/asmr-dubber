import json

import httpx
import pytest

from asmr_dubber import translation as tr
from asmr_dubber.errors import TranslationError
from asmr_dubber.models import Sentence


def sentences(count):
    return [
        Sentence(id=f"s{i:06d}", start_seconds=i, end_seconds=i + 1, source_text="認識")
        for i in range(1, count + 1)
    ]


def response(items):
    return json.dumps({"corrections": items}, ensure_ascii=False)


def item(sid, start, end, key="p000001"):
    return {
        "id": sid,
        "text": "模型文字不可信",
        "script_spans": [{"id": key, "start": start, "end": end}],
    }


def test_one_script_across_two_asr_sentences():
    original = "こちらに座って、少し待っていてね。"
    recognized = sentences(2)
    content = response([item("s000001", 0, 8), item("s000002", 8, len(original))])
    result, ends, spans = tr._validated_script_spans(
        content, recognized, [("p000001", original)], {}
    )
    assert "".join(result.values()) == original
    assert ends == {"p000001": len(original)}
    assert len(spans) == 2
    assert recognized[0].start_seconds == 1


def test_legacy_partial_text_is_safely_partitioned():
    content = response(
        [
            {"id": "s000001", "script_ids": ["p000001"], "text": "はい、"},
            {"id": "s000002", "script_ids": ["p000001"], "text": "どうぞ。"},
        ]
    )
    result, _, _ = tr._validated_script_spans(
        content, sentences(2), [("p000001", "はい、どうぞ。")], {}
    )
    assert list(result.values()) == ["はい、", "どうぞ。"]


def test_empty_legacy_correction_does_not_consume_whole_script():
    content = response(
        [
            {"id": "s000001", "script_ids": ["p000001"], "text": ""},
            {"id": "s000002", "script_ids": ["p000001"], "text": "はい。"},
        ]
    )
    result, ends, _ = tr._validated_script_spans(content, sentences(2), [("p000001", "はい。")], {})
    assert result == {"s000001": "", "s000002": "はい。"}
    assert ends == {"p000001": 3}


def test_adjacent_ranges_do_not_insert_spaces_into_english_words():
    content = response(
        [
            {
                "id": "s000001",
                "text": "ignored",
                "script_spans": [
                    {"id": "p000001", "start": 0, "end": 2},
                    {"id": "p000001", "start": 2, "end": 5},
                ],
            }
        ]
    )
    result, _, _ = tr._validated_script_spans(content, sentences(1), [("p000001", "hello")], {})
    assert result == {"s000001": "hello"}


@pytest.mark.parametrize("start,end", [(0, 4), (4, 99), (5, 7), (4, 4)])
def test_overlap_holes_invalid_ranges_rejected(start, end):
    with pytest.raises(TranslationError):
        tr._validated_script_spans(
            response([item("s000001", start, end)]),
            sentences(1),
            [("p000001", "はい、どうぞ。")],
            {"p000001": 4},
        )


def test_distinct_repeated_lines_not_deduplicated():
    data = [("p000001", "はい。"), ("p000002", "はい。")]
    result, _, _ = tr._validated_script_spans(
        response([item("s000001", 0, 3), item("s000002", 0, 3, "p000002")]), sentences(2), data, {}
    )
    assert list(result.values()) == ["はい。", "はい。"]


def test_cannot_advance_after_partial_line():
    data = [("p000001", "はい、どうぞ。"), ("p000002", "次。")]
    with pytest.raises(TranslationError, match="尚有未分配"):
        tr._validated_script_spans(
            response([item("s000001", 0, 3), item("s000002", 0, 2, "p000002")]),
            sentences(2),
            data,
            {},
        )
    with pytest.raises(TranslationError, match="尚有未分配"):
        tr._validated_script_spans(
            response([item("s000001", 0, 2, "p000002")]), sentences(1), data, {"p000001": 3}
        )


def test_cross_batch_remainder_and_targeted_retry(monkeypatch):
    monkeypatch.setattr(tr.time, "sleep", lambda _: None)
    calls = []

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        calls.append(prompt)
        if len(calls) == 1:
            items = [item(f"s{i:06d}", i - 1, i) for i in range(1, 9)]
        elif len(calls) == 2:
            assert '"available_start":8' in prompt
            items = [item("s000009", 0, 10)]
        else:
            assert "被重复分配" in prompt and "第 2 次" in prompt
            items = [item("s000009", 8, 10)]
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": response(items)}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result, _ = tr.reconcile_script_sentences(
            sentences(9),
            ["あいうえおかきくけこ"],
            source_language="ja",
            target="source",
            provider="deepseek",
            model="test",
            api_key="test",
            client=client,
        )
    assert "".join(result.values()) == "あいうえおかきくけこ"
    assert len(calls) == 3


def test_failure_report_has_text_locations_and_conflict_times(tmp_path, monkeypatch):
    monkeypatch.setattr(tr.time, "sleep", lambda _: None)

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": response([item("s000001", 0, 3), item("s000002", 0, 3)])
                        },
                    }
                ]
            },
        )

    diagnostic = tmp_path / "imports" / "script-error.json"
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(TranslationError, match="导入后第 1 行"),
    ):
        tr.reconcile_script_sentences(
            sentences(2),
            ["はい。"],
            source_language="ja",
            target="source",
            provider="deepseek",
            model="test",
            api_key="test",
            client=client,
            diagnostic_path=diagnostic,
        )
    report = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert report["script"][0]["text"] == "はい。"
    assert "s000001" in report["error"] and "s000002" in report["error"]
    assert report["recognized"][1]["start_seconds"] == 2
