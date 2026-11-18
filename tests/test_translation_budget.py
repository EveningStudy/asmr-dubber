import json

import httpx
import pytest

from asmr_dubber import translation as translation_module
from asmr_dubber.models import Sentence
from asmr_dubber.translation import length_budgets, translate_sentences


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(translation_module.time, "sleep", lambda _seconds: None)


def _line(index: int, start: float, end: float) -> Sentence:
    return Sentence(
        id=f"s{index:06d}", start_seconds=start, end_seconds=end, source_text=f"台詞{index}です。"
    )


def test_budget_uses_the_pause_before_the_next_line_up_to_one_second() -> None:
    lines = [_line(1, 0.0, 1.0), _line(2, 1.5, 2.0), _line(3, 10.0, 12.0)]
    budgets = length_budgets(lines, lines, 4.5)
    assert budgets["s000001"] == round(1.5 * 4.5)  # runs into the 0.5 s pause
    assert budgets["s000002"] == round(1.5 * 4.5)  # long pause: capped at end + 1 s
    assert budgets["s000003"] == round(3.0 * 4.5)  # last line: end + 1 s
    assert length_budgets([_line(4, 0.0, 0.1)], [_line(4, 0.0, 0.1)], 4.5)["s000004"] == 5


def _request_items(request: httpx.Request) -> tuple[list[dict], str]:
    content = json.loads(request.content)["messages"][-1]["content"]
    # The output schema is also a JSON array; the target items are the one with "source".
    start = content.rindex("[{", 0, content.index('"source"'))
    end = content.index("}]", start) + 2
    return json.loads(content[start:end]), content


def _answer(items: list[dict]) -> httpx.Response:
    rows = [{"id": item["id"], "zh": "好", "kind": "speech"} for item in items]
    content = json.dumps({"translations": rows}, ensure_ascii=False)
    return httpx.Response(
        200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    )


@pytest.mark.parametrize(
    ("provider", "target", "key"), [("deepseek", "zh", "max_chars"), ("openai", "en", "max_words")]
)
def test_language_models_receive_a_length_budget(provider, target, key) -> None:
    lines = [_line(1, 0.0, 2.0), _line(2, 2.5, 3.0)]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items, content = _request_items(request)
        assert all(key in item for item in items)
        seen.append(content)
        return _answer(items)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            lines,
            api_key="test-key",
            model="m",
            provider=provider,
            **({"base_url": "https://example.invalid/v1"} if provider != "deepseek" else {}),
            target_language=target,
            client=client,
            length_budget_rate=4.5,
        )
    assert seen and all(key in content and "上限" in content for content in seen)


def test_no_budget_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        items, content = _request_items(request)
        assert all("max_chars" not in item for item in items)
        assert "上限，用于让配音" not in content
        return _answer(items)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences([_line(1, 0.0, 2.0)], api_key="k", model="m", client=client)


def test_auto_tightening_runs_once_and_clears_too_long_flag() -> None:
    lines = [_line(1, 0.0, 1.0)]
    calls: list[tuple[list[dict], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items, content = _request_items(request)
        calls.append((items, content))
        zh = "这是一段明显超出预算的译文内容" if len(calls) == 1 else "简短译文"
        body = json.dumps(
            {"translations": [{"id": items[0]["id"], "zh": zh, "kind": "speech"}]},
            ensure_ascii=False,
        )
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines, api_key="k", model="m", client=client, length_budget_rate=4.5
        )
    assert len(calls) == 2
    assert "existing_zh" not in calls[0][0][0]
    assert calls[1][0][0]["existing_zh"] == "这是一段明显超出预算的译文内容"
    assert "existing_zh" in calls[1][1]
    assert report.translated == 1
    assert report.flagged == []
    assert lines[0].zh_text == "简短译文"


def test_auto_tightening_preserves_existing_text_on_error() -> None:
    lines = [_line(1, 0.0, 1.0)]
    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        items, _ = _request_items(request)
        if step == 1:
            body = json.dumps(
                {
                    "translations": [
                        {"id": items[0]["id"], "zh": "这句还是太长超过预算好多字", "kind": "speech"}
                    ]
                },
                ensure_ascii=False,
            )
            return httpx.Response(
                200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
            )
        return httpx.Response(500, text="upstream error")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines, api_key="k", model="m", client=client, length_budget_rate=4.5
        )
    assert lines[0].zh_text == "这句还是太长超过预算好多字"
    assert lines[0].status == "translated"
    assert report.failed == []
    assert [(flag.sentence_id, flag.code) for flag in report.flagged] == [("s000001", "too_long")]


def test_explicit_compress_targets_only_over_budget_and_wires_contract_ui(tmp_path) -> None:
    from importlib.resources import files

    from asmr_dubber.api_contract import TaskRequest

    lines = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=2.0,
            source_text="原文1",
            zh_text="短句",
            status="translated",
        ),
        Sentence(
            id="s000002",
            start_seconds=2.5,
            end_seconds=3.5,
            source_text="原文2",
            zh_text="这句译文明显比预留时间允许的字数要多出不少",
            status="translated",
        ),
    ]
    seen_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items, _ = _request_items(request)
        seen_ids.extend(item["id"] for item in items)
        rows = [{"id": item["id"], "zh": "简短点", "kind": "speech"} for item in items]
        body = json.dumps({"translations": rows}, ensure_ascii=False)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines, api_key="k", model="m", client=client, length_budget_rate=4.5, compress=True
        )
    assert seen_ids == ["s000002"]
    assert lines[0].zh_text == "短句"
    assert lines[1].zh_text == "简短点"
    assert report.translated == 1 and report.flagged == []

    req = TaskRequest(kind="translate", project=str(tmp_path / "p.json"), revision=1, compress=True)
    assert req.compress is True
    frontend = files("asmr_dubber").joinpath("frontend")
    assert "compressTooLong" in frontend.joinpath("project_translation.js").read_text(
        encoding="utf-8"
    )
    assert "data-compress" in frontend.joinpath("project_review.js").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("provider", "target", "key"), [("deepseek", "zh", "max_chars"), ("openai", "en", "max_words")]
)
def test_split_retry_preserves_length_budgets_and_rule(
    provider: str, target: str, key: str
) -> None:
    lines = [_line(1, 0.0, 1.0), _line(2, 1.4, 2.2), _line(3, 3.0, 4.5), _line(4, 5.0, 6.0)]
    expected = length_budgets(lines, lines, 4.5)
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items, content = _request_items(request)
        sizes.append(len(items))
        assert all(item.get(key) == expected[item["id"]] for item in items)
        assert key in content and "上限，用于让配音和原声对齐" in content
        if len(items) > 1:
            return httpx.Response(500, text="transient failure")
        return _answer(items)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines,
            api_key="test-key",
            model="m",
            provider=provider,
            **({"base_url": "https://example.invalid/v1"} if provider != "deepseek" else {}),
            target_language=target,
            client=client,
            length_budget_rate=4.5,
        )
    assert 4 in sizes and 2 in sizes and 1 in sizes
    assert report.translated == 4 and report.failed == []


@pytest.mark.parametrize(
    ("provider", "target", "key"), [("deepseek", "zh", "max_chars"), ("openai", "en", "max_words")]
)
def test_blank_semantic_single_retry_preserves_length_budget_and_rule(
    provider: str, target: str, key: str
) -> None:
    lines = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.5,
            source_text="これは大切な長い文章なので最後まで翻訳してください。",
        ),
        Sentence(
            id="s000002",
            start_seconds=2.0,
            end_seconds=3.5,
            source_text="次の台詞もきちんと翻訳してください。",
        ),
    ]
    expected = length_budgets(lines, lines, 4.5)
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        items, content = _request_items(request)
        ids = [item["id"] for item in items]
        calls.append(ids)
        assert all(item.get(key) == expected[item["id"]] for item in items)
        assert key in content and "上限，用于让配音和原声对齐" in content
        if len(calls) == 1:
            rows = [
                {"id": "s000001", "zh": "", "kind": "speech"},
                {"id": "s000002", "zh": "ok", "kind": "speech"},
            ]
        else:
            rows = [{"id": "s000001", "zh": "ok", "kind": "speech"}]
        body = json.dumps({"translations": rows}, ensure_ascii=False)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines,
            api_key="test-key",
            model="m",
            provider=provider,
            **({"base_url": "https://example.invalid/v1"} if provider != "deepseek" else {}),
            target_language=target,
            client=client,
            length_budget_rate=4.5,
        )
    assert calls == [["s000001", "s000002"], ["s000001"]]
    assert report.translated == 2 and report.failed == []


def test_multi_batch_compress_keeps_context_and_exposes_shortened_earlier_batches() -> None:
    lines = [
        Sentence(
            id=f"s{index:06d}",
            start_seconds=index * 2.0,
            end_seconds=index * 2.0 + 1.0,
            source_text=f"原文{index}です。",
            zh_text=f"初版超长译文{index}需要精简字数才能配得上原声时长",
            status="translated",
        )
        for index in range(1, 19)
    ]
    batch_messages: list[tuple[list[str], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        joined = "\n".join(msg["content"] for msg in payload["messages"])
        items, _ = _request_items(request)
        ids = [item["id"] for item in items]
        batch_messages.append((ids, joined))
        rows = [{"id": item["id"], "zh": f"短译{item['id']}", "kind": "speech"} for item in items]
        body = json.dumps({"translations": rows}, ensure_ascii=False)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = translate_sentences(
            lines,
            api_key="k",
            model="m",
            client=client,
            length_budget_rate=4.5,
            compress=True,
            send_context=True,
            context_sentences=24,
            memory_sentences=50,
        )
    assert [len(ids) for ids, _ in batch_messages] == [16, 2]
    assert "初版超长译文17需要精简字数才能配得上原声时长" in batch_messages[0][1]
    assert "短译s000001" in batch_messages[1][1]
    assert "短译s000016" in batch_messages[1][1]
    assert report.translated == 18
