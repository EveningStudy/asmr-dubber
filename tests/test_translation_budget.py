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
