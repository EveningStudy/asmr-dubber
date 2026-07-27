import json
import re

import httpx
import pytest

from asmr_dubber.errors import TranslationError
from asmr_dubber.models import Sentence
from asmr_dubber.translation import translate_sentences, validate_translation


def test_validates_exact_translation_ids_and_order() -> None:
    content = json.dumps(
        {
            "translations": [
                {"id": "s000001", "zh": "让我们开始吧。"},
                {"id": "s000002", "zh": "接下来。"},
            ]
        },
        ensure_ascii=False,
    )
    result = validate_translation(content, ["s000001", "s000002"])
    assert result["s000001"] == "让我们开始吧。"


def test_rejects_missing_or_reordered_ids() -> None:
    content = '{"translations":[{"id":"s000002","zh":"下一句"}]}'
    with pytest.raises(TranslationError):
        validate_translation(content, ["s000001", "s000002"])


def test_accepts_empty_translation_for_nonverbal_audio() -> None:
    content = '{"translations":[{"id":"s000001","zh":""}]}'
    assert validate_translation(content, ["s000001"]) == {"s000001": ""}


def test_empty_llm_translation_disables_chinese_dubbing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"translations":[{"id":"s000001","zh":""}]}'},
                    }
                ]
            },
        )

    sentence = Sentence(
        id="s000001",
        start_seconds=0.0,
        end_seconds=1.0,
        ja_text="ふふふ……",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            [sentence],
            api_key="secret-test-key",
            model="deepseek-v4-pro",
            client=client,
        )

    assert sentence.zh_text == ""
    assert sentence.enabled is False
    assert sentence.status == "skipped_filler"


def _target_ids(request: httpx.Request) -> list[str]:
    payload = json.loads(request.content)
    return re.findall(r'"id":"(s\d+)"', payload["messages"][-1]["content"])


def _translation_response(ids: list[str]) -> httpx.Response:
    content = json.dumps(
        {"translations": [{"id": sentence_id, "zh": f"译文{sentence_id}"} for sentence_id in ids]},
        ensure_ascii=False,
    )
    return httpx.Response(
        200,
        json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
    )


def test_deepseek_request_uses_pro_non_thinking_and_prior_translation_memory() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["Authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"translations":[{"id":"s000002","zh":"开始吧。"}]}'
                        },
                    }
                ]
            },
        )

    sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.0,
            ja_text="お姉ちゃん。",
            zh_text="姐姐。",
        ),
        Sentence(
            id="s000002",
            start_seconds=1.0,
            end_seconds=2.0,
            ja_text="始めましょう。",
        ),
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            sentences,
            api_key="secret-test-key",
            model="deepseek-v4-pro",
            client=client,
        )

    payload = seen["payload"]
    assert seen["authorization"] == "Bearer secret-test-key"
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert payload["max_tokens"] >= 16_384
    assert payload["response_format"] == {"type": "json_object"}
    assert "姐姐。" in payload["messages"][2]["content"]
    assert "secret-test-key" not in json.dumps(payload)
    assert sentences[1].zh_text == "开始吧。"


def test_translation_can_omit_full_context_and_translation_memory() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"translations":[{"id":"s000002","zh":"开始吧。"}]}'
                        },
                    }
                ]
            },
        )

    sentences = [
        Sentence(
            id="s000001",
            start_seconds=0.0,
            end_seconds=1.0,
            ja_text="お姉ちゃん。",
            zh_text="姐姐。",
        ),
        Sentence(
            id="s000002",
            start_seconds=1.0,
            end_seconds=2.0,
            ja_text="始めましょう。",
        ),
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            sentences,
            api_key="secret-test-key",
            model="deepseek-v4-pro",
            send_context=False,
            client=client,
        )

    messages = seen["payload"]["messages"]
    assert len(messages) == 2
    assert "完整日文转写" not in json.dumps(messages, ensure_ascii=False)
    assert "姐姐。" not in json.dumps(messages, ensure_ascii=False)


def test_translation_batches_are_small_and_checkpointed() -> None:
    calls: list[list[str]] = []
    checkpoints = 0

    def handler(request: httpx.Request) -> httpx.Response:
        ids = _target_ids(request)
        calls.append(ids)
        return _translation_response(ids)

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1

    sentences = [
        Sentence(
            id=f"s{index:06d}",
            start_seconds=float(index),
            end_seconds=float(index + 1),
            ja_text="短い文。",
        )
        for index in range(1, 34)
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            sentences,
            api_key="secret-test-key",
            model="deepseek-v4-pro",
            client=client,
            on_batch=checkpoint,
        )

    assert [len(ids) for ids in calls] == [16, 16, 1]
    assert checkpoints == 3
    assert all(sentence.zh_text for sentence in sentences)


def test_length_response_splits_batch_without_repeating_same_request() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = _target_ids(request)
        calls.append(ids)
        if len(ids) > 1:
            return httpx.Response(
                200,
                json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
            )
        return _translation_response(ids)

    sentences = [
        Sentence(
            id=f"s{index:06d}",
            start_seconds=float(index),
            end_seconds=float(index + 1),
            ja_text="はい。",
        )
        for index in range(1, 5)
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            sentences,
            api_key="secret-test-key",
            model="deepseek-v4-pro",
            client=client,
        )

    assert calls == [
        ["s000001", "s000002", "s000003", "s000004"],
        ["s000001", "s000002"],
        ["s000001"],
        ["s000002"],
        ["s000003", "s000004"],
        ["s000003"],
        ["s000004"],
    ]
    assert all(sentence.zh_text for sentence in sentences)


def test_single_line_length_response_increases_budget() -> None:
    budgets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        budgets.append(payload["max_tokens"])
        ids = _target_ids(request)
        if len(budgets) == 1:
            return httpx.Response(
                200,
                json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
            )
        return _translation_response(ids)

    sentence = Sentence(id="s000001", start_seconds=0.0, end_seconds=1.0, ja_text="はい。")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            [sentence],
            api_key="secret-test-key",
            model="deepseek-v4-pro",
            client=client,
        )

    assert budgets == [16_384, 32_768]
    assert sentence.zh_text == "译文s000001"


def test_deepseek_auth_error_is_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid key")

    sentences = [Sentence(id="s000001", start_seconds=0.0, end_seconds=1.0, ja_text="はい。")]
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(TranslationError, match="HTTP 401"),
    ):
        translate_sentences(
            sentences,
            api_key="bad-key",
            model="deepseek-v4-pro",
            client=client,
        )
    assert calls == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "openai_compatible"])
def test_llm_provider_adapters_keep_strict_sentence_ids(provider: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if provider in {"openai", "openai_compatible"}:
            assert request.headers.get("Authorization") == "Bearer provider-key"
            assert payload["response_format"] == {"type": "json_object"}
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": ('{"translations":[{"id":"s000001","zh":"你好。"}]}')
                            },
                        }
                    ]
                },
            )
        if provider == "anthropic":
            assert request.headers["x-api-key"] == "provider-key"
            return httpx.Response(
                200,
                json={
                    "stop_reason": "end_turn",
                    "content": [
                        {
                            "type": "text",
                            "text": '{"translations":[{"id":"s000001","zh":"你好。"}]}',
                        }
                    ],
                },
            )
        assert request.headers["x-goog-api-key"] == "provider-key"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": ('{"translations":[{"id":"s000001","zh":"你好。"}]}')}
                            ]
                        },
                    }
                ]
            },
        )

    sentence = Sentence(id="s000001", start_seconds=0.0, end_seconds=1.0, ja_text="こんにちは。")
    base_urls = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "openai_compatible": "http://127.0.0.1:11434/v1",
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            [sentence],
            api_key="provider-key",
            model="test-model",
            base_url=base_urls[provider],
            provider=provider,
            client=client,
        )
    assert sentence.zh_text == "你好。"


@pytest.mark.parametrize("provider", ["deepl", "google_translate", "microsoft_translate"])
def test_machine_translation_provider_adapters(provider: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if provider == "deepl":
            assert request.headers["Authorization"] == "DeepL-Auth-Key provider-key"
            return httpx.Response(200, json={"translations": [{"text": "晚安。"}]})
        if provider == "google_translate":
            assert request.headers["x-goog-api-key"] == "provider-key"
            return httpx.Response(
                200,
                json={"data": {"translations": [{"translatedText": "晚安。"}]}},
            )
        assert request.headers["Ocp-Apim-Subscription-Key"] == "provider-key"
        assert request.headers["Ocp-Apim-Subscription-Region"] == "eastasia"
        return httpx.Response(200, json=[{"translations": [{"text": "晚安。"}]}])

    sentence = Sentence(id="s000001", start_seconds=0.0, end_seconds=1.0, ja_text="おやすみ。")
    base_urls = {
        "deepl": "https://api.deepl.com",
        "google_translate": "https://translation.googleapis.com/language/translate/v2",
        "microsoft_translate": "https://api.cognitive.microsofttranslator.com",
    }
    models = {
        "deepl": "prefer_quality_optimized",
        "google_translate": "nmt",
        "microsoft_translate": "general",
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        translate_sentences(
            [sentence],
            api_key="provider-key",
            model=models[provider],
            base_url=base_urls[provider],
            provider=provider,
            microsoft_region="eastasia",
            client=client,
        )
    assert sentence.zh_text == "晚安。"
