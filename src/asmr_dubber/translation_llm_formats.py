"""Provider-specific REST request/response builders for LLM translation adapters."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from .api_contracts import APIContractError, merge_extra_body
from .errors import TranslationError
from .translation_schema import _NonRetryableTranslationError

OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {
        "deepseek",
        "bailian",
        "doubao",
        "openai",
        "openai_compatible",
        "sensenova",
    }
)


def _openai_provider_controls(
    provider: str,
    model: str,
    base_url: str = "",
) -> dict[str, object]:
    """Return non-thinking controls accepted by each provider's official REST API."""

    host = base_url.casefold()
    if provider == "deepseek" or "api.deepseek.com" in host:
        return {"thinking": {"type": "disabled"}}
    if provider == "bailian" or "dashscope.aliyuncs.com" in host:
        return {"enable_thinking": False}
    if provider == "doubao" or "volces.com" in host or "volcengine.com" in host:
        if model.casefold().startswith("doubao-seed-2-0-"):
            return {"reasoning_effort": "minimal"}
        return {"thinking": {"type": "disabled"}}
    if provider == "sensenova" or "sensenova.cn" in host:
        return {"reasoning_effort": "none"}
    if (
        provider == "openai"
        and model.casefold().startswith("gpt-5")
        and "pro" not in model.casefold()
    ):
        return {"reasoning_effort": "none"}
    return {}


def _request_deepseek_chat(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    job_id: str,
    temperature: float,
    top_p: float,
    extra_body: str,
) -> tuple[str, bool]:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        # Translation is a constrained text-to-text mapping.  Thinking
        # consumes the same completion budget as the final JSON and can
        # end the request before any usable content is returned.
        "thinking": {"type": "disabled"},
        "user_id": job_id,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        payload = merge_extra_body(
            payload,
            extra_body,
            label="翻译附加请求参数",
            reserved={"model", "messages", "stream"},
        )
    except APIContractError as exc:
        raise _NonRetryableTranslationError(str(exc)) from exc
    response = client.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if response.status_code in {401, 402, 403, 422}:
        raise _NonRetryableTranslationError(
            f"DeepSeek API 拒绝请求（HTTP {response.status_code}）：{response.text[:500]}",
            response.status_code,
        )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        return "", True
    content = choice.get("message", {}).get("content") or ""
    return content, False


def _request_openai_chat(
    client: httpx.Client,
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    job_id: str,
    temperature: float,
    top_p: float,
    extra_body: str,
    raise_for_status: Callable[[httpx.Response], None],
) -> tuple[str, bool]:
    token_field = "max_completion_tokens" if provider in {"openai", "sensenova"} else "max_tokens"
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        token_field: max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": {"type": "json_object"},
        "stream": False,
        "user": job_id,
    }
    payload.update(_openai_provider_controls(provider, model, base_url))
    if provider == "sensenova":
        # SenseNova's compatibility gateway uses a strict request schema.
        payload.pop("user", None)
    try:
        payload = merge_extra_body(
            payload,
            extra_body,
            label="翻译附加请求参数",
            reserved={"model", "messages", "stream"},
        )
    except APIContractError as exc:
        raise TranslationError(str(exc)) from exc
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = client.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
    )
    # Some local OpenAI-compatible servers implement chat but not JSON mode.
    if provider in {"openai_compatible", "sensenova"} and response.status_code == 400:
        payload.pop("response_format", None)
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
    raise_for_status(response)
    data = response.json()
    choice = data["choices"][0]
    return choice.get("message", {}).get("content") or "", (choice.get("finish_reason") == "length")


def _request_anthropic_messages(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    raise_for_status: Callable[[httpx.Response], None],
) -> tuple[str, bool]:
    user_messages = [item for item in messages if item["role"] != "system"]
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": user_messages,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    response = client.post(
        f"{base_url}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    raise_for_status(response)
    data = response.json()
    content = "".join(
        part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"
    )
    return content, data.get("stop_reason") == "max_tokens"


def _request_gemini_content(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    raise_for_status: Callable[[httpx.Response], None],
) -> tuple[str, bool]:
    user_text = "\n\n".join(item["content"] for item in messages if item["role"] != "system")
    generation_config: dict[str, object] = {
        "responseMimeType": "application/json",
        "maxOutputTokens": max_output_tokens,
    }
    if model.startswith("gemini-3.1-pro"):
        generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
    elif model.startswith(("gemini-3.6", "gemini-3.5")):
        generation_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
    if not model.startswith(("gemini-3.6", "gemini-3.5-flash-lite")):
        generation_config.update(
            temperature=temperature,
            topP=top_p,
        )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": generation_config,
    }
    response = client.post(
        f"{base_url}/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
    )
    raise_for_status(response)
    data = response.json()
    candidate = data["candidates"][0]
    content = "".join(
        part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
    )
    return content, candidate.get("finishReason") in {"MAX_TOKENS", "LENGTH"}
