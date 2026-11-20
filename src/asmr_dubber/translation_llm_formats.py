"""Provider-specific REST request/response builders for LLM translation adapters."""

from __future__ import annotations

import re
from collections.abc import Callable

import httpx

from .api_contracts import APIContractError, merge_extra_body, parse_json_object
from .errors import TranslationError
from .translation_schema import _FATAL_HTTP_STATUS, _NonRetryableTranslationError

_RECOVERABLE_400_MODERATION = (
    "content exists risk",
    "data_inspection_failed",
    "content_policy",
    "inappropriate content",
    "safety",
    "sensitive",
)
_RECOVERABLE_400_CONTEXT = (
    "context_length",
    "maximum context",
    "too many tokens",
    "token limit",
    "input is too long",
    "prompt is too long",
)
_RECOVERABLE_400_TRANSIENT = (
    "temporary",
    "try again",
    "overloaded",
    "upstream",
    "busy",
    "timeout",
)


def _raise_for_translation_status(response: httpx.Response, provider_label: str) -> None:
    status = response.status_code
    if status == 400:
        body_lower = response.text[:1000].casefold()
        if any(marker in body_lower for marker in _RECOVERABLE_400_MODERATION):
            raise _NonRetryableTranslationError(
                f"{provider_label} API 拦截本批内容（HTTP 400）：{response.text[:500]}",
                status_code=400,
                fatal=False,
            )
        if any(marker in body_lower for marker in _RECOVERABLE_400_CONTEXT):
            raise TranslationError(
                f"{provider_label} API 上下文超限（HTTP 400）：{response.text[:500]}"
            )
        if any(marker in body_lower for marker in _RECOVERABLE_400_TRANSIENT):
            response.raise_for_status()
        raise _NonRetryableTranslationError(
            f"{provider_label} API 拒绝请求（HTTP 400）：{response.text[:500]}",
            status_code=400,
            fatal=True,
        )
    if status in _FATAL_HTTP_STATUS:
        raise _NonRetryableTranslationError(
            f"{provider_label} API 拒绝请求（HTTP {status}）：{response.text[:500]}",
            status_code=status,
            fatal=True,
        )
    response.raise_for_status()


_OPENAI_REASONING_MODEL = re.compile(
    r"^(?:o[134](?:-|$)|gpt-5(?:\.\d+)?-pro(?:-|$))",
    re.IGNORECASE,
)

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
        raise _NonRetryableTranslationError(str(exc), fatal=True) from exc
    response = client.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    _raise_for_translation_status(response, "DeepSeek")
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
    include_response_format: bool = True,
    on_response_format_unsupported: Callable[[], None] | None = None,
) -> tuple[str, bool]:
    token_field = "max_completion_tokens" if provider in {"openai", "sensenova"} else "max_tokens"
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        token_field: max_output_tokens,
        "stream": False,
        "user": job_id,
    }
    if not _OPENAI_REASONING_MODEL.match(model.strip()):
        payload["temperature"] = temperature
        payload["top_p"] = top_p
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}
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
        raise _NonRetryableTranslationError(str(exc), fatal=True) from exc
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = client.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
    )
    # Some local OpenAI-compatible servers implement chat but not JSON mode.
    if (
        provider in {"openai_compatible", "sensenova"}
        and include_response_format
        and response.status_code == 400
    ):
        if on_response_format_unsupported is not None:
            on_response_format_unsupported()
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
    extra_body: str = "{}",
    raise_for_status: Callable[[httpx.Response], None],
) -> tuple[str, bool]:
    del top_p  # Anthropic rejects sending temperature and top_p together.
    user_messages = [item for item in messages if item["role"] != "system"]
    payload: dict[str, object] = {
        "model": model,
        "system": system_prompt,
        "messages": user_messages,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }
    try:
        extra = parse_json_object(extra_body, label="翻译附加请求参数")
        payload = merge_extra_body(
            payload,
            extra_body,
            label="翻译附加请求参数",
            reserved={"model", "messages", "system", "stream"},
        )
    except APIContractError as exc:
        raise _NonRetryableTranslationError(str(exc), fatal=True) from exc
    if "top_p" in payload and "temperature" not in extra:
        payload.pop("temperature", None)
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/v1"):
        clean_base = clean_base[:-3]
    response = client.post(
        f"{clean_base}/v1/messages",
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
    extra_body: str = "{}",
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
    payload: dict[str, object] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": generation_config,
    }
    try:
        payload = merge_extra_body(
            payload,
            extra_body,
            label="翻译附加请求参数",
            reserved={"contents", "systemInstruction"},
        )
    except APIContractError as exc:
        raise _NonRetryableTranslationError(str(exc), fatal=True) from exc
    extra_config = payload.get("generationConfig")
    if isinstance(extra_config, dict) and extra_config is not generation_config:
        merged_config = dict(generation_config)
        merged_config.update(extra_config)
        payload["generationConfig"] = merged_config
    response = client.post(
        f"{base_url}/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
    )
    raise_for_status(response)
    data = response.json()
    prompt_feedback = data.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        raise _NonRetryableTranslationError(
            f"Gemini 安全策略拦截了翻译请求（blockReason={prompt_feedback['blockReason']}）。",
            fatal=False,
        )
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise _NonRetryableTranslationError(
            "Gemini 未返回候选结果（可能被安全策略拦截）。",
            fatal=False,
        )
    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    parts = candidate.get("content", {}).get("parts", [])
    content = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("thought") is not True
    )
    if not content.strip() and finish_reason in {
        "SAFETY",
        "RECITATION",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
    }:
        raise _NonRetryableTranslationError(
            f"Gemini 安全策略拦截了翻译输出（finishReason={finish_reason}）。",
            fatal=False,
        )
    return content, finish_reason in {"MAX_TOKENS", "LENGTH"}
