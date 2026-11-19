"""LLM translation provider adapters (DeepSeek and multi-provider REST clients)."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from typing import Self

import httpx

from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .errors import TranslationError
from .languages import SourceLanguage, source_language_label
from .models import Sentence
from .translation_llm_formats import (
    OPENAI_COMPATIBLE_PROVIDERS,
    _raise_for_translation_status,
    _request_anthropic_messages,
    _request_deepseek_chat,
    _request_gemini_content,
    _request_openai_chat,
)
from .translation_prompts import (
    _deepseek_translation_request,
    _json_lines,
    budget_rule,
    default_translation_prompt,
    sound_policy_rule,
)
from .translation_schema import (
    TranslationChunk,
    TranslationEntry,
    _NonRetryableTranslationError,
    _OutputLengthTranslationError,
    validate_translation_entries,
)


def _default_http_client(label: str, *, timeout: float) -> httpx.Client:
    from . import translation

    return translation.logged_http_client(label, timeout=timeout)


def _retry_sleep(attempt: int) -> None:
    time.sleep(min(20.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.75)))


class _BaseLLMTranslator:
    """Shared HTTP client lifecycle and context message assembly for LLM adapters."""

    system_prompt: str
    source_language: SourceLanguage
    sound_policy: str
    max_retries: int
    client: httpx.Client
    _owns_client: bool

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _prefix_messages(
        self,
        full_context: str,
        translation_memory: str,
        *,
        memory_prompt: str,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if full_context != "[]":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"完整{source_language_label(self.source_language)}转写，"
                        "仅供保持人物称谓和上下文一致：\n" + full_context
                    ),
                }
            )
        if translation_memory != "[]":
            messages.append({"role": "user", "content": memory_prompt + translation_memory})
        return messages


class DeepSeekTranslator(_BaseLLMTranslator):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 900.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
        system_prompt: str = "",
        temperature: float = 0.1,
        top_p: float = 1.0,
        minimum_output_tokens: int = 16_384,
        source_language: SourceLanguage = "ja",
        extra_body: str = "{}",
        sound_policy: str = "skip",
    ) -> None:
        if not api_key.strip():
            raise TranslationError("缺少 DeepSeek API Key。请在 UI 中填写或设置 DEEPSEEK_API_KEY。")
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.sound_policy = sound_policy
        self.system_prompt = system_prompt.strip() or default_translation_prompt(
            source_language, sound_policy=sound_policy
        )
        self.temperature = temperature
        self.top_p = top_p
        self.minimum_output_tokens = minimum_output_tokens
        self.source_language = source_language
        self.extra_body = extra_body
        self.client = client or _default_http_client("DeepSeek 翻译 API", timeout=timeout_seconds)
        self._owns_client = client is None

    def _initial_max_tokens(
        self, sentences: Iterable[Sentence], *, include_existing_zh: bool = False
    ) -> int:
        char_count = sum(
            len(item.source_text) + (len(item.zh_text) if include_existing_zh else 0)
            for item in sentences
        )
        return min(
            65_536,
            max(self.minimum_output_tokens, char_count * 4 + 4_096),
        )

    def _request(
        self,
        messages: list[dict[str, str]],
        job_id: str,
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, bool]:
        tokens = self.minimum_output_tokens if max_tokens is None else max_tokens
        return _request_deepseek_chat(
            self.client,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            max_tokens=tokens,
            job_id=job_id,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body=self.extra_body,
        )

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, TranslationEntry]:
        expected_ids = [sentence.id for sentence in chunk.sentences]
        target = _json_lines(
            chunk.sentences,
            field="both" if chunk.compress else "source",
            budgets=chunk.budgets,
            budget_key=chunk.budget_key,
        )
        max_tokens = self._initial_max_tokens(chunk.sentences, include_existing_zh=chunk.compress)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            messages = self._prefix_messages(
                full_context,
                translation_memory,
                memory_prompt="以下是已经确认的译法。后续称谓、专名和语气必须与其一致：\n",
            )
            messages.append(
                {
                    "role": "user",
                    "content": _deepseek_translation_request(
                        target, expected_ids, attempt, self.sound_policy
                    )
                    + budget_rule(chunk),
                }
            )
            try:
                content, output_limited = self._request(messages, job_id, max_tokens=max_tokens)
                if output_limited:
                    length_error = _OutputLengthTranslationError(
                        f"DeepSeek 输出达到长度上限（max_tokens={max_tokens}）。"
                    )
                    # A multi-line batch is more reliably recovered by splitting
                    # it in the caller.  A single pathological line gets two
                    # larger budgets before failing clearly.
                    if len(chunk.sentences) > 1 or max_tokens >= 65_536:
                        raise length_error
                    max_tokens = min(65_536, max_tokens * 2)
                    last_error = length_error
                    continue
                if not content.strip():
                    raise TranslationError("DeepSeek 偶发返回了空内容。")
                return validate_translation_entries(content, expected_ids)
            except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                _retry_sleep(attempt)
        raise TranslationError(
            f"DeepSeek 翻译在 {self.max_retries} 次尝试后仍失败：{last_error}"
        ) from last_error


class LLMTranslator(_BaseLLMTranslator):
    """Provider-specific REST adapters sharing the strict translation contract."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        system_prompt: str = "",
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        timeout_seconds: float = 900.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        extra_body: str = "{}",
        sound_policy: str = "skip",
    ) -> None:
        if not api_key.strip() and provider != "openai_compatible":
            raise TranslationError(f"{provider} 缺少 API Key。")
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.sound_policy = sound_policy
        self.system_prompt = system_prompt.strip() or default_translation_prompt(
            source_language, sound_policy=sound_policy
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.source_language = source_language
        self.extra_body = extra_body
        self._use_response_format = True
        self.client = client or _default_http_client(
            f"{provider} 翻译 API", timeout=timeout_seconds
        )
        self._owns_client = client is None

    def _disable_response_format(self) -> None:
        self._use_response_format = False

    def _messages(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        attempt: int,
    ) -> list[dict[str, str]]:
        target = _json_lines(
            chunk.sentences,
            field="both" if chunk.compress else "source",
            budgets=chunk.budgets,
            budget_key=chunk.budget_key,
        )
        # Every item carries ``kind``; the sound rule matches the configured policy.
        correction = (
            '\n每项只输出 id、zh、kind 三个字段，kind 为 "speech" 或 "sound"。'
            + sound_policy_rule(self.sound_policy)
            + budget_rule(chunk)
        )
        if attempt > 1:
            correction += (
                f"\n这是第 {attempt} 次校验重试：务必返回全部 id 和原顺序；"
                '没有实义的项 kind 写 "sound"，不要省略。'
            )
        messages = self._prefix_messages(
            full_context,
            translation_memory,
            memory_prompt="以下是已经确认的译法，请保持称谓和专名一致：\n",
        )
        messages.append(
            {
                "role": "user",
                "content": "请翻译以下目标项并只输出 json：\n" + target + correction,
            }
        )
        return messages

    def _raise_for_status(self, response: httpx.Response) -> None:
        _raise_for_translation_status(response, self.provider)

    def _request(self, messages: list[dict[str, str]], job_id: str) -> tuple[str, bool]:
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            return _request_openai_chat(
                self.client,
                provider=self.provider,
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                max_output_tokens=self.max_output_tokens,
                job_id=job_id,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body=self.extra_body,
                raise_for_status=self._raise_for_status,
                include_response_format=self._use_response_format,
                on_response_format_unsupported=self._disable_response_format,
            )
        if self.provider == "anthropic":
            return _request_anthropic_messages(
                self.client,
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body=self.extra_body,
                raise_for_status=self._raise_for_status,
            )
        if self.provider == "gemini":
            return _request_gemini_content(
                self.client,
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body=self.extra_body,
                raise_for_status=self._raise_for_status,
            )
        raise TranslationError(f"不支持的大模型翻译服务：{self.provider}")

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, TranslationEntry]:
        expected_ids = [sentence.id for sentence in chunk.sentences]
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                content, output_limited = self._request(
                    self._messages(chunk, full_context, translation_memory, attempt),
                    job_id,
                )
                if output_limited:
                    raise _OutputLengthTranslationError(
                        f"{self.provider} 输出达到长度上限（{self.max_output_tokens} tokens）。"
                    )
                if not content.strip():
                    raise TranslationError(f"{self.provider} 返回了空内容。")
                return validate_translation_entries(content, expected_ids)
            except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                _retry_sleep(attempt)
        raise TranslationError(
            f"{self.provider} 翻译在 {self.max_retries} 次尝试后仍失败：{last_error}"
        ) from last_error
