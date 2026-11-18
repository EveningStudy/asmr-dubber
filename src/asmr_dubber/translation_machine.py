"""Machine translation service adapters (DeepL, Google Cloud Translation, Microsoft Translator)."""

from __future__ import annotations

import html
from typing import Self

import httpx

from .errors import TranslationError
from .languages import MACHINE_TRANSLATION_LANGUAGE_CODES, SourceLanguage
from .translation_schema import (
    _FATAL_HTTP_STATUS,
    TranslationChunk,
    _NonRetryableTranslationError,
)

MACHINE_TRANSLATION_PROVIDERS = frozenset({"deepl", "google_translate", "microsoft_translate"})


def _default_http_client(label: str, *, timeout: float) -> httpx.Client:
    from . import translation

    return translation.logged_http_client(label, timeout=timeout)


class _BaseMachineTranslator:
    """Shared transport lifecycle, error mapping, and line-count validation for MT services."""

    provider: str = ""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        target_language: str = "zh",
        provider: str | None = None,
    ) -> None:
        resolved_provider = provider or self.provider
        if not api_key.strip():
            raise TranslationError(f"{resolved_provider} 缺少 API Key。")
        self.provider = resolved_provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.source_language: SourceLanguage = source_language
        self.target_language = target_language
        self.client = client or _default_http_client(
            f"{self.provider} 翻译 API", timeout=timeout_seconds
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request_translations(self, texts: list[str], source_code: str) -> list[str]:
        raise TranslationError(f"不支持的机器翻译服务：{self.provider}")

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        _full_context: str,
        _translation_memory: str,
        _job_id: str,
    ) -> dict[str, str]:
        if self.provider not in MACHINE_TRANSLATION_LANGUAGE_CODES:
            raise TranslationError(f"不支持的机器翻译服务：{self.provider}")
        texts = [sentence.source_text for sentence in chunk.sentences]
        source_code = MACHINE_TRANSLATION_LANGUAGE_CODES[self.provider][self.source_language]
        try:
            translated = self._request_translations(texts, source_code)
        except httpx.HTTPStatusError as exc:
            response = exc.response
            message = (
                f"{self.provider} API 请求失败（HTTP {response.status_code}）："
                f"{response.text[:500]}"
            )
            if response.status_code in _FATAL_HTTP_STATUS:
                raise _NonRetryableTranslationError(message, response.status_code) from exc
            raise TranslationError(message) from exc
        except TranslationError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise TranslationError(f"{self.provider} 返回格式无效：{exc}") from exc
        if len(translated) != len(chunk.sentences) or any(not item.strip() for item in translated):
            raise TranslationError(f"{self.provider} 返回的译文数量或内容无效。")
        return {
            sentence.id: text.strip()
            for sentence, text in zip(chunk.sentences, translated, strict=True)
        }


class DeepLTranslator(_BaseMachineTranslator):
    provider = "deepl"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        deepl_formality: str = "default",
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        target_language: str = "zh",
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            client=client,
            source_language=source_language,
            target_language=target_language,
        )
        self.deepl_formality = deepl_formality

    def _request_translations(self, texts: list[str], source_code: str) -> list[str]:
        payload: dict[str, object] = {
            "text": texts,
            "source_lang": source_code,
            "target_lang": "EN-US" if self.target_language == "en" else "ZH-HANS",
            "model_type": self.model,
        }
        if self.deepl_formality != "default":
            payload["formality"] = self.deepl_formality
        response = self.client.post(
            f"{self.base_url}/v2/translate",
            headers={
                "Authorization": f"DeepL-Auth-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return [item["text"] for item in response.json()["translations"]]


class GoogleTranslator(_BaseMachineTranslator):
    provider = "google_translate"

    def _request_translations(self, texts: list[str], source_code: str) -> list[str]:
        response = self.client.post(
            self.base_url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "q": texts,
                "source": source_code,
                "target": "en" if self.target_language == "en" else "zh-CN",
                "format": "text",
                "model": self.model,
            },
        )
        response.raise_for_status()
        return [
            html.unescape(item["translatedText"])
            for item in response.json()["data"]["translations"]
        ]


class MicrosoftTranslator(_BaseMachineTranslator):
    provider = "microsoft_translate"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        microsoft_region: str = "",
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        target_language: str = "zh",
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            client=client,
            source_language=source_language,
            target_language=target_language,
        )
        self.microsoft_region = microsoft_region.strip()

    def _request_translations(self, texts: list[str], source_code: str) -> list[str]:
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json; charset=UTF-8",
        }
        if self.microsoft_region:
            headers["Ocp-Apim-Subscription-Region"] = self.microsoft_region
        response = self.client.post(
            f"{self.base_url}/translate",
            params={
                "api-version": "3.0",
                "from": source_code,
                "to": "en" if self.target_language == "en" else "zh-Hans",
            },
            headers=headers,
            json=[{"Text": text} for text in texts],
        )
        response.raise_for_status()
        return [item["translations"][0]["text"] for item in response.json()]


class MachineTranslationAPI(_BaseMachineTranslator):
    """Unified machine-translation facade dispatching to provider-specific adapters."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        deepl_formality: str,
        microsoft_region: str,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
        source_language: SourceLanguage = "ja",
        target_language: str = "zh",
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            client=client,
            source_language=source_language,
            target_language=target_language,
            provider=provider,
        )
        self.deepl_formality = deepl_formality
        self.microsoft_region = microsoft_region.strip()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, str]:
        from . import translation

        if self.provider == "deepl":
            adapter: _BaseMachineTranslator = translation.DeepLTranslator(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                deepl_formality=self.deepl_formality,
                client=self.client,
                source_language=self.source_language,
                target_language=self.target_language,
            )
        elif self.provider == "google_translate":
            adapter = translation.GoogleTranslator(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                client=self.client,
                source_language=self.source_language,
                target_language=self.target_language,
            )
        elif self.provider == "microsoft_translate":
            adapter = translation.MicrosoftTranslator(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                microsoft_region=self.microsoft_region,
                client=self.client,
                source_language=self.source_language,
                target_language=self.target_language,
            )
        else:
            raise TranslationError(f"不支持的机器翻译服务：{self.provider}")
        return adapter.translate_chunk(chunk, full_context, translation_memory, job_id)
