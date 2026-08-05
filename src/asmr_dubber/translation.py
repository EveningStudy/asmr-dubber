from __future__ import annotations

import html
import json
import random
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .errors import TranslationError
from .languages import (
    MACHINE_TRANSLATION_LANGUAGE_CODES,
    SourceLanguage,
    SpeechSourceLanguage,
    source_language_label,
)
from .models import Sentence
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)

Progress = Callable[[str, int, int], None]


class _NonRetryableTranslationError(TranslationError):
    """Authentication, billing, permission, and request-schema failures."""


class _OutputLengthTranslationError(TranslationError):
    """The completion budget ended before DeepSeek finished the JSON response."""


class TranslationItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    zh: str

    @field_validator("id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("zh")
    @classmethod
    def normalize_translation(cls, value: str) -> str:
        return value.strip()


class TranslationEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    translations: list[TranslationItem] = Field(min_length=1)


class ScriptCorrection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    text: str

    @field_validator("id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class ScriptCorrectionEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    corrections: list[ScriptCorrection] = Field(min_length=1)


@dataclass(frozen=True)
class TranslationChunk:
    sentences: list[Sentence]


def _load_prompt(name: str) -> str:
    return files("asmr_dubber.prompts").joinpath(name).read_text(encoding="utf-8").strip()


_SYSTEM_PROMPT_TEMPLATE = _load_prompt("translation.md")
_DEEPSEEK_STRUCTURE_PROMPT = _load_prompt("translation-structure.md")
_SCRIPT_RECONCILIATION_PROMPT = _load_prompt("script-reconciliation.md")

LLM_RECONCILIATION_PROVIDERS = frozenset(
    {"deepseek", "openai", "anthropic", "gemini", "openai_compatible"}
)

_SOURCE_PROMPT_VALUES: dict[SpeechSourceLanguage, tuple[str, str]] = {
    "ja": ("日语", "日语中的「あ」「え」「う」「ん」「ふふ」「えーと」"),
    "en": ("英语", "英语中的 “uh”“um”“erm”"),
}


def default_translation_prompt(source_language: SourceLanguage = "ja") -> str:
    """Render the packaged prompt for one source language."""

    prompt_language: SpeechSourceLanguage = "en" if source_language == "en" else "ja"
    language, filler_examples = _SOURCE_PROMPT_VALUES[prompt_language]
    return _SYSTEM_PROMPT_TEMPLATE.replace("{{SOURCE_LANGUAGE}}", language).replace(
        "{{FILLER_EXAMPLES}}",
        filler_examples,
    )


# Backward-compatible Japanese default for callers that imported this name.
SYSTEM_PROMPT = default_translation_prompt("ja")


def _deepseek_translation_request(
    target: str,
    expected_ids: list[str],
    attempt: int,
) -> str:
    output_schema = json.dumps(
        {"translations": [{"id": sentence_id, "zh": ""} for sentence_id in expected_ids]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    retry_note = (
        ""
        if attempt == 1
        else (
            f"这是第 {attempt} 次校验重试。上一次输出没有通过结构校验；"
            "这次必须逐字沿用结构模板中的键名、id 和顺序。"
        )
    )
    return (
        _DEEPSEEK_STRUCTURE_PROMPT.replace("{{RETRY_NOTE}}", retry_note)
        .replace("{{OUTPUT_SCHEMA}}", output_schema)
        .replace("{{TARGET_JSON}}", target)
        .strip()
    )


def _chunks(
    sentences: list[Sentence],
    max_lines: int = 16,
    max_chars: int = 4_000,
) -> list[TranslationChunk]:
    result: list[TranslationChunk] = []
    current: list[Sentence] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.source_text)
        if current and (len(current) >= max_lines or count + length > max_chars):
            result.append(TranslationChunk(current))
            current = []
            count = 0
        current.append(sentence)
        count += length
    if current:
        result.append(TranslationChunk(current))
    return result


def _json_lines(sentences: Iterable[Sentence], field: str = "source") -> str:
    items = []
    for sentence in sentences:
        item = {"id": sentence.id, "source": sentence.source_text}
        if field == "both" and sentence.zh_text:
            item["existing_zh"] = sentence.zh_text
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _translation_memory(sentences: Iterable[Sentence]) -> str:
    items = [
        {"id": sentence.id, "source": sentence.source_text, "zh": sentence.zh_text}
        for sentence in sentences
        if sentence.zh_text
    ]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _context_for_chunk(
    sentences: list[Sentence],
    chunk: TranslationChunk,
    maximum_sentences: int,
) -> str:
    if maximum_sentences <= 0:
        return "[]"
    positions = {sentence.id: index for index, sentence in enumerate(sentences)}
    indices = [positions[item.id] for item in chunk.sentences if item.id in positions]
    if not indices:
        return "[]"
    first, last = min(indices), max(indices)
    room = max(0, maximum_sentences - (last - first + 1))
    start = max(0, first - room // 2)
    end = min(len(sentences), last + 1 + (room - (first - start)))
    if end - start < maximum_sentences:
        start = max(0, end - maximum_sentences)
    return _json_lines(sentences[start:end], field="both")


def _bounded_translation_memory(
    sentences: list[Sentence],
    chunk: TranslationChunk,
    maximum_sentences: int,
) -> str:
    if maximum_sentences <= 0:
        return "[]"
    current_ids = {sentence.id for sentence in chunk.sentences}
    confirmed = [
        sentence for sentence in sentences if sentence.zh_text and sentence.id not in current_ids
    ]
    return _translation_memory(confirmed[-maximum_sentences:])


def _extract_json(content: str) -> dict:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TranslationError(f"翻译模型返回的不是有效 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise TranslationError("翻译模型 JSON 顶层必须是对象。")
    return parsed


def validate_translation(content: str, expected_ids: list[str]) -> dict[str, str]:
    try:
        envelope = TranslationEnvelope.model_validate(_extract_json(content))
    except (ValidationError, TranslationError) as exc:
        raise TranslationError(f"翻译结构校验失败：{exc}") from exc
    actual_ids = [item.id for item in envelope.translations]
    if actual_ids != expected_ids:
        missing = [item for item in expected_ids if item not in actual_ids]
        extra = [item for item in actual_ids if item not in expected_ids]
        raise TranslationError(f"翻译返回的 id/顺序不一致；缺少={missing[:8]}，多出={extra[:8]}")
    return {item.id: item.zh for item in envelope.translations}


def validate_script_reconciliation(content: str, expected_ids: list[str]) -> dict[str, str]:
    try:
        envelope = ScriptCorrectionEnvelope.model_validate(_extract_json(content))
    except (ValidationError, TranslationError) as exc:
        raise TranslationError(f"台本校对返回的结构无效：{exc}") from exc
    actual_ids = [item.id for item in envelope.corrections]
    if actual_ids != expected_ids:
        missing = [item for item in expected_ids if item not in actual_ids]
        extra = [item for item in actual_ids if item not in expected_ids]
        raise TranslationError(
            f"台本校对返回的 id/顺序不一致；缺少={missing[:8]}，多出={extra[:8]}"
        )
    return {item.id: item.text for item in envelope.corrections}


class DeepSeekTranslator:
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
    ) -> None:
        if not api_key.strip():
            raise TranslationError("缺少 DeepSeek API Key。请在 UI 中填写或设置 DEEPSEEK_API_KEY。")
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.system_prompt = system_prompt.strip() or default_translation_prompt(source_language)
        self.temperature = temperature
        self.top_p = top_p
        self.minimum_output_tokens = minimum_output_tokens
        self.source_language = source_language
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> DeepSeekTranslator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, str]:
        expected_ids = [sentence.id for sentence in chunk.sentences]
        target = _json_lines(chunk.sentences)
        max_tokens = min(
            65_536,
            max(
                self.minimum_output_tokens,
                sum(len(item.source_text) for item in chunk.sentences) * 4 + 4_096,
            ),
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
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
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "以下是已经确认的译法。后续称谓、专名和语气必须与其一致：\n"
                            + translation_memory
                        ),
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": _deepseek_translation_request(target, expected_ids, attempt),
                }
            )
            payload = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens,
                # Translation is a constrained text-to-text mapping.  Thinking
                # consumes the same completion budget as the final JSON and can
                # end the request before any usable content is returned.
                "thinking": {"type": "disabled"},
                "user_id": job_id,
                "stream": False,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code in {401, 402, 403, 422}:
                    raise _NonRetryableTranslationError(
                        f"DeepSeek API 拒绝请求（HTTP {response.status_code}）："
                        f"{response.text[:500]}"
                    )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
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
                content = choice.get("message", {}).get("content") or ""
                if not content.strip():
                    raise TranslationError("DeepSeek 偶发返回了空内容。")
                return validate_translation(content, expected_ids)
            except _NonRetryableTranslationError:
                raise
            except _OutputLengthTranslationError:
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                delay = min(20.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.75))
                time.sleep(delay)
        raise TranslationError(
            f"DeepSeek 翻译在 {self.max_retries} 次尝试后仍失败：{last_error}"
        ) from last_error


class LLMTranslator:
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
    ) -> None:
        if not api_key.strip() and provider != "openai_compatible":
            raise TranslationError(f"{provider} 缺少 API Key。")
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt.strip() or default_translation_prompt(source_language)
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.source_language = source_language
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> LLMTranslator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _messages(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        attempt: int,
    ) -> list[dict[str, str]]:
        target = _json_lines(chunk.sentences)
        correction = (
            ""
            if attempt == 1
            else f"\n这是第 {attempt} 次校验重试：务必返回全部 id 和原顺序；无实义项用空 zh。"
        )
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
            messages.append(
                {
                    "role": "user",
                    "content": "以下是已经确认的译法，请保持称谓和专名一致：\n"
                    + translation_memory,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": "请翻译以下目标项并只输出 json：\n" + target + correction,
            }
        )
        return messages

    def _request(self, messages: list[dict[str, str]], job_id: str) -> tuple[str, bool]:
        if self.provider in {"deepseek", "openai", "openai_compatible"}:
            token_field = "max_completion_tokens" if self.provider == "openai" else "max_tokens"
            payload: dict[str, object] = {
                "model": self.model,
                "messages": messages,
                token_field: self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "response_format": {"type": "json_object"},
                "stream": False,
                "user": job_id,
            }
            if self.provider == "deepseek":
                payload["thinking"] = {"type": "disabled"}
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            # Some local OpenAI-compatible servers implement chat but not JSON mode.
            if self.provider == "openai_compatible" and response.status_code == 400:
                payload.pop("response_format", None)
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            self._raise_for_status(response)
            data = response.json()
            choice = data["choices"][0]
            return choice.get("message", {}).get("content") or "", choice.get(
                "finish_reason"
            ) == "length"

        if self.provider == "anthropic":
            user_messages = [item for item in messages if item["role"] != "system"]
            payload = {
                "model": self.model,
                "system": self.system_prompt,
                "messages": user_messages,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            response = self.client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            self._raise_for_status(response)
            data = response.json()
            content = "".join(
                part.get("text", "")
                for part in data.get("content", [])
                if part.get("type") == "text"
            )
            return content, data.get("stop_reason") == "max_tokens"

        if self.provider == "gemini":
            user_text = "\n\n".join(
                item["content"] for item in messages if item["role"] != "system"
            )
            generation_config: dict[str, object] = {
                "responseMimeType": "application/json",
                "maxOutputTokens": self.max_output_tokens,
            }
            if not self.model.startswith(("gemini-3.6", "gemini-3.5-flash-lite")):
                generation_config.update(
                    temperature=self.temperature,
                    topP=self.top_p,
                )
            payload = {
                "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generationConfig": generation_config,
            }
            response = self.client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            self._raise_for_status(response)
            data = response.json()
            candidate = data["candidates"][0]
            content = "".join(
                part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
            )
            return content, candidate.get("finishReason") in {"MAX_TOKENS", "LENGTH"}

        raise TranslationError(f"不支持的大模型翻译服务：{self.provider}")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code in {400, 401, 402, 403, 404, 422}:
            raise _NonRetryableTranslationError(
                f"{self.provider} API 拒绝请求（HTTP {response.status_code}）："
                f"{response.text[:500]}"
            )
        response.raise_for_status()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        full_context: str,
        translation_memory: str,
        job_id: str,
    ) -> dict[str, str]:
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
                return validate_translation(content, expected_ids)
            except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                raise
            except TranslationError as exc:
                last_error = exc
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(20.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.75)))
        raise TranslationError(
            f"{self.provider} 翻译在 {self.max_retries} 次尝试后仍失败：{last_error}"
        ) from last_error


@dataclass(frozen=True)
class ScriptReconciliationBatch:
    recognized: list[Sentence]
    script: list[tuple[str, str]]


def _script_reconciliation_batches(
    recognized: list[Sentence],
    script_lines: list[str],
    *,
    maximum_recognized: int = 8,
    context_lines: int = 6,
) -> list[ScriptReconciliationBatch]:
    if not recognized:
        raise TranslationError("没有可供台本校对的识别结果。")
    if not script_lines:
        raise TranslationError("台本中没有可供校对的文字。")
    total_recognized = len(recognized)
    total_script = len(script_lines)
    batches: list[ScriptReconciliationBatch] = []
    for start in range(0, total_recognized, maximum_recognized):
        end = min(total_recognized, start + maximum_recognized)
        estimated_start = int(start * total_script / total_recognized)
        estimated_end = int(end * total_script / total_recognized)
        window_start = max(0, estimated_start - context_lines)
        window_end = min(total_script, estimated_end + context_lines)
        if window_end - window_start > 48:
            window_end = min(total_script, window_start + 48)
        if window_end <= window_start:
            window_end = min(total_script, window_start + 1)
        script = [
            (f"p{index + 1:06d}", text)
            for index, text in enumerate(script_lines[window_start:window_end], start=window_start)
        ]
        batches.append(ScriptReconciliationBatch(recognized[start:end], script))
    return batches


def _script_reconciliation_messages(
    batch: ScriptReconciliationBatch,
    *,
    source_language: SourceLanguage,
    target: Literal["source", "zh"],
    attempt: int,
) -> list[dict[str, str]]:
    output_label = source_language_label(source_language) if target == "source" else "中文"
    retry_note = (
        ""
        if attempt == 1
        else f"这是第 {attempt} 次校验重试。必须返回全部识别 id，且顺序完全一致。"
    )
    recognized_payload = []
    for sentence in batch.recognized:
        item: dict[str, object] = {
            "id": sentence.id,
            "start": round(sentence.start_seconds, 3),
            "end": round(sentence.end_seconds, 3),
            "recognized": sentence.source_text,
        }
        if target == "zh":
            item["translation"] = sentence.zh_text
        recognized_payload.append(item)
    script_payload = [{"id": line_id, "text": text} for line_id, text in batch.script]
    recognized_json = json.dumps(recognized_payload, ensure_ascii=False, separators=(",", ":"))
    script_json = json.dumps(script_payload, ensure_ascii=False, separators=(",", ":"))
    system = (
        _SCRIPT_RECONCILIATION_PROMPT.replace("{{OUTPUT_LABEL}}", output_label)
        .replace("{{RETRY_NOTE}}", retry_note)
        .replace("{{RECOGNIZED_JSON}}", recognized_json)
        .replace("{{SCRIPT_JSON}}", script_json)
        .strip()
    )
    user = (
        "请只根据下面两组数据完成校对。输出结构必须是："
        + json.dumps(
            {"corrections": [{"id": sentence.id, "text": ""} for sentence in batch.recognized]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n请根据系统消息中的数据完成校对。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def reconcile_script_sentences(
    recognized: list[Sentence],
    script_lines: list[str],
    *,
    source_language: SourceLanguage,
    target: Literal["source", "zh"],
    provider: str,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_output_tokens: int = 16_384,
    job_id: str = "asmr_dubber_script_review",
    progress: Progress | None = None,
    on_batch: Callable[[], None] | None = None,
    client: httpx.Client | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Use an LLM to map an untimed script onto ASR timing without changing time ranges."""

    if provider not in LLM_RECONCILIATION_PROVIDERS:
        raise TranslationError(
            "台本校对需要大模型翻译服务；请选择 DeepSeek、OpenAI、Claude、Gemini "
            "或本地/自定义 OpenAI-compatible。"
        )
    batches = _script_reconciliation_batches(recognized, script_lines)
    translator = LLMTranslator(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt="",
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        client=client,
        source_language=source_language,
    )
    cancel_translation = translator.close
    callback_signal = register_cancel_callback(cancel_translation, cancel_event)
    corrections: dict[str, str] = {}
    report: list[dict[str, object]] = []
    try:
        with translator:
            for index, batch in enumerate(batches):
                check_cancelled(cancel_event)
                expected_ids = [sentence.id for sentence in batch.recognized]
                last_error: Exception | None = None
                result: dict[str, str] | None = None
                for attempt in range(1, 4):
                    check_cancelled(cancel_event)
                    messages = _script_reconciliation_messages(
                        batch,
                        source_language=source_language,
                        target=target,
                        attempt=attempt,
                    )
                    # Anthropic and Gemini use this field as their system instruction;
                    # OpenAI-compatible providers also receive the explicit system message.
                    translator.system_prompt = messages[0]["content"]
                    try:
                        content, output_limited = translator._request(messages, job_id)
                        if output_limited:
                            raise _OutputLengthTranslationError(
                                f"台本校对第 {index + 1} 批输出达到长度上限。"
                            )
                        if not content.strip():
                            raise TranslationError("台本校对模型返回了空内容。")
                        result = validate_script_reconciliation(content, expected_ids)
                        break
                    except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                        raise
                    except TranslationError as exc:
                        last_error = exc
                    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                        last_error = exc
                    if attempt < 3:
                        time.sleep(min(8.0, attempt * 1.5 + random.uniform(0.0, 0.5)))
                if result is None:
                    raise TranslationError(
                        f"台本校对第 {index + 1}/{len(batches)} 批失败：{last_error}"
                    ) from last_error
                corrections.update(result)
                report.append(
                    {
                        "recognized_ids": expected_ids,
                        "script_ids": [line_id for line_id, _ in batch.script],
                        "corrections": result,
                    }
                )
                if on_batch:
                    on_batch()
                if progress:
                    progress(
                        f"台本校对第 {index + 1}/{len(batches)} 批完成",
                        index + 1,
                        len(batches),
                    )
                check_cancelled(cancel_event)
    finally:
        unregister_cancel_callback(cancel_translation, callback_signal)
    return corrections, report


class MachineTranslationAPI:
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
    ) -> None:
        if not api_key.strip():
            raise TranslationError(f"{provider} 缺少 API Key。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.deepl_formality = deepl_formality
        self.microsoft_region = microsoft_region.strip()
        self.source_language = source_language
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> MachineTranslationAPI:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        _full_context: str,
        _translation_memory: str,
        _job_id: str,
    ) -> dict[str, str]:
        texts = [sentence.source_text for sentence in chunk.sentences]
        if self.source_language == "zh":
            raise TranslationError("中文源文本不需要再次翻译为中文。")
        language_codes = MACHINE_TRANSLATION_LANGUAGE_CODES[self.provider]
        source_code = language_codes["en" if self.source_language == "en" else "ja"]
        try:
            if self.provider == "deepl":
                payload: dict[str, object] = {
                    "text": texts,
                    "source_lang": source_code,
                    "target_lang": "ZH-HANS",
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
                translated = [item["text"] for item in response.json()["translations"]]
            elif self.provider == "google_translate":
                response = self.client.post(
                    self.base_url,
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json={
                        "q": texts,
                        "source": source_code,
                        "target": "zh-CN",
                        "format": "text",
                        "model": self.model,
                    },
                )
                response.raise_for_status()
                translated = [
                    html.unescape(item["translatedText"])
                    for item in response.json()["data"]["translations"]
                ]
            elif self.provider == "microsoft_translate":
                headers = {
                    "Ocp-Apim-Subscription-Key": self.api_key,
                    "Content-Type": "application/json; charset=UTF-8",
                }
                if self.microsoft_region:
                    headers["Ocp-Apim-Subscription-Region"] = self.microsoft_region
                response = self.client.post(
                    f"{self.base_url}/translate",
                    params={"api-version": "3.0", "from": source_code, "to": "zh-Hans"},
                    headers=headers,
                    json=[{"Text": text} for text in texts],
                )
                response.raise_for_status()
                translated = [item["translations"][0]["text"] for item in response.json()]
            else:
                raise TranslationError(f"不支持的机器翻译服务：{self.provider}")
        except httpx.HTTPStatusError as exc:
            response = exc.response
            raise TranslationError(
                f"{self.provider} API 请求失败（HTTP {response.status_code}）："
                f"{response.text[:500]}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise TranslationError(f"{self.provider} 返回格式无效：{exc}") from exc
        if len(translated) != len(chunk.sentences) or any(not item.strip() for item in translated):
            raise TranslationError(f"{self.provider} 返回的译文数量或内容无效。")
        return {
            sentence.id: text.strip()
            for sentence, text in zip(chunk.sentences, translated, strict=True)
        }


def translate_sentences(
    sentences: list[Sentence],
    api_key: str,
    model: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    provider: str = "deepseek",
    source_language: SourceLanguage = "ja",
    system_prompt: str = "",
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_output_tokens: int = 16_384,
    deepl_formality: str = "default",
    microsoft_region: str = "",
    send_context: bool = True,
    context_sentences: int = 24,
    memory_sentences: int = 50,
    job_id: str = "asmr_dubber",
    progress: Progress | None = None,
    on_batch: Callable[[], None] | None = None,
    client: httpx.Client | None = None,
    cancel_event: CancellationSignal | None = None,
) -> None:
    check_cancelled(cancel_event)
    pending = [sentence for sentence in sentences if sentence.enabled and not sentence.zh_text]
    if not pending:
        if progress:
            progress("所有句子已有中文翻译", 1, 1)
        return
    if source_language == "zh":
        raise TranslationError("中文源文本不需要再次翻译为中文。")
    batches = _chunks(pending)
    if provider == "deepseek":
        translator: DeepSeekTranslator | LLMTranslator | MachineTranslationAPI = DeepSeekTranslator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            client=client,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            minimum_output_tokens=max_output_tokens,
            source_language=source_language,
        )
    elif provider in {"openai", "anthropic", "gemini", "openai_compatible"}:
        translator = LLMTranslator(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            client=client,
            source_language=source_language,
        )
    elif provider in {"deepl", "google_translate", "microsoft_translate"}:
        translator = MachineTranslationAPI(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            deepl_formality=deepl_formality,
            microsoft_region=microsoft_region,
            client=client,
            source_language=source_language,
        )
    else:
        raise TranslationError(f"未知翻译服务：{provider}")

    cancel_translation = translator.close
    callback_signal = register_cancel_callback(cancel_translation, cancel_event)
    try:
        with translator:
            index = 0
            while index < len(batches):
                check_cancelled(cancel_event)
                chunk = batches[index]
                if progress:
                    progress(
                        f"{provider} · {model} 翻译第 {index + 1}/{len(batches)} 批",
                        index,
                        len(batches),
                    )
                try:
                    translations = translator.translate_chunk(
                        chunk,
                        (
                            _context_for_chunk(sentences, chunk, context_sentences)
                            if send_context
                            else "[]"
                        ),
                        (
                            _bounded_translation_memory(sentences, chunk, memory_sentences)
                            if send_context
                            else "[]"
                        ),
                        job_id,
                    )
                except _OutputLengthTranslationError as exc:
                    if len(chunk.sentences) == 1:
                        raise TranslationError(
                            f"句子 {chunk.sentences[0].id} 即使提高输出预算仍超出长度上限。"
                        ) from exc
                    midpoint = len(chunk.sentences) // 2
                    batches[index : index + 1] = [
                        TranslationChunk(chunk.sentences[:midpoint]),
                        TranslationChunk(chunk.sentences[midpoint:]),
                    ]
                    if progress:
                        progress(
                            f"第 {index + 1} 批输出过长，已自动拆成两个小批次",
                            index,
                            len(batches),
                        )
                    continue
                except TranslationError:
                    check_cancelled(cancel_event)
                    raise
                for sentence in chunk.sentences:
                    translation = translations[sentence.id].strip()
                    sentence.zh_text = translation
                    if translation:
                        sentence.status = "translated"
                    else:
                        sentence.enabled = False
                        sentence.tts_file = None
                        sentence.tts_cache_key = None
                        sentence.tts_duration_seconds = None
                        sentence.status = "skipped_filler"
                    sentence.error = None
                if on_batch:
                    on_batch()
                check_cancelled(cancel_event)
                index += 1
                if progress:
                    progress(f"已翻译 {index}/{len(batches)} 批", index, len(batches))
    finally:
        unregister_cancel_callback(cancel_translation, callback_signal)
