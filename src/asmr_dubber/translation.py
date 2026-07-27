from __future__ import annotations

import gc
import html
import json
import random
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .environment import resolve_transformers_model_source
from .errors import TranslationError
from .models import Sentence

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


@dataclass(frozen=True)
class TranslationChunk:
    sentences: list[Sentence]


SYSTEM_PROMPT = """你是日语音声、广播剧和 ASMR 的简体中文配音翻译。

输出用于直接合成中文语音，只保留有明确语义或交际作用的内容。

规则：
1. 每个输入 id 必须输出一项并保持原顺序，不得合并、拆分、遗漏或改写 id。
2. 忠实翻译实义台词。中文应自然、简洁、适合朗读，并保持称呼、敬语、语气和专名一致。
3. 如果整句只有笑声、哭声、喘息、呻吟、亲吻声、呼吸声、拟声效果、舞台提示、无意义重复音
   或拉长音，zh 必须是空字符串。不得把声音效果翻成"哈哈哈""啊啊啊""嗯嗯嗯"等中文配音。
4. 如果非语言声音与实义台词混在同一句，只翻译实义台词；删除句首、句中、句尾无语义的
   「あ」「え」「う」「ん」「ふふ」「えーと」等填充音。具有答复、否定、确认、呼唤或惊讶
   等明确交际作用时才翻译。
5. 不因内容含耳语、成人表达或口语省略而弱化、解释或删改实义信息。
6. 不输出注释、括号说明、音效标记、罗马音或日文原文。
7. 只输出严格 JSON：
{"translations":[{"id":"s000001","zh":"中文台词；无实义时为空字符串"}]}
"""


# Hunyuan Hy-MT2 only emits the translation itself, so it cannot follow the
# strict JSON envelope used by the LLM adapters.  These prompts match the
# official ZH<=>XX templates published in the model card, with terminology
# injection reused for translation-memory consistency.  See:
# https://huggingface.co/tencent/Hy-MT2-1.8B
HUNYUAN_TRANSLATION_PROMPT = (
    "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n{source_text}"
)
HUNYUAN_TERMINOLOGY_PROMPT = (
    "参考下面的翻译：\n{terminology}\n\n"
    "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n{source_text}"
)

# Hunyuan Hy-MT2 publishes these recommended generation parameters in its
# model card; they are kept fixed here because the model was tuned for them.
_HUNYUAN_DEFAULT_TEMPERATURE = 0.7
_HUNYUAN_DEFAULT_TOP_P = 0.6
_HUNYUAN_DEFAULT_TOP_K = 20
_HUNYUAN_DEFAULT_REPETITION_PENALTY = 1.05

# Pure non-verbal Japanese lines should not produce a Chinese line.  Hy-MT2 is
# a translation model and will otherwise emit a literal transcription such as
# "呵呵呵" for "ふふふ", which the dubbing pipeline would then synthesise.
# This matches the LLM SYSTEM_PROMPT rule that non-speech gets an empty zh.
_FILLER_ONLY_PATTERN = re.compile(
    r"^[\s\u3000]*("
    r"\u3042\u3063|\u3042\u30fc|\u3042\u306f\u306f+|\u3042\u306f+"
    r"|\u3046\u3063|\u3046\u30fc|\u3046\u3093+|\u3046\u308f+"
    r"|\u3048\u3063|\u3048\u30fc"
    r"|\u304a\u3063|\u304a\u30fc|\u304a\u3044"
    r"|\u304b\u306f+|\u304f\u3063|\u304f\u30fc"
    r"|\u3050\u3063|\u3050\u306c+|\u3050\u30fc"
    r"|\u3053\u3053\u3053+|\u3051\u3063|\u3051\u30fc"
    r"|\u3053\u3093\u3053\u3093+|\u3053\u30fc\u3093"
    r"|\u3055\u3063|\u3057\u3045+|\u3057\u30fc+"
    r"|\u3059\u3063|\u3059\u30fc+|\u3059\u3093\u3081+"
    r"|\u305d\u30fc+|\u305d\u3063"
    r"|\u305f\u3063|\u3066\u3044\u3063|\u3066\u3044+"
    r"|\u306a\u304b+|\u306a\u30fc+"
    r"|\u306b\u3083\u30fc+|\u306b\u3083\u3063|\u306b\u30fc+"
    r"|\u306d\u30fc+|\u306d\u3048+|\u306d\u3063"
    r"|\u306f\u3063|\u306f\u3066+|\u306f\u30fc+|\u306f\u3044+"
    r"|\u3072\u3083\u30fc+|\u3072\u3083\u3063"
    r"|\u3072\u3063|\u3072\u30fc+|\u3072\u3044+"
    r"|\u3075\u30fc+|\u3075\u3045+|\u3075\u3063|\u3075\u3066+|\u3075\u3093+|\u3075\u3080+"
    r"|\u3078\u30fc+|\u3078\u3063"
    r"|\u307b\u30fc+|\u307b\u3063|\u307b\u308d+"
    r"|\u307e\u30fc+|\u307e\u3063"
    r"|\u3080\u3063|\u3080\u30fc+|\u3080\u308b+"
    r"|\u3081\u30fc+|\u3081\u3063"
    r"|\u3082\u3046+|\u3082\u3063"
    r"|\u3084\u3063|\u3084\u30fc+|\u3084\u308c+"
    r"|\u3086\u3063|\u3086\u30fc+"
    r"|\u3088\u3063|\u3088\u30fc+|\u3088\u3044+"
    r"|\u308f\u30fc+|\u308f\u3063"
    r"|\u3093\u30fc+|\u3093\u3063"
    r"|\u3093\u3082+|\u3046\u3093+"
    r"|[\u3041-\u309f]{0,3}[\u3002\u3001\uff01\uff1f\u3002]*"
    r")[\s\u3000\u3002\u3001\uff01\uff1f\u301c\uff5e\u2026]*$"
)


def _is_fillter_only(ja_text: str) -> bool:
    """Return True for lines that are only non-verbal sounds or punctuation."""
    if not ja_text.strip():
        return True
    return bool(_FILLER_ONLY_PATTERN.match(ja_text.strip()))


def _cleanup_cuda() -> None:
    """Free GPU memory after a local Transformers session, mirroring ASR/TTS."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


class HunyuanTranslator:
    """Local Transformers adapter for Tencent Hunyuan Hy-MT2.

    Loads the model like the bundled ASR/TTS backends (resolve_transformers_model_source
    + AutoModelForCausalLM) and translates each sentence individually, because the
    model is trained to emit only the translation rather than a JSON envelope.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        temperature: float = _HUNYUAN_DEFAULT_TEMPERATURE,
        top_p: float = _HUNYUAN_DEFAULT_TOP_P,
        top_k: int = _HUNYUAN_DEFAULT_TOP_K,
        repetition_penalty: float = _HUNYUAN_DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = 4_096,
    ) -> None:
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.model_id = model.strip()
        self.device = device.strip() or "cpu"
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.max_new_tokens = int(max_new_tokens)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                "缺少 transformers 或 torch；请在“设备与模型”页安装本地默认运行时。"
            ) from exc
        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            try:
                if not torch.cuda.is_available():
                    raise TranslationError(
                        "未检测到 PyTorch CUDA；请在设置中将翻译设备切换为 cpu，"
                        "或重新安装带 CUDA 的 PyTorch。"
                    )
            except (AttributeError, RuntimeError) as exc:
                raise TranslationError(f"无法检测 CUDA 设备：{exc}") from exc
        torch.set_float32_matmul_precision("high")
        source, revision = resolve_transformers_model_source(self.model_id)
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "cuda:0" if use_cuda else "cpu"
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                revision=revision,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                source,
                revision=revision,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self._torch = torch
        except (OSError, ValueError, RuntimeError) as exc:
            raise TranslationError(
                f"无法加载 Hunyuan Hy-MT2 翻译模型 {self.model_id}：{exc}"
            ) from exc

    def _generate(self, source_text: str, terminology: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        if terminology.strip():
            prompt = HUNYUAN_TERMINOLOGY_PROMPT.format(
                terminology=terminology,
                source_text=source_text,
            )
        else:
            prompt = HUNYUAN_TRANSLATION_PROMPT.format(source_text=source_text)
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self._model.device)
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        # Slice off the prompt tokens before decoding so the model's own
        # echoed instructions are not mistaken for the translation.
        if isinstance(inputs, dict):
            input_length = inputs["input_ids"].shape[-1]
        else:
            input_length = inputs.shape[-1]
        generated = outputs[0][input_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def translate_chunk(
        self,
        chunk: TranslationChunk,
        _full_context: str,
        translation_memory: str,
        _job_id: str,
    ) -> dict[str, str]:
        self._load()
        # Reuse confirmed translations as Hy-MT2 terminology so character names
        # and tone stay consistent with the LLM provider contract.
        terminology = translation_memory.strip() if translation_memory != "[]" else ""
        try:
            results: dict[str, str] = {}
            for sentence in chunk.sentences:
                if _is_fillter_only(sentence.ja_text):
                    results[sentence.id] = ""
                    continue
                translation = self._generate(sentence.ja_text, terminology)
                results[sentence.id] = translation
            return results
        except TranslationError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise TranslationError(f"Hunyuan Hy-MT2 翻译失败：{exc}") from exc

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None
        _cleanup_cuda()

    def __enter__(self) -> HunyuanTranslator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _chunks(
    sentences: list[Sentence],
    max_lines: int = 16,
    max_chars: int = 4_000,
) -> list[TranslationChunk]:
    result: list[TranslationChunk] = []
    current: list[Sentence] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.ja_text)
        if current and (len(current) >= max_lines or count + length > max_chars):
            result.append(TranslationChunk(current))
            current = []
            count = 0
        current.append(sentence)
        count += length
    if current:
        result.append(TranslationChunk(current))
    return result


def _json_lines(sentences: Iterable[Sentence], field: str = "ja") -> str:
    items = []
    for sentence in sentences:
        item = {"id": sentence.id, "ja": sentence.ja_text}
        if field == "both" and sentence.zh_text:
            item["existing_zh"] = sentence.zh_text
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _translation_memory(sentences: Iterable[Sentence]) -> str:
    items = [
        {"id": sentence.id, "ja": sentence.ja_text, "zh": sentence.zh_text}
        for sentence in sentences
        if sentence.zh_text
    ]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


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


class DeepSeekTranslator:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 900.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        top_p: float = 1.0,
        minimum_output_tokens: int = 16_384,
    ) -> None:
        if not api_key.strip():
            raise TranslationError("缺少 DeepSeek API Key。请在 UI 中填写或设置 DEEPSEEK_API_KEY。")
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.system_prompt = system_prompt.strip() or SYSTEM_PROMPT
        self.temperature = temperature
        self.top_p = top_p
        self.minimum_output_tokens = minimum_output_tokens
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
                sum(len(item.ja_text) for item in chunk.sentences) * 4 + 4_096,
            ),
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            correction = (
                ""
                if attempt == 1
                else f"\n这是第 {attempt} 次校验重试：务必返回全部 id 和原顺序；无实义项用空 zh。"
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": "完整日文转写，仅供保持人物称谓和上下文一致：\n" + full_context,
                },
            ]
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
                    "content": "请翻译以下目标项并输出 json：\n" + target + correction,
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
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        timeout_seconds: float = 900.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip() and provider != "openai_compatible":
            raise TranslationError(f"{provider} 缺少 API Key。")
        if not model.strip():
            raise TranslationError("翻译模型 ID 不能为空。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt.strip() or SYSTEM_PROMPT
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
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
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "完整日文转写，仅供保持人物称谓和上下文一致：\n" + full_context,
            },
        ]
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
        if self.provider in {"openai", "openai_compatible"}:
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
    ) -> None:
        if not api_key.strip():
            raise TranslationError(f"{provider} 缺少 API Key。")
        self.provider = provider
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.deepl_formality = deepl_formality
        self.microsoft_region = microsoft_region.strip()
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
        texts = [sentence.ja_text for sentence in chunk.sentences]
        try:
            if self.provider == "deepl":
                payload: dict[str, object] = {
                    "text": texts,
                    "source_lang": "JA",
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
                        "source": "ja",
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
                    params={"api-version": "3.0", "from": "ja", "to": "zh-Hans"},
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
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_output_tokens: int = 16_384,
    deepl_formality: str = "default",
    microsoft_region: str = "",
    job_id: str = "asmr_dubber",
    progress: Progress | None = None,
    on_batch: Callable[[], None] | None = None,
    client: httpx.Client | None = None,
    device: str = "cuda",
) -> None:
    pending = [sentence for sentence in sentences if sentence.enabled and not sentence.zh_text]
    if not pending:
        if progress:
            progress("所有句子已有中文翻译", 1, 1)
        return
    batches = _chunks(pending)
    # This exact prefix is reused for every batch, enabling DeepSeek's prefix cache.
    full_context = _json_lines(sentences)
    if provider == "deepseek":
        translator: (
            DeepSeekTranslator
            | LLMTranslator
            | MachineTranslationAPI
            | HunyuanTranslator
        ) = (
            DeepSeekTranslator(
                api_key=api_key,
                model=model,
                base_url=base_url,
                client=client,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                minimum_output_tokens=max_output_tokens,
            )
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
        )
    elif provider == "hunyuan_mt":
        translator = HunyuanTranslator(
            model=model,
            device=device,
            max_new_tokens=max_output_tokens,
        )
    else:
        raise TranslationError(f"未知翻译服务：{provider}")

    with translator:
        index = 0
        while index < len(batches):
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
                    full_context,
                    _translation_memory(sentences),
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
            index += 1
            if progress:
                progress(f"已翻译 {index}/{len(batches)} 批", index, len(batches))
