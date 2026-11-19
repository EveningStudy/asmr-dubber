"""LLM-assisted reconciliation of untimed scripts onto ASR timing."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from .constants import DEFAULT_DEEPSEEK_BASE_URL
from .errors import TranslationError
from .languages import SourceLanguage, source_language_label
from .models import Sentence
from .storage import atomic_write_text
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)
from .translation_prompts import _SCRIPT_RECONCILIATION_PROMPT
from .translation_schema import (
    Progress,
    _NonRetryableTranslationError,
    _OutputLengthTranslationError,
)
from .translation_script_spans import _validated_script_spans

logger = logging.getLogger(__name__)

LLM_RECONCILIATION_PROVIDERS = frozenset(
    {
        "deepseek",
        "bailian",
        "doubao",
        "openai",
        "anthropic",
        "gemini",
        "openai_compatible",
        "sensenova",
    }
)


@dataclass(frozen=True)
class ScriptReconciliationBatch:
    recognized: list[Sentence]
    script: list[tuple[str, str]]


def _script_fragments(script_lines: list[str]) -> list[str]:
    """Split long script rows at sentence boundaries before assigning them once."""

    fragments: list[str] = []
    for raw in script_lines:
        line = str(raw or "").strip()
        if not line:
            continue
        sentences = [
            item.strip() for item in re.split(r"(?<=[。！？!?])\s*|(?<=…)\s*", line) if item.strip()
        ]
        for sentence in sentences or [line]:
            if len(sentence) <= 80:
                fragments.append(sentence)
                continue
            clauses = [
                item.strip()
                for item in re.split(r"(?<=[，、,；;：:])\s*", sentence)
                if item.strip()
            ]
            fragments.extend(clauses or [sentence])
    return fragments


def _recognized_script_batches(
    recognized: list[Sentence],
    *,
    maximum_recognized: int = 8,
) -> list[list[Sentence]]:
    if not recognized:
        raise TranslationError("没有可供台本校对的识别结果。")
    return [
        recognized[start : start + maximum_recognized]
        for start in range(0, len(recognized), maximum_recognized)
    ]


def _script_reconciliation_messages(
    batch: ScriptReconciliationBatch,
    *,
    source_language: SourceLanguage,
    target: Literal["source", "zh"],
    attempt: int,
    last_error: str = "",
    consumed_ends: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    output_label = source_language_label(source_language) if target == "source" else "中文"
    retry_note = (
        "" if attempt == 1 else f"这是第 {attempt} 次校验重试。请修正以下具体错误：{last_error}"
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
    script_payload = [
        {
            "id": line_id,
            "text": text,
            "length": len(text),
            "available_start": (consumed_ends or {}).get(line_id, 0),
        }
        for line_id, text in batch.script
    ]
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
            {
                "corrections": [
                    {"id": sentence.id, "script_quotes": [], "text": ""}
                    for sentence in batch.recognized
                ]
            },
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
    extra_body: str = "{}",
    diagnostic_path: Path | None = None,
    script_source: str = "",
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Use an LLM to map an untimed script onto ASR timing without changing time ranges."""
    from . import translation

    if provider not in LLM_RECONCILIATION_PROVIDERS:
        raise TranslationError(
            "台本校对需要大模型翻译服务；请选择 DeepSeek、百炼、豆包、商汤、OpenAI、Claude、"
            "Gemini 或本地/自定义 OpenAI-compatible。"
        )
    fragments = _script_fragments(script_lines)
    if not fragments:
        raise TranslationError("台本中没有可供校对的文字。")
    recognized_batches = _recognized_script_batches(recognized)
    script_items = [(f"p{index:06d}", text) for index, text in enumerate(fragments, start=1)]
    script_lookup = dict(script_items)
    script_position = {script_id: index for index, (script_id, _text) in enumerate(script_items)}
    script_cursor = 0
    consumed_script_ids: set[str] = set()
    consumed_ends: dict[str, int] = {}
    origins = []
    for line_number, line in enumerate(script_lines, start=1):
        origins.extend([line_number] * len(_script_fragments([line])))
    translator = translation.LLMTranslator(
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
        extra_body=extra_body,
    )
    cancel_translation = translator.close
    callback_signal = register_cancel_callback(cancel_translation, cancel_event)
    corrections: dict[str, str] = {}
    report: list[dict[str, object]] = []
    try:
        with translator:
            for index, recognized_batch in enumerate(recognized_batches):
                check_cancelled(cancel_event)
                remaining_batches = len(recognized_batches) - index
                remaining_script = max(1, len(script_items) - script_cursor)
                expected_span = (remaining_script + remaining_batches - 1) // remaining_batches
                window_size = min(48, max(12, expected_span + 8))
                available_script = script_items[script_cursor : script_cursor + window_size]
                batch = ScriptReconciliationBatch(recognized_batch, available_script)
                expected_ids = [sentence.id for sentence in batch.recognized]
                last_error: Exception | None = None
                result: dict[str, str] | None = None
                selected_script_ids: list[str] = []
                new_ends: dict[str, int] = {}
                allocations: list[dict[str, object]] = []
                content = ""
                review_ids: list[str] = []
                for attempt in range(1, 4):
                    check_cancelled(cancel_event)
                    messages = _script_reconciliation_messages(
                        batch,
                        source_language=source_language,
                        target=target,
                        attempt=attempt,
                        last_error=str(last_error or ""),
                        consumed_ends=consumed_ends,
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
                        result, new_ends, allocations = _validated_script_spans(
                            content,
                            batch.recognized,
                            available_script,
                            consumed_ends,
                        )
                        selected_script_ids = list(dict.fromkeys(str(a["id"]) for a in allocations))
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
                    # Transport/auth failures must not be reported as a successful review.
                    if not isinstance(last_error, TranslationError):
                        raise TranslationError(f"台本校对请求失败：{last_error}") from last_error
                    ids_in_error = re.findall(r"\bp\d{6}\b", str(last_error))
                    locations = [
                        f"{key}=导入后第 {origins[script_position[key]]} 行"
                        for key in dict.fromkeys(ids_in_error)
                        if key in script_position
                    ]
                    diagnostic = {
                        "source": script_source,
                        "batch": index + 1,
                        "error": str(last_error),
                        "recognized": [s.model_dump() for s in batch.recognized],
                        "script": [
                            {"id": key, "parsed_line": origins[script_position[key]], "text": text}
                            for key, text in batch.script
                        ],
                        "response": content,
                        "completed_batches": report,
                    }
                    if diagnostic_path is not None:
                        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_text(
                            diagnostic_path, json.dumps(diagnostic, ensure_ascii=False, indent=2)
                        )
                    logger.warning("台本校对降级为逐句重试；位置：%s；详见项目诊断", locations)
                    result, new_ends, allocations = {}, dict(consumed_ends), []
                    for sentence in batch.recognized:
                        check_cancelled(cancel_event)
                        messages = _script_reconciliation_messages(
                            ScriptReconciliationBatch([sentence], available_script),
                            source_language=source_language,
                            target=target,
                            attempt=2,
                            last_error=str(last_error),
                            consumed_ends=new_ends,
                        )
                        translator.system_prompt = messages[0]["content"]
                        try:
                            single_content, limited = translator._request(messages, job_id)
                            if limited:
                                raise _OutputLengthTranslationError(
                                    "台本逐句校对达到输出长度上限。"
                                )
                            single, next_ends, assigned = _validated_script_spans(
                                single_content,
                                [sentence],
                                available_script,
                                new_ends,
                            )
                            if not single.get(sentence.id):
                                raise TranslationError("逐句重试没有可靠匹配")
                        except (_NonRetryableTranslationError, _OutputLengthTranslationError):
                            raise
                        except (
                            TranslationError,
                            httpx.HTTPError,
                            KeyError,
                            IndexError,
                            TypeError,
                            ValueError,
                        ) as single_exc:
                            check_cancelled(cancel_event)
                            review_ids.append(sentence.id)
                            result[sentence.id] = (
                                sentence.source_text if target == "source" else sentence.zh_text
                            )
                            logger.warning(
                                "%s 待人工校对，保留原文（%s）；详见项目诊断",
                                sentence.id,
                                single_exc,
                            )
                        else:
                            result.update(single)
                            new_ends = next_ends
                            allocations.extend(assigned)
                    selected_script_ids = list(dict.fromkeys(str(a["id"]) for a in allocations))
                corrections.update(result)
                consumed_ends = new_ends
                consumed_script_ids.update(
                    key for key in selected_script_ids if new_ends[key] == len(script_lookup[key])
                )
                if selected_script_ids:
                    last_key = selected_script_ids[-1]
                    script_cursor = script_position[last_key] + (last_key in consumed_script_ids)
                report.append(
                    {
                        "recognized_ids": expected_ids,
                        "recognized": [
                            {
                                "id": s.id,
                                "start": s.start_seconds,
                                "end": s.end_seconds,
                                "original": s.source_text,
                            }
                            for s in batch.recognized
                        ],
                        "available_script_ids": [line_id for line_id, _ in batch.script],
                        "consumed_script_ids": selected_script_ids,
                        "corrections": result,
                        "allocations": allocations,
                        "script_source": script_source,
                        "needs_review_ids": review_ids,
                    }
                )
                if on_batch:
                    on_batch()
                if progress:
                    progress(
                        f"台本校对第 {index + 1}/{len(recognized_batches)} 批完成"
                        + (f"；{len(review_ids)} 句待人工校对，已保留原文" if review_ids else ""),
                        index + 1,
                        len(recognized_batches),
                    )
                check_cancelled(cancel_event)
    finally:
        unregister_cancel_callback(cancel_translation, callback_signal)
    unused = [
        script_id for script_id, _text in script_items if script_id not in consumed_script_ids
    ]
    if not any(corrections.values()) and not any(entry.get("needs_review_ids") for entry in report):
        if diagnostic_path is not None:
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                diagnostic_path,
                json.dumps(
                    {"error": "未匹配到任何台本文字", "batches": report, "script": script_items},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        raise TranslationError(
            "台本没有可靠匹配到任何识别区间；未应用空结果，请核对音轨、台本和语言。"
            + (f"详细报告：{diagnostic_path}" if diagnostic_path else "")
        )
    if unused:
        report.append(
            {
                "unused_script_ids": unused,
                "remaining_script": [
                    {
                        "id": key,
                        "start": consumed_ends.get(key, 0),
                        "text": text[consumed_ends.get(key, 0) :],
                    }
                    for key, text in script_items
                    if key in unused
                ],
                "reason": "这些台本片段没有可靠匹配到 ASR 时间区间，需要人工检查",
            }
        )
    return corrections, report
