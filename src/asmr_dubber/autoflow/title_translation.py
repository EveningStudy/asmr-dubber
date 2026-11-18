"""Folder and track title translation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import plan_files, runlog
from .domain import (
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .work_state import task_key

TITLE_TRANSLATION_PROMPT = """你负责把日语作品文件夹名称和音频曲目标题翻译成自然、简洁的简体中文。
保留人物名、编号、括号、符号和作品专有名词，不要省略成人内容，不要解释。
如果输入只是 RJ、VJ、BJ 等作品编号，译文原样保留该编号。
每个输入 id 必须输出一项，顺序和 id 必须完全一致，译文不得为空。
只输出严格 JSON：
{"translations":[{"id":"title0001","zh":"中文标题"}]}"""


def identifier_only_name(value: str) -> bool:
    """Return whether a folder name is only a catalogue/product identifier."""

    return bool(
        re.fullmatch(
            r"(?i)(?:rj|vj|bj|cien|ci-en)?[\s._-]*\d+",
            value.strip(),
        )
    )


def translate_titles(
    state: dict[str, Any],
    paths: ToolPaths,
) -> dict[str, str]:
    try:
        from asmr_dubber.models import Sentence
        from asmr_dubber.translation import translate_sentences
        from asmr_dubber.user_settings import (
            PROVIDER_PRESETS,
            load_user_settings,
            resolve_api_key,
        )
    except Exception as exc:
        raise VideoPreparerError(f"无法加载 ASMR Dubber 的翻译工具：{exc}") from exc

    try:
        settings = load_user_settings()
        provider = settings.translation_provider
        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            raise VideoPreparerError(f"ASMR Dubber 中选择了未知翻译服务：{provider}")
        model = settings.translation_model or str(preset.get("default_model") or "")
        base_url = settings.translation_base_url.strip() or str(preset.get("base_url") or "")
        temperature = settings.translation_temperature
        top_p = settings.translation_top_p
        max_tokens = settings.translation_max_output_tokens
        api_key = resolve_api_key(provider)
    except Exception as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 的翻译设置或密钥：{exc}") from exc

    cached = {
        str(key): str(value)
        for key, value in (state.get("title_translations") or {}).items()
        if str(value).strip()
    }
    folder_name_original = str(
        state.get("folder_name_original") or Path(state["source_folder"]).name
    ).strip()
    existing_folder_translation = str(state.get("folder_name_translation") or "").strip()
    if identifier_only_name(folder_name_original) and not existing_folder_translation:
        existing_folder_translation = folder_name_original
    folder_sentence = Sentence(
        id="folder0000",
        start_seconds=0.0,
        end_seconds=1.0,
        source_text=folder_name_original,
        target_text=existing_folder_translation,
    )
    sentences = [folder_sentence]
    for index, item in enumerate(state["timeline"], start=1):
        filename = str(item.get("relative_path") or item["filename"])
        sentences.append(
            Sentence(
                id=f"title{index:04d}",
                start_seconds=float(index),
                end_seconds=float(index + 1),
                source_text=str(item["title_ja"]),
                target_text=cached.get(filename, ""),
            )
        )
    source_languages = {
        str(item.get("source_language") or "ja").casefold() for item in state["timeline"]
    }
    source_language = "en" if source_languages == {"en"} else "ja"

    try:
        translate_sentences(
            sentences,
            api_key=api_key,
            provider=provider,
            source_language=source_language,
            model=model,
            base_url=base_url,
            system_prompt=TITLE_TRANSLATION_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            extra_body=settings.translation_extra_body,
            deepl_formality=settings.translation_deepl_formality,
            microsoft_region=settings.translation_microsoft_region,
            send_context=True,
            context_sentences=100,
            memory_sentences=50,
            job_id=f"video_preparer_{task_key(Path(state['source_folder']))}",
            progress=lambda message, current, total: runlog.emit(f"  {message}"),
        )
    except Exception as exc:
        if folder_sentence.zh_text.strip():
            state["folder_name_translation"] = folder_sentence.zh_text.strip()
        for item, sentence in zip(state["timeline"], sentences[1:], strict=True):
            if sentence.zh_text.strip():
                key = str(item.get("relative_path") or item["filename"])
                cached[key] = sentence.zh_text.strip()
        state["title_translations"] = cached
        raise VideoPreparerError(f"文件夹名称与短音频标题翻译失败：{exc}") from exc

    translated: dict[str, str] = {}
    missing: list[str] = []
    folder_name_translation = folder_sentence.zh_text.strip() or folder_name_original
    state["folder_name_original"] = folder_name_original
    state["folder_name_translation"] = folder_name_translation
    for item, sentence in zip(state["timeline"], sentences[1:], strict=True):
        title = sentence.zh_text.strip()
        key = str(item.get("relative_path") or item["filename"])
        if (
            not title
            and sentence.source_text.strip()
            and not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", sentence.source_text)
        ):
            title = sentence.source_text.strip()
            runlog.emit(f"标题翻译为空，保留原名称：{key} → {title}")
        if not title:
            missing.append(key)
            continue
        translated[key] = title
    state["title_translations"] = translated
    if missing:
        raise VideoPreparerError("以下名称或标题翻译为空：" + "、".join(missing))
    return translated


def translated_plan_titles(
    plan_id: str,
    source_folder: Path,
    sources: list[AudioSource],
    paths: ToolPaths,
    *,
    translate_work_title: bool = True,
    translate_track_titles: bool = True,
) -> tuple[str, dict[str, str]]:
    cached = plan_files.load_plan_metadata(plan_id)
    expected_keys = [item.relative_path or item.path.name for item in sources]
    original_titles = {
        key: (source.title_ja.strip() or Path(key).stem)
        for key, source in zip(expected_keys, sources, strict=True)
    }
    policy = {
        "work": bool(translate_work_title),
        "tracks": bool(translate_track_titles),
    }
    cached_policy = cached.get("title_translation_policy")
    policy_matches = not isinstance(cached_policy, dict) or cached_policy == policy
    translations = (
        {
            str(key): str(value)
            for key, value in (cached.get("title_translations") or {}).items()
            if str(value).strip()
        }
        if policy_matches and translate_track_titles
        else dict(original_titles)
        if not translate_track_titles
        else {}
    )
    folder_translation = (
        str(cached.get("folder_name_translation") or "").strip()
        if policy_matches and translate_work_title
        else source_folder.name
        if not translate_work_title
        else ""
    )
    inferred_folder_translation = False
    if not folder_translation and identifier_only_name(source_folder.name):
        folder_translation = source_folder.name
        inferred_folder_translation = True
    if folder_translation and all(
        str(translations.get(key) or "").strip() for key in expected_keys
    ):
        if inferred_folder_translation or cached_policy != policy:
            updated_cache = dict(cached)
            updated_cache["folder_name_original"] = source_folder.name
            updated_cache["folder_name_translation"] = folder_translation
            updated_cache["title_translations"] = translations
            updated_cache["title_translation_policy"] = policy
            plan_files.save_plan_metadata(plan_id, updated_cache)
        return folder_translation, translations

    state: dict[str, Any] = {
        "source_folder": str(source_folder),
        "folder_name_original": source_folder.name,
        "folder_name_translation": folder_translation,
        "title_translations": translations,
        "timeline": [
            {
                "filename": item.path.name,
                "relative_path": item.relative_path or item.path.name,
                "title_ja": item.title_ja,
                "source_language": item.source_language,
            }
            for item in sources
        ],
    }
    try:
        translations = translate_titles(state, paths)
    except VideoPreparerError:
        plan_files.save_plan_metadata(
            plan_id,
            {
                "source_folder": str(source_folder),
                "folder_name_original": source_folder.name,
                "folder_name_translation": str(state.get("folder_name_translation") or ""),
                "title_translations": dict(state.get("title_translations") or {}),
                "title_translation_policy": policy,
            },
        )
        raise
    folder_translation = str(state["folder_name_translation"])
    plan_files.save_plan_metadata(
        plan_id,
        {
            "source_folder": str(source_folder),
            "folder_name_original": source_folder.name,
            "folder_name_translation": folder_translation,
            "title_translations": translations,
            "title_translation_policy": policy,
        },
    )
    return folder_translation, translations
