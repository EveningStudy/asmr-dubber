"""Per-speaker reference profile assembly and override mutations."""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from .. import pipeline
from ..audio import probe_audio
from ..errors import ProjectError
from ..lifecycle import invalidate_outputs
from ..models import DubProject, SpeakerReference, save_project
from ..user_settings import store_reference_audio
from ..voice_reference import speaker_reference_sentence
from .media import discard_upload

_CANONICAL_SPEAKER = re.compile(r"^S(\d+)$", re.I)


def speaker_sort_key(speaker_id: str) -> tuple[int, int, str]:
    match = _CANONICAL_SPEAKER.fullmatch(speaker_id.strip())
    if match:
        return (0, int(match.group(1)), speaker_id)
    return (1, 0, speaker_id.casefold())


def _sentence_choice_label(
    sentence: Any,
    *,
    tag: str = "",
    recommended: bool = False,
) -> tuple[str, str]:
    duration = sentence.end_seconds - sentence.start_seconds
    prefix = f"[{tag}] " if tag else ("★ 推荐 · " if recommended else "")
    warning = "⚠ 过短 · " if duration < 1.5 else ""
    excerpt = " ".join((sentence.source_text or sentence.zh_text).split())[:42] or "（无文本）"
    return (f"{prefix}{warning}{sentence.id} · {duration:.1f}s · {excerpt}", sentence.id)


def speaker_voice_profiles(
    project: DubProject,
    media_register: Callable[[Path], str | None],
) -> list[dict[str, Any]]:
    by_id: dict[str, list[Any]] = {}
    for sentence in project.sentences:
        spk = (sentence.speaker_id or "").strip()
        if spk:
            by_id.setdefault(spk, []).append(sentence)
    all_speakers = sorted(
        set(by_id) | {key.strip() for key in project.speaker_references if key.strip()},
        key=speaker_sort_key,
    )
    if not all_speakers:
        return []

    profiles: list[dict[str, Any]] = []
    for spk in all_speakers:
        own_sentences = by_id.get(spk, [])
        sample = next(
            (
                item
                for item in own_sentences
                if item.enabled and (item.source_text.strip() or item.zh_text.strip())
            ),
            None,
        )
        auto_id: str | None = None
        if sample is not None:
            with suppress(Exception):
                auto_id = speaker_reference_sentence(project, sample, ignore_override=True)[0].id

        override = project.speaker_references.get(spk)
        mode = "auto"
        sentence_id = auto_id
        audio_path = ""
        audio_name = ""
        audio_url: str | None = None
        text = ""
        language = "auto"
        if override is not None:
            if override.source == "external" and override.audio_path.strip():
                mode = "external"
                audio_path = override.audio_path.strip()
                resolved = Path(audio_path).expanduser().resolve()
                audio_name = override.audio_label.strip() or resolved.name
                text = override.text
                language = override.language
                if resolved.is_file():
                    with suppress(Exception):
                        probe_audio(resolved)
                        audio_url = media_register(resolved)
            elif override.source == "project_sentence" and override.sentence_id:
                if any(item.id == override.sentence_id for item in project.sentences):
                    mode = "project_sentence"
                    sentence_id = override.sentence_id

        own_choices = [
            _sentence_choice_label(item, recommended=(item.id == auto_id)) for item in own_sentences
        ]
        other_choices = [
            _sentence_choice_label(item, tag=(item.speaker_id or "未标注"))
            for item in project.sentences
            if (item.speaker_id or "").strip() != spk
        ]
        profiles.append(
            {
                "speaker_id": spk,
                "sentence_count": len(own_sentences),
                "enabled_count": sum(1 for item in own_sentences if item.enabled),
                "mode": mode,
                "sentence_id": sentence_id,
                "auto_sentence_id": auto_id,
                "audio_path": audio_path,
                "audio_name": audio_name,
                "audio_url": audio_url,
                "text": text,
                "language": language,
                "choices": own_choices + other_choices,
            }
        )
    return profiles


def set_speaker_reference(
    project_path: str,
    *,
    speaker_id: str,
    mode: str,
    sentence: str = "",
    external: str = "",
    external_name: str = "",
    text: str = "",
    language: str = "auto",
) -> str:
    spk = str(speaker_id or "").strip()
    if not spk or len(spk) > 32:
        raise ProjectError("说话人编号不能为空且不能超过 32 个字符。")
    if mode not in {"auto", "project_sentence", "external"}:
        raise ProjectError(f"未知的参考音色模式：{mode}")
    if language not in {"auto", "ja", "en", "zh"}:
        raise ProjectError(f"未知外部参考音频语言：{language}")
    project, directory = pipeline.reload_project(project_path)
    if mode == "auto":
        project.speaker_references.pop(spk, None)
        message = f"已恢复 {spk} 自动选择参考音色。"
    elif mode == "project_sentence":
        target = str(sentence or "").strip()
        if not target or not any(item.id == target for item in project.sentences):
            raise ProjectError(f"项目中找不到参考句：{target}")
        project.speaker_references[spk] = SpeakerReference(
            source="project_sentence",
            sentence_id=target,
        )
        message = f"已将 {target} 设为 {spk} 的参考音色。"
    else:
        raw_external = str(external or "").strip()
        if not raw_external:
            raise ProjectError(f"请先为 {spk} 选择外部参考音频。")
        existing = project.speaker_references.get(spk)
        ext_path = Path(raw_external).expanduser().resolve()
        if (
            existing is not None
            and existing.source == "external"
            and existing.audio_path.strip()
            and Path(existing.audio_path).expanduser().resolve() == ext_path
            and ext_path.is_file()
        ):
            stored = ext_path
            label = external_name.strip() or existing.audio_label.strip() or stored.name
        else:
            stored = store_reference_audio(raw_external)
            discard_upload(raw_external)
            label = external_name.strip() or stored.name
        project.speaker_references[spk] = SpeakerReference(
            source="external",
            audio_path=str(stored),
            audio_label=label,
            text=str(text or "").strip(),
            language=cast(Any, language),
        )
        message = f"已为 {spk} 指定外部参考音色：{label}"

    for item in project.sentences:
        if (item.speaker_id or "").strip() != spk:
            continue
        item.tts_file = None
        item.tts_cache_key = None
        item.tts_warning = None
        item.tts_candidates = []
        item.tts_duration_seconds = None
        item.reference_file = None
        if item.status in {"synthesized", "error"}:
            item.status = "translated" if item.zh_text else "pending"
        item.clear_error()
    invalidate_outputs(project)
    save_project(project, directory)
    return message
