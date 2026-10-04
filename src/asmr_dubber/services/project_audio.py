from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
import uuid
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from .. import pipeline
from ..audio import extract_reference, verify_source
from ..errors import ProjectError
from ..lifecycle import invalidate_outputs
from ..model_registry import TTS_BACKENDS
from ..models import (
    DubProject,
    save_project,
)
from ..platforms import portable_home
from ..user_settings import store_reference_audio
from ..voice_reference import shared_reference_sentence

_EDGE_TTS_PREVIEW_TEXT = "你好，欢迎使用 ASMR Dubber。"

_EDGE_TTS_PREVIEW_LOCK = threading.Lock()

_UI_CLEANUP_LOCK = threading.Lock()

_UI_CLEANUP_TIMES: dict[Path, float] = {}


def ui_stage_directory() -> Path:
    """Return the only local directory exposed to Gradio as an output allowlist."""

    destination = portable_home() / "temp" / "ui"
    destination.mkdir(parents=True, exist_ok=True)
    with _UI_CLEANUP_LOCK:
        now = time.monotonic()
        if now - _UI_CLEANUP_TIMES.get(destination, -float("inf")) < 300:
            return destination
        if len(_UI_CLEANUP_TIMES) >= 32:
            _UI_CLEANUP_TIMES.clear()
        _UI_CLEANUP_TIMES[destination] = now
    cutoff = time.time() - 24 * 3600
    for candidate in destination.rglob("*"):
        try:
            marker = candidate.with_name(candidate.name + ".lease")
            if (
                candidate.is_file()
                and candidate.suffix != ".lease"
                and (marker.stat().st_mtime if marker.is_file() else candidate.stat().st_mtime)
                < cutoff
            ):
                candidate.unlink()
                marker.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def stage_for_ui(
    path: Path | None,
    *,
    category: str = "exports",
    preserve_name: bool = True,
) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    identity = sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[
        :20
    ]
    directory = ui_stage_directory() / category
    directory.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.casefold() or ".bin"
    destination = directory / (
        f"{identity}_{path.name}" if preserve_name else f"{identity}{suffix}"
    )
    if destination.is_file() and destination.stat().st_size == stat.st_size:
        destination.with_name(destination.name + ".lease").touch()
        return str(destination.resolve())
    destination.with_name(destination.name + ".lease").touch()
    try:
        os.link(path, destination)
    except FileExistsError:
        pass
    except OSError:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(path, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination.resolve())


def preview_edge_tts_voice(voice: str) -> str:
    """Create a small cached Edge TTS sample for the settings-page player."""

    voice_id = str(voice or "").strip()
    if not voice_id:
        raise ProjectError("请先选择 Edge TTS 音色。")
    if not all(character.isalnum() or character == "-" for character in voice_id):
        raise ProjectError("Edge TTS 音色 ID 格式无效。")

    preview_text = (
        "Hello. This is an English voice preview."
        if voice_id.startswith("en-")
        else _EDGE_TTS_PREVIEW_TEXT
    )
    spec = TTS_BACKENDS["edge_tts"]
    identity = sha256(f"{spec.default_model}|{voice_id}|{preview_text}".encode()).hexdigest()[:20]
    destination = ui_stage_directory() / "edge-tts-previews" / f"{identity}.mp3"
    destination.parent.mkdir(parents=True, exist_ok=True)

    with _EDGE_TTS_PREVIEW_LOCK:
        if destination.is_file() and destination.stat().st_size > 0:
            return str(destination.resolve())
        temporary = destination.with_name(f".{destination.name}.tmp.mp3")
        temporary.unlink(missing_ok=True)
        try:
            try:
                import edge_tts
            except ImportError as exc:
                raise ProjectError(
                    "Edge TTS 运行依赖缺失；请重新运行 setup 修复基础依赖。"
                ) from exc

            async def save_preview() -> None:
                communicator = edge_tts.Communicate(
                    preview_text,
                    voice=voice_id,
                )
                await communicator.save(str(temporary))

            asyncio.run(save_preview())
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise ProjectError("Edge TTS 没有生成试听音频。")
            temporary.replace(destination)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"Edge TTS 试听失败：{exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination.resolve())


def reference_picker(
    project_path: str, *, include_preview: bool = True
) -> tuple[list[tuple[str, str]], str | None, str | None]:
    """Return project sentence choices and a staged preview for the selected anchor."""

    project, directory = pipeline.reload_project(project_path)
    if not project.sentences:
        return [], None, None

    recommended_id: str | None = None
    with suppress(Exception):
        recommended_id = shared_reference_sentence(project).id
    selected = project.settings.tts_reference_sentence_id or recommended_id
    choices: list[tuple[str, str]] = []
    for sentence in project.sentences:
        duration = sentence.end_seconds - sentence.start_seconds
        prefix = "★ 推荐 · " if sentence.id == recommended_id else ""
        warning = "⚠ 过短 · " if duration < 1.5 else ""
        excerpt = " ".join((sentence.source_text or sentence.zh_text).split())[:42] or "（无文本）"
        choices.append(
            (
                f"{prefix}{warning}{sentence.id} · {duration:.1f}s · {excerpt}",
                sentence.id,
            )
        )
    valid_ids = {value for _, value in choices}
    if selected not in valid_ids:
        selected = choices[0][1]
    return (
        choices,
        selected,
        reference_preview(project, directory, selected) if include_preview else None,
    )


def reference_preview(
    project: DubProject,
    directory: Path,
    sentence_id: str | None,
) -> str | None:
    sentence = next((item for item in project.sentences if item.id == sentence_id), None)
    if sentence is None:
        return None
    source = verify_source(directory, project.source)
    preview_dir = ui_stage_directory() / "reference-previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    identity = sha256(
        (
            f"{project.source.sha256}|{sentence.id}|{sentence.start_seconds:.6f}|"
            f"{sentence.end_seconds:.6f}|{project.settings.reference_padding_seconds:.6f}"
        ).encode()
    ).hexdigest()[:20]
    destination = preview_dir / f"{sentence.id}_{identity}.wav"
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.name}.tmp.wav")
        try:
            extract_reference(
                source,
                temporary,
                sentence.start_seconds,
                sentence.end_seconds,
                project.settings.reference_padding_seconds,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination.resolve())


def preview_reference(project_path: str, sentence_id: str | None) -> str | None:
    project, directory = pipeline.reload_project(project_path)
    return reference_preview(project, directory, sentence_id)


def select_reference(project_path: str, sentence_id: str) -> tuple[str, str | None]:
    project, directory = pipeline.reload_project(project_path)
    if sentence_id and not any(item.id == sentence_id for item in project.sentences):
        raise ProjectError(f"项目中找不到参考句：{sentence_id}")
    project.settings.tts_reference_sentence_id = sentence_id or None
    invalidate_outputs(project)
    save_project(project, directory)
    return (
        f"已把 {sentence_id} 设为项目统一音色参考。" if sentence_id else "已恢复自动选择音色参考。",
        reference_preview(project, directory, sentence_id),
    )


def select_autoflow_project_reference(
    project_path: str,
    sentence_id: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    source_text: str | None = None,
) -> tuple[str, str | None]:
    """Use one analyzed project sentence for the active AutoFlow task."""

    project, directory = pipeline.reload_project(project_path)
    if not any(item.id == sentence_id for item in project.sentences):
        raise ProjectError(f"项目中找不到参考句：{sentence_id}")
    sentence = next(item for item in project.sentences if item.id == sentence_id)
    start = sentence.start_seconds if start_seconds is None else float(start_seconds)
    end = sentence.end_seconds if end_seconds is None else float(end_seconds)
    if not (0 <= start < end <= project.source.duration_seconds):
        raise ProjectError("参考范围必须在源音频内，且结束时间大于开始时间。")
    if (start, end, source_text) != (sentence.start_seconds, sentence.end_seconds, None):
        sentence.start_seconds = start
        sentence.end_seconds = end
        if source_text is not None:
            sentence.source_text = str(source_text).strip()
        sentence.tts_file = None
        sentence.tts_cache_key = None
        invalidate_outputs(project)
    project.settings.tts_reference_source = "project_sentence"
    project.settings.tts_reference_sentence_id = sentence_id
    if project.settings.tts_backend in {"indextts2_5", "indextts2"}:
        project.settings.tts_index_speaker_source = "project_reference"
    save_project(project, directory)
    return (
        f"已为当前批量任务选择项目片段 {sentence_id}。",
        reference_preview(project, directory, sentence_id),
    )


def select_autoflow_external_reference(
    project_path: str,
    audio_path: str | os.PathLike[str],
    *,
    text: str = "",
    language: str = "auto",
) -> tuple[str, str]:
    """Store and use an external voice reference for the active AutoFlow task."""

    if language not in {"auto", "ja", "en", "zh"}:
        raise ProjectError(f"未知外部参考音频语言：{language}")
    stored = store_reference_audio(audio_path)
    project, directory = pipeline.reload_project(project_path)
    project.settings.tts_reference_source = "external"
    project.settings.tts_external_reference_audio = str(stored)
    project.settings.tts_external_reference_text = str(text or "").strip()
    project.settings.tts_external_reference_language = cast(Any, language)
    project.settings.tts_reference_sentence_id = None
    if project.settings.tts_backend in {"indextts2_5", "indextts2"}:
        project.settings.tts_index_speaker_source = "external"
    save_project(project, directory)
    return f"已为当前批量任务导入外部参考音频：{stored.name}", str(stored)
