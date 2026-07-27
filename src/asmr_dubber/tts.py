from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from .errors import SynthesisError
from .hashing import cached_sha256_file
from .models import DubProject, Sentence
from .voice_reference import (
    STABLE_CLONE_MODES,
    reference_plan_hash,
)
from .voice_reference import (
    shared_reference_sentence as _shared_reference_sentence,
)

Progress = Callable[[str, int, int], None]
_STABLE_CLONE_MODES = STABLE_CLONE_MODES


def shared_reference_sentence(project: DubProject) -> Sentence:
    return _shared_reference_sentence(project)


def shared_reference_plan_hash(project: DubProject) -> str:
    return reference_plan_hash(project)


def _index_reference_payload(project: DubProject, sentence: Sentence) -> dict[str, object]:
    references: dict[str, object] = {
        "sentence_reference": {
            "id": sentence.id,
            "start": sentence.start_seconds,
            "end": sentence.end_seconds,
            "ja": sentence.ja_text,
        }
    }
    speaker_source = project.settings.tts_index_speaker_source
    emotion_source = project.settings.tts_index_emotion_source
    if "project_reference" in {speaker_source, emotion_source}:
        shared = shared_reference_sentence(project)
        references["project_reference"] = {
            "id": shared.id,
            "start": shared.start_seconds,
            "end": shared.end_seconds,
            "ja": shared.ja_text,
        }

    payload: dict[str, object] = {}
    if speaker_source == "external":
        path = Path(project.settings.tts_external_reference_audio).expanduser()
        if not path.is_file():
            raise SynthesisError(f"找不到 IndexTTS2 外部音色参考：{path}")
        payload["speaker"] = cached_sha256_file(path)
    else:
        payload["speaker"] = references[speaker_source]

    if emotion_source == "external":
        path = Path(project.settings.tts_index_external_emotion_audio).expanduser()
        if not path.is_file():
            raise SynthesisError(f"找不到 IndexTTS2 外部情绪参考：{path}")
        payload["emotion"] = cached_sha256_file(path)
    elif emotion_source in references:
        payload["emotion"] = references[emotion_source]
    else:
        payload["emotion"] = emotion_source
    return payload


def tts_cache_key(project: DubProject, sentence: Sentence) -> str:
    """Return a stable cache key using one cached digest per external reference."""

    settings = project.settings
    payload: dict[str, object] = {
        "source_sha256": project.source.sha256,
        "backend": settings.tts_backend,
        "model": settings.tts_model,
        "reference_source": settings.tts_reference_source,
        "clone_mode": settings.tts_clone_mode,
        "device": settings.tts_device,
        "speed": settings.tts_speed,
        "temperature": settings.tts_temperature,
        "top_p": settings.tts_top_p,
        "api_base_url": settings.tts_api_base_url,
        "model_path": settings.tts_model_path,
        "config_path": settings.tts_config_path,
        "executable": settings.tts_executable,
        "index_fp16": settings.tts_index_use_fp16,
        "index_emo_alpha": settings.tts_index_emo_alpha,
        "index_speaker_source": settings.tts_index_speaker_source,
        "index_emotion_source": settings.tts_index_emotion_source,
        "index_emo_text": settings.tts_index_emo_text,
        "gpt_top_k": settings.tts_gpt_top_k,
        "gpt_split": settings.tts_gpt_text_split_method,
        "gpt_sample_steps": settings.tts_gpt_sample_steps,
        "cosyvoice_mode": settings.tts_cosyvoice_mode,
        "zh": sentence.zh_text,
        "sentence_id": sentence.id,
        "implementation": "supported-backends-v3",
    }
    if settings.tts_backend == "indextts2":
        payload["index_references"] = _index_reference_payload(project, sentence)
        payload["reference_padding"] = settings.reference_padding_seconds
    elif (
        settings.tts_reference_source == "external"
        or settings.tts_clone_mode in _STABLE_CLONE_MODES
    ):
        payload["reference_plan"] = shared_reference_plan_hash(project)
        payload["seed"] = settings.random_seed
    else:
        payload["reference_plan"] = {
            "start": sentence.start_seconds,
            "end": sentence.end_seconds,
            "ja": sentence.ja_text,
            "padding": settings.reference_padding_seconds,
        }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def synthesize_sentences(
    project: DubProject,
    project_dir: Path,
    source: Path,
    force: bool = False,
    sentence_ids: Iterable[str] | None = None,
    progress: Progress | None = None,
    on_sentence: Callable[[], None] | None = None,
) -> list[str]:
    from .tts_backends import synthesize_with_selected_backend

    return synthesize_with_selected_backend(
        project,
        project_dir,
        source,
        force=force,
        sentence_ids=sentence_ids,
        progress=progress,
        on_sentence=on_sentence,
    )
