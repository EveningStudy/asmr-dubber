from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .audio import extract_reference
from .errors import SynthesisError
from .filtering import is_japanese_filler_only
from .hashing import cached_sha256_file
from .models import DubProject, Sentence

STABLE_CLONE_MODES = {
    "stable_reference",
}


@dataclass(frozen=True)
class VoiceReference:
    path: Path
    text: str
    identity: str
    sentence: Sentence | None = None
    emotion_path: Path | None = None
    emotion_identity: str | None = None


def _content_length(text: str) -> int:
    return sum(char.isalnum() or "\u3040" <= char <= "\u30ff" for char in text)


def _available_text(sentence: Sentence) -> str:
    return sentence.ja_text or sentence.zh_text


def shared_reference_sentence(project: DubProject) -> Sentence:
    """Resolve the frozen project-level voice anchor, or select one deterministically."""
    configured = project.settings.tts_reference_sentence_id
    if configured:
        for sentence in project.sentences:
            if sentence.id == configured:
                return sentence

    candidates = [
        sentence
        for sentence in project.sentences
        if _content_length(_available_text(sentence)) >= 4
        and not is_japanese_filler_only(_available_text(sentence))
        and sentence.end_seconds - sentence.start_seconds >= 1.5
    ]
    if not candidates:
        raise SynthesisError("找不到至少 1.5 秒且包含有效台词的片段作为统一声纹参考。")

    def score(sentence: Sentence) -> tuple[float, ...]:
        duration = sentence.end_seconds - sentence.start_seconds
        content = _content_length(_available_text(sentence))
        density = content / duration
        preferred_duration = 1.0 if 5.0 <= duration <= 15.0 else 0.0
        return (
            preferred_duration,
            -abs(duration - 7.0),
            -abs(density - 5.0),
            min(float(content), 40.0),
            -sentence.start_seconds,
        )

    return max(candidates, key=score)


def reference_plan_hash(project: DubProject) -> str:
    if project.settings.tts_reference_source == "external":
        path = Path(project.settings.tts_external_reference_audio).expanduser().resolve()
        if not path.is_file():
            raise SynthesisError(f"找不到设置中的外部参考音频：{path}")
        payload = {
            "source": "external",
            "sha256": cached_sha256_file(path),
            "text": project.settings.tts_external_reference_text,
        }
    else:
        reference = shared_reference_sentence(project)
        payload = {
            "source": "project_sentence",
            "source_sha256": project.source.sha256,
            "sentence_id": reference.id,
            "start": reference.start_seconds,
            "end": reference.end_seconds,
            "ja": reference.ja_text,
            "padding": project.settings.reference_padding_seconds,
        }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prepare_voice_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    sentence: Sentence,
) -> VoiceReference:
    """Return the selected external/shared/per-sentence reference for any TTS backend."""
    if project.settings.tts_reference_source == "external":
        path = Path(project.settings.tts_external_reference_audio).expanduser().resolve()
        if not path.is_file():
            raise SynthesisError(f"找不到设置中的外部参考音频：{path}")
        identity = reference_plan_hash(project)
        return VoiceReference(
            path=path,
            text=project.settings.tts_external_reference_text.strip(),
            identity=f"external:{identity}",
        )

    stable = project.settings.tts_clone_mode in STABLE_CLONE_MODES
    reference_sentence = shared_reference_sentence(project) if stable else sentence
    if stable:
        project.settings.tts_reference_sentence_id = reference_sentence.id
        prefix = "voice_anchor"
        identity = reference_plan_hash(project)
    else:
        prefix = sentence.id
        payload = (
            f"{project.source.sha256}|{sentence.id}|{sentence.start_seconds:.6f}|"
            f"{sentence.end_seconds:.6f}|{sentence.ja_text}|"
            f"{project.settings.reference_padding_seconds:.6f}"
        )
        identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    destination = project_dir / "references" / f"{prefix}_{identity[:16]}.wav"
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.stem}.tmp.wav")
        try:
            extract_reference(
                source=source,
                destination=temporary,
                start_seconds=reference_sentence.start_seconds,
                end_seconds=reference_sentence.end_seconds,
                padding_seconds=project.settings.reference_padding_seconds,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return VoiceReference(
        path=destination,
        text=reference_sentence.ja_text,
        identity=f"project:{identity}",
        sentence=reference_sentence,
    )


def _index_external_reference(path_text: str, *, role: str) -> VoiceReference:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise SynthesisError(f"找不到 IndexTTS2 外部{role}参考音频：{path}")
    identity = cached_sha256_file(path)
    return VoiceReference(
        path=path,
        text="",
        identity=f"index-external-{role}:{identity}",
    )


def _index_sentence_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    reference_sentence: Sentence,
    *,
    role: str,
    shared: bool,
) -> VoiceReference:
    payload = (
        f"{project.source.sha256}|index-{role}|{reference_sentence.id}|"
        f"{reference_sentence.start_seconds:.6f}|{reference_sentence.end_seconds:.6f}|"
        f"{reference_sentence.ja_text}|{project.settings.reference_padding_seconds:.6f}"
    )
    identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    prefix = f"index_{role}_anchor" if shared else f"index_{role}_{reference_sentence.id}"
    destination = project_dir / "references" / f"{prefix}_{identity[:16]}.wav"
    if not destination.is_file():
        temporary = destination.with_name(f".{destination.stem}.tmp.wav")
        try:
            extract_reference(
                source=source,
                destination=temporary,
                start_seconds=reference_sentence.start_seconds,
                end_seconds=reference_sentence.end_seconds,
                padding_seconds=project.settings.reference_padding_seconds,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return VoiceReference(
        path=destination,
        text=reference_sentence.ja_text,
        identity=f"index-{role}:{identity}",
        sentence=reference_sentence,
    )


def prepare_index_speaker_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    sentence: Sentence,
) -> VoiceReference:
    source_id = project.settings.tts_index_speaker_source
    if source_id == "external":
        return _index_external_reference(
            project.settings.tts_external_reference_audio,
            role="speaker",
        )
    shared = source_id == "project_reference"
    reference_sentence = shared_reference_sentence(project) if shared else sentence
    return _index_sentence_reference(
        project,
        project_dir,
        source,
        reference_sentence,
        role="speaker",
        shared=shared,
    )


def prepare_index_emotion_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    sentence: Sentence,
    speaker_reference: VoiceReference,
) -> VoiceReference | None:
    source_id = project.settings.tts_index_emotion_source
    if source_id == "text":
        return None
    if source_id == "speaker_reference":
        return speaker_reference
    if source_id == "external":
        return _index_external_reference(
            project.settings.tts_index_external_emotion_audio,
            role="emotion",
        )
    shared = source_id == "project_reference"
    reference_sentence = shared_reference_sentence(project) if shared else sentence
    if (
        speaker_reference.sentence is not None
        and speaker_reference.sentence.id == reference_sentence.id
    ):
        return speaker_reference
    return _index_sentence_reference(
        project,
        project_dir,
        source,
        reference_sentence,
        role="emotion",
        shared=shared,
    )
