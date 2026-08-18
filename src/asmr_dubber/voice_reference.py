from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .audio import extract_reference, probe_audio
from .errors import SynthesisError
from .hashing import cached_sha256_file
from .languages import SourceLanguage
from .models import DubProject, Sentence

STABLE_CLONE_MODES = {
    "stable_reference",
}

_INDEX_REFERENCE_MIN_SECONDS = 1.0
_INDEX_REFERENCE_CACHE_VERSION = "index-reference-v2-min-1s"
_INDEX_REFERENCE_DURATION_TOLERANCE = 0.01


@dataclass(frozen=True)
class VoiceReference:
    path: Path
    text: str
    identity: str
    language: SourceLanguage = "ja"
    sentence: Sentence | None = None
    emotion_path: Path | None = None
    emotion_identity: str | None = None


def shared_reference_sentence(project: DubProject) -> Sentence:
    """Resolve the frozen project-level voice anchor, or select one deterministically."""
    configured = project.settings.tts_reference_sentence_id
    if configured:
        for sentence in project.sentences:
            if sentence.id == configured:
                return sentence

    candidates = [sentence for sentence in project.sentences if sentence.source_text]
    if not candidates:
        candidates = [sentence for sentence in project.sentences if sentence.zh_text]
    if not candidates:
        raise SynthesisError("找不到包含源文或中文台词的片段作为统一声纹参考。")

    return max(
        candidates,
        key=lambda sentence: (
            sentence.end_seconds - sentence.start_seconds,
            -sentence.start_seconds,
        ),
    )


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
        if (
            project.settings.tts_external_reference_language != "auto"
            or project.source_language != "ja"
        ):
            payload["language"] = (
                project.source_language
                if project.settings.tts_external_reference_language == "auto"
                else project.settings.tts_external_reference_language
            )
    else:
        reference = shared_reference_sentence(project)
        payload = {
            "source": "project_sentence",
            "source_sha256": project.source.sha256,
            "sentence_id": reference.id,
            "start": reference.start_seconds,
            "end": reference.end_seconds,
            "ja": reference.source_text,
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
            language=(
                project.source_language
                if project.settings.tts_external_reference_language == "auto"
                else project.settings.tts_external_reference_language
            ),
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
            f"{sentence.end_seconds:.6f}|{sentence.source_text}|"
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
        text=reference_sentence.source_text or reference_sentence.zh_text,
        identity=f"project:{identity}",
        language=project.source_language if reference_sentence.source_text else "zh",
        sentence=reference_sentence,
    )


def _index_external_reference(path_text: str, *, role: str) -> VoiceReference:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise SynthesisError(f"找不到 IndexTTS2 外部{role}参考音频：{path}")
    identity = cached_sha256_file(path)
    try:
        duration = probe_audio(path, sha256=identity).duration_seconds
    except Exception as exc:
        raise SynthesisError(f"无法读取 IndexTTS2 外部{role}参考音频：{path}: {exc}") from exc
    if duration + _INDEX_REFERENCE_DURATION_TOLERANCE < _INDEX_REFERENCE_MIN_SECONDS:
        raise SynthesisError(
            f"IndexTTS2 外部{role}参考音频只有 {duration:.2f} 秒；"
            f"请使用至少 {_INDEX_REFERENCE_MIN_SECONDS:g} 秒的音频。"
        )
    return VoiceReference(
        path=path,
        text="",
        identity=f"index-external-{role}:{identity}",
        language="zh",
    )


def _index_reference_bounds(
    project: DubProject,
    reference_sentence: Sentence,
) -> tuple[float, float]:
    """Expand an IndexTTS2 reference inside the source without changing the sentence."""

    source_duration = float(project.source.duration_seconds)
    if source_duration + _INDEX_REFERENCE_DURATION_TOLERANCE < _INDEX_REFERENCE_MIN_SECONDS:
        raise SynthesisError(
            f"源音频只有 {source_duration:.2f} 秒，无法准备 IndexTTS2 参考音频；"
            f"至少需要 {_INDEX_REFERENCE_MIN_SECONDS:g} 秒。"
        )

    padding = project.settings.reference_padding_seconds
    start = max(0.0, min(source_duration, reference_sentence.start_seconds - padding))
    end = max(start, min(source_duration, reference_sentence.end_seconds + padding))
    if end - start + _INDEX_REFERENCE_DURATION_TOLERANCE >= _INDEX_REFERENCE_MIN_SECONDS:
        return start, end

    missing = _INDEX_REFERENCE_MIN_SECONDS - (end - start)
    start -= missing / 2
    end += missing / 2
    if start < 0:
        end = min(source_duration, end - start)
        start = 0.0
    if end > source_duration:
        start = max(0.0, start - (end - source_duration))
        end = source_duration

    if end - start + _INDEX_REFERENCE_DURATION_TOLERANCE < _INDEX_REFERENCE_MIN_SECONDS:
        raise SynthesisError(f"{reference_sentence.id} 附近没有足够长的音频可作为 IndexTTS2 参考。")
    return start, end


def _valid_cached_index_reference(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        duration = float(sf.info(path).duration)
    except (OSError, RuntimeError):
        return False
    return duration + _INDEX_REFERENCE_DURATION_TOLERANCE >= _INDEX_REFERENCE_MIN_SECONDS


def _index_sentence_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    reference_sentence: Sentence,
    *,
    role: str,
    shared: bool,
) -> VoiceReference:
    start_seconds, end_seconds = _index_reference_bounds(project, reference_sentence)
    payload = (
        f"{_INDEX_REFERENCE_CACHE_VERSION}|{project.source.sha256}|index-{role}|"
        f"{reference_sentence.id}|{start_seconds:.6f}|{end_seconds:.6f}|"
        f"{reference_sentence.source_text}"
    )
    identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    prefix = f"index_{role}_anchor" if shared else f"index_{role}_{reference_sentence.id}"
    destination = project_dir / "references" / f"{prefix}_{identity[:16]}.wav"
    if not _valid_cached_index_reference(destination):
        destination.unlink(missing_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp.wav")
        try:
            extract_reference(
                source=source,
                destination=temporary,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            if not _valid_cached_index_reference(temporary):
                raise SynthesisError(
                    f"{reference_sentence.id} 的 IndexTTS2 {role}参考音频不足 "
                    f"{_INDEX_REFERENCE_MIN_SECONDS:g} 秒。"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return VoiceReference(
        path=destination,
        text=reference_sentence.source_text or reference_sentence.zh_text,
        identity=f"index-{role}:{identity}",
        language=project.source_language if reference_sentence.source_text else "zh",
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
