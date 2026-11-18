from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from heapq import nlargest
from pathlib import Path

import numpy as np
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
_INDEX_REFERENCE_CACHE_VERSION = "index-reference-v3-quality-fallback"
_INDEX_REFERENCE_DURATION_TOLERANCE = 0.01
_REFERENCE_TEXT = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]", re.UNICODE)
logger = logging.getLogger(__name__)
_REFERENCE_SELECTION: ContextVar[tuple | None] = ContextVar("reference_selection", default=None)


@contextmanager
def reference_selection_scope(project: DubProject) -> Iterator[None]:
    """Freeze an operation's reference lookup index, never cache across edits."""
    active = _REFERENCE_SELECTION.get()
    if active is not None and active[0] is project:
        yield
        return
    by_id = {sentence.id: sentence for sentence in project.sentences}
    candidates = [s for s in project.sentences if s.source_text.strip() or s.zh_text.strip()]
    ranked = nlargest(
        2,
        candidates,
        key=lambda s: (
            s.end_seconds - s.start_seconds,
            len(_REFERENCE_TEXT.findall(s.source_text or s.zh_text)),
            -s.start_seconds,
        ),
    )
    shared = shared_reference_sentence(project) if candidates else None
    token = _REFERENCE_SELECTION.set((project, by_id, shared, ranked))
    try:
        yield
    finally:
        _REFERENCE_SELECTION.reset(token)


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
    active = _REFERENCE_SELECTION.get()
    if active is not None and active[0] is project:
        selected = active[1].get(configured) if configured else active[2]
        if selected is not None:
            return selected
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


_PICK_FRAME_SECONDS = 0.02
_PICK_MAX_CANDIDATES = 24


def _frame_levels(audio: np.ndarray, rate: int) -> np.ndarray:
    hop = max(1, round(rate * _PICK_FRAME_SECONDS))
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    count = len(mono) // hop
    if count == 0:
        return np.zeros(0)
    frames = mono[: count * hop].astype(np.float64).reshape(count, hop)
    return np.sqrt(np.mean(frames**2, axis=1))


def _duration_score(seconds: float) -> float:
    if 5.0 <= seconds <= 12.0:
        return 1.0
    if seconds < 5.0:
        return max(0.0, (seconds - 2.0) / 3.0)
    return max(0.0, 1.0 - (seconds - 12.0) / 8.0)


def pick_reference_sentence(project: DubProject, source: Path) -> Sentence | None:
    """Choose the voice anchor by quality, not just length (F3).

    Scores each candidate line on length (5-12 s is best), how far its voice
    stands above the recording's noise floor, and how much of it is voiced:
    a clear, normally spoken line gives a steadier cloned timbre than a long
    whisper or a line full of effects.  Deterministic for a given source.
    """
    from .filtering import is_japanese_filler_only
    from .voice_match import voiced_ratio

    candidates = [
        s
        for s in project.sentences
        if s.enabled
        and s.source_text.strip()
        and 2.0 <= s.end_seconds - s.start_seconds <= 20.0
        and not is_japanese_filler_only(s.source_text)
    ]
    if not candidates:
        return None
    candidates = sorted(
        candidates, key=lambda s: (-(s.end_seconds - s.start_seconds), s.start_seconds)
    )[:_PICK_MAX_CANDIDATES]
    try:
        with sf.SoundFile(source) as audio:
            rate = audio.samplerate
            levels = np.concatenate(
                [
                    _frame_levels(block, rate)
                    for block in audio.blocks(blocksize=rate * 30, dtype="float32", always_2d=True)
                ]
            )
            floor = float(np.percentile(levels[levels > 0], 10)) if np.any(levels > 0) else 1e-9
            scored = []
            for sentence in candidates:
                audio.seek(max(0, round(sentence.start_seconds * rate)))
                frames = round((sentence.end_seconds - sentence.start_seconds) * rate)
                clip = audio.read(frames, dtype="float32", always_2d=True)
                clip_levels = _frame_levels(clip, rate)
                if not len(clip_levels) or clip_levels.max() <= 0:
                    continue
                active = clip_levels[clip_levels >= clip_levels.max() * 0.1]
                snr_db = 20 * math.log10(max(float(np.mean(active)), 1e-9) / max(floor, 1e-9))
                score = (
                    _duration_score(sentence.end_seconds - sentence.start_seconds)
                    + 0.5 * min(max(snr_db, 0.0) / 30.0, 1.0)
                    + 0.5 * voiced_ratio(clip.mean(axis=1), rate)
                )
                scored.append((score, -sentence.start_seconds, sentence))
    except (OSError, RuntimeError, ValueError):
        return None
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def pick_whisper_reference_sentence(project: DubProject, source: Path) -> Sentence | None:
    """Choose a clean whispered line as the project's whisper voice anchor (W5).

    Scores lines with low ``voiced_ratio`` (<= 0.26) that still stand clearly above
    the room's noise floor, so whispered dubs are synthesized from an actual whisper
    rather than forcing a modal reference through post-hoc breath conversion.
    """
    from .filtering import is_japanese_filler_only
    from .voice_match import voiced_ratio

    candidates = [
        s
        for s in project.sentences
        if s.enabled
        and s.source_text.strip()
        and 1.8 <= s.end_seconds - s.start_seconds <= 20.0
        and not is_japanese_filler_only(s.source_text)
    ]
    if not candidates:
        return None
    candidates = sorted(
        candidates, key=lambda s: (-(s.end_seconds - s.start_seconds), s.start_seconds)
    )[:_PICK_MAX_CANDIDATES]
    try:
        with sf.SoundFile(source) as audio:
            rate = audio.samplerate
            levels = np.concatenate(
                [
                    _frame_levels(block, rate)
                    for block in audio.blocks(blocksize=rate * 30, dtype="float32", always_2d=True)
                ]
            )
            floor = float(np.percentile(levels[levels > 0], 10)) if np.any(levels > 0) else 1e-9
            scored = []
            for sentence in candidates:
                audio.seek(max(0, round(sentence.start_seconds * rate)))
                frames = round((sentence.end_seconds - sentence.start_seconds) * rate)
                clip = audio.read(frames, dtype="float32", always_2d=True)
                clip_levels = _frame_levels(clip, rate)
                if not len(clip_levels) or clip_levels.max() <= 0:
                    continue
                active = clip_levels[clip_levels >= clip_levels.max() * 0.1]
                snr_db = 20 * math.log10(max(float(np.mean(active)), 1e-9) / max(floor, 1e-9))
                if snr_db < 4.5:
                    continue
                vr = voiced_ratio(clip.mean(axis=1), rate)
                if vr > 0.26:
                    continue
                dur = sentence.end_seconds - sentence.start_seconds
                dur_score = 1.0 if 3.5 <= dur <= 10.0 else max(0.0, 1.0 - abs(dur - 6.0) / 8.0)
                whisper_score = 1.0 - min(vr / 0.28, 1.0)
                score = dur_score + 0.5 * min(max(snr_db, 0.0) / 25.0, 1.0) + 0.8 * whisper_score
                scored.append((score, -sentence.start_seconds, sentence))
    except (OSError, RuntimeError, ValueError):
        return None
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def annotate_sentence_voice_modes(project: DubProject, source: Path) -> None:
    """Classify sentences into ``modal`` vs ``whisper`` from ``source`` audio."""
    from .voice_match import voiced_ratio

    if not project.sentences or not source.is_file():
        return
    try:
        with sf.SoundFile(source) as audio:
            rate = audio.samplerate
            for sentence in project.sentences:
                if sentence.end_seconds - sentence.start_seconds < 0.20:
                    continue
                lo = max(0, min(audio.frames, round(sentence.start_seconds * rate)))
                hi = max(lo, min(audio.frames, round(sentence.end_seconds * rate)))
                if hi - lo < rate // 10:
                    continue
                audio.seek(lo)
                clip = audio.read(hi - lo, dtype="float32", always_2d=True)
                # Pick whichever stereo ear is louder so single-ear whispers aren't halved.
                ear_rms = np.sqrt(np.mean(clip.astype(np.float64) ** 2, axis=0))
                mono = clip[:, int(np.argmax(ear_rms))] if clip.shape[1] >= 2 else clip[:, 0]
                vr = voiced_ratio(mono, rate)
                sentence.voice_mode = "whisper" if vr < 0.24 else "modal"
    except (OSError, RuntimeError, ValueError):
        pass


def effective_project_reference_sentence(
    project: DubProject,
    sentence: Sentence,
) -> tuple[Sentence, bool]:
    """Return ``(anchor_sentence, is_whisper_anchor)`` for IndexTTS speaker prompt."""
    if (
        project.settings.tts_dual_voice_anchor
        and sentence.voice_mode == "whisper"
        and project.settings.tts_whisper_reference_sentence_id
    ):
        whisper_id = project.settings.tts_whisper_reference_sentence_id
        anchor = next((s for s in project.sentences if s.id == whisper_id), None)
        if anchor is not None:
            return anchor, True
    return shared_reference_sentence(project), False


def effective_index_emotion_weight(project: DubProject, sentence: Sentence) -> float:
    """Boost emotion conditioning on whispers, and back off on QC retries to stop slurring."""
    base = float(project.settings.tts_index_emo_alpha)
    weight = (
        max(base, 0.85)
        if project.settings.tts_dual_voice_anchor and sentence.voice_mode == "whisper"
        else base
    )
    if sentence.tts_attempt == 1:
        return min(weight, 0.30)
    if sentence.tts_attempt >= 2:
        return min(weight, 0.10)
    return weight


def fallback_reference_sentence(project: DubProject, sentence: Sentence) -> Sentence | None:
    """Choose a deterministic project reference that is not the rejected sentence."""

    configured = project.settings.tts_reference_sentence_id
    active = _REFERENCE_SELECTION.get()
    if active is not None and active[0] is project:
        selected = active[1].get(configured)
        if (
            selected is not None
            and selected.id != sentence.id
            and (selected.source_text.strip() or selected.zh_text.strip())
        ):
            return selected
        return next((s for s in active[3] if s.id != sentence.id), None)
    if configured and configured != sentence.id:
        selected = next((item for item in project.sentences if item.id == configured), None)
        if selected is not None and (selected.source_text.strip() or selected.zh_text.strip()):
            return selected
    candidates = [
        item
        for item in project.sentences
        if item.id != sentence.id and (item.source_text.strip() or item.zh_text.strip())
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.end_seconds - item.start_seconds,
            len(_REFERENCE_TEXT.findall(item.source_text or item.zh_text)),
            -item.start_seconds,
        ),
    )


def reference_audio_signature(project: DubProject, source: Path) -> str:
    """Identify how reference audio is cut, so caches never mix the two methods."""
    mode = project.settings.reference_audio_mode
    origin = "vocals" if source.name == "vocals.wav" and "separation" in source.parts else "mix"
    return f"reference-audio-v2|{mode}|{origin}"


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
    fullband = project.settings.reference_audio_mode != "legacy"
    if fullband:
        identity = hashlib.sha256(
            f"{identity}|{reference_audio_signature(project, source)}".encode()
        ).hexdigest()
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
                fullband=fullband,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return VoiceReference(
        path=destination,
        text=reference_sentence.source_text or reference_sentence.zh_text,
        identity=f"project:{identity}",
        language=project.source_language
        if reference_sentence.source_text
        else project.settings.tts_target_language,
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


def _index_reference_is_usable(sentence: Sentence, path: Path) -> bool:
    """Reject only clearly invalid current-sentence references, not quiet ASMR speech."""

    text = sentence.source_text.strip() or sentence.zh_text.strip()
    text_units = len(_REFERENCE_TEXT.findall(text))
    if text_units == 0:
        return False
    try:
        with sf.SoundFile(path) as audio:
            peak = 0.0
            sum_squares = 0.0
            sample_count = 0
            for block in audio.blocks(blocksize=max(1, int(audio.samplerate)), dtype="float32"):
                if block.size == 0:
                    continue
                absolute = abs(block)
                peak = max(peak, float(absolute.max(initial=0.0)))
                sum_squares += float((block.astype("float64") ** 2).sum())
                sample_count += int(block.size)
    except (OSError, RuntimeError, ValueError):
        return False
    if sample_count == 0:
        return False
    rms = math.sqrt(sum_squares / sample_count)
    # The limits are intentionally far below normal whisper levels. They only
    # catch empty/zero-filled slices and avoid classifying quiet ASMR as silence.
    return peak >= 2e-5 and rms >= 2e-6


def _fallback_index_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    sentence: Sentence,
    reference: VoiceReference,
    *,
    role: str,
) -> VoiceReference:
    if _index_reference_is_usable(sentence, reference.path):
        return reference
    fallback = fallback_reference_sentence(project, sentence)
    if fallback is None:
        logger.warning(
            "%s 的 IndexTTS2 %s参考无效，但项目中没有可用的统一参考句", sentence.id, role
        )
        return reference
    logger.warning(
        "%s 的 IndexTTS2 %s参考明显无效，自动改用项目参考句 %s",
        sentence.id,
        role,
        fallback.id,
    )
    return _index_sentence_reference(
        project,
        project_dir,
        source,
        fallback,
        role=role,
        shared=True,
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
    start_seconds, end_seconds = _index_reference_bounds(project, reference_sentence)
    payload = (
        f"{_INDEX_REFERENCE_CACHE_VERSION}|{project.source.sha256}|index-{role}|"
        f"{reference_sentence.id}|{start_seconds:.6f}|{end_seconds:.6f}|"
        f"{reference_sentence.source_text}"
    )
    fullband = project.settings.reference_audio_mode != "legacy"
    if fullband:
        payload += f"|{reference_audio_signature(project, source)}"
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
                fullband=fullband,
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
        language=project.source_language
        if reference_sentence.source_text
        else project.settings.tts_target_language,
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
    if shared:
        reference_sentence, _ = effective_project_reference_sentence(project, sentence)
    else:
        reference_sentence = sentence
    reference = _index_sentence_reference(
        project,
        project_dir,
        source,
        reference_sentence,
        role="speaker",
        shared=shared,
    )
    if shared:
        return reference
    return _fallback_index_reference(
        project,
        project_dir,
        source,
        sentence,
        reference,
        role="speaker",
    )


def prepare_index_emotion_reference(
    project: DubProject,
    project_dir: Path,
    source: Path,
    sentence: Sentence,
    speaker_reference: VoiceReference,
) -> VoiceReference | None:
    source_id = project.settings.tts_index_emotion_source
    if source_id in {"text", "vector"}:
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
    reference = _index_sentence_reference(
        project,
        project_dir,
        source,
        reference_sentence,
        role="emotion",
        shared=shared,
    )
    if shared:
        return reference
    return _fallback_index_reference(
        project,
        project_dir,
        source,
        sentence,
        reference,
        role="emotion",
    )
