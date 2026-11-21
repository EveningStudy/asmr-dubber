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

import soundfile as sf

from .audio import extract_reference, probe_audio
from .audio_ffmpeg import (
    clip_snr_db,
    louder_ear_mono,
    open_audio_for_analysis,
    read_clip_for_window,
    source_noise_floor,
)
from .errors import AsmrDubberError, SynthesisError
from .hashing import cached_sha256_file
from .languages import SourceLanguage
from .models import DubProject, Sentence, SpeakerReference

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


_PICK_MAX_CANDIDATES = 24


def _duration_score(seconds: float) -> float:
    if 5.0 <= seconds <= 12.0:
        return 1.0
    if seconds < 5.0:
        return max(0.0, (seconds - 2.0) / 3.0)
    return max(0.0, 1.0 - (seconds - 12.0) / 8.0)


def _whisper_duration_score(seconds: float) -> float:
    if 3.5 <= seconds <= 10.0:
        return 1.0
    return max(0.0, 1.0 - abs(seconds - 6.0) / 8.0)


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
        candidates,
        key=lambda s: (
            -_duration_score(s.end_seconds - s.start_seconds),
            -(s.end_seconds - s.start_seconds),
            s.start_seconds,
        ),
    )[:_PICK_MAX_CANDIDATES]
    try:
        with open_audio_for_analysis(source) as audio:
            rate = audio.samplerate
            floor = source_noise_floor(audio, rate)
            scored = []
            for sentence in candidates:
                clip = read_clip_for_window(
                    audio, rate, sentence.start_seconds, sentence.end_seconds
                )
                snr_db = clip_snr_db(clip, rate, floor)
                if snr_db is None:
                    continue
                score = (
                    _duration_score(sentence.end_seconds - sentence.start_seconds)
                    + 0.5 * min(max(snr_db, 0.0) / 30.0, 1.0)
                    + 0.5 * voiced_ratio(louder_ear_mono(clip), rate)
                )
                scored.append((score, -sentence.start_seconds, sentence))
    except (OSError, RuntimeError, ValueError, AsmrDubberError):
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
        candidates,
        key=lambda s: (
            -_whisper_duration_score(s.end_seconds - s.start_seconds),
            -(s.end_seconds - s.start_seconds),
            s.start_seconds,
        ),
    )[:_PICK_MAX_CANDIDATES]
    try:
        with open_audio_for_analysis(source) as audio:
            rate = audio.samplerate
            floor = source_noise_floor(audio, rate)
            scored = []
            for sentence in candidates:
                clip = read_clip_for_window(
                    audio, rate, sentence.start_seconds, sentence.end_seconds
                )
                snr_db = clip_snr_db(clip, rate, floor)
                if snr_db is None or snr_db < 4.5:
                    continue
                vr = voiced_ratio(louder_ear_mono(clip), rate)
                if vr > 0.26:
                    continue
                dur = sentence.end_seconds - sentence.start_seconds
                whisper_score = 1.0 - min(vr / 0.28, 1.0)
                score = (
                    _whisper_duration_score(dur)
                    + 0.5 * min(max(snr_db, 0.0) / 25.0, 1.0)
                    + 0.8 * whisper_score
                )
                scored.append((score, -sentence.start_seconds, sentence))
    except (OSError, RuntimeError, ValueError, AsmrDubberError):
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
        with open_audio_for_analysis(source) as audio:
            rate = audio.samplerate
            for sentence in project.sentences:
                if sentence.end_seconds - sentence.start_seconds < 0.20:
                    sentence.voice_mode = "modal"
                    continue
                clip = read_clip_for_window(
                    audio, rate, sentence.start_seconds, sentence.end_seconds
                )
                if len(clip) < rate // 10:
                    sentence.voice_mode = "modal"
                    continue
                vr = voiced_ratio(louder_ear_mono(clip), rate)
                sentence.voice_mode = "whisper" if vr < 0.24 else "modal"
    except (OSError, RuntimeError, ValueError, AsmrDubberError):
        pass


def project_has_multiple_speakers(project: DubProject) -> bool:
    """Return True when enabled sentences span two or more distinct speaker IDs."""
    speakers = {
        s.speaker_id.strip()
        for s in project.sentences
        if s.enabled and s.speaker_id and s.speaker_id.strip()
    }
    return len(speakers) >= 2


def _sentence_by_id(project: DubProject, sentence_id: str | None) -> Sentence | None:
    if not sentence_id:
        return None
    active = _REFERENCE_SELECTION.get()
    if active is not None and active[0] is project:
        return active[1].get(sentence_id)
    return next((item for item in project.sentences if item.id == sentence_id), None)


def configured_speaker_reference(
    project: DubProject, sentence: Sentence
) -> SpeakerReference | None:
    """Return a valid per-speaker reference override configured on ``project``, if any."""
    spk = (sentence.speaker_id or "").strip()
    if not spk:
        return None
    ref = project.speaker_references.get(spk)
    if ref is None:
        return None
    if ref.source == "external":
        return ref if ref.audio_path.strip() else None
    if ref.source == "project_sentence" and _sentence_by_id(project, ref.sentence_id) is not None:
        return ref
    return None


def speaker_reference_sentence(
    project: DubProject,
    sentence: Sentence,
    source: Path | None = None,
    *,
    ignore_override: bool = False,
) -> tuple[Sentence, bool]:
    """Select the clearest modal or whisper voice anchor for ``sentence.speaker_id``."""
    from .filtering import is_japanese_filler_only

    if not ignore_override:
        override = configured_speaker_reference(project, sentence)
        if override is not None and override.source == "project_sentence":
            anchor = _sentence_by_id(project, override.sentence_id)
            if anchor is not None:
                return anchor, anchor.voice_mode == "whisper"

    target_spk = (sentence.speaker_id or "").strip()
    same_speaker = [
        s
        for s in project.sentences
        if s.enabled
        and (s.speaker_id or "").strip() == target_spk
        and (s.source_text.strip() or s.zh_text.strip())
    ]
    if not same_speaker:
        return shared_reference_sentence(project), False

    non_filler = [
        s
        for s in same_speaker
        if not is_japanese_filler_only(s.source_text)
        and s.status not in {"skipped_filler", "skipped_sound", "skipped_hallucination"}
    ]
    if non_filler:
        same_speaker = non_filler
    if source is not None and source.is_file():
        try:
            with open_audio_for_analysis(source) as audio:
                rate = audio.samplerate
                floor = source_noise_floor(audio, rate)
                clean = []
                for s in same_speaker:
                    clip = read_clip_for_window(audio, rate, s.start_seconds, s.end_seconds)
                    snr_db = clip_snr_db(clip, rate, floor)
                    if snr_db is not None and snr_db >= 4.5:
                        clean.append(s)
                if clean:
                    same_speaker = clean
        except (OSError, RuntimeError, ValueError, AsmrDubberError):
            pass

    if project.settings.tts_dual_voice_anchor and sentence.voice_mode == "whisper":
        whispers = [
            s
            for s in same_speaker
            if s.voice_mode == "whisper" and 1.2 <= (s.end_seconds - s.start_seconds) <= 20.0
        ]
        if whispers:
            best_w = max(
                whispers,
                key=lambda s: (
                    _duration_score(s.end_seconds - s.start_seconds),
                    len(_REFERENCE_TEXT.findall(s.source_text or s.zh_text)),
                    -s.start_seconds,
                ),
            )
            return best_w, True

    modals = [s for s in same_speaker if s.voice_mode != "whisper"] or same_speaker
    best_m = max(
        modals,
        key=lambda s: (
            _duration_score(s.end_seconds - s.start_seconds),
            s.end_seconds - s.start_seconds,
            len(_REFERENCE_TEXT.findall(s.source_text or s.zh_text)),
            -s.start_seconds,
        ),
    )
    return best_m, False


def effective_project_reference_sentence(
    project: DubProject,
    sentence: Sentence,
) -> tuple[Sentence, bool]:
    """Return ``(anchor_sentence, is_whisper_anchor)`` for IndexTTS speaker prompt."""
    override = configured_speaker_reference(project, sentence)
    if override is not None and override.source == "project_sentence":
        anchor = _sentence_by_id(project, override.sentence_id)
        if anchor is not None:
            return anchor, anchor.voice_mode == "whisper"
    if sentence.speaker_id and project_has_multiple_speakers(project):
        return speaker_reference_sentence(project, sentence)
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
    target_spk = (sentence.speaker_id or "").strip()
    multi_spk = bool(target_spk) and project_has_multiple_speakers(project)
    configured = project.settings.tts_reference_sentence_id
    active = _REFERENCE_SELECTION.get()
    if not multi_spk and active is not None and active[0] is project:
        selected = active[1].get(configured)
        if (
            selected is not None
            and selected.id != sentence.id
            and (selected.source_text.strip() or selected.zh_text.strip())
        ):
            return selected
        return next((s for s in active[3] if s.id != sentence.id), None)
    if not multi_spk and configured and configured != sentence.id:
        selected = next((item for item in project.sentences if item.id == configured), None)
        if selected is not None and (selected.source_text.strip() or selected.zh_text.strip()):
            return selected
    candidates = [
        item
        for item in project.sentences
        if item.id != sentence.id and (item.source_text.strip() or item.zh_text.strip())
    ]
    if multi_spk:
        same_spk = [item for item in candidates if (item.speaker_id or "").strip() == target_spk]
        if same_spk:
            candidates = same_spk
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
    override = configured_speaker_reference(project, sentence)
    if override is not None and override.source == "external":
        path = Path(override.audio_path).expanduser().resolve()
        if not path.is_file():
            raise SynthesisError(f"找不到说话人 {sentence.speaker_id} 的外部参考音频：{path}")
        resolved_lang: SourceLanguage = (
            project.source_language if override.language == "auto" else override.language
        )
        spk = (sentence.speaker_id or "").strip()
        raw_payload = f"{spk}|{cached_sha256_file(path)}|{override.text.strip()}|{resolved_lang}"
        identity = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        return VoiceReference(
            path=path,
            text=override.text.strip(),
            identity=f"speaker-external:{identity}",
            language=resolved_lang,
        )

    has_sentence_override = override is not None and override.source == "project_sentence"
    if not has_sentence_override and project.settings.tts_reference_source == "external":
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
    if has_sentence_override or (
        stable and sentence.speaker_id and project_has_multiple_speakers(project)
    ):
        reference_sentence, _ = speaker_reference_sentence(project, sentence)
        safe_spk = re.sub(r"[^A-Za-z0-9_-]+", "_", (sentence.speaker_id or "").strip()) or "S01"
        prefix = f"speaker_{safe_spk}"
        payload = (
            f"{project.source.sha256}|{reference_sentence.id}|{reference_sentence.start_seconds:.6f}|"
            f"{reference_sentence.end_seconds:.6f}|{reference_sentence.source_text}|"
            f"{project.settings.reference_padding_seconds:.6f}"
        )
        identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    elif stable:
        reference_sentence = shared_reference_sentence(project)
        project.settings.tts_reference_sentence_id = reference_sentence.id
        prefix = "voice_anchor"
        identity = reference_plan_hash(project)
    else:
        reference_sentence = sentence
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
    override = configured_speaker_reference(project, sentence)
    if override is not None:
        if override.source == "external":
            return _index_external_reference(override.audio_path, role="speaker")
        ref_sentence = _sentence_by_id(project, override.sentence_id)
        if ref_sentence is not None:
            return _index_sentence_reference(
                project,
                project_dir,
                source,
                ref_sentence,
                role="speaker",
                shared=True,
            )
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
    reference_sentence = (
        effective_project_reference_sentence(project, sentence)[0] if shared else sentence
    )
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
