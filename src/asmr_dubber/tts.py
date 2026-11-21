from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .errors import SynthesisError
from .filtering import has_speakable_text
from .hashing import cached_sha256_file
from .models import DubProject, ProjectSettings, Sentence
from .task_control import CancellationSignal, check_cancelled
from .voice_reference import (
    STABLE_CLONE_MODES,
    configured_speaker_reference,
    effective_index_emotion_weight,
    effective_project_reference_sentence,
    fallback_reference_sentence,
    project_has_multiple_speakers,
    reference_plan_hash,
    speaker_reference_sentence,
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
            "ja": sentence.source_text,
        }
    }
    speaker_source = project.settings.tts_index_speaker_source
    emotion_source = project.settings.tts_index_emotion_source
    if "sentence_reference" in {speaker_source, emotion_source}:
        fallback = fallback_reference_sentence(project, sentence)
        references["sentence_quality_fallback"] = (
            {
                "id": fallback.id,
                "start": fallback.start_seconds,
                "end": fallback.end_seconds,
                "ja": fallback.source_text,
                "zh": fallback.zh_text,
            }
            if fallback is not None
            else None
        )
        references["sentence_quality_fallback_version"] = 1
    if "project_reference" in {speaker_source, emotion_source}:
        shared, is_whisper = effective_project_reference_sentence(project, sentence)
        references["project_reference"] = {
            "id": shared.id,
            "start": shared.start_seconds,
            "end": shared.end_seconds,
            "ja": shared.source_text,
        }
        if is_whisper:
            references["whisper_anchor"] = True

    payload: dict[str, object] = {}
    override = configured_speaker_reference(project, sentence)
    if override is not None and override.source == "external":
        path = Path(override.audio_path).expanduser()
        if not path.is_file():
            raise SynthesisError(f"找不到说话人 {sentence.speaker_id} 的外部参考音频：{path}")
        payload["speaker"] = {"speaker_external": cached_sha256_file(path)}
    elif override is not None and override.source == "project_sentence":
        ref_sent = next((s for s in project.sentences if s.id == override.sentence_id), None)
        if ref_sent is not None:
            payload["speaker"] = {
                "id": ref_sent.id,
                "start": ref_sent.start_seconds,
                "end": ref_sent.end_seconds,
                "ja": ref_sent.source_text,
            }
        else:
            payload["speaker"] = references[speaker_source]
    elif speaker_source == "external":
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
    if "sentence_quality_fallback" in references:
        payload["fallback"] = references["sentence_quality_fallback"]
        payload["fallback_version"] = references["sentence_quality_fallback_version"]
    return payload


_ELLIPSIS = re.compile(r"[…⋯]+|\.{3,}|・{2,}|[—―]{2,}|[～〜~]{2,}")
_COMMA_RUN = re.compile(r"[，,、\s]*[，,、][，,、\s]*")
_COMMA_BEFORE_END = re.compile(r"[，,、\s]+([。！？!?」』）)])")
_EN_COMMA_RUN = re.compile(r"\s*(?:,\s*){2,}")
_EN_COMMA_BEFORE_END = re.compile(r",\s+([.!?\"')\]])")


def normalize_pauses(text: str, target_language: str = "zh") -> str:
    """Turn long-pause punctuation into ordinary commas before synthesis.

    Zero-shot TTS renders 「……」 as a second or more of silence, so a line full of
    ellipses runs far longer than the source it replaces.  Idempotent.
    """
    if target_language == "en":
        value = _ELLIPSIS.sub(", ", text)
        value = _EN_COMMA_RUN.sub(", ", value)
        value = _EN_COMMA_BEFORE_END.sub(r"\1", value)
        value = re.sub(r"\s{2,}", " ", value).strip().strip(",").strip()
        return value or text
    value = _ELLIPSIS.sub("，", text)
    value = _COMMA_RUN.sub("，", value)
    value = _COMMA_BEFORE_END.sub(r"\1", value)
    value = value.strip().strip("，,、").strip()
    return value or text


def spoken_text(settings: ProjectSettings, text: str) -> str:
    """The exact text sent to a TTS backend for a translated line."""
    from .zh_phonetics import apply_spoken_normalization

    value = (
        normalize_pauses(text, target_language=settings.tts_target_language)
        if settings.tts_normalize_pauses
        else text
    )
    if settings.tts_pronunciation_glossary or settings.tts_spoken_normalization:
        value = apply_spoken_normalization(
            value,
            glossary=settings.tts_pronunciation_glossary,
            normalize_numbers=settings.tts_spoken_normalization,
            target_language=settings.tts_target_language,
        )
    return value


def attempt_seed(seed: int, sentence: Sentence) -> int:
    """A different but reproducible seed for each automatic regeneration."""
    return (seed + 1009 * sentence.tts_attempt) % (2**32)


def target_duration_for_sentence(project: DubProject, sentence: Sentence) -> float | None:
    """Target spoken duration (100 ms quantized) for native S2Mel length control (W1).

    Respects both subsequent dubbed lines and preserved sound/breath obstacles so
    synthesized lines fit their acoustic slot natively without post-hoc speed-up.
    """
    settings = project.settings
    if not settings.tts_fit_source_duration or settings.tts_backend not in {
        "indextts2",
        "indextts2_5",
    }:
        return None
    own = max(0.25, sentence.end_seconds - sentence.start_seconds)
    next_start: float | None = None
    for other in project.sentences:
        if other.id == sentence.id or other.start_seconds <= sentence.start_seconds + 0.01:
            continue
        is_dubbed = (
            other.enabled
            and other.chinese_audio_enabled
            and (
                bool(other.zh_text)
                or other.status not in {"skipped_sound", "skipped_filler", "skipped_tts"}
            )
        )
        is_kept_sound = (
            other.original_audio_enabled is True
            or other.status == "skipped_sound"
            or (other.enabled and not other.chinese_audio_enabled)
        ) and (other.end_seconds - other.start_seconds >= 0.20)
        if (is_dubbed or is_kept_sound) and (
            next_start is None or other.start_seconds < next_start
        ):
            next_start = other.start_seconds
    gap = max(0.0, next_start - sentence.end_seconds - 0.06) if next_start is not None else 0.40
    target = own + min(gap, 0.85)
    return round(max(0.30, target) * 10) / 10.0


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
        "voice": settings.tts_voice,
        "volume": settings.tts_volume,
        "pitch": settings.tts_pitch,
        "emotion": settings.tts_emotion,
        "style_prompt": settings.tts_style_prompt,
        "temperature": settings.tts_temperature,
        "top_p": settings.tts_top_p,
        "api_base_url": settings.tts_api_base_url,
        "api_extra_body": settings.tts_api_extra_body,
        "model_path": settings.tts_model_path,
        "config_path": settings.tts_config_path,
        "executable": settings.tts_executable,
        "index_fp16": settings.tts_index_use_fp16,
        "index_emo_alpha": effective_index_emotion_weight(project, sentence),
        "index_speaker_source": settings.tts_index_speaker_source,
        "index_emotion_source": settings.tts_index_emotion_source,
        "index_emo_text": settings.tts_index_emo_text,
        "gpt_top_k": settings.tts_gpt_top_k,
        "gpt_split": settings.tts_gpt_text_split_method,
        "gpt_sample_steps": settings.tts_gpt_sample_steps,
        "cosyvoice_mode": settings.tts_cosyvoice_mode,
        "zh": spoken_text(settings, sentence.zh_text),
        "sentence_id": sentence.id,
        "implementation": (
            "supported-backends-v7-stable-seed"
            if settings.tts_backend == "indextts2_5"
            else "supported-backends-v6-indextts25"
        ),
    }
    if settings.tts_target_language != "zh":
        payload["target_language"] = settings.tts_target_language
    if settings.tts_backend in {"indextts2_5", "indextts2"}:
        payload["index_references"] = _index_reference_payload(project, sentence)
        payload["reference_padding"] = settings.reference_padding_seconds
        target_dur = target_duration_for_sentence(project, sentence)
        if target_dur is not None:
            payload["target_duration_100ms"] = round(target_dur * 10)
        if settings.tts_index_emotion_source == "vector":
            payload["emotion_vector"] = settings.tts_index25_emotion_vector
        if settings.tts_backend == "indextts2":
            defaults = ProjectSettings()
            index2_overrides = {
                key: getattr(settings, field_name)
                for key, field_name in (
                    ("seed", "random_seed"),
                    ("num_beams", "tts_index25_num_beams"),
                    ("do_sample", "tts_index25_do_sample"),
                    ("top_k", "tts_index25_top_k"),
                    ("repetition_penalty", "tts_index25_repetition_penalty"),
                    ("length_penalty", "tts_index25_length_penalty"),
                    ("max_mel_tokens", "tts_index25_max_mel_tokens"),
                    ("interval_silence_ms", "tts_index25_interval_silence_ms"),
                    ("max_text_tokens", "tts_index25_max_text_tokens"),
                )
                if getattr(settings, field_name) != getattr(defaults, field_name)
            }
            if index2_overrides:
                payload["index2_generation"] = index2_overrides
        if settings.tts_backend == "indextts2_5":
            payload["index25"] = {
                "model_path": settings.tts_index25_model_path,
                "config_path": settings.tts_index25_config_path,
                "language": settings.tts_target_language,
                "bf16": settings.tts_index25_use_bf16,
                "cuda_kernel": settings.tts_index25_use_cuda_kernel,
                "deepspeed": settings.tts_index25_use_deepspeed,
                "accel": settings.tts_index25_use_accel,
                "torch_compile": settings.tts_index25_use_torch_compile,
                "emotion_vector": settings.tts_index25_emotion_vector,
                "use_random": settings.tts_index25_use_random,
                "interval_silence_ms": settings.tts_index25_interval_silence_ms,
                "max_text_tokens": settings.tts_index25_max_text_tokens,
                "duration_factor": settings.tts_index25_duration_factor,
                "text_normalization": settings.tts_index25_text_normalization,
                "do_sample": settings.tts_index25_do_sample,
                "temperature": settings.tts_index25_temperature,
                "top_p": settings.tts_index25_top_p,
                "top_k": settings.tts_index25_top_k,
                "num_beams": settings.tts_index25_num_beams,
                "repetition_penalty": settings.tts_index25_repetition_penalty,
                "length_penalty": settings.tts_index25_length_penalty,
                "max_mel_tokens": settings.tts_index25_max_mel_tokens,
                "seed": settings.random_seed,
            }
    elif settings.tts_backend in {"edge_tts", "minimax", "generic_tts_api"} or (
        settings.tts_backend == "mimo_tts"
        and (settings.tts_model.strip() or "mimo-v2.5-tts-voiceclone") != "mimo-v2.5-tts-voiceclone"
    ):
        payload["reference_plan"] = "unused"
    elif (override := configured_speaker_reference(project, sentence)) is not None:
        if override.source == "external":
            path = Path(override.audio_path).expanduser()
            if not path.is_file():
                raise SynthesisError(f"找不到说话人 {sentence.speaker_id} 的外部参考音频：{path}")
            payload["reference_plan"] = {
                "speaker": sentence.speaker_id,
                "source": "external",
                "sha256": cached_sha256_file(path),
                "text": override.text,
                "language": override.language,
            }
        else:
            spk_ref, _ = speaker_reference_sentence(project, sentence)
            payload["reference_plan"] = {
                "speaker": sentence.speaker_id,
                "sentence_id": spk_ref.id,
                "start": spk_ref.start_seconds,
                "end": spk_ref.end_seconds,
                "padding": settings.reference_padding_seconds,
            }
        payload["seed"] = settings.random_seed
    elif (
        settings.tts_reference_source == "external"
        or settings.tts_clone_mode in _STABLE_CLONE_MODES
    ):
        if (
            settings.tts_reference_source != "external"
            and sentence.speaker_id
            and project_has_multiple_speakers(project)
        ):
            spk_ref, _ = speaker_reference_sentence(project, sentence)
            payload["reference_plan"] = {
                "speaker": sentence.speaker_id,
                "sentence_id": spk_ref.id,
                "start": spk_ref.start_seconds,
                "end": spk_ref.end_seconds,
                "padding": settings.reference_padding_seconds,
            }
        else:
            payload["reference_plan"] = shared_reference_plan_hash(project)
        payload["seed"] = settings.random_seed
    else:
        payload["reference_plan"] = {
            "start": sentence.start_seconds,
            "end": sentence.end_seconds,
            "ja": sentence.source_text,
            "padding": settings.reference_padding_seconds,
        }
    if settings.tts_normalize_pauses:
        payload["pauses"] = "normalized-v1"
    if sentence.tts_attempt:
        payload["attempt"] = sentence.tts_attempt
    if settings.reference_audio_mode != "legacy":
        payload["reference_audio"] = [
            "reference-audio-v2",
            settings.reference_audio_mode,
            settings.separation_enabled and settings.reference_audio_mode == "fullband_vocals",
        ]
    if project.source_language != "ja":
        payload["source_language"] = project.source_language
    if settings.tts_backend == "gpt_sovits":
        payload["seed"] = settings.random_seed
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=4096)
def _readable_audio(path: str, size: int, modified: int) -> bool:
    del size, modified
    try:
        with sf.SoundFile(path) as audio:
            if audio.frames <= 0 or audio.samplerate <= 0:
                return False
            return all(
                np.isfinite(block).all()
                for block in audio.blocks(blocksize=262144, dtype="float32")
            )
    except (OSError, RuntimeError, ValueError):
        return False


def valid_tts_cache(project: DubProject, sentence: Sentence, directory: Path) -> bool:
    from .audio import resolve_project_path

    if not sentence.tts_file or sentence.tts_cache_key != tts_cache_key(project, sentence):
        return False
    path = resolve_project_path(directory, sentence.tts_file, f"句子 {sentence.id} 的中文音频")
    try:
        stat = path.stat()
    except OSError:
        return False
    return _readable_audio(str(path), stat.st_size, stat.st_mtime_ns)


_QC_RETRIES = 2
_SPOKEN_UNITS = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]", re.UNICODE)
_EN_WORD_UNITS = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class _InspectedTake:
    """One take of a line as the automatic check saw it, enough to switch back to it."""

    attempt: int
    file: str | None
    duration: float | None
    cache_key: str | None
    reference_file: str | None
    warning: str | None
    broken: bool  # failed the gross check (silence, clipping, far too short/long)
    cer: float | None  # back-listening error rate, when it was measured

    @classmethod
    def of(cls, sentence: Sentence, broken: bool, cer: float | None) -> _InspectedTake:
        return cls(
            sentence.tts_attempt,
            sentence.tts_file,
            sentence.tts_duration_seconds,
            sentence.tts_cache_key,
            sentence.reference_file,
            sentence.tts_warning,
            broken,
            cer,
        )

    def badness(self) -> tuple[bool, float]:
        return self.broken, self.cer if self.cer is not None else 0.0


def _keep_best_take(directory: Path, sentence: Sentence, history: list[_InspectedTake]) -> None:
    """After every retry failed the check, keep the least bad take rather than the last one.

    A gross failure is worse than any recognition mismatch; among mismatches the lowest
    error rate wins, and a tie keeps the newest take.
    """
    from .audio import resolve_project_path

    best = min(reversed(history), key=_InspectedTake.badness)
    if best.attempt == sentence.tts_attempt or not best.file:
        return
    try:
        if not resolve_project_path(directory, best.file, sentence.id).is_file():
            return
    except Exception:
        return
    sentence.tts_attempt = best.attempt
    sentence.tts_file = best.file
    sentence.tts_duration_seconds = best.duration
    sentence.tts_cache_key = best.cache_key
    sentence.reference_file = best.reference_file
    sentence.tts_warning = best.warning
    sentence.status = "synthesized"
    sentence.error = None


def dub_problem(path: Path, text: str, target_language: str = "zh") -> str | None:
    """Return why a synthesized line looks broken, or None when it looks usable.

    Only gross failures are caught: no sound, far too short for the text (words
    dropped), far too long (rambling or repeated audio), or clipped samples.
    """
    from .audio import speech_bounds

    try:
        data, _rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError):
        return "无法读取配音文件"
    units = (
        len(_EN_WORD_UNITS.findall(text))
        if target_language == "en"
        else len(_SPOKEN_UNITS.findall(text))
    )
    level = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if data.size else 0.0
    if level < 10 ** (-60 / 20):
        return "配音几乎没有声音"
    start, end = speech_bounds(path)
    seconds = end - start
    if units >= 2 and seconds < 0.2:
        return "配音几乎没有声音"
    min_per_unit = 0.10 if target_language == "en" else 0.06
    max_per_unit = 1.2 if target_language == "en" else 0.6
    if units >= 3 and seconds < units * min_per_unit:
        return f"配音只有 {seconds:.1f} 秒，比文字短太多，可能漏字"
    if seconds > max(3.0, units * max_per_unit):
        return f"配音长达 {seconds:.1f} 秒，比文字长太多，可能有多余内容"
    if float(np.mean(np.abs(data) >= 0.999)) > 0.001:
        return "配音有削波失真"
    return None


def _evict_tts_daemons_if_low_vram(min_free_gb: float = 4.5) -> None:
    from .tts_daemon import daemon_pool_active, evict_tts_daemons

    if not daemon_pool_active():
        return
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= min_free_gb * (1024**3):
                return
    except Exception:
        return
    evict_tts_daemons()


def synthesize_sentences(
    project: DubProject,
    project_dir: Path,
    source: Path,
    force: bool = False,
    sentence_ids: Iterable[str] | None = None,
    progress: Progress | None = None,
    on_sentence: Callable[[], None] | None = None,
    cancel_event: CancellationSignal | None = None,
) -> list[str]:
    from .audio import resolve_project_path
    from .tts_backends import synthesize_with_selected_backend

    requested = set(sentence_ids) if sentence_ids is not None else None
    options = {"progress": progress, "on_sentence": on_sentence, "cancel_event": cancel_event}
    if project.settings.tts_reference_sentence_id is None and not any(
        sentence.tts_file for sentence in project.sentences
    ):
        from .voice_reference import pick_reference_sentence

        picked = pick_reference_sentence(project, source)
        if picked is not None:
            project.settings.tts_reference_sentence_id = picked.id
    if project.settings.tts_dual_voice_anchor:
        from .voice_reference import (
            annotate_sentence_voice_modes,
            pick_whisper_reference_sentence,
        )

        if any(s.voice_mode is None for s in project.sentences):
            annotate_sentence_voice_modes(project, source)
        if project.settings.tts_whisper_reference_sentence_id is None:
            whisper_picked = pick_whisper_reference_sentence(project, source)
            if whisper_picked is not None:
                project.settings.tts_whisper_reference_sentence_id = whisper_picked.id
    failures = synthesize_with_selected_backend(
        project, project_dir, source, force=force, sentence_ids=sentence_ids, **options
    )
    settings = project.settings
    if not settings.tts_quality_check and not settings.tts_asr_quality_check:
        return failures
    from .asr import recognition_session
    from .dub_qc import dub_asr_check

    first_attempt = {sentence.id: sentence.tts_attempt for sentence in project.sentences}
    takes: dict[str, list[_InspectedTake]] = {}
    inspect_ids: set[str] | None = requested
    for _round in range(_QC_RETRIES + 1):
        check_cancelled(cancel_event)
        if settings.tts_asr_quality_check:
            _evict_tts_daemons_if_low_vram(4.5)
        retry: list[Sentence] = []
        # Scope ASR QC models strictly to this inspection pass so GPU VRAM is freed
        # before any IndexTTS child process spawns for retries.
        with recognition_session():
            for sentence in project.sentences:
                check_cancelled(cancel_event)
                if not (
                    sentence.enabled
                    and sentence.chinese_audio_enabled
                    and sentence.tts_file
                    and has_speakable_text(sentence.zh_text)
                    and (inspect_ids is None or sentence.id in inspect_ids)
                ):
                    continue
                try:
                    path = resolve_project_path(project_dir, sentence.tts_file, sentence.id)
                except Exception:
                    continue
                expected = spoken_text(settings, sentence.zh_text)
                used = sentence.tts_attempt - first_attempt.get(sentence.id, 0)
                warning = (
                    dub_problem(path, expected, settings.tts_target_language)
                    if settings.tts_quality_check
                    else None
                )
                broken = warning is not None
                cer = None
                if warning is None and settings.tts_asr_quality_check:
                    warning, cer = dub_asr_check(path, expected, settings)
                sentence.tts_warning = warning
                takes.setdefault(sentence.id, []).append(_InspectedTake.of(sentence, broken, cer))
                if sentence.tts_warning and used < _QC_RETRIES:
                    retry.append(sentence)
        if not retry:
            break
        check_cancelled(cancel_event)
        snapshots: dict[
            str,
            tuple[
                int, str | None, float | None, str | None, str | None, str | None, Any, str | None
            ],
        ] = {}
        for sentence in retry:
            snapshots[sentence.id] = (
                sentence.tts_attempt,
                sentence.tts_file,
                sentence.tts_duration_seconds,
                sentence.tts_cache_key,
                sentence.reference_file,
                sentence.tts_warning,
                sentence.status,
                sentence.error,
            )
            sentence.tts_attempt += 1
        if progress:
            progress(f"配音自动检查：{len(retry)} 句不合格，换随机种子重新生成", 0, len(retry))
        again = synthesize_with_selected_backend(
            project,
            project_dir,
            source,
            force=True,
            sentence_ids=[sentence.id for sentence in retry],
            **options,
        )
        next_inspect_ids: set[str] = set()
        restored_ids: set[str] = set()
        for sentence in retry:
            valid_new = False
            if sentence.status != "error" and sentence.tts_file:
                try:
                    valid_new = resolve_project_path(
                        project_dir, sentence.tts_file, sentence.id
                    ).is_file()
                except Exception:
                    valid_new = False
            if valid_new:
                next_inspect_ids.add(sentence.id)
            else:
                (
                    sentence.tts_attempt,
                    sentence.tts_file,
                    sentence.tts_duration_seconds,
                    sentence.tts_cache_key,
                    sentence.reference_file,
                    sentence.tts_warning,
                    sentence.status,
                    sentence.error,
                ) = snapshots[sentence.id]
                restored_ids.add(sentence.id)
        # A retry that could not produce a new take keeps the earlier, usable one;
        # it is not a missing line, so it must not fail the whole run.
        failures = list(
            dict.fromkeys(
                [
                    *failures,
                    *(item for item in again if item.partition(":")[0] not in restored_ids),
                ]
            )
        )
        if not next_inspect_ids:
            break
        inspect_ids = next_inspect_ids
    for sentence in project.sentences:
        history = takes.get(sentence.id, [])
        if sentence.tts_warning and len(history) > 1:
            _keep_best_take(project_dir, sentence, history)
    return failures
