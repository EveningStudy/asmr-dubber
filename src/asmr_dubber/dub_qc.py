"""Closed-loop quality control for synthesized dubs (B12), residual speech (B13),
and sentence-boundary silence snapping (E2).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from .environment import cached_model_path, resolve_transformers_model_source
from .models import ProjectSettings, Sentence
from .zh_phonetics import zh_phonetic_char_cost, zh_syllable_map

logger = logging.getLogger(__name__)

_SPOKEN_CHARS = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]", re.UNICODE)
_JA_CONTENT_CHARS = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u3400-\u9fff]")
_JA_BREATH_NOISE = re.compile(r"^[ぁあぃいぅうぇえぉおんっーはふすちゅぱれろぺろ\s。、！？…・]+$")

_DIGIT_TO_ZH = str.maketrans("0123456789", "零一二三四五六七八九")
# Common simplified/traditional variants Whisper may emit on Mandarin audio.
_TRAD_TO_SIMP = str.maketrans(
    "這個人們來時會說愛聽風聲點開關門長間東車馬魚鳥語話請讓給對進還過後裡頭發見覺親體醫雙難願類顯顧",
    "这个人门来时会说爱听风声点开关门长间东车马鱼鸟语话请让给对进还过后里头发见觉亲体医双难愿类显顾",
)


def normalize_for_cer(text: str, language: str = "zh") -> str:
    """Normalize text to comparable spoken units so punctuation never inflates CER."""
    if language == "en":
        words = re.findall(r"[a-z0-9']+", text.lower())
        return " ".join(words)
    lowered = text.lower().translate(_DIGIT_TO_ZH).translate(_TRAD_TO_SIMP)
    chars = _SPOKEN_CHARS.findall(lowered)
    return "".join(chars)


def edit_distance(a: str | list[str], b: str | list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def character_error_rate(expected: str, recognized: str, language: str = "zh") -> float:
    """Compute normalized phonetic/word Error Rate in [0.0, +inf)."""
    ref_norm = normalize_for_cer(expected, language)
    hyp_norm = normalize_for_cer(recognized, language)
    if not ref_norm:
        return 0.0
    if not hyp_norm:
        return 1.0
    if language == "en":
        ref_units = ref_norm.split()
        hyp_units = hyp_norm.split()
        return edit_distance(ref_units, hyp_units) / max(1, len(ref_units))
    table = zh_syllable_map()
    if not table:
        return edit_distance(ref_norm, hyp_norm) / max(1, len(ref_norm))
    previous = [float(j) for j in range(len(hyp_norm) + 1)]
    for i, ch_a in enumerate(ref_norm, start=1):
        current = [float(i)]
        for j, ch_b in enumerate(hyp_norm, start=1):
            sub = zh_phonetic_char_cost(ch_a, ch_b, table)
            current.append(min(current[-1] + 1.0, previous[j] + 1.0, previous[j - 1] + sub))
        previous = current
    return previous[-1] / max(1, len(ref_norm))


def _read_16k_mono(path: Path) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(data):
        return np.zeros(0, dtype=np.float32)
    mono = data.mean(axis=1, dtype=np.float32)
    if rate != 16_000:
        mono = soxr.resample(mono, rate, 16_000, quality="HQ")
    arr = np.asarray(mono, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak >= 5e-4:
        rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        if rms > 1e-6:
            # Lift quiet whispered dubs toward -19 dBFS (up to +14 dB) for reliable ASR QC.
            gain = min(5.0, (10 ** (-19.0 / 20.0)) / rms, 0.95 / peak)
            if gain > 1.05:
                arr = (arr * gain).astype(np.float32)
    return arr


def transcribe_dub_clip(
    audio_path: Path,
    language: str,
    settings: ProjectSettings,
) -> str | None:
    """Recognize a synthesized dub clip using any locally installed ASR model.

    Returns ``None`` when no compatible local Chinese/English recognizer is installed,
    so QC degrades gracefully without failing synthesis.
    """
    from .asr import _session_model

    waveform = _read_16k_mono(audio_path)
    if len(waveform) < 800 or float(np.max(np.abs(waveform))) < 1e-5:
        return ""

    # 1. Prefer Qwen3-ASR if installed (supports zh & en natively in VRAM).
    qwen_id = "Qwen/Qwen3-ASR-1.7B"
    if cached_model_path(qwen_id) is not None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            source, revision = resolve_transformers_model_source(qwen_id)
            use_cuda = settings.asr_device.startswith("cuda") and torch.cuda.is_available()
            dtype = (
                torch.bfloat16
                if use_cuda and torch.cuda.is_bf16_supported()
                else (torch.float16 if use_cuda else torch.float32)
            )
            device_map = settings.asr_device if use_cuda else "cpu"
            kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device_map}
            if revision is not None:
                kwargs["revision"] = revision
            model = _session_model(
                f"dub-qc-qwen3:{source}:{device_map}",
                lambda: Qwen3ASRModel.from_pretrained(source, max_new_tokens=256, **kwargs),
            )
            lang_name = "English" if language == "en" else "Chinese"
            res = model.transcribe(
                audio=[(waveform, 16_000)],
                context="",
                language=lang_name,
                return_time_stamps=False,
            )
            return str(res[0].text or "").strip() if res else ""
        except Exception:
            logger.debug("Qwen3-ASR dub QC skipped; trying faster-whisper.", exc_info=True)

    # 2. Fallback to faster-whisper large-v2 if installed locally.
    fw_id = "Systran/faster-whisper-large-v2"
    fw_path = cached_model_path(fw_id)
    if fw_path is not None:
        try:
            from faster_whisper import WhisperModel

            device = "cuda" if settings.asr_device.startswith("cuda") else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            model = _session_model(
                f"dub-qc-fw:{fw_path}:{device}",
                lambda: WhisperModel(str(fw_path), device=device, compute_type=compute),
            )
            segments, _ = model.transcribe(
                waveform,
                language="en" if language == "en" else "zh",
                beam_size=2,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return "".join(seg.text for seg in segments).strip()
        except Exception:
            logger.debug("faster-whisper dub QC unavailable.", exc_info=True)

    return None


def dub_asr_check(
    audio_path: Path,
    expected_text: str,
    settings: ProjectSettings,
) -> tuple[str | None, float | None]:
    """Back-listen to a dub: the QC warning (>10% phonetic CER) and the CER itself.

    The CER is ``None`` when the line was not checked (no local recogniser, or an
    interjection too short to judge).
    """
    target_lang = settings.tts_target_language
    norm_expected = normalize_for_cer(expected_text, target_lang)
    unit_count = len(norm_expected.split()) if target_lang == "en" else len(norm_expected)
    if unit_count < 1:
        return None, None
    if unit_count == 1 and norm_expected in {"嗯", "啊", "唔", "哦", "噢", "喔", "哈", "嘿"}:
        return None, None
    recognized = transcribe_dub_clip(audio_path, target_lang, settings)
    if recognized is None:
        return None, None
    cer = character_error_rate(expected_text, recognized, target_lang)
    if cer > 0.10:
        preview = recognized[:24] if recognized else "（未识别出语音）"
        return f"配音回听字错率 {cer * 100:.0f}%（识别为「{preview}」）", cer
    return None, cer


def dub_asr_problem(
    audio_path: Path,
    expected_text: str,
    settings: ProjectSettings,
    *,
    is_final_attempt: bool = False,
) -> str | None:
    """Return a Chinese QC warning if ASR back-listening detects >10% phonetic CER (B12)."""
    return dub_asr_check(audio_path, expected_text, settings)[0]


def meaningful_japanese_residual(text: str) -> str:
    """Extract substantive Japanese characters, ignoring pure breath/sigh hallucinations."""
    cleaned = text.strip()
    if not cleaned or _JA_BREATH_NOISE.match(cleaned):
        return ""
    chars = "".join(_JA_CONTENT_CHARS.findall(cleaned))
    if len(chars) <= 1 and chars in {"あ", "ん", "っ", "は", "ふ", "え", "お", "う"}:
        return ""
    return chars


def scan_residual_japanese_windows(
    bed_audio_path: Path,
    entries: list[dict[str, Any]],
    settings: ProjectSettings,
) -> bool:
    """Run anime-whisper over subtracted replacement windows to flag audible Japanese (B13).

    Mutates ``entries`` in place with ``recognized_residual`` and ``residual_chars``.
    Returns ``True`` if an ASR model ran, ``False`` if unavailable.
    """
    from .asr import (
        _REVIEW_MODELS,
        _run_transformers_asr_pipeline,
        _session_model,
        recognition_session,
    )
    from .asr_candidates import ANIME_WHISPER_MODEL

    if not entries or cached_model_path(ANIME_WHISPER_MODEL) is None:
        return False
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError:
        return False

    clips: list[np.ndarray] = []
    indices: list[int] = []
    with sf.SoundFile(bed_audio_path) as bed:
        rate = bed.samplerate
        for idx, entry in enumerate(entries):
            lo = max(0, round(float(entry["start"]) * rate))
            hi = min(bed.frames, round(float(entry["end"]) * rate))
            if hi - lo < rate // 5:
                continue
            bed.seek(lo)
            raw = bed.read(hi - lo, dtype="float32", always_2d=True).mean(axis=1)
            if float(np.sqrt(np.mean(raw.astype(np.float64) ** 2))) < 10 ** (-52.0 / 20.0):
                continue
            mono_16k = soxr.resample(raw, rate, 16_000, quality="HQ") if rate != 16_000 else raw
            clips.append(np.asarray(mono_16k, dtype=np.float32))
            indices.append(idx)

    if not clips:
        for entry in entries:
            entry.setdefault("recognized_residual", "")
            entry.setdefault("residual_chars", 0)
        return True

    use_cuda = settings.asr_device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = settings.asr_device if use_cuda else "cpu"
    source, revision = resolve_transformers_model_source(ANIME_WHISPER_MODEL)

    def _load_pipe() -> Any:
        proc = AutoProcessor.from_pretrained(source, revision=revision)
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
            source, revision=revision, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        return pipeline(
            "automatic-speech-recognition",
            model=mdl,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            dtype=dtype,
            device=device,
        )

    def _run_scan() -> list[Any]:
        pipe = _session_model(f"anime-whisper:{source}:{revision}:{device}:{dtype}", _load_pipe)
        outputs = _run_transformers_asr_pipeline(
            pipe,
            [{"array": clip, "sampling_rate": 16_000} for clip in clips],
            batch_size=8,
            generate_kwargs={"language": "ja", "task": "transcribe", "no_repeat_ngram_size": 5},
        )
        return outputs if isinstance(outputs, list) else [outputs]

    if _REVIEW_MODELS.get() is None:
        with recognition_session():
            out_list = _run_scan()
    else:
        out_list = _run_scan()
    for entry in entries:
        entry.setdefault("recognized_residual", "")
        entry.setdefault("residual_chars", 0)
    for idx, result in zip(indices, out_list, strict=True):
        raw_text = str(result.get("text", "")).strip() if isinstance(result, dict) else ""
        content = meaningful_japanese_residual(raw_text)
        entries[idx]["recognized_residual"] = raw_text if content else ""
        entries[idx]["residual_chars"] = len(content)
        if len(content) >= 2 and "residual_speech" not in entries[idx]["reasons"]:
            entries[idx]["reasons"].insert(0, "residual_speech")
            entries[idx]["exceeds"] = True
    return True


def snap_sentences_to_silence(
    sentences: list[Sentence],
    audio_path: Path,
    *,
    outward_ms: float = 90.0,
    inward_ms: float = 50.0,
) -> list[Sentence]:
    """Snap sentence start/end timestamps to local 10 ms energy valleys (E2).

    Prevents forced-aligner jitter from slicing off leading consonants or trailing
    particles while guaranteeing neighbouring sentences never cross or overlap.
    """
    if not sentences or not audio_path.is_file():
        return sentences
    try:
        with sf.SoundFile(audio_path) as handle:
            rate = handle.samplerate
            hop = max(1, round(rate * 0.010))
            blocks: list[np.ndarray] = []
            for chunk in handle.blocks(blocksize=hop * 2000, dtype="float32", always_2d=True):
                mono_sq = np.max(chunk.astype(np.float64) ** 2, axis=1)
                pad = (-len(mono_sq)) % hop
                if pad:
                    mono_sq = np.pad(mono_sq, (0, pad))
                blocks.append(np.sqrt(mono_sq.reshape(-1, hop).mean(axis=1)))
            envelope = np.concatenate(blocks) if blocks else np.zeros(0)
            duration = handle.frames / rate
    except (OSError, RuntimeError):
        return sentences
    if len(envelope) < 4:
        return sentences

    hop_s = hop / rate
    outward = outward_ms / 1000.0
    inward = inward_ms / 1000.0
    ordered = sorted(sentences, key=lambda s: (s.start_seconds, s.end_seconds, s.id))

    def _valley(nominal: float, lo: float, hi: float) -> float:
        if hi <= lo + hop_s:
            return float(np.clip(nominal, lo, hi))
        i_lo = max(0, int(np.floor(lo / hop_s)))
        i_hi = min(len(envelope), int(np.ceil(hi / hop_s)) + 1)
        if i_hi - i_lo < 2:
            return float(np.clip(nominal, lo, hi))
        sec = envelope[i_lo:i_hi]
        valley_floor = float(sec.min()) * 1.35 + 1e-8
        nom_clamped = float(np.clip(nominal, lo, hi))
        nom_idx = min(len(envelope) - 1, max(0, int(nom_clamped / hop_s)))
        if envelope[nom_idx] <= valley_floor:
            return nom_clamped
        candidates = np.flatnonzero(sec <= valley_floor)
        times = (i_lo + candidates + 0.5) * hop_s
        best = float(times[np.argmin(np.abs(times - nominal))])
        return float(np.clip(best, lo, hi))

    snapped: list[Sentence] = []
    prev_end = 0.0
    for idx, sent in enumerate(ordered):
        next_start = ordered[idx + 1].start_seconds if idx + 1 < len(ordered) else duration
        lo_s = max(prev_end, sent.start_seconds - outward)
        hi_s = min(sent.end_seconds - 0.04, sent.start_seconds + inward)
        new_start = _valley(sent.start_seconds, lo_s, max(lo_s, hi_s))

        lo_e = max(new_start + 0.04, sent.end_seconds - inward)
        hi_e = min(next_start, duration, sent.end_seconds + outward)
        new_end = _valley(sent.end_seconds, min(lo_e, hi_e), hi_e)
        if new_end <= new_start + 0.02:
            new_start, new_end = sent.start_seconds, sent.end_seconds
        prev_end = new_end
        snapped.append(
            sent.model_copy(
                update={
                    "start_seconds": round(new_start, 3),
                    "end_seconds": round(new_end, 3),
                }
            )
        )
    return snapped


def detect_uncovered_vocal_sounds(
    sentences: list[Sentence],
    vocals_path: Path,
    *,
    min_duration_seconds: float = 0.50,
) -> list[Sentence]:
    """Add ``skipped_sound`` rows for sustained separated-vocal activity missed by ASR.

    Separated ``vocals.wav`` already excludes instrumental/room foley, so uncovered
    regions >= 0.50 s within 28 dB of the vocal peak are non-verbal kisses, ear-licks
    or breaths. Recording them protects their audio from dubbing overlap.
    """
    if not vocals_path.is_file():
        return sentences
    try:
        with sf.SoundFile(vocals_path) as handle:
            rate = handle.samplerate
            hop = max(1, round(rate * 0.020))
            chunks: list[np.ndarray] = []
            for block in handle.blocks(blocksize=hop * 1000, dtype="float32", always_2d=True):
                sq = np.mean(block.astype(np.float64) ** 2, axis=1)
                pad = (-len(sq)) % hop
                if pad:
                    sq = np.pad(sq, (0, pad))
                chunks.append(np.sqrt(sq.reshape(-1, hop).mean(axis=1)))
            env = np.concatenate(chunks) if chunks else np.zeros(0)
            total_dur = handle.frames / rate
    except (OSError, RuntimeError):
        return sentences
    if len(env) < 10:
        return sentences
    hop_s = hop / rate
    ref = float(np.percentile(env, 99))
    if ref <= 1e-6:
        return sentences
    active = env > max(ref * 10 ** (-28.0 / 20.0), 1e-5)
    covered = np.zeros(len(active), dtype=bool)
    for s in sentences:
        lo = max(0, int((s.start_seconds - 0.06) / hop_s))
        hi = min(len(covered), int(np.ceil((s.end_seconds + 0.06) / hop_s)))
        covered[lo:hi] = True
    uncovered = active & ~covered
    idx = np.flatnonzero(uncovered)
    if not len(idx):
        return sentences

    extra: list[Sentence] = []
    run_s = int(idx[0])
    run_p = int(idx[0])
    bridge = max(1, round(0.14 / hop_s))

    def _emit(a: int, b: int) -> None:
        t0 = round(a * hop_s, 3)
        t1 = round(min(total_dur, (b + 1) * hop_s), 3)
        if t1 - t0 < min_duration_seconds:
            return
        if any(min(t1, s.end_seconds) - max(t0, s.start_seconds) > 0.04 for s in sentences):
            return
        extra.append(
            Sentence(
                id=f"s8{len(extra):05d}",
                start_seconds=t0,
                end_seconds=t1,
                source_text="（吐息）",
                enabled=False,
                chinese_audio_enabled=False,
                status="skipped_sound",
            )
        )

    for x_raw in idx[1:]:
        x = int(x_raw)
        if x <= run_p + bridge:
            run_p = x
        else:
            _emit(run_s, run_p)
            run_s = run_p = x
    _emit(run_s, run_p)
    if not extra:
        return sentences
    merged = sorted([*sentences, *extra], key=lambda s: (s.start_seconds, s.end_seconds, s.id))
    for i, s in enumerate(merged, start=1):
        s.id = f"s{i:06d}"
    return merged
