"""Multi-speaker diarization and cross-chunk speaker stitching (B14 / Wave 2).

Uses Apache-2.0 CAM++ embeddings (``campplus.py``) when ``campplus_cn_common.bin``
is installed locally, with a pure-NumPy vocal-tract formant + pitch embedding
fallback so speaker clustering works deterministically in all environments.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .campplus import extract_campplus_embeddings, kaldi_fbank_80
from .models import Sentence

_MOSS_CHUNK_SPEAKER = re.compile(r"^c\d{4}:S\d+$")


def _estimate_f0_hz(clip_16k: np.ndarray, sample_rate: int = 16_000) -> tuple[float, float]:
    """Return ``(median_f0_hz, voiced_fraction)`` over active 40 ms frames."""
    size, hop = round(sample_rate * 0.040), round(sample_rate * 0.010)
    if len(clip_16k) < size:
        return 0.0, 0.0
    count = 1 + (len(clip_16k) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    frames = clip_16k[idx].astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active = rms > max(float(rms.max(initial=0.0)) * 0.12, 1e-5)
    if not np.any(active):
        return 0.0, 0.0
    voiced = frames[active] - np.mean(frames[active], axis=1, keepdims=True)
    spec = np.fft.rfft(voiced * np.hanning(size)[None, :], 4 * size, axis=1)
    corr = np.fft.irfft(np.abs(spec) ** 2, axis=1)
    corr /= np.maximum(corr[:, :1], 1e-20)
    lo_lag = max(1, round(sample_rate / 420.0))
    hi_lag = min(corr.shape[1] - 1, round(sample_rate / 75.0))
    sub = corr[:, lo_lag : hi_lag + 1]
    best_rel = np.argmax(sub, axis=1)
    peaks = sub[np.arange(len(sub)), best_rel]
    mask = peaks > 0.45
    voiced_frac = float(np.mean(mask))
    if not np.any(mask):
        return 0.0, voiced_frac
    lags = lo_lag + best_rel[mask]
    f0s = sample_rate / np.maximum(lags.astype(np.float64), 1.0)
    return float(np.median(f0s)), voiced_frac


def acoustic_timbre_embedding(clip_16k: np.ndarray) -> np.ndarray:
    """Pure-NumPy 36-D vocal-tract cepstral + F0 speaker vector (L2-normalized)."""
    sig = np.asarray(clip_16k, dtype=np.float32).reshape(-1)
    if len(sig) < 400 or float(np.max(np.abs(sig))) < 1e-6:
        return np.zeros(36, dtype=np.float32)
    fbank = kaldi_fbank_80(sig, 16_000)
    # Type-II DCT of 80-bin Mel filterbank yields vocal-tract MFCCs (formant envelope).
    k = np.arange(1, 17, dtype=np.float64)[:, None]
    n = np.arange(80, dtype=np.float64)[None, :]
    basis = np.cos(math.pi * k * (n + 0.5) / 80.0)
    cep = fbank.astype(np.float64) @ basis.T
    mean_cep = np.mean(cep, axis=0)
    std_cep = np.std(cep, axis=0) * 0.5
    f0_hz, voiced_frac = _estimate_f0_hz(sig, 16_000)
    # Encode log-F0 around 180 Hz center with 3 dimensions so pitch separation is clear.
    log_f0 = math.log2(f0_hz / 180.0) * 4.5 if f0_hz > 0.0 else 0.0
    pitch_feats = np.array(
        [log_f0, log_f0 * voiced_frac, voiced_frac * 1.5, float(np.mean(fbank[:, :20]))],
        dtype=np.float64,
    )
    vec = np.concatenate([mean_cep, std_cep, pitch_feats])
    norm = float(np.linalg.norm(vec))
    return (vec / max(norm, 1e-9)).astype(np.float32)


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-9)
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def _agglomerative_average_cosine(dist: np.ndarray, k: int) -> list[int]:
    n = dist.shape[0]
    if k <= 1 or n <= 1:
        return [0] * n
    k = min(k, n)
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best_pair = (0, 1)
        best_d = math.inf
        for i in range(len(clusters)):
            idx_i = clusters[i]
            for j in range(i + 1, len(clusters)):
                idx_j = clusters[j]
                d = float(np.mean(dist[np.ix_(idx_i, idx_j)]))
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        a, b = best_pair
        clusters[a].extend(clusters[b])
        del clusters[b]
    labels = [0] * n
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


def _silhouette_and_min_sep(
    embeddings: np.ndarray, dist: np.ndarray, labels: list[int]
) -> tuple[float, float]:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return -1.0, 0.0
    arr = np.asarray(labels, dtype=np.int32)
    sil_scores: list[float] = []
    for idx in range(len(labels)):
        own = arr == arr[idx]
        own[idx] = False
        if not np.any(own):
            # By definition (Rousseeuw 1987 / sklearn), singleton clusters have silhouette 0.0.
            sil_scores.append(0.0)
            continue
        a = float(np.mean(dist[idx, own]))
        b = min(
            float(np.mean(dist[idx, arr == other]))
            for other in unique
            if other != arr[idx] and np.any(arr == other)
        )
        denom = max(a, b, 1e-9)
        sil_scores.append((b - a) / denom)

    centroids: list[np.ndarray] = []
    for cid in unique:
        c = np.mean(embeddings[arr == cid], axis=0)
        c /= max(float(np.linalg.norm(c)), 1e-9)
        centroids.append(c)
    min_sep = min(
        1.0 - float(np.clip(np.dot(centroids[i], centroids[j]), -1.0, 1.0))
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    )
    return float(np.mean(sil_scores)), min_sep


def _canonicalize_labels(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = len(mapping)
        out.append(mapping[lbl])
    return out


def cluster_speaker_embeddings(
    embeddings: np.ndarray,
    *,
    num_speakers: int | None = None,
    max_speakers: int = 6,
    min_separation_cosine: float = 0.24,
) -> list[int]:
    """Cluster speaker embeddings into canonical chronological speaker indices ``0..K-1``."""
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1 or (num_speakers is not None and num_speakers <= 1):
        return [0] * n
    dist = _cosine_distance_matrix(np.asarray(embeddings, dtype=np.float64))
    if num_speakers is not None:
        return _canonicalize_labels(_agglomerative_average_cosine(dist, num_speakers))

    best_k = 1
    best_score = 0.16  # Require positive silhouette clarity before splitting a monologue.
    best_labels = [0] * n
    upper_k = min(max_speakers, n)
    for k in range(2, upper_k + 1):
        cand = _agglomerative_average_cosine(dist, k)
        counts = [cand.count(c) for c in set(cand)]
        if n >= 6 and min(counts) < 2 and max(counts) / n > 0.85:
            # Guard against a single cough/breath outlier creating a fake 2nd speaker.
            continue
        sil, min_sep = _silhouette_and_min_sep(embeddings, dist, cand)
        if min_sep >= min_separation_cosine and sil > best_score:
            best_score = sil
            best_k = k
            best_labels = cand
    return _canonicalize_labels(best_labels) if best_k > 1 else [0] * n


def _read_sentence_clips_16k(
    sentences: list[Sentence],
    audio_path: Path,
) -> tuple[list[int], list[np.ndarray]]:
    indices: list[int] = []
    clips: list[np.ndarray] = []
    if not sentences or not audio_path.is_file():
        return indices, clips
    with sf.SoundFile(audio_path) as audio:
        rate = audio.samplerate
        for idx, sent in enumerate(sentences):
            if sent.status == "skipped_sound" or sent.end_seconds - sent.start_seconds < 0.18:
                continue
            lo = max(0, min(audio.frames, round(sent.start_seconds * rate)))
            hi = max(lo, min(audio.frames, round(sent.end_seconds * rate)))
            if hi - lo < rate // 6:
                continue
            audio.seek(lo)
            raw = audio.read(hi - lo, dtype="float32", always_2d=True)
            ear_rms = np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=0))
            if float(ear_rms.max(initial=0.0)) < 1e-5:
                continue
            mono = raw[:, int(np.argmax(ear_rms))] if raw.shape[1] >= 2 else raw[:, 0]
            mono_16k = soxr.resample(mono, rate, 16_000, quality="HQ") if rate != 16_000 else mono
            indices.append(idx)
            clips.append(np.asarray(mono_16k, dtype=np.float32))
    return indices, clips


def _compute_embeddings(clips_16k: list[np.ndarray]) -> np.ndarray:
    camp = extract_campplus_embeddings(clips_16k)
    acoustic = np.stack([acoustic_timbre_embedding(c) for c in clips_16k], axis=0)
    if camp is None:
        return acoustic
    # Blend 88% CAM++ x-vector with 12% acoustic F0/formant features for pitch-distinct duets.
    combined = np.concatenate([camp * math.sqrt(0.88), acoustic * math.sqrt(0.12)], axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return (combined / np.maximum(norms, 1e-9)).astype(np.float32)


def stitch_moss_chunk_speakers(
    sentences: list[Sentence],
    audio_path: Path,
    *,
    num_speakers: int | None = None,
    max_speakers: int = 6,
) -> list[Sentence]:
    """Resolve MOSS chunk-local ``c0001:S01`` speaker IDs into global ``S01``, ``S02``."""
    chunk_tags = [
        s.speaker_id or ""
        for s in sentences
        if s.speaker_id and _MOSS_CHUNK_SPEAKER.match(s.speaker_id)
    ]
    if not chunk_tags:
        return sentences
    unique_chunks = {tag.split(":", 1)[0] for tag in chunk_tags}
    if len(unique_chunks) == 1 and num_speakers is None:
        # Single chunk: MOSS local IDs are already globally consistent; strip chunk prefix.
        mapping: dict[str, str] = {}
        updated: list[Sentence] = []
        for s in sentences:
            tag = s.speaker_id
            if tag and _MOSS_CHUNK_SPEAKER.match(tag):
                mapping.setdefault(tag, f"S{len(mapping) + 1:02d}")
                updated.append(s.model_copy(update={"speaker_id": mapping[tag]}))
            else:
                updated.append(s)
        return updated

    valid_indices, clips = _read_sentence_clips_16k(sentences, audio_path)
    if not clips:
        return sentences
    embs = _compute_embeddings(clips)
    by_tag: dict[str, list[np.ndarray]] = {}
    tag_order: list[str] = []
    for sent_idx, emb in zip(valid_indices, embs, strict=True):
        tag = sentences[sent_idx].speaker_id
        if tag and _MOSS_CHUNK_SPEAKER.match(tag):
            if tag not in by_tag:
                by_tag[tag] = []
                tag_order.append(tag)
            by_tag[tag].append(emb)
    if not tag_order:
        return sentences
    group_centroids = np.stack(
        [
            (m := np.mean(by_tag[t], axis=0)) / max(float(np.linalg.norm(m)), 1e-9)
            for t in tag_order
        ],
        axis=0,
    )
    group_labels = cluster_speaker_embeddings(
        group_centroids,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        min_separation_cosine=0.20,
    )
    tag_to_global = {
        tag: f"S{lbl + 1:02d}" for tag, lbl in zip(tag_order, group_labels, strict=True)
    }
    result: list[Sentence] = []
    for s in sentences:
        if s.speaker_id in tag_to_global:
            result.append(s.model_copy(update={"speaker_id": tag_to_global[s.speaker_id]}))
        elif s.speaker_id and _MOSS_CHUNK_SPEAKER.match(s.speaker_id):
            local = s.speaker_id.split(":", 1)[1]
            result.append(s.model_copy(update={"speaker_id": local}))
        else:
            result.append(s)
    return result


def diarize_sentences(
    sentences: list[Sentence],
    audio_path: Path,
    *,
    num_speakers: int | None = None,
    max_speakers: int = 6,
    preserve_existing: bool = False,
) -> list[Sentence]:
    """Assign ``S01``, ``S02``, ... ``speaker_id`` labels across ``sentences``."""
    if not sentences or not audio_path.is_file():
        return sentences
    if any(s.speaker_id and _MOSS_CHUNK_SPEAKER.match(s.speaker_id) for s in sentences):
        return stitch_moss_chunk_speakers(
            sentences,
            audio_path,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
        )
    if preserve_existing and all(s.speaker_id or not s.enabled for s in sentences):
        return sentences

    valid_indices, clips = _read_sentence_clips_16k(sentences, audio_path)
    if not clips:
        return sentences
    embs = _compute_embeddings(clips)
    # Use a slightly higher separation bar when CAM++ is active so modal vs whisper of
    # the same actor stays unified unless two distinct vocal tracts are present.
    min_sep = 0.28 if embs.shape[1] > 64 else 0.22
    raw_labels = cluster_speaker_embeddings(
        embs,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        min_separation_cosine=min_sep,
    )

    assigned: dict[int, str] = {
        sent_idx: f"S{lbl + 1:02d}" for sent_idx, lbl in zip(valid_indices, raw_labels, strict=True)
    }
    # Smooth isolated ultra-short (<0.45s) back-and-forth flips inside rapid speech.
    if len(valid_indices) >= 3 and len(set(raw_labels)) >= 2:
        for pos in range(1, len(valid_indices) - 1):
            cur_i = valid_indices[pos]
            prev_i = valid_indices[pos - 1]
            next_i = valid_indices[pos + 1]
            cur_s = sentences[cur_i]
            if (
                cur_s.end_seconds - cur_s.start_seconds < 0.45
                and assigned[prev_i] == assigned[next_i]
                and assigned[cur_i] != assigned[prev_i]
                and cur_s.start_seconds - sentences[prev_i].end_seconds < 0.80
                and sentences[next_i].start_seconds - cur_s.end_seconds < 0.80
            ):
                assigned[cur_i] = assigned[prev_i]

    output: list[Sentence] = []
    last_speaker = next(iter(assigned.values()), "S01")
    for idx, sent in enumerate(sentences):
        if preserve_existing and sent.speaker_id:
            last_speaker = sent.speaker_id
            output.append(sent)
            continue
        if sent.status == "skipped_sound":
            output.append(sent)
            continue
        spk = assigned.get(idx, last_speaker)
        last_speaker = spk
        output.append(sent.model_copy(update={"speaker_id": spk}))
    return output
