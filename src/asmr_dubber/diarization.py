"""Multi-speaker diarization and cross-chunk speaker stitching (B14 / Wave 2).

Uses Apache-2.0 CAM++ embeddings (``campplus.py``) when ``campplus_cn_common.bin``
is installed locally, with a pure-NumPy vocal-tract formant + pitch embedding
fallback so speaker clustering works deterministically in all environments.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .campplus import extract_campplus_embeddings, kaldi_fbank_80
from .models import Sentence
from .task_control import CancellationSignal, check_cancelled

Progress = Callable[[str, int, int], None]
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


def _average_linkage_merges(dist: np.ndarray) -> list[tuple[int, int]]:
    """Build an O(N^2) Lance-Williams average-linkage merge tree, respecting +inf cannot-links."""
    n = dist.shape[0]
    if n <= 1:
        return []
    d = np.array(dist, dtype=np.float64, copy=True)
    np.fill_diagonal(d, np.inf)
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=np.float64)
    row_min_idx = np.argmin(d, axis=1)
    row_min_val = d[np.arange(n), row_min_idx]
    merges: list[tuple[int, int]] = []
    for _ in range(n - 1):
        a = int(np.argmin(row_min_val))
        best_val = float(row_min_val[a])
        if not math.isfinite(best_val):
            break
        b = int(row_min_idx[a])
        if a == b:
            break
        if a > b:
            a, b = b, a
        merges.append((a, b))
        na, nb = sizes[a], sizes[b]
        new_row = (na * d[a] + nb * d[b]) / (na + nb)
        new_row[a] = np.inf
        new_row[b] = np.inf
        new_row[~active] = np.inf
        d[a, :] = new_row
        d[:, a] = new_row
        sizes[a] = na + nb
        active[b] = False
        d[b, :] = np.inf
        d[:, b] = np.inf
        row_min_val[b] = np.inf

        idx_a = int(np.argmin(d[a]))
        row_min_idx[a] = idx_a
        row_min_val[a] = d[a, idx_a]

        better = active & (d[:, a] < row_min_val)
        row_min_val[better] = d[better, a]
        row_min_idx[better] = a

        stale = active & ((row_min_idx == a) | (row_min_idx == b))
        stale[a] = False
        for i in np.flatnonzero(stale):
            idx = int(np.argmin(d[i]))
            row_min_idx[i] = idx
            row_min_val[i] = d[i, idx]
    return merges


def _labels_from_merges(n: int, merges: list[tuple[int, int]], k: int) -> list[int]:
    if n <= 0:
        return []
    k = max(1, min(k, n))
    limit = min(len(merges), n - k)
    parent = list(range(n))
    for a, b in merges[:limit]:
        parent[b] = a

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    return [find(i) for i in range(n)]


def _agglomerative_average_cosine(dist: np.ndarray, k: int) -> list[int]:
    n = dist.shape[0]
    if k <= 1 or n <= 1:
        return [0] * n
    merges = _average_linkage_merges(dist)
    return _labels_from_merges(n, merges, min(k, n))


def _silhouette_and_min_sep(
    embeddings: np.ndarray, dist: np.ndarray, labels: list[int]
) -> tuple[float, float]:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return -1.0, 0.0
    label_map = {cid: idx for idx, cid in enumerate(unique)}
    n = len(labels)
    arr = np.fromiter((label_map[lbl] for lbl in labels), dtype=np.int32, count=n)
    k = len(unique)
    counts = np.bincount(arr, minlength=k)
    cluster_means = np.stack([np.mean(dist[:, arr == c], axis=1) for c in range(k)], axis=0)
    idx_range = np.arange(n)
    own_counts = counts[arr]
    own_means = cluster_means[arr, idx_range]
    a_vec = np.where(own_counts > 1, own_means * own_counts / np.maximum(own_counts - 1, 1), 0.0)
    other_means = cluster_means.copy()
    other_means[arr, idx_range] = np.inf
    b_vec = np.min(other_means, axis=0)
    denom = np.maximum(np.maximum(a_vec, b_vec), 1e-9)
    # By definition (Rousseeuw 1987 / sklearn), singleton clusters have silhouette 0.0.
    sil_scores = np.where(own_counts > 1, (b_vec - a_vec) / denom, 0.0)

    centroids: list[np.ndarray] = []
    for c in range(k):
        centroid = np.mean(embeddings[arr == c], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        centroids.append(centroid)
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
    cannot_link: Iterable[tuple[int, int]] | None = None,
) -> list[int]:
    """Cluster speaker embeddings into canonical chronological speaker indices ``0..K-1``."""
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    cos_dist = _cosine_distance_matrix(np.asarray(embeddings, dtype=np.float64))
    merge_dist = cos_dist.copy()
    for i, j in cannot_link or ():
        if 0 <= i < n and 0 <= j < n and i != j:
            merge_dist[i, j] = np.inf
            merge_dist[j, i] = np.inf
    merges = _average_linkage_merges(merge_dist)
    min_valid_k = n - len(merges)
    if num_speakers is not None:
        if num_speakers <= 1 and min_valid_k <= 1:
            return [0] * n
        return _canonicalize_labels(_labels_from_merges(n, merges, num_speakers))

    best_k = min_valid_k
    best_score = 0.16  # Require positive silhouette clarity before splitting a monologue.
    best_labels = _labels_from_merges(n, merges, min_valid_k) if min_valid_k > 1 else [0] * n
    upper_k = min(max_speakers, n)
    for k in range(max(2, min_valid_k), upper_k + 1):
        cand = _labels_from_merges(n, merges, k)
        actual_k = len(set(cand))
        if actual_k < 2:
            continue
        counts = [cand.count(c) for c in set(cand)]
        if n >= 6 and min(counts) < 2 and max(counts) / n > 0.85:
            # Guard against a single cough/breath outlier creating a fake 2nd speaker.
            continue
        sil, min_sep = _silhouette_and_min_sep(embeddings, cos_dist, cand)
        if n == 2 and actual_k == 2:
            if min_sep >= min_separation_cosine or min_valid_k >= 2:
                best_k = 2
                best_labels = cand
            continue
        if min_sep >= min_separation_cosine and sil > best_score:
            best_score = sil
            best_k = actual_k
            best_labels = cand
    return _canonicalize_labels(best_labels) if best_k > 1 else [0] * n


def stereo_pan_features(raw: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Return bounded 2-D binaural ILD cues ``[pan, pan_consistency]`` in ``[-1, 1]``."""
    if raw.ndim < 2 or raw.shape[1] < 2 or raw.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    left = raw[:, 0].astype(np.float64)
    right = raw[:, 1].astype(np.float64)
    size = max(1, round(sample_rate * 0.040))
    hop = max(1, round(sample_rate * 0.020))
    if len(left) < size:
        rms_l = np.array([math.sqrt(float(np.mean(left**2)))], dtype=np.float64)
        rms_r = np.array([math.sqrt(float(np.mean(right**2)))], dtype=np.float64)
    else:
        count = 1 + (len(left) - size) // hop
        idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
        rms_l = np.sqrt(np.mean(left[idx] ** 2, axis=1))
        rms_r = np.sqrt(np.mean(right[idx] ** 2, axis=1))
    frame_peak = np.maximum(rms_l, rms_r)
    active = frame_peak > max(float(frame_peak.max(initial=0.0)) * 0.12, 1e-5)
    if not np.any(active):
        return np.zeros(2, dtype=np.float32)
    total_l = math.sqrt(float(np.mean(rms_l[active] ** 2)))
    total_r = math.sqrt(float(np.mean(rms_r[active] ** 2)))
    ild_db = 20.0 * math.log10((total_r + 1e-7) / (total_l + 1e-7))
    pan = math.tanh(ild_db / 9.0)
    frame_ild = 20.0 * np.log10((rms_r[active] + 1e-7) / (rms_l[active] + 1e-7))
    pan_consistency = float(np.mean(np.tanh(frame_ild / 9.0)))
    return np.array([pan, pan_consistency], dtype=np.float32)


def _read_sentence_clips_and_pan(
    sentences: list[Sentence],
    audio_path: Path,
    *,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[list[int], list[np.ndarray], np.ndarray]:
    indices: list[int] = []
    clips: list[np.ndarray] = []
    pans: list[np.ndarray] = []
    if not sentences or not audio_path.is_file():
        return indices, clips, np.zeros((0, 2), dtype=np.float32)
    total = len(sentences)
    with sf.SoundFile(audio_path) as audio:
        rate = audio.samplerate
        for idx, sent in enumerate(sentences):
            check_cancelled(cancel_event)
            if progress and idx % 50 == 0:
                progress("正在读取说话人音频片段", idx, total)
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
            pan = stereo_pan_features(raw, rate)
            mono = raw[:, int(np.argmax(ear_rms))] if raw.shape[1] >= 2 else raw[:, 0]
            mono_16k = soxr.resample(mono, rate, 16_000, quality="HQ") if rate != 16_000 else mono
            indices.append(idx)
            clips.append(np.asarray(mono_16k, dtype=np.float32))
            pans.append(pan)
    pan_matrix = np.stack(pans, axis=0) if pans else np.zeros((0, 2), dtype=np.float32)
    return indices, clips, pan_matrix


def _read_sentence_clips_16k(
    sentences: list[Sentence],
    audio_path: Path,
    *,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    indices, clips, _ = _read_sentence_clips_and_pan(
        sentences, audio_path, progress=progress, cancel_event=cancel_event
    )
    return indices, clips


def _compute_embeddings(
    clips_16k: list[np.ndarray],
    *,
    stereo_pans: np.ndarray | None = None,
    num_speakers: int | None = None,
    cancel_event: CancellationSignal | None = None,
) -> np.ndarray:
    check_cancelled(cancel_event)
    camp = extract_campplus_embeddings(clips_16k)
    check_cancelled(cancel_event)
    acoustic_list: list[np.ndarray] = []
    for idx, clip in enumerate(clips_16k):
        if idx % 32 == 0:
            check_cancelled(cancel_event)
        acoustic_list.append(acoustic_timbre_embedding(clip))
    acoustic = np.stack(acoustic_list, axis=0)
    if camp is None:
        voice_embs = acoustic
    else:
        # Blend 88% CAM++ x-vector with 12% acoustic F0/formant features for pitch-distinct duets.
        combined = np.concatenate([camp * math.sqrt(0.88), acoustic * math.sqrt(0.12)], axis=1)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        voice_embs = (combined / np.maximum(norms, 1e-9)).astype(np.float32)

    if (
        stereo_pans is not None
        and len(stereo_pans) == len(clips_16k)
        and len(clips_16k) >= 2
        and float(np.std(stereo_pans[:, 0])) >= 0.18
    ):
        # Moderate binaural weight helps split left/right co-actors without splitting
        # one moving solo actor in auto mode; explicit num_speakers uses stronger panning.
        auto_weight = 0.115 if camp is not None else 0.107
        w_stereo = 0.22 if (num_speakers is not None and num_speakers >= 2) else auto_weight
        pan_feats = stereo_pans.astype(np.float32) * math.sqrt(w_stereo / 2.0)
        joined = np.concatenate([voice_embs * math.sqrt(1.0 - w_stereo), pan_feats], axis=1)
        norms = np.linalg.norm(joined, axis=1, keepdims=True)
        return (joined / np.maximum(norms, 1e-9)).astype(np.float32)
    return voice_embs


def stitch_moss_chunk_speakers(
    sentences: list[Sentence],
    audio_path: Path,
    *,
    num_speakers: int | None = None,
    max_speakers: int = 6,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> list[Sentence]:
    """Resolve MOSS chunk-local ``c0001:S01`` speaker IDs into global ``S01``, ``S02``."""
    check_cancelled(cancel_event)
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

    valid_indices, clips, pans = _read_sentence_clips_and_pan(
        sentences, audio_path, progress=progress, cancel_event=cancel_event
    )
    if not clips:
        return sentences
    if progress:
        progress("正在提取说话人声纹", 1, 3)
    embs = _compute_embeddings(
        clips, stereo_pans=pans, num_speakers=num_speakers, cancel_event=cancel_event
    )
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
    tag_chunks = [tag.split(":", 1)[0] for tag in tag_order]
    cannot_link = [
        (i, j)
        for i in range(len(tag_order))
        for j in range(i + 1, len(tag_order))
        if tag_chunks[i] == tag_chunks[j]
    ]
    if progress:
        progress("正在聚类匹配说话人", 2, 3)
    group_labels = cluster_speaker_embeddings(
        group_centroids,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        min_separation_cosine=0.20,
        cannot_link=cannot_link,
    )
    tag_to_global = {
        tag: f"S{lbl + 1:02d}" for tag, lbl in zip(tag_order, group_labels, strict=True)
    }
    used_globals = set(tag_to_global.values())
    fallback_map: dict[str, str] = {}
    next_fallback = len(used_globals) + 1
    result: list[Sentence] = []
    for s in sentences:
        if s.speaker_id in tag_to_global:
            result.append(s.model_copy(update={"speaker_id": tag_to_global[s.speaker_id]}))
        elif s.speaker_id and _MOSS_CHUNK_SPEAKER.match(s.speaker_id):
            if s.speaker_id not in fallback_map:
                while f"S{next_fallback:02d}" in used_globals:
                    next_fallback += 1
                fid = f"S{next_fallback:02d}"
                fallback_map[s.speaker_id] = fid
                used_globals.add(fid)
                next_fallback += 1
            result.append(s.model_copy(update={"speaker_id": fallback_map[s.speaker_id]}))
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
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
) -> list[Sentence]:
    """Assign ``S01``, ``S02``, ... ``speaker_id`` labels across ``sentences``."""
    check_cancelled(cancel_event)
    if not sentences or not audio_path.is_file():
        return sentences
    if any(s.speaker_id and _MOSS_CHUNK_SPEAKER.match(s.speaker_id) for s in sentences):
        return stitch_moss_chunk_speakers(
            sentences,
            audio_path,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
            progress=progress,
            cancel_event=cancel_event,
        )
    if preserve_existing and all(s.speaker_id or not s.enabled for s in sentences):
        return sentences

    valid_indices, clips, pans = _read_sentence_clips_and_pan(
        sentences, audio_path, progress=progress, cancel_event=cancel_event
    )
    if not clips:
        return sentences
    if progress:
        progress("正在提取说话人声纹", 1, 3)
    embs = _compute_embeddings(
        clips, stereo_pans=pans, num_speakers=num_speakers, cancel_event=cancel_event
    )
    check_cancelled(cancel_event)
    if progress:
        progress("正在聚类匹配说话人", 2, 3)
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
