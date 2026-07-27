"""Run opt-in local model smoke tests used during release validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf

from asmr_dubber.asr import transcribe_japanese
from asmr_dubber.audio import probe_audio
from asmr_dubber.model_registry import ASR_BACKENDS
from asmr_dubber.models import DubProject, ProjectSettings, Sentence
from asmr_dubber.tts import synthesize_sentences


def _progress(message: str, current: int, total: int) -> None:
    print(f"[{current}/{total}] {message}", flush=True)


def smoke_asr(
    audio: Path,
    *,
    backend: str,
    model: str,
    device: str,
    compute_type: str,
) -> dict[str, object]:
    settings = ProjectSettings(
        asr_backend=backend,
        asr_model=model,
        asr_device=device,
        asr_compute_type=compute_type,
        asr_batch_size=1,
    )
    sentences, language = transcribe_japanese(audio.resolve(), settings, _progress)
    if not sentences:
        raise RuntimeError(f"{backend} ASR smoke test returned no sentences")
    return {
        "backend": backend,
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "sentences": [
            {
                "start": item.start_seconds,
                "end": item.end_seconds,
                "text": item.ja_text,
            }
            for item in sentences
        ],
    }


def smoke_indextts(reference: Path, workdir: Path, model_dir: Path) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / f"source{reference.suffix.lower()}"
    shutil.copy2(reference, source)
    info = probe_audio(source)
    sentence = Sentence(
        id="s0001",
        start_seconds=0,
        end_seconds=min(info.duration_seconds, 3.5),
        ja_text="今日は一緒にゆっくり休みましょう。",
        zh_text="今天让我们一起慢慢休息吧。",
    )
    settings = ProjectSettings(
        tts_backend="indextts2",
        tts_model="IndexTTS2",
        tts_device="cuda",
        tts_clone_mode="stable_reference",
        tts_reference_sentence_id=sentence.id,
        tts_model_path=str(model_dir.resolve()),
        tts_config_path=str((model_dir / "config.yaml").resolve()),
        tts_index_use_fp16=True,
    )
    project = DubProject(source=info, settings=settings, sentences=[sentence])
    failures = synthesize_sentences(
        project,
        workdir,
        source,
        force=True,
        progress=_progress,
    )
    if failures or not sentence.tts_file:
        raise RuntimeError(f"IndexTTS2 smoke test failed: {failures}")
    output = workdir / sentence.tts_file
    audio_info = sf.info(output)
    if audio_info.frames <= 0:
        raise RuntimeError("IndexTTS2 smoke test produced empty audio")
    return {
        "backend": "indextts2",
        "output": str(output.resolve()),
        "duration": audio_info.duration,
        "sample_rate": audio_info.samplerate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-audio", type=Path)
    parser.add_argument("--asr-backend", choices=ASR_BACKENDS, default="parakeet_nemo")
    parser.add_argument("--asr-model")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument("--indextts-reference", type=Path)
    parser.add_argument("--indextts-model-dir", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path(".smoke-models"))
    args = parser.parse_args()
    if not args.asr_audio and not args.indextts_reference:
        parser.error("provide an ASR audio or IndexTTS2 reference")
    if args.indextts_reference and not args.indextts_model_dir:
        parser.error("--indextts-reference requires --indextts-model-dir")
    result: dict[str, object] = {}
    if args.asr_audio:
        model = args.asr_model or ASR_BACKENDS[args.asr_backend].default_model
        result["asr"] = smoke_asr(
            args.asr_audio,
            backend=args.asr_backend,
            model=model,
            device=args.asr_device,
            compute_type=args.asr_compute_type,
        )
    if args.indextts_reference:
        result["indextts"] = smoke_indextts(
            args.indextts_reference,
            args.workdir / "indextts",
            args.indextts_model_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
