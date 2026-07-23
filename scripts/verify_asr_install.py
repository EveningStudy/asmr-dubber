"""Run one installed ASR backend against a short audio file."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from asmr_dubber.asr import transcribe_japanese
from asmr_dubber.audio import make_analysis_copy
from asmr_dubber.models import ProjectSettings
from asmr_dubber.platforms import portable_home


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--decoder", choices=("tdt", "ctc"), default="tdt")
    args = parser.parse_args()

    source = Path(args.audio).expanduser().resolve()
    temporary = portable_home() / "temp" / f"verify-asr-{uuid.uuid4().hex}.wav"
    try:
        analysis = make_analysis_copy(source, temporary)
        settings = ProjectSettings(
            asr_backend=args.backend,
            asr_model=args.model,
            asr_device=args.device,
            asr_compute_type=args.compute_type,
            asr_parakeet_decoder=args.decoder,
            asr_batch_size=1,
        )
        sentences, language = transcribe_japanese(analysis, settings)
        print(
            json.dumps(
                {
                    "language": language,
                    "sentences": [
                        {
                            "start": sentence.start_seconds,
                            "end": sentence.end_seconds,
                            "text": sentence.ja_text,
                        }
                        for sentence in sentences
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
