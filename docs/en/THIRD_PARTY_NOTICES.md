[中文](../THIRD_PARTY_NOTICES.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Third-party software, models and services

ASMR Dubber code is [MIT licensed](../../LICENSE). This does not relicense model weights, runtimes, media, reference voices, services or generated content. This page identifies boundaries; the actual artifact's license/NOTICE/model card governs.

## Recognition and alignment

| Component | Upstream | License recorded by this project |
|---|---|---|
| CrispASR | [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) | MIT; model terms separate |
| Parakeet CTC 1.1B JA | [Model card](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja) | Apache-2.0 |
| Parakeet TDT/CTC 0.6B JA | [Model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | CC BY 4.0 |
| Kotoba-Whisper v2.2 | [Model card](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | Apache-2.0 |
| Faster-Whisper large-v2 | [Model card](https://huggingface.co/Systran/faster-whisper-large-v2) | MIT; converted from Whisper |
| Faster-Whisper code | [Repository](https://github.com/SYSTRAN/faster-whisper) | See shipped LICENSE; CTranslate2 separate |
| Japanese ASMR VAD | [Model card](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) | MIT |
| Qwen3 ForcedAligner | [Model card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache-2.0 |

Conversion, quantization and mirroring do not remove attribution or redistribution obligations.

## IndexTTS

[IndexTTS2 and IndexTTS-2.5](https://github.com/index-tts/index-tts) have separate upstream model-use terms, including the bilibili Model Use License Agreement. They are not relicensed under this application's MIT license. Read the pinned artifact's `LICENSE`, `LICENSE_ZH.txt` and notices before installation or redistribution; do not infer unrestricted commercial rights from public download availability.

## Runtimes and media

uv, CPython/python-build-standalone, FFmpeg, PyTorch/TorchAudio, Transformers, ONNX Runtime, CTranslate2, edge-tts and dependencies in `pyproject.toml`/`uv.lock` retain their own licenses. FFmpeg obligations depend on actual build/linkage. Preserve all packaged notices. Generate a final artifact inventory when redistributing runtimes or wheelhouses.

Optional audio-separator and each separation model have independent terms. The model catalog is not a redistribution license. Replicate/custom HTTP services are separate providers; verify the selected model's terms and data policy.

## External services

Edge, MiMo, MiniMax, GPT-SoVITS, CosyVoice, Fish and translation providers are client integrations. The application does not grant accounts, credits, server/model licenses or content rights. OpenAI-compatible describes a request format, not a license or provider identity.

Users are responsible for accounts, region, cost/quota, upload authorization, retention/training policies and output restrictions. Model licensing does not grant copyright, performer, personality, privacy or voice-cloning rights over input works. Do not use the tool for impersonation, fraud, harassment or unauthorized cloning.

## Redistribution checklist

Pin exact revisions; retain LICENSE/NOTICE/model cards; verify transformed-weight provenance; inventory transitive binary/wheel dependencies; inspect the final archive; and provide user-managed acquisition for components that cannot be redistributed. `DEPENDENCIES.json` covers core portable wheels, not all isolated model environments. Report discrepancies against the actual artifact license.
