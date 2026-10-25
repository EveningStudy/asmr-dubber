English | [中文](../MOSS_TRANSCRIBE_DIARIZE.md)

# MOSS-Transcribe-Diarize

Download MOSS on the Models page, select it in recognition settings, and expand **MOSS parameters**. Global settings affect new projects; existing projects keep their own settings.

The adapter runs in an isolated Python environment with Torch 2.11.0 and Transformers 5.6.0. Existing Transformers 4.x backends remain in their own environment. Parakeet remains the default recognizer because MOSS has not been compared against it on annotated Japanese whisper recordings.

## Controls

| Control | Default | Purpose |
|---|---|---|
| Chunk length | 300 seconds | Split near quiet boundaries; longer windows need more memory. |
| Precision | auto | BF16 on supported CUDA devices, otherwise CUDA FP16 or CPU FP32. |
| Attention | sdpa | Portable built-in attention; Flash Attention 2 needs a compatible additional installation. |
| Maximum output tokens | 16384 | Detect and reject truncation instead of saving an incomplete transcript. |
| Context budget | 131072 | Input plus reserved output must fit this limit. |
| Sampling | false | Official greedy inference baseline. |
| Temperature / Top P / Top K | 1.0 / 1.0 / 50 | Used only with sampling enabled. |
| Beam count | 1 | Increasing it is not proven to improve recognition. |
| Repetition penalty | 1.0 | Preserve real repeated words. |
| KV cache | true | Reuse decoder state. |
| Seed | 0 | Sampling reproducibility, without a cross-device bitwise guarantee. |
| Prompt | Official instruction | Includes the required timestamp and speaker format. Clearing restores the built-in prompt. |
| Hotwords | Empty | Append real names or domain terms. |
| Extra generation options | `{}` | Additional Transformers generation arguments. Dedicated controls and input/output contract fields cannot be overridden. |
| Total timeout | 7200 seconds | Includes loading and all chunks; adjustable up to 86400 seconds. |

These 17 fields are also listed with ranges in the [parameter reference](PARAMETERS.md). The allowed 5400-second chunk limit is not a claim that portable hardware can process a 90-minute window. Reduce chunks to 120 or 60 seconds if memory runs out.

## Model downloads and speaker labels

Each model file tries the [user-owned ModelScope mirror](https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Portable-Mirror/files) first, then the pinned official Hugging Face snapshot. The Apache-2.0 license falls back to the pinned official source repository. Sizes and SHA-256 hashes are checked before atomic publication; partial files support resuming. MOSS uses this fallback order even with the older global domestic-only policy.

The pinned model revision is `704aa4a9c304e8520be88901e0d1960158ef5b15`. Runtime dependencies use a separate installer with PyPI mirrors and uv's selected Torch source; no ModelScope offline dependency pack was published for this runtime.

Segment timestamps are preserved without inventing word timings. A label such as `c0001:S01` is stored in the project and displayed in selected-sentence details. Speaker IDs are local to each chunk. There is no automatic cross-chunk identity merging, separate TTS voice assignment, or change to existing subtitle formats. Cancellation and timeout stop the worker; failed output does not replace the previous sentence table.

## Verification limits

Windows CUDA/BF16/SDPA inference passed on an RTX 5070 Ti Laptop with 12 GB VRAM. A 10.435-second public speech clip and a 320.18-second repeated-clip test passed. The longer case produced 56 segments across two chunks in 90.11 seconds. It validates chunk handling, not natural long-recording accuracy. Every mirrored runtime file was downloaded anonymously and its SHA-256 matched.

Japanese ASMR accuracy, real multi-speaker diarization quality, Linux and CPU inference remain unevaluated. See the [validation report](MOSS_VALIDATION_2026-10-16.md), [defaults review](DEFAULTS_AUDIT_2026-10-16.md), [official source](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) and [model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize).
