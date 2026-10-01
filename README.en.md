[中文](README.md) | English

# ASMR Dubber

[![Release](https://img.shields.io/github/v/release/EveningStudy/asmr-dubber?label=release)](https://github.com/EveningStudy/asmr-dubber/releases/latest)
![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D4)
![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn Japanese, English or Chinese audio/video into Chinese or English dubbing, bilingual audio and subtitles. Supports voice cloning, batch processing and subtitle-only export. Recognition, translation, synthesis and mixing are separate stages; sentence text and timing remain editable. Language support depends on the selected backend.

## Demo

Headphones recommended.

### Sample 1 · RTF

Bilingual audio: the Japanese original is retained and mixed with Chinese dubbing processed by RTF (original spatial-cue transfer).

https://github.com/user-attachments/assets/2eddb029-1a6d-4daf-8f53-741b46141f4d

### Sample 2 · RTF

Bilingual audio: RTF-processed Chinese dubbing is mixed with the retained Japanese original.

https://github.com/user-attachments/assets/8068b982-95c0-4561-8d18-f82cb8cebb6e

### Sample 3 · RTF

Bilingual audio: the Japanese original plus RTF-processed Chinese dubbing. Compare it with the vocal-separation version below.

https://github.com/user-attachments/assets/4e7618af-ad3d-4931-9749-4b39816d01d8

### Sample 3 · RTF + vocal separation

Chinese replacement dub: separate Japanese vocals from the background, recognize and translate the speech, then synthesize Chinese dubbing. RTF transfers spatial cues from the original to the dub before mixing it with the separated background. Separation may leave Japanese speech or alter sound effects; complete removal is not guaranteed.

https://github.com/user-attachments/assets/a8841fb0-e4f1-4383-92bb-942805ff1adc

[Watch the introduction and demo of an older version on Bilibili](https://www.bilibili.com/video/BV1f43G6YEov/). Please refer to documents for current information. Media is used for demonstration; contact the maintainer about rights concerns.

## Install

On Windows, download the portable ZIP from [Releases](https://github.com/EveningStudy/asmr-dubber/releases/latest). Extract it completely to a short, writable path such as `D:\Apps\ASMR-Dubber`.

> Enable Windows long paths before use. If available, open Settings → System → Advanced → File Explorer → Enable long paths, then restart ASMR Dubber. Deep paths in third-party runtimes can otherwise fail even when the file exists. If your Windows version does not expose the toggle, use a short directory such as `D:\ASMR-Dubber` to reduce the risk.

![Windows long-path setting](assets/windows-enable-long-paths.png)

1. Run `ASMR-Dubber-Setup.exe`; choose a language and installation profile. Chinese is the default.
2. Run `ASMR-Dubber.exe` and keep its terminal open.
3. Check **Settings → Devices & models**, then configure ASR, translation and TTS.

Use **中文 / English** at the top of the web UI to switch interface language without restarting. This does not change the source language, prompts, dubbing target or project contents.

No preinstalled Python, uv, Git, FFmpeg or CUDA Toolkit is required. Downloads need system `curl.exe`; local GPU inference needs a compatible NVIDIA driver. See [Installation](docs/en/INSTALLATION.md).

Linux x86_64, including WSL2:

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

Requires `bash`, `curl`, `tar` and `getconf`. ARM64 and macOS are not supported.

| Profile | Contents |
|---|---|
| Core | UI, media tools, Edge TTS and API clients; no large models |
| Recommended | Core + two Japanese Parakeet models; IndexTTS2 on NVIDIA machines |
| Advanced | Recommended components + Kotoba, Faster-Whisper, ASMR VAD and Qwen3 alignment |

IndexTTS-2.5 is an optional installation, not part of these profiles. Local English ASR requires Faster-Whisper; the Japanese recommended profile is not an English model bundle.

## Workflow

Select Japanese, English or Chinese as the new-project source language in **ASR**, and Chinese or English as the **TTS dubbing language**. Batch jobs can produce bilingual mixes, replacement dubs or both, using the selected RTF and mixing settings; original media with subtitles and subtitle-only exports are also available. See [Input and dubbing languages](docs/en/USER_GUIDE.md#input-and-dubbing-languages) and [Batch output choices](docs/en/USER_GUIDE.md#batch-output-choices).

| Input | Route |
|---|---|
| Audio/video only | Create → ASR → correct source text → translate → synthesize → mix |
| Timed source-language subtitles | Import → translate → synthesize → mix |
| Chinese subtitles | Import as Chinese → synthesize → mix; no ASR or body translation |
| Untimed script | Import with estimated timing or ASR-assisted script matching; review manually |
| Multiple tracks/works | Scan → check tracks, scripts and languages → queue → run |
| Subtitles only | AutoFlow → subtitle files only → bilingual, source or translation; no finished audio/video |

[User guide](docs/en/USER_GUIDE.md) · [Existing-subtitle workflow](docs/en/SUBTITLE_WORKFLOW.md) · [Configuration](docs/en/CONFIGURATION.md)

### Save scope

**Defaults for new projects / Current project / Both** are distinct. Saving defaults does not update an open project. API keys have separate save buttons. Tab changes preserve drafts; browser refresh discards unsaved drafts. Reopen existing projects after refresh.

### Mixing

Mixing optionally uses **RTF (original spatial cue transfer)** to transfer stereo cues from the original recording to the target-language dub. Bilingual and replacement mixes reuse a valid RTF dubbing stem rather than computing it twice. See [Audio processing](docs/en/EXPERIMENTAL_AUDIO.md) for operation and parameters.

## Backends

| Stage | Supported interfaces |
|---|---|
| ASR | Parakeet (Japanese), Kotoba-Whisper (Japanese), Faster-Whisper (Japanese/English/Chinese with a suitable multilingual model), generic ASR API |
| Alignment | Backend timestamps, Qwen3 ForcedAligner |
| Local TTS | IndexTTS2, optional IndexTTS-2.5 |
| Online TTS | Edge, MiMo, MiniMax, IndexTTS2 API, GPT-SoVITS, CosyVoice, Fish, generic TTS API |
| Translation | DeepSeek, Bailian, Doubao, SenseNova, OpenAI, Claude, Gemini, OpenAI-compatible, DeepL, Google, Microsoft |

Edge needs internet but no key and does not clone voices. Other services need your account or server. ASMR Dubber does not deploy third-party servers. [Backend reference](docs/en/BACKENDS.md).

## Data and recovery

Default data lives in `.asmr-dubber`: projects, models, isolated runtimes, configuration, download caches and AutoFlow state. Back up the entire project directory, not just `project.json`. External project directories need separate backups. Retrying reuses valid caches; deleting the data directory is not routine troubleshooting.

Use **Settings → Storage & cleanup** to reclaim space: scan, select projects, then confirm. Only safe caches are selected by default. Analysis audio, RTF intermediates and separated stems require explicit selection and may need rebuilding later. Projects, source media, synthesized clips, final outputs and models are preserved. See [Project cache cleanup](docs/en/USER_GUIDE.md#project-cache-cleanup).

API keys are plaintext in `.asmr-dubber/config/secrets.json`. Translation sends text; ASR APIs send audio; external TTS/separation may upload reference audio. Check service terms and media rights before use. Do not publish private logs or projects without review.

## Documentation

[Complete documentation index](docs/en/INDEX.md)

| Category | Pages |
|---|---|
| Getting started | [Installation](docs/en/INSTALLATION.md) · [User guide and batch processing](docs/en/USER_GUIDE.md) |
| Subtitles and audio | [Existing-subtitle workflow](docs/en/SUBTITLE_WORKFLOW.md) · [Audio processing: RTF, separation and sentence mixing](docs/en/EXPERIMENTAL_AUDIO.md) |
| Configuration and troubleshooting | [Configuration](docs/en/CONFIGURATION.md) · [Backends](docs/en/BACKENDS.md) · [CLI](docs/en/CLI.md) · [Troubleshooting](docs/en/TROUBLESHOOTING.md) · [Support](SUPPORT.en.md) |
| Storage and recovery | [Project cache cleanup](docs/en/USER_GUIDE.md#project-cache-cleanup) · [Separation and RTF caches](docs/en/EXPERIMENTAL_AUDIO.md#batch-outputs-and-caches) |
| Development and maintenance | [Contributing](CONTRIBUTING.en.md) · [Architecture](docs/en/ARCHITECTURE.md) · [Prompts](docs/en/PROMPTS.md) · [Artifact maintenance](docs/en/MODELSCOPE_UPLOADS.md) · [Release notes](docs/en/RELEASE.md) |
| Security and licensing | [Security](SECURITY.en.md) · [Third-party notices](docs/en/THIRD_PARTY_NOTICES.md) · [Code of conduct](CODE_OF_CONDUCT.en.md) |

## License

Application code is MIT licensed. Model weights, voices, media and third-party runtimes retain their own terms. Obtain permission for voice cloning and redistribution; passing tests is not a quality or rights guarantee.
