[中文](../CONFIGURATION.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Configuration

## Scope and persistence

Defaults live in `.asmr-dubber/config/settings.json`. A new project copies them into `project.json`; existing projects do not continuously inherit them.

| Save scope | Changes |
|---|---|
| Defaults for new projects | Future projects only |
| Current project | Open project only |
| Both | Both copies |

Saving applies the whole settings draft, not only the visible tab. API keys have separate buttons. Tabs preserve drafts; refresh discards drafts and reloads defaults. Reopen an existing project after refresh. Explicitly loading project TTS settings replaces that portion of the draft.

ASR/VAD/alignment/review changes mark recognition stale; they do not instantly erase the sentence table. Revision checks prevent an older browser session from overwriting newer saved work.

Interface language is browser-local, defaults to Chinese and persists independently of project settings. It never changes prompts, source language, model IDs or Chinese dubbing targets.

## General and devices

An empty project-root setting uses `.asmr-dubber/projects`. External roots must be backed up separately. Python/Hugging Face endpoints are optional; the ordinary installer still follows the explicit mirror policy.

Device/model checks are static and do not load models. Available means dependencies/files pass checks, not that inference or perceptual quality has been verified. Install, repair and inference share runtime locks. External service entries require your server/account.

## ASR

| Control | Default / interpretation |
|---|---|
| Japanese backend | Parakeet CTC 1.1B JA GAL |
| English local backend | Faster-Whisper |
| Device / precision | CUDA / float16; use compatible CPU settings when needed |
| Batch / beam | 1 / 5; increasing uses more resources |
| Pause split / maximum sentence | 0.55 s / 15 s |
| Prompt | Optional names/terminology, not expected full transcript |
| Parakeet chunk / idle timeout | 120 s / 600 s; chunk range 15–600 s |
| Kotoba chunk | 30 s; range 5–120 s |

Previous-text conditioning applies to compatible Whisper paths; disable it when errors repeat across chunks. Parakeet punctuation restoration is off by default, avoiding an implicit punctuation-model download.

VAD defaults off. Backend VAD is available for compatible Parakeet/Faster-Whisper paths. Japanese ASMR VAD requires its ONNX model/runtime; threshold 0.5, minimum speech 250 ms, silence 100 ms, boundary padding 200 ms. Lower thresholds may retain whispers and noise. Qwen3 ForcedAligner adjusts timestamps without correcting text.

Multi-model review is experimental and may be worse than one model. Default is suggestions only; clip target 30 s, context 0.5 s, bounded extension to 90 s. Review does not call an LLM. Historical text-priority/prompt fields are compatibility data, not a new voting mechanism. [Review guide](AUDIO_REVIEW_TUTORIAL.md).

## Translation

LLM providers: DeepSeek, Bailian, Doubao, SenseNova, OpenAI, Claude, Gemini and OpenAI-compatible. Machine translation: DeepL, Google Cloud Translation and Microsoft Translator; these do not perform untimed-script matching.

| Control | Default |
|---|---|
| Provider/model | DeepSeek / `deepseek-v4-flash` |
| Temperature / Top P | 0.1 / 1.0 |
| Output token limit | 16384 |
| Neighbor context | Enabled, 24 sentences |
| Translation memory | 50 confirmed pairs |
| Extra request JSON | `{}` |

Models may be entered manually when available to your account. Endpoint, region and key must match. DeepL Free generally uses `https://api-free.deepl.com`; Azure requires the correct region; Google uses Basic v2 keys. Custom OpenAI-compatible endpoints may omit a key.

Prompts are stored separately by source language. Extra JSON cannot override protected payload fields. Recognition IDs/order remain stable; nonverbal entries may receive empty Chinese text. Requests use bounded context, not unlimited project history.

## TTS

IndexTTS2 is the default local choice; unavailable local files cause new projects to use Edge TTS. IndexTTS-2.5 is installed separately and does not replace IndexTTS2. Local timeout defaults to 600 s; HTTP concurrency defaults to 2, range 1–8.

| Backend | Reference / configuration |
|---|---|
| IndexTTS2 | Separate venv/checkpoints; FP16 on; emotion weight 0.5; speaker and emotion references independently selectable |
| IndexTTS-2.5 | Separate runtime; Chinese synthesis default; duration factor 1.0 (0.5–2.0); BF16 on where supported; text normalization on |
| Edge | `zh-CN-XiaoxiaoNeural`; no cloning/key; online |
| MiMo | Model-dependent voice ID, reference audio or voice-design instruction |
| MiniMax | Account voice ID; speed, volume, pitch, optional fixed emotion |
| GPT-SoVITS | Official api_v2 server; reference transcript and server-accessible path required |
| CosyVoice | Zero-shot needs reference transcript; cross-lingual uses reference audio |
| Fish | Reference audio and transcript through compatible `/v1/tts` |
| Generic TTS API | Text/model/voice via `/audio/speech`; no reference audio |
| IndexTTS2 API | Multipart `/v1/tts`; server manages model/GPU |

IndexTTS-2.5 generation defaults: 120 text tokens per segment, 200 ms internal gap, sampling enabled, temperature 0.8, top-p 0.8, top-k 30, beams 3, repetition penalty 10, length penalty 0, maximum acoustic tokens 1500. Random conditions are off. Eight emotion values represent happy, angry, sad, afraid, disgusted, melancholic, surprised and calm; they are normalized before applying weight.

CUDA kernels, DeepSpeed, GPT acceleration and `torch.compile` are optional, not compatibility fixes. Leave them disabled unless the runtime supports them. External APIs may upload audio; URLs are bounded and do not inherit credentials.

## Mixing

| Control | Behavior |
|---|---|
| Chinese offset | Default +500 ms relative to source timing |
| Fit-window | Speeds up conflicts only; default maximum 1.8×, allowed 1–4×; remaining overlap is possible |
| Sequential | No speed-up; sentences wait for previous Chinese speech and may extend duration |
| Source-relative loudness | Default Chinese level 8 dB below corresponding source |
| Uniform loudness | Default −30 RMS dBFS |
| Raw TTS loudness | No per-sentence normalization; gain and peak protection still apply |

Source-relative targets are bounded; automatic gain should not amplify silence without limit. Offset, gain, scheduling and RTF are mix-time operations and do not require TTS regeneration.

Separation/Chinese replacement are experimental, not recommended. RTF is independent and requires stereo source. Original-vocal per-sentence controls require separation; Chinese mute/gain does not. [Full routing and recovery](EXPERIMENTAL_AUDIO.md).

## Subtitles

Maximum characters per line: **8–500**, default 22. This is a wrapping ceiling, not a cue-merging rule. Minimum duration defaults to 1 s; maximum reading speed to 18 characters/s. These readability controls may extend display time. Choose original or dubbed timing explicitly.

Save to Current project/Both and regenerate to update an existing project's outputs. A defaults-only save cannot change existing project files. Existing SRT/LRC files are not rewritten at save time.

## AutoFlow

The batch page stores per-work output choices in queue entries. Defaults only initialize a new draft. Select subtitle files only for no finished audio/video, then bilingual/source/translation. Naming can use source audio names, content-based names or a custom stem. Per-track captions retain track filenames.

Global original-audio hard subtitles default off. Enabling them adds `原声字幕版.mp4`; harmonized timing uses the same source delay. Encoding failures are errors, not soft-subtitle fallback. Pure audio output is unaffected.

Timestamp extra text can appear before/after the timestamp list; default after. The work title stays first. These rules are read at queue start; running tasks retain a snapshot. Explicitly redo finished work to change outputs.

Track/work-title translation is independent of subtitle body translation. Per-track-only output does not create a merged deliverable; per-track-plus-merged reuses valid per-track results.

## Keys and environment

Keys are plaintext in `.asmr-dubber/config/secrets.json`, not project manifests or export files. Empty password fields do not delete saved keys; use Remove. Protect directory permissions and backups.

| Variable | Purpose |
|---|---|
| `ASMR_DUBBER_HOME` | Portable root override |
| `ASMR_DUBBER_DATA_DIR` | Data/model/cache root |
| `ASMR_DUBBER_CONFIG_DIR` | Settings/key root |
| `ASMR_DUBBER_PROJECTS` | Default project root |
| `ASMR_DUBBER_FFMPEG` | Custom FFmpeg executable |
| `ASMR_DUBBER_LOCAL_CACHE_ROOTS` | Read-only artifact reuse locations |
| `ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1` | Explicit external-source opt-in |
| `MODELSCOPE_API_TOKEN` | Private ModelScope access |
| `ASMR_DUBBER_UI_USERNAME` / `ASMR_DUBBER_UI_PASSWORD` | Non-loopback UI authentication |
| `ASMR_DUBBER_MAX_UPLOAD_SIZE` | Upload ceiling, default `20gb` |

Non-loopback binding requires authentication; absent a configured password, a random one is printed in the terminal. Do not expose the development UI directly to the internet.
