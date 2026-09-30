[中文](../USER_GUIDE.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# User guide

## Project cache cleanup

Open **Settings → Storage & cleanup**, choose a project directory, and click **Scan caches (no deletion)**. Review sizes and the file inventory, select projects, then confirm cleanup. Changing the directory or categories requires a new scan.

- **Safe caches**: input copies and chunks whose completed separation outputs pass integrity checks. Selected by default; incomplete or damaged separation jobs retain their restart checkpoints.
- **Rebuildable caches**: analysis audio, RTF references, dubbing stems and mix backgrounds. Future mixing may need to recompute them.
- **Separated stems**: vocals and backgrounds. Future processing must rerun the separation model, which can take considerably longer.

Projects, source media, corrected text, synthesized clips, final media, subtitles, models and runtimes are preserved. Locks protect active operations; files changed since scanning are skipped. Deletion is irreversible, but caches can be rebuilt. Cleanup is manual; it never deletes entire projects.

Batch separation loads the model once per processing pass while retaining chunk-based recovery. Bilingual and replacement mixes write directly to separate output directories and reuse a Chinese RTF stem. Changes to synthesized audio, timing, loudness or RTF parameters invalidate that cache.

## Input and dubbing languages

ASR accepts Japanese, English, or Chinese. Use multilingual Faster-Whisper or a compatible ASR API for English/Chinese. **TTS → Dubbing language** selects Chinese or English for both translation and synthesis. Matching source/target languages bypass translation. Interface language is independent.

Applying a different target language to a project backs up its old translations to `translations-language-revision.json`, then clears translation and sentence-audio references. Translate and synthesize again; source audio, recognition timestamps, and old audio files remain. IndexTTS 2/2.5 and GPT-SoVITS support English; custom APIs depend on the deployed model. Edge TTS automatically selects an English voice for English output; US/UK voices are also selectable.

## Batch output choices

Queue entries retain the selected TTS backend, model and reference settings. After changing the IndexTTS version or external voice, save new-project defaults (or both scopes), then edit and save existing queue entries. Current-project-only settings do not change batch defaults; started projects retain their own settings and results.

While waiting for a reference, edit its start/end times and transcript directly in the picker. “Use this clip” saves the edits before continuing synthesis; timing edits also update that sentence in the project timeline. You can also open the project and save the review table while the batch is waiting. Save table edits before confirming a reference. Timeout still resumes processing automatically; increase the wait duration before lengthy edits.

Under **Workspace → Batch processing**, select the audio result, then subtitle content, audio/video format, and track/merged layout.

| Result | Processing |
| --- | --- |
| Bilingual mix + original | Original plus target-language dubbing; honors RTF settings |
| Replacement dub + original | Separated background plus dubbing; separation required, experimental and not recommended |
| Both mixes + original | Shares ASR, translation and TTS; renders and stores each mix separately |
| Original media + subtitles | No TTS, separation or RTF; optional video subtitles |
| Subtitle files only | SRT/LRC without media output; source-only subtitles skip translation |

Every option supports bilingual, source-only, or translation-only subtitles. Translation-only uses the dubbing target language. Dubbing jobs still translate speech even when subtitles show only the source. Subtitle-file naming is configured in **Settings → AutoFlow → Subtitle files**.

Queued jobs retain their target language, RTF, separation, mixing and original-speech retention settings. Edit and save the queue entry to adopt changed defaults. Replacement output may still contain separation leakage or deliberately retained original speech; audition before publishing.

## Start

Run Setup once, then the launcher. Keep the terminal open while using the local web UI. Check Devices & models before selecting local backends. Interface language can be switched with 中文 / English; source-language and output-language settings are independent.

## 1. Create or open

Choose source language and upload audio/video, then Create project. The source is copied into the project. To resume, open a recent project or its `project.json`.

![Workspace](../../assets/screenshots/workbench.png)

Import timed source subtitles to skip ASR. Import timed Chinese subtitles to skip both ASR and body translation. Untimed scripts need estimated timing or ASR-assisted matching. Import replaces the sentence timeline, not the original media; back up important edits first.

## 2. Recognize

Choose a backend/model in ASR settings, save to Current project/Both and run ASR. Japanese supports Parakeet, Kotoba and Faster-Whisper; local English uses Faster-Whisper. A generic ASR API uploads audio to your configured endpoint.

![ASR settings](../../assets/screenshots/settings-asr.png)

VAD may miss quiet whispers. Default is no VAD preprocessing; backend VAD and the optional Japanese ASMR VAD are alternatives. Qwen3 alignment repositions existing text without verifying its accuracy.

Multi-model review is optional and experimental. Review proposals before translation; it may be worse than a single model. [Illustrated instructions](AUDIO_REVIEW_TUTORIAL.md).

![Review proposals](../../assets/tutorials/review-proposals.png)

An untimed script uses a different path: LLM quote matching against ASR intervals. If retries cannot establish a reliable match, the existing sentence survives with a manual-review note. Check diagnostics; do not treat a completed batch as a fully corrected transcript.

## 3. Edit and translate

Edit source text/timestamps in the sentence table, then save. The table pages 50 sentences at a time; paging retains all drafts. Clearing both texts deletes a row. Translation preserves sentence IDs and order; nonverbal content may have empty Chinese text.

Configure translation provider, model, endpoint and key. Keys use a separate Save button. LLM providers support structured script matching; DeepL/Google/Microsoft machine translation do not.

![Translation settings](../../assets/screenshots/settings-translation.png)

Japanese and English translation prompts are stored separately. Keep names/terminology concise. Do not paste credentials or unrelated instructions into prompts.

## 4. Voice reference

Choose a clean representative project sentence or external reference. A unified reference improves voice consistency; per-sentence references follow local delivery but may vary more. Reference requirements depend on the TTS backend. GPT-SoVITS and some other APIs need the exact reference transcript.

## 5. Synthesize and mix

Select TTS backend/model, save and synthesize Chinese speech. Existing valid sentence caches are reused. Editing Chinese text invalidates the affected synthesis; gain, offset, scheduling and RTF changes require only remixing.

![TTS settings](../../assets/screenshots/settings-tts.png)

IndexTTS2 and optional IndexTTS-2.5 use separate runtimes. Speaker and emotion references are independent. Edge TTS requires internet, no key, and does not clone voices. MiMo supports preset, cloned or designed voices depending on model. MiniMax uses an account voice ID; cloning is managed on its platform. External model servers are not started by this application.

### Timing and loudness

Fit-window mode speeds up only conflicting Chinese sentences, up to the configured limit; residual overlap may remain. Sequential mode avoids Chinese-on-Chinese overlap by delaying subsequent sentences and may extend the result. Global offset applies to Chinese speech, not source media.

![Mix settings](../../assets/screenshots/settings-mix.png)

Choose source-relative loudness, a uniform RMS target or raw TTS level. Peak protection remains active. Raising levels aggressively can amplify noise; louder is not automatically better.

### Separation and RTF

Separation and Chinese replacement are experimental, not recommended, and disabled by default. Enable separation before ASR if recognition should use the vocal stem. Replacement requires it; ordinary bilingual mixing does not.

RTF is independently enabled in Mixing & subtitles and uses original stereo cues. Short/silent references degrade to bounded level placement rather than aborting the mix.

The table's rightmost controls mute/adjust each original vocal or Chinese sentence. Original controls require separation. Chinese playback differs from Process Chinese: muting it retains text/cache. Save and remix. [Details](EXPERIMENTAL_AUDIO.md).

## 6. Subtitles

Choose bilingual, source or Chinese subtitles, and source or dubbing timing. SRT/LRC are written; video projects may also produce a subtitled video. For no audio/video deliverables, use AutoFlow's subtitle-files-only mode.

Line width accepts 8–500 characters. This wraps each cue without merging sentences. For an existing project save to Current project/Both, then regenerate. Minimum duration and reading-speed limits may extend display time beyond imported ends.

## Batch processing

Scan a work folder, inspect selected tracks, ordering, subtitle associations and languages, choose output, then add to the queue. Drag tracks/tasks or use their move controls. Scanning and queueing are distinct; save per-work options before execution.

![Batch workflow](../../assets/screenshots/batch-workflow.png)

Layouts: merged, per-track, or per-track plus merged. The combined output can reuse per-track work. Audio/video mode, subtitle content and filename policy are separate choices. Source-only subtitle output does not translate the body. All selected tracks need corresponding timed subtitles to bypass ASR completely.

Original-audio hard subtitles optionally add an encoded original-audio video; this costs time/space. Timestamp footer text can appear before or after the timestamp list, with the work title retained first. These AutoFlow rules are read when the queue starts; running tasks retain a snapshot.

Reprocessing finished work requires explicit replacement confirmation. Back up results before using it. Cancel/pause does not delete projects, models or completed outputs.

### Subtitle files only

In Batch processing, select **Subtitle files only**, then bilingual, source or translation. No finished audio or video is produced. Filename rules can preserve the audio basename. For timed subtitle import and ASR bypass conditions, see [Existing-subtitle workflow](SUBTITLE_WORKFLOW.md).

## Recovery and privacy

Keep the full project directory and any configured external project root. Resume from valid caches rather than deleting `.asmr-dubber`. Two browser sessions cannot overwrite newer revisions silently; reopen stale projects.

API keys are plaintext in the portable config directory. LLMs receive text/context; ASR and external voice/separation APIs may receive audio. Check rights and provider privacy terms. Logs and script diagnostics may contain private text; inspect before sharing.
