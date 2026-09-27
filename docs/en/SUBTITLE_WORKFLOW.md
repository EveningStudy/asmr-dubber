[中文](../SUBTITLE_WORKFLOW.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Existing subtitles and subtitle-only output

For translation-only subtitle files, selected timed Chinese subtitles can provide text and timing directly, without ASR or body translation. Source/bilingual output still needs source-language text. Select direct subtitle timing on the track.

Trusted subtitles do not need to cover every second of audio. Music, pauses and breaths without captions are not missing dialogue.

## Use supplied timing

1. Scan the work in Workspace → Batch processing.
2. Assign the correct SRT, VTT, ASS/SSA or LRC to every selected track.
3. Choose full use of subtitle text/timing, without ASR.
4. Verify each subtitle's language. Chinese subtitles remain Chinese even when the audio is Japanese.
5. Confirm complete per-track coverage before adding to the queue.

| Input | Processing |
|---|---|
| Chinese timed subtitles on every track | Skip ASR and body translation; synthesize directly |
| Japanese or English timed subtitles | Skip ASR; translate then synthesize |
| Chinese plus one source language | Keep Chinese; translate only source-language entries |
| Mixed Japanese and English in one merged project | Rejected; split tracks or use consistent language metadata |
| Missing subtitles on selected tracks | Supply them or deselect those tracks |
| Untimed TXT / retiming mode | ASR may still be needed for timing |

Merging adds each track's start offset to its subtitles. Parsing errors do not silently trigger ASR. Repeated/overlapping source captions are preserved rather than deduplicated.

LRC usually defines starts only: ends are inferred from the next entry or a final duration estimate. Imported timestamps are not forced TTS durations. Dubbing offsets, speed-up and subtitle readability settings can change output timing.

Work and track-title translation are separate switches. Disable them too if the task must never call a translation service.

## Subtitle files only

Select subtitle-only output, then **Bilingual**, **Source only** or **Translation only**. It produces SRT/LRC without synthesizing finished audio or encoding video. Source-only does not translate the body. Bilingual and translation-only use translation settings when no matching Chinese text is supplied.

Choose source filename, content-based name or a custom name in Settings → AutoFlow. Per-track subtitles retain the track filename. Internal working audio may still be created for ASR; no audio/video deliverable is produced.

Maximum characters per line is a wrapping limit, not a target cue length and not a sentence-merging control. Set 8–500 in Mixing & subtitles. Save to Current project/Both for existing projects and regenerate; old files are not rewritten by saving settings.

## Untimed scripts

The LLM selects literal script quotes; Python computes character ranges. Explicitly skipped stage directions are recorded. The same characters cannot be allocated twice, and ASR timing is not interpolated from character counts.

Invalid batches retry with diagnostics, then sentence by sentence. Unresolved sentences retain their source text/translation and are marked for manual review. Authentication, network and output-limit failures remain errors, not successful reviews. Inspect project diagnostics and the script report before dubbing.

## Redoing an incorrect task

Back up finished outputs. Rescan rather than resuming a stale queue snapshot, check language/timing on every track, then explicitly allow replacing the tool's prior results. Upgrading does not rewrite source media, source subtitles or existing deliverables.
