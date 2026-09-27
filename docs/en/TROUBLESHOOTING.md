[中文](../TROUBLESHOOTING.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Troubleshooting

Preserve the project, logs and completed results before repair. Record version/commit, installation profile, OS, selected backend/model and the exact failing stage. Static checks and screenshots alone do not establish inference quality.

## Missing DLL / backend exit

Open Settings → Devices & models → Dependency diagnostics & repair. Check first; repair only the identified runtime/backend. Microsoft runtime repair downloads from the official source, verifies signatures and may request elevation. Backend repair may replace its environment or download missing assets. It does not delete projects/models/results or rerun failed tasks.

Do not copy arbitrary DLLs into system/application directories. A bare exit code can indicate missing dependencies, incompatible CPU/GPU code, driver problems or a native crash; the diagnostic log is needed to distinguish them.

## Setup and downloads

| Symptom | Check |
|---|---|
| Setup cannot start | Complete extraction, `scripts`, `mirrors.json`, PowerShell and `curl.exe`; use a short writable path |
| File exists but cannot be opened | Windows path length, permissions, antivirus locking, synced folders |
| Interrupted download | Rerun Setup; retain verified files and partial download state |
| ModelScope failure | Exact URL/status, proxy, disk space, authentication only for private repos |
| Existing download ignored | Expected filename, size, SHA-256 and internal package manifest |
| Installed but warning remains | Distinguish optional selected backend from core runtime readiness |

Logs: `.asmr-dubber/logs/setup-*.log`. Do not disable checksums or rename unrelated artifacts to the expected filename. External download fallback requires explicit opt-in; a failure is not permission to change sources silently.

Moving the application can invalidate virtual-environment paths. Rerun Setup after moving. Keep runtime rollback backups until the new environment works. A partially interrupted install may retain archives, extraction directories and caches; inspect exact locations rather than deleting the whole data directory.

## UI and projects

If no browser opens, inspect the terminal and use its local URL. A refreshed page discards unsaved drafts, not saved project data. Reopen a project after refresh. A process restart is required for changed Python code, not ordinary saved settings.

When settings appear unchanged, check Save scope and the success message. Defaults-only does not update an existing project. A stale project revision means another session saved first; reopen instead of overwriting its work.

Large sentence tables are paginated. Saving commits all pages. Slow model initialization is distinct from a frozen editor; include sentence count, media duration, selected models and exact stage in a report.

## ASR and review

CUDA unavailable: inspect driver/runtime compatibility and device selection. VRAM capacity alone is insufficient. Use CPU-compatible precision or an API where appropriate. Out-of-memory: reduce batch/chunk size and avoid simultaneous GPU workloads.

Missing Kotoba/Faster-Whisper files indicate an incomplete snapshot, not necessarily a missing Python package. VAD/Qwen alignment controls appear only when their model/runtime is complete. Parakeet uses a separate executable; keep its private DLL environment intact. Punctuation restoration is off by default.

An empty review list usually means no eligible local models. Review completion means proposals, not automatic transcript replacement. Different model sentence splits are compared over common audio regions. Retry review alone instead of rerunning the primary ASR unnecessarily.

## Untimed-script matching fails

The model selects exact script quotes; the application computes offsets. Repeated/rewritten quotes and backward mappings are rejected. Explicitly skipped stage directions are recorded. After batch retries, single-sentence retries isolate failures; unresolved sentences keep their original text and receive manual-review notes in project diagnostics.

Inspect `imports/script-reconciliation*-error.json` for source lines, recognition times and model response. It contains private text. API authentication/network/token-limit failures remain task errors. Do not duplicate an entire script line into several timestamp intervals.

## Subtitles stay at the old width

The supported line limit is 8–500 characters. Save to Current project/Both for an existing project, then regenerate SRT/LRC. Defaults-only affects future projects. Saving does not rewrite exported files. The width is a maximum; short cues remain short and separate cues are not merged. A player's automatic screen wrapping is separate from line breaks in the SRT file.

## Batch repeats dialogue / unexpectedly runs ASR

Check every selected track's subtitle, language and timing mode. Timed Chinese subtitles bypass ASR/body translation when coverage is complete by track; silence need not be captioned. LRC has inferred ends. Untimed scripts still need timing. Do not resume a queue snapshot created with incorrect associations: back up results, rescan and explicitly redo.

## Translation and TTS

Verify provider, endpoint, model availability, region, key and quota. Keep JSON extra parameters valid and avoid overriding protected fields. Output-token exhaustion is not an empty successful translation.

Edge needs internet. External TTS servers must actually be running and may need access to reference paths. IndexTTS2 and 2.5 use different runtimes/checkpoints; repairing one does not install the other. For unstable voice, compare clean unified references before changing many inference parameters.

## Mixing and video

Clipping: lower gains and inspect peak limits. Delayed Chinese: check global offset and sequential scheduling. Overlap: fit-window speed caps may leave residual conflicts. Channel layout: inspect routing; RTF requires stereo original audio.

Short/silent RTF references fall back to level placement and log the change. Separation may remove breaths or leave speech residue; it is experimental. Original sentence gain/mute requires separation. Chinese playback mute preserves text/TTS and needs only remixing.

Video failures: retain FFmpeg diagnostics and check source codecs, output path and disk space. Original-audio hard-subtitle encoding fails explicitly rather than silently switching to soft subtitles.

## Reporting

Use [Support](../../SUPPORT.en.md). Provide reproducible steps and sanitized text logs, not screenshots alone. Never attach API keys, full config directories, private audio or unreviewed script diagnostics publicly.
