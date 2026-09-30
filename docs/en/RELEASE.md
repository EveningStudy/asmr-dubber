[中文](../RELEASE.md) | English

# ASMR Dubber 1.6.2

## Changes

- Fixed batch vocal-separation worker failures caused by temporarily locked request files. Reading and deletion use bounded retries, duplicate chunks are guarded, and returned audio files are validated.
- Completed separation chunks are preserved. Restart the application after updating, then retry the failed task.

## Features introduced in 1.6.1

- Vocal separation loads its model once per processing pass, retaining chunk-based cancellation and recovery.
- Bilingual and replacement mixes reuse the Chinese RTF stem. Audio, timing and relevant parameter changes invalidate the cache. Each variant writes directly to its own directory, reducing intermediate copies.
- Added **Settings → Storage & cleanup**: scan, select projects, then confirm. Safe caches, rebuildable caches and separated stems are distinct categories. Projects, source media, synthesized clips and final outputs are preserved; active or changed files are skipped.
- Japanese, English and Chinese ASR inputs; Chinese or English dubbing targets, subject to the selected backend's capabilities.
- Batch processing supports bilingual mixes, replacement dubs or both, alongside originals and subtitle-only outputs, using the selected RTF, separation and mixing settings.
- Fixed propagation of batch IndexTTS version and external voice-reference settings. Reference timing and source text can be edited while waiting for selection.
- Updated bilingual documentation, README demo explanations and cache-cleanup instructions.

## Download and upgrade

Windows: extract `ASMR-Dubber-windows-portable-v1.6.2.zip` completely, run `ASMR-Dubber-Setup.exe`, then `ASMR-Dubber.exe`. Enable long paths and use a short writable directory.

Linux x86_64: run `bash scripts/linux/setup.sh 推荐` from the source root, then `bash scripts/linux/run-ui.sh`.

Stop tasks and back up projects, settings and finished outputs before upgrading. Preserve `.asmr-dubber` and external project directories. Restart the application after updating; refreshing the browser does not load new code. Existing caches are not deleted automatically; cleanup requires confirmation.

[User guide](USER_GUIDE.md) · [Installation](INSTALLATION.md)

## Verification scope

Local checks covered real-model process reuse across separation chunks, cache hits, short-sample dual-variant mixing and subtitled video, and the browser scan/confirm/cleanup workflow. These checks do not establish long-job speedup ratios or guarantee every cloud service or hardware configuration.
