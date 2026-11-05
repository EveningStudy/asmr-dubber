English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/RELEASE.md)

# ASMR Dubber 2.2.0

## Fixes and improvements

- Corrected batch subtitle routing: subtitle-files-only exports SRT/LRC; original audio with subtitles retains media. Translation-only text, original filenames and title translation follow their settings.
- Existing subtitle queues adopt displayed intent with new execution identities. Old exports are retained until an explicit rebuild or output-folder change.
- Blank translations of meaningful speech are retried individually. Persistent failures remain enabled with errors, preserving successful translations. Older skipped speech can be restored; row appearance and remaining counts update immediately.
- Each queued work has its own waiting, running, reference-waiting, completed or failed state. Successful works remain visible in failed batches. Timestamped history can be hidden; empty queues no longer create failed tasks.
- New batch project directories include the work, track and task ID. Storage and cleanup previews show ownership, including older checkpoint labels without moving projects.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v2.2.0.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

From 2.x: close the program, back it up, and copy the new files over the existing program folder. Keep `.asmr-dubber`, which contains projects, models and settings.

From 1.x: close the program, delete the old `src` folder and `ASMR-Dubber-Setup.exe`, then copy the new files over the old folder. Keep and back up `.asmr-dubber`.

[Batch processing](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/en/BATCH.md) · [MOSS guide](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/en/MOSS_TRANSCRIBE_DIARIZE.md) · [Full parameters](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/en/PARAMETERS.md)

If you worked around the 2.1 mapping by choosing the opposite subtitle option, review queued selections after upgrading.
