English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/RELEASE.md)

# ASMR Dubber 2.1.0

## New and improved

- Added MOSS-Transcribe-Diarize 0.9B for Japanese, English and Chinese transcription and speaker labels within each chunk, with full inference controls. It uses an isolated runtime and tries the maintainer's ModelScope mirror before falling back to the official source. Parakeet remains the default recognizer.
- Restored cover previews when adding or editing batch works. Output defaults can be saved; new installs default to fully dubbed plus bilingual outputs. Fully dubbed output enables vocal separation for that work and requires its model to be installed.
- Added a shared external voice reference for the queue, including the current and remaining waiting works during a run. A durable audio copy and the queue selection survive restarts for unattended processing.
- Reduced the default automatic speed limit from 1.8× to 1.4×. Fully dubbed volume controls are directly visible, defaulting to source loudness with 0 dB relative level and 0 dB extra gain, subject to peak protection and loudness bounds. Existing projects and saved values retain their settings.
- Fixed the home checklist after saving a DeepSeek key. Notifications can be dismissed and expire automatically, repeated polling errors are suppressed, and language buttons have equal widths.
- Added application, browser and Windows launcher icons; refreshed both READMEs and parameter documentation.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v2.1.0.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

From 2.0.x: close the program, back it up, and copy the new files over the existing program folder. Keep `.asmr-dubber`, which contains projects, models and settings.

From 1.x: close the program, delete the old `src` folder and `ASMR-Dubber-Setup.exe`, then copy the new files over the old folder. Keep and back up `.asmr-dubber`.

[Batch processing](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/en/BATCH.md) · [MOSS guide](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/en/MOSS_TRANSCRIBE_DIARIZE.md) · [Full parameters](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/en/PARAMETERS.md)
