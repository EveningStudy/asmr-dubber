English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v3.0.0/docs/RELEASE.md)

# ASMR Dubber 3.0.0

The full list of changes is in the [changelog](https://github.com/EveningStudy/asmr-dubber/blob/v3.0.0/CHANGELOG.en.md).

## Highlights

- **A much better fully dubbed version (still experimental)**: dubbed sentences replace the original voice and everything else stays as recorded. Each dub is generated to fit the length of its line; on the evaluation clips the median start offset fell from about 0.5 s to about 0.03 s. Dubs match the original loudness, never talk over kept breaths or kisses, and the last line tries to finish before the recording ends. Voice-cloning references are 16 kHz again with the high band restored afterwards, which removes most crackle. When IndexTTS-2.5 is installed, the "Fully dubbed" preset uses it.
- **More accurate recognition**: the recommended recognizer is now anime-whisper, fine-tuned on anime and game voices and better on whispers and quiet lines. Qwen3-ASR and its Japanese anime/visual-novel fine-tune are also available. All have ModelScope China mirrors.
- **Translation**: each line gets a length budget from its timing, overlong translations are compressed automatically or on request, and lines can be revised by instruction. A failed line no longer stops the track; its subtitle falls back to the original text.
- **Sentence table**: thousands of lines scroll smoothly; play the original and the dub of each line, drag the waveform to adjust timing, use arrow keys, Space and Enter, and undo or redo edits.
- **Several speakers (experimental)**: speakers can be told apart automatically, each with their own cloned voice.
- **Batch**: recognition and dubbing models load once per queue; after a crash or a closed window the queue continues where it stopped without paying again for lines already translated; a work with a few failed lines still gets its output.
- **Also**: a first-run wizard, quality and output presets, a pronunciation glossary and spoken numbers, several dub takes per line, and English translations for every progress message.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v3.0.0.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

From 2.x: close the program, back it up, and copy the new files over the existing program folder. Keep `.asmr-dubber`, which contains projects, models and settings. The first launch checks and updates the core clients; installed speech models are not downloaded again.

Existing projects open directly and can be translated and exported further; dubs already generated stay valid. The first save upgrades a project to the 3.0 format and keeps the original file in the project's `backups/` folder. An upgraded project no longer opens in 2.x; use that backup to go back.

Saved settings are not changed, including the recognition and dubbing models you picked. To use the newly recommended anime-whisper, download it on the Models page and select it in the settings.

From 1.x: close the program, delete the old `src` folder and `ASMR-Dubber-Setup.exe`, then copy the new files over the old folder. Keep and back up `.asmr-dubber`.

## Known limitations

- The fully dubbed version is still experimental: separation can leave traces of the original voice or damage sound effects, and very quiet whispers that recognition misses keep their original voice.
- Local dubbing takes about twice the dubbed duration (measured on an RTX 5070 Ti laptop GPU); IndexTTS-2.5 needs 10 GB of video memory or more.
- Edge TTS uses Microsoft's online service; temporary failures and rate limits can occur, so retry failed sentences or choose another voice.
