English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v3.0.2/docs/RELEASE.md)

# ASMR Dubber 3.0.2

The full list of changes is in the [changelog](https://github.com/EveningStudy/asmr-dubber/blob/v3.0.2/CHANGELOG.en.md).

## Highlights

- **Prevent breathy lullabies or singing from hijacking automatic voice reference selection**:
  - `pick_reference_sentence` now scores conversational speech rate (penalizing elongated singing punctuation and slow humming), voicing-gated `500–2500 Hz` formant richness over `100–500 Hz` fundamental energy, a `0.65–0.93` modal voicing plateau, and separated vocal purity when cached `vocals.wav` exists, preventing a single breathy lullaby from making an entire project sound hoarse and airy. Multi-speaker anchor selection (`speaker_reference_sentence`) also uses speech-density scoring.
  - Continues using 16 kHz mono reference extraction so existing project dub caches stay valid without reintroducing full-band crackle.
- **Gate full-dub whisper conversion to genuinely whispered or breathy originals**:
  - `replacement_whisper_match` now triggers only when the original line itself is whispered or breathy (`voiced_ratio < 0.45`), preventing hoarse random-phase noise from being added to normal conversational lines.
- Includes all `3.0.1` improvements (IndexTTS2 8 GB VRAM offload, Windows advanced pack import & 8 GB transcription UX, Home project folder shortcut, and Multi-Speaker Phase 1).

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v3.0.2.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

From 3.0.0 / 3.0.1: close the app and copy the new files over your existing folder; your `.asmr-dubber` directory (projects, settings, models, and IndexTTS2 runtimes) stays intact and does not need re-downloading.

From 2.x: close the program, back it up, and copy the new files over the existing program folder. Keep `.asmr-dubber`, which contains projects, models and settings. The first launch checks and updates the core clients; installed speech models are not downloaded again.

Existing projects open directly and can be translated and exported further; dubs already generated stay valid. The first save upgrades a project to the 3.0 format and keeps the original file in the project's `backups/` folder. An upgraded project no longer opens in 2.x; use that backup to go back.

Saved settings are not changed automatically. If your saved Japanese recognizer is an older model (such as Faster-Whisper or Parakeet), startup shows a one-time prompt to switch to anime-whisper (downloading it if needed) or keep your current setting.

From 1.x: close the program, delete the old `src` folder and `ASMR-Dubber-Setup.exe`, then copy the new files over the old folder. Keep and back up `.asmr-dubber`.

## Known limitations

- The fully dubbed version relies on vocal separation: it can leave traces of the original voice or slightly affect sound effects, and very quiet whispers that recognition misses keep their original voice.
- Local dubbing takes about twice the dubbed duration (measured on an RTX 5070 Ti laptop GPU); with FP16 enabled, IndexTTS2 automatically offloads its semantic encoder to fit 8 GB GPUs, while IndexTTS-2.5 still recommends 10 GB of video memory or more.
- Edge TTS uses Microsoft's online service; temporary failures and rate limits can occur, so retry failed sentences or choose another voice.
