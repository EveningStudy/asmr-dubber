English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v3.0.1/docs/RELEASE.md)

# ASMR Dubber 3.0.1

The full list of changes is in the [changelog](https://github.com/EveningStudy/asmr-dubber/blob/v3.0.1/CHANGELOG.en.md).

## Highlights

- **8 GB VRAM dubbing & cleaner failure reporting**:
  - IndexTTS2 now lazy-loads `QwenEmotion` (saving ~1.1 GB VRAM unless text emotion is active), casts GPT weights to FP16 before moving to CUDA to avoid initialization spikes, and automatically offloads `w2v-bert-2.0` (~2.2 GB) to CPU on GPUs with `<= 9.5 GB` total or `<= 7 GB` free memory (reducing initial CUDA allocation from ~5.1 GB to ~2.9 GB);
  - Sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` on Windows; batch initialization failures no longer blame every sentence in a wall of text, and error toasts scroll cleanly while surfacing Windows virtual-memory guidance.
- **Windows dependency pack & 8 GB recognition UX**:
  - Windows advanced dependency pack import skips duplicate 2.9 GB SHA-256 verification, merges files via same-volume moves, flushes progress live, and resets the progress bar on non-percentage stages instead of appearing frozen at `100%`;
  - Added heartbeats, `local_files_only` offline fast-fail errors, MOSS chunk progress updates, and low-VRAM recognition batch caps so 8 GB GPUs do not silently stall at `0%`.
- **Manage projects right from Home**: Added an **Open projects folder** button next to Recent projects on the home page, with a note that deleting a project's folder removes the project without touching source media.
- **Multi-speaker diarization & per-speaker voices**:
  - Added binaural stereo ILD panning cues to speaker diarization and surfaced **Speaker count** in Step 1 basic settings;
  - Gave the sentence table its own **Speaker** column with inline color-coded selectors, custom labels (`+ New…`), speaker filtering, and undoable batch assignment;
  - Step 3 now lists each detected speaker (`S01`, `S02`, …) with auditioning and independent per-speaker voice overrides (auto clip, pinned line, or external reference audio), clearing only that speaker's cached dubs on change.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v3.0.1.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

From 3.0.0: close the app and copy the new files over your existing folder; your `.asmr-dubber` directory (projects, settings, models, and IndexTTS2 runtimes) stays intact and does not need re-downloading.

From 2.x: close the program, back it up, and copy the new files over the existing program folder. Keep `.asmr-dubber`, which contains projects, models and settings. The first launch checks and updates the core clients; installed speech models are not downloaded again.

Existing projects open directly and can be translated and exported further; dubs already generated stay valid. The first save upgrades a project to the 3.0 format and keeps the original file in the project's `backups/` folder. An upgraded project no longer opens in 2.x; use that backup to go back.

Saved settings are not changed automatically. If your saved Japanese recognizer is an older model (such as Faster-Whisper or Parakeet), startup shows a one-time prompt to switch to anime-whisper (downloading it if needed) or keep your current setting.

From 1.x: close the program, delete the old `src` folder and `ASMR-Dubber-Setup.exe`, then copy the new files over the old folder. Keep and back up `.asmr-dubber`.

## Known limitations

- The fully dubbed version relies on vocal separation: it can leave traces of the original voice or slightly affect sound effects, and very quiet whispers that recognition misses keep their original voice.
- Local dubbing takes about twice the dubbed duration (measured on an RTX 5070 Ti laptop GPU); with FP16 enabled, IndexTTS2 automatically offloads its semantic encoder to fit 8 GB GPUs, while IndexTTS-2.5 still recommends 10 GB of video memory or more.
- Edge TTS uses Microsoft's online service; temporary failures and rate limits can occur, so retry failed sentences or choose another voice.
