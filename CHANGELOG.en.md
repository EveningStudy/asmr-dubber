English | [中文](CHANGELOG.md)

# Changelog

All notable changes to ASMR Dubber are documented in this file.

## [Unreleased]

- Keep decimals and scientific notation together when wrapping Chinese subtitles if they fit on one line; longer numbers still use the existing hard-break fallback.

## [3.0.2] - 2026-11-22

### Automatic Voice Reference & Full-Dub Audio Quality Fixes
- **Prevent breathy lullabies from hijacking automatic voice reference selection** (`voice_reference.py`, `separation.py`, `pipeline.py`, `tts.py`, `services/project_audio.py`): `pick_reference_sentence` expands candidate pre-ranking to 48 lines (sweet spot `5.5–9.5 s`), scores normal speech rate while penalizing elongated singing marks (`・`/`ー`/`…`), adds voicing-gated `500–2500 Hz` formant richness over `100–500 Hz` fundamental energy, caps modal voicing credit on a `0.65–0.93` plateau, scores against cached separated `vocals.wav` purity when available, and aligns multi-speaker modal selection (`speaker_reference_sentence`) with speech-density scoring.
- **Gate `replacement_whisper_match` to genuinely whispered or breathy originals** (`audio_dsp.py`, `stem_cache.py`): `_VoiceMatcher.apply` now ramps whisper conversion smoothly across `original voiced_ratio` `0.45 -> 0.20`, stopping normal conversational lines from receiving hoarse random-phase noise; bumped `stem_cache` implementation version to `6`.

## [3.0.1] - 2026-11-21

### 8 GB VRAM Dubbing & Error Handling
- **IndexTTS2 low-VRAM initialization & CPU semantic offload** (`indextts2_worker.py`): lazy-loads `QwenEmotion` (~1.11 GiB) only when text emotion is requested, casts GPT weights to FP16 on CPU before CUDA placement, and offloads `w2v-bert-2.0` (~2.16 GiB) to CPU on GPUs with `<= 9.5 GiB` total or `<= 7.0 GiB` free VRAM (bringing initial VRAM down to ~2.89 GiB).
- **Windows CUDA allocator & concise failure UI**: defaults `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` on Windows; raises a single diagnostic error with pagefile guidance instead of marking every sentence failed on worker startup errors; caps error toast height with scrolling.

### Installer Import, 8 GB ASR Progress & Home Project Folder
- **Windows advanced dependency pack import**: skips duplicate 2.9 GiB SHA-256 verification, merges files via same-volume `os.replace`, enables unbuffered output (`python -u`) and phase progress reporting, and resets the UI progress bar between download and import stages.
- **8 GB transcription stall fixes**: adds `model_heartbeat` around FFmpeg analysis extraction, model loading, and alignment; passes `local_files_only` when offline/local; parses MOSS chunk progress; caps ASR batch size under low free VRAM.
- **Home folder management**: adds **Open projects folder** (auto-creating missing `projects/`) and delete guidance on Home.

### Multi-Speaker Phase 1 (Binaural Cues & Per-Speaker Voice Picker)
- **Binaural ILD cues & basic speaker count**: adds stereo interaural panning cues in `diarization.py` to separate left/right duets without splitting solo binaural actors, and promotes `asr_speaker_count` to Step 1 basic ASR controls.
- **Standalone speaker column, filtering & Step 3 per-speaker UI**: adds a dedicated **Speaker** column (`+ New…`), toolbar speaker filter, undoable batch speaker assignment, `/api/projects/speaker-reference`, and Step 3 **Reference voice per speaker** controls with targeted per-speaker cache invalidation.
- **README**: replaces Sample 3 bilingual and fully dubbed demo videos.

## [3.0.0] - 2026-11-20

### Full-Dub Speech Replacement 3.0 & Audio Engine
- **IndexTTS-2.5 Recommended for the Full Dub & Owned IndexTTS2 Worker**: The "Fully dubbed" preset switches to IndexTTS-2.5 when it is installed (and suggests installing it otherwise). Added dedicated `indextts2_worker.py` and upgraded `indextts25_worker.py` with native S2Mel `target_dur` duration control (`tts_fit_source_duration`), resident `--daemon` JSONL protocol, and reproducible per-take seeding. On the 7-clip benchmark, median onset error dropped to `38.0 ms` (`p95 = 107.6 ms`), and lines needing `>1.1x` speed-up dropped to `8%`.
- **Tempo Still Uses FFmpeg `atempo`, Now Also Slowing Down**: With native duration fitting, few lines still need a tempo change; those use `atempo` (`0.90x` slower to `1.25x` faster). It sounded cleaner than an in-tree phase-locked vocoder in blind listening.
- **Dual Modal/Whisper Voice Anchors & Acoustic Matching**: Projects automatically select separate voiced and whispered reference anchors (`tts_dual_voice_anchor`). Voice matching (`voice_match.py`) runs in a fused single-STFT pass with harmonic-selective whisperization (`replacement_whisper_match = 0.65`) and `60–260 Hz` proximity EQ.
- **Closed-Loop Pronunciation & Residual QC**: Added optional target-language ASR listen-back (`tts_asr_quality_check`) with Mandarin syllable/phonetic CER tolerance (`zh_phonetics.py`) and automatic seed/emotion backoff retries (keeping the lowest-CER take when retries still fail), plus post-mix source-language residual speech scanning (`replacement_residual_asr_check`) and vocal-band Wiener residual suppression (`replacement_residual_suppression`).
- **Bilingual Spatial Accuracy**: Multi-chunk spatial reference estimation and binaural power loudness matching improved bilingual interaural level error (`ild_error_db`) from `15.53 dB` to `3.39 dB` and ITD error from `0.72 ms` to `0.22 ms`.
- **16 kHz Voice-Cloning References to Remove Crackle**: Full-band references carried breath and mouth noise into IndexTTS and caused audible crackle in blind tests. New projects default to `reference_audio_mode = "legacy"` (16 kHz mono). Such dubs have nothing above 8 kHz, so "Restore highs" now rebuilds that band 2 dB hotter to offset what the binaural stage smooths away; on the benchmark the 8–12 kHz gap to the original went from −2.1 dB to −0.1 dB. Existing projects keep their setting.
- **Full-Dub Loudness Matched per Channel**: The source level was measured as the sum of both channels' power, which made every replacement line about 3 dB louder than the original voice. Lines now match the original level.

### Multi-Speaker Diarization & Per-Speaker Dubbing
- **Local Multi-Speaker Diarization (`asr_diarize_speakers`)**: Added pure-PyTorch `CAMPPlus` speaker embedding extraction on CPU (`campplus.py`, zero GPU VRAM contention) with silhouette-guided agglomerative cosine clustering (`diarization.py`) and cross-chunk MOSS speaker stitching.
- **Per-Speaker Voice Cloning & UI Override**: Multi-speaker projects automatically pick independent modal and whisper voice anchors per speaker (`S01`, `S02`, ...) and pass speaker tags into translation context; any sentence's speaker can be edited directly in the project table.

### Sentence Editor, Translation & Desktop UX
- **Virtual-Scrolling Sentence Editor & Interactive Waveform**: Replaced 50-row pagination with virtual scrolling (`1,000+` sentences smooth), per-row independent original-slice (`▶`) and dub (`▶`) audition, active-sentence `<canvas>` waveform with draggable start/end handles, keyboard navigation (`↑/↓`, `Space`, `Enter`), and `Ctrl+Z` / `Ctrl+Y` undo/redo.
- **Duration-Budget Translation, Compression & Revise-on-Request**: Added per-line duration budget (`translation_length_budget`), automatic and one-click over-budget tightening, interactive batch revision proposals, multi-take dub candidates (`+3`), and project-level pronunciation glossary + spoken Chinese number/percentage normalization (`tts_pronunciation_glossary`, `tts_spoken_normalization`).
- **Native Desktop File Handling & Lifecycle**: Added loopback-only native Windows file selection (`files/media`), same-volume hard links for video sources to avoid duplicate multi-GB copies (audio is still copied, so retagging the original cannot break a project), incremental task polling (`tasks/poll`), single-row save I/O optimization, first-run hardware/model/key wizard, quality & mix-type presets, global task panel, and a clean **Quit App** action.
- **Clear Storage Breakdown & Named Missing DLLs**: **Settings → Storage** lists projects, models, each backend environment (previously uncounted, often tens of GB), the main environment, the cache and offline model packs by size, each with an **Open folder** button; deleting IndexTTS on the Models page now also removes its ~9 GB environment. **Check runtime** reads the backend files' dependencies and names the exact missing DLL with its fix (VC++ runtime, Windows Media Foundation, NVIDIA driver). Recognition starts as **Start recognition** and asks before a re-run; the Translation step links straight to a subtitles-only export.

### Batch Engine, Architecture & Model Packs
- **In-Process Batch Execution & Resident Workers**: AutoFlow batch queues execute pipeline stages directly in-process (`inprocess_pipeline.py`) with resident ASR sessions and persistent IndexTTS daemon pools (`tts_daemon.py`), so recognition and dubbing models load once per queue (about 10%-14% of a 10-work run). Removed all blocking CLI `input()` / `print()` calls from `autoflow/`. Batch full-dub output folders are now named `完全配音版` (existing outputs under legacy `替换配音版` remain resumable/reusable).
- **Upgrade Prompt for Legacy Japanese Recognizers**: Users upgrading from 2.x whose saved default Japanese recognizer is still Faster-Whisper, Parakeet, or Kotoba-Whisper get a one-time startup prompt to switch to `anime-whisper` (starting download if needed) or keep their current setting.
- **Project Schema v4 & Modular Codebase**: Upgraded project manifests to `PROJECT_SCHEMA_VERSION = 4` (`Sentence.target_text` with `zh_text` compatibility property, `SentenceStatus` enum, structured `error_code` + recovery hints, and `ProjectSettings` domain submodels `asr` / `translation` / `tts` / `mix` / `separation` / `subtitles`). Split all oversized modules so every source file is `<= 800` lines (`frontend/*.js` and `services/*.py` `<= 500` lines).
- **ModelScope Domestic Mirror Expansion**: Added ModelScope domestic mirror model packs for `anime-whisper`, `qwen3-asr-1.7b`, and `qwen3-asr-1.7b-ja-anime-galgame`.

### Processing Speed (RTX 5070 Ti Laptop GPU, 12 GB VRAM)
Measured processing time divided by source audio duration (lower is faster):
- **Unattended 10-work batch** (about 9 minutes of audio, fully dubbed, IndexTTS2): about 19 minutes, `~2.1x` overall; all 10 works finished. Recognition and dubbing models load once per queue; model loading is about `10%–14%` of the total (vocal separation still loads once per work).
- **Vocal separation** (`Mel-Band RoFormer`): `~0.25x` on first run; reused when the same project is exported again.
- **Speech recognition** (`Faster-Whisper large-v2`): `~0.25x`.
- **Voice-cloned dubbing**: about `2x` the spoken line duration, similar for IndexTTS2 and IndexTTS-2.5; the resident worker takes about 30 s to load once.
- Translation depends on the chosen provider; mixing and voice matching take a few seconds.

### Migration from 2.2.x / 2.3
- Projects from `2.2.x` and `2.3` (`schema_version = 3`) open directly in `3.0.0` and can be translated, mixed and exported further; the regression test uses projects written by the released 2.2.1 code.
- On first save, `3.0.0` writes a timestamped backup under `backups/project-schema-v3-*.json` and upgrades the manifest to `schema_version = 4` (each sentence's `zh_text` becomes `target_text`). An upgraded project no longer opens in 2.x; use that backup to go back.
- Pre-existing legacy settings (`reference_audio_mode = "legacy"`, `tts_normalize_pauses = false`, `tts_fit_source_duration = false`, `tts_dual_voice_anchor = false`, `tts_spoken_normalization = false`) are preserved automatically on migrated projects so **existing synthesized dubs remain 100% cache-valid and are never forced to re-synthesize**.

## [2.2.1] - Previous Stable Release
- See the [v2.2.1 release notes](https://github.com/EveningStudy/asmr-dubber/releases/tag/v2.2.1); earlier versions are listed on GitHub Releases.
