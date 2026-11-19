English | [中文](CHANGELOG.md)

# Changelog

All notable changes to ASMR Dubber are documented in this file.

## [3.0.0] - 2026-11-18

### Full-Dub Speech Replacement 3.0 & Audio Engine
- **IndexTTS-2.5 Recommended for the Full Dub & Owned IndexTTS2 Worker**: The "Fully dubbed" preset switches to IndexTTS-2.5 when it is installed (and suggests installing it otherwise). Added dedicated `indextts2_worker.py` and upgraded `indextts25_worker.py` with native S2Mel `target_dur` duration control (`tts_fit_source_duration`), resident `--daemon` JSONL protocol, and reproducible per-take seeding. On the 7-clip benchmark, median onset error dropped to `38.0 ms` (`p95 = 107.6 ms`), and lines needing `>1.1x` speed-up dropped to `8%`.
- **Tempo Still Uses FFmpeg `atempo`, Now Also Slowing Down**: With native duration fitting, few lines still need a tempo change; those use `atempo` (`0.90x` slower to `1.25x` faster). It sounded cleaner than an in-tree phase-locked vocoder in blind listening.
- **Dual Modal/Whisper Voice Anchors & Acoustic Matching**: Projects automatically select separate voiced and whispered reference anchors (`tts_dual_voice_anchor`). Voice matching (`voice_match.py`) runs in a fused single-STFT pass with harmonic-selective whisperization (`replacement_whisper_match = 0.65`) and `60–260 Hz` proximity EQ.
- **Closed-Loop Pronunciation & Residual QC**: Added optional target-language ASR listen-back (`tts_asr_quality_check`) with Mandarin syllable/phonetic CER tolerance (`zh_phonetics.py`) and automatic seed/emotion backoff retries, plus post-mix source-language residual speech scanning (`replacement_residual_asr_check`) and vocal-band Wiener residual suppression (`replacement_residual_suppression`).
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

### Batch Engine, Architecture & Model Packs
- **In-Process Batch Execution & Resident Workers**: AutoFlow batch queues execute pipeline stages directly in-process (`inprocess_pipeline.py`) with resident ASR sessions and persistent IndexTTS daemon pools (`tts_daemon.py`), keeping model load overhead `< 10%` across multi-track queues. Removed all blocking CLI `input()` / `print()` calls from `autoflow/`.
- **Project Schema v4 & Modular Codebase**: Upgraded project manifests to `PROJECT_SCHEMA_VERSION = 4` (`Sentence.target_text` with `zh_text` compatibility property, `SentenceStatus` enum, structured `error_code` + recovery hints, and `ProjectSettings` domain submodels `asr` / `translation` / `tts` / `mix` / `separation` / `subtitles`). Split all oversized modules so every source file is `<= 800` lines (`frontend/*.js` and `services/*.py` `<= 500` lines).
- **ModelScope Domestic Mirror Expansion**: Added ModelScope domestic mirror model packs for `anime-whisper`, `qwen3-asr-1.7b`, and `qwen3-asr-1.7b-ja-anime-galgame`.

### Stage Processing Speed Reference (RTX 5070 Ti Laptop GPU, 12 GB VRAM)
Measured wall-clock processing time divided by source audio duration (`RTF`, lower is faster):
- **Vocal Separation** (`Mel-Band RoFormer`, cached by source SHA-256): `~0.18x–0.25x` on first run; `0.00x` when cached across re-exports.
- **Speech Recognition** (`anime-whisper` + ASMR VAD + Qwen3 ForcedAligner): `~0.12x–0.20x`.
- **Translation** (`deepseek-v4-flash` with context & duration budget): `~0.03x–0.06x`.
- **Voice-Cloned Dubbing** (`IndexTTS-2.5` / `IndexTTS2` resident worker): `~0.45x–0.85x` of spoken line duration (`< 10%` model-load share on 10-track queues).
- **Mixing, Spatial RTF & Voice Matching**: `~0.03x–0.06x`.

### Migration from 2.2.x / 2.3
- Projects from `2.2.x` and `2.3` (`schema_version = 3`) open directly in `3.0.0` and can be translated, mixed and exported further; the regression test uses projects written by the released 2.2.1 code.
- On first save, `3.0.0` writes a timestamped backup under `backups/project-schema-v3-*.json` and upgrades the manifest to `schema_version = 4` (each sentence's `zh_text` becomes `target_text`). An upgraded project no longer opens in 2.x; use that backup to go back.
- Pre-existing legacy settings (`reference_audio_mode = "legacy"`, `tts_normalize_pauses = false`, `tts_fit_source_duration = false`, `tts_dual_voice_anchor = false`, `tts_spoken_normalization = false`) are preserved automatically on migrated projects so **existing synthesized dubs remain 100% cache-valid and are never forced to re-synthesize**.

## [2.2.1] - Previous Stable Release
- See [docs/en/RELEASE.md](docs/en/RELEASE.md) and Git history up to tag `v2.2.1`.
