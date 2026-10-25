English | [中文](../DEFAULTS_AUDIT_2026-10-16.md)

# Defaults review · 2026-10-16

The review covers all **218 settings** and compares schema defaults with the saved global settings on the development machine. Existing projects and saved preferences were not changed. The [Chinese report](../DEFAULTS_AUDIT_2026-10-16.md) contains the complete field-by-field assessment and 17 saved-value differences; [the English parameter reference](PARAMETERS.md) lists every default, range and visibility rule.

Defaults are practical starting points, not a universally optimal configuration. Only MOSS received actual GPU inference tests in this change. The rest of this review checks implementation and parameter contracts, with conditional listening recommendations rather than a completed model-quality benchmark.

| Area | Recommendation |
|---|---|
| DeepSeek model name | Update `deepseek-v4-flash` to the current official name `deepseek-flash`. The old alias already routes to V4.1, so renaming alone does not improve quality. |
| Binaural spatial transfer | Enable RTF for binaural ASMR when transfer is wanted. The user's saved setting already enables it; the schema default is off. |
| Automatic acceleration | Compare 1.25–1.4 for natural whispers against the schema's 1.8 and the saved 1.7. Lower limits may require shorter translations or different scheduling. |
| Scheduling | `fit_window` respects source windows; the saved `sequential` setting favors natural flow but can introduce delay. Replacement output should start with zero dubbing offset. |
| Voice reference | Enable manual selection for finished work. The saved batch setting disables the wait. Clean, continuous speech from the correct character matters more than blindly raising decoding values. |
| Multi-model review | Start with `suggest`, rather than the saved `conservative`, when enabling review. MOSS adds a different model family. Review is currently disabled. |
| Whisper previous-text conditioning | Try disabling it for repetition loops and long silences; cross-window text consistency may decrease. |
| Emotion reference | Keep per-sentence references for expressive variation. Compare a stable reference if quiet or noisy references make delivery unstable. The saved strength is 0.55, against a schema default of 0.5. |
| Separation | Do not enable it as a general improvement for bilingual mixes. Compare MDXC's native segment size against the current forced size before changing defaults. |
| Default recognizer | Keep Parakeet until MOSS has been evaluated on actual Japanese whisper recordings. Successful English execution does not establish Japanese ASMR superiority. |

Keep peak protection, stable speaker references, translation context, fixed seeds and conservative acceleration options unless a specific comparison justifies changing them. Device, path, timeout, container, naming and download controls are not quality rankings. Parameters belonging to inactive backends may have no effect.

Sources: [MOSS official inference](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), [DeepSeek model names and alias routing](https://api-docs.deepseek.com/quick_start/pricing/), [Faster-Whisper conditioning documentation](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), and the current application's ASR/TTS/mixing code. The saved settings of every historical project were not inspected, and no paid translation or synthesis calls were made.
