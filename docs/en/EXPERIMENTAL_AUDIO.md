[中文](../EXPERIMENTAL_AUDIO.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Audio processing: RTF, vocal separation and sentence mixing

Separation and Chinese replacement are **experimental, not recommended**; both are off by default. Separation can damage whispers, breaths and mouth sounds, and Japanese speech may remain in the background. RTF can alter timbre, phase and proximity. Neither promises lossless replacement.

## Routing

Separation has its own settings tab between General and ASR. RTF, mix mode and original-vocal retention live under Mixing & subtitles. RTF does not require separation, but requires a stereo original reference.

| Mode | Output |
|---|---|
| Bilingual | Original recording + Chinese speech |
| Chinese replacement | Separated background + Chinese speech; requires separation |
| Replacement with retention | Above + selected original-vocal intervals |

Conflicting replacement/separation settings are rejected on save; the application does not silently switch modes. Retention controls appear only for replacement mode.

1. Enable separation, select a downloaded model, save to the project and run ASR. Recognition and review use the separated vocals; the original recording remains intact.
2. Translate, edit and synthesize normally. Voice-cloning references still use the original audio.
3. Enable RTF if wanted, then mix. It uses matching stereo intervals to estimate relative level, spectrum and phase. No HRTF or Meta model is used.

## Per-sentence controls

The right side of the sentence editor contains original-vocal mode, original gain, Chinese playback and Chinese gain.

- Original mode inherits global behavior by default. Override with Keep or Mute. These controls affect only the separated vocal stem and require separation, including in bilingual mode.
- Gain is an adjustment on top of global processing: default 0 dB, range −60 to +12 dB. Final peak protection still applies.
- Chinese playback mutes only the final voice, not the text or TTS cache. It is distinct from Process Chinese.
- Save the table and remix. No ASR, translation or TTS rerun is needed for gain/mute edits.
- Overlapping original intervals do not duplicate vocals. Explicit mute wins on overlap.

## Original-vocal retention

Default: none. Manual mode accepts sentence IDs separated by commas/spaces. Automatic mode selects disabled sentences or sentences without playable Chinese, including filtered content. This is not a nonverbal-sound classifier: it may restore Japanese dialogue. Listen before exporting.

Only the separated vocal signal is restored; the background is not added twice. Sounds outside recognized sentence intervals may still be lost. Add a sentence interval and select its ID if manual retention is needed.

## Local backend

Independent runtime: `.asmr-dubber/runtimes/separation/.venv`. Model directory: `.asmr-dubber/models/separation`. The adapter uses audio-separator 0.47.0 and supports its Mel/BS-RoFormer, MDXC, MDX, VR and Demucs interfaces.

Baseline: `vocals_mel_band_roformer.ckpt` by Kimberley Jensen. It is a music-vocal separator, not an ASMR dialogue/effects model. Bandit and SAM Audio are not local integrations. A catalog entry does not mean its weights are downloaded or tested.

Advanced JSON exposes inference parameters for the selected architecture, not arbitrary training/network configuration. The isolated ONNX environment uses CPU; the PyTorch RoFormer path can use CUDA. Check each model's license.

Installation/download requires explicit consent. Sources are PyPI, PyTorch, GitHub and Hugging Face, separate from the ordinary Setup ModelScope policy. The default model is hash-pinned; other downloads record local integrity metadata. Inference refuses implicit downloads.

## Cloud contracts

| Backend | Contract |
|---|---|
| Replicate | Fixed 64-hex version ID, configurable audio/output fields and parameters; inline audio limited to 8 MiB per request |
| Custom HTTP | Multipart file + `model` + JSON-string `parameters`; JSON response containing a vocal-stem HTTPS URL |

Replicate output paths are relative to `output`, for example `vocals` or `0`. Use chunks of at most 30 seconds as a starting point. Cloud processing requires upload consent and may cost money. Failed jobs are not automatically resubmitted. Downloaded outputs do not inherit credentials and redirects are rejected. Mock-tested contracts are not live-provider validation.

## RTF and recovery

References shorter than 0.1 s use bounded left/right gain instead of phase estimation; silence or missing reference signal falls back to center. The log records the fallback. Chinese duration and scheduling remain unchanged, and no neighboring audio is borrowed. Corrupt/nonfinite samples still raise errors.

RTF runs after duration scheduling and loudness processing. Strength controls dry/wet mixing; level/color controls tune cue transfer; FFT/hop parameters control time-frequency resolution. Mono/multichannel originals do not provide the required stereo reference. RTF changes require only remixing.

Separation runs in chunks and reuses valid caches. Model/device/parameter changes create a distinct cache. Background is the exact residual `original − estimated vocals`; this preserves reconstruction, not perceptual separation quality. Test a short excerpt before processing a full work.
