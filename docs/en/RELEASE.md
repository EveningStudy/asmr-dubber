[中文](../RELEASE.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# ASMR Dubber 1.6.0

## Highlights

- Chinese/English web UI with browser-local language selection; bilingual Setup, README and documentation.
- Optional vocal separation using a local runtime or cloud interface. Original recordings and ordinary workflows are preserved.
- RTF (original spatial cue transfer), bilingual/replacement mixing, original-vocal return and per-sentence gain/mute. Mix-only edits reuse valid TTS caches.
- Untimed-script matching uses literal quotes instead of model-counted offsets. Invalid batches retry per sentence; unresolved matches retain existing text and flag manual review.
- Subtitle line width supports 8–500 characters. Save scope and regeneration requirements are explicit.
- Translation-only batch subtitles now use selected timed Chinese subtitles directly, without unnecessary ASR or translation API credentials.
- Reduced repeated language-layer DOM scanning. Documentation navigation now includes audio processing consistently.

Vocal separation and Chinese replacement remain off by default and marked experimental in the application. RTF is independent of separation and needs a stereo original reference. Short or silent references degrade gracefully rather than aborting the whole mix.

## Download and upgrade

Windows: extract `ASMR-Dubber-windows-portable-v1.6.0.zip` completely, run `ASMR-Dubber-Setup.exe`, then `ASMR-Dubber.exe`. Enable long paths and use a short writable directory.

Linux x86_64: run `bash scripts/linux/setup.sh 推荐 en`, then `bash scripts/linux/run-ui.sh`.

Stop tasks and back up projects, settings and finished outputs before upgrading. Preserve `.asmr-dubber` and any external project directories. Restart the application after updating; browser refresh alone does not reload Python code. Re-export subtitles or remix only when the changed settings require it.

[Installation](INSTALLATION.md) · [User guide](USER_GUIDE.md) · [Audio processing](EXPERIMENTAL_AUDIO.md)

## Validation scope

Release checks include direct browser interaction and local media output inspection. These do not guarantee recognition, translation or synthesis quality for every model or input. Paid cloud services and a clean-machine installation are not covered by the local manual smoke check.
