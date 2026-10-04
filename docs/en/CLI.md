[中文](../CLI.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# CLI

Use the platform wrappers to preserve portable Python, FFmpeg, model paths and DLL isolation:

```powershell
.\scripts\windows\run-cli.ps1 --help
```

```bash
bash scripts/linux/run-cli.sh --help
```

`<project>` means a project directory or its `project.json`. Commands below use Windows syntax; on Linux replace the wrapper with `bash scripts/linux/run-cli.sh`.

## Stages

```powershell
.\scripts\windows\run-cli.ps1 doctor --no-network
.\scripts\windows\run-cli.ps1 create 'D:\Media\input.wav' --source-language ja
.\scripts\windows\run-cli.ps1 analyze '<project>'
.\scripts\windows\run-cli.ps1 translate '<project>'
.\scripts\windows\run-cli.ps1 synthesize '<project>'
.\scripts\windows\run-cli.ps1 mix '<project>'
.\scripts\windows\run-cli.ps1 subtitles '<project>' --language bilingual
```

`doctor` checks the configured backends as well as core dependencies; a missing selected backend can return nonzero while the UI itself works. Without `--no-network`, it may contact the translation endpoint.

Create accepts `--projects-root`, `--source-language ja|en|zh`, `--offset-ms` and `--max-speed`. Source media is copied into the new project. Analyze/translate/synthesize accept `--force`; use it only when cached work should be replaced. Normal translation processes enabled sentences without Chinese text.

## Selected sentences and timing

```powershell
.\scripts\windows\run-cli.ps1 synthesize '<project>' --sentence s000001 --sentence s000004
.\scripts\windows\run-cli.ps1 set-timing '<project>' --offset-ms 500 --mode fit-window --max-speed 1.8
```

Sentence IDs are in the manifest/export table. Timing changes invalidate mixed outputs, not valid sentence TTS. `sequential` waits for previous Chinese speech; `fit-window` speeds conflicts up to 1–4× and may leave overlap at the cap.

Mix outputs follow project settings: final mix, Chinese stem or both. Muted Chinese sentences do not require usable synthesis. RTF and sentence gains are mix-time controls; original-vocal overrides require separation.

## Subtitles and import

```powershell
.\scripts\windows\run-cli.ps1 import-transcript '<project>' '.\subtitle.vtt' --kind zh
.\scripts\windows\run-cli.ps1 subtitles '<project>' --language source
```

Subtitle languages: `bilingual`, `zh`, `source` (`ja` remains a compatibility alias). The command writes SRT/LRC and may produce subtitled video for a video project. AutoFlow subtitle-files-only is the route for no finished media outputs. Readability/timing settings come from the project.

## Models

```powershell
.\scripts\windows\run-cli.ps1 install-backend parakeet_nemo
.\scripts\windows\run-cli.ps1 install-backend faster_whisper
.\scripts\windows\run-cli.ps1 install-backend indextts2_5
.\scripts\windows\run-cli.ps1 list-model-packs
.\scripts\windows\run-cli.ps1 import-model-packs --all
.\scripts\windows\run-cli.ps1 prepare-model-pack qwen3-forced-aligner
.\scripts\windows\run-cli.ps1 verify-asr --help
```

Install only the intended backend; other IDs include `kotoba_whisper` and `indextts2`. Static readiness is not actual inference. `verify-asr` runs a real short sample and English needs the matching source-language option. Do not disable artifact verification to force installation.

## Settings, batch and serving

```bash
bash scripts/linux/run-cli.sh settings show
bash scripts/linux/run-cli.sh settings set tts_backend edge_tts
bash scripts/linux/run-cli.sh settings set-translation-key deepseek
bash scripts/linux/run-cli.sh batch --help
```

Keys should be entered through the dedicated prompt/UI or environment, not command arguments that may enter shell history. Settings defaults do not retroactively update existing projects.

```powershell
.\scripts\windows\run-cli.ps1 batch 'D:\Media\Work' --mode audio --layout merged
.\scripts\windows\run-cli.ps1 ui --host 127.0.0.1 --port 7860
```

The web queue exposes more per-track choices and review interactions than a short CLI invocation. Check `--help` before scripting. Non-loopback serving requires authentication; credentials can be supplied through `ASMR_DUBBER_UI_USERNAME`/`ASMR_DUBBER_UI_PASSWORD`. Do not publish the local UI directly to the internet.

Check process exit codes, not just printed success words. Preserve projects/logs after failure; retries reuse valid caches. Cancellation does not mean completed outputs should be deleted.
