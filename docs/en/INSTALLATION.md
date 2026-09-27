[中文](../INSTALLATION.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Installation

## Requirements

Windows 10/11 x86_64, a writable NTFS directory and system `curl.exe`; or Linux x86_64 with `bash`, `curl`, `tar` and `getconf`. macOS, ARM64 and 32-bit systems are unsupported. WSL2 runs the Linux path.

Use a short path such as `D:\ASMR-Dubber`, not Program Files, a ZIP viewer or a live-synced folder. Enable Windows long paths when the OS exposes that option; see the [README illustration](../../README.en.md#install). No preinstalled Python, uv, Git, FFmpeg, CUDA Toolkit or PowerShell 7 is required. Windows PowerShell 5.1 is supported.

Local CPU inference is possible for supported backends but often much slower. CUDA requires a compatible NVIDIA driver. VRAM figures are approximate loading thresholds, not stability guarantees: Parakeet roughly 6 GB recommended; Kotoba/Faster-Whisper often benefit from 6 GB; IndexTTS2/2.5 prefer 10 GB or more. Keep batch size 1 on constrained devices. AMD/Intel GPUs are not CUDA devices.

## Profiles

| Profile | Installed size / suggested free space | Contents |
|---|---|---|
| Core | ~2 GB / 5 GB | UI, media tools, Edge TTS and API clients |
| Recommended | ~24–28 GB / 35 GB | Core + Parakeet CTC 1.1B JA GAL and TDT/CTC 0.6B JA; IndexTTS2 on NVIDIA |
| Advanced | ~33–39 GB / 50 GB | Recommended + Kotoba v2.2, Faster-Whisper large-v2, Japanese ASMR VAD and Qwen3 ForcedAligner 0.6B |

Without NVIDIA, automatic IndexTTS2 installation is skipped and space usage is lower. IndexTTS-2.5 is optional from Devices & models, not included in any profile. It has a separate runtime and approximately 15 GB of model/dependency data, plus installation/backup space.

## Windows

Download the portable ZIP from [Releases](https://github.com/EveningStudy/asmr-dubber/releases/latest), extract completely, then run `ASMR-Dubber-Setup.exe`. Choose Chinese (default) or English, then profile 1/2/3; Enter selects Recommended. Run `ASMR-Dubber.exe` after completion. The language prompt is separate from profile selection. `--lang=en` and `--lang=zh` skip the language prompt.

```powershell
.\ASMR-Dubber-Setup.exe --lang=en
.\scripts\windows\setup.ps1 -Profile Core
.\scripts\windows\setup.ps1 -Profile Recommended -SkipRecommendedTTS
```

Interrupted installations can be rerun. Verified completed downloads are reused; valid partial files remain resumable. Keep `.asmr-dubber/logs/setup-*.log` for diagnosis. Installation language does not alter model selection or downloads.

## Linux

```bash
bash scripts/linux/setup.sh 基础
bash scripts/linux/setup.sh 推荐
bash scripts/linux/setup.sh 进阶
bash scripts/linux/run-ui.sh
```

Choose one setup profile, not all three. To omit recommended TTS:

For English setup messages: `bash scripts/linux/setup.sh Recommended en`. Raw third-party tool logs keep their original language.

```bash
ASMR_DUBBER_SKIP_RECOMMENDED_TTS=1 bash scripts/linux/setup.sh 推荐
```

Scripts prepare a project-local runtime rather than modifying system Python or global PATH. Server use is available through `scripts/linux/run-cli.sh`.

## Downloads and offline packages

Ordinary Setup follows `mirrors.json`: ModelScope artifacts first, configured domestic Python mirror where permitted, external sources disabled unless explicitly allowed. Fixed artifacts are checked by size/SHA-256 and model packages by internal manifest. Do not alter hashes to bypass failures.

```powershell
$env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS = '1'
.\ASMR-Dubber-Setup.exe
```

This opt-in affects only the process. Private ModelScope repositories may require `MODELSCOPE_API_TOKEN`; do not put it in shared logs/scripts.

Place intact ASMR Dubber model ZIPs in `model-packs`, then rerun the appropriate Setup profile or use Scan & import in Devices & models. For multipart RAR downloads, first extract the complete ZIP using the archive tool; the application does not read RAR directly.

```powershell
.\scripts\windows\run-cli.ps1 list-model-packs
.\scripts\windows\run-cli.ps1 import-model-packs --all
```

Experimental separation uses its own explicit download consent and external sources; it is not installed by these profiles. [Separation setup](EXPERIMENTAL_AUDIO.md).

## Verify and repair

Devices & models performs static checks without loading models. For a CLI report:

```powershell
.\scripts\windows\run-cli.ps1 doctor --no-network
```

Use `verify-asr --help` for actual short-audio inference. Static availability is not inference or quality validation. A wheel supplies Python CLI/API code, not the full portable installer assets; use the portable/source distribution for in-app local model installation.

Missing DLLs: run runtime diagnostics in Devices & models, then the explicit repair action. Task errors diagnose without automatically installing software. Do not download individual DLLs from arbitrary sites.

## Move, back up, remove

Stop tasks and close the launcher. Copy the full program/data directory and any external project root. After moving, rerun Setup to repair environment paths. Secrets are plaintext and move with the directory.

Routine repair does not require deleting `.asmr-dubber`. Keep runtime rollback backups until the repaired environment works. To uninstall, stop the app/services, back up required projects, then remove the application directory; separately handle any configured external data and revoke exposed keys.
