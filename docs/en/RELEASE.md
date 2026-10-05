English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.1/docs/RELEASE.md)

# ASMR Dubber 2.0.1

## Fixes

- 2.0.0 failed to start after extraction (a warning about an incomplete bundled wheelhouse, followed by a failed installation). Fixed.
- The program no longer fails to start after its folder is moved.
- IndexTTS-2.5 acceleration options (DeepSpeed, CUDA Kernel and others) are skipped when the runtime cannot provide them, instead of failing every sentence.

## If you already downloaded 2.0.0

Download 2.0.1 again. If you would rather not, delete the `.asmr-dubber\venv` folder inside the program directory and run `ASMR-Dubber.exe`; it rebuilds the runtime online.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v2.0.1.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

Upgrading from 1.x: close the program, delete the `src` folder and `ASMR-Dubber-Setup.exe` from the old folder, then copy all files of the new version over it. Keep the `.asmr-dubber` folder: it holds your projects, models and settings. Back it up first.

Everything new in 2.0.0 is listed under [v2.0.0](https://github.com/EveningStudy/asmr-dubber/releases/tag/v2.0.0).

[User guide](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.1/docs/en/USER_GUIDE.md) · [Installation and models](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.1/docs/en/INSTALLATION.md)
