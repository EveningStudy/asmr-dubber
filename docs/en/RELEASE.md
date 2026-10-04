English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.0/docs/RELEASE.md)

# ASMR Dubber 2.0.0

The interface has been rebuilt from scratch and installation works differently.

## The new interface

- Four entries on the left: Projects, Batch, Models, Settings.
- A project follows four steps: Recognition, Translation, Dubbing, Export. Sentences are on the left and only the current step's settings are on the right.
- Every parameter is still there, folded away until you need it.
- Settings changed inside a project affect only that project. **Settings → New project defaults** affects only projects created afterwards. The "save scope" option is gone.
- Changes save automatically. There is no Save button.

## The new installation

- `ASMR-Dubber-Setup.exe` and installation profiles are gone. Extract the archive and run `ASMR-Dubber.exe`.
- Models are downloaded on demand from the **Models** page, with pause and resume.
- The bottom left of the interface tells you when a new version is available and can download and install it for you.

## Other changes

- Uploaded source files are removed once used and no longer take up disk space.
- **Translate everything again** asks for confirmation first, so hand-edited translations are not overwritten by accident.
- The documentation has been rewritten with screenshots of the new interface.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v2.0.0.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

Upgrading from 1.x: close the program, delete the `src` folder and `ASMR-Dubber-Setup.exe` from the old folder, then copy all files of the new version over it. Keep the `.asmr-dubber` folder: it holds your projects, models and settings. Back it up first.

On Linux x86_64, run `bash scripts/linux/setup.sh Core` from the source directory, then `bash scripts/linux/run-ui.sh`.

[User guide](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.0/docs/en/USER_GUIDE.md) · [Installation and models](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.0/docs/en/INSTALLATION.md)
