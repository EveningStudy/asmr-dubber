<div align="center">

<img src="design/icon.svg" width="112" alt="ASMR Dubber">

# ASMR Dubber

**Dub audio works into another language. Wherever the original voice is, the dub is there too.**

Recognition, translation, voice-cloned dubbing, mixing and subtitles in one program.<br>
Works with Japanese, English and Chinese audio or video, and dubs into Chinese or English.

[![Release](https://img.shields.io/github/v/release/EveningStudy/asmr-dubber?label=release&color=6d55eb)](https://github.com/EveningStudy/asmr-dubber/releases/latest)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20%7C%20Linux%20x86__64-1d1d1f)
[![License](https://img.shields.io/badge/license-MIT-6e6e73)](LICENSE)

[简体中文](README.md) · **English**

[Download](https://github.com/EveningStudy/asmr-dubber/releases/latest) · [Get started](#get-started-in-three-minutes) · [User guide](docs/en/USER_GUIDE.md) · [All docs](docs/en/INDEX.md)

<br>

<img src="assets/screenshots/en/step3-done.png" alt="The project workspace: sentences on the left, settings for the current step on the right" width="880">

</div>

<br>

## Listen first

Headphones recommended.

The dubbed demos use ASMR Dubber 3.0 factory defaults and the corresponding output presets. The bilingual mix retains the Japanese original and adds Chinese dubbing processed with RTF (spatial-cue transfer from the original). The fully dubbed mix also uses RTF and separates vocals so it can replace the original voice only in dubbed sentences, leaving the rest as recorded. Separation may leave some Japanese speech or alter sound effects. Speech that recognition misses keeps its original voice, so complete removal is not guaranteed.

### Sample 1

**Original**

https://github.com/user-attachments/assets/67e3dbf2-4727-4d9c-8f06-38eeee6fbce1

**Bilingual (RTF)**

https://github.com/user-attachments/assets/3508d200-9477-48a2-baa7-27b5be211cf1

**Fully dubbed (RTF + vocal separation)**

https://github.com/user-attachments/assets/324b242b-ace9-4643-a4be-46545ae80b02

### Sample 2

**Original**

https://github.com/user-attachments/assets/75549f29-1435-448a-851a-86f2828f356a

**Bilingual (RTF)**

https://github.com/user-attachments/assets/1e335590-c1b3-4a20-ba96-ecf77a0423d3

**Fully dubbed (RTF + vocal separation)**

https://github.com/user-attachments/assets/f5099846-7f9d-4c65-aef6-c24e1393c1a2

### Sample 3

**Original**

https://github.com/user-attachments/assets/f1e55d2d-be2a-45b6-90a3-76a92ff47920

**Bilingual (RTF)**

https://github.com/user-attachments/assets/314eec6d-f903-4c6b-ae8f-52882f897ba0

**Fully dubbed (RTF + vocal separation)**

https://github.com/user-attachments/assets/619dfe0f-2a4b-42fb-a769-d75eed1dbd53

### Sample 4 (18+)

**Original**

https://github.com/user-attachments/assets/74c76761-def1-47b3-a58c-03de69483bad

**Bilingual (RTF)**

https://github.com/user-attachments/assets/0939c7d6-1f04-4a24-a393-25b57614aec7

**Fully dubbed (RTF + vocal separation)**

https://github.com/user-attachments/assets/85fdff01-d78c-4fa6-a93c-8b56a182981c

[Watch the introduction and demo of an older version on Bilibili](https://www.bilibili.com/video/BV1f43G6YEov/). Please refer to the documents for current information.

[More audio examples](https://space.bilibili.com/3747523753675973) made with this program. Results vary with the source material, models and settings.

Media is used for demonstration; contact the maintainer about rights concerns.

## What it does

| | |
|---|---|
| 🎧 **Any result you want** | A bilingual mix of original plus dub, a fully dubbed mix in which dubbed sentences replace the original voice, a separate dub track, translated subtitles only, or a video with a cover image. |
| 🧭 **Spatial following** | Voices in these works move from ear to ear and from far to near. The program analyses the position and distance of each original line and places the dub in the same spot instead of leaving it in the centre. |
| 🎼 **Vocal separation** | Splits the original into voice and background and removes the original voice only inside dubbed sentences, making the fully dubbed mix. Laughter, sighs and anything recognition did not pick up keep the original voice. |
| 🗣️ **Voice cloning & multi-speaker** | Pick one line of the original as a reference and the dub speaks in that voice (IndexTTS-2.5 and IndexTTS2 on your own GPU); multi-speaker works can automatically separate speakers and clone each voice independently. |
| ✍️ **Edit any sentence** | Virtual-scrolling editor handles 1,000+ lines smoothly with draggable waveform handles, independent original/dub clip audition, keyboard shortcuts and undo/redo; only edited lines are re-dubbed. |
| 📦 **A whole work in one go** | Hand a work folder to the Batch page. Every track runs through the full pipeline and can be merged into one file or turned into a video. |
| 📝 **Use existing subtitles** | Original-language subtitles skip recognition. Translated subtitles skip translation too. You can also export subtitles only. |
| 🎚️ **Every parameter is exposed** | Over two hundred settings, folded away until you need them. Changes inside a project affect only that project. |

## Get started in three minutes

The steps below use the most common case, Japanese dubbed into Chinese, as the example.

**1. Download and run.** Get the archive from [Releases](https://github.com/EveningStudy/asmr-dubber/releases/latest), extract it to a short path such as `D:\ASMR-Dubber`, and double-click `ASMR-Dubber.exe`. The interface opens in your browser. You do not need to install Python or anything else first.

On the first run Windows may show "Windows protected your PC". This appears because the program has no paid code-signing certificate, not because something is wrong. Click **More info**, then **Run anyway**.

<img src="assets/windows-smartscreen-more-info.png" width="420" alt="SmartScreen prompt: click More info, then Run anyway">

**2. Download models.** Open **Models** and click **Download both**. With an NVIDIA GPU this installs anime-whisper for Japanese recognition and IndexTTS2 for voice cloning; without one, recognition uses the lighter Parakeet. For the fully dubbed version, also download IndexTTS-2.5. If the original is English or Chinese, download Faster-Whisper for recognition instead.

![The Models page](assets/screenshots/en/models.png)

**3. Add a translation key.** Open **Settings → Cloud services** and enter an API key for DeepSeek or any provider you have an account with.

**4. Make something.** Go back to **Projects**, drop in an audio file, pick the original and dubbing languages, and follow the four steps at the top: **Recognition → Translation → Dubbing → Export**.

<img src="assets/screenshots/en/new-project.png" width="560" alt="New project">

No GPU? It still works. Use the built-in Edge TTS for dubbing (online, fixed voices) and run recognition on the CPU, which is slower.

The full walkthrough with screenshots is in the [user guide](docs/en/USER_GUIDE.md). For Linux, see [Installation and models](docs/en/INSTALLATION.md#linux-and-wsl2).

## Documentation

| I want to… | Read |
|---|---|
| Install the program and download models | [Installation and models](docs/en/INSTALLATION.md) |
| Dub one audio file from start to finish | [User guide](docs/en/USER_GUIDE.md) |
| Process a whole work folder | [Batch processing](docs/en/BATCH.md) |
| Use existing subtitles or a script | [Subtitles and scripts](docs/en/SUBTITLE_WORKFLOW.md) |
| Tune spatial following and volume, or make a fully dubbed mix | [Mixing and spatial following](docs/en/EXPERIMENTAL_AUDIO.md) |
| Have several models cross-check recognition | [Multi-model review](docs/en/AUDIO_REVIEW_TUTORIAL.md) |
| Switch recognition models, dubbing services or translation providers | [Models and services](docs/en/BACKENDS.md) |
| Understand where settings live, free disk space, back up | [Settings and storage](docs/en/CONFIGURATION.md) |
| Look up a parameter | [Parameter reference](docs/en/PARAMETERS.md) |
| Fix an error | [Troubleshooting](docs/en/TROUBLESHOOTING.md) |

For developers: [Changelog](CHANGELOG.en.md) · [Architecture](docs/en/ARCHITECTURE.md) · [Command line](docs/en/CLI.md) · [Contributing](CONTRIBUTING.en.md) · [All docs](docs/en/INDEX.md)

## Things to know

- **Your data lives in the program folder.** Projects, models and settings are under `.asmr-dubber`. Copy the whole folder to back up or move.
- **API keys are stored in plain text** at `.asmr-dubber/config/secrets.json`. Do not share that folder.
- **Cloud services receive your content.** Translation sends the script text. Cloud dubbing sends the translation and the reference audio. Nothing is uploaded when you use local models only.
- **The fully dubbed mix relies on vocal separation.** It can leave traces of the original voice, keep the original where a quiet whisper was not recognized, and can slightly affect sound effects.

## License

The project code is released under the [MIT](LICENSE) license. Models, runtimes, cloud services, input works and generated content are subject to their own terms; open-source code does not grant rights to them. See [third-party notices](docs/en/THIRD_PARTY_NOTICES.md).

## Friends

- [LINUX DO](https://linux.do/): a new kind of community
