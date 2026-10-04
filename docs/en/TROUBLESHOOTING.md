English | [中文](../TROUBLESHOOTING.md)

[← All docs](INDEX.md)

# Troubleshooting

## Do these two things first

**1. Read the error.** When a task fails, the task box on the right shows the reason, and **Log** expands to show more. Most problems explain themselves there.

**2. Run the environment check.** Open **Settings → Logs and diagnostics** and click **Start check**.

![Logs and diagnostics](../../assets/screenshots/en/logs.png)

It checks the GPU driver, model files and runtime libraries. Click the matching **Repair** for whatever it reports.

Whatever the problem, **do not delete the `.asmr-dubber` folder.** Your projects and models are in it.

## Starting up

**Double-clicking the EXE does nothing, or a window flashes and disappears**

- Make sure you extracted the archive and are not running from inside it.
- Put the program on a short path such as `D:\ASMR-Dubber`, not deep inside folders with spaces and non-ASCII names.
- Check whether antivirus software blocked it.

**The black window appears but no browser opens**

Open `http://127.0.0.1:7860` yourself. The black window also prints this address.

**The port is already in use**

Usually the previous run was not fully closed. Close the earlier black window and try again.

**Startup is very slow**

The first start, and the first start after updating or moving, prepares the runtime and takes a few minutes. Antivirus scanning many small files also slows it down.

## Downloading models

**A download fails or is very slow**

- Switch the download source at the top right of the Models page.
- Check free disk space. Models are unpacked after download and need more room than the size shown.
- Click **Download** again. What is already downloaded is not fetched twice.

**The download finished but the model shows as not installed or incomplete**

Expand **Details** for that model to see what is missing. Then go to **Settings → Logs and diagnostics**, pick it under **Repair a model runtime**, and click **Repair**.

**Errors like "the file exists but cannot be found"**

Most likely the path is too long. Move the program to a shorter path and enable Windows long path support. See [Installation and models](INSTALLATION.md#1-download-and-extract).

**DLL or VC++ errors**

Click **Repair Microsoft VC++ runtime** under Logs and diagnostics.

## Recognition

**Out of memory**

- Close other programs using the GPU, such as games or other AI software.
- Under **More settings** for recognition, set the batch size to 1 and reduce the chunk length.
- Use a smaller model, or switch the device to CPU.

**English or Chinese audio comes out as nonsense**

Parakeet and Kotoba only recognise Japanese. Set **Audio language** correctly when creating the project and use Faster-Whisper.

**Quiet speech is missing**

Turn **Silence detection** off and run recognition again. It skips very quiet parts, and whispers get caught by it.

**Strange sentences appear in long silences**

The opposite: turn **Silence detection** on. For Japanese works you can use the ASMR-specific detector, which must be downloaded first.

**The progress bar does not move for a long time**

Loading the model shows no progress, which is normal. Expand **Log** and see whether new lines are appearing. Only if nothing new shows up for several minutes is it stuck.

## Translation

**401, 403, or "invalid key"**

The key is wrong, or was entered under a different provider. Enter it again under **Settings → Cloud services**. Also check that **Translation provider** in the project is the one you entered a key for.

**404 or "model not found"**

The model name is wrong. Check which models your account can use in the provider's console and enter the exact name.

**Some sentences have no translation**

Laughter, breathing, "mm" and similar sounds are not translated. That is normal. If a sentence with real content is empty, expand the task log to look for errors, then click **Translate remaining** again.

**Translation overwrote my edits**

**Translate remaining** never touches existing translations. **Translate everything again** does, and it asks for confirmation first.

## Dubbing

**The voice does not sound like the original at all**

- Make sure you are using a cloning model such as IndexTTS2. Edge TTS has fixed voices and will not resemble the original.
- Try another voice reference. One that is too short, too noisy, or has more than one speaker throws the voice off. See the [user guide](USER_GUIDE.md#choose-a-voice-reference).

**I changed one sentence but many need regenerating**

Editing a translation affects only that sentence. If many are redone, you changed something else as well: the dubbing model, the voice reference, or dubbing parameters. Any of those invalidates the earlier dubs.

**The first sentence takes a long time to start**

The model has to load onto the GPU first. Later sentences are much faster.

**Pause does not stop immediately**

The sentence being generated has to finish first. A cloud request already sent also has to return.

## Export

**The dub and the original overlap and are hard to follow**

On the export step, expand **Dubbing timing** and increase **Dub offset (ms)** so the dub comes in later.

**The dub speeds up and slows down**

When the translation is longer than the original line, the program speeds it up. Shortening the translation is the best fix. You can also lower **Maximum speed-up on overlap**, or switch the timing mode so lines are pushed back instead.

**The dub is too loud or too quiet**

Expand **Volume** and adjust **Relative to original**.

**A channel error appears with spatial following on**

Spatial following needs a stereo original. Turn it off for mono audio.

**Export says a sentence has no translation**

That sentence has original text but an empty translation. Fill it in, or untick the sentence, and export again.

**The original voice is still audible in the replacement mix**

Vocal separation cannot be perfectly clean. This is a limit of the technology today. Also check whether **Original vocal reinsertion** is on, since it deliberately puts some of the original back.

## Settings

**I changed a setting and nothing happened**

You probably changed it in the wrong place. **Settings → New project defaults** only affects projects created afterwards. To change the current project, change it inside the project. See [Settings and storage](CONFIGURATION.md#where-to-change-settings).

Batch works the same way: works already in the queue do not follow the defaults. Click **Edit** and save again.

**The interface is in English but the dub is still Chinese**

The interface language and the dubbing language are separate. Choose the dubbing language under **Dub into** on the dubbing step.

## Still stuck

Click **Export** under **Settings → Logs and diagnostics** and attach the log file to an [issue](https://github.com/EveningStudy/asmr-dubber/issues), along with:

- what you did and which step failed
- your GPU model and VRAM
- which recognition and dubbing models you use

API keys in the log are hidden automatically, but file paths and work names are not. Look it over before sending.
