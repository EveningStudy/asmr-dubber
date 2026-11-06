English | [中文](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.1/docs/RELEASE.md)

# ASMR Dubber 2.2.1

## Fixes and improvements

- Updated Edge TTS to 7.3.1 with Microsoft's new authentication protocol, fixing HTTP 403 during synthesis and voice previews with older clients.
- The Windows launcher and launch scripts detect incompatible clients and repair existing core environments after an upgrade. Portable releases bundle compatible dependencies for offline repair first. Linux launch scripts also detect and repair older clients.
- Synthesis and previews share retries, deadlines and cancellation. Synthesis respects the single-sentence timeout; previews wait at most 30 seconds. Silent service connections no longer wait indefinitely.
- Separate messages for authentication refusal, rate limits, DNS failures, timeouts and missing audio. Failures remove temporary files while retaining successful synthesis caches.

## Download and upgrade

On Windows, download `ASMR-Dubber-windows-portable-v2.2.1.zip`, extract all of it to a short path, and run `ASMR-Dubber.exe`.

From 2.x: close the program, back it up, and copy the new files over the existing program folder. Keep `.asmr-dubber`, which contains projects, models and settings. The first launch checks and updates the core clients without downloading speech models again.

From 1.x: close the program, delete the old `src` folder and `ASMR-Dubber-Setup.exe`, then copy the new files over the old folder. Keep and back up `.asmr-dubber`.

Edge TTS uses Microsoft's online service. Temporary service or voice failures and rate limits can still occur; retry failed sentences or choose another voice.
