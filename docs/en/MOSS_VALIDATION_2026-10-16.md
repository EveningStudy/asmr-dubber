English | [中文](../MOSS_VALIDATION_2026-10-16.md)

# MOSS validation · 2026-10-16

## Summary and environment

The change adds an isolated recognizer, 17 adjustable parameters, speaker-label persistence/display, managed model downloads/removal and runtime diagnostics. Existing uncommitted changes were preserved. No Git commit or push was made.

Inference environment: Windows, RTX 5070 Ti Laptop 12 GB, Torch 2.11.0+cu130, Transformers 5.6.0, BF16 and SDPA. UI testing used `http://127.0.0.1:7867/`, 1440×1000 and 390×900, with temporary settings. Browser plugin not available; the existing Playwright/Edge route was used without installing browser dependencies.

## Checks

| Check | Result |
|---|---|
| Identity, meaningful rendering, no error overlay | Passed: ASMR Dubber, Models and Settings pages. |
| Console and page errors | Passed: empty error list. |
| Conditional controls | Passed: MOSS controls follow backend selection; sampling fields follow the sampling toggle. |
| Save and reload | Passed: custom chunk length 240 seconds survives refresh. The schema default remains 300. |
| Desktop and narrow viewport | Passed: no horizontal overflow; the entire context token count is visible. |
| Speaker persistence | Passed: reopening a saved GPU result displays `c0001:S01`. |
| Short speech inference | Passed: 10.435-second public English speech returned text, timestamps and a speaker label. |
| Chunked inference | Passed: 320.18-second repeated-clip input, two chunks, 56 segments, 90.11 seconds; last segment ends at 318.87 seconds. |
| MOSS regression tests | Passed: 10 tests covering source order, resume, corruption, failure, cancellation, parsing, settings, process registration, status and removal. |
| ModelScope publication | Passed: 19 files committed to the existing mirror, zero deletions. |
| Anonymous mirror download | Passed: every byte of all 18 runtime files was hashed and matched. |
| Ruff and artifact contract | Passed. |

Final offline regression: **712 passed, 3 skipped, 1 deselected in 32.79 seconds**, using `pytest -m "not network and not gpu and not slow"`. The real GPU and anonymous mirror-download checks were run separately. Detailed output is in `.asmr-dubber/temp/moss-pytest-final.log`.

Interaction path: Models → Settings / Recognition → choose MOSS → expand controls → change chunk length → reload → enable sampling → switch backend. A separate saved-project check verified the speaker label in sentence details.

Main files: `moss.py`, `moss_worker.py`, `moss_manifest.json`, `asr.py`, the backend registry, parameter layout and model services. Commands included `pytest -m 'not network and not gpu and not slow'`, `ruff check`, the artifact verifier and project-style Playwright scripts.

Japanese whisper accuracy, real multi-speaker diarization, Linux and CPU inference remain untested. Repeated speech validates chunk handling, not natural long-recording accuracy. Cross-chunk speaker identity and per-speaker TTS routing are not implemented.

## Screenshots

![Desktop settings](../../assets/screenshots/zh/moss-settings.png)
![Narrow settings](../../assets/screenshots/zh/moss-settings-mobile.png)
![Persisted speaker label](../../assets/screenshots/zh/moss-speaker.png)
