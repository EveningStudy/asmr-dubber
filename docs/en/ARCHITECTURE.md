[中文](../ARCHITECTURE.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# Architecture

## Boundaries

UI and CLI call the same pipeline. Projects own settings snapshots and sentence state; global settings initialize future projects. Model execution, installation and media processing remain outside presentation callbacks.

| Area | Modules |
|---|---|
| Entry points | `ui.py`, `ui_services.py`, `cli.py` |
| Orchestration | `pipeline.py`, `autoflow/engine.py`, task controllers |
| ASR/review | ASR adapters, `asr_review.py`, `review_services.py` |
| Translation/script matching | `translation.py`, runtime prompt assets |
| Synthesis | `tts.py`, `tts_backends.py`, `voice_reference.py` |
| Timing/audio/subtitles | `timing.py`, `audio.py`, `subtitles.py` |
| Separation/RTF | `separation.py`, `separation_worker.py`, `experimental_mix.py`, `spatial_rtf.py`, `spatial_cues.py` |
| State | `models.py`, `storage.py`, `user_settings.py`, `lifecycle.py` |
| Runtime/distribution | `model_registry.py`, `runtime_manager.py`, model packages, mirror/platform helpers |
| Browser language | `localization.py`, `locales/en.json`, `locales/switch.js` |

## Project lifecycle

Create copies source media and records integrity metadata. ASR produces intervals; optional separation changes analysis input, not the preserved source. Primary recognition is saved before auxiliary review. Suggestions are version-protected against later manual edits.

Translation preserves IDs/order, with bounded context and memory. Untimed scripts use literal quotes mapped monotonically onto ASR intervals. Python owns offsets; skips are audited. Invalid batches retry by sentence. Manual-review notes distinguish fallback text from validated script text.

TTS caches include relevant text/model/reference parameters. Synthesis writes unique temporary files and validates audio before replacing caches. Mix-only gain, muting, offset, scheduling and RTF changes must not trigger resynthesis.

Mixing schedules Chinese events, applies loudness/RTF and per-sentence gain, then combines the chosen source bed. Separated background is the residual of the source and estimated vocals. Explicit mute overrides overlapping original-vocal retention; it does not duplicate the background. Muted Chinese events still participate in timing planning.

Subtitles use project line width, minimum duration and reading-speed constraints. Save-time settings changes invalidate outputs; writing new subtitle files remains an explicit generation action.

## Persistence and concurrency

Use storage locks and atomic replacement, never direct partial manifest writes. Revision checks reject stale project saves. Task cancellation preserves validated outputs and resumable caches. Installation and inference share runtime exclusion.

AutoFlow keeps per-work plans and state snapshots, with explicit output-replacement consent. Complete timed-subtitle coverage is evaluated per selected track, not per second of waveform. Track-title translation is distinct from subtitle body translation.

## Security and portability

Default data lives under `.asmr-dubber`; configured external roots change the backup boundary. API secrets are plaintext in a separate config file and must not enter project exports. Manifest paths are validated against their project boundary.

The UI stages downloadable outputs under the controlled UI directory; it does not expose arbitrary filesystem paths. Non-loopback serving requires authentication; public share tunnels remain off.

Ordinary download contracts pin name, size, hash and package manifest. External fallback is explicit. Experimental separation has a distinct download/upload consent boundary and isolated runtime. Output-URL retrieval must not forward service credentials.

## Frontend

Sentence editing uses a native paginated table to avoid per-keystroke backend requests. Paging retains all drafts. Language selection is browser-local and changes presentation only; editable values, prompts, media names and model IDs must remain untouched. Catalog additions require switch/restore and draft-preservation tests.

## Extension checklist

A backend needs registry capabilities, validated settings, adapter, static readiness checks, install/repair/cancel behavior, pinned artifacts, complete cache keys, relevant UI fields, tests, a real short-sample check and license/privacy documentation. A dropdown entry alone is not an integration.
