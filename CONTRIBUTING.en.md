[中文](CONTRIBUTING.md) | English

# Contributing

Read [Architecture](docs/en/ARCHITECTURE.md) and the relevant [backend contract](docs/en/BACKENDS.md) first. Discuss new model families, artifact size, schemas and workflow changes before implementation. Use [Support](SUPPORT.en.md) for usage questions and [Security](SECURITY.en.md) for private vulnerability reports. The [Code of Conduct](CODE_OF_CONDUCT.en.md) applies.

## Development

Python 3.12 and uv; keep the development venv separate from the user's portable runtime.

```bash
uv sync --locked --extra dev
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pyright
uv run --no-sync python scripts/verify_modelscope_artifacts.py
uv run --no-sync pytest
uv run --no-sync asmr-dubber ui --host 127.0.0.1 --port 7860
```

Use an isolated `ASMR_DUBBER_HOME` in tests; never load or overwrite a developer's real configuration. Tests must not implicitly download large models or make paid requests.

## Code contracts

- UI/CLI share pipeline operations; do not duplicate model logic in callbacks.
- Registry IDs/capabilities, validated settings and adapters must agree.
- Use boundary-checked paths, locks, revisions and atomic writes.
- Preserve state on failure/cancellation. No silent model switches or downloads.
- Bound long-audio chunks, context windows, concurrency and response sizes.
- Secrets and private media must not enter logs/exports.
- Keep default persistent data portable unless the user configures external roots.

A backend requires static readiness, installation/repair/cancel behavior, pinned artifacts, complete cache keys, relevant UI controls, tests, short-sample inference evidence and license/privacy notes. A dropdown alone is insufficient.

## Validation

Cover storage conflicts/failure, path escape, timestamp boundaries, channel layout, peak handling, subtitle wrapping, malformed network responses, resume/hash verification and install/inference locking.

UI changes need save/refresh/reopen checks in one process, project autosave/default isolation and queue snapshots, long/paginated tables, dynamic backend choices and draft preservation. Language switching must preserve text, filenames, model IDs and API values. Test Chinese as the first-visit default and English persistence.

Windows installer scripts must parse under PowerShell 5.1 and 7; preserve required UTF-8 BOMs. Artifact changes update mirror/lock/constants/tests together. Real model tests use already installed weights and licensed short audio; report hardware, revisions, precision and tested boundaries.

## Documentation and releases

Keep operational instructions concise: conditions, action, result and failure boundary. Use neutral example paths. Preserve useful screenshots. Add matching English pages and top-of-page language links; runtime prompt templates are code assets and do not get navigation links.

Do not commit portable data, models, media, caches, logs or secrets. Regenerate dependency locks with uv. PR descriptions should explain the user problem, implementation choice, state/cache/network impact and verification performed.

Release tooling reads `docs/RELEASE.md`; ensure it describes only the intended release. Version changes alone are not publication. Record platform-specific clean-install/upgrade/inference coverage rather than claiming all platforms from unit tests.
