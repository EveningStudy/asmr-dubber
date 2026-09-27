[中文](../MODELSCOPE_UPLOADS.md) | English

[Documentation index](INDEX.md) · [README](../../README.en.md)

# ModelScope artifact maintenance

Published artifact paths are immutable. Changes to contents require a new filename/revision and synchronized contracts, not overwriting bytes behind an existing URL.

## Sources of truth

`mirrors.json`, `modelscope-artifacts.lock.json`, model-package constants and platform dependency contracts must agree on filenames, byte lengths and SHA-256. Internal manifests additionally enumerate safe relative paths and file hashes. Review the actual pinned revision and its license, not just the upstream latest page.

Portable mirrors, Windows dependency archives, model ZIPs, IndexTTS source/runtime assets and platform wheelhouses are distinct products. A model package is not a complete application installation. Linux and Windows wheels are not interchangeable.

## Build and verify

Use repository scripts rather than hand-editing lock files:

```powershell
.\.asmr-dubber\venv\Scripts\python.exe .\scripts\create-model-packs.py --help
.\.asmr-dubber\venv\Scripts\python.exe .\scripts\verify_modelscope_artifacts.py
```

Model ZIPs must preserve licenses/notices and their manifest. Do not include private media, credentials, developer paths or unneeded caches. Fixed IndexTTS source revisions and the corresponding wheelhouse/model set must be compatible. IndexTTS-2.5 uses independent artifacts and does not replace the IndexTTS2 contract.

## Publish procedure

1. Build in an isolated environment using pinned inputs.
2. Record exact bytes and hashes; verify archive paths and manifest.
3. Upload under a new immutable name.
4. Download the published artifact and verify it again.
5. Update all contract references together; run artifact/downloader tests.
6. Exercise resume, cancellation, wrong-size/hash rejection and read-only cache reuse.
7. Test the intended installation profile on the target platform before release.

Do not automatically enable overseas sources because an artifact is missing. Current users should retain resumable files and receive a clear failure. Token-gated repositories require explicit credentials; tokens never belong in repository URLs, documentation or logs.

## Read-only cache reuse

`ASMR_DUBBER_LOCAL_CACHE_ROOTS` or Windows Setup's `-LocalCacheRoot` can reuse another installation's verified artifacts. The source directory stays read-only. Files that fail the current contract must not be adopted because their name happens to match.

Clean installation, upgrade, offline import and real inference are separate validations. Report platform, commit, installation path and exact verification coverage. Core-wheel dependency inventories do not cover every isolated model runtime.
