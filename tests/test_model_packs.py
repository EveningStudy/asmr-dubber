from __future__ import annotations

import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from asmr_dubber import cli as cli_module
from asmr_dubber import environment
from asmr_dubber.constants import (
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    OPTIONAL_ASR_MODEL_REVISIONS,
)
from asmr_dubber.model_packs import (
    MODEL_PACK_MANIFEST,
    ModelPackError,
    ModelPackSource,
    build_model_pack,
    discover_model_packs,
    import_discovered_model_packs,
    import_model_pack,
    imported_hf_snapshot_path,
    inspect_model_pack,
)
from asmr_dubber.platforms import current_platform


def _manifest(
    path: str,
    payload: bytes,
    *,
    pack_id: str = "test-pack",
    platforms: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack_id": pack_id,
        "display_name": "Test Pack",
        "pack_version": "0.0.0",
        "platforms": platforms or [current_platform().id],
        "architectures": ["any"],
        "files": [
            {
                "path": path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def _write_pack(
    archive: Path,
    manifest: dict[str, object],
    payloads: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(archive, "w", allowZip64=True) as handle:
        handle.writestr(MODEL_PACK_MANIFEST, json.dumps(manifest))
        for path, payload in payloads.items():
            handle.writestr(path, payload)


def test_build_import_and_reuse_model_pack(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"model-weights")
    config_payload = b'{"model":"test"}\n'
    (source / "config.json").write_bytes(config_payload)
    archive = tmp_path / "model-packs" / "test.zip"
    events: list[tuple[str, int, int]] = []

    manifest = build_model_pack(
        archive,
        pack_id="test-pack",
        display_name="Test Pack",
        pack_version="0.0.0",
        platforms=(current_platform().id,),
        architectures=("any",),
        sources=(ModelPackSource(source, "models/test"),),
        progress=lambda message, current, total: events.append((message, current, total)),
    )

    assert archive.is_file()
    assert archive.with_suffix(".zip.sha256").is_file()
    assert manifest.uncompressed_bytes == len(b"model-weights") + len(config_payload)
    inspection = inspect_model_pack(archive)
    assert inspection.manifest == manifest
    assert inspection.compatible

    home = tmp_path / "portable"
    first = import_model_pack(archive, home=home)
    assert first.installed_files == 2
    assert first.reused_files == 0
    assert not first.already_installed
    assert (home / "models/test/model.bin").read_bytes() == b"model-weights"

    second = import_model_pack(archive, home=home)
    assert second.installed_files == 0
    assert second.reused_files == 2
    assert second.already_installed
    assert events


def test_discovery_reports_valid_and_invalid_archives(tmp_path: Path) -> None:
    directory = tmp_path / "model-packs"
    directory.mkdir()
    payload = b"ok"
    valid = directory / "valid.zip"
    _write_pack(
        valid,
        _manifest("models/test/model.bin", payload),
        {"payload/models/test/model.bin": payload},
    )
    (directory / "broken.zip").write_bytes(b"not a zip")

    inspections = discover_model_packs(directory)

    assert [inspection.archive.name for inspection in inspections] == [
        "broken.zip",
        "valid.zip",
    ]
    assert inspections[0].manifest is None
    assert inspections[1].manifest is not None


def test_discovery_ignores_windows_recommended_dependency_bundle(tmp_path: Path) -> None:
    directory = tmp_path / "model-packs"
    directory.mkdir()
    dependency_bundle = directory / "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip"
    dependency_bundle.write_bytes(b"handled by the Windows dependency importer")

    assert discover_model_packs(directory) == []


def test_bulk_import_rejects_a_corrupt_archive_in_the_inbox(tmp_path: Path) -> None:
    directory = tmp_path / "model-packs"
    directory.mkdir()
    (directory / "broken.zip").write_bytes(b"not a zip")

    with pytest.raises(ModelPackError, match="broken.zip"):
        import_discovered_model_packs(
            directory=directory,
            home=tmp_path / "portable",
        )


def test_bulk_import_accepts_an_empty_inbox(tmp_path: Path) -> None:
    directory = tmp_path / "model-packs"
    directory.mkdir()

    assert import_discovered_model_packs(directory=directory) == []


def test_cli_all_accepts_an_empty_model_pack_inbox(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "import_discovered_model_packs", lambda **_kwargs: [])

    result = CliRunner().invoke(cli_module.app, ["import-model-packs", "--all"])

    assert result.exit_code == 0
    assert "未发现" in result.stdout


def test_cli_forwards_selected_pack_ids(monkeypatch) -> None:
    calls: list[set[str] | None] = []

    def fake_import(**kwargs):
        calls.append(kwargs["pack_ids"])
        return []

    monkeypatch.setattr(cli_module, "import_discovered_model_packs", fake_import)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "import-model-packs",
            "--all",
            "--pack-id",
            "parakeet-ja-windows",
            "--pack-id",
            "kotoba-whisper-v2.2",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"parakeet-ja-windows", "kotoba-whisper-v2.2"}]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.bin",
        "models/../../outside.bin",
        "/models/absolute.bin",
        "cache/huggingface/model.bin",
        "models\\windows-path.bin",
        "models/CON/file.bin",
        "models/trailing./file.bin",
    ],
)
def test_manifest_rejects_unsafe_or_disallowed_targets(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    payload = b"bad"
    archive = tmp_path / "unsafe.zip"
    _write_pack(
        archive,
        _manifest(unsafe_path, payload),
        {f"payload/{unsafe_path}": payload},
    )

    inspection = inspect_model_pack(archive)

    assert inspection.manifest is None
    assert inspection.error


def test_archive_rejects_unlisted_file(tmp_path: Path) -> None:
    payload = b"model"
    archive = tmp_path / "unlisted.zip"
    _write_pack(
        archive,
        _manifest("models/test/model.bin", payload),
        {
            "payload/models/test/model.bin": payload,
            "payload/models/test/extra.bin": b"not declared",
        },
    )

    inspection = inspect_model_pack(archive)

    assert inspection.manifest is None
    assert "未声明" in inspection.error


def test_archive_rejects_traversal_directory_entry(tmp_path: Path) -> None:
    payload = b"model"
    archive = tmp_path / "traversal-directory.zip"
    with zipfile.ZipFile(archive, "w", allowZip64=True) as handle:
        handle.writestr(
            MODEL_PACK_MANIFEST,
            json.dumps(_manifest("models/test/model.bin", payload)),
        )
        handle.writestr("../outside/", b"")
        handle.writestr("payload/models/test/model.bin", payload)

    inspection = inspect_model_pack(archive)

    assert inspection.manifest is None
    assert "不安全" in inspection.error


def test_archive_rejects_duplicate_zip_paths(tmp_path: Path) -> None:
    payload = b"model"
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w", allowZip64=True) as handle:
        handle.writestr(
            MODEL_PACK_MANIFEST,
            json.dumps(_manifest("models/test/model.bin", payload)),
        )
        handle.writestr("payload/models/test/model.bin", payload)
        with pytest.warns(UserWarning):
            handle.writestr("payload/models/test/model.bin", payload)

    inspection = inspect_model_pack(archive)

    assert inspection.manifest is None
    assert "重复" in inspection.error


def test_import_rejects_platform_mismatch(tmp_path: Path) -> None:
    payload = b"model"
    archive = tmp_path / "other-platform.zip"
    _write_pack(
        archive,
        _manifest(
            "models/test/model.bin",
            payload,
            platforms=["definitely-not-this-platform"],
        ),
        {"payload/models/test/model.bin": payload},
    )

    inspection = inspect_model_pack(archive)
    assert inspection.manifest is not None
    assert not inspection.compatible
    with pytest.raises(ModelPackError, match="不支持当前系统"):
        import_model_pack(archive, home=tmp_path / "home")


def test_import_rejects_corrupt_payload_without_writing(tmp_path: Path) -> None:
    expected = b"expected"
    archive = tmp_path / "corrupt.zip"
    _write_pack(
        archive,
        _manifest("models/test/model.bin", expected),
        {"payload/models/test/model.bin": b"corrupt!"},
    )
    home = tmp_path / "home"

    with pytest.raises(ModelPackError):
        import_model_pack(archive, home=home)

    assert not (home / "models/test/model.bin").exists()


def test_cached_model_path_prefers_imported_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "kotoba-tech/kotoba-whisper-v2.2"
    revision = OPTIONAL_ASR_MODEL_REVISIONS[model_id]
    snapshot = imported_hf_snapshot_path(model_id, revision, home=tmp_path)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"offline model")
    monkeypatch.setattr(environment, "portable_home", lambda: tmp_path)

    assert environment.cached_model_path(model_id) == snapshot.resolve()


def test_indextts_model_pack_definition_covers_every_required_resource(
    tmp_path: Path,
) -> None:
    home = tmp_path / "portable"
    checkpoints = home / "runtimes/index-tts/checkpoints"
    for relative in INDEXTTS_REQUIRED_FILES:
        target = checkpoints / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
    for relative in INDEXTTS_REQUIRED_DIRS:
        required_dir = checkpoints / relative
        required_dir.mkdir(parents=True, exist_ok=True)
        (required_dir / "test-resource.bin").write_bytes(b"checkpoint directory")

    script = Path(__file__).parents[1] / "scripts/create-model-packs.py"
    namespace = runpy.run_path(str(script))
    sources_for_pack = namespace["_sources"]
    sources_for_pack.__globals__["portable_home"] = lambda: home
    sources = sources_for_pack("indextts2-checkpoints")
    manifest = build_model_pack(
        tmp_path / "indextts.zip",
        pack_id="indextts2-checkpoints",
        display_name="IndexTTS2 test pack",
        pack_version="0.0.0",
        platforms=(current_platform().id,),
        architectures=("any",),
        sources=sources,
    )
    targets = {file.path for file in manifest.files}
    prefix = "runtimes/index-tts/checkpoints/"

    assert {prefix + name for name in INDEXTTS_REQUIRED_FILES} <= targets
    for name in INDEXTTS_REQUIRED_DIRS:
        assert any(target.startswith(prefix + name + "/") for target in targets)


def test_parakeet_model_pack_definition_includes_model_notices(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts/create-model-packs.py"
    namespace = runpy.run_path(str(script))
    sources_for_pack = namespace["_sources"]
    sources_for_pack.__globals__["portable_home"] = lambda: tmp_path

    targets = {source.target for source in sources_for_pack("parakeet-ja-windows")}

    assert "models/parakeet/APACHE-2.0.txt" in targets
    assert "models/parakeet/MODEL_NOTICES.txt" in targets
