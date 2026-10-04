from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_advanced_dependency_pack_and_analysis_model_packs_are_reused() -> None:
    mirrors = json.loads((ROOT / "mirrors.json").read_text(encoding="utf-8"))

    assert mirrors["modelscope_artifacts"]["windows_advanced_dependency_archives"] == [
        "https://modelscope.cn/models/EveningStudyW/"
        "ASMR-Dubber-Windows-Advanced/resolve/master/"
        "ASMR-Dubber-Windows-Advanced-Dependencies-v1.0.0.zip"
    ]
    assert "qwen3-forced-aligner" in mirrors["model_pack_sources"]
    assert "whisper-vad-asmr-onnx" in mirrors["model_pack_sources"]

    setup = "\n".join(
        (ROOT / "scripts/windows" / name).read_text(encoding="utf-8-sig")
        for name in ("setup.ps1", "setup-application.ps1", "setup-models.ps1")
    )
    dependencies = (ROOT / "scripts/windows/recommended-dependencies.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "Import-ASMRDubberAdvancedDependencies" in setup
    assert "qwen_asr" in dependencies
    assert "onnxruntime" in dependencies
    advanced_function = dependencies.split("function Import-ASMRDubberAdvancedDependencies", 1)[1]
    assert (
        "[switch]$MergeExisting"
        in advanced_function.split("if (Test-ASMRDubberAdvancedDependencies", 1)[0]
    )
    assert "bafd2268de9a83bbf391ba8918d1798d24f703b023af70e8f623b2dbffc9a178" in (dependencies)


def test_windows_powershell_scripts_are_compatible_with_legacy_utf8_detection() -> None:
    scripts = sorted((ROOT / "scripts").rglob("*.ps1")) + sorted((ROOT / "launcher").rglob("*.ps1"))
    assert scripts
    for script in scripts:
        assert script.read_bytes().startswith(b"\xef\xbb\xbf"), script
        assert "utf8NoBOM" not in script.read_text(encoding="utf-8-sig")


def test_windows_native_process_arguments_use_shared_quoting() -> None:
    mirrors = (ROOT / "scripts/mirrors.ps1").read_text(encoding="utf-8")
    assert "ConvertTo-ASMRDubberWindowsCommandLineArgument" in mirrors
    assert "$StartInfo.Arguments = Join-ASMRDubberWindowsCommandLine" in mirrors

    for relative in (
        "scripts/windows/setup.ps1",
        "scripts/windows/install-backend.ps1",
        "scripts/windows/install-indextts2.ps1",
        "scripts/windows/install-parakeet.ps1",
        "scripts/windows/run-cli.ps1",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "ProcessStartInfo]::new" not in source


def test_webui_backend_installer_resolves_project_root_and_reuses_setup_downloads() -> None:
    source = (ROOT / "scripts/windows/install-backend.ps1").read_text(encoding="utf-8-sig")

    assert 'Resolve-Path (Join-Path $PSScriptRoot "..\\..")' in source
    assert "Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)" not in source
    assert "Get-ASMRDubberWheelhouse" in source
    assert "Invoke-ASMRDubberUvOfflineWheelhouse" in source
    assert 'ArchiveMirrorName "windows_application_wheelhouse_archives"' in source
    assert 'ArchiveMirrorName "windows_cuda_wheelhouse_archives"' in source
    assert "Import-ASMRDubberAdvancedDependencies" in source
    assert "-MergeExisting" in source


def test_indextts25_is_webui_only_and_does_not_change_setup_profiles() -> None:
    setup_files = (
        "scripts/windows/setup.ps1",
        "scripts/linux/setup.sh",
        "scripts/windows/recommended-dependencies.ps1",
        "scripts/windows/create-recommended-dependency-pack.ps1",
        "scripts/import_windows_dependency_pack.py",
    )
    for relative in setup_files:
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        assert "indextts25" not in source.casefold()
        assert "indextts2_5" not in source.casefold()
        assert "indextts-2.5" not in source.casefold()

    windows = (ROOT / "scripts/windows/install-indextts25.ps1").read_text(encoding="utf-8-sig")
    linux = (ROOT / "scripts/linux/install-indextts25.sh").read_text(encoding="utf-8")
    for source in (windows, linux):
        assert "indextts2_5-checkpoints" in source
        assert "requirements.txt" in source
        assert "--offline" in source
        assert "--no-index" in source
        assert "indextts" in source
    assert 'Join-Path $_.Directory.FullName "indextts"' in windows
    assert 'Join-Path $IndexWheelhouse "requirements.txt"' in windows
    assert '"--find-links", $IndexWheelhouse, "--requirement", $Requirements' in windows
    assert 'test -d "$(dirname "$1")/indextts"' in linux
    assert 'REQUIREMENTS="$ASMR_WHEELHOUSE_RESULT/requirements.txt"' in linux
    assert '--requirement "$REQUIREMENTS"' in linux
