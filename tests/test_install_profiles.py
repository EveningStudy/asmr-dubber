from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES

ROOT = Path(__file__).parents[1]


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell behavior")
def test_windows_downloader_reuses_verified_destination_without_network(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    assert powershell is not None
    destination = tmp_path / "complete.zip"
    partial = Path(f"{destination}.partial")
    destination.write_bytes(b"already complete")
    partial.write_bytes(b"stale partial")
    expected = hashlib.sha256(destination.read_bytes()).hexdigest()

    def quote(value: Path) -> str:
        return str(value).replace("'", "''")

    script = f"""
$ErrorActionPreference = "Stop"
. '{quote(ROOT / "scripts/mirrors.ps1")}'
function Get-ASMRDubberGitHubUrls {{ throw "network candidate resolution was reached" }}
$result = Invoke-ASMRDubberDownload `
    -Configuration ([pscustomobject]@{{}}) `
    -Url "https://invalid.example.test/complete.zip" `
    -Destination '{quote(destination)}' `
    -Sha256 "{expected}" `
    -Resume
if (-not $result.StartsWith("existing:")) {{ throw "existing file was not reused" }}
"""
    subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
    )

    assert destination.read_bytes() == b"already complete"
    assert not partial.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell behavior")
def test_windows_downloader_adapts_to_legacy_curl(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    assert powershell is not None
    legacy_curl = tmp_path / "legacy-curl.cmd"
    modern_curl = tmp_path / "modern-curl.cmd"
    legacy_curl.write_text(
        '@echo off\nif "%~1"=="--retry-all-errors" exit /b 2\nexit /b 0\n',
        encoding="ascii",
    )
    modern_curl.write_text("@exit /b 0\n", encoding="ascii")

    def quote(value: Path) -> str:
        return str(value).replace("'", "''")

    script = f"""
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
. '{quote(ROOT / "scripts/mirrors.ps1")}'
$legacy = @(Get-ASMRDubberCurlCommonArguments -CurlPath '{quote(legacy_curl)}')
$modern = @(Get-ASMRDubberCurlCommonArguments -CurlPath '{quote(modern_curl)}')
[pscustomobject]@{{ legacy = $legacy; modern = $modern }} |
    ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert "--retry-all-errors" not in result["legacy"]
    assert "--retry-all-errors" in result["modern"]
    assert "--retry" in result["legacy"]


def test_windows_setup_exposes_three_chinese_monotonic_profiles() -> None:
    setup = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")

    assert '"基础", "推荐", "进阶", "Core", "Recommended", "Advanced"' in setup
    assert '"Full"' not in setup
    profile_switch = setup.split("$Extra = switch ($Profile)", 1)[1]
    recommended = _between(profile_switch, '"推荐" {', '"进阶" {')
    advanced = _between(profile_switch, '"进阶" {', "\n}\n\n$RecommendedDependenciesReady")

    assert "$InstallParakeet = $true" in recommended
    assert "$InstallRecommendedTTS = -not $SkipRecommendedTTS" in recommended
    assert "asr-faster-whisper" not in recommended
    assert "asr-kotoba-whisper" not in recommended

    assert "$InstallAdvancedModels = $true" in advanced
    assert "$InstallRecommendedTTS = -not $SkipRecommendedTTS" in advanced
    assert "asr-faster-whisper" in advanced
    assert "asr-kotoba-whisper" in advanced
    assert "asr-forced-aligner" in advanced
    assert "asr-asmr-vad" in advanced

    cuda_block = setup.split("if ($InstallAdvancedModels -and $NvidiaSmi)", 1)[1]
    cuda_block = cuda_block.split('Write-Host "正在安装应用依赖', 1)[0]
    assert '"torch==2.11.0+cu130"' in cuda_block
    assert '"torchaudio==2.11.0+cu130"' in cuda_block


def test_windows_setup_reports_native_runtime_problems_without_stopping() -> None:
    setup = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")
    report = _between(
        setup,
        "function Write-ASMRDubberNativeRuntimeReport {",
        "\n}\n\nNew-Item",
    )

    for dll in (
        "MSVCP140.dll",
        "VCOMP140.DLL",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
    ):
        assert dll in report
    assert "仅报告，不会中止安装" in report
    assert "Setup 仍会继续" in report
    assert "throw" not in report.casefold()
    assert "Write-ASMRDubberNativeRuntimeReport" in setup


def test_linux_setup_exposes_same_three_chinese_profiles() -> None:
    setup = (ROOT / "scripts/linux/setup.sh").read_text(encoding="utf-8")

    assert "基础|推荐|进阶" in setup
    assert "Full)" not in setup
    profile_switch = setup.split("INSTALL_PARAKEET=0", 1)[1]
    recommended = _between(profile_switch, "  推荐)\n", "  进阶)\n")
    advanced = _between(profile_switch, "  进阶)\n", "esac")

    assert "INSTALL_PARAKEET=1" in recommended
    assert "INSTALL_RECOMMENDED_TTS=1" in recommended
    assert "asr-faster-whisper" not in recommended
    assert "asr-kotoba-whisper" not in recommended

    assert "INSTALL_ADVANCED_MODELS=1" in advanced
    assert "INSTALL_RECOMMENDED_TTS=1" in advanced
    assert "asr-faster-whisper" in advanced
    assert "asr-kotoba-whisper" in advanced
    assert "asr-forced-aligner" in advanced
    assert "asr-asmr-vad" in advanced


def test_every_setup_profile_installs_and_verifies_online_api_clients() -> None:
    windows = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8-sig")
    runtime_checks = (ROOT / "scripts/windows/recommended-dependencies.ps1").read_text(
        encoding="utf-8-sig"
    )
    dependency_builder = (
        ROOT / "scripts/windows/create-recommended-dependency-pack.ps1"
    ).read_text(encoding="utf-8-sig")
    linux = (ROOT / "scripts/linux/setup.sh").read_text(encoding="utf-8")
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    edge_version = next(
        package["version"] for package in lock["package"] if package["name"] == "edge-tts"
    )

    api_install = windows.index('"edge-tts==7.2.8"')
    application_install = windows.index('Write-Host "正在安装应用依赖')
    assert application_install < api_install
    assert f'"edge-tts=={edge_version}"' in windows
    assert '"httpx>=0.28.0"' in windows
    assert "Test-ASMRDubberApiClientRuntime" in runtime_checks
    assert "Test-ASMRDubberApplicationRuntime" in runtime_checks
    assert "import edge_tts, httpx" in runtime_checks
    assert '("edge-tts", "edge_tts")' in (ROOT / "src/asmr_dubber/cli.py").read_text(
        encoding="utf-8"
    )
    assert "$ApplicationDependenciesReady = Test-ASMRDubberApplicationRuntime" in windows
    assert "Test-ASMRDubberCoreRuntime -PortableRoot $DataRoot" in windows
    assert "edge_tts" in dependency_builder
    assert "httpx" in dependency_builder
    assert "import asmr_dubber.ui, av, edge_tts, gradio, httpx" in linux
    assert "国内软件源补齐应用依赖" in windows
    assert "国内软件源补齐应用依赖" in linux

    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "uv export --frozen --no-dev --extra ui" in release
    assert "vendor\\windows-core-wheelhouse" in release
    assert "--only-binary=:all:" in release
    assert "windows-core-wheelhouse-smoke" in release
    assert "--offline --find-links $coreWheelhouse" in release
    assert "import asmr_dubber.ui, edge_tts, gradio, httpx, setuptools" in release
    for required_wheel in ("edge_tts", "editables", "gradio", "httpx", "hatchling"):
        assert f'"{required_wheel}-*.whl"' in release


def test_windows_setup_installs_application_before_validating_api_clients() -> None:
    windows = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8-sig")
    application_install = windows.index(
        "$ApplicationDependenciesReady = Test-ASMRDubberApplicationRuntime"
    )
    api_validation = windows.index(
        "$ApiClientsReady = Test-ASMRDubberApiClientRuntime", application_install
    )
    final_validation = windows.index(
        "Test-ASMRDubberCoreRuntime -PortableRoot $DataRoot", api_validation
    )

    assert application_install < api_validation < final_validation
    assert "$BundledCoreWheelhouse" in windows[application_install:api_validation]


def test_model_packs_are_imported_before_profile_downloads() -> None:
    windows = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts/linux/setup.sh").read_text(encoding="utf-8")

    for setup, advanced_download in (
        (windows, 'download-models", "--backend", "进阶语音识别"'),
        (linux, "download-models --backend 进阶语音识别"),
    ):
        import_position = setup.index("import-model-packs")
        advanced_position = setup.index(advanced_download)
        assert import_position < advanced_position


def test_setup_imports_only_model_packs_belonging_to_the_selected_tier() -> None:
    windows = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts/linux/setup.sh").read_text(encoding="utf-8")

    windows_packs = windows.split("$LocalPackIds = switch ($Profile)", 1)[1]
    windows_recommended = _between(
        windows_packs,
        '"推荐" {',
        '"进阶" {',
    )
    windows_advanced = _between(windows_packs, '"进阶" {', "\n    }\n}")
    assert "parakeet-ja-windows" in windows_recommended
    assert "indextts2-checkpoints" in windows_recommended
    assert "kotoba-whisper-v2.2" not in windows_recommended
    assert "faster-whisper-large-v2" not in windows_recommended
    assert "indextts2-checkpoints" in windows_advanced
    assert "kotoba-whisper-v2.2" in windows_advanced
    assert "faster-whisper-large-v2" in windows_advanced
    assert "qwen3-forced-aligner" in windows_advanced
    assert "whisper-vad-asmr-onnx" in windows_advanced

    linux_packs = linux.split("PACK_ARGUMENTS=(import-model-packs --all)", 1)[1]
    linux_recommended = _between(linux_packs, "    推荐)\n", "    进阶)\n")
    linux_advanced = _between(linux_packs, "    进阶)\n", "  esac")
    assert "--pack-id parakeet-ja-linux" in linux_recommended
    assert "--pack-id indextts2-checkpoints" in linux_recommended
    assert "--pack-id kotoba-whisper-v2.2" not in linux_recommended
    assert "--pack-id faster-whisper-large-v2" not in linux_recommended
    assert "--pack-id indextts2-checkpoints" in linux_advanced
    assert "--pack-id kotoba-whisper-v2.2" in linux_advanced
    assert "--pack-id faster-whisper-large-v2" in linux_advanced
    assert "--pack-id qwen3-forced-aligner" in linux_advanced
    assert "--pack-id whisper-vad-asmr-onnx" in linux_advanced


def test_indextts_installers_reuse_complete_checkpoints_before_download() -> None:
    windows = (ROOT / "scripts/windows/install-indextts2.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts/linux/install-indextts2.sh").read_text(encoding="utf-8")

    for installer, download_marker, check_marker in (
        (windows, '"download", "--source", "modelscope"', '"check", "--model-dir"'),
        (linux, "download --source modelscope", 'check --model-dir "$MODEL_DIR"'),
    ):
        definition_position = installer.index(
            "from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES"
        )
        download_position = installer.index(download_marker)
        check_position = installer.index(check_marker)
        assert definition_position < download_position < check_position
        assert "本地 checkpoints 已完整，无需联网下载" in installer


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell 5.1 behavior")
def test_indextts_checkpoint_probe_works_in_windows_powershell_51(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe")
    assert powershell is not None
    installer = (ROOT / "scripts/windows/install-indextts2.ps1").read_text(encoding="utf-8")
    validation_script = _between(installer, "$ValidationScript = @'", "'@")
    assert "$ValidationResult = $ValidationScript | & $AppPython - $ModelDir" in installer
    assert "-c $ValidationScript" not in installer

    model_dir = tmp_path / "模型 checkpoints"
    model_dir.mkdir()

    def quote(value: str | Path) -> str:
        return str(value).replace("'", "''")

    def probe() -> str:
        command = f"""
$ErrorActionPreference = "Stop"
$ValidationScript = @'
{validation_script.strip()}
'@
$ValidationResult = $ValidationScript | & '{quote(sys.executable)}' - '{quote(model_dir)}'
if ($LASTEXITCODE -ne 0) {{ throw "checkpoint probe failed" }}
Write-Output $ValidationResult
"""
        encoded_command = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        result = subprocess.run(
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_command,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    assert probe() == "missing"

    for relative in INDEXTTS_REQUIRED_FILES:
        target = model_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for relative in INDEXTTS_REQUIRED_DIRS:
        (model_dir / relative).mkdir(parents=True, exist_ok=True)

    assert probe() == "ready"


def test_indextts_source_archive_prefers_modelscope_and_keeps_github_fallback() -> None:
    mirrors = json.loads((ROOT / "mirrors.json").read_text(encoding="utf-8"))
    sources = mirrors["indextts2_source_archives"]
    assert sources[0].startswith("https://modelscope.cn/")
    assert sources[-1].startswith("https://github.com/")
    assert "13495845e3028f0bb6ca1462ad22aa0e76349e40.zip" in sources[0]

    windows = (ROOT / "scripts/windows/install-indextts2.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts/linux/install-indextts2.sh").read_text(encoding="utf-8")
    assert '-Name "indextts2_source_archives"' in windows
    assert 'asmr_mirror_list "$ROOT" indextts2_source_archives' in linux


def test_windows_recommended_prefers_verified_modelscope_dependency_pack() -> None:
    mirrors = json.loads((ROOT / "mirrors.json").read_text(encoding="utf-8"))
    sources = mirrors["windows_recommended_dependency_archives"]
    assert sources == [
        "https://modelscope.cn/models/EveningStudyW/"
        "ASMR-Dubber-Windows-Recommended/resolve/master/"
        "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip"
    ]

    setup = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")
    helper = (ROOT / "scripts/windows/recommended-dependencies.ps1").read_text(encoding="utf-8")
    importer = (ROOT / "scripts/import_windows_dependency_pack.py").read_text(encoding="utf-8")
    assert setup.index("Import-ASMRDubberRecommendedDependencies") < setup.index(
        'Write-Host "正在安装应用依赖'
    )
    assert "windows_recommended_dependency_archives" in helper
    assert "Get-ASMRDubberFileSha256 -Path $Archive" in helper
    assert "RecommendedDependencyPackSize = 0" not in helper
    assert "__WINDOWS_RECOMMENDED_DEPENDENCY_PACK_SHA256__" not in helper
    assert "ALLOWED_PREFIXES" in importer
    assert "MAX_UNCOMPRESSED_BYTES" in importer


def test_windows_indextts_can_reuse_relocatable_preinstalled_environment() -> None:
    installer = (ROOT / "scripts/windows/install-indextts2.ps1").read_text(encoding="utf-8")
    assert "Test-ASMRDubberIndexRuntimeDependencies" in installer
    assert '"-m", "indextts.cli_v2", "check"' in installer
    assert '"-m", "indextts.cli_v2", "download"' in installer
    assert '$AllowedBootstrapDirectories = @("checkpoints", ".venv")' in installer
    assert "$_.PSIsContainer -and $_.Name -in $AllowedBootstrapDirectories" in installer


def test_windows_recommended_dependency_pack_builder_has_expected_components() -> None:
    builder = (ROOT / "scripts/windows/create-recommended-dependency-pack.ps1").read_text(
        encoding="utf-8"
    )
    for component in (
        "application-ui",
        "indextts2",
        "ffmpeg-shared",
        "cpython-3.11.13-windows-x86_64-none",
    ):
        assert component in builder


def test_windows_recommended_portable_builder_uses_verified_complete_payloads() -> None:
    builder = (ROOT / "scripts/windows/create-recommended-portable.ps1").read_text(encoding="utf-8")
    for component in (
        "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip",
        "ASMR-Dubber-ModelPack-parakeet-ja-windows-v0.2.1.zip",
        "ASMR-Dubber-ModelPack-indextts2-checkpoints-v0.2.1.zip",
        "doctor",
        "--no-network",
    ):
        assert component in builder
    assert "Get-ASMRDubberFileSha256 -Path $Pack.Path" in builder


def test_windows_setup_prompt_maps_all_profiles_and_shows_space() -> None:
    source = (ROOT / "launcher/windows/ASMRDubberSetup.cs").read_text(encoding="utf-8")

    assert 'if (input == "1") return "基础";' in source
    assert 'if (input == "" || input == "2") return "推荐";' in source
    assert 'if (input == "3") return "进阶";' in source
    assert 'input == "4"' not in source
    for estimate in ("约 2 GB", "约 24–28 GB", "约 33–39 GB"):
        assert estimate in source
    for model in (
        "Parakeet CTC 1.1B JA GAL",
        "Parakeet TDT/CTC 0.6B JA",
        "Kotoba-Whisper v2.2",
        "Faster-Whisper large-v2",
        "日语 ASMR 专用 Whisper VAD ONNX",
        "Qwen3 ForcedAligner 0.6B",
        "IndexTTS2 checkpoints",
    ):
        assert model in source


def test_windows_launcher_sources_match_release_version() -> None:
    for name in ("ASMRDubberLauncher.cs", "ASMRDubberSetup.cs"):
        source = (ROOT / "launcher/windows" / name).read_text(encoding="utf-8")
        assert 'AssemblyVersion("1.2.0.0")' in source
        assert 'AssemblyFileVersion("1.2.0.0")' in source


def test_windows_launcher_uses_path_scoped_mutex_dynamic_port_and_product_marker() -> None:
    source = (ROOT / "launcher/windows/ASMRDubberLauncher.cs").read_text(encoding="utf-8")

    assert "ProjectPathHash(root)" in source
    assert "FindAvailablePort(7860, 100)" in source
    assert 'ProductMarker = "asmr-dubber-product-marker"' in source
    assert "body.IndexOf(ProductMarker" in source


def test_advanced_dependency_pack_and_analysis_model_packs_are_reused() -> None:
    mirrors = json.loads((ROOT / "mirrors.json").read_text(encoding="utf-8"))

    assert mirrors["modelscope_artifacts"]["windows_advanced_dependency_archives"] == [
        "https://modelscope.cn/models/EveningStudyW/"
        "ASMR-Dubber-Windows-Advanced/resolve/master/"
        "ASMR-Dubber-Windows-Advanced-Dependencies-v1.0.0.zip"
    ]
    assert "qwen3-forced-aligner" in mirrors["model_pack_sources"]
    assert "whisper-vad-asmr-onnx" in mirrors["model_pack_sources"]

    setup = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8-sig")
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
