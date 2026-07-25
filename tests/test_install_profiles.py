from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_windows_setup_exposes_four_monotonic_profiles() -> None:
    setup = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")

    assert '[ValidateSet("Core", "Recommended", "Advanced", "Full")]' in setup
    profile_switch = setup.split("$Extra = switch ($Profile)", 1)[1]
    recommended = _between(profile_switch, '"Recommended" {', '"Advanced" {')
    advanced = _between(profile_switch, '"Advanced" {', '"Full" {')
    full = _between(profile_switch, '"Full" {', "\n}\n\nif ($InstallDefaultModels")

    assert "$InstallParakeet = $true" in recommended
    assert "$InstallRecommendedTTS = -not $SkipRecommendedTTS" in recommended
    assert "asr-faster-whisper" not in recommended
    assert "asr-kotoba-whisper" not in recommended

    assert "$InstallAdvancedModels = $true" in advanced
    assert "$InstallRecommendedTTS = -not $SkipRecommendedTTS" in advanced
    assert "asr-faster-whisper" in advanced
    assert "asr-kotoba-whisper" in advanced

    assert "$InstallAdvancedModels = $true" in full
    assert "$InstallDefaultModels = $true" in full
    assert "local-default" in full
    assert "asr-openai-whisper" in full
    assert "asr-funasr" in full

    cuda_block = setup.split("if ($InstallAdvancedModels -and $NvidiaSmi)", 1)[1]
    cuda_block = cuda_block.split('Write-Host "正在安装应用依赖', 1)[0]
    assert '"torch==2.11.0+cu130"' in cuda_block
    assert '"torchaudio==2.11.0+cu130"' in cuda_block


def test_linux_setup_exposes_same_four_profiles() -> None:
    setup = (ROOT / "scripts/linux/setup.sh").read_text(encoding="utf-8")

    assert "Core|Recommended|Advanced|Full" in setup
    profile_switch = setup.split("INSTALL_PARAKEET=0", 1)[1]
    recommended = _between(profile_switch, "  Recommended)\n", "  Advanced)\n")
    advanced = _between(profile_switch, "  Advanced)\n", "  Full)\n")
    full = _between(profile_switch, "  Full)\n", "esac")

    assert "INSTALL_PARAKEET=1" in recommended
    assert "INSTALL_RECOMMENDED_TTS=1" in recommended
    assert "asr-faster-whisper" not in recommended
    assert "asr-kotoba-whisper" not in recommended

    assert "INSTALL_ADVANCED_MODELS=1" in advanced
    assert "INSTALL_RECOMMENDED_TTS=1" in advanced
    assert "asr-faster-whisper" in advanced
    assert "asr-kotoba-whisper" in advanced

    assert "INSTALL_ADVANCED_MODELS=1" in full
    assert "INSTALL_DEFAULT_MODELS=1" in full
    assert "local-default" in full
    assert "asr-openai-whisper" in full
    assert "asr-funasr" in full


def test_model_packs_are_imported_before_profile_downloads() -> None:
    windows = (ROOT / "scripts/windows/setup.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts/linux/setup.sh").read_text(encoding="utf-8")

    for setup, advanced_download in (
        (windows, 'download-models", "--backend", "advanced-asr"'),
        (linux, "download-models --backend advanced-asr"),
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
        '"Recommended" {',
        '"Advanced" {',
    )
    windows_advanced = _between(windows_packs, '"Advanced" {', '"Full" {')
    assert "parakeet-ja-windows" in windows_recommended
    assert "indextts2-checkpoints" in windows_recommended
    assert "kotoba-whisper-v2.2" not in windows_recommended
    assert "faster-whisper-large-v2" not in windows_recommended
    assert "indextts2-checkpoints" in windows_advanced
    assert "kotoba-whisper-v2.2" in windows_advanced
    assert "faster-whisper-large-v2" in windows_advanced

    linux_packs = linux.split("PACK_ARGUMENTS=(import-model-packs --all)", 1)[1]
    linux_recommended = _between(linux_packs, "    Recommended)\n", "    Advanced)\n")
    linux_advanced = _between(linux_packs, "    Advanced)\n", "  esac")
    assert "--pack-id parakeet-ja-linux" in linux_recommended
    assert "--pack-id indextts2-checkpoints" in linux_recommended
    assert "--pack-id kotoba-whisper-v2.2" not in linux_recommended
    assert "--pack-id faster-whisper-large-v2" not in linux_recommended
    assert "--pack-id indextts2-checkpoints" in linux_advanced
    assert "--pack-id kotoba-whisper-v2.2" in linux_advanced
    assert "--pack-id faster-whisper-large-v2" in linux_advanced


def test_indextts_installers_reuse_complete_checkpoints_before_download() -> None:
    windows = (ROOT / "scripts/windows/install-indextts2.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts/linux/install-indextts2.sh").read_text(encoding="utf-8")

    for installer, download_marker, check_marker in (
        (windows, '@("download", "--source", "modelscope"', '@("check", "--model-dir"'),
        (linux, "download --source modelscope", 'check --model-dir "$MODEL_DIR"'),
    ):
        definition_position = installer.index(
            "from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES"
        )
        download_position = installer.index(download_marker)
        check_position = installer.index(check_marker)
        assert definition_position < download_position < check_position
        assert "本地 checkpoints 已完整，无需联网下载" in installer


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


def test_windows_setup_prompt_maps_all_profiles_and_shows_space() -> None:
    source = (ROOT / "launcher/windows/ASMRDubberSetup.cs").read_text(encoding="utf-8")

    assert 'if (input == "1") return "Core";' in source
    assert 'if (input == "" || input == "2") return "Recommended";' in source
    assert 'if (input == "3") return "Advanced";' in source
    assert 'if (input == "4") return "Full";' in source
    for estimate in ("约 2 GB", "约 24–28 GB", "约 30–35 GB", "约 42–48 GB"):
        assert estimate in source


def test_windows_launcher_sources_match_release_version() -> None:
    for name in ("ASMRDubberLauncher.cs", "ASMRDubberSetup.cs"):
        source = (ROOT / "launcher/windows" / name).read_text(encoding="utf-8")
        assert 'AssemblyVersion("0.3.2.0")' in source
        assert 'AssemblyFileVersion("0.3.2.0")' in source
