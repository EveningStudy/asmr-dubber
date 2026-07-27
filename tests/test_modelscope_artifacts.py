from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_modelscope_contract_verifier_passes_without_network() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_modelscope_artifacts.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "contract OK" in completed.stdout


def test_every_setup_artifact_has_a_modelscope_first_source() -> None:
    mirrors = json.loads((ROOT / "mirrors.json").read_text(encoding="utf-8"))
    artifacts = mirrors["modelscope_artifacts"]
    required = {
        "uv_archives_windows",
        "uv_archives_linux",
        "python312_windows_archives",
        "python312_linux_archives",
        "python311_windows_archives",
        "python311_linux_archives",
        "windows_recommended_dependency_archives",
        "windows_advanced_dependency_archives",
        "windows_application_wheelhouse_archives",
        "windows_cuda_wheelhouse_archives",
        "indextts2_wheelhouse_archives_windows",
        "linux_application_wheelhouse_archives",
        "indextts2_wheelhouse_archives_linux",
        "indextts2_source_archives",
        "ffmpeg_shared_archives_windows",
        "crispasr_windows_cpu_archives",
        "crispasr_windows_cuda_archives",
        "crispasr_linux_cpu_archives",
        "crispasr_linux_cuda_archives",
        "parakeet_11b_model_files",
        "parakeet_06b_model_files",
    }

    assert required <= artifacts.keys()
    assert mirrors["download_policy"] == {
        "provider": "modelscope-first",
        "allow_external": False,
    }


def test_installers_never_unconditionally_fall_back_to_huggingface_or_github() -> None:
    sources = [
        ROOT / "scripts/windows/install-indextts2.ps1",
        ROOT / "scripts/windows/install-parakeet.ps1",
        ROOT / "scripts/linux/install-indextts2.sh",
        ROOT / "scripts/linux/install-parakeet.sh",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    assert "ModelScope 不可用，改用 Hugging Face" not in combined
    assert "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS" in (ROOT / "scripts/mirrors.ps1").read_text(
        encoding="utf-8"
    )
    assert "asmr_external_downloads_allowed" in combined


def test_empty_huggingface_policy_is_safe_for_launchers_and_network_probe() -> None:
    windows_consumers = [
        ROOT / "scripts/windows/setup.ps1",
        ROOT / "scripts/windows/run-cli.ps1",
        ROOT / "scripts/windows/install-indextts2.ps1",
    ]
    for path in windows_consumers:
        source = path.read_text(encoding="utf-8-sig")
        assert "Set-ASMRDubberHuggingFaceEnvironment" in source
        assert "$HuggingFaceEndpoints[0]" not in source

    linux_mirrors = (ROOT / "scripts/mirrors.sh").read_text(encoding="utf-8")
    network_bridge = (ROOT / "scripts/network-bridge.sh").read_text(encoding="utf-8")
    assert "unset HF_ENDPOINT" in linux_mirrors
    assert "https://modelscope.cn/robots.txt" in network_bridge
    assert "https://huggingface.co/robots.txt" not in network_bridge


def test_dependency_pack_builder_excludes_setuptools_test_fixtures() -> None:
    builder = (ROOT / "scripts/windows/create-recommended-dependency-pack.ps1").read_text(
        encoding="utf-8"
    )

    assert "Lib\\site-packages\\pkg_resources\\tests" in builder
