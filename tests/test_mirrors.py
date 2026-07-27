import json
import shutil
import subprocess
from pathlib import Path

import pytest

from asmr_dubber.mirrors import (
    MIRROR_CONFIG_PATH,
    download_candidates,
    external_downloads_allowed,
    github_url_candidates,
    hf_hub_download_with_fallback,
    mirror_candidates,
    modelscope_artifact_urls,
)


def test_mirror_candidates_keep_order_and_official_fallback(tmp_path: Path) -> None:
    config = tmp_path / "mirrors.json"
    config.write_text(
        json.dumps(
            {
                "huggingface_endpoints": [
                    "https://mirror.example.test/",
                    "http://unsafe.example.test",
                ]
            }
        ),
        encoding="utf-8",
    )

    assert mirror_candidates("huggingface_endpoints", path=config) == (
        "https://mirror.example.test",
        "https://huggingface.co",
    )


def test_invalid_mirror_config_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "mirrors.json"
    config.write_text('{"pypi_indexes": "not-an-array"}', encoding="utf-8")

    with pytest.raises(ValueError, match="必须是数组"):
        mirror_candidates("pypi_indexes", path=config)


def test_github_candidates_end_with_direct_official_url() -> None:
    url = "https://github.com/example/project/releases/download/v1/file.zip"
    candidates = github_url_candidates(url)

    assert candidates[-1] == url
    assert candidates[0].endswith("/" + url)


def test_huggingface_download_automatically_uses_next_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", "1")
    calls: list[str] = []

    def fake_download(*, endpoint: str, **_kwargs) -> str:
        calls.append(endpoint)
        if endpoint == "https://hf-mirror.com":
            raise OSError("mirror unavailable")
        return "cached-file"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    assert hf_hub_download_with_fallback(repo_id="example/model", filename="config.json") == (
        "cached-file"
    )
    assert calls == ["https://hf-mirror.com", "https://huggingface.co"]


def test_release_download_policy_is_modelscope_first_and_blocks_external(monkeypatch) -> None:
    monkeypatch.delenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", raising=False)

    assert external_downloads_allowed(MIRROR_CONFIG_PATH) is False
    candidates = download_candidates("uv_archives_windows")
    assert candidates
    assert candidates[0].startswith("https://modelscope.cn/")
    assert all("github.com" not in candidate for candidate in candidates)
    assert download_candidates("huggingface_endpoints") == ()


def test_external_downloads_require_explicit_environment_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", "1")

    candidates = download_candidates("uv_archives_windows")
    assert candidates[0].startswith("https://modelscope.cn/")
    assert any("github.com" in candidate for candidate in candidates)


def test_preferred_external_download_requires_explicit_opt_in(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "mirrors.json"
    config.write_text(
        json.dumps(
            {
                "download_policy": {"allow_external": False},
                "pypi_indexes": ["https://pypi.tuna.tsinghua.edu.cn/simple"],
            }
        ),
        encoding="utf-8",
    )
    preferred = "https://pypi.org/simple"

    monkeypatch.delenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", raising=False)
    assert preferred not in download_candidates("pypi_indexes", preferred=preferred, path=config)

    monkeypatch.setenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS", "1")
    assert download_candidates("pypi_indexes", preferred=preferred, path=config)[0] == preferred


def test_powershell_preferred_external_download_obeys_policy() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    script = MIRROR_CONFIG_PATH.parent / "scripts" / "mirrors.ps1"
    quoted_script = str(script).replace("'", "''")
    command = f"""
    $ErrorActionPreference = 'Stop'
    $WarningPreference = 'SilentlyContinue'
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
    . '{quoted_script}'
    $configuration = [pscustomobject]@{{
        download_policy = [pscustomobject]@{{ allow_external = $false }}
        pypi_indexes = @('https://pypi.tuna.tsinghua.edu.cn/simple')
    }}
    Remove-Item Env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS -ErrorAction SilentlyContinue
    $blocked = @(Get-ASMRDubberMirrorList -Configuration $configuration `
        -Name 'pypi_indexes' -Preferred 'https://pypi.org/simple')
    $env:HF_ENDPOINT = 'https://huggingface.co'
    $blockedHf = @(Set-ASMRDubberHuggingFaceEnvironment `
        -Configuration $configuration)
    $blockedHfEnvironment = $env:HF_ENDPOINT
    $env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS = '1'
    $allowed = @(Get-ASMRDubberMirrorList -Configuration $configuration `
        -Name 'pypi_indexes' -Preferred 'https://pypi.org/simple')
    $allowedHf = @(Set-ASMRDubberHuggingFaceEnvironment `
        -Configuration $configuration -Preferred 'https://huggingface.co')
    [pscustomobject]@{{
        blocked = $blocked
        allowed = $allowed
        blocked_hf = $blockedHf
        blocked_hf_environment = $blockedHfEnvironment
        allowed_hf = $allowedHf
        allowed_hf_environment = $env:HF_ENDPOINT
    }} |
        ConvertTo-Json -Compress -Depth 4
    """
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)

    assert "https://pypi.org/simple" not in result["blocked"]
    assert result["allowed"][0] == "https://pypi.org/simple"
    assert result["blocked_hf"] == []
    assert result["blocked_hf_environment"] is None
    assert result["allowed_hf"][0] == "https://huggingface.co"
    assert result["allowed_hf_environment"] == "https://huggingface.co"


def test_modelscope_artifact_aliases_expand_relative_paths() -> None:
    urls = modelscope_artifact_urls("python312_windows_archives")

    assert len(urls) == 1
    assert urls[0].startswith(
        "https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Portable-Mirror/resolve/master/"
    )
    assert "cpython-3.12.13%2B20260718" in urls[0]
