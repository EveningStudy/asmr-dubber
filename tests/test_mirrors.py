import json
from pathlib import Path

import pytest

from asmr_dubber.mirrors import (
    github_url_candidates,
    hf_hub_download_with_fallback,
    mirror_candidates,
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
