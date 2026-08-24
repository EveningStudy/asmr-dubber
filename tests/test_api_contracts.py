from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest

from asmr_dubber.api_contracts import (
    APIContractError,
    endpoint_url,
    merge_extra_body,
    write_audio_response,
)


def test_endpoint_url_and_extra_body_are_strict() -> None:
    assert endpoint_url("http://127.0.0.1:8000/v1", "/audio/speech") == (
        "http://127.0.0.1:8000/v1/audio/speech"
    )
    assert (
        merge_extra_body(
            {"model": "m", "input": "x"},
            '{"reasoning_effort":"none"}',
            label="参数",
            reserved={"model", "input"},
        )["reasoning_effort"]
        == "none"
    )
    with pytest.raises(APIContractError, match="不能覆盖"):
        merge_extra_body({"model": "m"}, '{"model":"other"}', label="参数", reserved={"model"})


def test_write_audio_response_accepts_binary_and_base64_json(tmp_path: Path) -> None:
    binary = b"RIFF-test-audio"
    out_binary = tmp_path / "binary.wav"
    write_audio_response(
        httpx.Response(200, content=binary, headers={"content-type": "audio/wav"}),
        out_binary,
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
    )
    assert out_binary.read_bytes() == binary

    out_json = tmp_path / "json.wav"
    response = httpx.Response(
        200,
        json={"audio_base64": base64.b64encode(binary).decode("ascii")},
    )
    write_audio_response(
        response,
        out_json,
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
    )
    assert out_json.read_bytes() == binary
