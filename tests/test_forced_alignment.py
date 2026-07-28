from types import SimpleNamespace

import pytest

from asmr_dubber.forced_alignment import (
    _alignment_device_map,
    _alignment_dtype,
    _map_group_items_to_sentences,
)
from asmr_dubber.models import Sentence


class _Processor:
    @staticmethod
    def clean_token(value: str) -> str:
        return "".join(character for character in value if character.isalnum())


def test_aligner_uses_fp16_on_cuda_without_bf16_support() -> None:
    torch = SimpleNamespace(
        float16=object(),
        float32=object(),
        bfloat16=object(),
        cuda=SimpleNamespace(is_bf16_supported=lambda: False),
    )

    assert _alignment_dtype(torch, "cuda") is torch.float16
    assert _alignment_device_map("cuda") == "cuda:0"
    assert _alignment_device_map("cuda:1") == "cuda:1"


def test_aligner_keeps_bf16_on_supported_cuda_and_float32_on_cpu() -> None:
    torch = SimpleNamespace(
        float16=object(),
        float32=object(),
        bfloat16=object(),
        cuda=SimpleNamespace(is_bf16_supported=lambda: True),
    )

    assert _alignment_dtype(torch, "cuda") is torch.bfloat16
    assert _alignment_dtype(torch, "cpu") is torch.float32


def test_group_alignment_maps_by_characters_when_token_splits_differ() -> None:
    sentences = [
        Sentence(id="s000001", start_seconds=0, end_seconds=1, ja_text="今日は。"),
        Sentence(id="s000002", start_seconds=1, end_seconds=2, ja_text="晴れです！"),
    ]
    items = [
        SimpleNamespace(text="今日", start_time=0.1, end_time=0.3),
        SimpleNamespace(text="は", start_time=0.3, end_time=0.5),
        SimpleNamespace(text="晴れ", start_time=0.6, end_time=0.9),
        SimpleNamespace(text="です", start_time=0.9, end_time=1.2),
    ]

    mapped = _map_group_items_to_sentences(_Processor(), sentences, items)

    assert [[item.text for item in group] for group in mapped] == [
        ["今日", "は"],
        ["晴れ", "です"],
    ]


def test_group_alignment_rejects_changed_text() -> None:
    sentences = [
        Sentence(id="s000001", start_seconds=0, end_seconds=1, ja_text="今日は。"),
    ]
    items = [SimpleNamespace(text="明日", start_time=0.1, end_time=0.3)]

    with pytest.raises(ValueError, match="文字单元与台本不一致"):
        _map_group_items_to_sentences(_Processor(), sentences, items)
