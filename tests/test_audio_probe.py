from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from asmr_dubber.audio import probe_audio


def test_probe_audio_replaces_invalid_optional_metadata(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio-placeholder")
    received: dict[str, object] = {}

    class FakeContainer:
        duration = None

        def __init__(self) -> None:
            context = SimpleNamespace(
                sample_rate=48_000,
                channels=2,
                layout=SimpleNamespace(name="stereo"),
                name="mp3float",
            )
            stream = SimpleNamespace(
                duration=480_000,
                time_base=Fraction(1, 48_000),
                rate=48_000,
                codec_context=context,
            )
            self.streams = SimpleNamespace(audio=[stream], video=[])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_open(path: str, **kwargs):
        received["path"] = path
        received.update(kwargs)
        return FakeContainer()

    monkeypatch.setattr("av.open", fake_open)

    info = probe_audio(source, sha256="a" * 64)

    assert received == {"path": str(source.resolve()), "metadata_errors": "replace"}
    assert info.duration_seconds == 10.0
    assert info.sample_rate == 48_000
    assert info.channels == 2
