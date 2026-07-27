import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import asr
from asmr_dubber.asr import _clock
from asmr_dubber.errors import AsmrDubberError
from asmr_dubber.models import ProjectSettings


def test_clock_formats_long_audio_positions() -> None:
    assert _clock(89.9) == "1:29"
    assert _clock(3_661.2) == "1:01:01"


def test_faster_whisper_uses_explicit_batched_pipeline_without_enabling_vad(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    calls: list[dict[str, object]] = []

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("sequential path should not be used")

    class FakeBatchedPipeline:
        def __init__(self, *, model):
            assert isinstance(model, FakeWhisperModel)

        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            word = SimpleNamespace(word="声", start=0.1, end=0.4)
            segment = SimpleNamespace(text="声", words=[word], start=0.1, end=0.4)
            return [segment], SimpleNamespace(language="ja")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            synchronize=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(
            WhisperModel=FakeWhisperModel,
            BatchedInferencePipeline=FakeBatchedPipeline,
        ),
    )

    sentences, _ = asr._transcribe_faster_whisper(
        audio_path,
        ProjectSettings(
            asr_backend="faster_whisper",
            asr_batch_size=4,
            asr_vad_filter=False,
        ),
        None,
    )

    assert len(sentences) == 1
    assert calls[0]["batch_size"] == 4
    assert calls[0]["vad_filter"] is False
    assert calls[0]["without_timestamps"] is False


def test_kotoba_faster_avoids_invalid_teacher_alignment_heads(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    calls: list[dict[str, object]] = []

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            segment = SimpleNamespace(text="声", words=None, start=0.1, end=0.4)
            return [segment], SimpleNamespace(language="ja")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            synchronize=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(
            WhisperModel=FakeWhisperModel,
            BatchedInferencePipeline=object,
        ),
    )
    monkeypatch.setattr(asr, "resolve_model_source", lambda model_id: model_id)

    sentences, _ = asr._transcribe_faster_whisper(
        audio_path,
        ProjectSettings(
            asr_backend="faster_whisper",
            asr_model="kotoba-tech/kotoba-whisper-v2.0-faster",
            asr_batch_size=1,
            asr_condition_on_previous_text=True,
            asr_kotoba_chunk_seconds=15,
        ),
        None,
    )

    assert len(sentences) == 1
    assert calls[0]["word_timestamps"] is False
    assert calls[0]["condition_on_previous_text"] is False
    assert calls[0]["chunk_length"] == 15


def test_kotoba_transformers_refuses_incomplete_model_without_downloading(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    monkeypatch.setattr(asr, "cached_model_path", lambda _model_id: None)

    with pytest.raises(
        AsmrDubberError,
        match=r"模型尚未完整下载.*不会在后台自动下载",
    ):
        asr._transcribe_kotoba_whisper(
            audio_path,
            ProjectSettings(
                asr_backend="kotoba_whisper",
                asr_model="kotoba-tech/kotoba-whisper-v2.1",
                asr_device="cpu",
            ),
            None,
        )


def test_transformers_asr_pipeline_skips_optional_torchcodec() -> None:
    transformers = pytest.importorskip("transformers")
    asr_pipeline = transformers.pipelines.automatic_speech_recognition

    original = asr_pipeline.is_torchcodec_available

    def fake_pipe(inputs, **kwargs):
        assert asr_pipeline.is_torchcodec_available() is False
        return {"inputs": inputs, "kwargs": kwargs}

    result = asr._run_transformers_asr_pipeline(
        fake_pipe,
        {"array": np.zeros(16, dtype=np.float32), "sampling_rate": 16_000},
        return_timestamps=True,
    )

    assert result["kwargs"]["return_timestamps"] is True
    assert asr_pipeline.is_torchcodec_available is original


def test_parakeet_ctc_zero_width_words_fall_back_to_complete_segment() -> None:
    payload = {
        "crispasr": {"language": "ja"},
        "transcription": [
            {
                "text": "えっうそでしょ?",
                "offsets": {"from": 0, "to": 3680},
                "words": [
                    {"text": "えっ", "offsets": {"from": 0, "to": 0}},
                    {"text": "うそ", "offsets": {"from": 2240, "to": 2880}},
                    {"text": "でしょ?", "offsets": {"from": 2880, "to": 3680}},
                ],
            }
        ],
    }

    sentences, language = asr._parse_crispasr_payload(payload, ProjectSettings())

    assert language == "ja"
    assert "".join(sentence.ja_text for sentence in sentences) == "えっうそでしょ?"
    assert sentences[0].start_seconds == 0
    assert sentences[-1].end_seconds == 3.68


def test_parakeet_ctc_11_uses_token_timestamps_for_sentence_splitting() -> None:
    payload = {
        "crispasr": {"language": "ja", "backend": "fastconformer-ctc"},
        "transcription": [
            {
                "text": "今日は大丈夫?もう寝ます。",
                "offsets": {"from": 0, "to": 4300},
                "tokens": [
                    {"text": "今日", "offsets": {"from": 100, "to": 700}},
                    {"text": "は", "offsets": {"from": 700, "to": 900}},
                    {"text": "大丈夫", "offsets": {"from": 1000, "to": 1900}},
                    {"text": "?", "offsets": {"from": 1900, "to": 2000}},
                    {"text": "もう", "offsets": {"from": 2900, "to": 3400}},
                    {"text": "寝ます", "offsets": {"from": 3500, "to": 4200}},
                    {"text": "。", "offsets": {"from": 4200, "to": 4300}},
                    {"text": "", "offsets": {"from": 4300, "to": 4400}},
                ],
            }
        ],
    }

    sentences, language = asr._parse_crispasr_payload(payload, ProjectSettings())

    assert language == "ja"
    assert [sentence.ja_text for sentence in sentences] == [
        "今日は大丈夫?",
        "もう寝ます。",
    ]
    assert sentences[0].end_seconds == 1.9
    assert sentences[1].start_seconds == 2.9


def test_parakeet_stages_non_ascii_windows_input_at_ascii_path(tmp_path, monkeypatch) -> None:
    source = tmp_path / "日本語💫" / "analysis.wav"
    source.parent.mkdir()
    source.write_bytes(b"audio")
    run_directory = tmp_path / "portable" / "temp" / "asr" / "run-1"
    run_directory.mkdir(parents=True)
    monkeypatch.setattr(
        asr,
        "current_platform",
        lambda: SimpleNamespace(is_windows=True),
    )

    staged = asr._parakeet_input_path(source, run_directory)

    assert staged == run_directory / "input.wav"
    assert str(staged).isascii()
    assert staged.read_bytes() == b"audio"
    assert source.read_bytes() == b"audio"


def test_parakeet_command_pins_backend_cache_and_supported_vad(tmp_path, monkeypatch) -> None:
    portable = tmp_path / "portable"
    executable = portable / "runtimes" / "crispasr" / "bin" / "crispasr"
    model = portable / "models" / "parakeet" / "parakeet-ctc-1.1b-ja-f16.gguf"
    executable.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    executable.touch()
    model.touch()
    audio = tmp_path / "analysis.wav"
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)
    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            calls.append(command)
            output_base = command[command.index("-of") + 1]
            payload = {
                "crispasr": {"language": "ja"},
                "transcription": [
                    {
                        "text": "声。",
                        "offsets": {"from": 100, "to": 500},
                    }
                ],
            }
            Path(output_base).with_suffix(".json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            self.stdout = iter(())
            self.returncode = None

        def wait(self):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(asr, "portable_home", lambda: portable)
    monkeypatch.setattr(
        asr,
        "current_platform",
        lambda: SimpleNamespace(is_windows=False),
    )
    monkeypatch.setattr(asr.subprocess, "Popen", FakeProcess)

    sentences, _ = asr._transcribe_parakeet(
        audio,
        ProjectSettings(
            asr_vad_filter=True,
            asr_vad_min_silence_ms=650,
        ),
        None,
    )

    command = calls[0]
    assert command[command.index("--backend") + 1] == "parakeet"
    assert command[command.index("--cache-dir") + 1] == str(portable / "cache" / "crispasr")
    assert command[command.index("-vm") + 1] == "silero"
    assert "--chunk-seconds" not in command
    assert "whisper-vad" not in command
    assert sentences[0].ja_text == "声。"
    assert not list((portable / "temp" / "asr").glob("parakeet-*"))


@pytest.mark.parametrize(
    ("model_id", "model_filename"),
    [
        (
            "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo",
            "parakeet-ctc-1.1b-ja-f16.gguf",
        ),
        (
            "nvidia/parakeet-tdt_ctc-0.6b-ja",
            "parakeet-tdt-0.6b-ja.gguf",
        ),
    ],
)
def test_parakeet_long_audio_is_split_before_both_backends(
    tmp_path,
    monkeypatch,
    model_id,
    model_filename,
) -> None:
    portable = tmp_path / "portable"
    executable = portable / "runtimes" / "crispasr" / "bin" / "crispasr"
    model = portable / "models" / "parakeet" / model_filename
    executable.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    executable.touch()
    model.touch()
    audio = tmp_path / "long.wav"
    sf.write(audio, np.zeros(66_000, dtype=np.float32), 1_000)
    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            calls.append(command)
            output_base = command[command.index("-of") + 1]
            if model_id == "nvidia/parakeet-tdt_ctc-0.6b-ja" and len(calls) == 2:
                self.stdout = iter(())
                self.returncode = None
                return
            payload = {
                "crispasr": {"language": "ja"},
                "transcription": [
                    {
                        "text": "声。",
                        "words": [
                            {
                                "text": "声",
                                "offsets": {"from": 100, "to": 500},
                            }
                        ],
                    }
                ],
            }
            Path(output_base).with_suffix(".json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            self.stdout = iter(())
            self.returncode = None

        def wait(self):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(asr, "portable_home", lambda: portable)
    monkeypatch.setattr(
        asr,
        "current_platform",
        lambda: SimpleNamespace(is_windows=False),
    )
    monkeypatch.setattr(asr.subprocess, "Popen", FakeProcess)

    sentences, _ = asr._transcribe_parakeet(
        audio,
        ProjectSettings(asr_model=model_id, asr_chunk_seconds=30),
        None,
    )

    assert len(calls) == 3
    assert all("--chunk-seconds" not in command for command in calls)
    assert all(
        Path(command[command.index("-f") + 1]).name.startswith("chunk-") for command in calls
    )
    expected_sentences = 2 if model_id == "nvidia/parakeet-tdt_ctc-0.6b-ja" else 3
    assert len(sentences) == expected_sentences
    assert sentences[1].start_seconds > (45 if expected_sentences == 2 else 20)
    assert not list((portable / "temp" / "asr").glob("parakeet-*"))


def test_parakeet_error_keeps_leading_argument_diagnostic(tmp_path, monkeypatch) -> None:
    portable = tmp_path / "portable"
    executable = portable / "runtimes" / "crispasr" / "bin" / "crispasr"
    model = portable / "models" / "parakeet" / "parakeet-ctc-1.1b-ja-f16.gguf"
    executable.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    executable.touch()
    model.touch()
    audio = tmp_path / "analysis.wav"
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            self.stdout = iter(["argument error: invalid option\n", *(["help line\n"] * 80)])
            self.returncode = None

        def wait(self):
            self.returncode = 2
            return 2

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(asr, "portable_home", lambda: portable)
    monkeypatch.setattr(
        asr,
        "current_platform",
        lambda: SimpleNamespace(is_windows=False),
    )
    monkeypatch.setattr(asr.subprocess, "Popen", FakeProcess)

    with pytest.raises(AsmrDubberError, match="argument error: invalid option"):
        asr._transcribe_parakeet(audio, ProjectSettings(), None)
