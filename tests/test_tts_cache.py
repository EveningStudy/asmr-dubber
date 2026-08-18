import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.errors import OperationCancelledError, SynthesisError
from asmr_dubber.languages import SourceLanguage
from asmr_dubber.models import AudioInfo, DubProject, Sentence
from asmr_dubber.task_control import CancellationToken
from asmr_dubber.tts import shared_reference_sentence, tts_cache_key
from asmr_dubber.tts_backends import (
    _cosyvoice_runner,
    _edge_tts_runner,
    _fish_runner,
    _gpt_sovits_runner,
    _indextts_command,
    _load_indextts,
    _mimo_runner,
    _minimax_runner,
    _synthesize_indextts_cli_batch,
    synthesize_with_selected_backend,
)
from asmr_dubber.voice_reference import (
    VoiceReference,
    prepare_index_emotion_reference,
    prepare_index_speaker_reference,
)


def _project() -> DubProject:
    return DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="a" * 64,
            duration_seconds=12.0,
            sample_rate=48_000,
            channels=2,
        ),
        sentences=[
            Sentence(
                id="s000001",
                start_seconds=0.2,
                end_seconds=1.8,
                ja_text="始めましょう。",
                zh_text="让我们开始吧。",
            ),
            Sentence(
                id="s000002",
                start_seconds=2.0,
                end_seconds=8.0,
                ja_text="これは十分に長くて明瞭な音色の参考文章です。",
                zh_text="这是一句足够长而清晰的音色参考。",
            ),
        ],
    )


def test_tts_cache_tracks_synthesis_inputs_but_not_mix_only_settings() -> None:
    project = _project()
    sentence = project.sentences[0]
    original = tts_cache_key(project, sentence)

    project.settings.chinese_dubbing_offset_ms = -200
    project.settings.chinese_max_auto_speed = 1.35
    project.settings.chinese_dubbing_timing_mode = "sequential"
    project.settings.chinese_gain_db = -3.0
    project.settings.chinese_target_active_rms_dbfs = -34.0
    assert tts_cache_key(project, sentence) == original

    project.settings.tts_speed = 1.1
    assert tts_cache_key(project, sentence) != original
    project.settings.tts_speed = 1.0
    project.settings.tts_voice = "another-voice"
    assert tts_cache_key(project, sentence) != original
    project.settings.tts_voice = ""
    sentence.zh_text = "现在开始吧。"
    assert tts_cache_key(project, sentence) != original


def test_shared_reference_is_deterministic_and_affects_every_tts_key() -> None:
    project = _project()
    target, anchor = project.sentences
    project.settings.tts_backend = "gpt_sovits"
    project.settings.tts_model = "GPT-SoVITS-v4"

    assert shared_reference_sentence(project).id == anchor.id
    original = tts_cache_key(project, target)
    target.start_seconds = 0.4
    target.end_seconds = 2.0
    target.ja_text = "开始吧。"
    assert tts_cache_key(project, target) == original

    anchor.end_seconds = 8.5
    assert tts_cache_key(project, target) != original


def test_shared_reference_can_be_selected_for_chinese_only_script() -> None:
    project = _project()
    for sentence in project.sentences:
        sentence.ja_text = ""

    assert shared_reference_sentence(project).id == "s000002"


def test_shared_reference_uses_longest_japanese_clip() -> None:
    project = _project()
    project.sentences.append(
        Sentence(
            id="s000003",
            start_seconds=9.0,
            end_seconds=19.0,
            ja_text="これは別の参考です。",
            zh_text="这是另一段参考。",
        )
    )

    assert shared_reference_sentence(project).id == "s000003"


def test_shared_reference_prefers_japanese_before_chinese_fallback() -> None:
    project = _project()
    project.sentences.append(
        Sentence(
            id="s000003",
            start_seconds=9.0,
            end_seconds=16.0,
            ja_text="",
            zh_text="这段中文音频更长。",
        )
    )

    assert shared_reference_sentence(project).id == "s000002"


def test_index_default_uses_shared_speaker_and_current_sentence_emotion(
    tmp_path: Path,
) -> None:
    project = _project()
    target, anchor = project.sentences
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 12, dtype=np.float32), 16_000, subtype="FLOAT")

    speaker = prepare_index_speaker_reference(project, tmp_path, source, target)
    emotion = prepare_index_emotion_reference(project, tmp_path, source, target, speaker)

    assert speaker.sentence is anchor
    assert emotion is not None
    assert emotion.sentence is target
    assert emotion.path != speaker.path


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [
        (0.01, 0.06),
        (11.94, 11.99),
    ],
)
def test_index_short_sentence_reference_is_expanded_inside_source(
    tmp_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    project = _project()
    target = project.sentences[0]
    target.start_seconds = start_seconds
    target.end_seconds = end_seconds
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 12, dtype=np.float32), 16_000, subtype="FLOAT")

    speaker = prepare_index_speaker_reference(project, tmp_path, source, target)
    emotion = prepare_index_emotion_reference(project, tmp_path, source, target, speaker)

    assert emotion is not None
    assert sf.info(emotion.path).duration == pytest.approx(1.0, abs=0.01)


def test_index_short_cached_reference_is_rebuilt(tmp_path: Path) -> None:
    project = _project()
    target = project.sentences[0]
    target.start_seconds = 0.01
    target.end_seconds = 0.06
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 12, dtype=np.float32), 16_000, subtype="FLOAT")

    speaker = prepare_index_speaker_reference(project, tmp_path, source, target)
    first = prepare_index_emotion_reference(project, tmp_path, source, target, speaker)
    assert first is not None
    sf.write(first.path, np.zeros(800, dtype=np.float32), 16_000, subtype="FLOAT")

    rebuilt = prepare_index_emotion_reference(project, tmp_path, source, target, speaker)

    assert rebuilt is not None
    assert rebuilt.path == first.path
    assert sf.info(rebuilt.path).duration == pytest.approx(1.0, abs=0.01)


def test_index_external_reference_rejects_unsafe_short_audio(tmp_path: Path) -> None:
    project = _project()
    external = tmp_path / "short.wav"
    sf.write(external, np.zeros(800, dtype=np.float32), 16_000, subtype="FLOAT")
    project.settings.tts_index_speaker_source = "external"
    project.settings.tts_external_reference_audio = str(external)

    with pytest.raises(SynthesisError, match="至少 1 秒"):
        prepare_index_speaker_reference(
            project,
            tmp_path,
            tmp_path / "source.wav",
            project.sentences[0],
        )


def test_index_cache_tracks_independent_emotion_reference() -> None:
    project = _project()
    sentence = project.sentences[0]
    project.settings.tts_reference_sentence_id = project.sentences[1].id
    original = tts_cache_key(project, sentence)

    sentence.start_seconds = 0.3
    assert tts_cache_key(project, sentence) != original


def test_indextts_direct_runner_passes_separate_emotion_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project()
    project.settings.tts_model_path = str(tmp_path / "checkpoints")
    project.settings.tts_config_path = str(tmp_path / "checkpoints" / "config.yaml")
    Path(project.settings.tts_model_path).mkdir()
    Path(project.settings.tts_config_path).touch()
    calls: list[dict[str, object]] = []

    class FakeIndexTTS2:
        def __init__(self, **_kwargs):
            pass

        def infer(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "indextts", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "indextts.infer_v2",
        SimpleNamespace(IndexTTS2=FakeIndexTTS2),
    )
    speaker = tmp_path / "speaker.wav"
    emotion = tmp_path / "emotion.wav"
    reference = VoiceReference(
        speaker,
        "音色",
        "speaker",
        emotion_path=emotion,
        emotion_identity="emotion",
    )

    run, cleanup = _load_indextts(project)
    try:
        run(project.sentences[0], reference, tmp_path / "output.wav")
    finally:
        cleanup()

    assert calls[0]["spk_audio_prompt"] == str(speaker)
    assert calls[0]["emo_audio_prompt"] == str(emotion)
    assert calls[0]["emo_alpha"] == project.settings.tts_index_emo_alpha


def test_indextts_prefers_relocatable_python_module_command(tmp_path: Path) -> None:
    project = _project()
    model_dir = tmp_path / "index-tts" / "checkpoints"
    python = model_dir.parent / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    model_dir.mkdir()
    project.settings.tts_model_path = str(model_dir)

    command = _indextts_command(project)
    assert command is not None
    assert Path(command[0]).samefile(python)
    assert command[1:] == ["-m", "indextts.cli_v2"]


def test_external_backend_and_reference_change_tts_cache(tmp_path: Path) -> None:
    project = _project()
    sentence = project.sentences[0]
    project.settings.tts_backend = "gpt_sovits"
    project.settings.tts_model = "GPT-SoVITS-v4"
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(16_000, dtype=np.float32), 16_000, subtype="FLOAT")
    project.settings.tts_reference_source = "external"
    project.settings.tts_external_reference_audio = str(reference)
    project.settings.tts_external_reference_text = "これは参考音声です。"

    original = tts_cache_key(project, sentence)
    project.settings.tts_external_reference_text = "修正した参考テキストです。"
    assert tts_cache_key(project, sentence) != original


def test_external_tts_respects_bounded_request_concurrency(tmp_path: Path, monkeypatch) -> None:
    project = _project()
    project.settings.tts_backend = "gpt_sovits"
    project.settings.tts_model = "GPT-SoVITS-v4"
    project.settings.tts_request_concurrency = 3
    project.settings.tts_reference_source = "external"
    project.sentences = [
        project.sentences[0].model_copy(
            update={
                "id": f"s{index:06d}",
                "start_seconds": float(index),
                "end_seconds": index + 0.5,
            }
        )
        for index in range(1, 6)
    ]
    source = tmp_path / "source.wav"
    source.touch()
    reference_path = tmp_path / "reference.wav"
    sf.write(reference_path, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")
    project.settings.tts_external_reference_audio = str(reference_path)
    project.settings.tts_external_reference_text = "参考です。"
    reference = VoiceReference(reference_path, "参考です。", "shared")
    active = 0
    maximum = 0
    lock = threading.Lock()

    def run(_sentence, _reference, output):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.05)
            sf.write(output, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        "asmr_dubber.tts_backends._runner",
        lambda _project: (run, lambda: None),
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_voice_reference",
        lambda *_args: reference,
    )

    failures = synthesize_with_selected_backend(project, tmp_path, source)

    assert failures == []
    assert maximum == 3
    assert all(sentence.status == "synthesized" for sentence in project.sentences)


def test_external_tts_cancel_closes_active_client_without_waiting_for_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project()
    project.sentences = [project.sentences[0]]
    project.settings.tts_backend = "gpt_sovits"
    project.settings.tts_model = "GPT-SoVITS-v4"
    source = tmp_path / "source.wav"
    source.touch()
    reference_path = tmp_path / "reference.wav"
    sf.write(reference_path, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")
    reference = VoiceReference(reference_path, "参考です。", "shared")
    started = threading.Event()
    client_closed = threading.Event()
    signal = CancellationToken()

    def run(_sentence, _reference, _output):
        started.set()
        assert client_closed.wait(timeout=2)
        raise RuntimeError("client closed")

    monkeypatch.setattr(
        "asmr_dubber.tts_backends._runner",
        lambda _project: (run, client_closed.set),
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_voice_reference",
        lambda *_args: reference,
    )

    canceller = threading.Thread(
        target=lambda: (started.wait(timeout=2), signal.set()),
        daemon=True,
    )
    canceller.start()
    started_at = time.monotonic()
    with pytest.raises(OperationCancelledError):
        synthesize_with_selected_backend(
            project,
            tmp_path,
            source,
            cancel_event=signal,
        )
    canceller.join(timeout=2)

    assert time.monotonic() - started_at < 2
    assert client_closed.is_set()
    assert not list((tmp_path / "chinese").glob("*.tmp.wav"))


def test_edge_tts_does_not_prepare_or_record_reference_audio(tmp_path: Path, monkeypatch) -> None:
    project = _project()
    project.settings.tts_backend = "edge_tts"
    project.settings.tts_model = "edge-tts"
    source = tmp_path / "source.wav"
    source.touch()

    def run(_sentence, reference, output):
        assert reference.identity == "unused"
        sf.write(output, np.zeros(800, dtype=np.float32), 8_000, subtype="FLOAT")

    monkeypatch.setattr(
        "asmr_dubber.tts_backends._runner",
        lambda _project: (run, lambda: None),
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_voice_reference",
        lambda *_args: pytest.fail("Edge TTS must not prepare reference audio"),
    )

    assert synthesize_with_selected_backend(project, tmp_path, source) == []
    assert all(sentence.reference_file is None for sentence in project.sentences)


class _Response:
    is_error = False
    status_code = 200
    text = ""
    content = b"mock-wave"


class _RecordingClient:
    instances: ClassVar[list["_RecordingClient"]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def post(self, url, **kwargs):
        files = kwargs.get("files")
        uploaded = None
        if files:
            uploaded = files["prompt_wav"][1].read()
        self.calls.append({"url": url, **kwargs, "uploaded": uploaded})
        return _Response()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("reference_language", ["ja", "en", "zh"])
def test_gpt_sovits_http_contract(
    tmp_path: Path,
    monkeypatch,
    reference_language: SourceLanguage,
) -> None:
    _RecordingClient.instances.clear()
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    project = _project()
    project.settings.tts_api_base_url = "http://127.0.0.1:9880"
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference")
    reference = VoiceReference(
        reference_path,
        "Reference text.",
        "shared",
        language=reference_language,
    )
    output = tmp_path / "output.wav"

    run, cleanup = _gpt_sovits_runner(project)
    run(project.sentences[0], reference, output)
    cleanup()

    client = _RecordingClient.instances[0]
    call = client.calls[0]
    assert call["url"] == "http://127.0.0.1:9880/tts"
    assert call["json"]["text_lang"] == "zh"
    assert call["json"]["prompt_lang"] == reference_language
    assert call["json"]["ref_audio_path"] == str(reference_path)
    assert output.read_bytes() == b"mock-wave"
    assert client.closed is True


@pytest.mark.parametrize(
    ("mode", "endpoint", "has_prompt_text"),
    [
        ("zero_shot", "inference_zero_shot", True),
        ("cross_lingual", "inference_cross_lingual", False),
    ],
)
def test_cosyvoice_http_contract(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    endpoint: str,
    has_prompt_text: bool,
) -> None:
    _RecordingClient.instances.clear()
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    project = _project()
    project.settings.tts_api_base_url = "http://127.0.0.1:50000"
    project.settings.tts_cosyvoice_mode = mode
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference-audio")
    reference = VoiceReference(reference_path, "参考です。", "shared")
    output = tmp_path / "output.wav"

    run, cleanup = _cosyvoice_runner(project)
    run(project.sentences[0], reference, output)
    cleanup()

    client = _RecordingClient.instances[0]
    call = client.calls[0]
    assert call["url"] == f"http://127.0.0.1:50000/{endpoint}"
    assert ("prompt_text" in call["data"]) is has_prompt_text
    assert call["uploaded"] == b"reference-audio"
    assert output.read_bytes() == b"mock-wave"


def test_fish_speech_http_contract_and_authorization(tmp_path: Path, monkeypatch) -> None:
    _RecordingClient.instances.clear()
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    monkeypatch.setattr("asmr_dubber.tts_backends.saved_service_key", lambda _name: "secret")
    project = _project()
    project.settings.tts_backend = "fish_speech"
    project.settings.tts_api_base_url = "https://fish.example/api"
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference-audio")
    reference = VoiceReference(reference_path, "参考です。", "shared")
    output = tmp_path / "output.wav"

    run, cleanup = _fish_runner(project)
    run(project.sentences[0], reference, output)
    cleanup()

    client = _RecordingClient.instances[0]
    call = client.calls[0]
    assert client.kwargs["headers"] == {"Authorization": "Bearer secret"}
    assert call["url"] == "https://fish.example/api/v1/tts"
    assert call["json"]["references"][0]["text"] == "参考です。"
    assert output.read_bytes() == b"mock-wave"
    assert client.closed is True


def _mock_httpx_client(monkeypatch, handler):
    real_client = httpx.Client
    clients = []

    def factory(**kwargs):
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", factory)
    return clients


def test_edge_tts_contract_and_wav_conversion(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class FakeCommunicate:
        def __init__(self, text, *, voice, rate):
            calls.append((text, voice, rate))

        async def save(self, path):
            Path(path).write_bytes(b"mock-mp3")

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate))

    def fake_ffmpeg(arguments, **_kwargs):
        assert Path(arguments[2]).read_bytes() == b"mock-mp3"
        sf.write(Path(arguments[-1]), np.zeros(2400, dtype=np.float32), 24_000)

    monkeypatch.setattr("asmr_dubber.audio._run_ffmpeg", fake_ffmpeg)
    project = _project()
    project.settings.tts_backend = "edge_tts"
    project.settings.tts_voice = "zh-CN-XiaoyiNeural"
    project.settings.tts_speed = 1.15
    output = tmp_path / "output.wav"

    run, cleanup = _edge_tts_runner(project)
    run(project.sentences[0], VoiceReference(Path(), "", "unused"), output)
    cleanup()

    assert calls == [("让我们开始吧。", "zh-CN-XiaoyiNeural", "+15%")]
    assert sf.info(output).frames == 2400


def test_mimo_voice_clone_http_contract(tmp_path: Path, monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"audio": {"data": "bW9jay13YXZl"}}}]},
        )

    clients = _mock_httpx_client(monkeypatch, handler)
    monkeypatch.setattr("asmr_dubber.tts_backends.saved_service_key", lambda _name: "mimo-key")
    project = _project()
    project.settings.tts_backend = "mimo_tts"
    project.settings.tts_model = "mimo-v2.5-tts-voiceclone"
    project.settings.tts_api_base_url = "https://api.xiaomimimo.com/v1"
    project.settings.tts_style_prompt = "轻声耳语"
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference-audio")
    reference = VoiceReference(reference_path, "", "shared")
    output = tmp_path / "output.wav"

    run, cleanup = _mimo_runner(project)
    run(project.sentences[0], reference, output)
    cleanup()

    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.xiaomimimo.com/v1/chat/completions"
    assert request.headers["api-key"] == "mimo-key"
    assert payload["messages"][0] == {"role": "user", "content": "轻声耳语"}
    assert payload["messages"][1]["content"] == "让我们开始吧。"
    assert payload["audio"]["voice"].startswith("data:audio/wav;base64,")
    assert output.read_bytes() == b"mock-wave"
    assert clients[0].is_closed is True


def test_minimax_http_contract_and_hex_audio(tmp_path: Path, monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {"audio": b"mock-wave".hex(), "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    clients = _mock_httpx_client(monkeypatch, handler)
    monkeypatch.setattr("asmr_dubber.tts_backends.saved_service_key", lambda _name: "minimax-key")
    project = _project()
    project.settings.tts_backend = "minimax"
    project.settings.tts_model = "speech-2.8-hd"
    project.settings.tts_api_base_url = "https://api.minimaxi.com"
    project.settings.tts_voice = "female-shaonv"
    project.settings.tts_speed = 0.9
    project.settings.tts_volume = 1.2
    project.settings.tts_pitch = -2
    project.settings.tts_emotion = "calm"
    output = tmp_path / "output.wav"

    run, cleanup = _minimax_runner(project)
    run(project.sentences[0], VoiceReference(Path(), "", "unused"), output)
    cleanup()

    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.minimaxi.com/v1/t2a_v2"
    assert request.headers["Authorization"] == "Bearer minimax-key"
    assert payload["voice_setting"] == {
        "voice_id": "female-shaonv",
        "speed": 0.9,
        "vol": 1.2,
        "pitch": -2,
        "emotion": "calm",
    }
    assert payload["audio_setting"]["format"] == "wav"
    assert payload["output_format"] == "hex"
    assert output.read_bytes() == b"mock-wave"
    assert clients[0].is_closed is True


@pytest.mark.parametrize(
    "runner,backend",
    [(_mimo_runner, "mimo_tts"), (_minimax_runner, "minimax")],
)
def test_cloud_tts_requires_saved_key(runner, backend, monkeypatch) -> None:
    monkeypatch.setattr("asmr_dubber.tts_backends.saved_service_key", lambda _name: "")
    project = _project()
    project.settings.tts_backend = backend
    project.settings.tts_api_base_url = "https://example.test"

    with pytest.raises(SynthesisError, match="API Key"):
        runner(project)


def test_indextts_cancel_after_child_exit_is_not_reported_as_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project()
    project.sentences = [project.sentences[0]]
    project.settings.tts_model_path = str(tmp_path / "checkpoints")
    model_dir = Path(project.settings.tts_model_path)
    model_dir.mkdir()
    (model_dir / "config.yaml").touch()
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 2, dtype=np.float32), 16_000)
    reference_path = tmp_path / "reference.wav"
    sf.write(reference_path, np.zeros(16_000, dtype=np.float32), 16_000)
    reference = VoiceReference(reference_path, "参考です。", "shared")
    (tmp_path / "chinese").mkdir()
    signal = threading.Event()

    class FakeProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = iter(())
            self.returncode = None

        def wait(self):
            signal.set()
            self.returncode = 1
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("asmr_dubber.tts_backends._indextts_command", lambda _project: ["python"])
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_index_speaker_reference",
        lambda *_args: reference,
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_index_emotion_reference",
        lambda *_args: reference,
    )
    monkeypatch.setattr("asmr_dubber.tts_backends.subprocess.Popen", FakeProcess)

    with pytest.raises(OperationCancelledError):
        _synthesize_indextts_cli_batch(
            project,
            tmp_path,
            source,
            project.sentences,
            None,
            None,
            signal,
        )
