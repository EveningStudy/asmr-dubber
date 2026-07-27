import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from asmr_dubber.models import AudioInfo, DubProject, Sentence
from asmr_dubber.tts import shared_reference_sentence, tts_cache_key
from asmr_dubber.tts_backends import (
    _indextts_command,
    _load_indextts,
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

    project.settings.global_overlap_seconds = 2.0
    project.settings.chinese_gain_db = -3.0
    project.settings.chinese_target_active_rms_dbfs = -34.0
    assert tts_cache_key(project, sentence) == original

    project.settings.tts_speed = 1.1
    assert tts_cache_key(project, sentence) != original
    project.settings.tts_speed = 1.0
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
