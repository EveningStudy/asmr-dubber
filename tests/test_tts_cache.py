import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber.errors import SynthesisError
from asmr_dubber.models import AudioInfo, DubProject, Sentence
from asmr_dubber.tts import _enable_voxcpm_prompt_cache, shared_reference_sentence, tts_cache_key
from asmr_dubber.tts_backends import (
    _indextts_command,
    _load_indextts,
    _load_qwen3,
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
            duration_seconds=3.0,
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
            )
        ],
    )


def test_tts_cache_tracks_clone_inputs_but_not_mix_only_settings() -> None:
    project = _project()
    project.settings.tts_backend = "voxcpm2"
    project.settings.tts_model = "openbmb/VoxCPM2"
    sentence = project.sentences[0]
    original = tts_cache_key(project, sentence)

    project.settings.global_overlap_seconds = 2.0
    project.settings.chinese_gain_db = -3.0
    project.settings.chinese_target_active_rms_dbfs = -34.0
    project.settings.chinese_stem_peak_dbfs = -4.0
    assert tts_cache_key(project, sentence) == original

    project.settings.tts_clone_mode = "ultimate"
    assert tts_cache_key(project, sentence) != original

    project.settings.tts_clone_mode = "reference_only"
    sentence.zh_text = "现在开始吧。"
    assert tts_cache_key(project, sentence) != original


def test_shared_reference_is_deterministic_and_affects_every_tts_key() -> None:
    project = _project()
    project.settings.tts_backend = "voxcpm2"
    project.settings.tts_model = "openbmb/VoxCPM2"
    target = project.sentences[0]
    anchor = Sentence(
        id="s000002",
        start_seconds=2.0,
        end_seconds=8.0,
        ja_text="これは十分に長くて明瞭な音色の参考文章です。",
        zh_text="这是一句足够长而清晰的音色参考。",
    )
    project.source.duration_seconds = 12.0
    project.sentences.append(anchor)

    assert shared_reference_sentence(project).id == "s000002"
    original = tts_cache_key(project, target)
    target.start_seconds = 0.4
    target.end_seconds = 2.0
    target.ja_text = "开始吧。"
    assert tts_cache_key(project, target) == original

    project.settings.tts_clone_mode = "stable_voice_sentence_style"
    styled = tts_cache_key(project, target)
    target.start_seconds = 0.5
    assert tts_cache_key(project, target) != styled

    anchor.end_seconds = 8.5
    assert tts_cache_key(project, target) != original


def test_index_default_uses_shared_speaker_and_current_sentence_emotion(
    tmp_path: Path,
) -> None:
    project = _project()
    target = project.sentences[0]
    anchor = Sentence(
        id="s000002",
        start_seconds=2.0,
        end_seconds=8.0,
        ja_text="これは十分に長くて明瞭な音色の参考文章です。",
        zh_text="这是一句足够长而清晰的音色参考。",
    )
    project.source.duration_seconds = 9.0
    project.sentences.append(anchor)
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000 * 9, dtype=np.float32), 16_000, subtype="FLOAT")

    speaker = prepare_index_speaker_reference(project, tmp_path, source, target)
    emotion = prepare_index_emotion_reference(project, tmp_path, source, target, speaker)

    assert speaker.sentence is anchor
    assert emotion is not None
    assert emotion.sentence is target
    assert emotion.path != speaker.path


def test_index_cache_tracks_independent_emotion_reference() -> None:
    project = _project()
    sentence = project.sentences[0]
    project.settings.tts_reference_sentence_id = sentence.id
    original = tts_cache_key(project, sentence)

    sentence.start_seconds = 0.3
    assert tts_cache_key(project, sentence) != original

    project.settings.tts_index_emotion_source = "speaker_reference"
    shared_emotion = tts_cache_key(project, sentence)
    sentence.start_seconds = 0.4
    assert tts_cache_key(project, sentence) != shared_emotion


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

    fake_package = SimpleNamespace()
    fake_infer = SimpleNamespace(IndexTTS2=FakeIndexTTS2)
    monkeypatch.setitem(sys.modules, "indextts", fake_package)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", fake_infer)
    speaker = tmp_path / "speaker.wav"
    emotion = tmp_path / "emotion.wav"
    output = tmp_path / "output.wav"
    reference = VoiceReference(
        speaker,
        "音色",
        "speaker",
        emotion_path=emotion,
        emotion_identity="emotion",
    )

    run, cleanup = _load_indextts(project)
    try:
        run(project.sentences[0], reference, output)
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


def test_backend_and_external_reference_change_tts_cache(tmp_path: Path) -> None:
    project = _project()
    sentence = project.sentences[0]
    original = tts_cache_key(project, sentence)

    project.settings.tts_backend = "qwen3_tts"
    project.settings.tts_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert tts_cache_key(project, sentence) != original

    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(16_000, dtype=np.float32), 16_000, subtype="FLOAT")
    project.settings.tts_reference_source = "external"
    project.settings.tts_external_reference_audio = str(reference)
    project.settings.tts_external_reference_text = "これは参考音声です。"
    external = tts_cache_key(project, sentence)
    project.settings.tts_external_reference_text = "修正した参考テキストです。"
    assert tts_cache_key(project, sentence) != external


def test_backend_rejects_clone_mode_it_does_not_support(tmp_path: Path) -> None:
    project = _project()
    project.settings.tts_backend = "indextts2"
    project.settings.tts_model = "IndexTTS2"
    project.settings.tts_clone_mode = "stable_hifi"

    with pytest.raises(SynthesisError, match="不支持参考策略"):
        synthesize_with_selected_backend(project, tmp_path, tmp_path / "source.wav")


def test_qwen_voice_clone_prompt_is_reused_for_same_reference(tmp_path: Path, monkeypatch) -> None:
    project = _project()
    project.settings.tts_backend = "qwen3_tts"
    project.settings.tts_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    project.settings.tts_device = "cpu"
    prompt_calls: list[dict[str, object]] = []
    generation_calls: list[dict[str, object]] = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def create_voice_clone_prompt(self, **kwargs):
            prompt_calls.append(kwargs)
            return {"encoded": len(prompt_calls)}

        def generate_voice_clone(self, **kwargs):
            generation_calls.append(kwargs)
            return [np.zeros(240, dtype=np.float32)], 24_000

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeModel))

    reference_path = tmp_path / "reference.wav"
    sf.write(reference_path, np.zeros(1_600, dtype=np.float32), 16_000, subtype="FLOAT")
    reference = VoiceReference(reference_path, "参考です。", "same-reference")
    second = project.sentences[0].model_copy(update={"id": "s000002", "zh_text": "第二句。"})
    run, cleanup = _load_qwen3(project)
    try:
        run(project.sentences[0], reference, tmp_path / "first.wav")
        run(second, reference, tmp_path / "second.wav")
    finally:
        cleanup()

    assert len(prompt_calls) == 1
    assert len(generation_calls) == 2
    assert all("voice_clone_prompt" in item for item in generation_calls)
    assert all("ref_audio" not in item for item in generation_calls)


def test_voxcpm_public_prompt_builder_is_cached_without_replacing_generate() -> None:
    calls: list[dict[str, object]] = []

    class FakeTTSModel:
        def build_prompt_cache(self, **kwargs):
            calls.append(kwargs)
            return {"call": len(calls)}

    model = SimpleNamespace(tts_model=FakeTTSModel())
    _enable_voxcpm_prompt_cache(model)
    first = model.tts_model.build_prompt_cache(reference_wav_path="same.wav")
    second = model.tts_model.build_prompt_cache(reference_wav_path="same.wav")
    third = model.tts_model.build_prompt_cache(reference_wav_path="different.wav")

    assert first is second
    assert third is not first
    assert len(calls) == 2
