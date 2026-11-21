from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import asr, asr_runners, moss
from asmr_dubber.errors import AsmrDubberError, SynthesisError, classify_error
from asmr_dubber.indextts2_worker import (
    _CpuSemanticModelProxy,
    _LazyQwenEmotion,
    _should_offload_semantic_model,
)
from asmr_dubber.models import AudioInfo, DubProject, ProjectSettings, Sentence
from asmr_dubber.services import models as models_service
from asmr_dubber.services import storage_report
from asmr_dubber.tts_backends import _synthesize_indextts25_batch, _synthesize_indextts_cli_batch
from asmr_dubber.voice_reference import VoiceReference

ROOT = Path(__file__).parents[1]


def _load_advanced_importer():
    script = ROOT / "scripts/import_windows_advanced_dependency_pack.py"
    spec = importlib.util.spec_from_file_location("adv_importer_fixes_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_advanced_pack(path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "pack_id": "windows-advanced-dependencies",
        "pack_version": "1.0.0",
        "platform": "windows",
        "architecture": "x86_64",
        "python": "3.12",
        "components": [
            "application-ui",
            "advanced-asr",
            "qwen-forced-aligner",
            "asmr-vad",
            "pytorch-cu130",
        ],
    }
    files = {
        "payload/venv/Scripts/python.exe": b"advanced-python",
        "payload/venv/Lib/site-packages/pydantic/__init__.py": b"",
        "payload/venv/Lib/site-packages/torch/__init__.py": b"",
        "payload/venv/Lib/site-packages/torchaudio/__init__.py": b"",
        "payload/venv/Lib/site-packages/transformers/__init__.py": b"",
        "payload/venv/Lib/site-packages/faster_whisper/__init__.py": b"",
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as handle:
        handle.writestr("dependency-pack.json", json.dumps(manifest))
        for name, content in files.items():
            handle.writestr(name, content)


def test_advanced_importer_emits_progress_cleans_stale_staging_and_moves_files(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adv = _load_advanced_importer()
    archive = tmp_path / "advanced.zip"
    _write_advanced_pack(archive)
    portable = tmp_path / "portable"
    (portable / "venv/Scripts").mkdir(parents=True)
    (portable / "venv/Scripts/python.exe").write_bytes(b"existing-python")
    stale = portable / "t/a-stale-leftover"
    stale.mkdir(parents=True)
    (stale / "leftover.txt").write_text("stale", encoding="utf-8")

    replace_calls: list[tuple[str, str]] = []
    real_replace = adv.os.replace

    def recording_replace(src, dst):
        replace_calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(adv.os, "replace", recording_replace)
    adv.import_pack(archive, portable, "a" * 64, merge_existing=True)

    output = capsys.readouterr().out
    assert not stale.exists()
    assert "导入 Windows 进阶依赖包：校验 0%" in output
    assert "导入 Windows 进阶依赖包：解压" in output
    assert "导入 Windows 进阶依赖包：合并运行库" in output
    assert "导入 Windows 进阶依赖包：完成 100%" in output
    assert any(
        dst.endswith("faster_whisper\\__init__.py") or dst.endswith("faster_whisper/__init__.py")
        for _, dst in replace_calls
    )
    assert all(".advanced-dependency-pack-new" not in src for src, _ in replace_calls)
    assert (portable / "venv/Lib/site-packages/faster_whisper/__init__.py").is_file()


def test_models_download_progress_parses_import_percent_and_resets_stale_100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    events: list[tuple[str, float | int, float | int]] = []

    def fake_install(_identifier, *, progress, log_callback, cancel_event):
        progress(" 100 100M  100 100M    0     0  50.0M      0  0:00:02  0:00:02 --:--:-- 50.0M")
        progress("正在导入 Windows 进阶依赖包...")
        progress("导入 Windows 进阶依赖包：解压 42%（100/238）")
        progress("导入 Windows 进阶依赖包：合并运行库 85%（85/100）")
        return "ok"

    monkeypatch.setattr(models_service.runtime, "install_backend", fake_install)
    models_service.download(
        {"model": "anime_whisper"},
        lambda msg, cur, tot: events.append((msg, cur, tot)),
        SimpleNamespace(is_set=lambda: False),
    )

    assert events[0][1:] == (100.0, 100)
    assert events[1] == ("正在导入 Windows 进阶依赖包...", 0, 0)
    assert events[2][1:] == (42.0, 100)
    assert events[3][1:] == (85.0, 100)


def test_faster_whisper_resolves_shorthand_and_fails_fast_when_offline_and_uncached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "english.wav"
    audio_path.touch()
    seen: list[tuple[str, dict[str, object]]] = []

    class StrictWhisperModel:
        def __init__(
            self,
            model_size_or_path,
            device="auto",
            compute_type="default",
            local_files_only=False,
        ):
            seen.append(
                (
                    model_size_or_path,
                    {
                        "device": device,
                        "compute_type": compute_type,
                        "local_files_only": local_files_only,
                    },
                )
            )
            raise OSError("LocalEntryNotFoundError: cannot find snapshot in local cache")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=StrictWhisperModel, BatchedInferencePipeline=object),
    )
    monkeypatch.setattr(asr_runners, "external_downloads_allowed", lambda: False)
    monkeypatch.setattr(asr, "resolve_model_source", lambda model_id: model_id)
    monkeypatch.setattr(asr, "cached_model_path", lambda _model_id: None)

    progress_updates: list[tuple[str, int, int]] = []
    with pytest.raises(AsmrDubberError, match=r"Faster-Whisper 模型尚未完整下载：large-v3"):
        asr_runners._transcribe_faster_whisper(
            audio_path,
            ProjectSettings(asr_backend="faster_whisper", asr_model="large-v3", asr_device="cpu"),
            lambda msg, cur, tot: progress_updates.append((msg, cur, tot)),
            source_language="en",
        )

    assert seen == [
        (
            "Systran/faster-whisper-large-v3",
            {"device": "cpu", "compute_type": "int8", "local_files_only": True},
        )
    ]
    assert any("加载 Faster-Whisper" in msg for msg, _, _ in progress_updates)


def test_moss_runner_advances_progress_across_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.ones(16_000 * 3, dtype=np.float32) * 0.1, 16_000)
    monkeypatch.setattr(moss, "runtime_python", lambda: tmp_path / "python")
    (tmp_path / "python").touch()
    monkeypatch.setattr(moss, "complete_model", lambda **kwargs: True)
    monkeypatch.setattr(asr, "portable_home", lambda: tmp_path)

    class ChunkedProcess:
        def __init__(self, command, **kwargs):
            self._polls = 0
            self.returncode = 0
            stdout_file = kwargs["stdout"]
            stdout_file.write(json.dumps({"progress": "MOSS 加载模型（cuda, float16）"}) + "\n")
            stdout_file.write(json.dumps({"progress": "MOSS 识别分块 1/3"}) + "\n")
            stdout_file.write(json.dumps({"progress": "MOSS 识别分块 2/3"}) + "\n")
            stdout_file.write(json.dumps({"progress": "MOSS 识别分块 3/3"}) + "\n")
            stdout_file.flush()
            result = {
                "segments": [{"text": "Hi", "start": 0.1, "end": 1.0, "speaker": "c0001:S01"}]
            }
            Path(command[-1]).write_text(json.dumps(result), encoding="utf-8")

        def poll(self):
            self._polls += 1
            return None if self._polls == 1 else 0

    monkeypatch.setattr(asr.subprocess, "Popen", ChunkedProcess)
    monkeypatch.setattr(asr, "terminate_process_tree", lambda _process: None)

    updates: list[tuple[str, int, int]] = []
    settings = ProjectSettings(
        asr_backend="moss_transcribe_diarize",
        asr_model=moss.MODEL_ID,
        asr_moss_chunk_seconds=15,
    )
    rows, _ = asr.transcribe_source(
        audio,
        settings,
        "en",
        progress=lambda msg, cur, tot: updates.append((msg, cur, tot)),
    )
    assert len(rows) == 1
    assert any(cur > 0 for _, cur, _ in updates), updates


def test_open_folder_creates_missing_projects_directory_and_opens_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    projects_dir = tmp_path / "custom-projects"
    opened: list[Path] = []
    monkeypatch.setattr(storage_report, "open_directory", lambda path: opened.append(path) or path)

    result = storage_report.open_folder("projects", str(projects_dir))
    assert projects_dir.is_dir()
    assert opened == [projects_dir]
    assert result == {"path": str(projects_dir)}


def test_indextts2_low_vram_helpers_behave_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeEmotion:
        def __init__(self, model_dir: str) -> None:
            calls.append(f"init:{model_dir}")

        def inference(self, text: str) -> dict[str, float]:
            calls.append(f"infer:{text}")
            return {"calm": 1.0}

    lazy = _LazyQwenEmotion("emo-dir", FakeEmotion, low_vram=False)
    assert calls == []
    assert lazy.inference("开心") == {"calm": 1.0}
    assert lazy.inference("平静") == {"calm": 1.0}
    assert calls == ["init:emo-dir", "infer:开心", "infer:平静"]

    class FakeTensor:
        def __init__(self, device: str) -> None:
            self.device = device

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return FakeTensor("cpu")

    seen_devices: list[tuple[str, str]] = []

    class FakeModule:
        extra_attr = "kept"

        def eval(self) -> FakeModule:
            return self

        def __call__(self, x, *, attention_mask=None):
            seen_devices.append((x.device, attention_mask.device))
            return "ok"

    proxy = _CpuSemanticModelProxy(FakeModule())
    assert proxy.to("cuda") is proxy
    assert proxy.eval() is proxy
    assert proxy.extra_attr == "kept"
    assert proxy(FakeTensor("cuda"), attention_mask=FakeTensor("cuda")) == "ok"
    assert seen_devices == [("cpu", "cpu")]

    fake_torch_8g = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda _dev: (6 * 1024**3, 8 * 1024**3))
    )
    fake_torch_24g = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda _dev: (20 * 1024**3, 24 * 1024**3))
    )
    monkeypatch.delenv("ASMR_DUBBER_INDEXTTS2_LOW_VRAM", raising=False)
    assert _should_offload_semantic_model(fake_torch_8g, "cuda") is True
    assert _should_offload_semantic_model(fake_torch_24g, "cuda") is False
    assert _should_offload_semantic_model(fake_torch_8g, "cpu") is False
    monkeypatch.setenv("ASMR_DUBBER_INDEXTTS2_LOW_VRAM", "1")
    assert _should_offload_semantic_model(fake_torch_24g, "cuda") is True
    monkeypatch.setenv("ASMR_DUBBER_INDEXTTS2_LOW_VRAM", "0")
    assert _should_offload_semantic_model(fake_torch_8g, "cuda") is False


def _batch_oom_project(tmp_path: Path) -> tuple[DubProject, Path, VoiceReference]:
    source = tmp_path / "source.wav"
    ref_path = tmp_path / "ref.wav"
    sf.write(source, np.zeros(16_000 * 3, dtype=np.float32), 16_000)
    sf.write(ref_path, np.zeros(16_000, dtype=np.float32), 16_000)
    (tmp_path / "chinese").mkdir(exist_ok=True)
    sentences = [
        Sentence(
            id=f"s{i:06d}",
            start_seconds=0.1 * i,
            end_seconds=0.1 * i + 0.08,
            ja_text=f"テスト{i}",
            zh_text=f"测试{i}",
        )
        for i in range(1, 5)
    ]
    project = DubProject(
        source=AudioInfo(
            path="source.wav",
            sha256="b" * 64,
            duration_seconds=3.0,
            sample_rate=16_000,
            channels=1,
        ),
        sentences=sentences,
    )
    return project, source, VoiceReference(ref_path, "参考", "shared")


@pytest.mark.parametrize("backend", ["indextts2", "indextts2_5"])
def test_indextts_global_oom_init_failure_does_not_mark_every_sentence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    project, source, reference = _batch_oom_project(tmp_path)
    runtime_root = tmp_path / backend
    model_dir = runtime_root / "checkpoints"
    model_dir.mkdir(parents=True)
    config = model_dir / "config.yaml"
    config.touch()
    if backend == "indextts2":
        project.settings.tts_model_path = str(model_dir)
        project.settings.tts_config_path = str(config)
        monkeypatch.setattr("asmr_dubber.tts_backends._indextts_command", lambda _p: ["python"])
    else:
        project.settings.tts_index25_model_path = str(model_dir)
        project.settings.tts_index25_config_path = str(config)
        monkeypatch.setattr(
            "asmr_dubber.tts_backends._indextts25_command",
            lambda _p: ["python", "indextts25_worker.py"],
        )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_index_speaker_reference", lambda *_: reference
    )
    monkeypatch.setattr(
        "asmr_dubber.tts_backends.prepare_index_emotion_reference", lambda *_: reference
    )

    captured_env: dict[str, str] = {}

    class OomProcess:
        def __init__(self, _command, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            self.stdout = iter(("ERROR: model initialization failed: CUDA out of memory.\n",))
            self.returncode = 3

        def wait(self):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr("asmr_dubber.tts_backends.subprocess.Popen", OomProcess)

    runner = (
        _synthesize_indextts_cli_batch if backend == "indextts2" else _synthesize_indextts25_batch
    )
    with pytest.raises(
        SynthesisError, match=r"模型初始化失败：显卡显存或系统虚拟内存不足"
    ) as exc_info:
        runner(project, tmp_path, source, project.sentences, None, None, None)

    message = str(exc_info.value)
    assert message.count("CUDA out of memory") == 1
    assert all(sentence.error is None for sentence in project.sentences)
    assert all(sentence.status != "error" for sentence in project.sentences)
    code, hint = classify_error(exc_info.value)
    assert code == "CUDA_OOM"
    assert "虚拟内存" in hint
    if os.name == "nt":
        assert captured_env.get("PYTORCH_CUDA_ALLOC_CONF") == "max_split_size_mb:128"


def test_home_projects_folder_button_and_capped_error_toast() -> None:
    html = (ROOT / "src/asmr_dubber/frontend/index.html").read_text(encoding="utf-8")
    css = (ROOT / "src/asmr_dubber/frontend/styles.css").read_text(encoding="utf-8")
    projects_js = (ROOT / "src/asmr_dubber/frontend/projects.js").read_text(encoding="utf-8")
    app_js = (ROOT / "src/asmr_dubber/frontend/app.js").read_text(encoding="utf-8")

    assert 'id="openProjectsFolder"' in html
    assert "直接删除对应项目文件夹即可删除整个项目" in html
    assert "storage/open" in projects_js and "'projects'" in projects_js
    assert "max-height: min(40vh, 320px)" in css and "overflow-y: auto" in css

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    session_uri = (
        "data:text/javascript;base64,"
        + base64.b64encode((ROOT / "src/asmr_dubber/frontend/session.js").read_bytes()).decode()
    )
    fn_match = app_js.split("function formatTaskErrorNotice(task) {", 1)[1].split(
        "\nlet polling =", 1
    )[0]
    script = f"""
const {{ state, t }} = await import({json.dumps(session_uri)});
state.boot = {{ labels: {{}}, locales: {{}} }};
function formatTaskErrorNotice(task) {{
{fn_match}
const out = formatTaskErrorNotice({{
  error_hint: '显存提示',
  error: 'X'.repeat(800),
}});
if (
  !out.startsWith('显存提示\\n') ||
  !out.includes('…（完整信息请看下方日志）') ||
  out.length >= 600
) {{
  throw new Error('Unexpected toast formatting: ' + out);
}}
"""
    res = subprocess.run(
        [node, "--input-type=module", "-"],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res.returncode == 0, res.stderr
