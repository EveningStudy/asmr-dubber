from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import asr
from asmr_dubber.autoflow import inprocess_pipeline, processes, project_cli
from asmr_dubber.models import DubProject, ProjectSettings, Sentence, load_project, save_project
from asmr_dubber.pipeline import (
    analyze_project,
    create_project,
    mix_project,
    synthesize_project,
    translate_project,
)
from asmr_dubber.tts_daemon import (
    batch_worker_session,
    daemon_pool_active,
    run_resident_indextts_batch,
)


def _write_test_wav(
    path: Path, seconds: float = 3.0, rate: int = 16_000, *, mono: bool = False
) -> Path:
    t = np.arange(round(seconds * rate), dtype=np.float64) / rate
    sig = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(path, sig if mono else np.column_stack([sig, sig]), rate)
    return path


def test_inprocess_batch_pipeline_executes_without_cli_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    media = _write_test_wav(tmp_path / "track01.wav", seconds=2.5)
    paths = SimpleNamespace(
        cli_command=(sys.executable, "-m", "asmr_dubber.cli"),
        asmr_root=tmp_path,
        asmr_home=tmp_path / "home",
    )
    assert inprocess_pipeline.can_run_in_process(paths) is True

    cli_spawns: list[list[str]] = []
    real_popen = subprocess.Popen

    def spy_popen(cmd, *args, **kwargs):
        argv = [str(x) for x in cmd] if isinstance(cmd, (list, tuple)) else [str(cmd)]
        if any("asmr_dubber.cli" in item for item in argv):
            cli_spawns.append(argv)
        return real_popen(cmd, *args, **kwargs)

    def fake_transcribe(_audio, _settings, source_language="ja", **_kw):
        return (
            [
                Sentence(id="s000001", start_seconds=0.2, end_seconds=1.1, source_text="おはよう"),
                Sentence(id="s000002", start_seconds=1.3, end_seconds=2.2, source_text="おやすみ"),
            ],
            source_language,
        )

    def fake_synth(project: DubProject, project_dir: Path, _source: Path, **_kw):
        from asmr_dubber.tts import tts_cache_key

        out_dir = project_dir / "chinese"
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in project.sentences:
            if not (s.enabled and s.zh_text):
                continue
            wav = out_dir / f"{s.id}.wav"
            _write_test_wav(wav, seconds=0.6)
            s.tts_file = str(wav.relative_to(project_dir))
            s.tts_duration_seconds = 0.6
            s.tts_cache_key = tts_cache_key(project, s)
            s.status = "synthesized"
        return []

    with (
        patch.object(subprocess, "Popen", side_effect=spy_popen),
        patch("asmr_dubber.pipeline.transcribe_source", side_effect=fake_transcribe),
        patch(
            "asmr_dubber.pipeline.translate_project",
            side_effect=lambda p, d, **_k: (
                [
                    setattr(s, "zh_text", f"译{s.id}") or setattr(s, "status", "translated")
                    for s in p.sentences
                ],
                save_project(p, d),
            ),
        ),
        patch(
            "asmr_dubber.tts_backends.synthesize_with_selected_backend",
            side_effect=fake_synth,
        ),
    ):
        manifest = project_cli.create_asmr_project(paths, media, source_language="ja")
        proj, proj_dir = load_project(manifest)
        proj.settings.separation_enabled = False
        proj.settings.spatial_rtf_enabled = False
        proj.settings.tts_quality_check = False
        save_project(proj, proj_dir)

        assert project_cli.run_asmr_cli(paths, "analyze", str(manifest)) is None
        res_tr = project_cli.run_asmr_cli(paths, "translate", str(manifest))
        assert res_tr is not None and res_tr["ok"] is True
        res_tts = project_cli.run_asmr_cli(paths, "synthesize", str(manifest))
        assert res_tts is not None and res_tts["ok"] is True
        res_mix = project_cli.run_asmr_cli(paths, "mix", str(manifest))
        assert res_mix is not None and res_mix["ok"] is True
        assert (
            project_cli.run_asmr_cli(paths, "subtitles", str(manifest), "--language", "bilingual")
            is None
        )

    assert cli_spawns == []
    # Verify custom process runner monkeypatches still fall back to subprocess stub cleanly.
    seen_fallback: list[list[str]] = []
    monkeypatch.setattr(processes, "run_process", lambda cmd, **_kw: seen_fallback.append(cmd))
    assert inprocess_pipeline.can_run_in_process(paths) is False
    project_cli.run_asmr_cli(paths, "subtitles", str(manifest))
    assert len(seen_fallback) == 1


def test_ten_track_batch_resident_asr_and_tts_daemon_load_under_10_percent(tmp_path: Path) -> None:
    daemon_script = tmp_path / "mock_indextts_daemon.py"
    daemon_script.write_text(
        """import json, pathlib, sys, time
import numpy as np, soundfile as sf
# Simulate one-time model load (0.04s) vs per-track synthesis (0.06s x 10 = 0.60s)
t0 = time.perf_counter()
time.sleep(0.04)
load_elapsed = time.perf_counter() - t0
pathlib.Path(sys.argv[1]).write_text(str(load_elapsed), encoding="utf-8")
print("READY: mock_daemon", flush=True)
for raw in sys.stdin:
    msg = json.loads(raw)
    if msg.get("action") == "quit":
        break
    batch = pathlib.Path(msg["batch_file"])
    for line in batch.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        time.sleep(0.06)
        out = pathlib.Path(task["output"])
        sf.write(out, np.zeros(8000, dtype="float32"), 16000)
        print(f"Generated: {task['id']}", flush=True)
    print(json.dumps({"event": "batch_done", "exit_code": 0}), flush=True)
""",
        encoding="utf-8",
    )
    load_marker = tmp_path / "daemon_loads.txt"
    asr_loads = 0

    def load_asr() -> str:
        nonlocal asr_loads
        asr_loads += 1
        time.sleep(0.02)
        return "resident-asr"

    from collections import deque

    t_start = time.perf_counter()
    with batch_worker_session():
        assert daemon_pool_active() is True
        for track_idx in range(10):
            # Nested recognition_session inside per-track ASR/QC must reuse outer resident cache!
            with asr.recognition_session():
                model = asr._session_model("anime-whisper:cuda", load_asr)
                assert model == "resident-asr"
            manifest = tmp_path / f"batch_{track_idx}.jsonl"
            out_wav = tmp_path / f"out_{track_idx}.wav"
            manifest.write_text(
                json.dumps({"id": f"s{track_idx}", "output": str(out_wav)}) + "\n",
                encoding="utf-8",
            )
            generated: list[str] = []
            rc = run_resident_indextts_batch(
                worker_id="indextts25",
                label="IndexTTS-2.5",
                daemon_command=[sys.executable, str(daemon_script), str(load_marker)],
                cwd=tmp_path,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                manifest=manifest,
                total_items=1,
                timeout_seconds=30.0,
                progress=None,
                cancel_event=None,
                on_line=lambda ln, g=generated: (
                    (g.append(ln) or True) if ln.startswith("Generated:") else False
                ),
                output_tail=deque(maxlen=20),
            )
            assert rc == 0
            assert out_wav.is_file()
            assert len(generated) == 1

    total_elapsed = time.perf_counter() - t_start
    daemon_load_s = float(load_marker.read_text(encoding="utf-8"))
    total_load_s = daemon_load_s + 0.02
    assert asr_loads == 1
    assert total_load_s / total_elapsed < 0.10


def test_crash_kill_and_resume_across_analyze_translate_synthesize_and_mix_loses_zero_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    media = _write_test_wav(tmp_path / "episode.wav", seconds=8.0)
    project, proj_dir = create_project(
        media,
        tmp_path / "projects",
        settings=ProjectSettings(
            separation_enabled=False,
            spatial_rtf_enabled=False,
            tts_backend="edge_tts",
            tts_quality_check=False,
            translation_provider="openai_compatible",
            translation_base_url="http://127.0.0.1:9999/v1",
        ),
    )

    # 1. Crash mid-analyze -> resume completes analyze cleanly.
    analyze_attempts = 0

    def flaky_transcribe(_audio, _settings, source_language="ja", **_kw):
        nonlocal analyze_attempts
        analyze_attempts += 1
        if analyze_attempts == 1:
            raise KeyboardInterrupt("simulated kill during ASR")
        return (
            [
                Sentence(
                    id=f"s{i:06d}",
                    start_seconds=round((i - 1) * 1.2 + 0.1, 2),
                    end_seconds=round((i - 1) * 1.2 + 1.0, 2),
                    source_text=f"テスト文第{i}番です",
                )
                for i in range(1, 5)
            ],
            source_language,
        )

    with patch("asmr_dubber.pipeline.transcribe_source", side_effect=flaky_transcribe):
        with pytest.raises(KeyboardInterrupt):
            analyze_project(project, proj_dir)
        project, proj_dir = load_project(proj_dir)
        assert project.sentences == []
        analyze_project(project, proj_dir)

    project, proj_dir = load_project(proj_dir)
    assert len(project.sentences) == 4

    # 2. Crash mid-translate after chunk 1 -> resume translates ONLY chunk 2 (zero re-billing!).
    api_batches: list[list[str]] = []

    def two_chunk_builder(sentences, *_args, **_kwargs):
        from asmr_dubber.translation import TranslationChunk

        items = list(sentences)
        if len(items) > 2:
            return [TranslationChunk(items[:2]), TranslationChunk(items[2:])]
        return [TranslationChunk(items)]

    def fake_translate_chunk(chunk, *_args, **_kwargs):
        from asmr_dubber.translation import TranslationEntry

        ids = [s.id for s in chunk.sentences]
        api_batches.append(ids)
        if len(api_batches) == 2:
            raise KeyboardInterrupt("simulated kill on chunk 2")
        return {
            s.id: TranslationEntry(zh=f"中文翻译_{s.id}", kind="speech") for s in chunk.sentences
        }

    with (
        patch("asmr_dubber.translation._chunks", side_effect=two_chunk_builder),
        patch(
            "asmr_dubber.translation.LLMTranslator.translate_chunk",
            side_effect=fake_translate_chunk,
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            translate_project(project, proj_dir, api_key="local")

        # Reload from disk after crash: chunk 1 (s000001, s000002) MUST already be persisted!
        project, proj_dir = load_project(proj_dir)
        assert [bool(s.zh_text) for s in project.sentences] == [True, True, False, False]

        # Resume translation: ONLY s000003 & s000004 are sent to the translation API!
        translate_project(project, proj_dir, api_key="local")

    project, proj_dir = load_project(proj_dir)
    assert all(s.zh_text.startswith("中文翻译_") for s in project.sentences)
    assert api_batches[0] == ["s000001", "s000002"]
    assert api_batches[-1] == ["s000003", "s000004"]
    assert set(api_batches[0]).isdisjoint(api_batches[-1])

    # 3. Crash mid-synthesize after 2 sentences -> resume synthesizes ONLY remaining 2 sentences!
    synth_calls: list[str] = []

    def crash_mid_runner(_project):
        def run(sentence, _reference, temporary: Path) -> None:
            if len(synth_calls) >= 2:
                raise KeyboardInterrupt("simulated kill during sentence 3 TTS")
            synth_calls.append(sentence.id)
            _write_test_wav(temporary, seconds=0.6, mono=True)

        return run, lambda: None

    with patch("asmr_dubber.tts_backends._runner", side_effect=crash_mid_runner):
        with pytest.raises(KeyboardInterrupt):
            synthesize_project(project, proj_dir)

        project, proj_dir = load_project(proj_dir)
        done_after_crash = [s.id for s in project.sentences if s.tts_file]
        assert done_after_crash == ["s000001", "s000002"]

        # Resume without crashing: ONLY s000003 and s000004 are synthesized!
        synth_calls.clear()
        with patch(
            "asmr_dubber.tts_backends._runner",
            side_effect=lambda _p: (
                lambda s, _ref, tmp: (
                    synth_calls.append(s.id),
                    _write_test_wav(tmp, seconds=0.6, mono=True),
                )[1],
                lambda: None,
            ),
        ):
            synthesize_project(project, proj_dir)

    project, proj_dir = load_project(proj_dir)
    assert synth_calls == ["s000003", "s000004"]
    assert all(s.tts_file for s in project.sentences)

    # 4. Crash mid-mix -> no partial output left behind; resume succeeds cleanly!
    with (
        patch(
            "asmr_dubber.pipeline.mix_original_and_stem",
            side_effect=KeyboardInterrupt("kill mix"),
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        mix_project(project, proj_dir)
    project, proj_dir = load_project(proj_dir)
    assert project.output_file is None

    final_out = mix_project(project, proj_dir)
    assert final_out.is_file()
