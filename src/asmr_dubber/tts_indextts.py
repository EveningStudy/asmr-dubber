from __future__ import annotations

import gc
import json
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import OperationCancelledError, SynthesisError
from .models import DubProject, Sentence
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_process,
    terminate_process_tree,
    unregister_process,
)
from .tts import (
    attempt_seed,
    spoken_text,
    target_duration_for_sentence,
    tts_cache_key,
)
from .voice_reference import VoiceReference, effective_index_emotion_weight

if TYPE_CHECKING:
    import subprocess

Progress = Callable[[str, int, int], None]


def _load_indextts(project: DubProject) -> tuple[Any, Callable[[], None]]:
    try:
        from indextts.infer_v2 import IndexTTS2
    except ImportError as exc:
        raise SynthesisError(
            "IndexTTS2 未安装；请克隆官方 index-tts 仓库并在其兼容环境运行本程序。"
        ) from exc
    config = Path(project.settings.tts_config_path).expanduser()
    model_dir = Path(project.settings.tts_model_path).expanduser()
    if not config.is_file() or not model_dir.is_dir():
        raise SynthesisError("IndexTTS2 需要有效的 config.yaml 路径和 checkpoints 模型目录。")
    model = IndexTTS2(
        cfg_path=str(config.resolve()),
        model_dir=str(model_dir.resolve()),
        use_fp16=project.settings.tts_index_use_fp16,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        kwargs: dict[str, Any] = {
            "spk_audio_prompt": str(reference.path),
            "text": spoken_text(project.settings, sentence.zh_text),
            "output_path": str(output),
            "emo_alpha": project.settings.tts_index_emo_alpha,
            "use_random": False,
            "verbose": False,
        }
        if project.settings.tts_index_emotion_source == "text":
            kwargs.update(
                use_emo_text=True,
                emo_text=project.settings.tts_index_emo_text.strip()
                or spoken_text(project.settings, sentence.zh_text),
            )
        elif reference.emotion_path is not None:
            kwargs["emo_audio_prompt"] = str(reference.emotion_path)
        model.infer(**kwargs)

    def cleanup() -> None:
        nonlocal model
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return run, cleanup


def _indextts_cli(project: DubProject) -> Path | None:
    model_dir = Path(project.settings.tts_model_path).expanduser().resolve()
    candidates = [
        model_dir.parent / ".venv" / "bin" / "indextts2",
        model_dir.parent / ".venv" / "Scripts" / "indextts2.exe",
    ]
    configured = project.settings.tts_executable.strip()
    if configured and "indextts" in Path(configured).name.lower():
        candidates.insert(0, Path(configured).expanduser().resolve())
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _indextts_command(project: DubProject) -> list[str] | None:
    """Prefer the relocatable venv interpreter over an absolute-path EXE shim."""
    from . import tts_backends

    model_dir = Path(project.settings.tts_model_path).expanduser().resolve()
    runtime_root = model_dir.parent
    python_candidates = (
        runtime_root / ".venv" / "Scripts" / "python.exe",
        runtime_root / ".venv" / "bin" / "python",
    )
    python = next((candidate for candidate in python_candidates if candidate.is_file()), None)
    if python is not None:
        return [str(python), "-m", "indextts.cli_v2"]
    executable = tts_backends._indextts_cli(project)
    return [str(executable)] if executable is not None else None


def _synthesize_indextts_cli_batch(
    project: DubProject,
    project_dir: Path,
    source: Path,
    pending: list[Sentence],
    progress: Progress | None,
    on_sentence: Callable[[], None] | None,
    cancel_event: CancellationSignal | None,
) -> list[str]:
    from . import tts_backends

    project_dir = project_dir.resolve()
    source = source.resolve()
    command_prefix = tts_backends._indextts_command(project)
    if command_prefix is None:
        raise SynthesisError(
            "找不到 IndexTTS2 独立运行环境。请把模型放在官方仓库的 checkpoints 目录，"
            "并在该仓库运行 uv sync。"
        )
    model_dir = Path(project.settings.tts_model_path).expanduser().resolve()
    if not (model_dir / "config.yaml").is_file():
        raise SynthesisError(f"IndexTTS2 模型目录缺少 config.yaml：{model_dir}")
    tts_dir = project_dir / "chinese"
    manifest = tts_dir / ".indextts2_batch.jsonl"
    tasks: list[tuple[Sentence, VoiceReference, Path]] = []
    lines: list[str] = []
    settings = project.settings
    use_custom_worker = (
        len(command_prefix) >= 3
        and command_prefix[-2:] == ["-m", "indextts.cli_v2"]
        and Path(__file__).with_name("indextts2_worker.py").is_file()
    )
    for sentence in pending:
        check_cancelled(cancel_event)
        reference = tts_backends.prepare_index_speaker_reference(
            project, project_dir, source, sentence
        )
        emotion = tts_backends.prepare_index_emotion_reference(
            project,
            project_dir,
            source,
            sentence,
            reference,
        )
        output = (tts_dir / f".{sentence.id}.indextts2.tmp.wav").resolve()
        output.unlink(missing_ok=True)
        emo_weight = effective_index_emotion_weight(project, sentence)
        payload: dict[str, Any] = {
            "text": spoken_text(settings, sentence.zh_text),
            "voice": str(reference.path.resolve()),
            "output": str(output),
        }
        if use_custom_worker:
            payload.update(
                {
                    "id": sentence.id,
                    "sentence_id": sentence.id,
                    "seed": attempt_seed(settings.random_seed, sentence),
                    "do_sample": settings.tts_index25_do_sample,
                    "temperature": settings.tts_temperature,
                    "top_p": settings.tts_top_p,
                    "top_k": settings.tts_index25_top_k,
                    "num_beams": settings.tts_index25_num_beams,
                    "repetition_penalty": settings.tts_index25_repetition_penalty,
                    "length_penalty": settings.tts_index25_length_penalty,
                    "max_mel_tokens": settings.tts_index25_max_mel_tokens,
                    "interval_silence_ms": settings.tts_index25_interval_silence_ms,
                    "max_text_tokens": settings.tts_index25_max_text_tokens,
                    "emotion_weight": emo_weight,
                }
            )
            target_dur = target_duration_for_sentence(project, sentence)
            if target_dur is not None:
                payload["target_duration_seconds"] = target_dur
        if settings.tts_index_emotion_source == "text":
            payload["emotion_text"] = settings.tts_index_emo_text.strip() or spoken_text(
                settings, sentence.zh_text
            )
            payload["emotion_weight"] = emo_weight
        elif settings.tts_index_emotion_source == "vector":
            payload["emotion_vector"] = settings.tts_index25_emotion_vector
            payload["emotion_weight"] = emo_weight
        elif emotion is not None:
            payload["emotion_audio"] = str(emotion.path.resolve())
            payload["emotion_weight"] = emo_weight
        lines.append(json.dumps(payload, ensure_ascii=False))
        tasks.append((sentence, reference, output))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if use_custom_worker:
        worker_script = Path(__file__).with_name("indextts2_worker.py").resolve()
        command = [
            command_prefix[0],
            str(worker_script),
            "--batch-file",
            str(manifest),
            "--model-dir",
            str(model_dir),
            "--device",
            settings.tts_device,
            "--fp16" if settings.tts_index_use_fp16 else "--no-fp16",
        ]
    else:
        command = [
            *command_prefix,
            "batch",
            "--batch-file",
            str(manifest),
            "--model-dir",
            str(model_dir),
            "--device",
            settings.tts_device,
            "--fp16" if settings.tts_index_use_fp16 else "--no-fp16",
            "--force",
        ]
    tts_backends.logger.info(
        "IndexTTS2 开始：device=%s fp16=%s sentences=%d",
        project.settings.tts_device,
        project.settings.tts_index_use_fp16,
        len(tasks),
    )
    if progress:
        progress(f"IndexTTS2 独立环境批量生成 {len(tasks)} 句", 0, len(tasks))
    output_tail: deque[str] = deque(maxlen=200)
    return_code = -1
    committed: set[str] = set()
    batch_error: SynthesisError | None = None
    process: subprocess.Popen[str] | None = None
    cancelled: OperationCancelledError | None = None
    try:
        env = tts_backends.isolated_runtime_environment("index-tts")
        if use_custom_worker:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            runtime_source = str(model_dir.parent)
            existing_pythonpath = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (runtime_source, existing_pythonpath) if part
            )
        if (
            use_custom_worker
            and tts_backends.daemon_pool_active()
            and tts_backends.subprocess.Popen is tts_backends._DEFAULT_SUBPROCESS_POPEN
        ):
            daemon_cmd = [
                command[0],
                command[1],
                "--daemon",
                *command[4:],
            ]

            def _on_v2_line(line: str) -> bool:
                if "Generated:" not in line:
                    return False
                for sentence, reference, temporary in tasks:
                    if str(temporary) in line and sentence.id not in committed:
                        tts_backends._commit_clip(
                            project, project_dir, sentence, reference, temporary
                        )
                        committed.add(sentence.id)
                        if on_sentence:
                            on_sentence()
                return True

            # A worker that dies part-way (crash, out of memory) is restarted once and
            # only the lines it never delivered are resubmitted.
            return_code = tts_backends.run_resident_with_one_restart(
                manifest=manifest,
                entries=[
                    (sentence.id, line) for (sentence, _, _), line in zip(tasks, lines, strict=True)
                ],
                committed=committed,
                label="IndexTTS2",
                progress=progress,
                worker_id="indextts2",
                daemon_command=daemon_cmd,
                cwd=model_dir.parent,
                env=env,
                timeout_seconds=project.settings.tts_timeout_seconds,
                cancel_event=cancel_event,
                on_line=_on_v2_line,
                output_tail=output_tail,
            )
            check_cancelled(cancel_event)
        else:
            process = tts_backends.subprocess.Popen(
                command,
                start_new_session=os.name != "nt",
                cwd=model_dir.parent if use_custom_worker else None,
                stdout=tts_backends.subprocess.PIPE,
                stderr=tts_backends.subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            register_process(process, cancel_event)
            messages: queue.Queue[str | None] = queue.Queue()

            def read_output() -> None:
                assert process is not None and process.stdout is not None
                try:
                    for line in process.stdout:
                        messages.put(line)
                finally:
                    messages.put(None)

            reader = threading.Thread(target=read_output, name="indextts-output", daemon=True)
            reader.start()
            deadline = time.monotonic() + project.settings.tts_timeout_seconds
            generated_count = 0
            stream_closed = False
            started_at = time.monotonic()
            last_heartbeat = started_at
            while not stream_closed:
                check_cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    terminate_process_tree(process)
                    raise SynthesisError(
                        f"IndexTTS2 批量生成超过 {project.settings.tts_timeout_seconds:g} 秒/句，"
                        "已停止独立进程。"
                    )
                try:
                    line = messages.get(timeout=0.25)
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        break
                    now = time.monotonic()
                    if progress and now - last_heartbeat >= 2.0:
                        elapsed = int(now - started_at)
                        progress(
                            f"IndexTTS2 正在加载模型/生成，已等待 {elapsed} 秒"
                            f"（完成 {generated_count}/{len(tasks)} 句）",
                            generated_count,
                            len(tasks),
                        )
                        last_heartbeat = now
                    continue
                if line is None:
                    stream_closed = True
                    continue
                output_tail.append(line)
                if "Generated:" in line:
                    generated_count = min(len(tasks), generated_count + 1)
                    deadline = time.monotonic() + project.settings.tts_timeout_seconds
                    for sentence, reference, temporary in tasks:
                        if str(temporary) in line and sentence.id not in committed:
                            tts_backends._commit_clip(
                                project, project_dir, sentence, reference, temporary
                            )
                            committed.add(sentence.id)
                            if on_sentence:
                                on_sentence()
                    if progress:
                        progress(
                            f"IndexTTS2 已生成 {generated_count}/{len(tasks)} 句",
                            generated_count,
                            len(tasks),
                        )
            while process.poll() is None:
                check_cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    terminate_process_tree(process)
                    raise SynthesisError("IndexTTS2 退出超时；已保留完成的逐句缓存。")
                with suppress(tts_backends.subprocess.TimeoutExpired):
                    process.wait(timeout=0.2)
            return_code = process.wait()
            reader.join(timeout=1)
            check_cancelled(cancel_event)
    except OperationCancelledError as exc:
        cancelled = exc
    except SynthesisError as exc:
        batch_error = exc
    except OSError as exc:
        raise SynthesisError(f"无法启动 IndexTTS2 独立环境：{exc}") from exc
    finally:
        if process is not None:
            unregister_process(process, cancel_event)
            if process.poll() is None:
                terminate_process_tree(process)
        manifest.unlink(missing_ok=True)

    detail = tts_backends.readable_output_tail(output_tail)
    failures: list[str] = []
    for index, (sentence, reference, temporary) in enumerate(tasks, start=1):
        output = tts_backends._cache_output(project, sentence, project_dir)
        if sentence.id in committed:
            continue
        if batch_error is not None:
            temporary.unlink(missing_ok=True)
            continue
        if cancelled is not None and not temporary.is_file():
            continue
        try:
            if not temporary.is_file():
                raise SynthesisError(
                    f"IndexTTS2 未生成本句（批处理退出码 {return_code}）：{detail}"
                )
            frame_count, sample_rate = tts_backends._validate_output(temporary)
            temporary.replace(output)
            sentence.reference_file = str(reference.path)
            sentence.tts_file = str(output.relative_to(project_dir))
            sentence.tts_duration_seconds = frame_count / sample_rate
            sentence.tts_cache_key = tts_cache_key(project, sentence)
            sentence.status = "synthesized"
            sentence.error = None
        except Exception as exc:
            sentence.status = "error"
            sentence.error = str(exc)
            failures.append(f"{sentence.id}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)
        if on_sentence:
            on_sentence()
        if progress:
            progress(f"已处理 {index}/{len(tasks)} 句 IndexTTS2 配音", index, len(tasks))
    if batch_error is not None:
        raise batch_error
    if cancelled is not None:
        raise cancelled
    if return_code != 0 and not failures:
        raise SynthesisError(f"IndexTTS2 批处理失败（{return_code}）：{detail}")
    tts_backends.logger.info(
        "IndexTTS2 完成：device=%s sentences=%d failures=%d",
        project.settings.tts_device,
        len(tasks),
        len(failures),
    )
    return failures
