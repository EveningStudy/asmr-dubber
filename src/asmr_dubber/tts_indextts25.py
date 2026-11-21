from __future__ import annotations

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


def _indextts25_command(project: DubProject) -> list[str] | None:
    model_dir = Path(project.settings.tts_index25_model_path).expanduser().resolve()
    runtime_root = model_dir.parent
    python_candidates = (
        runtime_root / ".venv" / "Scripts" / "python.exe",
        runtime_root / ".venv" / "bin" / "python",
    )
    python = next((candidate for candidate in python_candidates if candidate.is_file()), None)
    worker = Path(__file__).with_name("indextts25_worker.py").resolve()
    if python is None or not worker.is_file():
        return None
    return [str(python), str(worker)]


def _synthesize_indextts25_batch(
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
    command_prefix = tts_backends._indextts25_command(project)
    if command_prefix is None:
        raise SynthesisError("找不到 IndexTTS-2.5 独立运行环境。请在“设备与模型”中安装或修复。")
    model_dir = Path(project.settings.tts_index25_model_path).expanduser().resolve()
    config = Path(project.settings.tts_index25_config_path).expanduser().resolve()
    if config != model_dir / "config.yaml" or not config.is_file():
        raise SynthesisError(f"IndexTTS-2.5 配置文件无效：{config}")
    tts_dir = project_dir / "chinese"
    manifest = tts_dir / ".indextts25_batch.jsonl"
    tasks: list[tuple[Sentence, VoiceReference, Path]] = []
    lines: list[str] = []
    settings = project.settings
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
        output = (tts_dir / f".{sentence.id}.indextts25.tmp.wav").resolve()
        output.unlink(missing_ok=True)
        emo_weight = effective_index_emotion_weight(project, sentence)
        payload: dict[str, Any] = {
            "id": sentence.id,
            "sentence_id": sentence.id,
            "text": spoken_text(project.settings, sentence.zh_text),
            "voice": str(reference.path.resolve()),
            "output": str(output),
            "language": settings.tts_target_language,
            "emotion_weight": emo_weight,
            "use_random": settings.tts_index25_use_random,
            "interval_silence_ms": settings.tts_index25_interval_silence_ms,
            "max_text_tokens": settings.tts_index25_max_text_tokens,
            "duration_factor": settings.tts_index25_duration_factor,
            "text_normalization": settings.tts_index25_text_normalization,
            "do_sample": settings.tts_index25_do_sample,
            "top_p": settings.tts_index25_top_p,
            "top_k": settings.tts_index25_top_k,
            "temperature": settings.tts_index25_temperature,
            "length_penalty": settings.tts_index25_length_penalty,
            "num_beams": settings.tts_index25_num_beams,
            "repetition_penalty": settings.tts_index25_repetition_penalty,
            "max_mel_tokens": settings.tts_index25_max_mel_tokens,
            "seed": attempt_seed(settings.random_seed, sentence),
        }
        target_dur = target_duration_for_sentence(project, sentence)
        if target_dur is not None:
            payload["target_duration_seconds"] = target_dur
        if settings.tts_index_emotion_source == "text":
            payload["emotion_text"] = settings.tts_index_emo_text.strip() or spoken_text(
                project.settings, sentence.zh_text
            )
        elif settings.tts_index_emotion_source == "vector":
            payload["emotion_vector"] = settings.tts_index25_emotion_vector
        elif emotion is not None:
            payload["emotion_audio"] = str(emotion.path.resolve())
        lines.append(json.dumps(payload, ensure_ascii=False))
        tasks.append((sentence, reference, output))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        *command_prefix,
        "--batch-file",
        str(manifest),
        "--model-dir",
        str(model_dir),
        "--device",
        settings.tts_device,
        "--bf16" if settings.tts_index25_use_bf16 else "--no-bf16",
        "--cuda-kernel" if settings.tts_index25_use_cuda_kernel else "--no-cuda-kernel",
        "--deepspeed" if settings.tts_index25_use_deepspeed else "--no-deepspeed",
        "--accel" if settings.tts_index25_use_accel else "--no-accel",
        "--torch-compile" if settings.tts_index25_use_torch_compile else "--no-torch-compile",
    ]
    tts_backends.logger.info(
        "IndexTTS-2.5 开始：device=%s bf16=%s language=%s sentences=%d",
        settings.tts_device,
        settings.tts_index25_use_bf16,
        settings.tts_index25_language,
        len(tasks),
    )
    if progress:
        progress(f"IndexTTS-2.5 独立环境批量生成 {len(tasks)} 句", 0, len(tasks))
    output_tail: deque[str] = deque(maxlen=200)
    return_code = -1
    committed: set[str] = set()
    line_failures: list[str] = []
    batch_error: SynthesisError | None = None
    process: subprocess.Popen[str] | None = None
    cancelled: OperationCancelledError | None = None
    try:
        env = tts_backends.isolated_runtime_environment("index-tts-2.5")
        if os.name == "nt":
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        runtime_source = str(model_dir.parent)
        existing_pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (runtime_source, existing_pythonpath) if part
        )
        if (
            tts_backends.daemon_pool_active()
            and tts_backends.subprocess.Popen is tts_backends._DEFAULT_SUBPROCESS_POPEN
        ):
            daemon_cmd = [
                *command_prefix,
                "--daemon",
                *command[len(command_prefix) + 2 :],
            ]

            def _on_v25_line(line: str) -> bool:
                if not line.startswith("Generated:"):
                    return False
                generated_id = line.partition(":")[2].strip()
                for sentence, reference, temporary in tasks:
                    if generated_id == sentence.id and sentence.id not in committed:
                        try:
                            tts_backends._commit_clip(
                                project, project_dir, sentence, reference, temporary
                            )
                        except SynthesisError as exc:
                            sentence.status = "error"
                            sentence.error = str(exc)
                            line_failures.append(f"{sentence.id}: {exc}")
                            temporary.unlink(missing_ok=True)
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
                label="IndexTTS-2.5",
                progress=progress,
                worker_id="indextts25",
                daemon_command=daemon_cmd,
                cwd=model_dir.parent,
                env=env,
                timeout_seconds=settings.tts_timeout_seconds,
                cancel_event=cancel_event,
                on_line=_on_v25_line,
                output_tail=output_tail,
            )
            check_cancelled(cancel_event)
        else:
            process = tts_backends.subprocess.Popen(
                command,
                start_new_session=os.name != "nt",
                cwd=model_dir.parent,
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

            reader = threading.Thread(target=read_output, name="indextts25-output", daemon=True)
            reader.start()
            deadline = time.monotonic() + settings.tts_timeout_seconds
            generated_count = 0
            stream_closed = False
            started_at = time.monotonic()
            last_heartbeat = started_at
            while not stream_closed:
                check_cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    terminate_process_tree(process)
                    raise SynthesisError(
                        f"IndexTTS-2.5 批量生成超过 {settings.tts_timeout_seconds:g} 秒/句，"
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
                            f"IndexTTS-2.5 正在加载模型/生成，已等待 {elapsed} 秒"
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
                if line.startswith("Generated:"):
                    generated_count = min(len(tasks), generated_count + 1)
                    deadline = time.monotonic() + settings.tts_timeout_seconds
                    generated_id = line.partition(":")[2].strip()
                    for sentence, reference, temporary in tasks:
                        if generated_id == sentence.id and sentence.id not in committed:
                            try:
                                tts_backends._commit_clip(
                                    project, project_dir, sentence, reference, temporary
                                )
                            except SynthesisError as exc:
                                sentence.status = "error"
                                sentence.error = str(exc)
                                line_failures.append(f"{sentence.id}: {exc}")
                                temporary.unlink(missing_ok=True)
                            committed.add(sentence.id)
                            if on_sentence:
                                on_sentence()
                    if progress:
                        progress(
                            f"IndexTTS-2.5 已生成 {generated_count}/{len(tasks)} 句",
                            generated_count,
                            len(tasks),
                        )
            while process.poll() is None:
                check_cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    terminate_process_tree(process)
                    raise SynthesisError("IndexTTS-2.5 退出超时；已保留完成的逐句缓存。")
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
        raise SynthesisError(f"无法启动 IndexTTS-2.5 独立环境：{exc}") from exc
    finally:
        if process is not None:
            unregister_process(process, cancel_event)
            if process.poll() is None:
                terminate_process_tree(process)
        manifest.unlink(missing_ok=True)

    detail = tts_backends.readable_output_tail(output_tail)
    if (
        batch_error is None
        and cancelled is None
        and return_code != 0
        and not committed
        and not any(temporary.is_file() for _, _, temporary in tasks)
    ):
        for _, _, temporary in tasks:
            temporary.unlink(missing_ok=True)
        if "out of memory" in detail.casefold():
            raise SynthesisError(
                f"IndexTTS-2.5 模型初始化失败：显卡显存或系统虚拟内存不足"
                f"（退出码 {return_code}）。\n{detail}"
            )
        raise SynthesisError(
            f"IndexTTS-2.5 模型初始化或批处理启动失败（退出码 {return_code}）：{detail}"
        )
    failures: list[str] = list(line_failures)
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
                    f"IndexTTS-2.5 未生成本句（批处理退出码 {return_code}）：{detail}"
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
            progress(
                f"已处理 {index}/{len(tasks)} 句 IndexTTS-2.5 配音",
                index,
                len(tasks),
            )
    if batch_error is not None:
        raise batch_error
    if cancelled is not None:
        raise cancelled
    if return_code != 0 and not failures:
        raise SynthesisError(f"IndexTTS-2.5 批处理失败（{return_code}）：{detail}")
    tts_backends.logger.info(
        "IndexTTS-2.5 完成：device=%s sentences=%d failures=%d",
        settings.tts_device,
        len(tasks),
        len(failures),
    )
    return failures
