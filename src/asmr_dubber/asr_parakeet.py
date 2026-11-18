from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import soundfile as sf

from .errors import AsmrDubberError
from .filtering import has_speakable_text
from .languages import SpeechSourceLanguage
from .models import ProjectSettings, Sentence
from .segmentation import TimedToken
from .task_control import CancellationSignal, check_cancelled

Progress = Callable[[str, int, int], None]


def _crispasr_executable() -> Path:
    from . import asr

    name = "crispasr.exe" if asr.current_platform().is_windows else "crispasr"
    return asr.portable_home() / "runtimes" / "crispasr" / "bin" / name


def _parakeet_model_path(model_id: str) -> Path:
    from . import asr

    directory = asr.portable_home() / "models" / "parakeet"
    filenames = {
        "nvidia/parakeet-tdt_ctc-0.6b-ja": "parakeet-tdt-0.6b-ja.gguf",
        (
            "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
        ): "parakeet-ctc-1.1b-ja-f16.gguf",
    }
    filename = filenames.get(model_id)
    if filename is None:
        raise AsmrDubberError(f"不支持的 Parakeet 日语模型：{model_id}")
    return directory / filename


def _parakeet_input_path(analysis_audio: Path, run_directory: Path) -> Path:
    """Stage a non-ASCII Windows source inside the portable ASR run directory."""
    from . import asr

    resolved = analysis_audio.resolve()
    if not asr.current_platform().is_windows or str(resolved).isascii():
        return resolved
    staged = run_directory / "input.wav"
    try:
        # This is instant when the project and portable runtime share a volume.
        os.link(resolved, staged)
    except OSError:
        # Cross-volume projects cannot be hard-linked.  The temporary copy is
        # still isolated from the source and is removed with run_directory.
        shutil.copyfile(resolved, staged)
    return staged


def _transcribe_parakeet(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
    window_inputs: list[Path] | None = None,
    window_results: dict[str, list[Sentence]] | None = None,
    window_errors: dict[str, str] | None = None,
) -> tuple[list[Sentence], str]:
    from . import asr

    check_cancelled(cancel_event)
    if source_language != "ja":
        raise AsmrDubberError("当前安装的 Parakeet 模型仅支持日语；英语项目请使用 Faster-Whisper。")
    executable = asr._crispasr_executable()
    model_path = asr._parakeet_model_path(settings.asr_model)
    if not executable.is_file() or not model_path.is_file():
        raise AsmrDubberError(
            "Parakeet/CrispASR 未安装完整；请在“设备与模型”页选择 Parakeet 后点击安装。"
        )
    portable_root = asr.portable_home().resolve()
    run_directory = portable_root / "temp" / "asr" / f"parakeet-{uuid.uuid4().hex}"
    run_directory.mkdir(parents=True, exist_ok=False)
    cache_directory = portable_root / "cache" / "crispasr"
    cache_directory.mkdir(parents=True, exist_ok=True)
    native_input = asr._parakeet_input_path(analysis_audio, run_directory)
    use_portable_relative_paths = (
        asr.current_platform().is_windows and not str(portable_root).isascii()
    )
    command_directory = portable_root if use_portable_relative_paths else asr.PROJECT_ROOT

    def command_path(path: Path) -> str:
        resolved = path.resolve()
        if not use_portable_relative_paths:
            return str(resolved)
        try:
            relative = resolved.relative_to(portable_root)
        except ValueError:
            return str(resolved)
        rendered = str(relative)
        return rendered if rendered.isascii() else str(resolved)

    base_command = [
        str(executable),
        "--backend",
        "parakeet",
        # Optional FireRedPunc must not be implicitly downloaded during ASR.
        "--no-punctuation",
        "--cache-dir",
        command_path(cache_directory),
        "-m",
        command_path(model_path),
        "-l",
        "ja",
        "-ojf",
        "-pp",
        "-t",
        str(max(1, min(os.cpu_count() or 4, settings.asr_batch_size * 4))),
    ]
    if settings.asr_model == "nvidia/parakeet-tdt_ctc-0.6b-ja":
        base_command.extend(("--parakeet-decoder", settings.asr_parakeet_decoder))
    if settings.asr_vad_mode == "backend":
        base_command.extend(
            (
                "--vad",
                "-vm",
                "silero",
                "-vsd",
                str(settings.asr_vad_min_silence_ms),
            )
        )
    if settings.asr_initial_prompt:
        base_command.extend(("--hotwords", settings.asr_initial_prompt))
    if not settings.asr_device.startswith("cuda"):
        base_command.append("--no-gpu")
    environment = asr.isolated_runtime_environment("crispasr")
    environment["CRISPASR_CACHE_DIR"] = command_path(cache_directory)
    # FastConformer attention still grows with the input duration in CrispASR
    # 0.8.x. Supplying a multi-hour file directly can therefore request tens of
    # GiB even when --chunk-seconds is present. CrispASR accepts several input
    # files in one invocation, so create bounded files and pass all of them to
    # one process: peak memory stays bounded while the model is loaded once.
    configured_chunk_seconds = settings.asr_chunk_seconds or asr._PARAKEET_AUTO_CHUNK_SECONDS
    duration_seconds = float(sf.info(native_input).duration)
    if duration_seconds <= 0:
        raise AsmrDubberError("Parakeet 输入音频为空。")

    def prepare_chunks(
        chunk_seconds: float,
        attempt: int,
    ) -> tuple[int, list[tuple[Path, int, int]]]:
        if window_inputs is not None:
            chunk_directory = run_directory / f"chunks-{attempt:02d}"
            chunk_directory.mkdir(parents=True, exist_ok=False)
            window_chunks = []
            sample_rate = 16000
            for index, path in enumerate(window_inputs, start=1):
                check_cancelled(cancel_event)
                chunk_file = chunk_directory / f"chunk-{index:06d}.wav"
                data, sample_rate = sf.read(path, dtype="float32")
                sf.write(chunk_file, data, sample_rate, subtype="PCM_16")
                window_chunks.append((chunk_file, 0, len(data)))
            return sample_rate, window_chunks
        sample_rate, ranges = asr._audio_file_chunk_ranges(native_input, chunk_seconds)
        chunk_directory = run_directory / f"chunks-{attempt:02d}"
        chunk_directory.mkdir(parents=True, exist_ok=False)
        chunks: list[tuple[Path, int, int]] = []
        with sf.SoundFile(native_input) as source_audio:
            for index, (start_sample, end_sample) in enumerate(ranges, start=1):
                check_cancelled(cancel_event)
                chunk_file = chunk_directory / f"chunk-{index:06d}.wav"
                source_audio.seek(start_sample)
                waveform = source_audio.read(
                    end_sample - start_sample,
                    dtype="float32",
                    always_2d=False,
                )
                sf.write(chunk_file, waveform, sample_rate, subtype="PCM_16")
                chunks.append((chunk_file, start_sample, end_sample))
        return sample_rate, chunks

    def command_batches(chunk_files: list[Path]) -> list[list[Path]]:
        # CreateProcess has a 32,767-character command-line ceiling on Windows.
        # Keep margin for quoting and environment wrappers. Typical recordings
        # still fit in one batch; only extremely long ones reload per batch.
        limit = 28_000 if asr.current_platform().is_windows else 120_000
        base_length = sum(len(argument) + 3 for argument in base_command)
        batches: list[list[Path]] = []
        current: list[Path] = []
        current_length = base_length
        for chunk_file in chunk_files:
            argument_length = len(command_path(chunk_file)) + 3
            if current and current_length + argument_length > limit:
                batches.append(current)
                current = []
                current_length = base_length
            current.append(chunk_file)
            current_length += argument_length
        if current:
            batches.append(current)
        return batches

    def run_batch(
        command: list[str],
        *,
        batch_index: int,
        batch_count: int,
        completed_before: int,
        total_chunks: int,
    ) -> tuple[int, list[str], int]:
        output_lines: list[str] = []
        process = asr.subprocess.Popen(
            command,
            start_new_session=os.name != "nt",
            cwd=command_directory,
            env=environment,
            stdout=asr.subprocess.PIPE,
            stderr=asr.subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        asr.register_process(process, cancel_event)
        stdout = process.stdout
        assert stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output,
            name="parakeet-output-reader",
            daemon=True,
        )
        reader.start()
        started_at = time.monotonic()
        last_activity = started_at
        last_heartbeat = started_at
        output_closed = False
        completed_in_batch = 0
        try:
            while not output_closed:
                check_cancelled(cancel_event)
                now = time.monotonic()
                idle_seconds = now - last_activity
                if idle_seconds >= settings.asr_timeout_seconds:
                    asr.terminate_process_tree(process)
                    raise AsmrDubberError(
                        f"Parakeet 连续 {settings.asr_timeout_seconds:g} 秒没有任何响应，"
                        "已安全停止。可在设置中增加“连续无响应超时”。"
                    )
                try:
                    line = output_queue.get(timeout=0.25)
                except queue.Empty:
                    if progress and now - last_heartbeat >= 2:
                        elapsed = now - started_at
                        progress(
                            f"Parakeet 正在处理有界分块 {completed_before + completed_in_batch}/"
                            f"{total_chunks}（批次 {batch_index}/{batch_count}，"
                            f"已运行 {elapsed:.0f} 秒）",
                            completed_before + completed_in_batch,
                            total_chunks,
                        )
                        last_heartbeat = now
                    if process.poll() is not None and not reader.is_alive():
                        break
                    continue
                if line is None:
                    output_closed = True
                    continue
                last_activity = time.monotonic()
                value = line.strip()
                if not value:
                    continue
                output_lines.append(value)
                if "transcribed " in value.casefold() and " audio in " in value.casefold():
                    completed_in_batch += 1
                    if progress:
                        progress(
                            f"Parakeet 已完成 {completed_before + completed_in_batch}/"
                            f"{total_chunks} 个分块",
                            completed_before + completed_in_batch,
                            total_chunks,
                        )
                if progress and ("progress" in value.lower() or "%" in value):
                    progress(
                        f"Parakeet：{value[-180:]}",
                        completed_before + completed_in_batch,
                        total_chunks,
                    )
            reader.join(timeout=1)
            return_code = process.wait()
        finally:
            asr.unregister_process(process, cancel_event)
            if process.poll() is None:
                asr.terminate_process_tree(process)
        check_cancelled(cancel_event)
        asr.logger.info(
            "Parakeet 批次结束：batch=%s/%s return_code=%s elapsed=%.1fs",
            batch_index,
            batch_count,
            return_code,
            time.monotonic() - started_at,
        )
        return return_code, output_lines, completed_in_batch

    def diagnostic_text(lines: list[str]) -> str:
        selected = [*lines[:12], "…", *lines[-30:]] if len(lines) > 42 else lines
        return "\n".join(selected).strip()

    def is_memory_failure(return_code: int, detail: str) -> bool:
        lowered = detail.casefold()
        return return_code in {3221225477, -1073741819} or any(
            marker in lowered
            for marker in (
                "out of memory",
                "cudaerroroutofmemory",
                "cudamalloc failed",
                "failed to allocate cuda",
                "failed to allocate cuda0",
            )
        )

    try:
        chunk_seconds = float(configured_chunk_seconds)
        attempt = 0
        while True:
            attempt += 1
            sample_rate, chunks = prepare_chunks(chunk_seconds, attempt)
            batches = command_batches([chunk[0] for chunk in chunks])
            asr.logger.info(
                "Parakeet 开始：model=%s duration=%.1fs chunk=%.1fs chunks=%s batches=%s device=%s",
                settings.asr_model,
                duration_seconds,
                chunk_seconds,
                len(chunks),
                len(batches),
                settings.asr_device,
            )
            if progress:
                progress(
                    f"Parakeet 识别 {asr._clock(duration_seconds)} 音频："
                    f"{len(chunks)} 个显存有界分块，模型每批只加载一次",
                    0,
                    len(chunks),
                )

            completed = 0
            failed: tuple[int, str] | None = None
            for batch_index, batch in enumerate(batches, start=1):
                check_cancelled(cancel_event)
                command = [*base_command, *(command_path(path) for path in batch)]
                return_code, output_lines, reported_completed = run_batch(
                    command,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    completed_before=completed,
                    total_chunks=len(chunks),
                )
                if return_code != 0:
                    failed = (return_code, diagnostic_text(output_lines))
                    break
                completed += max(reported_completed, len(batch))

            if failed is not None:
                return_code, detail = failed
                minimum_chunk_seconds = 15.0
                if is_memory_failure(return_code, detail) and chunk_seconds > minimum_chunk_seconds:
                    next_chunk_seconds = max(minimum_chunk_seconds, chunk_seconds / 2)
                    asr.logger.warning(
                        "Parakeet 显存不足，自动缩小分块：%.1fs -> %.1fs",
                        chunk_seconds,
                        next_chunk_seconds,
                    )
                    if progress:
                        progress(
                            f"当前分块需要的显存超过设备容量；自动从 {chunk_seconds:g} 秒"
                            f"缩小到 {next_chunk_seconds:g} 秒后重试",
                            0,
                            1,
                        )
                    chunk_seconds = next_chunk_seconds
                    continue
                suffix = (
                    "；已自动尝试最小 15 秒分块，模型本身仍超出当前可用显存。"
                    "可关闭其它占用显存的程序，或选择较小的 Parakeet 0.6B 模型。"
                    if is_memory_failure(return_code, detail)
                    else ""
                )
                raise AsmrDubberError(
                    f"Parakeet 识别失败（退出码 {return_code}）：{detail[:4000]}{suffix}"
                )

            all_tokens: list[TimedToken] = []
            full_text_parts: list[str] = []
            language = "ja"
            for window_index, (chunk_file, start_sample, _end_sample) in enumerate(chunks):
                check_cancelled(cancel_event)
                result_file = chunk_file.with_suffix(".json")
                if not result_file.is_file():
                    # CrispASR may return success without JSON for a fully silent
                    # input. Treat it as an empty chunk rather than a failure.
                    if window_results is not None and window_inputs is not None:
                        window_results[window_inputs[window_index].stem] = []
                    continue
                try:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise AsmrDubberError(
                        f"Parakeet 分块没有生成有效结果：{result_file.name}：{exc}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise AsmrDubberError(f"Parakeet JSON 顶层格式无效：{result_file.name}")
                tokens, full_text, chunk_language = asr._crispasr_payload_tokens(payload)
                if window_results is not None and window_inputs is not None:
                    identifier = window_inputs[window_index].stem
                    try:
                        window_results[identifier] = (
                            asr._finish_tokens(
                                tokens, full_text, chunk_language, settings, source_language
                            )[0]
                            if has_speakable_text(full_text)
                            else []
                        )
                    except AsmrDubberError as exc:
                        if window_errors is None:
                            raise
                        window_errors[identifier] = str(exc)
                    continue
                offset_seconds = start_sample / sample_rate
                all_tokens.extend(
                    TimedToken(
                        text=token.text,
                        start_seconds=token.start_seconds + offset_seconds,
                        end_seconds=token.end_seconds + offset_seconds,
                    )
                    for token in tokens
                )
                full_text_parts.append(full_text)
                if chunk_language:
                    language = chunk_language
            if window_results is not None:
                return [], language
            return asr._finish_tokens(
                all_tokens,
                "".join(full_text_parts),
                language,
                settings,
                source_language,
            )
    finally:
        shutil.rmtree(run_directory, ignore_errors=True)
        asr._cleanup_cuda()
