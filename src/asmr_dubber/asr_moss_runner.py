from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .errors import AsmrDubberError
from .languages import SpeechSourceLanguage
from .models import ProjectSettings, Sentence
from .task_control import CancellationSignal, check_cancelled

Progress = Callable[[str, int, int], None]


def _transcribe_moss(
    analysis_audio: Path,
    settings: ProjectSettings,
    progress: Progress | None,
    cancel_event: CancellationSignal | None = None,
    *,
    source_language: SpeechSourceLanguage = "ja",
) -> tuple[list[Sentence], str]:
    from . import asr, moss

    if not moss.runtime_python().is_file():
        raise AsmrDubberError("请先在模型页安装 MOSS-Transcribe-Diarize。")
    valid_moss_models = {moss.MODEL_ID, str(moss.model_directory()), ""}
    if settings.asr_model not in valid_moss_models:
        raise AsmrDubberError("MOSS 只支持经过校验的内置固定快照。")
    if not moss.complete_model(verify=True):
        raise AsmrDubberError("MOSS 模型缺失或校验失败；请在模型页重新下载。")
    check_cancelled(cancel_event)
    rate, windows = asr._audio_file_chunk_ranges(analysis_audio, settings.asr_moss_chunk_seconds)
    run_directory = asr.portable_home() / "temp" / "asr" / f"moss-{uuid.uuid4().hex}"
    run_directory.mkdir(parents=True)
    request_file = run_directory / "request.json"
    output_file = run_directory / "result.json"
    log_file = run_directory / "worker.log"
    request_file.write_text(
        json.dumps(
            {
                "audio": str(analysis_audio.resolve()),
                "model": str(moss.model_directory()),
                "sample_rate": rate,
                "windows": windows,
                "language": source_language,
                "settings": settings.model_dump(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    process = None
    try:
        with log_file.open("w", encoding="utf-8") as output:
            process = asr.subprocess.Popen(
                [
                    str(moss.runtime_python()),
                    str(Path(__file__).with_name("moss_worker.py")),
                    "--request",
                    str(request_file),
                    "--output",
                    str(output_file),
                ],
                cwd=moss.runtime_root(),
                env=asr.isolated_runtime_environment("moss-transcribe-diarize"),
                stdout=output,
                stderr=asr.subprocess.STDOUT,
                creationflags=asr.subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            asr.register_process(process, cancel_event)
            deadline = time.monotonic() + settings.asr_moss_timeout_seconds
            current_window = 0
            started_at = time.monotonic()
            last_heartbeat = started_at
            last_message = f"MOSS 初始化中（共 {len(windows)} 段）"
            with log_file.open(encoding="utf-8", errors="replace") as reader:
                while process.poll() is None:
                    check_cancelled(cancel_event)
                    if time.monotonic() > deadline:
                        raise AsmrDubberError("MOSS 推理超时；请缩短分块或增加 MOSS 超时。")
                    saw_progress = False
                    for line in reader:
                        try:
                            message = json.loads(line).get("progress")
                        except (json.JSONDecodeError, AttributeError):
                            continue
                        if message:
                            saw_progress = True
                            last_message = str(message)
                            match = re.search(r"MOSS\s+(?:识别)?分块\s+(\d+)", last_message)
                            if match:
                                current_window = max(
                                    current_window,
                                    min(len(windows), int(match.group(1)) - 1),
                                )
                            if progress:
                                progress(last_message, current_window, len(windows))
                    now = time.monotonic()
                    if saw_progress:
                        last_heartbeat = now
                    elif progress and now - last_heartbeat >= 2.0:
                        progress(
                            f"{last_message} · 仍在处理，已运行 {now - started_at:.0f} 秒",
                            current_window,
                            len(windows),
                        )
                        last_heartbeat = now
                    time.sleep(0.1)
        check_cancelled(cancel_event)
        if process.returncode or not output_file.is_file():
            detail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise AsmrDubberError(f"MOSS 识别失败（退出码 {process.returncode}）：{detail}")
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        sentences = []
        for segment in payload["segments"]:
            # Segment-level timestamps do not justify inventing word timings.
            # Keep speaker turns intact instead of redistributing their duration.
            token = asr._finite_token(segment["text"], segment["start"], segment["end"])
            if token:
                sentences.append(
                    Sentence(
                        id=f"s{len(sentences) + 1:06d}",
                        source_text=token.text,
                        start_seconds=token.start_seconds,
                        end_seconds=token.end_seconds,
                        speaker_id=segment["speaker"],
                    )
                )
        if not sentences and asr._REVIEW_MODELS.get() is None:
            raise AsmrDubberError("MOSS 未识别到带有效时间戳的语音；原有句子表已保留。")
        return sentences, source_language
    finally:
        if process is not None:
            asr.terminate_process_tree(process)
            asr.unregister_process(process, cancel_event)
        shutil.rmtree(run_directory, ignore_errors=True)
