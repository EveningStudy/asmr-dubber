from __future__ import annotations

import logging
import subprocess as subprocess
import threading
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .api_contracts import (
    APIContractError as APIContractError,
)
from .api_contracts import (
    bearer_headers as bearer_headers,
)
from .api_contracts import (
    endpoint_url as endpoint_url,
)
from .api_contracts import (
    merge_extra_body as merge_extra_body,
)
from .api_contracts import (
    write_audio_response as write_audio_response,
)
from .api_logging import logged_http_client as logged_http_client
from .api_logging import safe_api_url
from .errors import OperationCancelledError, SynthesisError
from .filtering import has_speakable_text
from .model_registry import TTS_BACKENDS
from .models import DubProject, Sentence
from .platforms import isolated_runtime_environment as isolated_runtime_environment
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    unregister_cancel_callback,
)
from .task_control import (
    register_process as register_process,
)
from .task_control import (
    terminate_process_tree as terminate_process_tree,
)
from .task_control import (
    unregister_process as unregister_process,
)
from .tts import (
    attempt_seed as attempt_seed,
)
from .tts import (
    spoken_text as spoken_text,
)
from .tts import (
    target_duration_for_sentence as target_duration_for_sentence,
)
from .tts import (
    tts_cache_key,
    valid_tts_cache,
)
from .tts_daemon import (
    daemon_pool_active as daemon_pool_active,
)
from .tts_daemon import (
    readable_output_tail as readable_output_tail,
)
from .tts_daemon import (
    run_resident_indextts_batch as run_resident_indextts_batch,
)
from .tts_daemon import (
    run_resident_with_one_restart as run_resident_with_one_restart,
)
from .tts_http_runners import (
    _cosyvoice_runner as _cosyvoice_runner,
)
from .tts_http_runners import (
    _edge_tts_runner as _edge_tts_runner,
)
from .tts_http_runners import (
    _fish_runner as _fish_runner,
)
from .tts_http_runners import (
    _generic_tts_api_runner as _generic_tts_api_runner,
)
from .tts_http_runners import (
    _gpt_sovits_runner as _gpt_sovits_runner,
)
from .tts_http_runners import (
    _indextts_api_runner as _indextts_api_runner,
)
from .tts_http_runners import (
    _mimo_runner as _mimo_runner,
)
from .tts_http_runners import (
    _minimax_runner as _minimax_runner,
)
from .tts_indextts import (
    _indextts_cli as _indextts_cli,
)
from .tts_indextts import (
    _indextts_command as _indextts_command,
)
from .tts_indextts import (
    _load_indextts as _load_indextts,
)
from .tts_indextts import (
    _synthesize_indextts_cli_batch as _synthesize_indextts_cli_batch,
)
from .tts_indextts25 import (
    _indextts25_command as _indextts25_command,
)
from .tts_indextts25 import (
    _synthesize_indextts25_batch as _synthesize_indextts25_batch,
)
from .user_settings import saved_service_key as saved_service_key
from .voice_reference import (
    VoiceReference,
    prepare_index_emotion_reference,
    prepare_index_speaker_reference,
    prepare_voice_reference,
)
from .voice_reference import (
    effective_index_emotion_weight as effective_index_emotion_weight,
)

Progress = Callable[[str, int, int], None]
logger = logging.getLogger(__name__)
_DEFAULT_SUBPROCESS_POPEN = subprocess.Popen


def _cache_output(project: DubProject, sentence: Sentence, directory: Path) -> Path:
    return directory / "chinese" / f"{sentence.id}-{tts_cache_key(project, sentence)}.wav"


def _commit_clip(
    project: DubProject,
    directory: Path,
    sentence: Sentence,
    reference: VoiceReference,
    temporary: Path,
) -> None:
    frames, rate = _validate_output(temporary)
    output = _cache_output(project, sentence, directory)
    temporary.replace(output)
    sentence.reference_file = str(reference.path)
    sentence.tts_file = output.relative_to(directory).as_posix()
    sentence.tts_duration_seconds = frames / rate
    sentence.tts_cache_key = tts_cache_key(project, sentence)
    sentence.status = "synthesized"
    sentence.error = None


def _selected_sentences(project: DubProject, sentence_ids: Iterable[str] | None) -> list[Sentence]:
    requested = set(sentence_ids) if sentence_ids is not None else None
    return [
        sentence
        for sentence in project.sentences
        if sentence.enabled
        and sentence.chinese_audio_enabled
        and has_speakable_text(sentence.zh_text)
        and (requested is None or sentence.id in requested)
    ]


def _validate_output(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise SynthesisError(f"TTS（语音合成）没有生成输出文件：{path}")
    try:
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise SynthesisError("TTS（语音合成）输出为空或采样率无效。")
        if info.channels == 1 and info.format in {"WAV", "RF64"} and info.subtype == "FLOAT":
            # Most local backends already emit the mixer's canonical format.
            # Validate it in bounded blocks without reading and rewriting the
            # whole sentence a second time.
            with sf.SoundFile(path) as audio:
                while True:
                    block = audio.read(262_144, dtype="float32", always_2d=False)
                    if block.size == 0:
                        break
                    if not np.isfinite(block).all():
                        raise SynthesisError("TTS（语音合成）输出包含无效采样。")
            return int(info.frames), int(info.samplerate)
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        if isinstance(exc, SynthesisError):
            raise
        raise SynthesisError(f"TTS（语音合成）输出不是可读取的音频：{path}: {exc}") from exc
    if waveform.size == 0 or sample_rate <= 0 or not np.isfinite(waveform).all():
        raise SynthesisError("TTS（语音合成）输出为空或包含无效采样。")
    mono = np.asarray(np.mean(waveform, axis=1, dtype=np.float32), dtype=np.float32)
    # Generated audio is canonicalized for the mixer; no loudness normalization is applied here.
    sf.write(path, mono, sample_rate, format="WAV", subtype="FLOAT")
    return int(mono.size), int(sample_rate)


def _require_reference_text(project: DubProject, reference: VoiceReference) -> None:
    spec = TTS_BACKENDS[project.settings.tts_backend]
    if spec.reference_text == "required" and not reference.text.strip():
        raise SynthesisError(
            f"{spec.label} 的高质量克隆需要参考音频对应文本。"
            "请在设置 → TTS（语音合成）→ 外部参考文本中填写，或改用项目内参考句。"
        )


def _uses_reference_audio(project: DubProject) -> bool:
    if project.settings.tts_backend == "mimo_tts":
        return project.settings.tts_model == "mimo-v2.5-tts-voiceclone"
    return TTS_BACKENDS[project.settings.tts_backend].reference_audio


def _empty_reference() -> VoiceReference:
    return VoiceReference(path=Path(), text="", identity="unused", language="zh")


def _runner(
    project: DubProject,
    api_key: str | None = None,
) -> tuple[Any, Callable[[], None]]:
    backend = project.settings.tts_backend
    if backend == "indextts2_5":
        raise SynthesisError("IndexTTS-2.5 独立运行环境不可用；请在“设备与模型”中安装或修复。")
    if backend == "indextts2":
        return _load_indextts(project)
    if backend == "indextts2_api":
        return _indextts_api_runner(project, api_key)
    if backend == "generic_tts_api":
        return _generic_tts_api_runner(project, api_key)
    if backend == "gpt_sovits":
        return _gpt_sovits_runner(project)
    if backend == "cosyvoice":
        return _cosyvoice_runner(project)
    if backend == "fish_speech":
        return _fish_runner(project, api_key)
    if backend == "edge_tts":
        return _edge_tts_runner(project)
    if backend == "mimo_tts":
        return _mimo_runner(project, api_key)
    if backend == "minimax":
        return _minimax_runner(project, api_key)
    raise SynthesisError(f"未知 TTS（语音合成）模型后端：{backend}")


def synthesize_with_selected_backend(
    project: DubProject,
    project_dir: Path,
    source: Path,
    force: bool = False,
    sentence_ids: Iterable[str] | None = None,
    progress: Progress | None = None,
    on_sentence: Callable[[], None] | None = None,
    cancel_event: CancellationSignal | None = None,
) -> list[str]:
    check_cancelled(cancel_event)
    spec = TTS_BACKENDS.get(project.settings.tts_backend)
    if spec is None:
        raise SynthesisError(f"未知 TTS（语音合成）模型后端：{project.settings.tts_backend}")
    if project.settings.tts_clone_mode not in spec.clone_modes:
        raise SynthesisError(
            f"{spec.label} 不支持参考策略 {project.settings.tts_clone_mode}；"
            "请在设置 → TTS（语音合成）中选择该后端提供的模式。"
        )
    requested = set(sentence_ids) if sentence_ids is not None else None
    muted = [
        sentence
        for sentence in project.sentences
        if sentence.enabled
        and not sentence.chinese_audio_enabled
        and (requested is None or sentence.id in requested)
    ]
    skipped = [
        sentence
        for sentence in project.sentences
        if sentence.enabled
        and sentence.chinese_audio_enabled
        and sentence.zh_text
        and not has_speakable_text(sentence.zh_text)
        and (requested is None or sentence.id in requested)
    ]
    skipped_changed = False
    for sentence in skipped:
        skipped_changed = skipped_changed or any(
            (
                sentence.reference_file is not None,
                sentence.tts_file is not None,
                sentence.tts_duration_seconds is not None,
                sentence.tts_cache_key is not None,
                sentence.status != "skipped_tts",
                sentence.error is not None,
            )
        )
        sentence.reference_file = None
        sentence.tts_file = None
        sentence.tts_duration_seconds = None
        sentence.tts_cache_key = None
        sentence.status = "skipped_tts"
        sentence.error = None
    if skipped_changed and on_sentence:
        on_sentence()
    selected = _selected_sentences(project, sentence_ids)
    if not selected:
        if skipped or muted:
            if progress:
                label = "无需配音内容" if muted else "无可朗读内容"
                progress(f"已跳过 {len(skipped) + len(muted)} 句{label}", 1, 1)
            return []
        raise SynthesisError("没有可生成的句子；请先完成翻译并确认句子已启用。")
    tts_dir = project_dir / "chinese"
    tts_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        for sentence in selected:
            recovered = _cache_output(project, sentence, project_dir)
            candidate = sentence.model_copy(
                update={
                    "tts_file": recovered.relative_to(project_dir).as_posix(),
                    "tts_cache_key": tts_cache_key(project, sentence),
                }
            )
            if recovered.is_file() and valid_tts_cache(project, candidate, project_dir):
                sentence.tts_file = candidate.tts_file
                sentence.tts_cache_key = candidate.tts_cache_key
                sentence.tts_duration_seconds = float(sf.info(recovered).duration)
                sentence.status = "synthesized"
                sentence.error = None
    pending = [
        sentence
        for sentence in selected
        if force or not valid_tts_cache(project, sentence, project_dir)
    ]
    if not pending:
        if progress:
            progress("中文配音缓存完整，无需重新生成", 1, 1)
        return []
    if project.settings.tts_backend == "indextts2_5" and _indextts25_command(project) is not None:
        return _synthesize_indextts25_batch(
            project,
            project_dir,
            source,
            pending,
            progress,
            on_sentence,
            cancel_event,
        )
    if project.settings.tts_backend == "indextts2_5":
        raise SynthesisError("找不到 IndexTTS-2.5 独立运行环境。请在“设备与模型”中安装或修复。")
    if project.settings.tts_backend == "indextts2" and _indextts_command(project) is not None:
        return _synthesize_indextts_cli_batch(
            project,
            project_dir,
            source,
            pending,
            progress,
            on_sentence,
            cancel_event,
        )
    if progress:
        progress(f"加载 {spec.label}", 0, len(pending))
    request_workers = (
        1
        if project.settings.tts_backend == "edge_tts"
        else min(project.settings.tts_request_concurrency, len(pending))
    )
    logger.info(
        "TTS 后端开始：后端=%s 模型=%s 运行方式=%s 地址=%s 句数=%d 并发=%d",
        project.settings.tts_backend,
        project.settings.tts_model,
        spec.runtime,
        safe_api_url(project.settings.tts_api_base_url) if spec.runtime == "http" else "本地",
        len(pending),
        request_workers if spec.runtime == "http" else 1,
    )
    run, cleanup = _runner(project)
    cleanup_lock = threading.Lock()
    cleanup_done = False

    def cleanup_once() -> None:
        nonlocal cleanup_done
        with cleanup_lock:
            if cleanup_done:
                return
            cleanup_done = True
        cleanup()

    callback_signal = (
        register_cancel_callback(cleanup_once, cancel_event) if spec.runtime == "http" else None
    )
    failures: list[str] = []
    if spec.runtime == "http":
        from contextvars import copy_context

        items = iter(pending)
        completed_count = 0

        def generate_one(
            sentence: Sentence, reference: VoiceReference, output: Path
        ) -> tuple[int, int]:
            check_cancelled(cancel_event)
            temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.wav")
            try:
                run(sentence, reference, temporary)
                check_cancelled(cancel_event)
                frame_count, sample_rate = _validate_output(temporary)
                temporary.replace(output)
                return frame_count, sample_rate
            finally:
                temporary.unlink(missing_ok=True)

        try:
            with ThreadPoolExecutor(max_workers=max(1, request_workers)) as executor:
                futures: dict[Future[tuple[int, int]], tuple[Sentence, VoiceReference, Path]] = {}

                def fill_window() -> None:
                    nonlocal completed_count
                    while len(futures) < max(1, request_workers) * 2:
                        check_cancelled(cancel_event)
                        sentence = next(items, None)
                        if sentence is None:
                            break
                        try:
                            reference = (
                                prepare_voice_reference(project, project_dir, source, sentence)
                                if _uses_reference_audio(project)
                                else _empty_reference()
                            )
                            _require_reference_text(project, reference)
                            output = _cache_output(project, sentence, project_dir)
                            future = executor.submit(
                                copy_context().run, generate_one, sentence, reference, output
                            )
                            futures[future] = (sentence, reference, output)
                        except OperationCancelledError:
                            raise
                        except Exception as exc:
                            sentence.status = "error"
                            sentence.error = str(exc)
                            failures.append(f"{sentence.id}: {exc}")
                            completed_count += 1
                            if on_sentence:
                                on_sentence()

                fill_window()
                while futures:
                    check_cancelled(cancel_event)
                    done, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        sentence, reference, output = futures.pop(future)
                        try:
                            frame_count, sample_rate = future.result()
                            sentence.reference_file = (
                                str(reference.path) if _uses_reference_audio(project) else None
                            )
                            sentence.tts_file = output.relative_to(project_dir).as_posix()
                            sentence.tts_duration_seconds = frame_count / sample_rate
                            sentence.tts_cache_key = tts_cache_key(project, sentence)
                            sentence.status = "synthesized"
                            sentence.error = None
                        except OperationCancelledError:
                            raise
                        except Exception as exc:
                            sentence.status = "error"
                            sentence.error = str(exc)
                            failures.append(f"{sentence.id}: {exc}")
                        completed_count += 1
                        if on_sentence:
                            on_sentence()
                        if progress:
                            progress(
                                f"外部 TTS 已处理 {completed_count}/{len(pending)} 句",
                                completed_count,
                                len(pending),
                            )
                    fill_window()
        finally:
            unregister_cancel_callback(cleanup_once, callback_signal)
            cleanup_once()
        return failures
    try:
        for index, sentence in enumerate(pending, start=1):
            check_cancelled(cancel_event)
            output = _cache_output(project, sentence, project_dir)
            try:
                if project.settings.tts_backend in {"indextts2_5", "indextts2"}:
                    reference = prepare_index_speaker_reference(
                        project,
                        project_dir,
                        source,
                        sentence,
                    )
                    emotion = prepare_index_emotion_reference(
                        project,
                        project_dir,
                        source,
                        sentence,
                        reference,
                    )
                    if emotion is not None:
                        reference = VoiceReference(
                            path=reference.path,
                            text=reference.text,
                            identity=reference.identity,
                            language=reference.language,
                            sentence=reference.sentence,
                            emotion_path=emotion.path,
                            emotion_identity=emotion.identity,
                        )
                else:
                    reference = prepare_voice_reference(project, project_dir, source, sentence)
                _require_reference_text(project, reference)
                if progress:
                    progress(
                        f"{sentence.id}：{spec.label} 生成中文",
                        index - 1,
                        len(pending),
                    )
                temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.wav")
                try:
                    run(sentence, reference, temporary)
                    frame_count, sample_rate = _validate_output(temporary)
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
                sentence.reference_file = str(reference.path)
                sentence.tts_file = str(output.relative_to(project_dir))
                sentence.tts_duration_seconds = frame_count / sample_rate
                sentence.tts_cache_key = tts_cache_key(project, sentence)
                sentence.status = "synthesized"
                sentence.error = None
            except OperationCancelledError:
                raise
            except Exception as exc:
                sentence.status = "error"
                sentence.error = str(exc)
                failures.append(f"{sentence.id}: {exc}")
            if on_sentence:
                on_sentence()
            if progress:
                progress(f"已处理 {index}/{len(pending)} 句中文配音", index, len(pending))
    finally:
        if callback_signal is not None:
            unregister_cancel_callback(cleanup_once, callback_signal)
        cleanup_once()
    return failures
