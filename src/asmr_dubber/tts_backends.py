from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import mimetypes
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import project_file_exists
from .errors import OperationCancelledError, SynthesisError
from .model_registry import TTS_BACKENDS
from .models import DubProject, Sentence
from .platforms import isolated_runtime_environment
from .task_control import (
    CancellationSignal,
    check_cancelled,
    register_cancel_callback,
    register_process,
    terminate_process_tree,
    unregister_cancel_callback,
    unregister_process,
)
from .tts import tts_cache_key
from .user_settings import saved_service_key
from .voice_reference import (
    VoiceReference,
    prepare_index_emotion_reference,
    prepare_index_speaker_reference,
    prepare_voice_reference,
)

Progress = Callable[[str, int, int], None]
logger = logging.getLogger(__name__)


def _selected_sentences(project: DubProject, sentence_ids: Iterable[str] | None) -> list[Sentence]:
    requested = set(sentence_ids) if sentence_ids is not None else None
    return [
        sentence
        for sentence in project.sentences
        if sentence.enabled and sentence.zh_text and (requested is None or sentence.id in requested)
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
            "text": sentence.zh_text,
            "output_path": str(output),
            "emo_alpha": project.settings.tts_index_emo_alpha,
            "use_random": False,
            "verbose": False,
        }
        if project.settings.tts_index_emotion_source == "text":
            kwargs.update(
                use_emo_text=True,
                emo_text=project.settings.tts_index_emo_text.strip() or sentence.zh_text,
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
    model_dir = Path(project.settings.tts_model_path).expanduser().resolve()
    runtime_root = model_dir.parent
    python_candidates = (
        runtime_root / ".venv" / "Scripts" / "python.exe",
        runtime_root / ".venv" / "bin" / "python",
    )
    python = next((candidate for candidate in python_candidates if candidate.is_file()), None)
    if python is not None:
        return [str(python), "-m", "indextts.cli_v2"]
    executable = _indextts_cli(project)
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
    project_dir = project_dir.resolve()
    source = source.resolve()
    command_prefix = _indextts_command(project)
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
    for sentence in pending:
        check_cancelled(cancel_event)
        reference = prepare_index_speaker_reference(project, project_dir, source, sentence)
        emotion = prepare_index_emotion_reference(
            project,
            project_dir,
            source,
            sentence,
            reference,
        )
        output = (tts_dir / f".{sentence.id}.indextts2.tmp.wav").resolve()
        output.unlink(missing_ok=True)
        payload: dict[str, Any] = {
            "text": sentence.zh_text,
            "voice": str(reference.path.resolve()),
            "output": str(output),
        }
        if project.settings.tts_index_emotion_source == "text":
            payload["emotion_text"] = (
                project.settings.tts_index_emo_text.strip() or sentence.zh_text
            )
            payload["emotion_weight"] = project.settings.tts_index_emo_alpha
        elif emotion is not None:
            payload["emotion_audio"] = str(emotion.path.resolve())
            payload["emotion_weight"] = project.settings.tts_index_emo_alpha
        lines.append(json.dumps(payload, ensure_ascii=False))
        tasks.append((sentence, reference, output))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        *command_prefix,
        "batch",
        "--batch-file",
        str(manifest),
        "--model-dir",
        str(model_dir),
        "--device",
        project.settings.tts_device,
        "--fp16" if project.settings.tts_index_use_fp16 else "--no-fp16",
        "--force",
    ]
    logger.info(
        "IndexTTS2 开始：device=%s fp16=%s sentences=%d",
        project.settings.tts_device,
        project.settings.tts_index_use_fp16,
        len(tasks),
    )
    if progress:
        progress(f"IndexTTS2 独立环境批量生成 {len(tasks)} 句", 0, len(tasks))
    output_tail: deque[str] = deque(maxlen=200)
    return_code = -1
    process: subprocess.Popen[str] | None = None
    cancelled: OperationCancelledError | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=isolated_runtime_environment("index-tts"),
        )
        register_process(process, cancel_event)
        messages: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, name="indextts-output", daemon=True)
        reader.start()
        deadline = time.monotonic() + (project.settings.tts_timeout_seconds * max(1, len(tasks)))
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
                if progress:
                    progress(
                        f"IndexTTS2 已生成 {generated_count}/{len(tasks)} 句",
                        generated_count,
                        len(tasks),
                    )
        return_code = process.wait()
        reader.join(timeout=1)
        # Cancellation can terminate the child between the queue check and
        # process.wait(). Re-check here so a killed process is reported as a
        # user cancellation rather than a misleading synthesis failure.
        check_cancelled(cancel_event)
    except OperationCancelledError as exc:
        cancelled = exc
    except OSError as exc:
        raise SynthesisError(f"无法启动 IndexTTS2 独立环境：{exc}") from exc
    finally:
        if process is not None:
            unregister_process(process, cancel_event)
            if process.poll() is None:
                terminate_process_tree(process)
        manifest.unlink(missing_ok=True)

    detail = "".join(output_tail)[-3000:]
    failures: list[str] = []
    for index, (sentence, reference, temporary) in enumerate(tasks, start=1):
        output = tts_dir / f"{sentence.id}.wav"
        if cancelled is not None and not temporary.is_file():
            continue
        try:
            if not temporary.is_file():
                raise SynthesisError(
                    f"IndexTTS2 未生成本句（批处理退出码 {return_code}）：{detail}"
                )
            frame_count, sample_rate = _validate_output(temporary)
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
    if cancelled is not None:
        raise cancelled
    if return_code != 0 and not failures:
        raise SynthesisError(f"IndexTTS2 批处理失败（{return_code}）：{detail}")
    logger.info(
        "IndexTTS2 完成：device=%s sentences=%d failures=%d",
        project.settings.tts_device,
        len(tasks),
        len(failures),
    )
    return failures


def _gpt_sovits_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    import httpx

    url = f"{project.settings.tts_api_base_url.rstrip('/')}/tts"
    client = httpx.Client(timeout=project.settings.tts_timeout_seconds)

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        payload = {
            "text": sentence.zh_text,
            "text_lang": "zh",
            "ref_audio_path": str(reference.path),
            "prompt_lang": reference.language,
            "prompt_text": reference.text,
            "top_k": project.settings.tts_gpt_top_k,
            "top_p": project.settings.tts_top_p,
            "temperature": project.settings.tts_temperature,
            "text_split_method": project.settings.tts_gpt_text_split_method,
            "speed_factor": project.settings.tts_speed,
            "seed": project.settings.random_seed,
            "media_type": "wav",
            "streaming_mode": False,
            "sample_steps": project.settings.tts_gpt_sample_steps,
        }
        response = client.post(url, json=payload)
        if response.is_error:
            raise SynthesisError(f"GPT-SoVITS API {response.status_code}: {response.text[:500]}")
        output.write_bytes(response.content)

    return run, client.close


def _cosyvoice_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    import httpx

    mode = project.settings.tts_cosyvoice_mode
    endpoint = "inference_zero_shot" if mode == "zero_shot" else "inference_cross_lingual"
    url = f"{project.settings.tts_api_base_url.rstrip('/')}/{endpoint}"
    client = httpx.Client(timeout=project.settings.tts_timeout_seconds)

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        data = {"tts_text": sentence.zh_text}
        if mode == "zero_shot":
            data["prompt_text"] = reference.text
        with reference.path.open("rb") as handle:
            response = client.post(
                url,
                data=data,
                files={"prompt_wav": (reference.path.name, handle, "audio/wav")},
            )
        if response.is_error:
            raise SynthesisError(f"CosyVoice API {response.status_code}: {response.text[:500]}")
        output.write_bytes(response.content)

    return run, client.close


def _fish_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    import httpx

    base = project.settings.tts_api_base_url.rstrip("/")
    url = base if base.endswith("/v1/tts") else f"{base}/v1/tts"
    key = saved_service_key(f"tts:{project.settings.tts_backend}")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = httpx.Client(headers=headers, timeout=project.settings.tts_timeout_seconds)
    reference_payloads: dict[tuple[str, str], str] = {}
    payload_lock = threading.Lock()

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        reference_key = (str(reference.path.resolve()), reference.text)
        with payload_lock:
            encoded_reference = reference_payloads.get(reference_key)
            if encoded_reference is None:
                encoded_reference = base64.b64encode(reference.path.read_bytes()).decode("ascii")
                reference_payloads[reference_key] = encoded_reference
        payload = {
            "text": sentence.zh_text,
            "format": "wav",
            "normalize": True,
            "references": [
                {
                    "audio": encoded_reference,
                    "text": reference.text,
                }
            ],
        }
        response = client.post(url, json=payload)
        if response.is_error:
            raise SynthesisError(f"Fish Speech API {response.status_code}: {response.text[:500]}")
        output.write_bytes(response.content)

    def cleanup() -> None:
        reference_payloads.clear()
        client.close()

    return run, cleanup


def _edge_tts_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    try:
        import edge_tts
    except ImportError as exc:
        raise SynthesisError("Edge TTS 运行依赖缺失；请重新运行 setup 修复基础依赖。") from exc

    from .audio import _run_ffmpeg

    voice = project.settings.tts_voice.strip() or TTS_BACKENDS["edge_tts"].default_voice
    speed_percent = round((project.settings.tts_speed - 1.0) * 100)
    rate = f"{speed_percent:+d}%"

    def run(sentence: Sentence, _reference: VoiceReference, output: Path) -> None:
        encoded = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.edge.mp3")

        async def synthesize() -> None:
            communicator = edge_tts.Communicate(sentence.zh_text, voice=voice, rate=rate)
            await communicator.save(str(encoded))

        try:
            asyncio.run(synthesize())
            _run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(encoded),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_f32le",
                    str(output),
                ]
            )
        except SynthesisError:
            raise
        except Exception as exc:
            raise SynthesisError(f"Edge TTS 生成失败：{exc}") from exc
        finally:
            encoded.unlink(missing_ok=True)

    return run, lambda: None


def _mimo_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    import httpx

    base = project.settings.tts_api_base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    key = saved_service_key("tts:mimo_tts")
    if not key:
        raise SynthesisError("小米 MiMo TTS 尚未保存 API Key。")
    client = httpx.Client(
        headers={"api-key": key, "Content-Type": "application/json"},
        timeout=project.settings.tts_timeout_seconds,
    )
    encoded_references: dict[str, str] = {}
    payload_lock = threading.Lock()

    def reference_data_uri(reference: VoiceReference) -> str:
        resolved = str(reference.path.resolve())
        with payload_lock:
            cached = encoded_references.get(resolved)
            if cached is not None:
                return cached
            mime = mimetypes.guess_type(reference.path.name)[0] or "audio/wav"
            if mime in {"audio/x-wav", "audio/vnd.wave"}:
                mime = "audio/wav"
            encoded = base64.b64encode(reference.path.read_bytes()).decode("ascii")
            value = f"data:{mime};base64,{encoded}"
            encoded_references[resolved] = value
            return value

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        model = project.settings.tts_model.strip() or TTS_BACKENDS["mimo_tts"].default_model
        style = project.settings.tts_style_prompt.strip()
        messages = [
            {"role": "user", "content": style},
            {"role": "assistant", "content": sentence.zh_text},
        ]
        audio: dict[str, object] = {"format": "wav"}
        if model == "mimo-v2.5-tts-voiceclone":
            audio["voice"] = reference_data_uri(reference)
        elif model == "mimo-v2.5-tts-voicedesign":
            if not style:
                messages[0]["content"] = "温柔自然的中文女声，语速平稳。"
            audio["optimize_text_preview"] = True
        else:
            audio["voice"] = (
                project.settings.tts_voice.strip() or TTS_BACKENDS["mimo_tts"].default_voice
            )
        response = client.post(
            url,
            json={"model": model, "messages": messages, "audio": audio, "stream": False},
        )
        if response.is_error:
            raise SynthesisError(f"小米 MiMo TTS API {response.status_code}: {response.text[:500]}")
        try:
            encoded_audio = response.json()["choices"][0]["message"]["audio"]["data"]
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SynthesisError("小米 MiMo TTS 返回格式无效，未找到可解码的音频。") from exc
        if not audio_bytes:
            raise SynthesisError("小米 MiMo TTS 返回了空音频。")
        output.write_bytes(audio_bytes)

    def cleanup() -> None:
        encoded_references.clear()
        client.close()

    return run, cleanup


def _minimax_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    import httpx

    base = project.settings.tts_api_base_url.rstrip("/")
    url = base if base.endswith("/v1/t2a_v2") else f"{base}/v1/t2a_v2"
    key = saved_service_key("tts:minimax")
    if not key:
        raise SynthesisError("MiniMax TTS 尚未保存 API Key。")
    client = httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=project.settings.tts_timeout_seconds,
    )

    def run(sentence: Sentence, _reference: VoiceReference, output: Path) -> None:
        voice_setting: dict[str, object] = {
            "voice_id": project.settings.tts_voice.strip() or TTS_BACKENDS["minimax"].default_voice,
            "speed": project.settings.tts_speed,
            "vol": project.settings.tts_volume,
            "pitch": project.settings.tts_pitch,
        }
        emotion = project.settings.tts_emotion.strip()
        if emotion and emotion != "auto":
            voice_setting["emotion"] = emotion
        payload = {
            "model": project.settings.tts_model or TTS_BACKENDS["minimax"].default_model,
            "text": sentence.zh_text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": 32_000,
                "bitrate": 128_000,
                "format": "wav",
                "channel": 1,
            },
            "language_boost": "Chinese",
            "output_format": "hex",
            "subtitle_enable": False,
        }
        response = client.post(url, json=payload)
        if response.is_error:
            raise SynthesisError(f"MiniMax TTS API {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
            base_response = data.get("base_resp") or {}
            if base_response.get("status_code", 0) != 0:
                raise SynthesisError(
                    "MiniMax TTS 拒绝请求：" + str(base_response.get("status_msg", "未知错误"))
                )
            audio_bytes = bytes.fromhex(data["data"]["audio"])
        except SynthesisError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SynthesisError("MiniMax TTS 返回格式无效，未找到可解码的音频。") from exc
        if not audio_bytes:
            raise SynthesisError("MiniMax TTS 返回了空音频。")
        output.write_bytes(audio_bytes)

    return run, client.close


def _runner(project: DubProject) -> tuple[Any, Callable[[], None]]:
    backend = project.settings.tts_backend
    if backend == "indextts2":
        return _load_indextts(project)
    if backend == "gpt_sovits":
        return _gpt_sovits_runner(project)
    if backend == "cosyvoice":
        return _cosyvoice_runner(project)
    if backend == "fish_speech":
        return _fish_runner(project)
    if backend == "edge_tts":
        return _edge_tts_runner(project)
    if backend == "mimo_tts":
        return _mimo_runner(project)
    if backend == "minimax":
        return _minimax_runner(project)
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
    selected = _selected_sentences(project, sentence_ids)
    if not selected:
        raise SynthesisError("没有可生成的句子；请先完成翻译并确认句子已启用。")
    tts_dir = project_dir / "chinese"
    tts_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        sentence
        for sentence in selected
        if force
        or not sentence.tts_file
        or not project_file_exists(
            project_dir,
            sentence.tts_file,
            f"句子 {sentence.id} 的中文音频",
        )
        or sentence.tts_cache_key != tts_cache_key(project, sentence)
    ]
    if not pending:
        if progress:
            progress("中文配音缓存完整，无需重新生成", 1, 1)
        return []
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
        prepared: list[tuple[Sentence, VoiceReference, Path]] = []
        completed_count = 0
        for sentence in pending:
            check_cancelled(cancel_event)
            try:
                reference = (
                    prepare_voice_reference(project, project_dir, source, sentence)
                    if _uses_reference_audio(project)
                    else _empty_reference()
                )
                _require_reference_text(project, reference)
                prepared.append((sentence, reference, tts_dir / f"{sentence.id}.wav"))
            except OperationCancelledError:
                raise
            except Exception as exc:
                sentence.status = "error"
                sentence.error = str(exc)
                failures.append(f"{sentence.id}: {exc}")
                completed_count += 1
                if on_sentence:
                    on_sentence()

        def generate_one(
            sentence: Sentence,
            reference: VoiceReference,
            output: Path,
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
            workers = min(project.settings.tts_request_concurrency, len(prepared))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures: dict[Future[tuple[int, int]], tuple[Sentence, VoiceReference, Path]] = {
                    executor.submit(generate_one, sentence, reference, output): (
                        sentence,
                        reference,
                        output,
                    )
                    for sentence, reference, output in prepared
                }
                for future in as_completed(futures):
                    check_cancelled(cancel_event)
                    sentence, reference, output = futures[future]
                    try:
                        frame_count, sample_rate = future.result()
                        sentence.reference_file = (
                            str(reference.path) if _uses_reference_audio(project) else None
                        )
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
                    completed_count += 1
                    if on_sentence:
                        on_sentence()
                    if progress:
                        progress(
                            f"外部 TTS（语音合成）已处理 {completed_count}/{len(pending)} 句",
                            completed_count,
                            len(pending),
                        )
        finally:
            unregister_cancel_callback(cleanup_once, callback_signal)
            cleanup_once()
        return failures
    try:
        for index, sentence in enumerate(pending, start=1):
            check_cancelled(cancel_event)
            output = tts_dir / f"{sentence.id}.wav"
            try:
                if project.settings.tts_backend == "indextts2":
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
