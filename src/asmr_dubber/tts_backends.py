from __future__ import annotations

import base64
import gc
import json
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import project_file_exists
from .errors import SynthesisError
from .model_registry import TTS_BACKENDS
from .models import DubProject, Sentence
from .platforms import isolated_runtime_environment
from .tts import tts_cache_key
from .user_settings import saved_service_key
from .voice_reference import VoiceReference, prepare_voice_reference

Progress = Callable[[str, int, int], None]


def _selected_sentences(project: DubProject, sentence_ids: Iterable[str] | None) -> list[Sentence]:
    requested = set(sentence_ids) if sentence_ids is not None else None
    return [
        sentence
        for sentence in project.sentences
        if sentence.enabled and sentence.zh_text and (requested is None or sentence.id in requested)
    ]


def _validate_output(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise SynthesisError(f"TTS 没有生成输出文件：{path}")
    try:
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise SynthesisError("TTS 输出为空或采样率无效。")
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
                        raise SynthesisError("TTS 输出包含无效采样。")
            return int(info.frames), int(info.samplerate)
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        if isinstance(exc, SynthesisError):
            raise
        raise SynthesisError(f"TTS 输出不是可读取的音频：{path}: {exc}") from exc
    if waveform.size == 0 or sample_rate <= 0 or not np.isfinite(waveform).all():
        raise SynthesisError("TTS 输出为空或包含无效采样。")
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    # Generated audio is canonicalized for the mixer; no loudness normalization is applied here.
    sf.write(path, mono, sample_rate, format="WAV", subtype="FLOAT")
    return int(mono.size), int(sample_rate)


def _require_reference_text(project: DubProject, reference: VoiceReference) -> None:
    spec = TTS_BACKENDS[project.settings.tts_backend]
    if spec.reference_text == "required" and not reference.text.strip():
        if project.settings.tts_backend == "qwen3_tts" and project.settings.tts_qwen_x_vector_only:
            return
        raise SynthesisError(
            f"{spec.label} 的高质量克隆需要参考音频对应文本。"
            "请在设置 → TTS → 外部参考文本中填写，或改用项目内参考句。"
        )


def _load_qwen3(project: DubProject) -> tuple[Any, Callable[[], None]]:
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SynthesisError("Qwen3-TTS 未安装；建议按设置页说明使用独立环境安装。") from exc
    dtype = torch.bfloat16 if project.settings.tts_device.startswith("cuda") else torch.float32
    model = Qwen3TTSModel.from_pretrained(
        project.settings.tts_model,
        device_map="cuda:0" if project.settings.tts_device.startswith("cuda") else "cpu",
        dtype=dtype,
    )
    clone_prompts: dict[tuple[str, str, bool], Any] = {}

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        reference_key = (
            str(reference.path.resolve()),
            reference.text,
            project.settings.tts_qwen_x_vector_only,
        )
        kwargs: dict[str, Any] = {
            "text": sentence.zh_text,
            "language": "Chinese",
            "temperature": project.settings.tts_temperature,
            "top_p": project.settings.tts_top_p,
        }
        create_prompt = getattr(model, "create_voice_clone_prompt", None)
        if callable(create_prompt):
            # The official API documents this object as reusable across
            # generations.  Stable-reference projects now encode the voice
            # prompt once instead of repeating the same work for every line.
            if reference_key not in clone_prompts:
                clone_prompts[reference_key] = create_prompt(
                    ref_audio=str(reference.path),
                    ref_text=reference.text or None,
                    x_vector_only_mode=project.settings.tts_qwen_x_vector_only,
                )
            kwargs["voice_clone_prompt"] = clone_prompts[reference_key]
        else:
            # Compatibility fallback for older qwen-tts releases.
            kwargs.update(
                ref_audio=str(reference.path),
                ref_text=reference.text or None,
                x_vector_only_mode=project.settings.tts_qwen_x_vector_only,
            )
        instruction = project.settings.tts_control_instruction.strip()
        if instruction:
            kwargs["instruct"] = instruction
        wavs, sample_rate = model.generate_voice_clone(**kwargs)
        waveform = np.asarray(wavs[0], dtype=np.float32).squeeze()
        sf.write(output, waveform, int(sample_rate), format="WAV", subtype="FLOAT")

    def cleanup() -> None:
        nonlocal model
        clone_prompts.clear()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return run, cleanup


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
        if project.settings.tts_index_use_emo_text:
            kwargs.update(
                use_emo_text=True,
                emo_text=project.settings.tts_index_emo_text.strip() or sentence.zh_text,
            )
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


def _synthesize_indextts_cli_batch(
    project: DubProject,
    project_dir: Path,
    source: Path,
    pending: list[Sentence],
    progress: Progress | None,
    on_sentence: Callable[[], None] | None,
) -> list[str]:
    project_dir = project_dir.resolve()
    source = source.resolve()
    executable = _indextts_cli(project)
    if executable is None:
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
        reference = prepare_voice_reference(project, project_dir, source, sentence)
        output = (tts_dir / f".{sentence.id}.indextts2.tmp.wav").resolve()
        output.unlink(missing_ok=True)
        payload: dict[str, Any] = {
            "text": sentence.zh_text,
            "voice": str(reference.path.resolve()),
            "output": str(output),
        }
        if project.settings.tts_index_use_emo_text:
            payload["emotion_text"] = (
                project.settings.tts_index_emo_text.strip() or sentence.zh_text
            )
            payload["emotion_weight"] = project.settings.tts_index_emo_alpha
        lines.append(json.dumps(payload, ensure_ascii=False))
        tasks.append((sentence, reference, output))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        str(executable),
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
    if progress:
        progress(f"IndexTTS2 独立环境批量生成 {len(tasks)} 句", 0, len(tasks))
    output_tail: deque[str] = deque(maxlen=200)
    return_code = -1
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
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
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
    except OSError as exc:
        raise SynthesisError(f"无法启动 IndexTTS2 独立环境：{exc}") from exc
    finally:
        manifest.unlink(missing_ok=True)

    detail = "".join(output_tail)[-3000:]
    failures: list[str] = []
    for index, (sentence, reference, temporary) in enumerate(tasks, start=1):
        output = tts_dir / f"{sentence.id}.wav"
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
    if return_code != 0 and not failures:
        raise SynthesisError(f"IndexTTS2 批处理失败（{return_code}）：{detail}")
    return failures


def _load_xtts(project: DubProject) -> tuple[Any, Callable[[], None]]:
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise SynthesisError("Coqui TTS/XTTS-v2 未安装；请使用其兼容独立环境。") from exc
    model = TTS(project.settings.tts_model).to(project.settings.tts_device)

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        model.tts_to_file(
            text=sentence.zh_text,
            speaker_wav=str(reference.path),
            language="zh-cn",
            file_path=str(output),
            speed=project.settings.tts_speed,
        )

    def cleanup() -> None:
        nonlocal model
        del model
        gc.collect()

    return run, cleanup


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
            "prompt_lang": "ja",
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


def _f5_runner(project: DubProject) -> Callable[[Sentence, VoiceReference, Path], None]:
    executable = project.settings.tts_executable.strip()
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved or not Path(resolved).is_file():
        raise SynthesisError(f"找不到 F5-TTS CLI：{executable}")

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        command = [
            str(resolved),
            "--model",
            project.settings.tts_model,
            "--ref_audio",
            str(reference.path),
            "--ref_text",
            reference.text,
            "--gen_text",
            sentence.zh_text,
            "--output_dir",
            str(output.parent),
            "--output_file",
            output.name,
            "--nfe_step",
            str(project.settings.tts_f5_nfe_steps),
            "--cfg_strength",
            str(project.settings.tts_f5_cfg_strength),
            "--speed",
            str(project.settings.tts_speed),
            "--device",
            project.settings.tts_device,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=project.settings.tts_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-1500:]
            raise SynthesisError(f"F5-TTS CLI 失败（{completed.returncode}）：{detail}")

    return run


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

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        reference_key = (str(reference.path.resolve()), reference.text)
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


def _runner(project: DubProject) -> tuple[Any, Callable[[], None]]:
    backend = project.settings.tts_backend
    if backend == "qwen3_tts":
        return _load_qwen3(project)
    if backend == "indextts2":
        return _load_indextts(project)
    if backend == "xtts_v2":
        return _load_xtts(project)
    if backend == "gpt_sovits":
        return _gpt_sovits_runner(project)
    if backend == "cosyvoice":
        return _cosyvoice_runner(project)
    if backend == "f5_tts":
        return _f5_runner(project), lambda: None
    if backend == "fish_speech":
        return _fish_runner(project)
    raise SynthesisError(f"未知 TTS 模型后端：{backend}")


def synthesize_with_selected_backend(
    project: DubProject,
    project_dir: Path,
    source: Path,
    force: bool = False,
    sentence_ids: Iterable[str] | None = None,
    progress: Progress | None = None,
    on_sentence: Callable[[], None] | None = None,
) -> list[str]:
    spec = TTS_BACKENDS.get(project.settings.tts_backend)
    if spec is None:
        raise SynthesisError(f"未知 TTS 模型后端：{project.settings.tts_backend}")
    if project.settings.tts_clone_mode not in spec.clone_modes:
        raise SynthesisError(
            f"{spec.label} 不支持参考策略 {project.settings.tts_clone_mode}；"
            "请在设置 → TTS 中选择该后端提供的模式。"
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
    if project.settings.tts_backend == "indextts2" and _indextts_cli(project) is not None:
        return _synthesize_indextts_cli_batch(
            project,
            project_dir,
            source,
            pending,
            progress,
            on_sentence,
        )
    if progress:
        progress(f"加载 {spec.label}", 0, len(pending))
    run, cleanup = _runner(project)
    failures: list[str] = []
    try:
        for index, sentence in enumerate(pending, start=1):
            output = tts_dir / f"{sentence.id}.wav"
            try:
                reference = prepare_voice_reference(project, project_dir, source, sentence)
                _require_reference_text(project, reference)
                if progress:
                    progress(
                        f"{sentence.id}：{spec.label} 生成中文",
                        index - 1,
                        len(pending),
                    )
                temporary = output.with_name(f".{output.stem}.tmp.wav")
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
            except Exception as exc:
                sentence.status = "error"
                sentence.error = str(exc)
                failures.append(f"{sentence.id}: {exc}")
            if on_sentence:
                on_sentence()
            if progress:
                progress(f"已处理 {index}/{len(pending)} 句中文配音", index, len(pending))
    finally:
        cleanup()
    return failures
