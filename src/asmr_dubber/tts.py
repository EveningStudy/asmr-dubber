from __future__ import annotations

import gc
import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import extract_reference, project_file_exists
from .constants import MODEL_REVISIONS
from .environment import require_cuda, resolve_model_source
from .errors import SynthesisError
from .models import DubProject, Sentence
from .voice_reference import (
    STABLE_CLONE_MODES,
    prepare_voice_reference,
    reference_plan_hash,
)
from .voice_reference import (
    shared_reference_sentence as _shared_reference_sentence,
)

Progress = Callable[[str, int, int], None]


def _sentence_seed(base: int, sentence_id: str) -> int:
    suffix = "".join(char for char in sentence_id if char.isdigit())
    return base + (int(suffix) if suffix else 0)


_STABLE_CLONE_MODES = STABLE_CLONE_MODES
_MIN_STYLE_PROMPT_SECONDS = 0.8


def _enable_voxcpm_prompt_cache(model: object) -> None:
    """Memoize VoxCPM's public prompt encoder while keeping generate() unchanged."""
    tts_model = getattr(model, "tts_model", None)
    original = getattr(tts_model, "build_prompt_cache", None)
    if not callable(original):
        return
    cached: dict[tuple[object, ...], object] = {}

    def build_once(*args: object, **kwargs: object) -> object:
        # VoxCPM's high-level API currently calls this with named scalar
        # arguments.  Unknown/new calling conventions retain upstream behavior.
        if args:
            return original(*args, **kwargs)
        key = (
            kwargs.get("prompt_text"),
            kwargs.get("prompt_wav_path"),
            kwargs.get("reference_wav_path"),
            kwargs.get("trim_silence_vad", False),
        )
        if key[1] is None and key[2] is None:
            return original(**kwargs)
        try:
            return cached[key]
        except KeyError:
            value = original(**kwargs)
            cached[key] = value
            return value

    tts_model.build_prompt_cache = build_once


def shared_reference_sentence(project: DubProject) -> Sentence:
    """Backward-compatible public entry point for the project voice picker."""
    return _shared_reference_sentence(project)


def shared_reference_plan_hash(project: DubProject) -> str:
    return reference_plan_hash(project)


def tts_cache_key(project: DubProject, sentence: Sentence) -> str:
    payload = {
        "source_sha256": project.source.sha256,
        "backend": project.settings.tts_backend,
        "model": project.settings.tts_model,
        "model_revision": MODEL_REVISIONS.get(project.settings.tts_model),
        "reference_source": project.settings.tts_reference_source,
        "clone_mode": project.settings.tts_clone_mode,
        "cfg_value": project.settings.tts_cfg_value,
        "inference_timesteps": project.settings.tts_inference_timesteps,
        "control_instruction": project.settings.tts_control_instruction,
        "device": project.settings.tts_device,
        "speed": project.settings.tts_speed,
        "temperature": project.settings.tts_temperature,
        "top_p": project.settings.tts_top_p,
        "api_base_url": project.settings.tts_api_base_url,
        "model_path": project.settings.tts_model_path,
        "config_path": project.settings.tts_config_path,
        "executable": project.settings.tts_executable,
        "qwen_x_vector_only": project.settings.tts_qwen_x_vector_only,
        "index_fp16": project.settings.tts_index_use_fp16,
        "index_emo_alpha": project.settings.tts_index_emo_alpha,
        "index_use_emo_text": project.settings.tts_index_use_emo_text,
        "index_emo_text": project.settings.tts_index_emo_text,
        "gpt_top_k": project.settings.tts_gpt_top_k,
        "gpt_split": project.settings.tts_gpt_text_split_method,
        "gpt_sample_steps": project.settings.tts_gpt_sample_steps,
        "cosyvoice_mode": project.settings.tts_cosyvoice_mode,
        "f5_nfe_steps": project.settings.tts_f5_nfe_steps,
        "f5_cfg": project.settings.tts_f5_cfg_strength,
        "zh": sentence.zh_text,
        "sentence_id": sentence.id,
        "implementation": "multi-backend-tts-v1",
    }
    uses_shared = (
        project.settings.tts_reference_source == "external"
        or project.settings.tts_clone_mode in _STABLE_CLONE_MODES
    )
    if uses_shared:
        payload.update(
            {
                "reference_plan": shared_reference_plan_hash(project),
                "seed": project.settings.random_seed,
            }
        )
        if project.settings.tts_clone_mode == "stable_voice_sentence_style":
            payload.update(
                {
                    "style_start": sentence.start_seconds,
                    "style_end": sentence.end_seconds,
                    "style_ja": sentence.ja_text,
                    "style_padding": project.settings.reference_padding_seconds,
                    "style_min_seconds": _MIN_STYLE_PROMPT_SECONDS,
                }
            )
    else:
        payload.update(
            {
                "start": sentence.start_seconds,
                "end": sentence.end_seconds,
                "ja": sentence.ja_text,
                "padding": project.settings.reference_padding_seconds,
                "seed": _sentence_seed(project.settings.random_seed, sentence.id),
            }
        )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _selected_sentences(
    project: DubProject,
    sentence_ids: Iterable[str] | None,
) -> list[Sentence]:
    requested = set(sentence_ids) if sentence_ids is not None else None
    return [
        sentence
        for sentence in project.sentences
        if sentence.enabled and sentence.zh_text and (requested is None or sentence.id in requested)
    ]


def synthesize_sentences(
    project: DubProject,
    project_dir: Path,
    source: Path,
    force: bool = False,
    sentence_ids: Iterable[str] | None = None,
    progress: Progress | None = None,
    on_sentence: Callable[[], None] | None = None,
) -> list[str]:
    if project.settings.tts_backend != "voxcpm2":
        from .tts_backends import synthesize_with_selected_backend

        return synthesize_with_selected_backend(
            project,
            project_dir,
            source,
            force=force,
            sentence_ids=sentence_ids,
            progress=progress,
            on_sentence=on_sentence,
        )
    use_cuda = project.settings.tts_device.startswith("cuda")
    if use_cuda:
        require_cuda()
    try:
        import torch
        from voxcpm import VoxCPM
    except ImportError as exc:
        raise SynthesisError("缺少 VoxCPM；请在“设备与模型”页安装 VoxCPM2。") from exc

    selected = _selected_sentences(project, sentence_ids)
    if not selected:
        raise SynthesisError("没有可生成的句子；请先完成翻译并确认句子已启用。")

    refs_dir = project_dir / "references"
    tts_dir = project_dir / "chinese"
    refs_dir.mkdir(parents=True, exist_ok=True)
    tts_dir.mkdir(parents=True, exist_ok=True)

    pending: list[Sentence] = []
    for sentence in selected:
        expected_key = tts_cache_key(project, sentence)
        cached = project_file_exists(
            project_dir,
            sentence.tts_file,
            f"句子 {sentence.id} 的中文音频",
        )
        if force or not cached or sentence.tts_cache_key != expected_key:
            pending.append(sentence)
        else:
            sentence.status = "synthesized"
            sentence.error = None

    if not pending:
        if progress:
            progress("中文配音缓存完整，无需重新生成", 1, 1)
        return []

    if progress:
        progress("加载 VoxCPM2 2B 跨语言音色克隆模型", 0, len(pending))
    torch.set_float32_matmul_precision("high")
    optimize = use_cuda and bool(shutil.which("g++") or shutil.which("clang++"))
    model = None
    failures: list[str] = []
    try:
        model = VoxCPM.from_pretrained(
            resolve_model_source(project.settings.tts_model),
            # VoxCPM2's optimize() checks for the canonical single-GPU name
            # "cuda"; "cuda:0" silently disables torch.compile.
            device="cuda" if use_cuda else "cpu",
            optimize=optimize,
            load_denoiser=False,
        )
        reusable_prompt = project.settings.tts_clone_mode in {
            "stable_reference",
            "stable_hifi",
        } or (
            project.settings.tts_reference_source == "external"
            and project.settings.tts_clone_mode != "stable_voice_sentence_style"
        )
        if reusable_prompt:
            _enable_voxcpm_prompt_cache(model)
        shared_reference: Path | None = None
        shared_prompt_text: str | None = None
        if (
            project.settings.tts_reference_source == "external"
            or project.settings.tts_clone_mode in _STABLE_CLONE_MODES
        ):
            prepared = prepare_voice_reference(project, project_dir, source, pending[0])
            shared_reference = prepared.path
            shared_prompt_text = prepared.text
            if progress:
                if prepared.sentence is not None:
                    duration = prepared.sentence.end_seconds - prepared.sentence.start_seconds
                    message = (
                        f"统一声纹锚点：{prepared.sentence.id}（{duration:.2f} 秒），所有中文复用"
                    )
                else:
                    message = "使用设置中的外部参考音频，所有中文复用该音色"
                progress(message, 0, len(pending))
        for index, sentence in enumerate(pending, start=1):
            reference = shared_reference or refs_dir / f"{sentence.id}.wav"
            output = tts_dir / f"{sentence.id}.wav"
            try:
                if shared_reference is None:
                    if progress:
                        progress(
                            f"{sentence.id}：提取本句原声作为音色参考",
                            index - 1,
                            len(pending),
                        )
                    extract_reference(
                        source=source,
                        destination=reference,
                        start_seconds=sentence.start_seconds,
                        end_seconds=sentence.end_seconds,
                        padding_seconds=project.settings.reference_padding_seconds,
                    )
                seed = (
                    project.settings.random_seed
                    if shared_reference is not None
                    else _sentence_seed(project.settings.random_seed, sentence.id)
                )
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                target_text = sentence.zh_text
                if (
                    project.settings.tts_clone_mode in {"reference_only", "stable_reference"}
                    and project.settings.tts_control_instruction.strip()
                ):
                    target_text = (
                        f"({project.settings.tts_control_instruction.strip()}){target_text}"
                    )
                generate_kwargs = {
                    "text": target_text,
                    "reference_wav_path": str(reference),
                    "cfg_value": project.settings.tts_cfg_value,
                    "inference_timesteps": project.settings.tts_inference_timesteps,
                    # VoxCPM calls its text/number normalization switch `normalize`;
                    # this does not normalize either the reference or output waveform.
                    "normalize": True,
                    "denoise": False,
                    "retry_badcase": True,
                    "retry_badcase_max_times": 3,
                }
                if project.settings.tts_clone_mode == "stable_voice_sentence_style":
                    style_duration = sentence.end_seconds - sentence.start_seconds
                    if style_duration >= _MIN_STYLE_PROMPT_SECONDS:
                        style_payload = (
                            f"{project.source.sha256}|{sentence.id}|{sentence.start_seconds:.6f}|"
                            f"{sentence.end_seconds:.6f}|{sentence.ja_text}|"
                            f"{project.settings.reference_padding_seconds:.6f}"
                        )
                        style_hash = hashlib.sha256(style_payload.encode("utf-8")).hexdigest()[:16]
                        style_reference = (
                            refs_dir / "style_prompts" / f"{sentence.id}_{style_hash}.wav"
                        )
                        if not style_reference.is_file():
                            temporary = style_reference.with_name(
                                f".{style_reference.stem}.tmp.wav"
                            )
                            try:
                                extract_reference(
                                    source=source,
                                    destination=temporary,
                                    start_seconds=sentence.start_seconds,
                                    end_seconds=sentence.end_seconds,
                                    padding_seconds=project.settings.reference_padding_seconds,
                                )
                                temporary.replace(style_reference)
                            finally:
                                temporary.unlink(missing_ok=True)
                        generate_kwargs.update(
                            {
                                # The shared anchor remains the isolated timbre
                                # reference.  This sentence is continuation context
                                # only, so it can contribute timing and emotion
                                # without redefining the project speaker identity.
                                "prompt_wav_path": str(style_reference),
                                "prompt_text": sentence.ja_text,
                            }
                        )
                    elif progress:
                        progress(
                            f"{sentence.id}：原句不足 {_MIN_STYLE_PROMPT_SECONDS:.1f} 秒，"
                            "仅使用统一声纹",
                            index - 1,
                            len(pending),
                        )
                elif project.settings.tts_clone_mode in {"ultimate", "stable_hifi"}:
                    prompt_text = shared_prompt_text or sentence.ja_text
                    generate_kwargs.update(
                        {
                            "prompt_wav_path": str(reference),
                            "prompt_text": prompt_text,
                        }
                    )
                waveform = np.asarray(model.generate(**generate_kwargs), dtype=np.float32).squeeze()
                if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
                    raise SynthesisError(f"模型返回的音频形状无效：{waveform.shape}")
                sample_rate = int(model.tts_model.sample_rate)
                if sample_rate <= 0:
                    raise SynthesisError(f"模型返回的采样率无效：{sample_rate}")
                sf.write(output, waveform, sample_rate, format="WAV", subtype="FLOAT")
                try:
                    sentence.reference_file = str(reference.relative_to(project_dir))
                except ValueError:
                    sentence.reference_file = str(reference)
                sentence.tts_file = str(output.relative_to(project_dir))
                sentence.tts_duration_seconds = float(waveform.size / sample_rate)
                sentence.tts_cache_key = tts_cache_key(project, sentence)
                sentence.status = "synthesized"
                sentence.error = None
            except torch.cuda.OutOfMemoryError as exc:
                sentence.status = "error"
                sentence.error = "GPU 显存不足"
                if on_sentence:
                    on_sentence()
                raise SynthesisError(
                    "VoxCPM2 运行时显存不足。请关闭占用 GPU 的其他程序后重试。"
                ) from exc
            except Exception as exc:  # Keep the project resumable after a bad single reference.
                sentence.status = "error"
                sentence.error = str(exc)
                failures.append(f"{sentence.id}: {exc}")
            if on_sentence:
                on_sentence()
            if progress:
                progress(f"已处理 {index}/{len(pending)} 句中文配音", index, len(pending))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return failures
