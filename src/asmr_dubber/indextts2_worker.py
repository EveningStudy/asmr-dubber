from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_MISSING_RESOURCE = 2
EXIT_RUNTIME_UNAVAILABLE = 3
EXIT_INFERENCE_ERROR = 4

_SAMPLING_RATE = 22050
_HOP_LENGTH = 256
_SILENT_TOKEN_V2 = 52


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASMR Dubber IndexTTS2 worker")
    parser.add_argument("--batch-file", default="")
    parser.add_argument("--daemon", action="store_true", help="Keep model resident across batches")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cuda-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deepspeed",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--accel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(task, dict):
            raise ValueError(f"line {line_number}: task must be an object")
        for field in ("id", "text", "voice", "output"):
            if not str(task.get(field, "")).strip():
                raise ValueError(f"line {line_number}: {field} is required")
        tasks.append(task)
    if not tasks:
        raise ValueError("batch file contains no tasks")
    return tasks


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _optional_acceleration(enabled: bool, module: str, option: str) -> bool:
    if not enabled:
        return False
    try:
        importlib.import_module(module)
    except (ImportError, OSError) as exc:
        print(f"WARNING: {option} ({module}) unavailable: {exc}; continuing", file=sys.stderr)
        return False
    return True


def _aux_paths(model_dir: Path) -> dict[str, str] | None:
    hf_cache = model_dir / "hf_cache"
    paths = {
        "w2v_bert": hf_cache / "w2v-bert-2.0",
        "semantic_codec": hf_cache / "semantic_codec_model.safetensors",
        "campplus": hf_cache / "campplus_cn_common.bin",
        "bigvgan": hf_cache / "bigvgan",
    }
    if (
        paths["w2v_bert"].is_dir()
        and paths["semantic_codec"].is_file()
        and paths["campplus"].is_file()
        and (paths["bigvgan"] / "bigvgan_generator.pt").is_file()
    ):
        return {key: str(value) for key, value in paths.items()}
    return None


def _infer_kwargs(task: dict[str, Any]) -> dict[str, Any]:
    emotion_text = str(task.get("emotion_text", "")).strip()
    emotion_audio = str(task.get("emotion_audio", "")).strip()
    vector = task.get("emotion_vector")
    if vector is not None:
        if not isinstance(vector, list) or len(vector) != 8:
            raise ValueError("emotion_vector must contain exactly 8 values")
        vector = [float(value) for value in vector]
        if any(value < 0.0 or value > 1.0 for value in vector):
            raise ValueError("emotion_vector values must be between 0 and 1")
    target_duration = task.get("target_duration_seconds")
    parsed_target = float(target_duration) if target_duration is not None else None
    if parsed_target is not None and parsed_target <= 0.0:
        parsed_target = None
    return {
        "spk_audio_prompt": str(Path(task["voice"]).resolve()),
        "text": str(task["text"]),
        "output_path": str(Path(task["output"]).resolve()),
        "emo_audio_prompt": str(Path(emotion_audio).resolve()) if emotion_audio else None,
        "emo_alpha": float(task.get("emotion_weight", 0.5)),
        "emo_vector": vector,
        "use_emo_text": bool(emotion_text),
        "emo_text": emotion_text or None,
        "use_random": bool(task.get("use_random", False)),
        "interval_silence": int(task.get("interval_silence_ms", 200)),
        "verbose": False,
        "max_text_tokens_per_segment": int(task.get("max_text_tokens", 120)),
        "do_sample": bool(task.get("do_sample", True)),
        "top_p": float(task.get("top_p", 0.8)),
        "top_k": int(task.get("top_k", 30)),
        "temperature": float(task.get("temperature", 0.8)),
        "length_penalty": float(task.get("length_penalty", 0.0)),
        "num_beams": int(task.get("num_beams", 3)),
        "repetition_penalty": float(task.get("repetition_penalty", 10.0)),
        "max_mel_tokens": int(task.get("max_mel_tokens", 1500)),
        "_duration_factor": float(task.get("duration_factor", 1.0)),
        "_target_duration_seconds": parsed_target,
        "_min_duration_factor": float(task.get("min_duration_factor", 0.76)),
        "_max_duration_factor": float(task.get("max_duration_factor", 1.0)),
    }


def _edge_silence_seconds(codes: Any, stop_token: int) -> float:
    try:
        row = codes[0].detach().cpu().tolist()
    except Exception:
        return 0.14
    if stop_token in row:
        row = row[: row.index(stop_token)]
    if not row:
        return 0.0
    lead = 0
    while lead < len(row) and row[lead] == _SILENT_TOKEN_V2:
        lead += 1
    trail = 0
    while trail < len(row) - lead and row[len(row) - 1 - trail] == _SILENT_TOKEN_V2:
        trail += 1
    return (lead + trail) * 1.72 * _HOP_LENGTH / _SAMPLING_RATE


def _spoken_duration_seconds(wav_path: Path) -> float:
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if not len(data) or rate <= 0:
        return 0.0
    mono = data.mean(axis=1)
    hop = max(1, round(rate * 0.010))
    count = len(mono) // hop
    if count == 0:
        return len(mono) / rate
    rms = np.sqrt(np.mean(mono[: count * hop].astype(np.float64).reshape(count, hop) ** 2, axis=1))
    peak = float(rms.max(initial=0.0))
    if peak <= 1e-7:
        return 0.0
    active = np.flatnonzero(rms >= max(peak * 10 ** (-38.0 / 20.0), 1e-6))
    if not len(active):
        return len(mono) / rate
    return float((active[-1] + 1 - active[0]) * hop / rate)


def _resynthesize_mel_length(
    model: Any,
    s_infer: Any,
    target_mel_frames: int,
    output_path: Path,
) -> None:
    import torch
    import torchaudio

    prompt_condition = model.cache_s2mel_prompt
    ref_mel = model.cache_mel
    style = model.cache_s2mel_style
    if prompt_condition is None or ref_mel is None or style is None:
        return
    device = s_infer.device
    with torch.no_grad():
        ylens = torch.tensor([max(8, int(target_mel_frames))], dtype=torch.long, device=device)
        cond = model.s2mel.models["length_regulator"](
            s_infer, ylens=ylens, n_quantizers=3, f0=None
        )[0]
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        lengths = torch.tensor([cat_condition.size(1)], dtype=torch.long, device=device)
        vc_target = model.s2mel.models["cfm"].inference(
            cat_condition,
            lengths,
            ref_mel,
            style,
            None,
            25,
            inference_cfg_rate=0.7,
        )
        vc_target = vc_target[:, :, ref_mel.size(-1) :]
        wav = model.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
        wav = torch.clamp(32767.0 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
    temp = output_path.with_name(f".{output_path.name}.refine.wav")
    try:
        torchaudio.save(str(temp), wav, _SAMPLING_RATE)
        temp.replace(output_path)
    finally:
        temp.unlink(missing_ok=True)


def _install_runtime_hooks(model: Any) -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "do_sample": True,
        "duration_factor": 1.0,
        "target_duration": None,
        "min_factor": 0.76,
        "max_factor": 1.0,
        "in_segment": False,
        "segment_calls": 0,
        "last_codes": None,
        "last_s_infer": None,
        "last_nat_frames": 0,
        "applied_factor": 1.0,
    }
    original_inference_speech = model.gpt.inference_speech
    s2mel = getattr(model, "s2mel", None)
    length_regulator = s2mel.models["length_regulator"] if s2mel is not None else None
    original_forward = length_regulator.forward if length_regulator is not None else None

    def inference_speech(*call_args: Any, **call_kwargs: Any) -> Any:
        call_kwargs["do_sample"] = state["do_sample"]
        out = original_inference_speech(*call_args, **call_kwargs)
        state["in_segment"] = True
        state["last_codes"] = out[0] if isinstance(out, tuple) and out else None
        return out

    if length_regulator is not None and original_forward is not None:

        def regulated_forward(x: Any, ylens: Any = None, *args: Any, **kwargs: Any) -> Any:
            if state["in_segment"] and ylens is not None:
                state["in_segment"] = False
                state["segment_calls"] += 1
                nat_frames = int(ylens[0].item())
                state["last_nat_frames"] = nat_frames
                state["last_s_infer"] = x.detach()
                factor = float(state["duration_factor"])
                target = state["target_duration"]
                if target is not None and nat_frames > 0:
                    nat_total = nat_frames * _HOP_LENGTH / _SAMPLING_RATE
                    edge = _edge_silence_seconds(state["last_codes"], int(model.stop_mel_token))
                    edge = min(edge, nat_total * 0.35)
                    nat_spoken = max(0.12, nat_total - edge)
                    if target < nat_spoken * 0.98 or state["max_factor"] > 1.0:
                        wanted_total = target + edge * 0.85
                        raw_ratio = wanted_total / max(nat_total, 0.12)
                        clamped = max(state["min_factor"], min(state["max_factor"], raw_ratio))
                        factor = factor * clamped
                state["applied_factor"] = factor
                if abs(factor - 1.0) > 1e-4:
                    ylens = torch.clamp((ylens.float() * factor).round().long(), min=8)
            return original_forward(x, ylens, *args, **kwargs)

        length_regulator.forward = regulated_forward
    model.gpt.inference_speech = inference_speech
    return state


def _run_batch_tasks(model: Any, hook_state: dict[str, Any], batch_file: Path) -> int:
    try:
        tasks = _load_tasks(batch_file)
        kwargs_by_task = [_infer_kwargs(task) for task in tasks]
    except (OSError, ValueError) as exc:
        print(f"ERROR: invalid batch file: {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT_ERROR

    for kwargs in kwargs_by_task:
        for path in (kwargs["spk_audio_prompt"], kwargs["emo_audio_prompt"]):
            if path and not Path(path).is_file():
                print(f"ERROR: reference audio does not exist: {path}", file=sys.stderr, flush=True)
                return EXIT_MISSING_RESOURCE

    failures = 0
    for task, raw_kwargs in zip(tasks, kwargs_by_task, strict=True):
        kwargs = dict(raw_kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        try:
            hook_state["do_sample"] = bool(kwargs.pop("do_sample"))
            hook_state["duration_factor"] = float(kwargs.pop("_duration_factor"))
            hook_state["target_duration"] = kwargs.pop("_target_duration_seconds")
            hook_state["min_factor"] = float(kwargs.pop("_min_duration_factor"))
            hook_state["max_factor"] = float(kwargs.pop("_max_duration_factor"))
            hook_state["in_segment"] = False
            hook_state["segment_calls"] = 0
            hook_state["last_codes"] = None
            hook_state["last_s_infer"] = None
            hook_state["last_nat_frames"] = 0
            hook_state["applied_factor"] = 1.0
            if kwargs["emo_vector"] is not None:
                kwargs["emo_vector"] = model.normalize_emo_vec(kwargs["emo_vector"])
            seed_key = str(task.get("sentence_id") or task["id"])
            identity = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:4], "big")
            _seed_everything((int(task.get("seed", 0)) + identity) % (2**32))
            result = model.infer(**kwargs)
            if result is None or not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("model returned without writing audio")
            target_sec = hook_state["target_duration"]
            if (
                target_sec is not None
                and hook_state["segment_calls"] == 1
                and hook_state["last_s_infer"] is not None
                and hook_state["last_nat_frames"] > 0
            ):
                spoken = _spoken_duration_seconds(output)
                if (
                    spoken > target_sec * 1.06
                    and hook_state["applied_factor"] > hook_state["min_factor"] + 0.02
                ):
                    refined = max(
                        hook_state["min_factor"],
                        hook_state["applied_factor"] * (target_sec / spoken) * 0.98,
                    )
                    target_frames = round(hook_state["last_nat_frames"] * refined)
                    _resynthesize_mel_length(
                        model, hook_state["last_s_infer"], target_frames, output
                    )
            print(f"Generated: {output} ({task['id']})", flush=True)
        except Exception as exc:
            failures += 1
            print(f"Failed: {task['id']}: {exc}", flush=True)
    return EXIT_INFERENCE_ERROR if failures else EXIT_SUCCESS


class _LazyQwenEmotion:
    """Delay loading QwenEmotion (1.11 GiB) until text emotion mode is actually used."""

    def __init__(self, model_dir: str, factory: Any, *, low_vram: bool = False) -> None:
        self.model_dir = model_dir
        self._factory = factory
        self._low_vram = low_vram
        self._inner: Any | None = None

    def inference(self, text_input: str) -> Any:
        inner = self._inner
        if inner is None:
            if self._low_vram:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            inner = self._factory(self.model_dir)
            self._inner = inner
        return inner.inference(text_input)


class _CpuSemanticModelProxy:
    """Keep w2v-bert-2.0 (2.16 GiB) on CPU while matching IndexTTS2's module interface."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def to(self, *_args: Any, **_kwargs: Any) -> _CpuSemanticModelProxy:
        return self

    def eval(self) -> _CpuSemanticModelProxy:
        self._module.eval()
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        cpu_args = tuple(
            arg.detach().cpu() if hasattr(arg, "detach") and hasattr(arg, "cpu") else arg
            for arg in args
        )
        cpu_kwargs = {
            key: (val.detach().cpu() if hasattr(val, "detach") and hasattr(val, "cpu") else val)
            for key, val in kwargs.items()
        }
        return self._module(*cpu_args, **cpu_kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def _should_offload_semantic_model(torch_mod: Any, device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    override = os.getenv("ASMR_DUBBER_INDEXTTS2_LOW_VRAM", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    try:
        free_bytes, total_bytes = torch_mod.cuda.mem_get_info(device)
        return total_bytes <= int(9.5 * 1024**3) or free_bytes <= int(7.0 * 1024**3)
    except Exception:
        try:
            total_bytes = torch_mod.cuda.get_device_properties(device).total_memory
            return total_bytes <= int(9.5 * 1024**3)
        except Exception:
            return False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "config.yaml").is_file():
        print(f"ERROR: model directory is incomplete: {model_dir}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    if not args.daemon:
        batch_file = Path(args.batch_file).expanduser().resolve()
        if not args.batch_file or not batch_file.is_file():
            print(f"ERROR: batch file does not exist: {batch_file}", file=sys.stderr)
            return EXIT_MISSING_RESOURCE
        # Validate batch syntax upfront before loading multi-GB weights.
        try:
            tasks = _load_tasks(batch_file)
            kwargs_by_task = [_infer_kwargs(task) for task in tasks]
        except (OSError, ValueError) as exc:
            print(f"ERROR: invalid batch file: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        for kwargs in kwargs_by_task:
            for path in (kwargs["spk_audio_prompt"], kwargs["emo_audio_prompt"]):
                if path and not Path(path).is_file():
                    print(f"ERROR: reference audio does not exist: {path}", file=sys.stderr)
                    return EXIT_MISSING_RESOURCE

    os.environ.setdefault("HF_HUB_CACHE", str(model_dir / "hf_cache"))
    try:
        import indextts.infer_v2 as infer_v2
        import torch
        from indextts.infer_v2 import IndexTTS2
    except (ImportError, OSError) as exc:
        print(f"ERROR: runtime unavailable: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: CUDA was selected but is not available in this runtime", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE

    effective_fp16 = bool(args.fp16) and not device.startswith("cpu")
    offload_semantic = _should_offload_semantic_model(torch, device)

    use_deepspeed = _optional_acceleration(bool(args.deepspeed), "deepspeed", "DeepSpeed")
    use_accel = _optional_acceleration(bool(args.accel), "flash_attn", "GPT acceleration")
    use_torch_compile = _optional_acceleration(bool(args.torch_compile), "triton", "torch.compile")

    orig_qwen_emotion = infer_v2.QwenEmotion
    orig_load_checkpoint = infer_v2.load_checkpoint
    orig_build_semantic_model = infer_v2.build_semantic_model
    try:
        infer_v2.QwenEmotion = lambda model_dir: _LazyQwenEmotion(
            model_dir, orig_qwen_emotion, low_vram=offload_semantic
        )
        if effective_fp16 and device.startswith("cuda"):

            def _load_gpt_half(model_obj: Any, path: str) -> Any:
                loaded = orig_load_checkpoint(model_obj, path)
                if isinstance(model_obj, infer_v2.UnifiedVoice):
                    model_obj.half()
                return loaded

            infer_v2.load_checkpoint = _load_gpt_half
        if offload_semantic:

            def _build_cpu_semantic(*call_args: Any, **call_kwargs: Any) -> Any:
                semantic_model, semantic_mean, semantic_std = orig_build_semantic_model(
                    *call_args, **call_kwargs
                )
                return _CpuSemanticModelProxy(semantic_model), semantic_mean, semantic_std

            infer_v2.build_semantic_model = _build_cpu_semantic

        model = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_fp16=effective_fp16,
            device=device,
            use_cuda_kernel=bool(args.cuda_kernel),
            use_deepspeed=use_deepspeed,
            use_accel=use_accel,
            use_torch_compile=use_torch_compile,
            aux_paths=_aux_paths(model_dir),
        )
    except Exception as exc:
        print(f"ERROR: model initialization failed: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE
    finally:
        infer_v2.QwenEmotion = orig_qwen_emotion
        infer_v2.load_checkpoint = orig_load_checkpoint
        infer_v2.build_semantic_model = orig_build_semantic_model

    if offload_semantic:

        def _cpu_get_emb(input_features: Any, attention_mask: Any) -> Any:
            with torch.no_grad():
                vq_emb = model.semantic_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                feat = vq_emb.hidden_states[17].to(model.device)
                return (feat - model.semantic_mean) / model.semantic_std

        model.get_emb = _cpu_get_emb

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        alloc_mib = torch.cuda.memory_allocated(device) / (1024**2)
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(
            f"INFO: IndexTTS2 ready (fp16={effective_fp16}, "
            f"low_vram_semantic_cpu={offload_semantic}, "
            f"allocated_mib={alloc_mib:.1f}, peak_mib={peak_mib:.1f})",
            flush=True,
        )

    hook_state = _install_runtime_hooks(model)
    if not args.daemon:
        return _run_batch_tasks(model, hook_state, Path(args.batch_file).expanduser().resolve())

    print("READY: indextts2_worker", flush=True)
    for raw_command in sys.stdin:
        text = raw_command.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            print(
                json.dumps(
                    {"event": "batch_done", "exit_code": EXIT_INPUT_ERROR, "error": str(exc)}
                ),
                flush=True,
            )
            continue
        if payload.get("action") == "quit":
            break
        target = Path(str(payload.get("batch_file", ""))).expanduser().resolve()
        code = _run_batch_tasks(model, hook_state, target)
        print(json.dumps({"event": "batch_done", "exit_code": code}), flush=True)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
