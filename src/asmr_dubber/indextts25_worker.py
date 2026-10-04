from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_MISSING_RESOURCE = 2
EXIT_RUNTIME_UNAVAILABLE = 3
EXIT_INFERENCE_ERROR = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASMR Dubber IndexTTS-2.5 worker")
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bf16",
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


def _effective_bf16(requested: bool, device: str) -> bool:
    if not requested or not device.startswith("cuda"):
        return False
    import torch

    supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if not supported:
        print("BF16 unavailable on this GPU; using full precision.", flush=True)
    return supported


def _optional_acceleration(enabled: bool, module: str, option: str) -> bool:
    try:
        _require_optional_module(enabled, module, option)
    except RuntimeError as exc:
        print(f"WARNING: {exc}; continuing without it", file=sys.stderr)
        return False
    return enabled


def _require_optional_module(enabled: bool, module: str, option: str) -> None:
    if not enabled:
        return
    try:
        importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{option} requires the optional dependency {module}, "
            "but it is not available in this IndexTTS-2.5 runtime"
        ) from exc


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
    return {
        "spk_audio_prompt": str(Path(task["voice"]).resolve()),
        "text": str(task["text"]),
        "output_path": str(Path(task["output"]).resolve()),
        "lang": str(task.get("language", "zh")),
        "emo_audio_prompt": str(Path(emotion_audio).resolve()) if emotion_audio else None,
        "emo_alpha": float(task.get("emotion_weight", 0.5)),
        "emo_vector": vector,
        "use_emo_text": bool(emotion_text),
        "emo_text": emotion_text or None,
        "use_random": bool(task.get("use_random", False)),
        "interval_silence": int(task.get("interval_silence_ms", 200)),
        "verbose": False,
        "max_text_tokens_per_segment": int(task.get("max_text_tokens", 120)),
        "duration_factor": float(task.get("duration_factor", 1.0)),
        "text_normalization": bool(task.get("text_normalization", True)),
        "do_sample": bool(task.get("do_sample", True)),
        "top_p": float(task.get("top_p", 0.8)),
        "top_k": int(task.get("top_k", 30)),
        "temperature": float(task.get("temperature", 0.8)),
        "length_penalty": float(task.get("length_penalty", 0.0)),
        "num_beams": int(task.get("num_beams", 3)),
        "repetition_penalty": float(task.get("repetition_penalty", 10.0)),
        "max_mel_tokens": int(task.get("max_mel_tokens", 1500)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    batch_file = Path(args.batch_file).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not batch_file.is_file():
        print(f"ERROR: batch file does not exist: {batch_file}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    if not (model_dir / "config.yaml").is_file():
        print(f"ERROR: model directory is incomplete: {model_dir}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE
    try:
        tasks = _load_tasks(batch_file)
        kwargs_by_task = [_infer_kwargs(task) for task in tasks]
    except (OSError, ValueError) as exc:
        print(f"ERROR: invalid batch file: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    missing_references = [
        path
        for kwargs in kwargs_by_task
        for path in (kwargs["spk_audio_prompt"], kwargs["emo_audio_prompt"])
        if path and not Path(path).is_file()
    ]
    if missing_references:
        print(f"ERROR: reference audio does not exist: {missing_references[0]}", file=sys.stderr)
        return EXIT_MISSING_RESOURCE

    try:
        import torch
        from indextts.infer_v2_5 import IndexTTS2
    except (ImportError, OSError) as exc:
        print(f"ERROR: runtime unavailable: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: CUDA was selected but is not available in this runtime", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE
    # These switches only make synthesis faster. A runtime that cannot provide one of them
    # still produces the same speech, so it continues without the acceleration.
    use_deepspeed = _optional_acceleration(bool(args.deepspeed), "deepspeed", "DeepSpeed")
    use_accel = _optional_acceleration(bool(args.accel), "flash_attn", "GPT acceleration")
    use_torch_compile = _optional_acceleration(bool(args.torch_compile), "triton", "torch.compile")
    use_qwen_emo = any(bool(kwargs["use_emo_text"]) for kwargs in kwargs_by_task)
    try:
        model = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_bf16=_effective_bf16(bool(args.bf16), device),
            device=device,
            use_cuda_kernel=bool(args.cuda_kernel),
            use_deepspeed=use_deepspeed,
            use_accel=use_accel,
            use_torch_compile=use_torch_compile,
            use_qwen_emo=use_qwen_emo,
        )
    except Exception as exc:
        print(f"ERROR: model initialization failed: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_UNAVAILABLE
    if bool(args.cuda_kernel) and not bool(model.use_cuda_kernel):
        print(
            "WARNING: BigVGAN CUDA kernel could not load in this runtime; "
            "continuing with the standard implementation",
            file=sys.stderr,
        )

    # The pinned upstream 2.5 release accepts ``do_sample`` in infer(), but
    # currently hard-codes True when it calls the GPT generator.  Override only
    # that keyword in the adapter so the UI switch has the promised effect,
    # without modifying the pinned third-party source tree.
    original_inference_speech = model.gpt.inference_speech
    current_do_sample = True

    def inference_speech(*call_args: Any, **call_kwargs: Any) -> Any:
        call_kwargs["do_sample"] = current_do_sample
        return original_inference_speech(*call_args, **call_kwargs)

    model.gpt.inference_speech = inference_speech

    failures = 0
    for task, kwargs in zip(tasks, kwargs_by_task, strict=True):
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        try:
            current_do_sample = bool(kwargs.pop("do_sample"))
            if kwargs["emo_vector"] is not None:
                kwargs["emo_vector"] = model.normalize_emo_vec(kwargs["emo_vector"])
            identity = int.from_bytes(hashlib.sha256(str(task["id"]).encode()).digest()[:4], "big")
            _seed_everything((int(task.get("seed", 0)) + identity) % (2**32))
            result = model.infer(**kwargs)
            if result is None or not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("model returned without writing audio")
            print(f"Generated: {task['id']}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"Failed: {task['id']}: {exc}", flush=True)
    return EXIT_INFERENCE_ERROR if failures else EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
