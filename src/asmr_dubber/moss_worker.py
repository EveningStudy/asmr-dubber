"""Standalone worker: runs only inside the MOSS Transformers 5 environment."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def parse_transcript(text: str, duration: float) -> list[dict]:
    pattern = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.S)
    result = []
    cursor = 0
    for match in pattern.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError("MOSS 返回了无法解析的转写内容；请检查提示词或生成参数。")
        start, end = float(match[1]), float(match[4])
        content = match[3].strip()
        if not all(map(math.isfinite, (start, end))) or end <= start or start >= duration:
            raise ValueError("MOSS 返回了无效或越界的时间戳。")
        if content:
            result.append(
                {"start": start, "end": min(duration, end), "speaker": match[2], "text": content}
            )
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError("MOSS 返回了不完整的转写；请增加输出 Token 上限或缩短分块。")
    return result


def report(message: str) -> None:
    print(json.dumps({"progress": message}, ensure_ascii=False), flush=True)


def run(request: dict) -> dict:
    import random

    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    settings = request["settings"]
    seed = settings["asr_moss_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = settings["asr_device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("MOSS 未检测到 CUDA，请选择 CPU 或修复独立环境。")
    precision = settings["asr_moss_dtype"]
    if precision == "auto":
        precision = (
            "bfloat16"
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else "float16"
            if device.startswith("cuda")
            else "float32"
        )
    dtype = getattr(torch, precision)
    if device == "cpu" and dtype != torch.float32:
        raise ValueError("MOSS CPU 推理请使用 auto 或 float32 精度。")
    report(f"MOSS 加载模型（{device}, {precision}）")
    model = (
        AutoModelForCausalLM.from_pretrained(
            request["model"],
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=settings["asr_moss_attention"],
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        request["model"],
        trust_remote_code=True,
        local_files_only=True,
    )
    rows = []
    transcripts = []
    extra = json.loads(settings["asr_moss_generation_params"])
    for index, window in enumerate(request["windows"]):
        report(f"MOSS 识别分块 {index + 1}/{len(request['windows'])}")
        samples, rate = sf.read(
            request["audio"],
            start=window[0],
            stop=window[1],
            dtype="float32",
            always_2d=True,
        )
        mono = samples.mean(axis=1)
        duration = len(mono) / rate
        if not np.any(mono):
            continue
        if rate != 16000:
            mono = soxr.resample(mono, rate, 16000)
        prompt = settings["asr_moss_prompt"]
        if settings["asr_moss_hotwords"].strip():
            prompt += "热词提示：" + settings["asr_moss_hotwords"].strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": request["audio"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=[mono], return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[-1]
        limit = settings["asr_moss_max_new_tokens"]
        if prompt_length + limit > settings["asr_moss_max_context"]:
            raise ValueError("MOSS 输入与输出超过上下文上限；请缩短分块或调整 Token 上限。")
        kwargs = {
            "max_new_tokens": limit,
            "do_sample": settings["asr_moss_do_sample"],
            "num_beams": settings["asr_moss_num_beams"],
            "repetition_penalty": settings["asr_moss_repetition_penalty"],
            "use_cache": settings["asr_moss_use_cache"],
        }
        if kwargs["do_sample"]:
            kwargs.update(
                temperature=settings["asr_moss_temperature"],
                top_p=settings["asr_moss_top_p"],
                top_k=settings["asr_moss_top_k"],
            )
        kwargs.update(extra)

        class Streamer:
            def __init__(self, prompt_length, window_index):
                self.count = -prompt_length
                self.window_index = window_index

            def put(self, tokens):
                self.count += tokens.numel()
                if self.count > 0 and self.count % 64 == 0:
                    report(f"MOSS 分块 {self.window_index + 1}：已生成 {self.count} Tokens")

            def end(self):
                pass

        if kwargs["num_beams"] == 1:
            kwargs["streamer"] = Streamer(prompt_length, index)
        autocast = device.startswith("cuda") and dtype != torch.float32
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=autocast):
            outputs = model.generate(**inputs, **kwargs)
        generated = outputs[0][prompt_length:]
        if generated.numel() >= limit:
            raise ValueError("MOSS 达到输出 Token 上限，结果未保存；请增加上限或缩短分块。")
        transcript = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        transcripts.append(transcript)
        for row in parse_transcript(transcript, duration):
            row["start"] += window[0] / request["sample_rate"]
            row["end"] += window[0] / request["sample_rate"]
            # Model speaker IDs are local to each inference window, never global identities.
            row["speaker"] = f"c{index + 1:04d}:{row['speaker']}"
            rows.append(row)
    return {"segments": rows, "transcripts": transcripts, "language": request["language"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(json.loads(Path(args.request).read_text(encoding="utf-8")))
        path = Path(args.output)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except Exception as exc:
        print(f"MOSS {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
