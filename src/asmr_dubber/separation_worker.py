"""Isolated audio-separator bridge. Inference forbids implicit downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if job.get("device") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import torch

        # Some Windows driver builds report availability despite an empty device list.
        torch.cuda.is_available = lambda: False
    from audio_separator.separator import Separator

    if job.get("device") == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("已选择 CUDA，但独立环境没有可用 GPU；请明确选择 CPU 后重试。")

    models = Path(job["models"])
    models.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model_file_dir=str(models),
        output_dir=job["output"],
        output_format="WAV",
        use_soundfile=True,
        normalization_threshold=job.get("normalization", 1.0),
        use_autocast=job.get("autocast", False),
        log_level=logging.INFO,
        **job.get("params", {}),
    )
    if job["action"] == "download":
        import requests

        def atomic_download(url, output_path):
            target = Path(output_path)
            if not target.resolve().is_relative_to(models.resolve()):
                raise RuntimeError("模型清单包含目录外路径。")
            if target.is_file():
                return
            if not url.startswith("https://"):
                raise RuntimeError("模型下载必须使用 HTTPS。")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            try:
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for block in response.iter_content(1024 * 1024):
                            output.write(block)
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)

        separator.download_file_if_not_exists = atomic_download
        separator.download_model_files(job["model"])
        if job["model"] == "vocals_mel_band_roformer.ckpt":
            for name, expected in {
                "vocals_mel_band_roformer.ckpt": (
                    "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
                ),
                "vocals_mel_band_roformer.yaml": (
                    "b958b29c8f7195f0d86bee6759a33980db675c4ecaf2fcaa80fa125828e6cd38"
                ),
            }.items():
                with (models / name).open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                        raise RuntimeError("默认分离模型哈希不符，拒绝加载：" + name)
        # Populate metadata and verify the checkpoint actually loads on this host.
        separator.load_model(job["model"])
        catalog = separator.list_supported_model_files()
        (models / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        manifest = {}
        for path in models.iterdir():
            if (
                path.is_file()
                and not path.name.endswith((".integrity.json", ".partial"))
                and path.name != "catalog.json"
            ):
                with path.open("rb") as handle:
                    manifest[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        (models / f"{job['model']}.integrity.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        print("MODEL_READY", flush=True)
        return

    def local_only(url, output_path):
        if not Path(output_path).is_file():
            raise RuntimeError(
                "缺少本地模型或元数据，请在设置中显式下载：" + Path(output_path).name
            )

    separator.download_file_if_not_exists = local_only
    # Models and catalogs must already be prepared; never let helpers phone home.
    import requests

    def no_network(*args, **kwargs):
        raise RuntimeError("分离推理禁止隐式联网，请先完成模型下载。")

    requests.sessions.Session.request = no_network
    separator.load_model(job["model"])
    if job.get("device") == "cuda" and str(separator.torch_device) != "cuda":
        raise RuntimeError("所选后端未使用 CUDA；请在设置中明确选择可用设备。")
    if job["action"] == "serve":
        request = Path(job["request"])
        while True:
            if not request.is_file():
                time.sleep(0.05)
                continue
            item = json.loads(request.read_text(encoding="utf-8"))
            request.unlink()
            if item.get("stop"):
                return
            # audio-separator keeps the destination on both objects.
            separator.output_dir = item["output"]
            separator.model_instance.output_dir = item["output"]
            separate_one(separator, {**job, **item})
    else:
        separate_one(separator, job)


def separate_one(separator, job):
    files = separator.separate(job["input"])
    stem = job.get("vocal_stem", "Vocals").casefold()
    selected = [f for f in files if f"({stem})" in Path(f).stem.casefold()]
    if len(selected) != 1:
        raise RuntimeError(f"不能唯一定位人声输出 {stem}，请核对模型输出轨名称：{files}")
    candidate = Path(selected[0])
    if not candidate.is_absolute():
        candidate = Path(job["output"]) / candidate
    result = Path(job["result"])
    temporary = result.with_suffix(".partial")
    temporary.write_text(json.dumps({"vocals": str(candidate.resolve())}), encoding="utf-8")
    os.replace(temporary, result)


if __name__ == "__main__":
    main()
