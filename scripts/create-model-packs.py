from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asmr_dubber import __version__  # noqa: E402
from asmr_dubber.constants import (  # noqa: E402
    ASMR_VAD_MODEL,
    DEFAULT_ALIGNER_MODEL,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    OPTIONAL_ASR_MODEL_REVISIONS,
)
from asmr_dubber.environment import cached_model_path  # noqa: E402
from asmr_dubber.model_packs import (  # noqa: E402
    ModelPackError,
    ModelPackSource,
    build_model_pack,
    imported_hf_snapshot_path,
    model_pack_directory,
)
from asmr_dubber.platforms import portable_home  # noqa: E402


@dataclass(frozen=True)
class PackDefinition:
    pack_id: str
    display_name: str
    platforms: tuple[str, ...]
    architectures: tuple[str, ...]


PACKS = {
    "parakeet-ja-windows": PackDefinition(
        pack_id="parakeet-ja-windows",
        display_name="Parakeet 日语 1.1B + 0.6B（Windows CrispASR）",
        platforms=("windows",),
        architectures=("x86_64",),
    ),
    "kotoba-whisper-v2.2": PackDefinition(
        pack_id="kotoba-whisper-v2.2",
        display_name="Kotoba-Whisper v2.2",
        platforms=("windows", "linux"),
        architectures=("any",),
    ),
    "faster-whisper-large-v2": PackDefinition(
        pack_id="faster-whisper-large-v2",
        display_name="Faster-Whisper large-v2",
        platforms=("windows", "linux"),
        architectures=("any",),
    ),
    "qwen3-forced-aligner": PackDefinition(
        pack_id="qwen3-forced-aligner",
        display_name="Qwen3 ForcedAligner 0.6B（独立时间戳对齐）",
        platforms=("windows", "linux"),
        architectures=("any",),
    ),
    "whisper-vad-asmr-onnx": PackDefinition(
        pack_id="whisper-vad-asmr-onnx",
        display_name="日语 ASMR 专用 Whisper VAD ONNX",
        platforms=("windows", "linux"),
        architectures=("any",),
    ),
    "indextts2-checkpoints": PackDefinition(
        pack_id="indextts2-checkpoints",
        display_name="IndexTTS2 官方 checkpoints",
        platforms=("windows", "linux"),
        architectures=("any",),
    ),
}


def _hf_sources(model_id: str) -> list[ModelPackSource]:
    revision = OPTIONAL_ASR_MODEL_REVISIONS[model_id]
    snapshot = cached_model_path(model_id)
    if snapshot is None:
        raise ModelPackError(f"本机没有完整的固定模型快照：{model_id}@{revision}")
    relative = imported_hf_snapshot_path(
        model_id,
        revision,
        home=Path("."),
    )
    return [ModelPackSource(snapshot, relative.as_posix())]


def _sources(pack_id: str) -> list[ModelPackSource]:
    home = portable_home()
    if pack_id == "parakeet-ja-windows":
        license_root = PROJECT_ROOT / "assets" / "model-licenses" / "parakeet"
        return [
            ModelPackSource(
                home / "models" / "parakeet" / "parakeet-ctc-1.1b-ja-f16.gguf",
                "models/parakeet/parakeet-ctc-1.1b-ja-f16.gguf",
            ),
            ModelPackSource(
                home / "models" / "parakeet" / "parakeet-tdt-0.6b-ja.gguf",
                "models/parakeet/parakeet-tdt-0.6b-ja.gguf",
            ),
            ModelPackSource(
                home / "runtimes" / "crispasr" / "bin",
                "runtimes/crispasr/bin",
            ),
            ModelPackSource(
                license_root / "APACHE-2.0.txt",
                "models/parakeet/APACHE-2.0.txt",
            ),
            ModelPackSource(
                license_root / "MODEL_NOTICES.txt",
                "models/parakeet/MODEL_NOTICES.txt",
            ),
        ]
    if pack_id == "kotoba-whisper-v2.2":
        return _hf_sources("kotoba-tech/kotoba-whisper-v2.2")
    if pack_id == "faster-whisper-large-v2":
        return _hf_sources("Systran/faster-whisper-large-v2")
    if pack_id == "qwen3-forced-aligner":
        return _hf_sources(DEFAULT_ALIGNER_MODEL)
    if pack_id == "whisper-vad-asmr-onnx":
        return _hf_sources(ASMR_VAD_MODEL)
    if pack_id == "indextts2-checkpoints":
        checkpoints = home / "runtimes" / "index-tts" / "checkpoints"
        relative_root = "runtimes/index-tts/checkpoints"
        required = [
            ModelPackSource(checkpoints / relative, f"{relative_root}/{relative}")
            for relative in sorted(INDEXTTS_REQUIRED_FILES)
        ]
        required.extend(
            ModelPackSource(checkpoints / relative, f"{relative_root}/{relative}")
            for relative in sorted(INDEXTTS_REQUIRED_DIRS)
        )
        for optional in ("LICENSE.txt", "LICENSE_ZH.txt", "README.md", "pinyin.vocab"):
            source = checkpoints / optional
            if source.is_file():
                required.append(ModelPackSource(source, f"{relative_root}/{optional}"))
        return required
    raise ModelPackError(f"未知模型包：{pack_id}")


def _progress(message: str, current: int, total: int) -> None:
    print(f"[{current}/{total}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从当前项目已安装的固定模型创建 ASMR Dubber 离线模型包。"
    )
    parser.add_argument(
        "packs",
        nargs="*",
        choices=tuple(PACKS),
        help="留空时创建全部模型包。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_pack_directory(PROJECT_ROOT),
        help="输出目录；默认是项目根目录 model-packs。",
    )
    parser.add_argument(
        "--pack-version",
        default=__version__,
        help="写入 manifest 和文件名的版本；默认使用程序版本。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已经存在的同名压缩包。",
    )
    arguments = parser.parse_args()
    selected = arguments.packs or list(PACKS)
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for pack_id in selected:
            definition = PACKS[pack_id]
            output = output_dir / (f"ASMR-Dubber-ModelPack-{pack_id}-v{arguments.pack_version}.zip")
            if output.exists() and not arguments.force:
                raise ModelPackError(f"输出已经存在：{output}；如需覆盖请添加 --force。")
            build_model_pack(
                output,
                pack_id=definition.pack_id,
                display_name=definition.display_name,
                pack_version=arguments.pack_version,
                platforms=definition.platforms,
                architectures=definition.architectures,
                sources=_sources(pack_id),
                log=print,
                progress=_progress,
            )
    except (ModelPackError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
