from pathlib import Path
from typing import Any

from ..constants import (
    INDEXTTS25_REQUIRED_DIRS,
    INDEXTTS25_REQUIRED_FILES,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
)
from ..model_packs import discover_model_packs, model_pack_directory
from ..platforms import runtime_executable_candidates


def indextts_installation_status(model_path: Any) -> str:
    text = str(model_path or "").strip()
    if not text:
        return "未填写 IndexTTS2 checkpoints 目录。"
    directory = Path(text).expanduser().resolve()
    executable = next(
        (
            candidate
            for candidate in runtime_executable_candidates(directory.parent, "indextts2")
            if candidate.is_file()
        ),
        None,
    )
    missing = sorted(
        [name for name in INDEXTTS_REQUIRED_FILES if not (directory / name).is_file()]
        + [name + "/" for name in INDEXTTS_REQUIRED_DIRS if not (directory / name).is_dir()]
    )
    if executable is None:
        return "运行环境未安装。请在“设备与模型”中安装 IndexTTS2。"
    if missing:
        preview = "、".join(missing[:5])
        suffix = f" 等 {len(missing)} 项" if len(missing) > 5 else ""
        return f"运行环境已安装，但模型不完整：缺少 {preview}{suffix}。"
    return f"IndexTTS2 已就绪：{executable}；模型目录：{directory}"


def indextts25_installation_status(model_path: Any) -> str:
    text = str(model_path or "").strip()
    if not text:
        return "未填写 IndexTTS-2.5 checkpoints 目录。"
    directory = Path(text).expanduser().resolve()
    executable = next(
        (
            candidate
            for candidate in runtime_executable_candidates(directory.parent, "python")
            if candidate.is_file()
        ),
        None,
    )
    missing = sorted(
        [name for name in INDEXTTS25_REQUIRED_FILES if not (directory / name).is_file()]
        + [name + "/" for name in INDEXTTS25_REQUIRED_DIRS if not (directory / name).is_dir()]
    )
    if executable is None:
        return "运行环境未安装。请在“设备与模型”中安装 IndexTTS-2.5。"
    if missing:
        preview = "、".join(missing[:5])
        suffix = f" 等 {len(missing)} 项" if len(missing) > 5 else ""
        return f"运行环境已安装，但模型不完整：缺少 {preview}{suffix}。"
    return f"IndexTTS-2.5 已就绪：{executable}；模型目录：{directory}"


def offline_model_pack_markdown() -> str:
    inbox = model_pack_directory()
    inspections = discover_model_packs(inbox)
    lines = [f"**本地模型包目录：** `{inbox}`"]
    if not inspections:
        lines.append("未发现 ZIP。把模型包原样放入该目录后再扫描。")
        return "  \n".join(lines)
    for inspection in inspections:
        manifest = inspection.manifest
        if manifest is None:
            lines.append(f"- **[无效] {inspection.archive.name}**：{inspection.error}")
            continue
        size = manifest.uncompressed_bytes / 1024**3
        state = "可导入" if inspection.compatible else f"不兼容：{inspection.error}"
        lines.append(f"- **[{state}] {manifest.display_name}** · {size:.2f} GiB")
    return "\n".join(lines)
