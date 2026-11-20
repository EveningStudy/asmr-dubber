"""Name the DLL a Windows program cannot load.

Windows reports a missing dependency only as ``0xC0000135`` ("a DLL was not
found") without the file name.  Reading the PE import tables of the program and
of the DLLs it ships with finds the exact file, so the advice can be specific:
the Visual C++ runtime, Windows Media Foundation (absent on N/KN and many
stripped-down Windows builds), the NVIDIA driver, or a damaged backend.

Only file headers are read (through ``mmap``); nothing is executed or changed.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Iterable
from pathlib import Path

_API_SETS = ("api-ms-win-", "ext-ms-win-")
_MAX_FILES = 400

VC_RUNTIME = frozenset(
    {
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vcomp140.dll",
        "concrt140.dll",
    }
)
MEDIA_FOUNDATION = frozenset(
    {"mfplat.dll", "mfreadwrite.dll", "mf.dll", "mfcore.dll", "mfuuid.dll", "evr.dll"}
)
NVIDIA_DRIVER = frozenset({"nvcuda.dll", "nvml.dll", "nvapi64.dll"})

ADVICE_VC = "微软 VC++ 2015–2022 x64 运行库：点“修复微软 VC++ 运行库”，装完重启电脑再检查。"
ADVICE_MEDIA = (
    "Windows 媒体功能（Media Foundation）：N/KN 版或精简版 Windows 没有它，VC++ 运行库修复不了。"
    "到 Windows“设置 → 应用 → 可选功能 → 添加可选功能”安装“媒体功能包”（Media Feature Pack），"
    "重启后再检查；精简版系统可能装不上，只能换用完整版 Windows 或改用其它识别模型。"
)
ADVICE_NVIDIA = "NVIDIA 显卡驱动：从 NVIDIA 官网安装或更新驱动；没有 NVIDIA 显卡就改用其它后端。"
ADVICE_CUDA = "后端自带的 CUDA 库不完整：在下方选中这个后端点“修复”，重新安装。"
ADVICE_OTHER = (
    "如果是后端自带的文件，选中这个后端点“修复”重新安装；"
    "否则把这个文件名和日志发给开发者。不要从陌生网站下载 DLL。"
)


def advice_for(dll: str) -> str:
    name = dll.lower()
    if name in VC_RUNTIME:
        return ADVICE_VC
    if name in MEDIA_FOUNDATION:
        return ADVICE_MEDIA
    if name in NVIDIA_DRIVER:
        return ADVICE_NVIDIA
    if name.startswith(("cudart", "cublas", "cudnn", "cufft", "curand", "cusparse", "nvrtc")):
        return ADVICE_CUDA
    return ADVICE_OTHER


def pe_imports(path: Path) -> list[str]:
    """DLL names in the import table of a PE file; empty when it cannot be read."""

    try:
        with (
            path.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
        ):
            if data[:2] != b"MZ":
                return []
            header = struct.unpack_from("<I", data, 0x3C)[0]
            if data[header : header + 4] != b"PE\0\0":
                return []
            sections = struct.unpack_from("<H", data, header + 6)[0]
            optional_size = struct.unpack_from("<H", data, header + 20)[0]
            optional = header + 24
            magic = struct.unpack_from("<H", data, optional)[0]
            directories = optional + (112 if magic == 0x20B else 96)
            import_rva = struct.unpack_from("<I", data, directories + 8)[0]
            if not import_rva:
                return []
            table = []
            for index in range(sections):
                entry = optional + optional_size + 40 * index
                virtual_size, address = struct.unpack_from("<II", data, entry + 8)
                raw = struct.unpack_from("<I", data, entry + 20)[0]
                table.append((address, max(virtual_size, 1), raw))

            def offset(rva: int) -> int | None:
                for address, size, raw in table:
                    if address <= rva < address + size:
                        return rva - address + raw
                return None

            names: list[str] = []
            cursor = offset(import_rva)
            while cursor is not None and cursor + 20 <= len(data) and len(names) < 512:
                name_rva = struct.unpack_from("<I", data, cursor + 12)[0]
                if not name_rva:
                    break
                start = offset(name_rva)
                if start is None:
                    break
                end = data.find(b"\0", start, start + 260)
                if end < 0:
                    break
                names.append(bytes(data[start:end]).decode("ascii", "replace"))
                cursor += 20
            return names
    except (OSError, ValueError, struct.error):
        return []


def _system_directories() -> list[Path]:
    root = Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
    return [root / "System32", root]


def missing_dlls(binaries: Iterable[Path], extra_dirs: Iterable[Path] = ()) -> dict[str, list[str]]:
    """Map each DLL that cannot be found to the files that import it.

    DLLs found next to a scanned file or in ``extra_dirs`` are followed, since a
    bundled DLL's own dependencies fail the same way; system DLLs are trusted.
    """

    system = _system_directories()
    extra = [Path(path) for path in extra_dirs]
    path_dirs = [Path(part) for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    queue = [Path(path) for path in binaries if Path(path).is_file()]
    seen = {str(path).lower() for path in queue}
    missing: dict[str, list[str]] = {}
    while queue and len(seen) <= _MAX_FILES:
        current = queue.pop(0)
        for name in pe_imports(current):
            lowered = name.lower()
            # API sets are virtual; pythonXY.dll comes from the interpreter that loads the module.
            if lowered.startswith(_API_SETS) or (
                lowered.startswith("python3") and lowered.endswith(".dll")
            ):
                continue
            local = [current.parent, *extra]
            found = next((d / name for d in local if (d / name).is_file()), None)
            if found is not None:
                key = str(found).lower()
                if key not in seen:
                    seen.add(key)
                    queue.append(found)
                continue
            if any((d / name).is_file() for d in (*system, *path_dirs)):
                continue
            missing.setdefault(name, [])
            if current.name not in missing[name]:
                missing[name].append(current.name)
    return missing


def describe_missing(missing: dict[str, list[str]]) -> list[str]:
    """One readable line per missing DLL, grouped advice last."""

    if not missing:
        return ["DLL 依赖检查：所需 DLL 都能找到。"]
    lines = ["DLL 依赖检查：找到缺失的文件："]
    advice: list[str] = []
    for name in sorted(missing, key=str.lower):
        users = "、".join(missing[name][:3])
        lines.append(f"- 缺少 {name}（{users} 需要）")
        hint = advice_for(name)
        if hint not in advice:
            advice.append(hint)
    lines.extend(f"→ {hint}" for hint in advice)
    return lines
