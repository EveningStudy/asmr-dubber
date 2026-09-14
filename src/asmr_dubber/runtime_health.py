"""Explicit runtime probes and opt-in repair; never manipulate project data."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .platforms import isolated_runtime_environment, portable_home, virtualenv_executable
from .storage import exclusive_file_lock

BACKENDS = {
    "parakeet_nemo": "Parakeet / CrispASR",
    "faster_whisper": "Faster-Whisper",
    "kotoba_whisper": "Kotoba-Whisper",
    "indextts2": "IndexTTS2",
    "indextts2_5": "IndexTTS-2.5",
}
_IMPORTS = {
    "faster_whisper": "import faster_whisper, ctranslate2",
    "kotoba_whisper": "import torch, torchaudio, transformers, accelerate",
    "indextts2": "import torch, torchaudio; import indextts.infer_v2",
    "indextts2_5": "import torch, torchaudio; import indextts.infer_v2_5",
}
_STATUS = {
    0xC0000135: (
        "Windows 找不到所需 DLL；可能缺少 VC++ 运行库、后端附带库或其下级依赖。"
        "退出码不能确定具体 DLL 名。"
    ),
    0xC000007B: "Windows 无法加载程序或 DLL：可能存在 32/64 位不匹配或文件损坏。",
    0xC0000139: "DLL 中找不到所需入口点：可能加载了不兼容版本的运行库。",
    0xC0000142: "DLL 初始化失败：请检查运行库、驱动和安全软件拦截记录。",
    0xC0000005: "原生程序发生访问冲突；可能涉及后端、驱动或内存，不能仅据此判定显存不足。",
    0xC000001D: "CPU 执行了不支持的指令：请检查后端对处理器指令集的要求。",
    0xC0000017: "系统无法分配内存；请检查可用内存及页面文件。",
}


def explain_runtime_error(message: str) -> str:
    """Explain known evidence, retaining the original diagnostic and unknown codes."""
    if "诊断（0x" in message:
        return message
    codes = re.findall(r"\b0[xX][0-9a-fA-F]{8}\b", message)
    codes += re.findall(
        r"(?:退出码|return[_ ]?code|exit[_ ]?code)\s*[:=：]?\s*(-?\d+)",
        message,
        flags=re.IGNORECASE,
    )
    if re.fullmatch(r"-?\d+", message.strip()):
        codes.append(message.strip())
    for token in codes:
        code = int(token, 16 if token.lower().startswith("0x") else 10) & 0xFFFFFFFF
        if code in _STATUS:
            return (
                f"{message}\n诊断（0x{code:08X}）：{_STATUS[code]}"
                "\n请前往“设置 → 设备与模型 → 运行依赖检测与修复”。不会自动安装或重跑任务。"
            )
    if any(
        word in message.lower()
        for word in ("dll load failed", "winerror 126", "winerror 127", "no module named")
    ):
        return (
            message + "\n运行依赖未能加载。请前往“设置 → 设备与模型 → 运行依赖检测与修复”，"
            "核对缺失包或库；不要从陌生网站下载 DLL。"
        )
    return message


def _probe(command: list[str], *, cwd: Path, runtime: str) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=isolated_runtime_environment(runtime),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return "未通过：启动/导入检测超过 60 秒；这不一定是缺库，请查看日志及安全软件。"
    except OSError as exc:
        return explain_runtime_error(f"未通过：{exc}")
    detail = (result.stdout + result.stderr).strip()[-3000:]
    if result.returncode:
        return explain_runtime_error(f"未通过：退出码 {result.returncode}\n{detail}")
    return "通过：启动/依赖导入成功（不代表模型推理已验证）。\n" + detail


def _check_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError("请选择支持的本地后端。")
    home = portable_home()
    runtime = {"indextts2": "index-tts", "indextts2_5": "index-tts-2.5"}.get(backend)
    if backend == "parakeet_nemo":
        binary = (
            home / "runtimes/crispasr/bin" / ("crispasr.exe" if os.name == "nt" else "crispasr")
        )
        if not binary.is_file():
            return "未安装：CrispASR 可执行文件不存在。请选择修复所选后端环境。"
        return _probe([str(binary), "--version"], cwd=home, runtime="crispasr")
    root = home / "runtimes" / runtime if runtime else home
    python = virtualenv_executable(root / ".venv", "python") if runtime else Path(sys.executable)
    if not python.is_file():
        return "未安装：独立 Python 运行环境不存在。请选择修复所选后端环境。"
    return _probe([str(python), "-c", _IMPORTS[backend]], cwd=root, runtime=runtime or backend)


def check_runtime_health(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError("请选择支持的本地后端。")
    with exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=1):
        lines = [f"检测后端：{BACKENDS[backend]}", "仅检测启动和依赖导入，不下载模型，不生成配音。"]
        if os.name == "nt":
            system = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32"
            for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "ucrtbase.dll"):
                state = (
                    "存在（仍需后端启动验证）"
                    if (system / name).is_file()
                    else "系统目录未发现（后端可能自带）"
                )
                lines.append(f"{name}：{state}")
            lines.append(
                "显卡驱动 nvcuda.dll："
                + (
                    "存在"
                    if (system / "nvcuda.dll").is_file()
                    else "未发现；CUDA 后端需检查 NVIDIA 驱动，CPU 使用不一定需要"
                )
            )
        lines.append(_check_backend(backend))
        lines.append("模型、项目和已有结果未修改。若仍缺 DLL，请提供日志和系统弹窗中的文件名。")
        return "\n".join(lines)


def repair_runtime_health(backend: str, action: str, confirmed: bool) -> str:
    if backend not in BACKENDS or action not in {"vc", "backend"}:
        raise ValueError("未知修复选项。")
    if not confirmed:
        return "尚未执行：请先确认下方安装范围。建议先点击检测，再选择修复方式。"
    if action == "backend":
        from .runtime_manager import install_backend

        result = install_backend(backend, force=True)
    else:
        if os.name != "nt":
            return (
                "微软 VC++ 运行库仅适用于 Windows；"
                "Linux 请修复所选后端环境或按发行版安装缺失系统库。"
            )
        script = Path(__file__).resolve().parents[2] / "scripts/windows/repair-vc-runtime.ps1"
        if not script.is_file():
            return "缺少便携修复脚本，请使用完整源码或便携包。"
        with exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=1):
            powershell = str(
                Path(os.environ.get("SYSTEMROOT", "C:/Windows"))
                / "System32/WindowsPowerShell/v1.0/powershell.exe"
            )
            completed = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            result = (completed.stdout + completed.stderr)[-5000:]
            if completed.returncode:
                return explain_runtime_error("运行库修复未完成：\n" + result)
    return result + "\n\n修复后复检：\n" + check_runtime_health(backend)
