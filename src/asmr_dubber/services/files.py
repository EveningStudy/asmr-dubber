"""Ask the operating system for a folder or media file; filesystem work stays in services."""

import base64
import os
import subprocess
from pathlib import Path

from ..platforms import open_directory


def _powershell_dialog(script: str) -> str:
    command = base64.b64encode(script.encode("utf-16-le")).decode()
    result = subprocess.run(
        ["powershell.exe", "-Sta", "-NoProfile", "-EncodedCommand", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return result.stdout.strip()


def folder():
    if os.name != "nt":
        return {"path": ""}
    script = (
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        "$shell=New-Object -ComObject Shell.Application; "
        "$folder=$shell.BrowseForFolder(0,'ASMR Dubber',0,0); "
        "if($folder){[Console]::Write($folder.Self.Path)}"
    )
    return {"path": _powershell_dialog(script)}


def media_file():
    if os.name != "nt":
        return {"path": "", "name": "", "native": False}
    script = (
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dlg=New-Object System.Windows.Forms.OpenFileDialog; "
        "$dlg.Title='ASMR Dubber'; "
        "$dlg.Filter='Audio & Video|*.wav;*.flac;*.mp3;*.m4a;*.aac;*.ogg;*.opus;"
        "*.mp4;*.mkv;*.mov;*.webm;*.avi|All Files|*.*'; "
        "if($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
        "{[Console]::Write($dlg.FileName)}"
    )
    chosen = _powershell_dialog(script)
    return {
        "path": chosen,
        "name": Path(chosen).name if chosen else "",
        "native": True,
    }


def open_output(path):
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise ValueError("输出文件夹不存在。")
    return {"path": str(open_directory(directory))}


def reference_upload(path):
    from .media import discard_upload
    from .settings import reference_upload as store

    stored = store(path)
    discard_upload(path)
    return {"path": stored}
