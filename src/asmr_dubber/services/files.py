"""Ask the operating system for a folder; filesystem work stays in services."""

import base64
import os
import subprocess
from pathlib import Path

from ..platforms import open_directory


def folder():
    if os.name != "nt":
        return {"path": ""}
    script = (
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        "$shell=New-Object -ComObject Shell.Application; "
        "$folder=$shell.BrowseForFolder(0,'ASMR Dubber',0,0); "
        "if($folder){[Console]::Write($folder.Self.Path)}"
    )
    command = base64.b64encode(script.encode("utf-16-le")).decode()
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-EncodedCommand", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return {"path": result.stdout.strip()}


def open_output(path):
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise ValueError("Output directory does not exist")
    return {"path": str(open_directory(directory))}


def reference_upload(path):
    from .media import discard_upload
    from .settings import reference_upload as store

    stored = store(path)
    discard_upload(path)
    return {"path": stored}
