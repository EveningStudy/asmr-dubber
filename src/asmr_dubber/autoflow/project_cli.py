"""Driving the ASMR Dubber CLI and reading its projects."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

from ..task_control import (
    check_cancelled,
)
from . import constants, processes, runlog, stages
from .constants import ReferenceEventCallback
from .domain import (
    REFERENCE_SELECTION_TIMEOUT_SECONDS,
    AudioSource,
    ToolPaths,
)
from .errors import StageFailedError, VideoPreparerError
from .fileio import atomic_copy
from .processes import run_process_captured

#: Optional callable given the CLI command name ("create", "analyze", "translate",
#: "synthesize", "mix", ...) just before it runs; the browser batch uses it to
#: show per-work stage progress.  Must never raise into the engine.
STAGE_LISTENER: Any = None


def announce_stage(arguments: Sequence[str]) -> None:
    if not arguments:
        return
    stage = arguments[0]
    runlog.emit_stage(stage)
    listener = STAGE_LISTENER
    if listener is not None:
        with suppress(Exception):
            listener(stage)


def run_asmr_cli(paths: ToolPaths, *arguments: str) -> dict[str, Any] | None:
    """Run one ``asmr-dubber`` command in-process when safe, else via CLI subprocess."""

    announce_stage(arguments)
    from .inprocess_pipeline import can_run_in_process, run_stage_in_process

    if can_run_in_process(paths):
        return run_stage_in_process(arguments)

    command = [*paths.cli_command, *arguments]

    if not arguments or arguments[0] not in stages.RESULT_STAGES:
        processes.run_process(command, cwd=paths.asmr_root)
        return None
    return stages.run_with_result(
        arguments[0],
        command,
        lambda full, tail: processes.run_process(full, cwd=paths.asmr_root, tail=tail),
        constants.STATE_ROOT / "results",
        failure=lambda stage, message, result: StageFailedError(stage, message, result),
    )


def run_budgeted_stage(
    paths: ToolPaths, state: dict[str, Any], stage: str, project_json: Path, *extra: str
) -> None:
    """Run a stage; a few lost sentences deliver the work as partial, more fail it."""

    try:
        run_asmr_cli(paths, stage, str(project_json), *extra)
    except StageFailedError as exc:
        if not stages.accept_partial(state, exc.stage, exc.result):
            raise StageFailedError(
                exc.stage, stages.failure_text(state, exc.stage, str(exc), exc.result), exc.result
            ) from exc
        runlog.log_event(f"{stage} 有少量句子失败，已在预算内继续：{state['partial'][-1]}")


def run_asmr_cli_captured(paths: ToolPaths, *arguments: str) -> str:
    announce_stage(arguments)
    command = [*paths.cli_command, *arguments]
    return run_process_captured(command, cwd=paths.asmr_root)


def configured_projects_root(paths: ToolPaths) -> Path:
    try:
        from asmr_dubber.user_settings import load_user_settings

        value = str(load_user_settings().projects_root or "").strip()
    except Exception as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 用户设置：{exc}") from exc
    return Path(value).expanduser().resolve() if value else (paths.asmr_home / "projects")


def known_project_manifests(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.rglob("project.json") if path.is_file()}


def create_asmr_project(
    paths: ToolPaths,
    video: Path,
    *,
    source_language: str = "ja",
    project_name: str | None = None,
) -> Path:
    projects_root = configured_projects_root(paths)
    from .inprocess_pipeline import can_run_in_process, create_project_in_process

    if can_run_in_process(paths):
        announce_stage(("create", str(video)))
        return create_project_in_process(
            video,
            projects_root=projects_root,
            source_language=source_language,
            project_name=project_name,
        )
    before = known_project_manifests(projects_root)
    started_ns = time.time_ns()
    output = run_asmr_cli_captured(
        paths,
        "create",
        str(video),
        "--projects-root",
        str(projects_root),
        "--source-language",
        source_language if source_language in {"ja", "en", "zh"} else "ja",
        *(("--project-name", project_name) if project_name else ()),
    )
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for raw_line in reversed(output.splitlines()):
        clean_line = ansi_escape.sub("", raw_line).strip().strip('"')
        if not clean_line.casefold().endswith("project.json"):
            continue
        candidate = Path(clean_line).expanduser().resolve()
        try:
            candidate.relative_to(projects_root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    after = known_project_manifests(projects_root)
    created = list(after - before)
    if not created:
        created = [path for path in after if path.stat().st_mtime_ns >= started_ns - 5_000_000_000]
    if not created:
        raise VideoPreparerError("ASMR Dubber 已返回成功，但没有找到新项目的 project.json。")
    created.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return created[0]


def read_project(project_json: Path) -> dict[str, Any]:
    try:
        return json.loads(project_json.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VideoPreparerError(f"无法读取 ASMR Dubber 项目：{project_json}: {exc}") from exc


def project_asset(project_json: Path, stored: Any) -> Path | None:
    text = str(stored or "").strip()
    if not text:
        return None
    candidate = (project_json.parent / text).resolve()
    try:
        candidate.relative_to(project_json.parent.resolve())
    except ValueError as exc:
        raise VideoPreparerError(f"项目输出路径越界：{text}") from exc
    return candidate if candidate.is_file() else None


def source_language_for_sources(sources: Sequence[AudioSource]) -> str:
    """Use the common source language; retain Japanese as the mixed-edition fallback."""

    languages = {str(item.source_language).casefold() for item in sources}
    return (
        next(iter(languages)) if len(languages) == 1 and languages <= {"ja", "en", "zh"} else "ja"
    )


def ensure_autoflow_project_outputs(project_json: Path) -> None:
    """Ensure an AutoFlow project can produce the promised bilingual mix."""

    try:
        from asmr_dubber.models import load_project, save_project

        project, project_dir = load_project(project_json)
        if project.settings.mix_output_mode == "stem":
            project.settings.mix_output_mode = "both"
            save_project(project, project_dir)
    except Exception as exc:
        raise VideoPreparerError(f"无法准备 ASMR Dubber 项目的混合输出设置：{exc}") from exc


def launch_asmr_ui(paths: ToolPaths, project_json: Path) -> None:
    if paths.powershell:
        with suppress(OSError):
            subprocess.run(
                [
                    paths.powershell,
                    "-NoProfile",
                    "-Command",
                    "Set-Clipboard -Value $args[0]",
                    str(project_json),
                ],
                check=False,
                capture_output=True,
            )
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    try:
        subprocess.Popen(
            list(paths.ui_command),
            cwd=paths.asmr_root,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise VideoPreparerError(f"无法启动 ASMR Dubber 网页：{exc}") from exc


def project_reference_id(project_json: Path) -> str:
    project = read_project(project_json)
    settings = project.get("settings") or {}
    return str(settings.get("tts_reference_sentence_id") or "").strip()


def project_has_external_reference(project_json: Path) -> bool:
    """Return whether the current 0.7.x project already has a usable reference."""

    project = read_project(project_json)
    settings = project.get("settings") or {}
    source = str(settings.get("tts_reference_source") or "project_sentence").strip()
    if settings.get("tts_backend") in {"indextts2", "indextts2_5"}:
        source = str(settings.get("tts_index_speaker_source") or "current_sentence")
    if source != "external":
        return False
    audio = Path(str(settings.get("tts_external_reference_audio") or "")).expanduser()
    if not audio.is_file():
        raise VideoPreparerError(
            f"已选择外部音色参考，但文件不存在：{audio}。请重新选择，不会自动替换参考。"
        )
    return True


def wait_for_reference(
    paths: ToolPaths,
    project_json: Path,
    *,
    timeout_seconds: int = REFERENCE_SELECTION_TIMEOUT_SECONDS,
    event_callback: ReferenceEventCallback | None = None,
    event_context: dict[str, Any] | None = None,
    launch_ui: bool = True,
) -> bool:
    if timeout_seconds <= 0:
        runlog.emit("\n  未指定统一参考音频，将使用主程序自动推荐的清晰片段。")
        return True

    request_id = uuid.uuid4().hex
    deadline_epoch = time.time() + timeout_seconds

    def emit(kind: str, **details: Any) -> None:
        if event_callback is None:
            return
        payload = {
            **(event_context or {}),
            "kind": kind,
            "request_id": request_id,
            "project_json": str(project_json.resolve()),
            "timeout_seconds": timeout_seconds,
            "deadline_epoch": deadline_epoch,
            **details,
        }
        try:
            event_callback(payload)
        except Exception as exc:
            # Browser/UI notifications are optional. A broken or closed page
            # must never stop the actual AutoFlow task.
            with suppress(Exception):
                runlog.log_event(f"参考音频页面通知失败，任务继续：{type(exc).__name__}: {exc}")

    emit("ready")
    runlog.emit("\n[4/5] 等你在网页中选择统一音色参考")
    runlog.emit(f"  项目：{project_json}")
    if launch_ui:
        runlog.emit("  项目路径已复制到剪贴板。网页会预选最近项目；请点击“打开项目”。")
        runlog.emit("  选择清晰片段后，必须点击“设为项目音色参考”。")
        runlog.emit("  如果还修改了表格，请同时点击“保存校对表格”。")
        launch_asmr_ui(paths, project_json)
    else:
        runlog.emit("  可在批量处理页展开当前任务下方的参考音频选择器。")
    runlog.emit(f"  {timeout_seconds} 秒内未选择则使用自动推荐的参考音频。")

    deadline = time.monotonic() + timeout_seconds
    notice_step = 15 if timeout_seconds <= 120 else 30
    next_notice = time.monotonic() + notice_step
    while True:
        reference_id = project_reference_id(project_json)
        if reference_id:
            runlog.emit(f"已检测到统一参考：{reference_id}")
            emit("selected", source="project_sentence", sentence_id=reference_id)
            return True
        if project_has_external_reference(project_json):
            runlog.emit("已检测到 ASMR Dubber 设置中的外部参考音频。")
            emit("selected", source="external")
            return True
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            runlog.emit("等待时间结束，将使用 ASMR Dubber 自动推荐的参考音频。")
            emit("timeout", source="automatic")
            return True
        if now >= next_notice:
            secs = max(1, math.ceil(remaining))
            if secs >= 60:
                mins, rem_s = divmod(secs, 60)
                span = f"{mins} 分 {rem_s} 秒" if rem_s else f"{mins} 分钟"
            else:
                span = f"{secs} 秒"
            runlog.emit(f"  仍在等待参考音频，剩余约 {span}……")
            next_notice = now + notice_step
        check_cancelled()
        time.sleep(min(2.0, remaining))


def project_target_language(project_json: Path) -> str:
    project = read_project(project_json)
    return str(project.get("settings", {}).get("tts_target_language") or "zh")


def project_source_language(project_json: Path) -> str:
    project = read_project(project_json)
    return str(project.get("source_language") or "ja")


def extract_shared_reference(project_json: Path, destination: Path) -> dict[str, str]:
    """把首个分轨项目的统一参考固化为后续项目可复用的外部参考。"""

    try:
        from asmr_dubber.models import load_project, save_project
        from asmr_dubber.voice_reference import (
            prepare_index_speaker_reference,
            prepare_voice_reference,
            shared_reference_sentence,
        )

        project, project_dir = load_project(project_json)
        sentence = shared_reference_sentence(project)
        source = (project_dir / project.source.path).resolve()
        prepare = (
            prepare_index_speaker_reference
            if project.settings.tts_backend in {"indextts2", "indextts2_5"}
            else prepare_voice_reference
        )
        reference = prepare(project, project_dir, source, sentence)
        save_project(project, project_dir)
    except Exception as exc:
        raise VideoPreparerError(f"无法固化分轨共用参考音频：{exc}") from exc

    atomic_copy(reference.path, destination)
    return {
        "audio": str(destination.resolve()),
        "text": reference.text,
        "language": reference.language,
    }


def apply_shared_reference(project_json: Path, reference: dict[str, str]) -> None:
    audio = Path(str(reference.get("audio") or "")).expanduser().resolve()
    if not audio.is_file():
        raise VideoPreparerError(f"分轨共用参考音频已经不存在：{audio}")
    try:
        from asmr_dubber.models import load_project, save_project

        project, project_dir = load_project(project_json)
        project.settings.tts_reference_source = "external"
        project.settings.tts_external_reference_audio = str(audio)
        project.settings.tts_external_reference_text = str(reference.get("text") or "")
        language = str(reference.get("language") or "auto")
        project.settings.tts_external_reference_language = cast(
            Literal["auto", "ja", "en", "zh"],
            language if language in {"ja", "en", "zh"} else "auto",
        )
        # IndexTTS2 有独立的音色参考选择；情绪仍可跟随当前句。
        project.settings.tts_index_speaker_source = "external"
        save_project(project, project_dir)
    except Exception as exc:
        raise VideoPreparerError(f"无法给分轨项目设置共用参考音频：{exc}") from exc
