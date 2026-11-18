"""In-process ASMR Dubber pipeline execution for AutoFlow batch queues (W17 / C1).

Eliminates per-stage Python CLI subprocess startup while preserving exact stage-result
reporting, cancellation, and fallback compatibility with custom CLI test stubs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .. import pipeline
from ..errors import AsmrDubberError, OperationCancelledError, ProjectError
from ..languages import SpeechSourceLanguage
from ..models import ProjectSettings
from ..stage_result import STAGE_LABELS, build_result
from ..subtitles import SubtitleLanguage
from ..task_control import check_cancelled
from ..user_settings import load_user_settings
from . import processes, runlog
from .domain import ToolPaths
from .errors import StageFailedError, VideoPreparerError


def can_run_in_process(paths: ToolPaths) -> bool:
    """Use in-process execution unless a caller/test overrode ``run_process`` or CLI."""
    if not processes.uses_default_process_runners():
        return False
    cmd = tuple(str(part) for part in paths.cli_command)
    if not cmd:
        return False
    exe_name = Path(cmd[0]).stem.casefold()
    cur_exe = Path(sys.executable).resolve()
    try:
        cmd_exe = Path(cmd[0]).expanduser().resolve()
        if cmd_exe == cur_exe:
            return True
    except OSError:
        pass
    if exe_name in {"asmr-dubber", "asmr_dubber"}:
        return True
    return len(cmd) >= 3 and cmd[1] == "-m" and cmd[2] in {"asmr_dubber.cli", "asmr_dubber"}


class _RunlogProgress:
    def __init__(self) -> None:
        self.last = ""

    def __call__(self, message: str, current: int, total: int) -> None:
        check_cancelled()
        label = f"[{current}/{total}] {message}" if total else message
        if label != self.last:
            runlog.emit(f"  {label}")
            self.last = label


def _flag_values(args: Sequence[str], flag: str, short: str | None = None) -> list[str]:
    values: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token in {flag, *([short] if short else [])} and i + 1 < len(args):
            values.append(args[i + 1])
            i += 2
            continue
        i += 1
    return values


def _last_flag(args: Sequence[str], flag: str, short: str | None = None) -> str | None:
    vals = _flag_values(args, flag, short)
    return vals[-1] if vals else None


def create_project_in_process(
    input_media: Path,
    *,
    projects_root: Path,
    source_language: str = "ja",
    project_name: str | None = None,
) -> Path:
    check_cancelled()
    lang = source_language if source_language in {"ja", "en", "zh"} else "ja"
    try:
        settings: ProjectSettings = load_user_settings().to_project_settings()
        _, directory = pipeline.create_project(
            input_media,
            projects_root,
            settings=settings,
            project_name=project_name,
            source_language=cast(SpeechSourceLanguage, lang),
        )
    except OperationCancelledError:
        raise
    except (AsmrDubberError, ValueError) as exc:
        raise VideoPreparerError(f"创建项目失败：{exc}") from exc
    manifest = (directory / "project.json").resolve()
    runlog.emit(str(manifest))
    return manifest


def run_stage_in_process(arguments: Sequence[str]) -> dict[str, Any] | None:
    """Execute one ASMR Dubber CLI command directly in the current Python process."""
    check_cancelled()
    if not arguments:
        return None
    stage = arguments[0]
    rest = list(arguments[1:])
    progress = _RunlogProgress()

    if stage == "create":
        if not rest:
            raise VideoPreparerError("create 缺少输入媒体路径。")
        media = Path(rest[0]).expanduser().resolve()
        root_str = _last_flag(rest[1:], "--projects-root")
        root = Path(root_str).expanduser().resolve() if root_str else media.parent
        lang = _last_flag(rest[1:], "--source-language") or "ja"
        name = _last_flag(rest[1:], "--project-name")
        create_project_in_process(
            media, projects_root=root, source_language=lang, project_name=name
        )
        return None

    if stage == "import-transcript":
        if len(rest) < 2:
            raise VideoPreparerError("import-transcript 缺少项目或字幕路径。")
        project_path = Path(rest[0])
        transcript_path = Path(rest[1])
        kind = _last_flag(rest[2:], "--kind") or "source"
        timing = (_last_flag(rest[2:], "--timing") or "estimate").replace("-", "_")
        try:
            project, directory = pipeline.reload_project(project_path)
            language = "zh" if kind == "zh" else project.source_language
            pipeline.import_project_transcript(
                project,
                directory,
                transcript_path=transcript_path,
                plain_timing=timing,
                script_language=language,
                progress=progress,
            )
        except OperationCancelledError:
            raise
        except (AsmrDubberError, ValueError) as exc:
            raise VideoPreparerError(f"导入字幕失败：{exc}") from exc
        return None

    if stage == "analyze":
        if not rest:
            raise VideoPreparerError("analyze 缺少项目路径。")
        project_path = Path(rest[0])
        force = "--force" in rest[1:]
        try:
            project, directory = pipeline.reload_project(project_path)
            pipeline.analyze_project(project, directory, force=force, progress=progress)
        except OperationCancelledError:
            raise
        except AsmrDubberError as exc:
            raise VideoPreparerError(f"语音识别失败：{exc}") from exc
        return None

    if stage == "subtitles":
        if not rest:
            raise VideoPreparerError("subtitles 缺少项目路径。")
        project_path = Path(rest[0])
        lang_raw = _last_flag(rest[1:], "--language", "-l") or "bilingual"
        if lang_raw == "ja":
            lang_raw = "source"
        variant = _last_flag(rest[1:], "--output-variant")
        try:
            project, directory = pipeline.reload_project(project_path)
            pipeline.generate_subtitles(
                project,
                directory,
                language=cast(SubtitleLanguage, lang_raw),
                progress=progress,
                output_variant=variant,
            )
        except OperationCancelledError:
            raise
        except (AsmrDubberError, ValueError) as exc:
            raise VideoPreparerError(f"生成字幕失败：{exc}") from exc
        return None

    if stage in STAGE_LABELS:
        if not rest:
            raise VideoPreparerError(f"{stage} 缺少项目路径。")
        project_path = Path(rest[0])
        force = "--force" in rest[1:]
        try:
            project, directory = pipeline.reload_project(project_path)
            if stage == "translate":
                pipeline.translate_project(project, directory, force=force, progress=progress)
            elif stage == "synthesize":
                sentence_ids = _flag_values(rest[1:], "--sentence", "-s") or None
                pipeline.synthesize_project(
                    project,
                    directory,
                    force=force,
                    sentence_ids=sentence_ids,
                    progress=progress,
                )
            elif stage == "mix":
                variant = _last_flag(rest[1:], "--output-variant")
                allow_missing = "--allow-missing" in rest[1:]
                pipeline.mix_project(
                    project,
                    directory,
                    progress=progress,
                    output_variant=variant,
                    allow_missing=allow_missing,
                )
        except (OperationCancelledError, KeyboardInterrupt):
            raise
        except (AsmrDubberError, ProjectError, ValueError) as exc:
            result = build_result(stage, project_path, exc)
            detail = str(result.get("error") or exc).strip()
            label = STAGE_LABELS[stage]
            raise StageFailedError(stage, f"{label}失败：{detail}"[:1500], result) from exc
        return build_result(stage, project_path)

    return None
