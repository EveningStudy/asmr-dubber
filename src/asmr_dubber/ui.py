from __future__ import annotations

import inspect
import io
import ipaddress
import logging
import os
import queue
import secrets
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from . import __version__
from .app_logging import application_log_path, configure_logging, recent_log_text
from .autoflow.ui_components import (
    QUEUE_LIST_CSS,
    QUEUE_LIST_JS,
    QUEUE_LIST_TEMPLATE,
    TRACK_LIST_CSS,
    TRACK_LIST_JS,
    TRACK_LIST_TEMPLATE,
)
from .autoflow.ui_services import (
    LAYOUT_LABELS as AUTOFLOW_LAYOUT_LABELS,
)
from .autoflow.ui_services import (
    MODE_LABELS as AUTOFLOW_MODE_LABELS,
)
from .autoflow.ui_services import (
    add_plan_to_queue,
    background_preview_path,
    build_plan_for_ui,
    edit_plan_for_ui,
    open_output_directory,
    preview_edition_for_ui,
    queue_items_for_ui,
    remove_plan_from_queue,
    reorder_queue_for_ui,
    reorder_tracks_for_ui,
    replace_plan_in_queue,
    scan_for_ui,
    set_track_subtitle_for_ui,
    toggle_plan_rebuild,
)
from .autoflow.ui_services import (
    recent_log_text as recent_autoflow_log_text,
)
from .autoflow.ui_services import (
    run_queue as run_autoflow_queue,
)
from .constants import (
    DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
    DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY,
    DEFAULT_CHINESE_RELATIVE_LOUDNESS_DB,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    MAX_CHINESE_AUTO_SPEED,
)
from .errors import InstallPausedError, OperationCancelledError
from .languages import SpeechSourceLanguage, source_language_label
from .model_packs import discover_model_packs, import_discovered_model_packs, model_pack_directory
from .model_registry import ASR_BACKENDS, CLONE_MODE_LABELS, TTS_BACKENDS
from .models import ProjectSettings, settings_for_source_language
from .platforms import portable_home, require_supported_platform, runtime_executable_candidates
from .runtime_manager import (
    asmr_vad_status,
    available_asr_review_choices,
    available_backend_models_markdown,
    available_timestamp_review_choices,
    backend_catalog_rows,
    hardware_markdown,
    install_backend,
    installable_backend_ids,
    recommended_stack_markdown,
    refresh_hardware,
)
from .task_control import CancellationToken, cancellation_scope
from .translation import default_translation_prompt
from .ui_services import (
    TABLE_HEADERS,
    TABLE_TYPES,
    ProjectView,
    analyze,
    apply_global_settings,
    create_project,
    import_transcript_data,
    load_view,
    mix,
    open_project_directory,
    preview_edge_tts_voice,
    preview_reference,
    recent_projects,
    reference_picker,
    save_table,
    select_autoflow_external_reference,
    select_autoflow_project_reference,
    select_reference,
    stage_for_ui,
    subtitles,
    synthesize,
    translate,
    ui_stage_directory,
)
from .user_settings import (
    PROVIDER_PRESETS,
    UserSettings,
    api_key_status,
    clear_api_key,
    clear_service_key,
    load_user_settings,
    save_api_key,
    save_service_key,
    save_user_settings,
    service_key_status,
    store_reference_audio,
)

APP_CSS = """
:root, body, .gradio-container {
    --font: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif;
    font-family: var(--font) !important;
}
.gradio-container {
    width: 100% !important;
    max-width: 1440px !important;
    min-width: 0 !important;
    margin-inline: auto !important;
}
:root .gradio-container > .main.fillable {
    padding-left: clamp(.75rem, 3vw, 2rem) !important;
    padding-right: clamp(.75rem, 3vw, 2rem) !important;
}
.gradio-container main,
.gradio-container .main,
.gradio-container .column,
.gradio-container .row,
.gradio-container [role="tabpanel"] { min-width: 0 !important; }
.gradio-container [role="tablist"] {
    max-width: 100%;
    overflow-x: auto !important;
    overflow-y: hidden;
    scrollbar-width: thin;
}
#asmr-dubber-product-marker { margin: .25rem 0 1rem; }
#asmr-dubber-product-marker h1 { margin: 0; font-size: clamp(1.75rem, 5vw, 2.45rem); }
#asmr-dubber-product-marker p { margin: .35rem 0 0; color: var(--body-text-color-subdued); }
#workflow-hint { border-left: 4px solid var(--color-accent); padding-left: .85rem; }
#workflow-hint #workflow-hint { border-left: 0 !important; padding-left: 0 !important; }
#project-start {
    border: 1px solid var(--border-color-primary) !important;
    border-left: 4px solid var(--color-accent) !important;
    border-radius: 10px !important;
    padding: clamp(.75rem, 2vw, 1.15rem) !important;
}
#project-start #project-start {
    border: 0 !important;
    padding: 0 !important;
}
#project-summary {
    margin: .25rem 0 .75rem !important;
    padding: .7rem .85rem !important;
    border-radius: 8px;
    background: var(--block-background-fill);
    color: var(--body-text-color-subdued);
}
#project-summary p { margin: 0 !important; }
#project-summary #project-summary {
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}
.workflow-actions button { min-height: 44px; font-weight: 600; }
#project-status {
    border-left: 4px solid var(--color-accent) !important;
    padding: .65rem .85rem !important;
    background: var(--block-background-fill);
}
#project-status p { margin: 0 !important; white-space: pre-wrap; }
#project-status #project-status { border-left: 0 !important; padding: 0 !important; }
#autoflow-start {
    border: 1px solid var(--border-color-primary) !important;
    border-left: 4px solid var(--color-accent) !important;
    border-radius: 10px !important;
    padding: clamp(.75rem, 2vw, 1.15rem) !important;
}
#autoflow-start #autoflow-start {
    border: 0 !important;
    padding: 0 !important;
}
#autoflow-status {
    border-left: 4px solid var(--color-accent) !important;
    padding: .65rem .85rem !important;
}
#autoflow-options {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 10px !important;
    padding: clamp(.75rem, 2vw, 1.15rem) !important;
    margin-top: .75rem !important;
}
#autoflow-options #autoflow-options {
    border: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
#autoflow-options-note,
#autoflow-settings-note {
    border-left: 4px solid var(--color-accent) !important;
    padding: .65rem .85rem !important;
    background: var(--block-background-fill);
    border-radius: 6px;
}
#autoflow-options-note p,
#autoflow-settings-note p { margin: 0 !important; }
#autoflow-options-note #autoflow-options-note,
#autoflow-settings-note #autoflow-settings-note {
    border-left: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}
.autoflow-section-title { margin-top: .35rem !important; }
.autoflow-table table { min-width: 720px; }
.status-panel textarea, .diagnostics-panel textarea { font-family: var(--font); }
.optional-section { opacity: .96; }
.sentence-table, .backend-table, .profile-table {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}
.sentence-table table { min-width: 760px; }
.backend-table table { min-width: 900px; }
.profile-table table { min-width: 640px; }
.gradio-container code { overflow-wrap: anywhere; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible {
    outline: 3px solid var(--color-accent) !important;
    outline-offset: 2px;
}
footer { display: none !important; }
@media (max-width: 640px) {
    .mobile-stack { flex-direction: column !important; }
    .mobile-stack > * { width: 100% !important; min-width: 0 !important; }
}
"""

CATALOG_HEADERS = ["后端", "支持级别", "设备兼容性", "状态", "说明", "磁盘占用"]
PROFILE_MARKDOWN = """
| 安装档位 | 包含内容 | 安装后约占用 | 建议可用空间 |
|---|---|---:|---:|
| 基础 | 程序、网页界面和外部 API 支持 | 2 GB | 5 GB |
| 推荐 | 基础 + Parakeet 1.1B/0.6B；NVIDIA 设备加 IndexTTS2 | 24–28 GB | 35 GB |
| 进阶 | 下方列出的 7 个固定模型及其运行依赖 | 33–39 GB | 50 GB |

**进阶固定模型：** Parakeet CTC 1.1B JA GAL、Parakeet TDT/CTC 0.6B JA、
`kotoba-tech/kotoba-whisper-v2.2`、`Systran/faster-whisper-large-v2`、日语 ASMR 专用
Whisper VAD ONNX、阿里 Qwen3 ForcedAligner 0.6B，以及 NVIDIA GPU 设备上的 IndexTTS2
checkpoints。不会自动下载 Kotoba v2.0/v2.1、Faster-Whisper large-v3 或其它识别模型。

Windows 双击 `ASMR-Dubber-Setup.exe`；Linux 运行
`bash scripts/linux/setup.sh 推荐`。英文 `Core`、`Recommended`、`Advanced` 仅作为旧命令行兼容别名。
"""

_INSTALLABLE = set(installable_backend_ids())
_PRIVATE_API: Any = False
_LLM_TRANSLATION_PROVIDERS = frozenset(
    {
        "deepseek",
        "bailian",
        "doubao",
        "openai",
        "anthropic",
        "gemini",
        "openai_compatible",
    }
)
_LOUDNESS_MODE_CHOICES = [
    ("跟随对应原声（推荐）", "source"),
    ("所有中文保持统一音量", "uniform"),
    ("保留 TTS 原始音量", "raw"),
]
_LOUDNESS_MODE_DESCRIPTIONS = {
    "source": "自动测量每句对应的原声音量，再按下面的相对值调整中文，适合保留场景强弱变化。",
    "uniform": "不跟随原片段强弱，把所有中文句子调整到同一个目标响度。",
    "raw": "不做逐句响度规范化，保留 TTS 输出音量；只应用手动微调和混音峰值保护。",
}
_NATIVE_VAD_LABELS = {
    "parakeet_nemo": "使用 Parakeet/CrispASR 自带 Silero VAD",
    "faster_whisper": "使用 Faster-Whisper 自带 Silero VAD",
}
logger = logging.getLogger(__name__)


def _speech_language(value: Any) -> SpeechSourceLanguage:
    return "en" if str(value or "ja") == "en" else "ja"


def _asr_backend_choices(language: SpeechSourceLanguage) -> list[tuple[str, str]]:
    if language == "en":
        spec = ASR_BACKENDS["faster_whisper"]
        return [(spec.label, spec.id)]
    return [(spec.label, key) for key, spec in ASR_BACKENDS.items()]


def _asr_models_for_language(
    backend_id: str,
    language: SpeechSourceLanguage,
) -> list[str]:
    if language == "en":
        if backend_id != "faster_whisper":
            return ["large-v2"]
        return [
            model
            for model in ASR_BACKENDS[backend_id].models
            if not model.startswith("kotoba-tech/")
        ]
    return list(ASR_BACKENDS[backend_id].models)


def asr_vad_choices(
    backend_id: str,
    source_language: SpeechSourceLanguage = "ja",
) -> list[tuple[str, str]]:
    """Return only VAD modes that can run for the selected backend right now."""

    choices = [("不做 VAD 预处理", "off")]
    native = _NATIVE_VAD_LABELS.get(str(backend_id))
    if native:
        choices.append((native, "backend"))
    if source_language == "ja" and asmr_vad_status().state == "ready":
        choices.append(("日语 ASMR 专用 Whisper VAD（独立预处理）", "asmr"))
    return choices


def _review_control_state(
    settings: ProjectSettings | UserSettings,
    source_language: SpeechSourceLanguage = "ja",
) -> tuple[
    list[tuple[str, str]],
    list[str],
    str | None,
    list[tuple[str, str]],
    str | None,
]:
    review_choices = available_asr_review_choices(settings)
    if source_language == "en":
        review_choices = [
            choice
            for choice in review_choices
            if choice[1].startswith("faster_whisper|") and "kotoba-tech/" not in choice[1]
        ]
    review_values = {value for _, value in review_choices}
    selected = [value for value in settings.asr_review_models if value in review_values]
    primary = f"{settings.asr_backend}|{settings.asr_model}"
    text_priority = settings.asr_review_text_priority_model
    if text_priority not in review_values:
        text_priority = primary if primary in review_values else None
    if text_priority is None and review_choices:
        text_priority = review_choices[0][1]

    timestamp_choices = available_timestamp_review_choices(settings)
    if source_language == "en":
        timestamp_choices = [
            choice
            for choice in timestamp_choices
            if choice[1].startswith("qwen_forced_aligner|")
            or (choice[1].startswith("faster_whisper|") and "kotoba-tech/" not in choice[1])
        ]
    timestamp_values = {value for _, value in timestamp_choices}
    timestamp_priority = settings.asr_review_timestamp_priority_model
    if timestamp_priority not in timestamp_values:
        timestamp_priority = primary if primary in timestamp_values else None
    if timestamp_priority is None and timestamp_choices:
        timestamp_priority = timestamp_choices[0][1]
    return (
        review_choices,
        selected,
        text_priority,
        timestamp_choices,
        timestamp_priority,
    )


class DownloadController:
    """Pause the active resumable installer without cancelling unrelated work."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._active: str | None = None

    def begin(self, label: str) -> None:
        with self._lock:
            self.cancel_event.clear()
            self._active = label

    def finish(self, label: str) -> None:
        with self._lock:
            if self._active == label:
                self._active = None

    def pause(self) -> str:
        with self._lock:
            if self._active is None:
                return "当前没有下载任务。"
            label = self._active
            self.cancel_event.set()
        return f"正在暂停 {label}；已完成的文件会保留，下次安装将继续。"


class ProjectTaskController:
    """Coordinate the one active project operation and its child processes."""

    def __init__(self, task_kind: str = "项目任务") -> None:
        self.cancel_event = CancellationToken()
        self._lock = threading.Lock()
        self._active: str | None = None
        self._task_kind = task_kind

    def begin(self, label: str) -> None:
        with self._lock:
            self.cancel_event.clear()
            self._active = label

    def finish(self, label: str) -> None:
        with self._lock:
            if self._active == label:
                self._active = None

    def cancel(self) -> str:
        with self._lock:
            if self._active is None:
                return f"当前没有正在执行的{self._task_kind}。"
            label = self._active
        self.cancel_event.set()
        return f"正在取消“{label}”… 已经完成并保存的内容会保留。"


class _AutoFlowLogWriter(io.TextIOBase):
    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit

    def write(self, value: str) -> int:
        text = str(value or "").replace("\r", "\n")
        for line in text.splitlines():
            if line.strip():
                self._emit(line.rstrip())
        return len(value)

    def flush(self) -> None:
        return None


def _autoflow_log_events(
    queue_payload: Any,
    controller: ProjectTaskController,
    *,
    heartbeat_seconds: float = 2.0,
) -> Iterator[tuple[str, bool, bool, str, list[str], dict[str, Any] | None]]:
    """Run one AutoFlow queue while streaming bounded, cancellable UI logs."""

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    lines: list[str] = []
    started = time.monotonic()

    def emit(message: str) -> None:
        events.put(("log", message))

    def append(message: str) -> None:
        lines.extend(line.rstrip() for line in str(message or "").splitlines() if line.strip())
        if len(lines) > 500:
            lines[:] = ["…较早的日志已省略…", *lines[-499:]]

    def run() -> None:
        writer = _AutoFlowLogWriter(emit)
        try:
            with (
                cancellation_scope(controller.cancel_event),
                redirect_stdout(writer),
                redirect_stderr(writer),
            ):
                result, outputs = run_autoflow_queue(
                    queue_payload,
                    cancel_event=controller.cancel_event,
                    reference_event_callback=lambda payload: events.put(
                        ("reference", dict(payload))
                    ),
                )
        except OperationCancelledError as exc:
            events.put(("cancelled", str(exc)))
        except Exception as exc:
            logger.exception("批量任务失败")
            events.put(("error", _safe_error(exc)))
        else:
            events.put(("done" if result == 0 else "partial", outputs))
        finally:
            controller.finish("自动处理队列")

    controller.begin("自动处理队列")
    threading.Thread(target=run, name="asmr-dubber-autoflow", daemon=True).start()
    append("开始处理队列。")
    yield "\n".join(lines), False, True, "自动处理已开始。", [], None

    heartbeat = max(0.1, heartbeat_seconds)
    while True:
        try:
            kind, payload = events.get(timeout=heartbeat)
        except queue.Empty:
            elapsed = int(time.monotonic() - started)
            yield (
                "\n".join(lines),
                False,
                True,
                f"仍在处理（已运行 {elapsed} 秒）。可以随时取消，已完成状态会保留。",
                [],
                None,
            )
            continue
        if kind == "log":
            append(str(payload))
            yield "\n".join(lines), False, True, "正在处理队列…", [], None
            continue
        if kind == "reference":
            reference_event = dict(payload) if isinstance(payload, dict) else {}
            event_kind = str(reference_event.get("kind") or "")
            work = str(reference_event.get("work") or "当前作品")
            if event_kind == "ready":
                append(f"{work} 可以选择参考音频；等待期间不操作会使用自动推荐。")
                message = f"“{work}”正在等待参考音频选择。"
            elif event_kind == "selected":
                append(f"{work} 已保存参考音频，任务继续。")
                message = f"“{work}”已保存参考音频，正在继续处理。"
            else:
                append(f"{work} 的等待时间结束，使用自动推荐的参考音频。")
                message = f"“{work}”未手动选择，已使用自动推荐。"
            yield "\n".join(lines), False, True, message, [], reference_event
            continue
        if kind == "done":
            outputs = [str(item) for item in payload]
            append("队列处理完成。")
            yield "\n".join(lines), True, True, "队列处理完成。", outputs, None
            return
        if kind == "partial":
            outputs = [str(item) for item in payload]
            append("队列已结束，其中至少一个作品处理失败；其余作品继续完成。")
            yield (
                "\n".join(lines),
                True,
                False,
                "队列已结束，但有作品失败。已有状态和成品均已保留，请查看日志。",
                outputs,
                None,
            )
            return
        if kind == "cancelled":
            append(str(payload))
            yield "\n".join(lines), True, False, str(payload), [], None
            return
        append(f"自动处理失败：{payload}")
        yield (
            "\n".join(lines),
            True,
            False,
            f"自动处理失败：{payload}\n详细信息已写入日志。",
            [],
            None,
        )
        return


def _install_backend_log_events(
    backend_id: str,
    *,
    installer: Callable[..., str] | None = None,
    heartbeat_seconds: float = 2.0,
    controller: DownloadController | None = None,
) -> Iterator[tuple[str, bool, bool]]:
    """Yield accumulated logs, whether work finished, and whether it succeeded."""

    install = installer or install_backend
    events: queue.Queue[tuple[str, str]] = queue.Queue()
    lines: list[str] = []
    started = time.monotonic()

    def append(message: str) -> None:
        rendered = str(message or "").replace("\r", "\n")
        additions = [line.strip() for line in rendered.splitlines() if line.strip()]
        lines.extend(additions)
        for line in additions:
            logger.info("安装任务 %s：%s", backend_id, line)
        if len(lines) > 240:
            lines[:] = ["…较早的日志已省略…", *lines[-239:]]

    def run() -> None:
        try:
            kwargs: dict[str, Any] = {"log_callback": lambda text: events.put(("log", text))}
            if controller is not None:
                kwargs["cancel_event"] = controller.cancel_event
            result = install(backend_id, **kwargs)
        except InstallPausedError as exc:
            events.put(("paused", str(exc)))
        except Exception as exc:
            events.put(("error", str(exc)))
        else:
            events.put(("done", result))
        finally:
            if controller is not None:
                controller.finish(backend_id)

    append(f"开始检查并安装：{backend_id}")
    if controller is not None:
        controller.begin(backend_id)
    threading.Thread(
        target=run,
        name=f"asmr-dubber-install-{backend_id or 'unknown'}",
        daemon=True,
    ).start()
    yield "\n".join(lines), False, True

    heartbeat = max(0.05, heartbeat_seconds)
    while True:
        try:
            kind, message = events.get(timeout=heartbeat)
        except queue.Empty:
            append(f"仍在处理中（已运行 {int(time.monotonic() - started)} 秒）…")
            yield "\n".join(lines), False, True
            continue
        if kind == "log":
            append(message)
            yield "\n".join(lines), False, True
            continue
        if kind == "done":
            if message.strip() and message.strip() not in "\n".join(lines):
                append(message)
            yield "\n".join(lines), True, True
            return
        if kind == "paused":
            append(message)
            yield "\n".join(lines), True, False
            return
        append(f"安装失败：{message}")
        yield "\n".join(lines), True, False
        return


class _StageProgress:
    def __init__(self, progress: Callable[..., Any] | None) -> None:
        self.progress = progress

    def __call__(self, message: str, current: int, total: int) -> None:
        if self.progress is None:
            return
        ratio = 0.0 if total <= 0 else min(1.0, max(0.0, current / total))
        self.progress((round(ratio * 100), 100), desc=message)


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


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        validation_error = cast(ValidationError, exc)
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}：{item['msg']}"
            for item in validation_error.errors()[:8]
        )
        return f"设置校验失败：{details}"
    return str(exc) or exc.__class__.__name__


def _view_values(view: ProjectView) -> tuple[Any, ...]:
    choices: list[tuple[str, str]] = []
    selected: str | None = None
    preview: str | None = None
    if view.manifest and view.rows:
        with suppress(Exception):
            choices, selected, preview = reference_picker(view.manifest)
    return (
        view.manifest,
        (
            f"**当前项目：** `{view.manifest}`  \n"
            f"**音频/台本语言：** {source_language_label(view.source_language)}"
        ),
        view.rows,
        _gr_update(value=view.output_audio, visible=bool(view.output_audio)),
        _gr_update(value=view.stem_audio, visible=bool(view.stem_audio)),
        _gr_update(value=view.output_video, visible=bool(view.output_video)),
        _gr_update(value=view.subtitle_files or None, visible=bool(view.subtitle_files)),
        _gr_update(value=view.subtitle_video, visible=bool(view.subtitle_video)),
        view.diagnostics,
        view.status,
        _gr_update(choices=choices, value=selected),
        preview,
    )


def _empty_project_updates(message: str) -> tuple[Any, ...]:
    return (*(_gr_update() for _ in range(9)), message, _gr_update(), _gr_update())


def _gr_update(**kwargs: Any) -> Any:
    import gradio as gr

    return gr.update(**kwargs)


def _run_project_action(
    action: Callable[..., ProjectView],
    *args: Any,
    cancel_event: CancellationToken | None = None,
) -> tuple[Any, ...]:
    action_name = getattr(action, "__name__", action.__class__.__name__)
    try:
        logger.info("项目任务开始：%s", action_name)
        with cancellation_scope(cancel_event):
            if cancel_event is None:
                result = action(*args)
            else:
                result = action(*args, cancel_event=cancel_event)
        logger.info("项目任务完成：%s", action_name)
        return _view_values(result)
    except OperationCancelledError as exc:
        logger.info("项目任务已取消：%s", action_name)
        if args:
            with suppress(Exception):
                current = load_view(str(args[0]))
                return _view_values(replace(current, status=str(exc)))
        return _empty_project_updates(str(exc))
    except Exception as exc:
        logger.exception("项目任务失败：%s", action_name)
        return _empty_project_updates(
            f"操作失败：{_safe_error(exc)}\n当前项目、表格和已有输出均已保留。"
            f"\n详细信息已写入日志：{application_log_path()}"
        )


def _settings_from_form(
    field_names: Sequence[str],
    values: Sequence[Any],
    speaker_upload: Any,
    emotion_upload: Any,
) -> UserSettings:
    current = load_user_settings().model_dump()
    form = dict(zip(field_names, values, strict=True))
    selected_language = str(
        form.get("default_source_language", current.get("default_source_language", "ja")) or "ja"
    )
    if selected_language not in {"ja", "en"}:
        raise ValueError(f"未知音频语言：{selected_language}")
    prompt_drafts = form.pop("translation_prompt_drafts", None)
    displayed_prompt = form.pop("translation_prompt", None)
    loudness_mode = form.pop("loudness_mode", None)
    source_ceiling = form.pop("loudness_source_ceiling_dbfs", None)
    uniform_target = form.pop("loudness_uniform_target_dbfs", None)
    raw_gain = form.pop("loudness_raw_gain_db", None)
    current.update(form)
    if loudness_mode is not None:
        mode = str(loudness_mode)
        if mode not in {"source", "uniform", "raw"}:
            raise ValueError(f"未知中文响度处理方式：{mode}")
        current["normalize_chinese_loudness"] = mode != "raw"
        current["match_source_loudness"] = mode == "source"
        if mode == "source" and source_ceiling is not None:
            current["chinese_target_active_rms_dbfs"] = source_ceiling
        elif mode == "uniform" and uniform_target is not None:
            current["chinese_target_active_rms_dbfs"] = uniform_target
        elif mode == "raw" and raw_gain is not None:
            current["chinese_gain_db"] = raw_gain
    if displayed_prompt is not None:
        drafts = dict(prompt_drafts) if isinstance(prompt_drafts, dict) else {}
        drafts[selected_language] = str(displayed_prompt or "")
        for language in ("ja", "en"):
            candidate = (
                str(
                    drafts.get(
                        language,
                        _translation_prompt_for_display(
                            current.get(f"translation_prompt_{language}", ""),
                            cast(SpeechSourceLanguage, language),
                        ),
                    )
                    or ""
                )
                .replace("\r\n", "\n")
                .strip()
            )
            built_in = default_translation_prompt(cast(SpeechSourceLanguage, language)).replace(
                "\r\n", "\n"
            )
            current[f"translation_prompt_{language}"] = "" if candidate == built_in else candidate
        # ProjectSettings keeps one active prompt. UserSettings stores one
        # override per source language and chooses it when a project is made.
        current["translation_prompt"] = ""
    primary_asr = f"{current.get('asr_backend', '')}|{current.get('asr_model', '')}"
    if not current.get("asr_review_text_priority_model"):
        current["asr_review_text_priority_model"] = primary_asr or DEFAULT_ASR_REVIEW_TEXT_PRIORITY
    if not current.get("asr_review_timestamp_priority_model"):
        current["asr_review_timestamp_priority_model"] = (
            primary_asr or DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY
        )
    if speaker_upload:
        current["tts_external_reference_audio"] = str(store_reference_audio(str(speaker_upload)))
    if emotion_upload:
        current["tts_index_external_emotion_audio"] = str(
            store_reference_audio(str(emotion_upload))
        )
    return UserSettings.model_validate(current)


def _provider_update(provider: Any) -> tuple[Any, ...]:
    provider_id = str(provider or "")
    preset = PROVIDER_PRESETS.get(provider_id, PROVIDER_PRESETS["deepseek"])
    return (
        _gr_update(choices=list(preset["models"]), value=preset["default_model"]),
        _gr_update(value=preset["base_url"]),
        str(preset["help"]),
        api_key_status(provider_id),
        _gr_update(visible=provider_id in _LLM_TRANSLATION_PROVIDERS),
        _gr_update(visible=provider_id == "deepl"),
        _gr_update(visible=provider_id == "microsoft_translate"),
    )


def _translation_prompt_for_display(
    value: Any,
    source_language: SpeechSourceLanguage = "ja",
) -> str:
    return str(value or "").strip() or default_translation_prompt(source_language)


def _translation_prompt_drafts(settings: UserSettings) -> dict[str, str]:
    return {
        language: _translation_prompt_for_display(
            settings.translation_prompt_for(cast(SpeechSourceLanguage, language)),
            cast(SpeechSourceLanguage, language),
        )
        for language in ("ja", "en")
    }


def _translation_prompt_note(language: SpeechSourceLanguage, prompt: str) -> str:
    label = source_language_label(language)
    built_in = default_translation_prompt(language).replace("\r\n", "\n").strip()
    current = str(prompt or "").replace("\r\n", "\n").strip()
    if current == built_in:
        return f"当前使用 **{label} → 中文内置 Prompt**。程序更新内置内容后会自动跟随。"
    return f"当前使用 **{label} → 中文自定义 Prompt**。另一种语言的 Prompt 不会被覆盖。"


def _translation_prompt_language_update(
    language: Any,
    current_prompt: Any,
    drafts: Any,
    active_language: Any,
) -> tuple[Any, dict[str, str], str, str]:
    selected = "en" if str(language or "ja") == "en" else "ja"
    previous = "en" if str(active_language or "ja") == "en" else "ja"
    values = dict(drafts) if isinstance(drafts, dict) else {}
    values[previous] = str(current_prompt or "")
    prompt = str(values.get(selected) or default_translation_prompt(selected)).strip()
    values[selected] = prompt
    return (
        _gr_update(
            label=f"翻译 Prompt（{source_language_label(selected)} → 中文）",
            value=prompt,
        ),
        values,
        selected,
        _translation_prompt_note(selected, prompt),
    )


def _reset_translation_prompt(
    language: Any,
    drafts: Any,
) -> tuple[Any, dict[str, str], str]:
    selected = "en" if str(language or "ja") == "en" else "ja"
    prompt = default_translation_prompt(selected)
    values = dict(drafts) if isinstance(drafts, dict) else {}
    values[selected] = prompt
    return (
        _gr_update(
            label=f"翻译 Prompt（{source_language_label(selected)} → 中文）",
            value=prompt,
        ),
        values,
        _translation_prompt_note(selected, prompt),
    )


def _loudness_mode(normalize: bool, match_source: bool) -> str:
    if not normalize:
        return "raw"
    return "source" if match_source else "uniform"


def _loudness_mode_update(mode: Any) -> tuple[Any, ...]:
    selected = str(mode or "source")
    if selected not in _LOUDNESS_MODE_DESCRIPTIONS:
        selected = "source"
    normalized = selected in {"source", "uniform"}
    return (
        _gr_update(visible=selected == "source"),
        _gr_update(visible=selected == "uniform"),
        _gr_update(visible=selected == "raw"),
        _gr_update(visible=selected == "source"),
        _gr_update(visible=normalized),
        _gr_update(visible=normalized),
        _LOUDNESS_MODE_DESCRIPTIONS[selected],
    )


_TRANSCRIPT_TIMING_CHOICES = [
    ("按台词长度估算（无需模型，之后手动校对）", "estimate"),
    ("Qwen3 ForcedAligner 自动对齐（需要进阶组件）", "qwen"),
    ("先运行 ASR/翻译，再用大模型校对台本（推荐）", "script_review"),
]
_CHINESE_TRANSCRIPT_TIMING_CHOICES = [
    _TRANSCRIPT_TIMING_CHOICES[0],
    _TRANSCRIPT_TIMING_CHOICES[2],
]


def _transcript_kind_update(script_kind: Any) -> Any:
    if str(script_kind or "source") == "zh":
        return _gr_update(
            choices=_CHINESE_TRANSCRIPT_TIMING_CHOICES,
            value="estimate",
        )
    return _gr_update(choices=_TRANSCRIPT_TIMING_CHOICES, value="estimate")


def _source_language_backend_update(language: Any, current_backend: Any) -> tuple[Any, str]:
    selected = _speech_language(language)
    backend_id = str(current_backend or "")
    if selected == "en" or backend_id not in ASR_BACKENDS:
        backend_id = "faster_whisper" if selected == "en" else "parakeet_nemo"
    note = (
        "英语项目使用 Faster-Whisper；日语专用的 Parakeet、Kotoba-Whisper 和 ASMR VAD 会自动隐藏。"
        if selected == "en"
        else "日语项目可以使用 Parakeet、Kotoba-Whisper 或 Faster-Whisper。"
    )
    return _gr_update(choices=_asr_backend_choices(selected), value=backend_id), note


def _asr_backend_update(
    backend: Any,
    current_vad_mode: Any = "off",
    source_language: Any = "ja",
) -> tuple[Any, ...]:
    language = _speech_language(source_language)
    backend_id = str(backend or "")
    if language == "en":
        backend_id = "faster_whisper"
    spec = ASR_BACKENDS.get(backend_id, ASR_BACKENDS["parakeet_nemo"])
    models = _asr_models_for_language(backend_id, language)
    default_model = spec.default_model if spec.default_model in models else models[0]
    vad_choices = asr_vad_choices(backend_id, language)
    vad_values = {value for _, value in vad_choices}
    vad_mode = str(current_vad_mode or "off")
    if vad_mode not in vad_values:
        vad_mode = "off"
    return (
        _gr_update(choices=models, value=default_model),
        f"{spec.help}\n\n{spec.setup}",
        _gr_update(visible=backend_id == "faster_whisper"),
        _gr_update(visible=backend_id == "faster_whisper"),
        _gr_update(visible=backend_id in {"kotoba_whisper", "faster_whisper"}),
        _gr_update(visible=backend_id in {"parakeet_nemo", "faster_whisper"}),
        _gr_update(visible=backend_id == "parakeet_nemo"),
        _gr_update(visible=backend_id == "parakeet_nemo"),
        _gr_update(visible=False),
        _gr_update(visible=backend_id == "kotoba_whisper"),
        _gr_update(choices=vad_choices, value=vad_mode),
        _gr_update(visible=vad_mode == "backend"),
        _gr_update(visible=vad_mode == "asmr"),
        _gr_update(visible=vad_mode == "asmr"),
        _gr_update(visible=vad_mode == "asmr"),
        _gr_update(visible=vad_mode == "asmr"),
    )


def _asr_model_update(backend: Any, model: Any) -> tuple[Any, Any]:
    backend_id = str(backend or "")
    model_id = str(model or "")
    return (
        _gr_update(
            visible=(
                backend_id == "parakeet_nemo" and model_id == "nvidia/parakeet-tdt_ctc-0.6b-ja"
            )
        ),
        _gr_update(
            visible=(
                backend_id == "kotoba_whisper"
                or (
                    backend_id == "faster_whisper"
                    and model_id == "kotoba-tech/kotoba-whisper-v2.0-faster"
                )
            )
        ),
    )


def _asr_vad_update(mode: Any) -> tuple[Any, ...]:
    value = str(mode or "off")
    return (
        _gr_update(visible=value == "backend"),
        _gr_update(visible=value == "asmr"),
        _gr_update(visible=value == "asmr"),
        _gr_update(visible=value == "asmr"),
        _gr_update(visible=value == "asmr"),
    )


def _review_visibility_update(enabled: Any) -> tuple[Any, ...]:
    update = _gr_update(visible=bool(enabled))
    return tuple(update.copy() for _ in range(6))


def _review_language_update(
    language: Any,
    enabled: Any,
    selected_models: Any,
    text_priority: Any,
    timestamp_priority: Any,
    backend: Any,
    model: Any,
) -> tuple[Any, Any, Any, Any]:
    source_language = _speech_language(language)
    settings = load_user_settings()
    choices = available_asr_review_choices(settings)
    timestamps = available_timestamp_review_choices(settings)
    if source_language == "en":
        choices = [
            choice
            for choice in choices
            if choice[1].startswith("faster_whisper|") and "kotoba-tech/" not in choice[1]
        ]
        timestamps = [
            choice
            for choice in timestamps
            if choice[1].startswith("qwen_forced_aligner|")
            or (choice[1].startswith("faster_whisper|") and "kotoba-tech/" not in choice[1])
        ]
    values = {value for _, value in choices}
    timestamp_values = {value for _, value in timestamps}
    selected = [str(item) for item in (selected_models or []) if str(item) in values]
    primary = f"{backend}|{model}"
    chosen_text = str(text_priority or "")
    if chosen_text not in values:
        chosen_text = primary if primary in values else (choices[0][1] if choices else "")
    chosen_timestamp = str(timestamp_priority or "")
    if chosen_timestamp not in timestamp_values:
        chosen_timestamp = (
            primary if primary in timestamp_values else (timestamps[0][1] if timestamps else "")
        )
    review_enabled = bool(enabled) and any(item != primary for item in selected)
    return (
        _gr_update(value=review_enabled, interactive=bool(choices)),
        _gr_update(choices=choices, value=selected, visible=review_enabled),
        _gr_update(choices=choices, value=chosen_text or None, visible=review_enabled),
        _gr_update(
            choices=timestamps,
            value=chosen_timestamp or None,
            visible=review_enabled,
        ),
    )


_TTS_DEFAULT_URLS = {
    "gpt_sovits": "http://127.0.0.1:9880",
    "cosyvoice": "http://127.0.0.1:50000",
    "fish_speech": "http://127.0.0.1:8080",
    "mimo_tts": "https://api.xiaomimimo.com/v1",
    "minimax": "https://api.minimaxi.com",
}


def _tts_model_uses_reference(backend: Any, model: Any) -> bool:
    backend_id = str(backend or "")
    if backend_id == "mimo_tts":
        return str(model or "") == "mimo-v2.5-tts-voiceclone"
    spec = TTS_BACKENDS.get(backend_id)
    return bool(spec and spec.reference_audio)


def _tts_model_controls_update(backend: Any, model: Any) -> tuple[Any, ...]:
    backend_id = str(backend or "")
    model_id = str(model or "")
    return (
        _gr_update(visible=_tts_model_uses_reference(backend_id, model_id)),
        _gr_update(
            visible=backend_id in {"edge_tts", "minimax"}
            or (backend_id == "mimo_tts" and model_id == "mimo-v2.5-tts")
        ),
        _gr_update(visible=backend_id == "minimax"),
        _gr_update(visible=backend_id == "mimo_tts"),
    )


def _tts_backend_update(backend: Any) -> tuple[Any, ...]:
    backend_id = str(backend or "")
    spec = TTS_BACKENDS.get(backend_id, TTS_BACKENDS["indextts2"])
    url = _TTS_DEFAULT_URLS.get(backend_id, "")
    return (
        _gr_update(choices=list(spec.models), value=spec.default_model),
        _gr_update(value=url),
        f"{spec.help}\n\n{spec.setup}",
        service_key_status(f"tts:{backend_id}", spec.api_key),
        _gr_update(visible=backend_id == "indextts2"),
        _gr_update(visible=_tts_model_uses_reference(backend_id, spec.default_model)),
        _gr_update(visible=backend_id == "gpt_sovits"),
        _gr_update(visible=backend_id == "cosyvoice"),
        _gr_update(visible=backend_id == "indextts2"),
        _gr_update(visible=backend_id != "indextts2"),
        _gr_update(visible=backend_id == "gpt_sovits"),
        _gr_update(visible=backend_id in {"gpt_sovits", "edge_tts", "minimax"}),
        _gr_update(choices=list(spec.voices), value=spec.default_voice),
        *_tts_model_controls_update(backend_id, spec.default_model)[1:],
    )


def _tts_service_visibility(backend: Any) -> tuple[Any, ...]:
    backend_id = str(backend or "")
    spec = TTS_BACKENDS.get(backend_id, TTS_BACKENDS["indextts2"])
    configurable_url = backend_id not in {"indextts2", "edge_tts"}
    return (
        _gr_update(visible=configurable_url),
        _gr_update(visible=spec.api_key),
        _gr_update(visible=spec.api_key),
        _gr_update(visible=spec.api_key),
    )


def _tts_detail_visibility(
    backend: Any,
    model: Any,
    reference_source: Any,
    index_speaker_source: Any,
    index_emotion_source: Any,
    cosyvoice_mode: Any,
) -> tuple[Any, ...]:
    """Hide reference controls that the active TTS mode cannot consume."""

    backend_id = str(backend or "")
    is_index = backend_id == "indextts2"
    uses_reference = _tts_model_uses_reference(backend_id, model)
    external_speaker = (
        str(index_speaker_source or "") == "external"
        if is_index
        else uses_reference and str(reference_source or "") == "external"
    )
    spec = TTS_BACKENDS.get(backend_id, TTS_BACKENDS["indextts2"])
    external_reference_text = (
        not is_index
        and uses_reference
        and spec.reference_text != "unused"
        and str(reference_source or "") == "external"
        and not (backend_id == "cosyvoice" and str(cosyvoice_mode or "") == "cross_lingual")
    )
    external_reference_language = (
        backend_id == "gpt_sovits" and str(reference_source or "") == "external"
    )
    return (
        _gr_update(visible=external_speaker),
        _gr_update(visible=external_reference_text),
        _gr_update(visible=external_reference_language),
        _gr_update(visible=is_index and str(index_emotion_source or "") == "external"),
        _gr_update(visible=is_index and str(index_emotion_source or "") == "text"),
    )


def _is_loopback(host: str) -> bool:
    value = host.strip().strip("[]").casefold()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _remote_auth(host: str) -> tuple[str, str] | None:
    if _is_loopback(host):
        return None
    username = os.getenv("ASMR_DUBBER_UI_USERNAME", "asmr").strip() or "asmr"
    password = os.getenv("ASMR_DUBBER_UI_PASSWORD", "").strip() or secrets.token_urlsafe(18)
    print(
        f"ASMR Dubber 已绑定非本机地址，登录账号：{username}；本次启动密码：{password}",
        flush=True,
    )
    return username, password


def build_app() -> Any:
    import gradio as gr

    stored = load_user_settings()
    selected_source_language = stored.default_source_language
    asr_stored = settings_for_source_language(
        stored.to_project_settings(source_language=selected_source_language),
        selected_source_language,
    )
    controller = DownloadController()
    task_controller = ProjectTaskController()
    autoflow_controller = ProjectTaskController("批量任务")
    asr_spec = ASR_BACKENDS[asr_stored.asr_backend]
    tts_spec = TTS_BACKENDS[stored.tts_backend]
    provider = PROVIDER_PRESETS.get(stored.translation_provider, PROVIDER_PRESETS["deepseek"])
    initial_loudness_mode = _loudness_mode(
        stored.normalize_chinese_loudness,
        stored.match_source_loudness,
    )
    initial_vad_choices = asr_vad_choices(asr_stored.asr_backend, selected_source_language)
    initial_vad_values = {value for _, value in initial_vad_choices}
    initial_vad_mode = asr_stored.asr_vad_mode
    if initial_vad_mode not in initial_vad_values:
        initial_vad_mode = "off"
    (
        initial_review_choices,
        initial_review_models,
        initial_review_text_priority,
        initial_timestamp_choices,
        initial_review_timestamp_priority,
    ) = _review_control_state(asr_stored, selected_source_language)
    initial_aligner_ready = any(
        value.startswith("qwen_forced_aligner|") for _, value in initial_timestamp_choices
    )
    initial_review_enabled = asr_stored.asr_review_enabled and bool(initial_review_models)
    recent = recent_projects(stored.projects_root or None)
    initial_recent = recent[0][1] if recent else None

    with gr.Blocks(title=f"ASMR Dubber {__version__}") as app:
        gr.HTML(
            "<header><h1>ASMR Dubber</h1>"
            f"<p>音频识别、翻译、中文配音与字幕制作 · {__version__}</p></header>",
            elem_id="asmr-dubber-product-marker",
        )

        with gr.Tabs():
            with gr.Tab("工作台", id="workspace"):
                gr.Markdown(
                    "选择一种工作方式：单个作品适合逐句校对；批量处理适合按作品扫描、选择并排队。",
                    elem_classes=["workspace-intro"],
                )
                with gr.Tabs():
                    with gr.Tab("单个作品", id="project-workspace"):
                        gr.Markdown(
                            "新建或打开项目后，按下面的 1–5 步处理。每一步都会自动保存。",
                            elem_id="workflow-hint",
                        )
                        with gr.Group(elem_id="project-start"):
                            gr.Markdown("### 新建或打开项目")
                            with gr.Row(elem_classes=["mobile-stack"]):
                                source_input = gr.File(
                                    label="原始音频或视频",
                                    file_types=["audio", "video"],
                                    type="filepath",
                                    scale=3,
                                )
                                with gr.Column(scale=1):
                                    create_button = gr.Button("新建项目", variant="primary")
                                    refresh_projects_button = gr.Button("刷新项目列表")
                            with gr.Row(elem_classes=["mobile-stack"]):
                                recent_project = gr.Dropdown(
                                    label="最近项目",
                                    choices=recent,
                                    value=initial_recent,
                                    allow_custom_value=True,
                                    info="也可以粘贴 project.json 的完整路径。",
                                    scale=3,
                                )
                                open_project_button = gr.Button("打开项目", scale=1)
                                open_project_directory_button = gr.Button("打开项目目录", scale=1)
                            project_path = gr.Textbox(
                                label="当前项目文件",
                                interactive=False,
                                visible=False,
                            )
                            project_summary = gr.Markdown(
                                "尚未打开项目。新建项目的音频语言在“设置 → "
                                "ASR（语音识别）”中选择。",
                                elem_id="project-summary",
                            )

                        with gr.Accordion(
                            "使用已有台本或字幕（可选）",
                            open=False,
                            elem_classes=["optional-section"],
                        ):
                            gr.Markdown(
                                "先新建或打开项目。带时间轴的字幕会直接导入；"
                                "TXT 和粘贴文本可以按长度估算，"
                                "也可以先运行识别/翻译，再让大模型按台本校对文字。导入会替换当前句子表，"
                                "原始音频不会改变。"
                            )
                            transcript_kind = gr.Radio(
                                label="导入内容",
                                choices=[
                                    ("原文台本或字幕（沿用当前项目语言，之后翻译）", "source"),
                                    ("中文配音稿或中文字幕（直接配音）", "zh"),
                                ],
                                value="source",
                            )
                            transcript_file = gr.File(
                                label="台本或字幕文件",
                                file_types=[".srt", ".vtt", ".ass", ".ssa", ".lrc", ".txt"],
                                type="filepath",
                            )
                            transcript_text = gr.Textbox(
                                label="粘贴纯文本",
                                placeholder="每个非空行作为一句；选择文件时，粘贴内容优先。",
                                lines=6,
                            )
                            plain_timing = gr.Radio(
                                label="纯文本台本的处理方式",
                                choices=_TRANSCRIPT_TIMING_CHOICES,
                                value="estimate",
                            )
                            gr.Markdown(
                                "智能台本校对会使用当前翻译设置中的大模型；"
                                "DeepL、Google Cloud Translation、Microsoft Translator 等"
                                "机器翻译服务不支持此方式。"
                            )
                            import_transcript_button = gr.Button(
                                "导入并建立句子时间轴",
                                variant="primary",
                            )

                        gr.Markdown("## 制作流程")
                        with gr.Row(
                            equal_height=True,
                            elem_classes=["workflow-actions", "mobile-stack"],
                        ):
                            asr_button = gr.Button("1 · 运行 ASR（语音识别）", variant="primary")
                            translate_button = gr.Button("2 · 翻译为中文")
                            save_table_button = gr.Button("3 · 保存校对表格")
                            synthesize_button = gr.Button(
                                "4 · 生成中文配音",
                                variant="primary",
                            )
                            mix_button = gr.Button("5 · 混音与输出")
                        cancel_task_button = gr.Button("取消当前执行", variant="stop")
                        status = gr.Markdown(
                            "请选择原始文件新建项目，或从最近项目中继续。",
                            elem_id="project-status",
                        )

                        sentence_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=cast(Any, TABLE_TYPES),
                            value=[],
                            interactive=True,
                            wrap=True,
                            column_count=len(TABLE_HEADERS),
                            label="句子校对表格",
                            elem_classes=["sentence-table"],
                        )
                        gr.Markdown(
                            "可以直接修改启用状态、时间、原文和中文。把一行的原文与中文都清空，"
                            "保存后会删除该句。"
                        )

                        with gr.Accordion(
                            "统一音色参考（可选）",
                            open=False,
                            elem_classes=["optional-section"],
                        ):
                            gr.Markdown("建议选择一段 5–15 秒、声音清楚且台词完整的片段。")
                            with gr.Row(elem_classes=["mobile-stack"]):
                                reference_sentence = gr.Dropdown(
                                    label="项目参考句",
                                    choices=[],
                                    allow_custom_value=False,
                                    scale=3,
                                )
                                save_reference_button = gr.Button("设为项目音色参考", scale=1)
                            reference_audio = gr.Audio(
                                label="参考句试听",
                                type="filepath",
                                interactive=False,
                            )

                        with (
                            gr.Accordion(
                                "生成字幕（可选）",
                                open=False,
                                elem_classes=["optional-section"],
                            ),
                            gr.Row(elem_classes=["mobile-stack"]),
                        ):
                            subtitle_language = gr.Radio(
                                label="字幕内容",
                                choices=[
                                    ("原文 + 中文", "bilingual"),
                                    ("仅中文", "zh"),
                                    ("仅原文", "source"),
                                ],
                                value="bilingual",
                            )
                            subtitle_button = gr.Button("生成字幕")

                        gr.Markdown("## 输出文件")
                        with gr.Row(elem_classes=["mobile-stack"]):
                            output_audio = gr.Audio(
                                label="混音成品",
                                type="filepath",
                                interactive=False,
                                visible=False,
                            )
                            stem_audio = gr.Audio(
                                label="中文克隆音轨",
                                type="filepath",
                                interactive=False,
                                visible=False,
                            )
                        with gr.Row(elem_classes=["mobile-stack"]):
                            output_video = gr.Video(
                                label="完成视频",
                                interactive=False,
                                visible=False,
                            )
                        with gr.Row(elem_classes=["mobile-stack"]):
                            subtitle_files = gr.File(
                                label="字幕文件（SRT / LRC）",
                                file_count="multiple",
                                interactive=False,
                                visible=False,
                            )
                            subtitle_video = gr.Video(
                                label="带字幕视频",
                                interactive=False,
                                visible=False,
                            )
                        with gr.Accordion(
                            "项目详情与诊断",
                            open=False,
                            elem_classes=["optional-section"],
                        ):
                            diagnostics = gr.Textbox(
                                label="诊断信息",
                                lines=6,
                                interactive=False,
                                elem_classes=["diagnostics-panel"],
                            )

                    with gr.Tab("批量处理", id="autoflow-workspace"):
                        gr.Markdown(
                            "批量处理适合一次处理一个 DLsite 作品目录中的多条音轨。"
                            "扫描后先确认本作品的选项，再加入队列；程序会按队列完成识别、翻译、"
                            "配音、混音、字幕和成品整理。"
                        )
                        autoflow_queue_state = gr.State([])
                        autoflow_track_state = gr.State([])
                        autoflow_edit_plan_state = gr.State("")
                        with gr.Group(elem_id="autoflow-start"):
                            gr.Markdown("### 1 · 扫描作品")
                            with gr.Row(elem_classes=["mobile-stack"]):
                                autoflow_folder = gr.Textbox(
                                    label="作品文件夹",
                                    placeholder=r"例如 D:\DLsite\RJxxxxxx",
                                    scale=4,
                                )
                                autoflow_scan_button = gr.Button(
                                    "扫描作品", variant="primary", scale=1
                                )
                            autoflow_scan_summary = gr.Markdown("尚未扫描作品。")

                        gr.Markdown("### 2 · 为当前作品选择处理方式")
                        gr.Markdown(
                            "下面的选项只影响刚刚扫描的这个作品。点击“加入处理队列”后，"
                            "选项会随任务一起保存；设置页里的 AutoFlow 默认值"
                            "只负责填充下一次任务。",
                            elem_id="autoflow-options-note",
                        )
                        with gr.Group(elem_id="autoflow-options"):
                            gr.Markdown(
                                "#### 音轨范围",
                                elem_classes=["autoflow-section-title"],
                            )
                            with gr.Row(elem_classes=["mobile-stack"]):
                                autoflow_edition = gr.Dropdown(
                                    label="音频版本",
                                    choices=[],
                                    interactive=True,
                                    info="同一作品可能有无损、压缩或不同语言版本；先选要处理的一组。",
                                    scale=3,
                                )
                                autoflow_include_bonus = gr.Checkbox(
                                    label="包含特典、样本和 Free Talk",
                                    value=stored.autoflow_include_bonus,
                                    info="勾选后把附加音轨也加入本次任务；不勾选只处理主要音轨。",
                                    scale=1,
                                )
                            gr.Markdown(
                                "拖动音轨卡片可以调整顺序；触屏或键盘操作时也可以使用右侧的上下按钮。"
                                "字幕会自动匹配并判断语言，也可以在每条音轨中更换或关闭。"
                            )
                            autoflow_tracks = gr.HTML(
                                value=[],
                                label="本次将处理的音轨",
                                html_template=TRACK_LIST_TEMPLATE,
                                css_template=TRACK_LIST_CSS,
                                js_on_load=TRACK_LIST_JS,
                                elem_id="autoflow-track-list",
                            )
                            autoflow_selection_summary = gr.Markdown("扫描后会在这里列出音轨。")

                            gr.Markdown(
                                "#### 成品类型与组织",
                                elem_classes=["autoflow-section-title"],
                            )
                            with gr.Row(elem_classes=["mobile-stack"]):
                                autoflow_mode = gr.Radio(
                                    label="输出类型",
                                    choices=[
                                        (label, value)
                                        for value, label in AUTOFLOW_MODE_LABELS.items()
                                    ],
                                    value=stored.autoflow_default_mode,
                                    info=(
                                        "纯音频=只生成音频；普通静态视频=原声从 0 秒开始；"
                                        "和谐静态视频=按设置降低并延后原声。"
                                    ),
                                )
                                autoflow_layout = gr.Radio(
                                    label="成品组织",
                                    choices=[
                                        (label, value)
                                        for value, label in AUTOFLOW_LAYOUT_LABELS.items()
                                    ],
                                    value=stored.autoflow_default_layout,
                                    info=(
                                        "合并成一部=适合直接发布；分别处理并输出=每条音轨独立处理，"
                                        "不生成合并版；"
                                        "分轨输出 + 合并版=两者都保留。"
                                    ),
                                )
                            with (
                                gr.Group(
                                    visible=stored.autoflow_default_mode != "audio"
                                ) as autoflow_video_group,
                                gr.Row(elem_classes=["mobile-stack"]),
                            ):
                                autoflow_background = gr.Dropdown(
                                    label="视频画面",
                                    choices=[("黑色背景", "black")],
                                    value="black",
                                    allow_custom_value=False,
                                    info="扫描后会直接选中推荐图片；也可以改选其它图片或黑色背景。",
                                    scale=2,
                                )
                                autoflow_background_preview = gr.Image(
                                    label="画面预览",
                                    type="filepath",
                                    interactive=False,
                                    height=220,
                                    buttons=[],
                                    scale=1,
                                )
                                autoflow_embed_subtitles = gr.Checkbox(
                                    label="在视频中内嵌双语字幕",
                                    value=stored.autoflow_embed_subtitles,
                                    info=(
                                        "勾选后字幕直接显示在视频里；无论是否内嵌，"
                                        "都会另外保留 SRT 和 LRC。"
                                    ),
                                )

                            with gr.Accordion("重做选项（通常不需要）", open=False):
                                autoflow_rebuild = gr.Checkbox(
                                    label="重做并替换本工具生成的旧结果",
                                    value=False,
                                    info=(
                                        "只清理所选作品输出目录里的 AutoFlow 成品和状态；"
                                        "不会修改原作品文件。普通续跑不要勾选。"
                                    ),
                                )
                            with gr.Row(elem_classes=["mobile-stack"]):
                                autoflow_add_button = gr.Button(
                                    "加入处理队列",
                                    variant="primary",
                                )
                                autoflow_cancel_edit_button = gr.Button(
                                    "取消编辑",
                                    visible="hidden",
                                )

                        gr.Markdown("### 3 · 处理队列")
                        gr.Markdown(
                            "拖动任务可以调整执行顺序。每个任务都可以重新编辑、移除，"
                            "也可以标记为重新处理。"
                        )
                        autoflow_queue_list = gr.HTML(
                            value=[],
                            label="处理队列",
                            html_template=QUEUE_LIST_TEMPLATE,
                            css_template=QUEUE_LIST_CSS,
                            js_on_load=QUEUE_LIST_JS,
                            elem_id="autoflow-queue-list",
                        )
                        autoflow_reference_state = gr.State({})
                        with gr.Accordion(
                            "当前作品的参考音频（可选）",
                            open=False,
                            visible=False,
                            elem_id="autoflow-reference-panel",
                            elem_classes=["optional-section"],
                        ) as autoflow_reference_panel:
                            autoflow_reference_info = gr.Markdown(
                                "到达参考音频步骤后，这里会显示当前作品。"
                            )
                            with gr.Row(elem_classes=["mobile-stack"]):
                                autoflow_reference_sentence = gr.Dropdown(
                                    label="项目内参考片段",
                                    choices=[],
                                    interactive=True,
                                    scale=3,
                                )
                                autoflow_reference_apply_button = gr.Button(
                                    "使用这个片段",
                                    variant="primary",
                                    scale=1,
                                )
                            autoflow_reference_audio = gr.Audio(
                                label="参考片段试听",
                                interactive=False,
                            )
                            gr.Markdown("也可以只为当前作品导入一段外部参考音频。")
                            with gr.Row(elem_classes=["mobile-stack"]):
                                autoflow_reference_upload = gr.File(
                                    label="外部参考音频",
                                    file_types=["audio"],
                                    type="filepath",
                                    scale=2,
                                )
                                autoflow_reference_external_text = gr.Textbox(
                                    label="音频对应原文（部分外部 TTS 需要）",
                                    placeholder="IndexTTS2 可以留空",
                                    scale=2,
                                )
                                autoflow_reference_external_language = gr.Dropdown(
                                    label="原文语言",
                                    choices=[
                                        ("跟随项目", "auto"),
                                        ("日语", "ja"),
                                        ("英语", "en"),
                                        ("中文", "zh"),
                                    ],
                                    value="auto",
                                    scale=1,
                                )
                            autoflow_reference_external_button = gr.Button("使用外部音频")
                            autoflow_reference_status = gr.Markdown(
                                "不做选择时，等待结束后会使用自动推荐的项目片段。"
                            )
                        autoflow_clear_button = gr.Button("清空队列")
                        with gr.Row(elem_classes=["mobile-stack"]):
                            autoflow_run_button = gr.Button("开始处理队列", variant="primary")
                            autoflow_cancel_button = gr.Button("取消当前批量任务", variant="stop")
                        autoflow_status = gr.Markdown(
                            "队列为空。完成的作品会写入各自的输出目录。",
                            elem_id="autoflow-status",
                        )
                        with gr.Row(elem_classes=["mobile-stack"]):
                            autoflow_output_selection = gr.Dropdown(
                                label="本次任务输出目录",
                                choices=[],
                                scale=3,
                            )
                            autoflow_open_output_button = gr.Button("打开输出目录", scale=1)
                        with gr.Accordion("运行日志", open=False):
                            autoflow_log = gr.Textbox(
                                label="自动处理日志",
                                value=recent_autoflow_log_text(),
                                lines=18,
                                interactive=False,
                            )

            with gr.Tab("设置", id="settings"):
                gr.Markdown(
                    "这里保存新项目的默认设置。已经打开的项目仍使用自己的设置；需要更新时，"
                    "请点击页面底部的“保存并应用到当前项目”。API 密钥保存在程序目录的"
                    " `.asmr-dubber/config/secrets.json`。"
                )
                settings_status = gr.Textbox(
                    label="设置状态",
                    value="尚未修改。",
                    interactive=False,
                    lines=3,
                )
                settings_components: dict[str, Any] = {}
                prompt_drafts = _translation_prompt_drafts(stored)
                settings_components["translation_prompt_drafts"] = gr.State(prompt_drafts)
                active_prompt_language = gr.State(selected_source_language)

                with gr.Tabs():
                    with gr.Tab("设备与模型", id="models"):
                        hardware = gr.Markdown(hardware_markdown())
                        recommendation = gr.Markdown(recommended_stack_markdown())
                        gr.Markdown(PROFILE_MARKDOWN, elem_classes=["profile-table"])
                        refresh_hardware_button = gr.Button("重新检测硬件与后端")
                        gr.Markdown("### ASR（语音识别）后端")
                        asr_catalog = gr.Dataframe(
                            headers=CATALOG_HEADERS,
                            datatype=cast(Any, ["str"] * len(CATALOG_HEADERS)),
                            value=backend_catalog_rows(stored, kind="asr"),
                            interactive=False,
                            wrap=True,
                            label="ASR（语音识别）兼容性与安装状态",
                            elem_classes=["backend-table"],
                        )
                        asr_available = gr.Markdown(
                            available_backend_models_markdown("asr", stored)
                        )
                        with gr.Row():
                            install_asr_choice = gr.Dropdown(
                                label="要安装/修复的 ASR（语音识别）后端",
                                choices=[
                                    (spec.label, backend_id)
                                    for backend_id, spec in ASR_BACKENDS.items()
                                    if backend_id in _INSTALLABLE
                                ],
                                value="parakeet_nemo",
                                scale=3,
                            )
                            install_asr_button = gr.Button("安装/修复", variant="primary")

                        gr.Markdown("### TTS（语音合成）后端")
                        tts_catalog = gr.Dataframe(
                            headers=CATALOG_HEADERS,
                            datatype=cast(Any, ["str"] * len(CATALOG_HEADERS)),
                            value=backend_catalog_rows(stored, kind="tts"),
                            interactive=False,
                            wrap=True,
                            label="TTS（语音合成）兼容性与安装状态",
                            elem_classes=["backend-table"],
                        )
                        tts_available = gr.Markdown(
                            available_backend_models_markdown("tts", stored)
                        )
                        with gr.Row():
                            install_tts_choice = gr.Dropdown(
                                label="要安装/修复的 TTS（语音合成）后端",
                                choices=[
                                    (spec.label, backend_id)
                                    for backend_id, spec in TTS_BACKENDS.items()
                                    if backend_id in _INSTALLABLE
                                ],
                                value="indextts2",
                                scale=3,
                            )
                            install_tts_button = gr.Button("安装/修复", variant="primary")
                        with gr.Row():
                            pause_download_button = gr.Button("暂停当前下载")
                            scan_packs_button = gr.Button("扫描并导入本地模型包")
                        model_pack_status = gr.Markdown(offline_model_pack_markdown())
                        install_log = gr.Textbox(
                            label="安装与模型包日志",
                            lines=12,
                            interactive=False,
                            autoscroll=True,
                        )

                    with gr.Tab("常规", id="general"):
                        settings_components["projects_root"] = gr.Textbox(
                            label="项目保存目录",
                            value=stored.projects_root,
                            placeholder=str(portable_home() / "projects"),
                            info="留空时使用程序目录内的 .asmr-dubber/projects。",
                        )
                        settings_components["huggingface_endpoint"] = gr.Textbox(
                            label="Hugging Face 下载端点",
                            value=stored.huggingface_endpoint,
                            placeholder="https://hf-mirror.com",
                        )
                        settings_components["pypi_index_url"] = gr.Textbox(
                            label="Python 软件源",
                            value=stored.pypi_index_url,
                            placeholder="https://pypi.org/simple",
                        )

                    with gr.Tab("ASR（语音识别）", id="asr"):
                        gr.Markdown("### 新建项目")
                        settings_components["default_source_language"] = gr.Radio(
                            label="新建媒体项目的音频语言",
                            choices=[("日语", "ja"), ("英语", "en")],
                            value=selected_source_language,
                            info=(
                                "新建项目时使用。当前已经打开的项目会保留自己的语言；"
                                "导入原文台本时直接沿用当前项目语言。"
                            ),
                        )
                        source_language_help = gr.Markdown(
                            "英语项目使用 Faster-Whisper；日语专用的 Parakeet、"
                            "Kotoba-Whisper 和 ASMR VAD 会自动隐藏。"
                            if selected_source_language == "en"
                            else "日语项目可以使用 Parakeet、Kotoba-Whisper 或 Faster-Whisper。"
                        )
                        gr.Markdown("### 识别方式")
                        settings_components["asr_backend"] = gr.Dropdown(
                            label="ASR（语音识别）后端",
                            choices=_asr_backend_choices(selected_source_language),
                            value=asr_stored.asr_backend,
                        )
                        settings_components["asr_model"] = gr.Dropdown(
                            label="ASR（语音识别）模型",
                            choices=_asr_models_for_language(
                                asr_stored.asr_backend,
                                selected_source_language,
                            ),
                            value=asr_stored.asr_model,
                            allow_custom_value=True,
                        )
                        asr_help = gr.Markdown(f"{asr_spec.help}\n\n{asr_spec.setup}")
                        with gr.Row():
                            settings_components["asr_device"] = gr.Dropdown(
                                label="识别设备",
                                choices=[("NVIDIA CUDA", "cuda"), ("CPU", "cpu")],
                                value=asr_stored.asr_device,
                            )
                            settings_components["asr_compute_type"] = gr.Dropdown(
                                label="计算精度",
                                choices=["float16", "float32", "int8_float16", "int8"],
                                value=asr_stored.asr_compute_type,
                                allow_custom_value=True,
                                visible=asr_stored.asr_backend == "faster_whisper",
                            )
                            settings_components["asr_batch_size"] = gr.Number(
                                label="批大小", value=asr_stored.asr_batch_size, precision=0
                            )
                            settings_components["asr_beam_size"] = gr.Number(
                                label="束搜索宽度（Beam Size）",
                                value=asr_stored.asr_beam_size,
                                precision=0,
                                visible=asr_stored.asr_backend == "faster_whisper",
                            )
                        with gr.Row():
                            settings_components["pause_split_seconds"] = gr.Number(
                                label="停顿切句秒数", value=stored.pause_split_seconds
                            )
                            settings_components["max_sentence_seconds"] = gr.Number(
                                label="单句最长秒数", value=stored.max_sentence_seconds
                            )
                            settings_components["asr_timeout_seconds"] = gr.Number(
                                label="Parakeet 连续无响应超时（秒）",
                                value=asr_stored.asr_timeout_seconds,
                                visible=asr_stored.asr_backend == "parakeet_nemo",
                                info="持续收到识别进度时不会因总耗时超过该值而停止。",
                            )
                        with gr.Accordion("VAD 与当前后端参数", open=False):
                            settings_components["asr_vad_mode"] = gr.Radio(
                                label="VAD（语音活动检测）预处理",
                                choices=initial_vad_choices,
                                value=initial_vad_mode,
                                info=(
                                    "只显示当前识别后端可用且本地已经安装的方式。"
                                    "ASMR 专用 VAD 不会修改原音频。"
                                ),
                            )
                            settings_components["asr_vad_min_silence_ms"] = gr.Number(
                                label="VAD 最短静音毫秒",
                                value=asr_stored.asr_vad_min_silence_ms,
                                precision=0,
                                visible=initial_vad_mode == "backend",
                            )
                            settings_components["asr_asmr_vad_threshold"] = gr.Number(
                                label="ASMR VAD 语音阈值",
                                value=asr_stored.asr_asmr_vad_threshold,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_asmr_vad_min_speech_ms"] = gr.Number(
                                label="ASMR VAD 最短语音毫秒",
                                value=asr_stored.asr_asmr_vad_min_speech_ms,
                                precision=0,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_asmr_vad_min_silence_ms"] = gr.Number(
                                label="ASMR VAD 最短静音毫秒",
                                value=asr_stored.asr_asmr_vad_min_silence_ms,
                                precision=0,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_asmr_vad_speech_pad_ms"] = gr.Number(
                                label="ASMR VAD 边界保留毫秒",
                                value=asr_stored.asr_asmr_vad_speech_pad_ms,
                                precision=0,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_condition_on_previous_text"] = gr.Checkbox(
                                label="使用上一段文字作为识别条件",
                                value=asr_stored.asr_condition_on_previous_text,
                                visible=asr_stored.asr_backend
                                in {"kotoba_whisper", "faster_whisper"},
                            )
                            settings_components["asr_initial_prompt"] = gr.Textbox(
                                label="识别提示词（人名、作品名或特殊读法）",
                                value=asr_stored.asr_initial_prompt,
                                lines=3,
                                visible=asr_stored.asr_backend
                                in {"parakeet_nemo", "faster_whisper"},
                            )
                            settings_components["asr_parakeet_decoder"] = gr.Radio(
                                label="Parakeet 解码头",
                                choices=[("TDT", "tdt"), ("CTC", "ctc")],
                                value=asr_stored.asr_parakeet_decoder,
                                visible=(
                                    asr_stored.asr_backend == "parakeet_nemo"
                                    and asr_stored.asr_model == "nvidia/parakeet-tdt_ctc-0.6b-ja"
                                ),
                            )
                            settings_components["asr_chunk_seconds"] = gr.Number(
                                label="Parakeet 分块秒数（15–600）",
                                value=asr_stored.asr_chunk_seconds,
                                visible=asr_stored.asr_backend == "parakeet_nemo",
                            )
                            settings_components["asr_kotoba_chunk_seconds"] = gr.Number(
                                label="Kotoba-Whisper 分块秒数（5–120）",
                                value=asr_stored.asr_kotoba_chunk_seconds,
                                visible=(
                                    asr_stored.asr_backend == "kotoba_whisper"
                                    or (
                                        asr_stored.asr_backend == "faster_whisper"
                                        and asr_stored.asr_model
                                        == "kotoba-tech/kotoba-whisper-v2.0-faster"
                                    )
                                ),
                            )
                        settings_components["asr_forced_alignment_enabled"] = gr.Checkbox(
                            label=("识别后使用 Qwen3 ForcedAligner 0.6B（阿里）重新计算时间戳"),
                            value=asr_stored.asr_forced_alignment_enabled,
                            visible=initial_aligner_ready,
                            info="识别文字不变；该模型只重新寻找每句话的起止时间。",
                        )
                        with gr.Accordion("多模型交叉校对（实验性）", open=False):
                            gr.Markdown("**实验性，不建议使用。**")
                            settings_components["asr_review_enabled"] = gr.Checkbox(
                                label="启用多 ASR（语音识别）+ 大模型交叉校对",
                                value=initial_review_enabled,
                                interactive=bool(initial_review_choices),
                                info="下面只列出本地已完整下载并可运行的识别模型。",
                            )
                            settings_components["asr_review_models"] = gr.CheckboxGroup(
                                label="参与校对的识别模型",
                                choices=initial_review_choices,
                                value=initial_review_models,
                                visible=initial_review_enabled,
                            )
                            settings_components["asr_review_text_priority_model"] = gr.Dropdown(
                                label="文字判断优先来源",
                                choices=initial_review_choices,
                                value=initial_review_text_priority,
                                visible=initial_review_enabled,
                            )
                            settings_components["asr_review_timestamp_priority_model"] = (
                                gr.Dropdown(
                                    label="最终时间戳来源",
                                    choices=initial_timestamp_choices,
                                    value=initial_review_timestamp_priority,
                                    visible=initial_review_enabled,
                                    info=(
                                        "可使用某个已下载 ASR 的自带时间戳，或用"
                                        " Qwen3 ForcedAligner 对齐校对后的最终文字。"
                                    ),
                                )
                            )
                            settings_components["asr_review_max_drift_seconds"] = gr.Number(
                                label="允许时间漂移秒数",
                                value=asr_stored.asr_review_max_drift_seconds,
                                visible=initial_review_enabled,
                            )
                            settings_components["asr_review_background"] = gr.Textbox(
                                label="作品、人物与场景背景",
                                value=asr_stored.asr_review_background,
                                lines=4,
                                visible=initial_review_enabled,
                            )
                            settings_components["asr_review_prompt"] = gr.Textbox(
                                label="ASR（语音识别）校对提示词（Prompt）",
                                value=asr_stored.asr_review_prompt,
                                lines=10,
                                visible=initial_review_enabled,
                            )

                    with gr.Tab("翻译", id="translation"):
                        settings_components["translation_provider"] = gr.Dropdown(
                            label="翻译服务",
                            choices=[
                                (str(item["label"]), key) for key, item in PROVIDER_PRESETS.items()
                            ],
                            value=stored.translation_provider,
                        )
                        settings_components["translation_model"] = gr.Dropdown(
                            label="翻译模型",
                            choices=list(provider["models"]),
                            value=stored.translation_model,
                            allow_custom_value=True,
                        )
                        settings_components["translation_base_url"] = gr.Textbox(
                            label="翻译 API（接口）基础地址",
                            value=stored.translation_base_url or str(provider["base_url"]),
                        )
                        translation_help = gr.Markdown(str(provider["help"]))
                        with gr.Group(
                            visible=stored.translation_provider in _LLM_TRANSLATION_PROVIDERS
                        ) as llm_translation_group:
                            with gr.Row():
                                settings_components["translation_temperature"] = gr.Number(
                                    label="随机度（Temperature）",
                                    value=stored.translation_temperature,
                                )
                                settings_components["translation_top_p"] = gr.Number(
                                    label="核采样概率（Top P）",
                                    value=stored.translation_top_p,
                                )
                                settings_components["translation_max_output_tokens"] = gr.Number(
                                    label="最大输出词元数（Token）",
                                    value=stored.translation_max_output_tokens,
                                    precision=0,
                                )
                            with gr.Row():
                                settings_components["translation_send_context"] = gr.Checkbox(
                                    label="发送相邻句上下文",
                                    value=stored.translation_send_context,
                                )
                                settings_components["translation_context_sentences"] = gr.Number(
                                    label="上下文句数",
                                    value=stored.translation_context_sentences,
                                    precision=0,
                                    visible=stored.translation_send_context,
                                )
                                settings_components["translation_memory_sentences"] = gr.Number(
                                    label="翻译记忆句数",
                                    value=stored.translation_memory_sentences,
                                    precision=0,
                                )
                            settings_components["translation_prompt"] = gr.Textbox(
                                label=(
                                    "翻译 Prompt"
                                    f"（{source_language_label(selected_source_language)} → 中文）"
                                ),
                                value=prompt_drafts[selected_source_language],
                                lines=12,
                                info=(
                                    "日语和英语分别保存。直接编辑会改为自定义；"
                                    "点击下方按钮可恢复当前语言的内置版本。"
                                ),
                            )
                            translation_prompt_note = gr.Markdown(
                                _translation_prompt_note(
                                    selected_source_language,
                                    prompt_drafts[selected_source_language],
                                )
                            )
                            reset_translation_prompt_button = gr.Button("恢复当前语言的内置 Prompt")
                        with gr.Group(
                            visible=stored.translation_provider == "deepl"
                        ) as deepl_translation_group:
                            settings_components["translation_deepl_formality"] = gr.Dropdown(
                                label="DeepL 正式程度",
                                choices=[
                                    "default",
                                    "more",
                                    "less",
                                    "prefer_more",
                                    "prefer_less",
                                ],
                                value=stored.translation_deepl_formality,
                                allow_custom_value=True,
                            )
                        with gr.Group(
                            visible=stored.translation_provider == "microsoft_translate"
                        ) as microsoft_translation_group:
                            settings_components["translation_microsoft_region"] = gr.Textbox(
                                label="Azure Translator 区域",
                                value=stored.translation_microsoft_region,
                            )
                        with gr.Accordion("翻译 API（接口）密钥", open=True):
                            translation_key = gr.Textbox(
                                label="API 密钥",
                                type="password",
                                placeholder="输入后点击保存；界面不会回显已保存密钥",
                            )
                            translation_key_status = gr.Textbox(
                                label="密钥状态",
                                value=api_key_status(stored.translation_provider),
                                interactive=False,
                            )
                            with gr.Row():
                                save_translation_key_button = gr.Button("保存当前服务密钥")
                                clear_translation_key_button = gr.Button("清除当前服务密钥")

                    with gr.Tab("TTS（语音合成）", id="tts"):
                        settings_components["tts_backend"] = gr.Dropdown(
                            label="TTS（语音合成）后端",
                            choices=[(spec.label, key) for key, spec in TTS_BACKENDS.items()],
                            value=stored.tts_backend,
                        )
                        settings_components["tts_model"] = gr.Dropdown(
                            label="TTS（语音合成）模型",
                            choices=list(tts_spec.models),
                            value=stored.tts_model,
                            allow_custom_value=True,
                        )
                        tts_help = gr.Markdown(f"{tts_spec.help}\n\n{tts_spec.setup}")
                        with gr.Group(
                            visible=(
                                stored.tts_backend in {"edge_tts", "minimax"}
                                or (
                                    stored.tts_backend == "mimo_tts"
                                    and stored.tts_model == "mimo-v2.5-tts"
                                )
                            )
                        ) as tts_voice_group:
                            with gr.Row(elem_classes=["mobile-stack"]):
                                settings_components["tts_voice"] = gr.Dropdown(
                                    label="音色 ID",
                                    choices=list(tts_spec.voices),
                                    value=stored.tts_voice or tts_spec.default_voice,
                                    allow_custom_value=True,
                                    info="可以直接填写服务商提供或账号中已创建的音色 ID。",
                                    scale=3,
                                )
                                edge_tts_preview_button = gr.Button(
                                    "试听音色",
                                    visible=stored.tts_backend == "edge_tts",
                                    scale=1,
                                )
                            edge_tts_preview_audio = gr.Audio(
                                label="Edge TTS 音色试听",
                                type="filepath",
                                interactive=False,
                                visible=False,
                            )
                            edge_tts_preview_status = gr.Markdown()
                        with gr.Group(visible=stored.tts_backend == "minimax") as minimax_group:
                            gr.Markdown("### MiniMax 参数")
                            with gr.Row():
                                settings_components["tts_volume"] = gr.Number(
                                    label="音量", value=stored.tts_volume
                                )
                                settings_components["tts_pitch"] = gr.Slider(
                                    label="音调",
                                    minimum=-12,
                                    maximum=12,
                                    step=1,
                                    value=stored.tts_pitch,
                                )
                                settings_components["tts_emotion"] = gr.Dropdown(
                                    label="情绪",
                                    choices=[
                                        ("自动", "auto"),
                                        ("平静", "calm"),
                                        ("开心", "happy"),
                                        ("悲伤", "sad"),
                                        ("愤怒", "angry"),
                                        ("恐惧", "fearful"),
                                        ("厌恶", "disgusted"),
                                        ("惊讶", "surprised"),
                                        ("耳语", "whipser"),
                                    ],
                                    value=stored.tts_emotion,
                                    allow_custom_value=True,
                                )
                        with gr.Group(visible=stored.tts_backend == "mimo_tts") as mimo_group:
                            settings_components["tts_style_prompt"] = gr.Textbox(
                                label="MiMo 语气与风格说明（可选）",
                                value=stored.tts_style_prompt,
                                lines=3,
                                info=(
                                    "用于控制语气、情绪和表达风格；文字设计音色模型留空时"
                                    "使用温柔自然的中文女声。"
                                ),
                            )
                        with gr.Row():
                            settings_components["tts_device"] = gr.Dropdown(
                                label="合成设备",
                                choices=[("NVIDIA CUDA", "cuda"), ("CPU/外部服务", "cpu")],
                                value=stored.tts_device,
                                visible=stored.tts_backend == "indextts2",
                            )
                            settings_components["tts_timeout_seconds"] = gr.Number(
                                label="单句超时秒数", value=stored.tts_timeout_seconds
                            )
                            settings_components["tts_request_concurrency"] = gr.Slider(
                                label="外部 API（接口）并发数",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=stored.tts_request_concurrency,
                                visible=stored.tts_backend != "indextts2",
                            )
                        settings_components["tts_speed"] = gr.Number(
                            label="语速",
                            value=stored.tts_speed,
                            visible=stored.tts_backend in {"gpt_sovits", "edge_tts", "minimax"},
                        )
                        with gr.Row(
                            visible=stored.tts_backend == "gpt_sovits"
                        ) as tts_sampling_group:
                            settings_components["tts_temperature"] = gr.Number(
                                label="随机度（Temperature）",
                                value=stored.tts_temperature,
                            )
                            settings_components["tts_top_p"] = gr.Number(
                                label="核采样概率（Top P）", value=stored.tts_top_p
                            )

                        saved_speaker = stored.tts_external_reference_audio or "无"
                        external_speaker_visible = _tts_model_uses_reference(
                            stored.tts_backend, stored.tts_model
                        ) and (
                            stored.tts_index_speaker_source == "external"
                            if stored.tts_backend == "indextts2"
                            else stored.tts_reference_source == "external"
                        )
                        with gr.Group(visible=external_speaker_visible) as external_speaker_group:
                            external_speaker_upload = gr.File(
                                label="新的外部音色参考音频（可选）",
                                file_types=["audio"],
                                type="filepath",
                            )
                            gr.Markdown(f"当前已保存音色参考：`{saved_speaker}`")
                        with gr.Group(
                            visible=(
                                stored.tts_backend != "indextts2"
                                and _tts_model_uses_reference(stored.tts_backend, stored.tts_model)
                            )
                        ) as generic_tts_group:
                            settings_components["tts_reference_source"] = gr.Radio(
                                label="参考音频来源",
                                choices=[
                                    ("项目内参考句", "project_sentence"),
                                    ("外部音频", "external"),
                                ],
                                value=stored.tts_reference_source,
                            )
                            settings_components["tts_clone_mode"] = gr.Radio(
                                label="参考策略",
                                choices=[(label, key) for key, label in CLONE_MODE_LABELS.items()],
                                value=stored.tts_clone_mode,
                            )
                            settings_components["tts_external_reference_text"] = gr.Textbox(
                                label="外部参考音频对应原文",
                                value=stored.tts_external_reference_text,
                                lines=3,
                                visible=(
                                    tts_spec.reference_text != "unused"
                                    and stored.tts_reference_source == "external"
                                    and not (
                                        stored.tts_backend == "cosyvoice"
                                        and stored.tts_cosyvoice_mode == "cross_lingual"
                                    )
                                ),
                            )
                            settings_components["tts_external_reference_language"] = gr.Dropdown(
                                label="外部参考音频语言",
                                choices=[
                                    ("跟随当前项目语言", "auto"),
                                    ("日语", "ja"),
                                    ("英语", "en"),
                                    ("中文", "zh"),
                                ],
                                value=stored.tts_external_reference_language,
                                visible=(
                                    stored.tts_backend == "gpt_sovits"
                                    and stored.tts_reference_source == "external"
                                ),
                            )

                        settings_components["tts_api_base_url"] = gr.Textbox(
                            label="TTS（语音合成）API（接口）基础地址",
                            value=stored.tts_api_base_url,
                            visible=stored.tts_backend not in {"indextts2", "edge_tts"},
                        )
                        tts_key = gr.Textbox(
                            label="TTS（语音合成）服务 API 密钥",
                            type="password",
                            visible=tts_spec.api_key,
                        )
                        tts_key_status = gr.Textbox(
                            label="TTS（语音合成）密钥状态",
                            value=service_key_status(f"tts:{stored.tts_backend}", tts_spec.api_key),
                            interactive=False,
                            visible=tts_spec.api_key,
                        )
                        with gr.Row(visible=tts_spec.api_key) as tts_key_buttons:
                            save_tts_key_button = gr.Button("保存当前 TTS 服务密钥")
                            clear_tts_key_button = gr.Button("清除当前 TTS 服务密钥")

                        with gr.Group(visible=stored.tts_backend == "indextts2") as index_group:
                            gr.Markdown("### IndexTTS2 参数")
                            settings_components["tts_model_path"] = gr.Textbox(
                                label="IndexTTS2 模型权重目录（checkpoints）",
                                value=stored.tts_model_path,
                            )
                            settings_components["tts_config_path"] = gr.Textbox(
                                label="IndexTTS2 配置文件（config.yaml）",
                                value=stored.tts_config_path,
                            )
                            index_status = gr.Textbox(
                                label="IndexTTS2 状态",
                                value=indextts_installation_status(stored.tts_model_path),
                                interactive=False,
                            )
                            settings_components["tts_index_use_fp16"] = gr.Checkbox(
                                label="使用半精度计算（FP16）",
                                value=stored.tts_index_use_fp16,
                            )
                            settings_components["tts_index_emo_alpha"] = gr.Slider(
                                label="情绪权重",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=stored.tts_index_emo_alpha,
                            )
                            settings_components["tts_index_speaker_source"] = gr.Dropdown(
                                label="音色参考来源",
                                choices=[
                                    ("项目统一参考句", "project_reference"),
                                    ("当前句", "sentence_reference"),
                                    ("外部音频", "external"),
                                ],
                                value=stored.tts_index_speaker_source,
                            )
                            settings_components["tts_index_emotion_source"] = gr.Dropdown(
                                label="情绪参考来源",
                                choices=[
                                    ("当前句", "sentence_reference"),
                                    ("项目统一参考句", "project_reference"),
                                    ("跟随音色参考", "speaker_reference"),
                                    ("外部音频", "external"),
                                    ("文字描述", "text"),
                                ],
                                value=stored.tts_index_emotion_source,
                            )
                            saved_emotion = stored.tts_index_external_emotion_audio or "无"
                            with gr.Group(
                                visible=stored.tts_index_emotion_source == "external"
                            ) as external_emotion_group:
                                external_emotion_upload = gr.File(
                                    label="新的外部情绪参考音频（可选）",
                                    file_types=["audio"],
                                    type="filepath",
                                )
                                gr.Markdown(f"当前已保存情绪参考：`{saved_emotion}`")
                            settings_components["tts_index_emo_text"] = gr.Textbox(
                                label="情绪文字描述",
                                value=stored.tts_index_emo_text,
                                lines=3,
                                visible=stored.tts_index_emotion_source == "text",
                            )
                        with gr.Group(visible=stored.tts_backend == "gpt_sovits") as gpt_group:
                            gr.Markdown("### GPT-SoVITS API 参数")
                            settings_components["tts_gpt_top_k"] = gr.Number(
                                label="候选数（Top K）",
                                value=stored.tts_gpt_top_k,
                                precision=0,
                            )
                            settings_components["tts_gpt_text_split_method"] = gr.Textbox(
                                label="文本切分方法", value=stored.tts_gpt_text_split_method
                            )
                            settings_components["tts_gpt_sample_steps"] = gr.Number(
                                label="采样步数", value=stored.tts_gpt_sample_steps, precision=0
                            )
                        with gr.Group(visible=stored.tts_backend == "cosyvoice") as cosy_group:
                            settings_components["tts_cosyvoice_mode"] = gr.Radio(
                                label="CosyVoice 模式",
                                choices=[("零样本", "zero_shot"), ("跨语言", "cross_lingual")],
                                value=stored.tts_cosyvoice_mode,
                            )

                    with gr.Tab("混音与字幕", id="mix-subtitles"):
                        with gr.Row():
                            settings_components["chinese_dubbing_offset_ms"] = gr.Number(
                                label="中文配音整体偏移（毫秒）",
                                value=stored.chinese_dubbing_offset_ms,
                                precision=0,
                                info="0 表示从原字幕开始；负数提前，正数延后。",
                            )
                            settings_components["chinese_dubbing_timing_mode"] = gr.Radio(
                                label="中文配音排程方式",
                                choices=[
                                    ("保持原时间点，冲突时自动加速", "fit_window"),
                                    ("上一句结束后再播放下一句", "sequential"),
                                ],
                                value=stored.chinese_dubbing_timing_mode,
                                info=(
                                    "顺延模式不自动加速，也不会让两句中文互相叠加；"
                                    "连续长句可能逐渐晚于原字幕。"
                                ),
                            )
                        with gr.Group(
                            visible=stored.chinese_dubbing_timing_mode == "fit_window"
                        ) as auto_speed_group:
                            settings_components["chinese_max_auto_speed"] = gr.Slider(
                                label="冲突时最大自动加速倍速",
                                minimum=1.0,
                                maximum=MAX_CHINESE_AUTO_SPEED,
                                step=0.05,
                                value=stored.chinese_max_auto_speed,
                                info=(
                                    "配音超过下一句开始时间时自动加速；最高可选 4×，"
                                    "达到上限后仍冲突则允许重叠。"
                                ),
                            )
                        gr.Markdown("### 中文配音音量")
                        settings_components["loudness_mode"] = gr.Radio(
                            label="音量处理方式",
                            choices=_LOUDNESS_MODE_CHOICES,
                            value=initial_loudness_mode,
                            info="只选择一种处理方式，避免多个开关互相依赖。",
                        )
                        loudness_mode_help = gr.Markdown(
                            _LOUDNESS_MODE_DESCRIPTIONS[initial_loudness_mode]
                        )
                        with gr.Group(
                            visible=initial_loudness_mode == "source"
                        ) as source_loudness_group:
                            settings_components["chinese_relative_loudness_db"] = gr.Slider(
                                label="中文相对原声音量（dB）",
                                minimum=-24.0,
                                maximum=12.0,
                                step=0.5,
                                value=stored.chinese_relative_loudness_db,
                                info=(
                                    "负数让中文更轻，0 接近原片段音量，正数让中文更突出；"
                                    "默认 -8 dB。"
                                ),
                            )
                        with gr.Group(
                            visible=initial_loudness_mode == "uniform"
                        ) as uniform_loudness_group:
                            settings_components["loudness_uniform_target_dbfs"] = gr.Slider(
                                label="统一中文目标响度（RMS dBFS）",
                                minimum=-50.0,
                                maximum=-16.0,
                                step=1.0,
                                value=stored.chinese_target_active_rms_dbfs,
                                info="数值越接近 0 越响；默认 -30 dBFS。",
                            )
                        with gr.Group(visible=initial_loudness_mode == "raw") as raw_loudness_group:
                            settings_components["loudness_raw_gain_db"] = gr.Slider(
                                label="TTS 原始音量微调（dB）",
                                minimum=-20.0,
                                maximum=12.0,
                                step=0.5,
                                value=stored.chinese_gain_db,
                                info="0 不改变原始音量；负数降低，正数提高。",
                            )
                        with gr.Accordion("高级响度参数", open=False):
                            gr.Markdown(
                                "默认值适合大多数项目。dBFS 数值越接近 0 越响；"
                                "峰值参数越低，保留的安全余量越多。"
                            )
                            with (
                                gr.Group(
                                    visible=initial_loudness_mode == "source"
                                ) as source_loudness_advanced_group,
                                gr.Row(),
                            ):
                                settings_components["chinese_min_active_rms_dbfs"] = gr.Number(
                                    label="自动匹配的最安静目标（RMS dBFS）",
                                    value=stored.chinese_min_active_rms_dbfs,
                                    info=(
                                        "原片段非常安静时，中文目标不会低于这个值，"
                                        "避免配音几乎听不见；默认 -42。"
                                    ),
                                )
                                settings_components["loudness_source_ceiling_dbfs"] = gr.Number(
                                    label="自动匹配的最响目标（RMS dBFS）",
                                    value=stored.chinese_target_active_rms_dbfs,
                                    info=(
                                        "原片段很响时，中文目标不会高于这个值，"
                                        "避免跟随结果过响；默认 -30。"
                                    ),
                                )
                            with (
                                gr.Group(
                                    visible=initial_loudness_mode in {"source", "uniform"}
                                ) as normalized_loudness_advanced_group,
                                gr.Row(),
                            ):
                                settings_components["chinese_max_loudness_boost_db"] = gr.Number(
                                    label="每句最大自动提升（dB）",
                                    value=stored.chinese_max_loudness_boost_db,
                                    info=(
                                        "一句中文太轻时最多自动提高多少；"
                                        "限制提升可避免底噪被过度放大，默认 12。"
                                    ),
                                )
                                settings_components["chinese_line_peak_dbfs"] = gr.Number(
                                    label="单句峰值上限（dBFS）",
                                    value=stored.chinese_line_peak_dbfs,
                                    info=("规范化后限制每句的瞬时峰值，防止单句削波；默认 -9。"),
                                )
                                settings_components["chinese_fade_ms"] = gr.Number(
                                    label="句首句尾淡入淡出（毫秒）",
                                    value=stored.chinese_fade_ms,
                                    info=(
                                        "在每句边缘加入极短渐变以减少爆音和咔哒声；"
                                        "过大可能吃掉短音节，默认 8 ms。"
                                    ),
                                )
                            with gr.Group(
                                visible=initial_loudness_mode in {"source", "uniform"}
                            ) as adjusted_loudness_gain_group:
                                settings_components["chinese_gain_db"] = gr.Number(
                                    label="自动处理后的整体微调（dB）",
                                    value=stored.chinese_gain_db,
                                    info=(
                                        "在自动匹配或统一响度之后，再整体提高或降低中文；"
                                        "通常保持 0，优先调整上面的主要音量参数。"
                                    ),
                                )
                            settings_components["chinese_stem_peak_dbfs"] = gr.Number(
                                label="中文轨叠加峰值上限（dBFS）",
                                value=stored.chinese_stem_peak_dbfs,
                                info=("多句中文重叠时限制合计峰值，避免中文中间轨削波；默认 -3。"),
                            )
                            reset_loudness_button = gr.Button(
                                "恢复响度参数默认值（不切换处理方式）"
                            )
                        settings_components["chinese_channel_routing"] = gr.Radio(
                            label="多声道路由",
                            choices=[("自动（优先中置）", "auto"), ("复制到全部声道", "all")],
                            value=stored.chinese_channel_routing,
                        )
                        settings_components["mix_output_mode"] = gr.Radio(
                            label="音频输出方式",
                            choices=[
                                ("混音成品和中文克隆音轨", "both"),
                                ("仅混音成品", "mixed"),
                                ("仅中文克隆音轨", "stem"),
                            ],
                            value=stored.mix_output_mode,
                            info=(
                                "“仅中文克隆音轨”会导出一条与原文件等长的 WAV，方便单独编辑。"
                                "之后改回包含混音成品的选项，可以直接重新混音，不会重新生成配音。"
                            ),
                        )
                        with gr.Group(visible=stored.mix_output_mode != "stem") as final_mix_group:
                            settings_components["mix_peak_protection"] = gr.Checkbox(
                                label="最终混音峰值保护", value=stored.mix_peak_protection
                            )
                            settings_components["mix_peak_limit_dbfs"] = gr.Number(
                                label="最终峰值上限（dBFS）", value=stored.mix_peak_limit_dbfs
                            )
                        settings_components["skip_japanese_fillers"] = gr.Checkbox(
                            label="日语项目跳过纯语气词",
                            value=stored.skip_japanese_fillers,
                        )
                        settings_components["reference_padding_seconds"] = gr.Number(
                            label="参考音频边缘扩展秒数",
                            value=stored.reference_padding_seconds,
                        )
                        settings_components["random_seed"] = gr.Number(
                            label="随机种子", value=stored.random_seed, precision=0
                        )
                        gr.Markdown("### 字幕")
                        settings_components["subtitle_timeline"] = gr.Radio(
                            label="字幕时间轴",
                            choices=[("原字幕时间", "source"), ("中文配音时间", "dubbing")],
                            value=stored.subtitle_timeline,
                        )
                        with gr.Row():
                            settings_components["subtitle_max_chars_per_line"] = gr.Number(
                                label="每行最多字符",
                                value=stored.subtitle_max_chars_per_line,
                                precision=0,
                            )
                            settings_components["subtitle_min_duration_seconds"] = gr.Number(
                                label="最短显示秒数",
                                value=stored.subtitle_min_duration_seconds,
                            )
                            settings_components["subtitle_max_cps"] = gr.Number(
                                label="最大每秒字符数", value=stored.subtitle_max_cps
                            )

                    with gr.Tab("自动处理", id="autoflow-settings"):
                        gr.Markdown(
                            "这里分成两类设置：固定规则会对所有作品生效；新作品默认值只负责"
                            "填充批量处理页，加入队列前仍可以按作品单独修改。识别、翻译、配音和混音"
                            "使用前面各页保存的主程序设置。",
                            elem_id="autoflow-settings-note",
                        )

                        gr.Markdown("### 固定规则（所有作品共用）")
                        settings_components["autoflow_output_folder_name"] = gr.Textbox(
                            label="成品输出文件夹名称",
                            value=stored.autoflow_output_folder_name,
                            info=(
                                "每个源作品目录下都会建立这个子文件夹，"
                                "用来保存 AutoFlow 成品和状态。"
                            ),
                        )
                        settings_components["autoflow_preferred_audio_formats"] = gr.Textbox(
                            label="同一作品多种音频格式时的选择顺序",
                            value=stored.autoflow_preferred_audio_formats,
                            info=(
                                "从左到右填写优先级，例如 wav,flac,ape,m4a,mp3；"
                                "只会选其中存在的一种。"
                            ),
                        )
                        gr.Markdown("#### 参考音频选择")
                        settings_components["autoflow_reference_wait_enabled"] = gr.Checkbox(
                            label="处理到参考音频时等待手动选择",
                            value=stored.autoflow_reference_wait_enabled,
                            info=(
                                "开启后，批量队列会显示选择器并提醒；关闭后直接使用"
                                "自动推荐的项目片段。"
                            ),
                        )
                        with gr.Group(
                            visible=stored.autoflow_reference_wait_enabled
                        ) as autoflow_reference_wait_group:
                            settings_components["autoflow_reference_wait_seconds"] = gr.Number(
                                label="每个作品最多等待（秒）",
                                value=stored.autoflow_reference_wait_seconds,
                                minimum=1,
                                maximum=3600,
                                precision=0,
                                info="默认 60 秒。到时未选择会自动继续，不会把队列判为失败。",
                            )
                        gr.Markdown("#### 和谐静态视频")
                        with gr.Row(elem_classes=["mobile-stack"]):
                            settings_components["autoflow_harmonized_volume_reduction_db"] = (
                                gr.Number(
                                    label="原声降低音量（dB）",
                                    value=stored.autoflow_harmonized_volume_reduction_db,
                                    info="填写正数；只影响选择“和谐静态视频”的任务。",
                                )
                            )
                            settings_components["autoflow_harmonized_delay_minutes"] = gr.Number(
                                label="原声、配音和字幕整体延后（分钟）",
                                value=stored.autoflow_harmonized_delay_minutes,
                                info="在成品开头加入相同长度的空档，让作品内容整体后移。",
                            )
                        settings_components["autoflow_timestamp_footer"] = gr.Textbox(
                            label="时间戳文档页脚",
                            value=stored.autoflow_timestamp_footer,
                            lines=5,
                            info="会写在总时间戳和分轨时间戳文档末尾；留空则不添加。所有作品共用。",
                        )
                        gr.Markdown("#### 标题文字")
                        with gr.Row(elem_classes=["mobile-stack"]):
                            settings_components["autoflow_translate_work_title"] = gr.Checkbox(
                                label="翻译作品文件夹名称",
                                value=stored.autoflow_translate_work_title,
                                info="用于成品标题和时间戳文档；关闭后保留原文件夹名称。",
                            )
                            settings_components["autoflow_translate_track_titles"] = gr.Checkbox(
                                label="翻译音轨标题",
                                value=stored.autoflow_translate_track_titles,
                                info="用于分轨文件名和曲目清单；关闭后保留原音轨标题。",
                            )

                        gr.Markdown("### 新作品默认值（可在批量处理页逐个覆盖）")
                        with gr.Row(elem_classes=["mobile-stack"]):
                            settings_components["autoflow_default_mode"] = gr.Radio(
                                label="默认输出类型",
                                choices=[
                                    (label, value) for value, label in AUTOFLOW_MODE_LABELS.items()
                                ],
                                value=stored.autoflow_default_mode,
                                info="打开批量处理页时填入；不会锁定每个作品的选择。",
                            )
                            settings_components["autoflow_default_layout"] = gr.Radio(
                                label="默认成品组织",
                                choices=[
                                    (label, value)
                                    for value, label in AUTOFLOW_LAYOUT_LABELS.items()
                                ],
                                value=stored.autoflow_default_layout,
                                info="只作为下一次批量任务的初始值。",
                            )
                        with gr.Row(elem_classes=["mobile-stack"]):
                            settings_components["autoflow_include_bonus"] = gr.Checkbox(
                                label="默认包含附加音轨",
                                value=stored.autoflow_include_bonus,
                                info=(
                                    "包括特典、样本、Free Talk 和闹钟等附加内容；批量页可按作品改。"
                                ),
                            )
                            settings_components["autoflow_embed_subtitles"] = gr.Checkbox(
                                label="默认在视频中内嵌字幕",
                                value=stored.autoflow_embed_subtitles,
                                info="只影响默认值；批量页可按作品关闭。",
                            )
                        with gr.Row(elem_classes=["mobile-stack"]):
                            settings_components["autoflow_background_policy"] = gr.Radio(
                                label="默认视频画面",
                                choices=[
                                    ("扫描后预选作品推荐图", "auto"),
                                    ("默认使用黑色背景", "black"),
                                ],
                                value=stored.autoflow_background_policy,
                                info="只作为视频任务的初始值；批量页扫描后可以改选作品图片。",
                            )

                with gr.Row():
                    save_defaults_button = gr.Button(
                        "仅保存为以后新项目默认值", variant="secondary"
                    )
                    apply_project_settings_button = gr.Button(
                        "保存并应用到当前项目", variant="primary"
                    )

            with gr.Tab("日志与诊断", id="logs"):
                gr.Markdown(
                    "程序运行日志保存在程序目录的 `.asmr-dubber/logs`。日志会自动轮转，"
                    "API 密钥和令牌会在写入前隐藏。遇到问题时可下载后随 Issue 一并提交。"
                )
                refresh_log_button = gr.Button("刷新日志")
                log_text = gr.Textbox(
                    label="最近日志",
                    value=recent_log_text(),
                    lines=22,
                    interactive=False,
                )
                log_file = gr.File(
                    label="下载完整日志",
                    value=stage_for_ui(application_log_path(), category="logs"),
                    interactive=False,
                )

        common_outputs = [
            project_path,
            project_summary,
            sentence_table,
            output_audio,
            stem_audio,
            output_video,
            subtitle_files,
            subtitle_video,
            diagnostics,
            status,
            reference_sentence,
            reference_audio,
        ]
        runtime_options = {
            "concurrency_id": "runtime_mutation",
            "concurrency_limit": 1,
            "show_progress": "full",
        }

        def run_project_task(
            label: str,
            action: Callable[..., ProjectView],
            *args: Any,
        ) -> tuple[Any, ...]:
            task_controller.begin(label)
            try:
                return _run_project_action(
                    action,
                    *args,
                    cancel_event=task_controller.cancel_event,
                )
            finally:
                task_controller.finish(label)

        def create_callback(
            source: Any,
            language: str,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "新建项目",
                create_project,
                source,
                language,
                _StageProgress(progress),
            )

        def open_callback(path: Any) -> tuple[Any, ...]:
            if not str(path or "").strip():
                return _empty_project_updates("请先选择最近项目或粘贴 project.json 路径。")
            return _run_project_action(load_view, str(path))

        def open_project_directory_callback(manifest: str) -> str:
            try:
                return open_project_directory(manifest)
            except Exception as exc:
                logger.exception("打开项目目录失败")
                return f"无法打开项目目录：{_safe_error(exc)}"

        def asr_callback(
            manifest: str,
            table: Any,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "ASR（语音识别）",
                analyze,
                manifest,
                table,
                _StageProgress(progress),
            )

        def import_transcript_callback(
            manifest: str,
            transcript: Any,
            text: str,
            timing: str,
            script_kind: str,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "导入台本/字幕",
                import_transcript_data,
                manifest,
                transcript,
                text,
                timing,
                script_kind,
                _StageProgress(progress),
            )

        def translate_callback(
            manifest: str,
            table: Any,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "翻译",
                translate,
                manifest,
                table,
                _StageProgress(progress),
            )

        def save_table_callback(manifest: str, table: Any) -> tuple[Any, ...]:
            return _run_project_action(save_table, manifest, table)

        def synthesize_callback(
            manifest: str,
            table: Any,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "TTS（语音合成）",
                synthesize,
                manifest,
                table,
                _StageProgress(progress),
            )

        def mix_callback(
            manifest: str,
            table: Any,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "混音与输出",
                mix,
                manifest,
                table,
                _StageProgress(progress),
            )

        def subtitle_callback(
            manifest: str,
            table: Any,
            language: str,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "生成字幕",
                subtitles,
                manifest,
                table,
                language,
                _StageProgress(progress),
            )

        def preview_reference_callback(manifest: str, sentence_id: str) -> Any:
            try:
                return preview_reference(manifest, sentence_id)
            except Exception:
                logger.exception("参考音频试听更新失败")
                return gr.update()

        def refresh_log_callback() -> tuple[str, str | None]:
            return (
                recent_log_text(),
                stage_for_ui(application_log_path(), category="logs"),
            )

        def edge_tts_preview_callback(voice: Any) -> tuple[Any, str]:
            try:
                preview = preview_edge_tts_voice(str(voice or ""))
                return gr.update(value=preview, visible=True), "试听已生成。"
            except Exception as exc:
                logger.exception("Edge TTS 音色试听失败")
                return gr.update(value=None, visible=False), f"试听失败：{_safe_error(exc)}"

        def refresh_projects_callback() -> Any:
            current = load_user_settings()
            choices = recent_projects(current.projects_root or None)
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def _stage_autoflow_background(path: str | None) -> str | None:
            return stage_for_ui(Path(path), category="autoflow-backgrounds") if path else None

        def autoflow_scan_callback(folder: Any, include_bonus: bool) -> tuple[Any, ...]:
            try:
                scanned = scan_for_ui(folder, include_bonus)
                return (
                    gr.update(value=scanned.folder),
                    gr.update(
                        choices=scanned.edition_choices,
                        value=scanned.selected_edition,
                    ),
                    gr.update(
                        choices=scanned.background_choices,
                        value=scanned.selected_background,
                    ),
                    _stage_autoflow_background(scanned.selected_background_preview),
                    scanned.source_payloads,
                    scanned.track_items,
                    scanned.summary,
                    "已载入推荐版本；可以调整选项后加入队列。",
                    "",
                    gr.update(value="加入处理队列"),
                    gr.update(visible="hidden"),
                )
            except Exception as exc:
                logger.exception("扫描自动处理作品失败")
                message = f"扫描失败：{_safe_error(exc)}"
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    message,
                    message,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        def autoflow_preview_callback(
            folder: Any,
            edition: Any,
            include_bonus: bool,
        ) -> tuple[Any, ...]:
            if not str(folder or "").strip() or not str(edition or "").strip():
                return gr.update(), gr.update(), "请先扫描作品。"
            try:
                view = preview_edition_for_ui(folder, edition, include_bonus)
                return view.source_payloads, view.track_items, view.summary
            except Exception as exc:
                logger.exception("更新自动处理音轨预览失败")
                return gr.update(), gr.update(), f"无法更新音轨预览：{_safe_error(exc)}"

        def _with_event_data(function: Callable[..., Any]) -> Callable[..., Any]:
            function.__annotations__["evt"] = gr.EventData
            return function

        @_with_event_data
        def autoflow_track_reorder_callback(
            source_payloads: Any,
            folder: Any,
            evt: gr.EventData,
        ) -> tuple[Any, ...]:
            try:
                view = reorder_tracks_for_ui(
                    folder,
                    source_payloads,
                    getattr(evt, "order", []),
                )
                return view.source_payloads, view.track_items, view.summary
            except Exception as exc:
                logger.exception("调整自动处理音轨顺序失败")
                return gr.update(), gr.update(), f"无法调整音轨顺序：{_safe_error(exc)}"

        @_with_event_data
        def autoflow_track_subtitle_callback(
            source_payloads: Any,
            folder: Any,
            evt: gr.EventData,
        ) -> tuple[Any, ...]:
            try:
                view = set_track_subtitle_for_ui(
                    folder,
                    source_payloads,
                    getattr(evt, "track_id", ""),
                    getattr(evt, "transcript", ""),
                    getattr(evt, "language", ""),
                )
                return view.source_payloads, view.track_items, view.summary
            except Exception as exc:
                logger.exception("更新自动处理字幕选择失败")
                return gr.update(), gr.update(), f"无法更新字幕选择：{_safe_error(exc)}"

        def autoflow_background_callback(folder: Any, background: Any) -> str | None:
            try:
                path = background_preview_path(folder, background)
                return _stage_autoflow_background(str(path) if path else None)
            except Exception:
                logger.exception("更新自动处理画面预览失败")
                return None

        def _autoflow_queue_updates(
            items: list[dict[str, Any]],
            message: str,
        ) -> tuple[Any, ...]:
            return items, queue_items_for_ui(items), message

        def autoflow_add_callback(
            queue_payload: Any,
            editing_plan_id: Any,
            folder: Any,
            edition: Any,
            source_payloads: Any,
            mode: Any,
            layout: Any,
            background: Any,
            embed_subtitles: bool,
            rebuild: bool,
        ) -> tuple[Any, ...]:
            try:
                plan = build_plan_for_ui(
                    folder,
                    edition,
                    source_payloads,
                    mode,
                    layout,
                    background,
                    embed_subtitles,
                    rebuild,
                )
                selected = str(editing_plan_id or "").strip()
                items = (
                    replace_plan_in_queue(queue_payload, selected, plan)
                    if selected
                    else add_plan_to_queue(queue_payload, plan)
                )
                action = "已更新队列任务" if selected else "已加入队列"
                queue_values = _autoflow_queue_updates(
                    items, f"{action}：{Path(str(plan['folder'])).name}。"
                )
                return (
                    *queue_values,
                    "",
                    gr.update(value="加入处理队列"),
                    gr.update(visible="hidden"),
                )
            except Exception as exc:
                logger.exception("加入自动处理队列失败")
                items = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
                return (
                    *_autoflow_queue_updates(items, f"无法保存队列任务：{_safe_error(exc)}"),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        @_with_event_data
        def autoflow_queue_reorder_callback(
            queue_payload: Any,
            evt: gr.EventData,
        ) -> tuple[Any, ...]:
            try:
                items = reorder_queue_for_ui(queue_payload, getattr(evt, "order", []))
                return _autoflow_queue_updates(items, "已更新队列顺序。")
            except Exception as exc:
                logger.exception("调整自动处理队列顺序失败")
                items = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
                return _autoflow_queue_updates(items, f"无法调整队列顺序：{_safe_error(exc)}")

        @_with_event_data
        def autoflow_queue_restart_callback(
            queue_payload: Any,
            evt: gr.EventData,
        ) -> tuple[Any, ...]:
            try:
                items = toggle_plan_rebuild(queue_payload, getattr(evt, "plan_id", ""))
                return _autoflow_queue_updates(items, "已更新这个任务的重新处理状态。")
            except Exception as exc:
                logger.exception("更新自动处理重做状态失败")
                items = [dict(item) for item in (queue_payload or []) if isinstance(item, dict)]
                return _autoflow_queue_updates(items, f"无法更新任务：{_safe_error(exc)}")

        @_with_event_data
        def autoflow_queue_remove_callback(
            queue_payload: Any,
            evt: gr.EventData,
        ) -> tuple[Any, ...]:
            items = remove_plan_from_queue(queue_payload, getattr(evt, "plan_id", ""))
            return (
                *_autoflow_queue_updates(items, "已从队列移除这个作品。"),
                "",
                gr.update(value="加入处理队列"),
                gr.update(visible="hidden"),
            )

        @_with_event_data
        def autoflow_queue_edit_callback(
            queue_payload: Any,
            evt: gr.EventData,
        ) -> tuple[Any, ...]:
            try:
                view = edit_plan_for_ui(queue_payload, getattr(evt, "plan_id", ""))
                return (
                    view.plan_id,
                    gr.update(value=view.folder),
                    gr.update(
                        choices=view.edition_choices,
                        value=view.selected_edition,
                    ),
                    gr.update(value=view.include_bonus),
                    view.source_payloads,
                    view.track_items,
                    view.scan_summary,
                    view.selection_summary,
                    gr.update(value=view.mode),
                    gr.update(value=view.layout),
                    gr.update(
                        choices=view.background_choices,
                        value=view.selected_background,
                    ),
                    _stage_autoflow_background(view.selected_background_preview),
                    gr.update(value=view.embed_subtitles),
                    gr.update(visible=view.mode != "audio"),
                    gr.update(value=view.rebuild),
                    gr.update(value="保存队列修改"),
                    gr.update(visible=True),
                    f"正在编辑队列中的“{Path(view.folder).name}”。修改后点击“保存队列修改”。",
                )
            except Exception as exc:
                logger.exception("载入自动处理队列任务失败")
                return (
                    *(gr.update() for _index in range(17)),
                    f"无法编辑队列任务：{_safe_error(exc)}",
                )

        def autoflow_cancel_edit_callback() -> tuple[Any, ...]:
            return (
                "",
                gr.update(value="加入处理队列"),
                gr.update(visible="hidden"),
                "已取消编辑；队列中的原任务没有改变。",
            )

        def autoflow_clear_callback() -> tuple[Any, ...]:
            return (
                *_autoflow_queue_updates([], "队列已清空。"),
                "",
                gr.update(value="加入处理队列"),
                gr.update(visible="hidden"),
            )

        def autoflow_run_callback(queue_payload: Any) -> Iterator[tuple[Any, ...]]:
            runtime_by_plan: dict[str, dict[str, Any]] = {}
            for logs, done, _success, message, outputs, reference_event in _autoflow_log_events(
                queue_payload,
                autoflow_controller,
            ):
                choices = [(Path(path).name, path) for path in outputs]
                queue_update: Any = gr.update()
                reference_updates: tuple[Any, ...] = tuple(gr.update() for _ in range(9))
                if reference_event:
                    event_kind = str(reference_event.get("kind") or "")
                    plan_id = str(reference_event.get("plan_id") or "")
                    work = str(reference_event.get("work") or "当前作品")
                    timeout_seconds = int(reference_event.get("timeout_seconds") or 0)
                    if event_kind == "ready":
                        runtime_by_plan[plan_id] = {
                            "reference_ready": True,
                            "request_id": str(reference_event.get("request_id") or ""),
                            "status": (
                                f"可以自选参考音频；最多等待 {timeout_seconds} 秒。"
                                "不操作会使用自动推荐。"
                            ),
                        }
                        project_json = str(reference_event.get("project_json") or "")
                        try:
                            sentence_choices, selected, preview = reference_picker(project_json)
                            picker_note = (
                                f"**{work}** 已完成识别和翻译。请选择一个项目片段，"
                                "或在下面导入外部音频。"
                            )
                        except Exception as exc:
                            logger.exception("载入批量任务参考音频选择器失败")
                            sentence_choices, selected, preview = [], None, None
                            picker_note = (
                                f"**{work}** 已进入参考音频步骤，但选择器载入失败："
                                f"{_safe_error(exc)}。不操作仍会按时使用自动推荐。"
                            )
                        reference_updates = (
                            dict(reference_event),
                            gr.update(
                                visible=True,
                                label=f"{work} · 自选参考音频（可选）",
                            ),
                            picker_note,
                            gr.update(choices=sentence_choices, value=selected),
                            gr.update(value=preview),
                            gr.update(value=None),
                            gr.update(value=""),
                            gr.update(value="auto"),
                            (
                                f"等待 {timeout_seconds} 秒；不做选择时，"
                                "程序会使用上方带“推荐”标记的项目片段。"
                            ),
                        )
                    else:
                        selected_by_user = event_kind == "selected"
                        source = str(reference_event.get("source") or "")
                        runtime_by_plan[plan_id] = {
                            "reference_ready": False,
                            "request_id": "",
                            "status": (
                                "已使用外部参考音频，任务继续。"
                                if selected_by_user and source == "external"
                                else "已使用所选项目片段，任务继续。"
                                if selected_by_user
                                else "等待时间结束，已使用自动推荐的项目片段。"
                            ),
                        }
                        reference_updates = (
                            {},
                            gr.update(visible=False),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(value=None),
                            gr.update(value=""),
                            gr.update(value="auto"),
                            gr.update(),
                        )
                    queue_update = queue_items_for_ui(
                        queue_payload,
                        runtime=runtime_by_plan,
                    )
                elif done:
                    reference_updates = (
                        {},
                        gr.update(visible=False),
                        *(gr.update() for _ in range(7)),
                    )
                yield (
                    logs,
                    message,
                    gr.update(
                        choices=choices,
                        value=choices[0][1] if choices else None,
                    ),
                    queue_update,
                    *reference_updates,
                )

        def _active_autoflow_reference_project(request: Any) -> str:
            if not isinstance(request, dict):
                raise ValueError("当前没有正在等待选择参考音频的作品。")
            project_json = str(request.get("project_json") or "").strip()
            if not project_json or not Path(project_json).is_file():
                raise ValueError("当前参考音频项目已经不存在。")
            deadline = float(request.get("deadline_epoch") or 0)
            if deadline and time.time() >= deadline:
                raise ValueError("这个作品的选择时间已经结束，程序将使用自动推荐。")
            return project_json

        def autoflow_reference_preview_callback(request: Any, sentence_id: Any) -> Any:
            try:
                project_json = _active_autoflow_reference_project(request)
                return preview_reference(project_json, str(sentence_id or ""))
            except Exception:
                logger.exception("更新批量任务参考音频试听失败")
                return gr.update()

        def autoflow_reference_apply_callback(
            request: Any,
            sentence_id: Any,
        ) -> tuple[Any, Any]:
            try:
                project_json = _active_autoflow_reference_project(request)
                if not str(sentence_id or "").strip():
                    raise ValueError("请先选择一个项目片段。")
                return select_autoflow_project_reference(project_json, str(sentence_id))
            except Exception as exc:
                logger.exception("保存批量任务项目参考音频失败")
                return f"无法保存参考音频：{_safe_error(exc)}", gr.update()

        def autoflow_reference_external_callback(
            request: Any,
            upload: Any,
            text: Any,
            language: Any,
        ) -> tuple[Any, Any]:
            try:
                project_json = _active_autoflow_reference_project(request)
                if not upload:
                    raise ValueError("请先选择要导入的外部音频。")
                return select_autoflow_external_reference(
                    project_json,
                    str(upload),
                    text=str(text or ""),
                    language=str(language or "auto"),
                )
            except Exception as exc:
                logger.exception("保存批量任务外部参考音频失败")
                return f"无法导入外部参考音频：{_safe_error(exc)}", gr.update()

        def autoflow_open_output_callback(path: Any) -> str:
            try:
                if not str(path or "").strip():
                    return "还没有可打开的输出目录。"
                return open_output_directory(path)
            except Exception as exc:
                logger.exception("打开自动处理输出目录失败")
                return f"无法打开输出目录：{_safe_error(exc)}"

        def pick_reference_callback(manifest: str, sentence_id: str) -> tuple[Any, Any]:
            try:
                return select_reference(manifest, sentence_id)
            except Exception as exc:
                return f"保存参考句失败：{_safe_error(exc)}", gr.update()

        create_button.click(
            create_callback,
            inputs=[source_input, settings_components["default_source_language"]],
            outputs=common_outputs,
            api_name="create_project",
            **runtime_options,
        )
        open_project_button.click(
            open_callback,
            inputs=[recent_project],
            outputs=common_outputs,
            api_name="open_project",
            **runtime_options,
        )
        open_project_directory_button.click(
            open_project_directory_callback,
            inputs=[project_path],
            outputs=[status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        refresh_projects_button.click(
            refresh_projects_callback,
            outputs=[recent_project],
            api_name=_PRIVATE_API,
            queue=False,
        )
        asr_button.click(
            asr_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            api_name="run_asr",
            **runtime_options,
        )
        import_transcript_button.click(
            import_transcript_callback,
            inputs=[
                project_path,
                transcript_file,
                transcript_text,
                plain_timing,
                transcript_kind,
            ],
            outputs=common_outputs,
            api_name="import_transcript",
            **runtime_options,
        )
        transcript_kind.change(
            _transcript_kind_update,
            inputs=[transcript_kind],
            outputs=[plain_timing],
            api_name=_PRIVATE_API,
            queue=False,
        )
        translate_button.click(
            translate_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            api_name="translate_project",
            **runtime_options,
        )
        save_table_button.click(
            save_table_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            api_name="save_sentence_table",
            **runtime_options,
        )
        synthesize_button.click(
            synthesize_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            api_name="synthesize_project",
            **runtime_options,
        )
        mix_button.click(
            mix_callback,
            inputs=[project_path, sentence_table],
            outputs=common_outputs,
            api_name="mix_project",
            **runtime_options,
        )
        subtitle_button.click(
            subtitle_callback,
            inputs=[project_path, sentence_table, subtitle_language],
            outputs=common_outputs,
            api_name="generate_subtitles",
            **runtime_options,
        )
        cancel_task_button.click(
            task_controller.cancel,
            outputs=[status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        reference_sentence.change(
            preview_reference_callback,
            inputs=[project_path, reference_sentence],
            outputs=[reference_audio],
            api_name=_PRIVATE_API,
            queue=False,
        )
        save_reference_button.click(
            pick_reference_callback,
            inputs=[project_path, reference_sentence],
            outputs=[status, reference_audio],
            api_name=_PRIVATE_API,
            **runtime_options,
        )
        autoflow_scan_button.click(
            autoflow_scan_callback,
            inputs=[autoflow_folder, autoflow_include_bonus],
            outputs=[
                autoflow_folder,
                autoflow_edition,
                autoflow_background,
                autoflow_background_preview,
                autoflow_track_state,
                autoflow_tracks,
                autoflow_scan_summary,
                autoflow_selection_summary,
                autoflow_edit_plan_state,
                autoflow_add_button,
                autoflow_cancel_edit_button,
            ],
            api_name="scan_autoflow_work",
            queue=False,
        )
        autoflow_edition.change(
            autoflow_preview_callback,
            inputs=[autoflow_folder, autoflow_edition, autoflow_include_bonus],
            outputs=[autoflow_track_state, autoflow_tracks, autoflow_selection_summary],
            api_name=_PRIVATE_API,
            queue=False,
            trigger_mode="always_last",
        )
        autoflow_include_bonus.change(
            autoflow_preview_callback,
            inputs=[autoflow_folder, autoflow_edition, autoflow_include_bonus],
            outputs=[autoflow_track_state, autoflow_tracks, autoflow_selection_summary],
            api_name=_PRIVATE_API,
            queue=False,
            trigger_mode="always_last",
        )
        autoflow_tracks.track_reorder(
            autoflow_track_reorder_callback,
            inputs=[autoflow_track_state, autoflow_folder],
            outputs=[autoflow_track_state, autoflow_tracks, autoflow_selection_summary],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_tracks.track_subtitle(
            autoflow_track_subtitle_callback,
            inputs=[autoflow_track_state, autoflow_folder],
            outputs=[autoflow_track_state, autoflow_tracks, autoflow_selection_summary],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_background.change(
            autoflow_background_callback,
            inputs=[autoflow_folder, autoflow_background],
            outputs=[autoflow_background_preview],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_mode.change(
            lambda mode: gr.update(visible=str(mode or "audio") != "audio"),
            inputs=[autoflow_mode],
            outputs=[autoflow_video_group],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_add_button.click(
            autoflow_add_callback,
            inputs=[
                autoflow_queue_state,
                autoflow_edit_plan_state,
                autoflow_folder,
                autoflow_edition,
                autoflow_track_state,
                autoflow_mode,
                autoflow_layout,
                autoflow_background,
                autoflow_embed_subtitles,
                autoflow_rebuild,
            ],
            outputs=[
                autoflow_queue_state,
                autoflow_queue_list,
                autoflow_status,
                autoflow_edit_plan_state,
                autoflow_add_button,
                autoflow_cancel_edit_button,
            ],
            api_name="add_autoflow_work",
            queue=False,
        )
        autoflow_cancel_edit_button.click(
            autoflow_cancel_edit_callback,
            outputs=[
                autoflow_edit_plan_state,
                autoflow_add_button,
                autoflow_cancel_edit_button,
                autoflow_status,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_queue_list.queue_reorder(
            autoflow_queue_reorder_callback,
            inputs=[autoflow_queue_state],
            outputs=[
                autoflow_queue_state,
                autoflow_queue_list,
                autoflow_status,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_queue_list.queue_restart(
            autoflow_queue_restart_callback,
            inputs=[autoflow_queue_state],
            outputs=[autoflow_queue_state, autoflow_queue_list, autoflow_status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_queue_list.queue_remove(
            autoflow_queue_remove_callback,
            inputs=[autoflow_queue_state],
            outputs=[
                autoflow_queue_state,
                autoflow_queue_list,
                autoflow_status,
                autoflow_edit_plan_state,
                autoflow_add_button,
                autoflow_cancel_edit_button,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_queue_list.queue_edit(
            autoflow_queue_edit_callback,
            inputs=[autoflow_queue_state],
            outputs=[
                autoflow_edit_plan_state,
                autoflow_folder,
                autoflow_edition,
                autoflow_include_bonus,
                autoflow_track_state,
                autoflow_tracks,
                autoflow_scan_summary,
                autoflow_selection_summary,
                autoflow_mode,
                autoflow_layout,
                autoflow_background,
                autoflow_background_preview,
                autoflow_embed_subtitles,
                autoflow_video_group,
                autoflow_rebuild,
                autoflow_add_button,
                autoflow_cancel_edit_button,
                autoflow_status,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_clear_button.click(
            autoflow_clear_callback,
            outputs=[
                autoflow_queue_state,
                autoflow_queue_list,
                autoflow_status,
                autoflow_edit_plan_state,
                autoflow_add_button,
                autoflow_cancel_edit_button,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_run_button.click(
            autoflow_run_callback,
            inputs=[autoflow_queue_state],
            outputs=[
                autoflow_log,
                autoflow_status,
                autoflow_output_selection,
                autoflow_queue_list,
                autoflow_reference_state,
                autoflow_reference_panel,
                autoflow_reference_info,
                autoflow_reference_sentence,
                autoflow_reference_audio,
                autoflow_reference_upload,
                autoflow_reference_external_text,
                autoflow_reference_external_language,
                autoflow_reference_status,
            ],
            api_name="run_autoflow_queue",
            concurrency_id="runtime_mutation",
            concurrency_limit=1,
            show_progress="minimal",
        )
        autoflow_reference_sentence.change(
            autoflow_reference_preview_callback,
            inputs=[autoflow_reference_state, autoflow_reference_sentence],
            outputs=[autoflow_reference_audio],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_reference_apply_button.click(
            autoflow_reference_apply_callback,
            inputs=[autoflow_reference_state, autoflow_reference_sentence],
            outputs=[autoflow_reference_status, autoflow_reference_audio],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_reference_upload.change(
            lambda path: gr.update(value=path) if path else gr.update(),
            inputs=[autoflow_reference_upload],
            outputs=[autoflow_reference_audio],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_reference_external_button.click(
            autoflow_reference_external_callback,
            inputs=[
                autoflow_reference_state,
                autoflow_reference_upload,
                autoflow_reference_external_text,
                autoflow_reference_external_language,
            ],
            outputs=[autoflow_reference_status, autoflow_reference_audio],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_cancel_button.click(
            autoflow_controller.cancel,
            outputs=[autoflow_status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        autoflow_open_output_button.click(
            autoflow_open_output_callback,
            inputs=[autoflow_output_selection],
            outputs=[autoflow_status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        refresh_log_button.click(
            refresh_log_callback,
            outputs=[log_text, log_file],
            api_name=_PRIVATE_API,
            queue=False,
        )

        source_language_event = settings_components["default_source_language"].change(
            _source_language_backend_update,
            inputs=[
                settings_components["default_source_language"],
                settings_components["asr_backend"],
            ],
            outputs=[settings_components["asr_backend"], source_language_help],
            api_name=_PRIVATE_API,
            queue=False,
        )
        source_language_event.then(
            _asr_backend_update,
            inputs=[
                settings_components["asr_backend"],
                settings_components["asr_vad_mode"],
                settings_components["default_source_language"],
            ],
            outputs=[
                settings_components["asr_model"],
                asr_help,
                settings_components["asr_compute_type"],
                settings_components["asr_beam_size"],
                settings_components["asr_condition_on_previous_text"],
                settings_components["asr_initial_prompt"],
                settings_components["asr_timeout_seconds"],
                settings_components["asr_chunk_seconds"],
                settings_components["asr_parakeet_decoder"],
                settings_components["asr_kotoba_chunk_seconds"],
                settings_components["asr_vad_mode"],
                settings_components["asr_vad_min_silence_ms"],
                settings_components["asr_asmr_vad_threshold"],
                settings_components["asr_asmr_vad_min_speech_ms"],
                settings_components["asr_asmr_vad_min_silence_ms"],
                settings_components["asr_asmr_vad_speech_pad_ms"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        ).then(
            _review_language_update,
            inputs=[
                settings_components["default_source_language"],
                settings_components["asr_review_enabled"],
                settings_components["asr_review_models"],
                settings_components["asr_review_text_priority_model"],
                settings_components["asr_review_timestamp_priority_model"],
                settings_components["asr_backend"],
                settings_components["asr_model"],
            ],
            outputs=[
                settings_components["asr_review_enabled"],
                settings_components["asr_review_models"],
                settings_components["asr_review_text_priority_model"],
                settings_components["asr_review_timestamp_priority_model"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        source_language_event.then(
            _translation_prompt_language_update,
            inputs=[
                settings_components["default_source_language"],
                settings_components["translation_prompt"],
                settings_components["translation_prompt_drafts"],
                active_prompt_language,
            ],
            outputs=[
                settings_components["translation_prompt"],
                settings_components["translation_prompt_drafts"],
                active_prompt_language,
                translation_prompt_note,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["asr_backend"].change(
            _asr_backend_update,
            inputs=[
                settings_components["asr_backend"],
                settings_components["asr_vad_mode"],
                settings_components["default_source_language"],
            ],
            outputs=[
                settings_components["asr_model"],
                asr_help,
                settings_components["asr_compute_type"],
                settings_components["asr_beam_size"],
                settings_components["asr_condition_on_previous_text"],
                settings_components["asr_initial_prompt"],
                settings_components["asr_timeout_seconds"],
                settings_components["asr_chunk_seconds"],
                settings_components["asr_parakeet_decoder"],
                settings_components["asr_kotoba_chunk_seconds"],
                settings_components["asr_vad_mode"],
                settings_components["asr_vad_min_silence_ms"],
                settings_components["asr_asmr_vad_threshold"],
                settings_components["asr_asmr_vad_min_speech_ms"],
                settings_components["asr_asmr_vad_min_silence_ms"],
                settings_components["asr_asmr_vad_speech_pad_ms"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["asr_model"].change(
            _asr_model_update,
            inputs=[settings_components["asr_backend"], settings_components["asr_model"]],
            outputs=[
                settings_components["asr_parakeet_decoder"],
                settings_components["asr_kotoba_chunk_seconds"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["asr_vad_mode"].change(
            _asr_vad_update,
            inputs=[settings_components["asr_vad_mode"]],
            outputs=[
                settings_components["asr_vad_min_silence_ms"],
                settings_components["asr_asmr_vad_threshold"],
                settings_components["asr_asmr_vad_min_speech_ms"],
                settings_components["asr_asmr_vad_min_silence_ms"],
                settings_components["asr_asmr_vad_speech_pad_ms"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["asr_review_enabled"].change(
            _review_visibility_update,
            inputs=[settings_components["asr_review_enabled"]],
            outputs=[
                settings_components["asr_review_models"],
                settings_components["asr_review_text_priority_model"],
                settings_components["asr_review_timestamp_priority_model"],
                settings_components["asr_review_max_drift_seconds"],
                settings_components["asr_review_background"],
                settings_components["asr_review_prompt"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["translation_provider"].change(
            _provider_update,
            inputs=[settings_components["translation_provider"]],
            outputs=[
                settings_components["translation_model"],
                settings_components["translation_base_url"],
                translation_help,
                translation_key_status,
                llm_translation_group,
                deepl_translation_group,
                microsoft_translation_group,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["translation_send_context"].change(
            lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[settings_components["translation_send_context"]],
            outputs=[settings_components["translation_context_sentences"]],
            api_name=_PRIVATE_API,
            queue=False,
        )
        reset_translation_prompt_button.click(
            _reset_translation_prompt,
            inputs=[
                settings_components["default_source_language"],
                settings_components["translation_prompt_drafts"],
            ],
            outputs=[
                settings_components["translation_prompt"],
                settings_components["translation_prompt_drafts"],
                translation_prompt_note,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        tts_backend_event = (
            settings_components["tts_backend"]
            .change(
                _tts_backend_update,
                inputs=[settings_components["tts_backend"]],
                outputs=[
                    settings_components["tts_model"],
                    settings_components["tts_api_base_url"],
                    tts_help,
                    tts_key_status,
                    index_group,
                    generic_tts_group,
                    gpt_group,
                    cosy_group,
                    settings_components["tts_device"],
                    settings_components["tts_request_concurrency"],
                    tts_sampling_group,
                    settings_components["tts_speed"],
                    settings_components["tts_voice"],
                    tts_voice_group,
                    minimax_group,
                    mimo_group,
                ],
                api_name=_PRIVATE_API,
                queue=False,
            )
            .then(
                _tts_service_visibility,
                inputs=[settings_components["tts_backend"]],
                outputs=[
                    settings_components["tts_api_base_url"],
                    tts_key,
                    tts_key_status,
                    tts_key_buttons,
                ],
                api_name=_PRIVATE_API,
                queue=False,
            )
        )
        tts_detail_inputs = [
            settings_components["tts_backend"],
            settings_components["tts_model"],
            settings_components["tts_reference_source"],
            settings_components["tts_index_speaker_source"],
            settings_components["tts_index_emotion_source"],
            settings_components["tts_cosyvoice_mode"],
        ]
        tts_detail_outputs = [
            external_speaker_group,
            settings_components["tts_external_reference_text"],
            settings_components["tts_external_reference_language"],
            external_emotion_group,
            settings_components["tts_index_emo_text"],
        ]
        tts_backend_event.then(
            _tts_detail_visibility,
            inputs=tts_detail_inputs,
            outputs=tts_detail_outputs,
            api_name=_PRIVATE_API,
            queue=False,
        )
        tts_backend_event.then(
            lambda backend: (
                gr.update(visible=str(backend or "") == "edge_tts"),
                gr.update(value=None, visible=False),
                "",
            ),
            inputs=[settings_components["tts_backend"]],
            outputs=[edge_tts_preview_button, edge_tts_preview_audio, edge_tts_preview_status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["tts_voice"].change(
            lambda: (gr.update(value=None, visible=False), ""),
            outputs=[edge_tts_preview_audio, edge_tts_preview_status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        edge_tts_preview_button.click(
            edge_tts_preview_callback,
            inputs=[settings_components["tts_voice"]],
            outputs=[edge_tts_preview_audio, edge_tts_preview_status],
            api_name="preview_edge_tts_voice",
            concurrency_limit=1,
            show_progress="full",
        )
        settings_components["tts_model"].change(
            _tts_model_controls_update,
            inputs=[settings_components["tts_backend"], settings_components["tts_model"]],
            outputs=[generic_tts_group, tts_voice_group, minimax_group, mimo_group],
            api_name=_PRIVATE_API,
            queue=False,
        )
        for detail_component in (
            settings_components["tts_model"],
            settings_components["tts_reference_source"],
            settings_components["tts_index_speaker_source"],
            settings_components["tts_index_emotion_source"],
            settings_components["tts_cosyvoice_mode"],
        ):
            detail_component.change(
                _tts_detail_visibility,
                inputs=tts_detail_inputs,
                outputs=tts_detail_outputs,
                api_name=_PRIVATE_API,
                queue=False,
            )
        settings_components["tts_model_path"].change(
            indextts_installation_status,
            inputs=[settings_components["tts_model_path"]],
            outputs=[index_status],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["chinese_dubbing_timing_mode"].change(
            lambda mode: gr.update(visible=str(mode or "fit_window") == "fit_window"),
            inputs=[settings_components["chinese_dubbing_timing_mode"]],
            outputs=[auto_speed_group],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["loudness_mode"].change(
            _loudness_mode_update,
            inputs=[settings_components["loudness_mode"]],
            outputs=[
                source_loudness_group,
                uniform_loudness_group,
                raw_loudness_group,
                source_loudness_advanced_group,
                normalized_loudness_advanced_group,
                adjusted_loudness_gain_group,
                loudness_mode_help,
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        reset_loudness_button.click(
            lambda: (
                DEFAULT_CHINESE_RELATIVE_LOUDNESS_DB,
                -30.0,
                0.0,
                -42.0,
                -30.0,
                12.0,
                -9.0,
                8.0,
                0.0,
                -3.0,
            ),
            outputs=[
                settings_components["chinese_relative_loudness_db"],
                settings_components["loudness_uniform_target_dbfs"],
                settings_components["loudness_raw_gain_db"],
                settings_components["chinese_min_active_rms_dbfs"],
                settings_components["loudness_source_ceiling_dbfs"],
                settings_components["chinese_max_loudness_boost_db"],
                settings_components["chinese_line_peak_dbfs"],
                settings_components["chinese_fade_ms"],
                settings_components["chinese_gain_db"],
                settings_components["chinese_stem_peak_dbfs"],
            ],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["mix_output_mode"].change(
            lambda mode: gr.update(visible=str(mode or "both") != "stem"),
            inputs=[settings_components["mix_output_mode"]],
            outputs=[final_mix_group],
            api_name=_PRIVATE_API,
            queue=False,
        )
        settings_components["autoflow_reference_wait_enabled"].change(
            lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[settings_components["autoflow_reference_wait_enabled"]],
            outputs=[autoflow_reference_wait_group],
            api_name=_PRIVATE_API,
            queue=False,
        )

        field_names = list(settings_components)
        field_components = [settings_components[name] for name in field_names]
        form_inputs = [*field_components, external_speaker_upload, external_emotion_upload]

        def parse_form(*values: Any) -> UserSettings:
            return _settings_from_form(
                field_names,
                values[: len(field_names)],
                values[-2],
                values[-1],
            )

        def save_defaults_callback(*values: Any) -> str:
            try:
                settings = parse_form(*values)
                path = save_user_settings(settings)
                return (
                    f"新项目默认值已保存：{path}\n"
                    "当前已打开的项目和已经扫描的批量任务没有改变；"
                    "需要时请点击“保存并应用到当前项目”。"
                )
            except Exception as exc:
                return f"保存设置失败：{_safe_error(exc)}"

        def apply_settings_callback(manifest: str, *values: Any) -> tuple[Any, ...]:
            try:
                settings = parse_form(*values)
            except Exception as exc:
                message = f"设置校验失败：{_safe_error(exc)}"
                return message, *_empty_project_updates(message)
            try:
                path = save_user_settings(settings)
            except Exception as exc:
                message = f"保存新项目默认值失败：{_safe_error(exc)}"
                return message, *_empty_project_updates(message)

            normalized_manifest = str(manifest or "").strip()
            if not normalized_manifest:
                message = (
                    f"新项目默认值已保存：{path}\n当前没有打开项目，因此没有可应用的当前项目。"
                )
                return (
                    message,
                    *_empty_project_updates("默认设置已保存；当前没有打开项目。"),
                )

            try:
                project_view = apply_global_settings(normalized_manifest, settings)
            except Exception as exc:
                project_message = f"新项目默认值已经保存，但应用到当前项目失败：{_safe_error(exc)}"
                settings_message = f"新项目默认值已保存：{path}\n{project_message}"
                return (
                    settings_message,
                    *_empty_project_updates(project_message),
                )

            settings_message = f"新项目默认值已保存：{path}\n{project_view.status}"
            return (
                settings_message,
                *_view_values(project_view),
            )

        save_defaults_button.click(
            save_defaults_callback,
            inputs=form_inputs,
            outputs=[settings_status],
            api_name="save_default_settings",
            **runtime_options,
        )
        apply_project_settings_button.click(
            apply_settings_callback,
            inputs=[project_path, *form_inputs],
            outputs=[
                settings_status,
                *common_outputs,
            ],
            api_name="apply_settings_to_project",
            **runtime_options,
        )

        def save_translation_key_callback(provider_id: str, key: str) -> str:
            try:
                save_api_key(provider_id, key)
                return api_key_status(provider_id)
            except Exception as exc:
                return f"保存密钥失败：{_safe_error(exc)}"

        def clear_translation_key_callback(provider_id: str) -> str:
            try:
                clear_api_key(provider_id)
                return api_key_status(provider_id)
            except Exception as exc:
                return f"清除密钥失败：{_safe_error(exc)}"

        save_translation_key_button.click(
            save_translation_key_callback,
            inputs=[settings_components["translation_provider"], translation_key],
            outputs=[translation_key_status],
            api_name=_PRIVATE_API,
            **runtime_options,
        )
        clear_translation_key_button.click(
            clear_translation_key_callback,
            inputs=[settings_components["translation_provider"]],
            outputs=[translation_key_status],
            api_name=_PRIVATE_API,
            **runtime_options,
        )

        def save_tts_key_callback(backend_id: str, key: str) -> str:
            try:
                save_service_key(f"tts:{backend_id}", key)
                return service_key_status(f"tts:{backend_id}", TTS_BACKENDS[backend_id].api_key)
            except Exception as exc:
                return f"保存密钥失败：{_safe_error(exc)}"

        def clear_tts_key_callback(backend_id: str) -> str:
            try:
                clear_service_key(f"tts:{backend_id}")
                return service_key_status(f"tts:{backend_id}", TTS_BACKENDS[backend_id].api_key)
            except Exception as exc:
                return f"清除密钥失败：{_safe_error(exc)}"

        save_tts_key_button.click(
            save_tts_key_callback,
            inputs=[settings_components["tts_backend"], tts_key],
            outputs=[tts_key_status],
            api_name=_PRIVATE_API,
            **runtime_options,
        )
        clear_tts_key_button.click(
            clear_tts_key_callback,
            inputs=[settings_components["tts_backend"]],
            outputs=[tts_key_status],
            api_name=_PRIVATE_API,
            **runtime_options,
        )

        availability_inputs = [
            settings_components["asr_review_enabled"],
            settings_components["asr_review_models"],
            settings_components["asr_review_text_priority_model"],
            settings_components["asr_review_timestamp_priority_model"],
            settings_components["asr_backend"],
            settings_components["asr_model"],
            settings_components["asr_vad_mode"],
            settings_components["asr_forced_alignment_enabled"],
        ]
        availability_outputs = [
            settings_components["asr_review_enabled"],
            settings_components["asr_review_models"],
            settings_components["asr_review_text_priority_model"],
            settings_components["asr_review_timestamp_priority_model"],
            settings_components["asr_vad_mode"],
            settings_components["asr_forced_alignment_enabled"],
        ]

        def model_availability_updates(*values: Any) -> tuple[Any, ...]:
            (
                current_review_enabled,
                current_review_models,
                current_text_priority,
                current_timestamp_priority,
                current_backend,
                current_model,
                current_vad_mode,
                current_forced_alignment,
            ) = values
            current = load_user_settings()
            review_choices = available_asr_review_choices(current)
            review_values = {value for _, value in review_choices}
            selected = [
                str(value)
                for value in (
                    current_review_models
                    if isinstance(current_review_models, (list, tuple))
                    else []
                )
                if str(value) in review_values
            ]
            primary = f"{current_backend}|{current_model}"
            text_priority = str(current_text_priority or "")
            if text_priority not in review_values:
                text_priority = primary if primary in review_values else ""
            if not text_priority and review_choices:
                text_priority = review_choices[0][1]

            timestamp_choices = available_timestamp_review_choices(current)
            timestamp_values = {value for _, value in timestamp_choices}
            timestamp_priority = str(current_timestamp_priority or "")
            if timestamp_priority not in timestamp_values:
                timestamp_priority = primary if primary in timestamp_values else ""
            if not timestamp_priority and timestamp_choices:
                timestamp_priority = timestamp_choices[0][1]
            aligner_ready = any(
                value.startswith("qwen_forced_aligner|") for _, value in timestamp_choices
            )

            vad_choices = asr_vad_choices(str(current_backend or ""))
            vad_values = {value for _, value in vad_choices}
            vad_mode = str(current_vad_mode or "off")
            if vad_mode not in vad_values:
                vad_mode = "off"
            review_enabled = bool(current_review_enabled and selected)
            return (
                gr.update(interactive=bool(review_choices), value=review_enabled),
                gr.update(
                    choices=review_choices,
                    value=selected,
                    visible=review_enabled,
                ),
                gr.update(
                    choices=review_choices,
                    value=text_priority or None,
                    visible=review_enabled,
                ),
                gr.update(
                    choices=timestamp_choices,
                    value=timestamp_priority or None,
                    visible=review_enabled,
                ),
                gr.update(choices=vad_choices, value=vad_mode),
                gr.update(
                    visible=aligner_ready,
                    value=bool(current_forced_alignment and aligner_ready),
                ),
            )

        def refreshed_runtime_values(*availability_values: Any) -> tuple[Any, ...]:
            current = load_user_settings()
            return (
                backend_catalog_rows(current, kind="asr"),
                backend_catalog_rows(current, kind="tts"),
                available_backend_models_markdown("asr", current),
                available_backend_models_markdown("tts", current),
                *model_availability_updates(*availability_values),
            )

        def install_callback(
            backend_id: str,
            *availability_values: Any,
        ) -> Iterator[tuple[Any, ...]]:
            if backend_id not in _INSTALLABLE:
                yield (
                    f"该后端不支持应用内安装：{backend_id}",
                    *refreshed_runtime_values(*availability_values),
                )
                return
            for log, done, _success in _install_backend_log_events(
                backend_id, controller=controller
            ):
                runtime_values = (
                    refreshed_runtime_values(*availability_values)
                    if done
                    else tuple(gr.update() for _ in range(10))
                )
                yield log, *runtime_values

        install_outputs = [
            install_log,
            asr_catalog,
            tts_catalog,
            asr_available,
            tts_available,
            *availability_outputs,
        ]
        install_asr_button.click(
            install_callback,
            inputs=[install_asr_choice, *availability_inputs],
            outputs=install_outputs,
            api_name="install_asr_backend",
            **runtime_options,
        )
        install_tts_button.click(
            install_callback,
            inputs=[install_tts_choice, *availability_inputs],
            outputs=install_outputs,
            api_name="install_tts_backend",
            **runtime_options,
        )
        pause_download_button.click(
            controller.pause,
            outputs=[install_log],
            api_name=_PRIVATE_API,
            queue=False,
        )

        def import_packs_callback(
            *availability_values: Any,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            lines: list[str] = []

            def report_progress(message: str, current: int, total: int) -> None:
                progress((current, max(1, total)), desc=message)

            try:
                results = import_discovered_model_packs(
                    log=lines.append,
                    progress=report_progress,
                )
                lines.append(f"已处理 {len(results)} 个模型包。")
            except Exception as exc:
                lines.append(f"导入失败：{_safe_error(exc)}")
            return (
                "\n".join(lines),
                offline_model_pack_markdown(),
                *refreshed_runtime_values(*availability_values),
            )

        scan_packs_button.click(
            import_packs_callback,
            inputs=availability_inputs,
            outputs=[
                install_log,
                model_pack_status,
                asr_catalog,
                tts_catalog,
                asr_available,
                tts_available,
                *availability_outputs,
            ],
            api_name="import_local_model_packs",
            **runtime_options,
        )

        def refresh_hardware_callback(*availability_values: Any) -> tuple[Any, ...]:
            profile = refresh_hardware()
            return (
                hardware_markdown(profile),
                recommended_stack_markdown(profile),
                *refreshed_runtime_values(*availability_values),
            )

        refresh_hardware_button.click(
            refresh_hardware_callback,
            inputs=availability_inputs,
            outputs=[
                hardware,
                recommendation,
                asr_catalog,
                tts_catalog,
                asr_available,
                tts_available,
                *availability_outputs,
            ],
            api_name=_PRIVATE_API,
            **runtime_options,
        )

    return app


def launch(host: str = "127.0.0.1", port: int = 7860) -> None:
    configure_logging()
    logger.info("启动 WebUI：host=%s port=%s", host, port)
    require_supported_platform()
    warnings.filterwarnings(
        "ignore",
        message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*",
        category=UserWarning,
        module=r"gradio\.routes",
    )
    app = build_app()
    stage = ui_stage_directory().resolve()
    auth = _remote_auth(host)
    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "share": False,
        "allowed_paths": [str(stage)],
        "max_file_size": os.getenv("ASMR_DUBBER_MAX_UPLOAD_SIZE", "20gb"),
        "footer_links": [],
        "show_error": False,
        "enable_monitoring": False,
        "strict_cors": True,
        "auth": auth,
        "auth_message": "ASMR Dubber 远程访问需要登录。" if auth else None,
        "css": APP_CSS,
        "theme": __import__("gradio").themes.Soft(
            primary_hue="violet",
            neutral_hue="slate",
            font=["Segoe UI", "Microsoft YaHei UI", "sans-serif"],
        ),
    }
    # Keep compatibility with a future Gradio release that may rename optional
    # launch parameters while retaining the security-critical allowlist/auth.
    accepted = inspect.signature(app.launch).parameters
    launch_kwargs = {key: value for key, value in launch_kwargs.items() if key in accepted}
    app.queue(default_concurrency_limit=1, max_size=8).launch(**launch_kwargs)


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
