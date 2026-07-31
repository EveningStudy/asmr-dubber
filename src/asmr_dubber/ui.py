from __future__ import annotations

import inspect
import ipaddress
import logging
import os
import queue
import secrets
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from .app_logging import application_log_path, configure_logging, recent_log_text
from .constants import (
    DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
    DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY,
    DEFAULT_CHINESE_RELATIVE_LOUDNESS_DB,
    INDEXTTS_REQUIRED_DIRS,
    INDEXTTS_REQUIRED_FILES,
    MAX_CHINESE_AUTO_SPEED,
)
from .errors import InstallPausedError, OperationCancelledError
from .model_packs import discover_model_packs, import_discovered_model_packs, model_pack_directory
from .model_registry import ASR_BACKENDS, CLONE_MODE_LABELS, TTS_BACKENDS
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
from .translation import SYSTEM_PROMPT
from .ui_services import (
    TABLE_HEADERS,
    TABLE_TYPES,
    ProjectView,
    analyze,
    apply_global_settings,
    create_project,
    import_transcript_data,
    load_view,
    preview_reference,
    recent_projects,
    reference_picker,
    save_table,
    select_reference,
    stage_for_ui,
    subtitles,
    synthesize_and_mix,
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
.workflow-hint { border-left: 4px solid var(--color-accent); padding-left: .85rem; }
.status-panel textarea, .diagnostics-panel textarea { font-family: var(--font); }
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
    {"deepseek", "openai", "anthropic", "gemini", "openai_compatible"}
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


def asr_vad_choices(backend_id: str) -> list[tuple[str, str]]:
    """Return only VAD modes that can run for the selected backend right now."""

    choices = [("不做 VAD 预处理", "off")]
    native = _NATIVE_VAD_LABELS.get(str(backend_id))
    if native:
        choices.append((native, "backend"))
    if asmr_vad_status().state == "ready":
        choices.append(("日语 ASMR 专用 Whisper VAD（独立预处理）", "asmr"))
    return choices


def _review_control_state(
    settings: UserSettings,
) -> tuple[
    list[tuple[str, str]],
    list[str],
    str | None,
    list[tuple[str, str]],
    str | None,
]:
    review_choices = available_asr_review_choices(settings)
    review_values = {value for _, value in review_choices}
    selected = [value for value in settings.asr_review_models if value in review_values]
    primary = f"{settings.asr_backend}|{settings.asr_model}"
    text_priority = settings.asr_review_text_priority_model
    if text_priority not in review_values:
        text_priority = primary if primary in review_values else None
    if text_priority is None and review_choices:
        text_priority = review_choices[0][1]

    timestamp_choices = available_timestamp_review_choices(settings)
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

    def __init__(self) -> None:
        self.cancel_event = CancellationToken()
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

    def cancel(self) -> str:
        with self._lock:
            if self._active is None:
                return "当前没有正在执行的项目任务。"
            label = self._active
        self.cancel_event.set()
        return f"正在取消“{label}”… 已经完成并保存的内容会保留。"


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
        view.rows,
        view.output_audio,
        view.output_video,
        view.subtitle_files,
        view.subtitle_video,
        view.diagnostics,
        view.status,
        _gr_update(choices=choices, value=selected),
        preview,
    )


def _empty_project_updates(message: str) -> tuple[Any, ...]:
    return (*(_gr_update() for _ in range(7)), message, _gr_update(), _gr_update())


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
    displayed_prompt = str(current.get("translation_prompt", "")).replace("\r\n", "\n").strip()
    if displayed_prompt == SYSTEM_PROMPT.replace("\r\n", "\n"):
        # Show the built-in prompt in the form without freezing a duplicate in
        # settings.json. Future built-in improvements still take effect when
        # the displayed text has not been edited.
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


def _translation_prompt_for_display(value: Any) -> str:
    return str(value or "").strip() or SYSTEM_PROMPT


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
    ("Qwen3 ForcedAligner 自动对齐（仅日语台本，需要进阶组件）", "qwen"),
]


def _transcript_language_update(language: Any) -> Any:
    if str(language or "ja") == "zh":
        return _gr_update(
            choices=[_TRANSCRIPT_TIMING_CHOICES[0]],
            value="estimate",
        )
    return _gr_update(choices=_TRANSCRIPT_TIMING_CHOICES, value="estimate")


def _asr_backend_update(backend: Any, current_vad_mode: Any = "off") -> tuple[Any, ...]:
    backend_id = str(backend or "")
    spec = ASR_BACKENDS.get(backend_id, ASR_BACKENDS["parakeet_nemo"])
    vad_choices = asr_vad_choices(backend_id)
    vad_values = {value for _, value in vad_choices}
    vad_mode = str(current_vad_mode or "off")
    if vad_mode not in vad_values:
        vad_mode = "off"
    return (
        _gr_update(choices=list(spec.models), value=spec.default_model),
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


_TTS_DEFAULT_URLS = {
    "gpt_sovits": "http://127.0.0.1:9880",
    "cosyvoice": "http://127.0.0.1:50000",
    "fish_speech": "http://127.0.0.1:8080",
}


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
        _gr_update(visible=backend_id != "indextts2"),
        _gr_update(visible=backend_id == "gpt_sovits"),
        _gr_update(visible=backend_id == "cosyvoice"),
        _gr_update(visible=backend_id == "indextts2"),
        _gr_update(visible=backend_id != "indextts2"),
        _gr_update(visible=backend_id == "gpt_sovits"),
    )


def _tts_service_visibility(backend: Any) -> tuple[Any, ...]:
    backend_id = str(backend or "")
    spec = TTS_BACKENDS.get(backend_id, TTS_BACKENDS["indextts2"])
    external = backend_id != "indextts2"
    return (
        _gr_update(visible=external),
        _gr_update(visible=spec.api_key),
        _gr_update(visible=spec.api_key),
        _gr_update(visible=spec.api_key),
    )


def _tts_detail_visibility(
    backend: Any,
    reference_source: Any,
    index_speaker_source: Any,
    index_emotion_source: Any,
    cosyvoice_mode: Any,
) -> tuple[Any, ...]:
    """Hide reference controls that the active TTS mode cannot consume."""

    backend_id = str(backend or "")
    is_index = backend_id == "indextts2"
    external_speaker = (
        str(index_speaker_source or "") == "external"
        if is_index
        else str(reference_source or "") == "external"
    )
    external_reference_text = (
        not is_index
        and str(reference_source or "") == "external"
        and not (backend_id == "cosyvoice" and str(cosyvoice_mode or "") == "cross_lingual")
    )
    return (
        _gr_update(visible=external_speaker),
        _gr_update(visible=external_reference_text),
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
    controller = DownloadController()
    task_controller = ProjectTaskController()
    asr_spec = ASR_BACKENDS[stored.asr_backend]
    tts_spec = TTS_BACKENDS[stored.tts_backend]
    provider = PROVIDER_PRESETS.get(stored.translation_provider, PROVIDER_PRESETS["deepseek"])
    initial_loudness_mode = _loudness_mode(
        stored.normalize_chinese_loudness,
        stored.match_source_loudness,
    )
    initial_vad_choices = asr_vad_choices(stored.asr_backend)
    initial_vad_values = {value for _, value in initial_vad_choices}
    initial_vad_mode = stored.asr_vad_mode
    if initial_vad_mode not in initial_vad_values:
        initial_vad_mode = "off"
    (
        initial_review_choices,
        initial_review_models,
        initial_review_text_priority,
        initial_timestamp_choices,
        initial_review_timestamp_priority,
    ) = _review_control_state(stored)
    initial_aligner_ready = any(
        value.startswith("qwen_forced_aligner|") for _, value in initial_timestamp_choices
    )
    initial_review_enabled = stored.asr_review_enabled and bool(initial_review_models)
    recent = recent_projects(stored.projects_root or None)
    initial_recent = recent[0][1] if recent else None

    with gr.Blocks(title="ASMR Dubber 0.4") as app:
        gr.HTML(
            "<header><h1>ASMR Dubber</h1>"
            "<p>把日语 ASMR 转成逐句同音色中文配音 · 便携版 0.4</p></header>",
            elem_id="asmr-dubber-product-marker",
        )

        with gr.Tabs():
            with gr.Tab("项目工作台", id="project-workspace"):
                gr.Markdown(
                    "**推荐顺序：** 新建/打开项目 → ASR（语音识别）→ 翻译并校对 → "
                    "TTS（语音合成）与混音。每一步都会保存项目，可随时关闭后继续。",
                    elem_classes=["workflow-hint"],
                )
                with gr.Row(elem_classes=["mobile-stack"]):
                    source_input = gr.File(
                        label="日语音频或视频",
                        file_types=["audio", "video"],
                        type="filepath",
                        scale=2,
                    )
                    with gr.Column(scale=1):
                        create_button = gr.Button("新建项目", variant="primary")
                        refresh_projects_button = gr.Button("刷新最近项目")
                with gr.Row():
                    recent_project = gr.Dropdown(
                        label="最近项目",
                        choices=recent,
                        value=initial_recent,
                        allow_custom_value=True,
                        info="也可以粘贴 project.json 的完整路径。",
                        scale=3,
                    )
                    open_project_button = gr.Button("打开项目", scale=1)
                project_path = gr.Textbox(
                    label="当前项目文件",
                    interactive=False,
                )

                with gr.Accordion("已有台本或字幕（可跳过 ASR）", open=False):
                    gr.Markdown(
                        "有时间戳的 SRT、VTT、ASS/SSA、LRC 会直接建立句子时间轴。"
                        "TXT 或粘贴文字按每个非空行作为一句，可按台词长度估算，"
                        "日语纯台本也可用进阶组件 Qwen3 ForcedAligner 自动对齐。"
                        "中文台本会直接写入中文配音列，跳过 ASR（语音识别）和翻译。"
                        "导入会替换当前句子表，但不会修改原音频。"
                    )
                    transcript_language = gr.Radio(
                        label="台本语言",
                        choices=[
                            ("日语台本（跳过 ASR，之后翻译）", "ja"),
                            ("中文配音文本（跳过 ASR 和翻译）", "zh"),
                        ],
                        value="ja",
                    )
                    transcript_file = gr.File(
                        label="台本或字幕文件",
                        file_types=[".srt", ".vtt", ".ass", ".ssa", ".lrc", ".txt"],
                        type="filepath",
                    )
                    transcript_text = gr.Textbox(
                        label="也可以直接粘贴纯台本",
                        placeholder="每个非空行作为一句；选择文件时，粘贴内容优先。",
                        lines=6,
                    )
                    plain_timing = gr.Radio(
                        label="纯文本台本如何生成时间轴",
                        choices=_TRANSCRIPT_TIMING_CHOICES,
                        value="estimate",
                    )
                    import_transcript_button = gr.Button(
                        "导入并建立句子时间轴",
                        variant="primary",
                    )

                gr.Markdown("## 处理步骤")
                with gr.Row(equal_height=True):
                    asr_button = gr.Button("1 · 运行 ASR（语音识别）", variant="primary")
                    translate_button = gr.Button("2 · 翻译日文")
                    save_table_button = gr.Button("3 · 保存校对表格")
                    synthesize_button = gr.Button("4 · TTS（语音合成）并混音", variant="primary")
                cancel_task_button = gr.Button("取消当前执行", variant="stop")

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
                    "可编辑启用状态、时间、日文和中文；清空一行的日文和中文即可删除"
                    "该句。状态与错误不会混入业务数据。"
                )

                with gr.Accordion("统一音色参考", open=False):
                    gr.Markdown(
                        "推荐选择 5–15 秒、清晰且包含完整台词的一句。只影响项目内统一音色模式。"
                    )
                    with gr.Row():
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

                with gr.Row():
                    subtitle_language = gr.Radio(
                        label="字幕内容",
                        choices=[
                            ("日中双语", "bilingual"),
                            ("仅中文", "zh"),
                            ("仅日文", "ja"),
                        ],
                        value="bilingual",
                    )
                    subtitle_button = gr.Button("生成字幕")

                with gr.Row():
                    output_audio = gr.Audio(
                        label="完成音频",
                        type="filepath",
                        interactive=False,
                    )
                    output_video = gr.Video(label="完成视频", interactive=False)
                with gr.Row():
                    subtitle_files = gr.File(
                        label="字幕文件（SRT / LRC）",
                        file_count="multiple",
                        interactive=False,
                    )
                    subtitle_video = gr.Video(label="带字幕视频", interactive=False)
                diagnostics = gr.Textbox(
                    label="项目诊断",
                    lines=6,
                    interactive=False,
                    elem_classes=["diagnostics-panel"],
                )
                status = gr.Textbox(
                    label="状态与错误",
                    value="请选择源文件新建项目，或打开最近项目。",
                    lines=4,
                    interactive=False,
                    elem_classes=["status-panel"],
                )

            with gr.Tab("设置", id="settings"):
                gr.Markdown(
                    "这里可以编辑**以后新项目的默认值**，也可以把同一份表单保存并应用到"
                    "当前项目。打开旧项目不会自动覆盖其设置。密钥以明文保存在程序目录的"
                    " `.asmr-dubber/config/secrets.json`；删除整个程序文件夹时会一并删除。"
                )
                settings_status = gr.Textbox(
                    label="设置状态",
                    value="尚未修改。",
                    interactive=False,
                    lines=3,
                )
                settings_components: dict[str, Any] = {}

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
                        settings_components["asr_backend"] = gr.Dropdown(
                            label="ASR（语音识别）后端",
                            choices=[(spec.label, key) for key, spec in ASR_BACKENDS.items()],
                            value=stored.asr_backend,
                        )
                        settings_components["asr_model"] = gr.Dropdown(
                            label="ASR（语音识别）模型",
                            choices=list(asr_spec.models),
                            value=stored.asr_model,
                            allow_custom_value=True,
                        )
                        asr_help = gr.Markdown(f"{asr_spec.help}\n\n{asr_spec.setup}")
                        with gr.Row():
                            settings_components["asr_device"] = gr.Dropdown(
                                label="识别设备",
                                choices=[("NVIDIA CUDA", "cuda"), ("CPU", "cpu")],
                                value=stored.asr_device,
                            )
                            settings_components["asr_compute_type"] = gr.Dropdown(
                                label="计算精度",
                                choices=["float16", "float32", "int8_float16", "int8"],
                                value=stored.asr_compute_type,
                                allow_custom_value=True,
                                visible=stored.asr_backend == "faster_whisper",
                            )
                            settings_components["asr_batch_size"] = gr.Number(
                                label="批大小", value=stored.asr_batch_size, precision=0
                            )
                            settings_components["asr_beam_size"] = gr.Number(
                                label="束搜索宽度（Beam Size）",
                                value=stored.asr_beam_size,
                                precision=0,
                                visible=stored.asr_backend == "faster_whisper",
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
                                value=stored.asr_timeout_seconds,
                                visible=stored.asr_backend == "parakeet_nemo",
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
                                value=stored.asr_vad_min_silence_ms,
                                precision=0,
                                visible=initial_vad_mode == "backend",
                            )
                            settings_components["asr_asmr_vad_threshold"] = gr.Number(
                                label="ASMR VAD 语音阈值",
                                value=stored.asr_asmr_vad_threshold,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_asmr_vad_min_speech_ms"] = gr.Number(
                                label="ASMR VAD 最短语音毫秒",
                                value=stored.asr_asmr_vad_min_speech_ms,
                                precision=0,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_asmr_vad_min_silence_ms"] = gr.Number(
                                label="ASMR VAD 最短静音毫秒",
                                value=stored.asr_asmr_vad_min_silence_ms,
                                precision=0,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_asmr_vad_speech_pad_ms"] = gr.Number(
                                label="ASMR VAD 边界保留毫秒",
                                value=stored.asr_asmr_vad_speech_pad_ms,
                                precision=0,
                                visible=initial_vad_mode == "asmr",
                            )
                            settings_components["asr_condition_on_previous_text"] = gr.Checkbox(
                                label="使用上一段文字作为识别条件",
                                value=stored.asr_condition_on_previous_text,
                                visible=stored.asr_backend in {"kotoba_whisper", "faster_whisper"},
                            )
                            settings_components["asr_initial_prompt"] = gr.Textbox(
                                label="日文识别提示词",
                                value=stored.asr_initial_prompt,
                                lines=3,
                                visible=stored.asr_backend in {"parakeet_nemo", "faster_whisper"},
                            )
                            settings_components["asr_parakeet_decoder"] = gr.Radio(
                                label="Parakeet 解码头",
                                choices=[("TDT", "tdt"), ("CTC", "ctc")],
                                value=stored.asr_parakeet_decoder,
                                visible=(
                                    stored.asr_backend == "parakeet_nemo"
                                    and stored.asr_model == "nvidia/parakeet-tdt_ctc-0.6b-ja"
                                ),
                            )
                            settings_components["asr_chunk_seconds"] = gr.Number(
                                label="Parakeet 分块秒数（15–600）",
                                value=stored.asr_chunk_seconds,
                                visible=stored.asr_backend == "parakeet_nemo",
                            )
                            settings_components["asr_kotoba_chunk_seconds"] = gr.Number(
                                label="Kotoba-Whisper 分块秒数（5–120）",
                                value=stored.asr_kotoba_chunk_seconds,
                                visible=(
                                    stored.asr_backend == "kotoba_whisper"
                                    or (
                                        stored.asr_backend == "faster_whisper"
                                        and stored.asr_model
                                        == "kotoba-tech/kotoba-whisper-v2.0-faster"
                                    )
                                ),
                            )
                        settings_components["asr_forced_alignment_enabled"] = gr.Checkbox(
                            label=("识别后使用 Qwen3 ForcedAligner 0.6B（阿里）重新计算时间戳"),
                            value=stored.asr_forced_alignment_enabled,
                            visible=initial_aligner_ready,
                            info="识别文字不变；该模型只重新寻找每句话的起止时间。",
                        )
                        with gr.Accordion("多模型交叉校对（进阶）", open=False):
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
                                value=stored.asr_review_max_drift_seconds,
                                visible=initial_review_enabled,
                            )
                            settings_components["asr_review_background"] = gr.Textbox(
                                label="作品、人物与场景背景",
                                value=stored.asr_review_background,
                                lines=4,
                                visible=initial_review_enabled,
                            )
                            settings_components["asr_review_prompt"] = gr.Textbox(
                                label="ASR（语音识别）校对提示词（Prompt）",
                                value=stored.asr_review_prompt,
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
                                label="自定义翻译 Prompt（留空使用内置）",
                                value=_translation_prompt_for_display(stored.translation_prompt),
                                lines=12,
                                info=(
                                    "当前直接显示实际使用的 Prompt。保持原样保存时仍跟随内置版本；"
                                    "修改后保存为自定义 Prompt。"
                                ),
                            )
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
                        with (
                            gr.Group(
                                visible=stored.tts_backend == "gpt_sovits"
                            ) as tts_sampling_group,
                            gr.Row(),
                        ):
                            settings_components["tts_speed"] = gr.Number(
                                label="语速", value=stored.tts_speed
                            )
                            settings_components["tts_temperature"] = gr.Number(
                                label="随机度（Temperature）",
                                value=stored.tts_temperature,
                            )
                            settings_components["tts_top_p"] = gr.Number(
                                label="核采样概率（Top P）", value=stored.tts_top_p
                            )

                        saved_speaker = stored.tts_external_reference_audio or "无"
                        external_speaker_visible = (
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
                            visible=stored.tts_backend != "indextts2"
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
                                label="外部参考音频对应日文",
                                value=stored.tts_external_reference_text,
                                lines=3,
                                visible=(
                                    stored.tts_reference_source == "external"
                                    and not (
                                        stored.tts_backend == "cosyvoice"
                                        and stored.tts_cosyvoice_mode == "cross_lingual"
                                    )
                                ),
                            )

                        settings_components["tts_api_base_url"] = gr.Textbox(
                            label="TTS（语音合成）API（接口）基础地址",
                            value=stored.tts_api_base_url,
                            visible=stored.tts_backend != "indextts2",
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
                        settings_components["mix_peak_protection"] = gr.Checkbox(
                            label="最终混音峰值保护", value=stored.mix_peak_protection
                        )
                        settings_components["mix_peak_limit_dbfs"] = gr.Number(
                            label="最终峰值上限（dBFS）", value=stored.mix_peak_limit_dbfs
                        )
                        settings_components["retain_chinese_stem"] = gr.Checkbox(
                            label="保留中文中间轨", value=stored.retain_chinese_stem
                        )
                        settings_components["skip_japanese_fillers"] = gr.Checkbox(
                            label="跳过纯日语语气词", value=stored.skip_japanese_fillers
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
                            choices=[("原日语时间", "source"), ("中文配音时间", "dubbing")],
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
            sentence_table,
            output_audio,
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

        def create_callback(source: Any, progress: gr.Progress = gr.Progress()) -> tuple[Any, ...]:
            return run_project_task(
                "新建项目",
                create_project,
                source,
                _StageProgress(progress),
            )

        def open_callback(path: Any) -> tuple[Any, ...]:
            if not str(path or "").strip():
                return _empty_project_updates("请先选择最近项目或粘贴 project.json 路径。")
            return _run_project_action(load_view, str(path))

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
            language: str,
            progress: gr.Progress = gr.Progress(),
        ) -> tuple[Any, ...]:
            return run_project_task(
                "导入台本/字幕",
                import_transcript_data,
                manifest,
                transcript,
                text,
                timing,
                language,
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
                "TTS（语音合成）与混音",
                synthesize_and_mix,
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

        def refresh_projects_callback() -> Any:
            current = load_user_settings()
            choices = recent_projects(current.projects_root or None)
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def pick_reference_callback(manifest: str, sentence_id: str) -> tuple[Any, Any]:
            try:
                return select_reference(manifest, sentence_id)
            except Exception as exc:
                return f"保存参考句失败：{_safe_error(exc)}", gr.update()

        create_button.click(
            create_callback,
            inputs=[source_input],
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
                transcript_language,
            ],
            outputs=common_outputs,
            api_name="import_transcript",
            **runtime_options,
        )
        transcript_language.change(
            _transcript_language_update,
            inputs=[transcript_language],
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
            api_name="synthesize_and_mix",
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
        refresh_log_button.click(
            refresh_log_callback,
            outputs=[log_text, log_file],
            api_name=_PRIVATE_API,
            queue=False,
        )

        settings_components["asr_backend"].change(
            _asr_backend_update,
            inputs=[
                settings_components["asr_backend"],
                settings_components["asr_vad_mode"],
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
            settings_components["tts_reference_source"],
            settings_components["tts_index_speaker_source"],
            settings_components["tts_index_emotion_source"],
            settings_components["tts_cosyvoice_mode"],
        ]
        tts_detail_outputs = [
            external_speaker_group,
            settings_components["tts_external_reference_text"],
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
        for detail_component in (
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
                    "当前已打开的项目没有改变；需要时请点击“保存并应用到当前项目”。"
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
                return message, *_empty_project_updates("默认设置已保存；当前没有打开项目。")

            try:
                project_view = apply_global_settings(normalized_manifest, settings)
            except Exception as exc:
                project_message = f"新项目默认值已经保存，但应用到当前项目失败：{_safe_error(exc)}"
                settings_message = f"新项目默认值已保存：{path}\n{project_message}"
                return settings_message, *_empty_project_updates(project_message)

            settings_message = f"新项目默认值已保存：{path}\n{project_view.status}"
            return settings_message, *_view_values(project_view)

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
            outputs=[settings_status, *common_outputs],
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
