from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .app_logging import configure_logging
from .asr import transcribe_source as transcribe_source
from .audio import make_analysis_copy as make_analysis_copy
from .cli_admin import (
    _parse_setting_value as _parse_setting_value,
)
from .cli_admin import (
    batch_command as batch_command,
)
from .cli_admin import (
    download_backend_models_command as download_backend_models_command,
)
from .cli_admin import (
    download_models_command as download_models_command,
)
from .cli_admin import (
    import_model_packs_command as import_model_packs_command,
)
from .cli_admin import (
    install_backend_command as install_backend_command,
)
from .cli_admin import (
    list_model_packs_command as list_model_packs_command,
)
from .cli_admin import (
    prepare_model_pack_command as prepare_model_pack_command,
)
from .cli_admin import (
    register_admin_commands,
)
from .cli_admin import (
    settings_app as settings_app,
)
from .cli_admin import (
    settings_clear_translation_key_command as settings_clear_translation_key_command,
)
from .cli_admin import (
    settings_clear_tts_key_command as settings_clear_tts_key_command,
)
from .cli_admin import (
    settings_set_command as settings_set_command,
)
from .cli_admin import (
    settings_set_translation_key_command as settings_set_translation_key_command,
)
from .cli_admin import (
    settings_set_tts_key_command as settings_set_tts_key_command,
)
from .cli_admin import (
    settings_show_command as settings_show_command,
)
from .cli_admin import (
    verify_asr_command as verify_asr_command,
)
from .constants import (
    ASMR_VAD_MODEL as ASMR_VAD_MODEL,
)
from .constants import (
    DEFAULT_ALIGNER_MODEL as DEFAULT_ALIGNER_MODEL,
)
from .environment import (
    cached_model_path as cached_model_path,
)
from .environment import (
    cuda_summary,
    ffmpeg_version,
)
from .errors import AsmrDubberError
from .languages import SpeechSourceLanguage
from .model_pack_download import (
    ModelPackDownloadError as ModelPackDownloadError,
)
from .model_pack_download import (
    prepare_remote_model_pack as prepare_remote_model_pack,
)
from .model_packs import (
    ModelPackError as ModelPackError,
)
from .model_packs import (
    discover_model_packs as discover_model_packs,
)
from .model_packs import (
    import_discovered_model_packs as import_discovered_model_packs,
)
from .model_packs import (
    import_model_pack as import_model_pack,
)
from .model_registry import ASR_BACKENDS, TTS_BACKENDS
from .models import ProjectSettings, save_project
from .pipeline import (
    analyze_project,
    create_project,
    default_projects_dir,
    export_transcript,
    generate_subtitles,
    import_project_transcript,
    mix_project,
    reload_project,
    synthesize_project,
    translate_project,
)
from .platforms import (
    current_platform,
    require_supported_platform,
)
from .platforms import (
    portable_home as portable_home,
)
from .runtime_manager import (
    backend_status,
    detect_hardware,
)
from .runtime_manager import (
    download_backend_models as download_backend_models,
)
from .runtime_manager import (
    install_backend as install_backend,
)
from .subtitles import SubtitleLanguage
from .user_settings import (
    PROVIDER_PRESETS,
    load_user_settings,
    resolve_api_key,
)
from .user_settings import (
    UserSettings as UserSettings,
)
from .user_settings import (
    clear_api_key as clear_api_key,
)
from .user_settings import (
    clear_service_key as clear_service_key,
)
from .user_settings import (
    save_api_key as save_api_key,
)
from .user_settings import (
    save_service_key as save_service_key,
)
from .user_settings import (
    save_user_settings as save_user_settings,
)


def _make_windows_stdio_loss_tolerant() -> None:
    """Do not crash when a legacy Windows code page cannot encode a path."""

    if not current_platform().is_windows:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")


_make_windows_stdio_loss_tolerant()
load_dotenv()
configure_logging()
app = typer.Typer(
    name="asmr-dubber",
    no_args_is_help=True,
    add_completion=False,
    help="日语/英语音声 → 逐句同音色中文复述（Windows / Linux）",
)
console = Console()


class ConsoleProgress:
    def __init__(self) -> None:
        self.last = ""

    def __call__(self, message: str, current: int, total: int) -> None:
        label = f"[{current}/{total}] {message}" if total else message
        if label != self.last:
            console.print(f"[cyan]{label}[/cyan]")
            self.last = label


def _fail(exc: Exception) -> NoReturn:
    console.print(f"[bold red]错误：[/bold red]{exc}")
    raise typer.Exit(code=1) from exc


ResultJsonOption = Annotated[
    Path | None,
    typer.Option("--result-json", help="把阶段结果和失败句写入 JSON，供批量引擎读取"),
]


def _stage_failed(
    stage: str, project_path: Path, result_json: Path | None, exc: Exception
) -> NoReturn:
    from .stage_result import write_result

    write_result(result_json, stage, project_path, exc)
    _fail(exc)


@app.command("create")
def create_command(
    input_media: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    projects_root: Annotated[Path | None, typer.Option("--projects-root")] = None,
    offset_ms: Annotated[int | None, typer.Option("--offset-ms")] = None,
    max_speed: Annotated[float | None, typer.Option("--max-speed")] = None,
    source_language: Annotated[str, typer.Option("--source-language")] = "ja",
    project_name: Annotated[str | None, typer.Option("--project-name")] = None,
) -> None:
    """从音频或视频建立项目并保存原始文件副本。"""
    try:
        if source_language not in {"ja", "en", "zh"}:
            raise ValueError("--source-language 只能是 ja、en 或 zh。")
        settings = load_user_settings().to_project_settings()
        if offset_ms is not None or max_speed is not None:
            values = settings.model_dump()
            if offset_ms is not None:
                values["chinese_dubbing_offset_ms"] = offset_ms
            if max_speed is not None:
                values["chinese_max_auto_speed"] = max_speed
            settings = ProjectSettings.model_validate(values)
        _, directory = create_project(
            input_media,
            projects_root,
            settings=settings,
            project_name=project_name,
            source_language=cast(SpeechSourceLanguage, source_language),
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(directory / "project.json")


@app.command("import-transcript")
def import_transcript_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    transcript: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    kind: Annotated[str, typer.Option("--kind", help="source=原文；zh=中文配音稿")] = "source",
    timing: Annotated[
        str,
        typer.Option(
            "--timing",
            help="无时间轴纯文本的处理方式：estimate、qwen 或 script-review",
        ),
    ] = "estimate",
) -> None:
    """导入 SRT/VTT/ASS/LRC 或纯文本台本，替换当前句子表。"""

    try:
        if kind not in {"source", "zh"}:
            raise ValueError("--kind 只能是 source 或 zh。")
        timing_value = timing.replace("-", "_")
        if timing_value not in {"estimate", "qwen", "script_review"}:
            raise ValueError("--timing 只能是 estimate、qwen 或 script-review。")
        project, directory = reload_project(project_path)
        language = "zh" if kind == "zh" else project.source_language
        result = import_project_transcript(
            project,
            directory,
            transcript_path=transcript,
            plain_timing=timing_value,
            script_language=language,
            progress=ConsoleProgress(),
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print_json(data=result, ensure_ascii=False)
    console.print(f"[green]{directory / 'exports' / 'transcript.json'}[/green]")


@app.command("analyze")
def analyze_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    force: Annotated[bool, typer.Option("--force", help="丢弃原识别结果并重跑")] = False,
) -> None:
    """使用项目设置中选定的 ASR（语音识别）后端识别、对齐和切句。"""
    try:
        project, directory = reload_project(project_path)
        analyze_project(project, directory, force=force, progress=ConsoleProgress())
    except AsmrDubberError as exc:
        _fail(exc)
    console.print(directory / "exports" / "transcript.json")


@app.command("translate")
def translate_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    force: Annotated[bool, typer.Option("--force", help="重新翻译所有句子")] = False,
    result_json: ResultJsonOption = None,
) -> None:
    """使用项目记录的翻译服务；API Key 从本机设置或环境变量读取。"""
    from .stage_result import write_result

    try:
        project, directory = reload_project(project_path)
        translate_project(project, directory, force=force, progress=ConsoleProgress())
    except AsmrDubberError as exc:
        _stage_failed("translate", project_path, result_json, exc)
    except Exception as exc:
        write_result(result_json, "translate", project_path, exc)
        raise
    write_result(result_json, "translate", project_path)
    console.print(directory / "exports" / "transcript.json")


@app.command("synthesize")
def synthesize_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    force: Annotated[bool, typer.Option("--force", help="忽略中文音频缓存")] = False,
    sentence: Annotated[
        list[str] | None, typer.Option("--sentence", "-s", help="只重做指定句子 id，可重复")
    ] = None,
    result_json: ResultJsonOption = None,
) -> None:
    """使用项目设置中选定的 TTS（语音合成）后端克隆全部中文。"""
    from .stage_result import write_result

    try:
        project, directory = reload_project(project_path)
        synthesize_project(
            project,
            directory,
            force=force,
            sentence_ids=sentence,
            progress=ConsoleProgress(),
        )
    except AsmrDubberError as exc:
        _stage_failed("synthesize", project_path, result_json, exc)
    except Exception as exc:
        write_result(result_json, "synthesize", project_path, exc)
        raise
    write_result(result_json, "synthesize", project_path)
    console.print(directory / "chinese")


@app.command("mix")
def mix_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    output_variant: Annotated[str | None, typer.Option("--output-variant")] = None,
    allow_missing: Annotated[
        bool,
        typer.Option("--allow-missing", help="缺配音的句子按不配音处理（保留原声），不中断混音"),
    ] = False,
    result_json: ResultJsonOption = None,
) -> None:
    """将中文轨与原轨相加；视频项目同时保留画面并输出视频。"""
    from .stage_result import write_result

    try:
        project, directory = reload_project(project_path)
        output = mix_project(
            project,
            directory,
            progress=ConsoleProgress(),
            output_variant=output_variant,
            allow_missing=allow_missing,
        )
    except (AsmrDubberError, ValueError) as exc:
        _stage_failed("mix", project_path, result_json, exc)
    except Exception as exc:
        write_result(result_json, "mix", project_path, exc)
        raise
    write_result(result_json, "mix", project_path)
    console.print(f"[bold green]{output}[/bold green]")
    if project.output_video_file:
        console.print(f"[bold green]{directory / project.output_video_file}[/bold green]")
    if project.chinese_stem_file:
        console.print(f"[bold green]{directory / project.chinese_stem_file}[/bold green]")


@app.command("subtitles")
def subtitles_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="bilingual、zh 或 source（ja 为兼容别名）"),
    ] = "bilingual",
    output_variant: Annotated[str | None, typer.Option("--output-variant")] = None,
) -> None:
    """独立生成 SRT/LRC；视频项目同时生成带字幕视频。"""
    try:
        if language == "ja":
            language = "source"
        if language not in {"bilingual", "zh", "source"}:
            raise ValueError("--language 只能是 bilingual、zh 或 source。")
        project, directory = reload_project(project_path)
        srt, lrc, video = generate_subtitles(
            project,
            directory,
            language=cast(SubtitleLanguage, language),
            progress=ConsoleProgress(),
            output_variant=output_variant,
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(f"[bold green]{srt}[/bold green]")
    console.print(f"[bold green]{lrc}[/bold green]")
    if video is not None:
        console.print(f"[bold green]{video}[/bold green]")


@app.command("run")
def run_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    start: Annotated[
        str,
        typer.Option("--start", help="从 analyze、translate、synthesize、mix 或 subtitles 开始"),
    ] = "analyze",
    stop: Annotated[
        str,
        typer.Option("--stop", help="在 analyze、translate、synthesize、mix 或 subtitles 结束"),
    ] = "subtitles",
    force: Annotated[bool, typer.Option("--force", help="重做范围内可缓存的阶段")] = False,
    subtitle_language: Annotated[
        str,
        typer.Option("--subtitle-language", help="bilingual、zh 或 source"),
    ] = "bilingual",
) -> None:
    """按顺序运行一个项目的多个阶段，适合无界面服务器任务。"""

    stages = ("analyze", "translate", "synthesize", "mix", "subtitles")
    try:
        if start not in stages or stop not in stages:
            raise ValueError(
                "--start/--stop 必须是 analyze、translate、synthesize、mix 或 subtitles。"
            )
        first = stages.index(start)
        last = stages.index(stop)
        if first > last:
            raise ValueError("--start 不能位于 --stop 之后。")
        if subtitle_language not in {"bilingual", "zh", "source"}:
            raise ValueError("--subtitle-language 只能是 bilingual、zh 或 source。")
        project, directory = reload_project(project_path)
        selected = stages[first : last + 1]
        for stage in selected:
            if stage == "analyze":
                if project.sentences and not force:
                    console.print("[dim]项目已有句子，保留现有时间轴。[/dim]")
                else:
                    analyze_project(project, directory, force=force, progress=ConsoleProgress())
            elif stage == "translate":
                translate_project(project, directory, force=force, progress=ConsoleProgress())
            elif stage == "synthesize":
                synthesize_project(project, directory, force=force, progress=ConsoleProgress())
            elif stage == "mix":
                mix_project(project, directory, progress=ConsoleProgress())
            else:
                generate_subtitles(
                    project,
                    directory,
                    language=cast(SubtitleLanguage, subtitle_language),
                    progress=ConsoleProgress(),
                )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(f"[bold green]流程完成：{directory / 'project.json'}[/bold green]")


@app.command("set-timing")
def set_timing_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    offset_ms: Annotated[
        int | None,
        typer.Option(
            "--offset-ms",
            help="中文相对原字幕开始时间的整体偏移；负数提前，正数延后",
        ),
    ] = None,
    max_speed: Annotated[
        float | None,
        typer.Option(
            "--max-speed",
            help="与下一句冲突时允许的最大自动加速倍速（1.0–4.0）",
        ),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option(
            "--mode",
            help="fit-window=冲突时自动加速；sequential=上一句结束后再播放下一句",
        ),
    ] = None,
) -> None:
    """修改配音排程；请重混，并重新生成跟随配音时间轴的字幕。"""
    try:
        if offset_ms is None and max_speed is None and mode is None:
            raise ValueError("请至少提供 --offset-ms、--max-speed 或 --mode。")
        if mode is not None and mode not in {"fit-window", "fit_window", "sequential"}:
            raise ValueError("--mode 只能是 fit-window 或 sequential。")
        project, directory = reload_project(project_path)
        settings = project.settings.model_dump()
        if offset_ms is not None:
            settings["chinese_dubbing_offset_ms"] = offset_ms
        if max_speed is not None:
            settings["chinese_max_auto_speed"] = max_speed
        if mode is not None:
            settings["chinese_dubbing_timing_mode"] = (
                "fit_window" if mode in {"fit-window", "fit_window"} else "sequential"
            )
        project.settings = ProjectSettings.model_validate(settings)
        from .lifecycle import invalidate_outputs

        invalidate_outputs(project, subtitles=project.settings.subtitle_timeline == "dubbing")
        save_project(project, directory)
        export_transcript(project, directory)
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(
        "中文配音排程已更新："
        f"整体偏移 {project.settings.chinese_dubbing_offset_ms:+d} ms，"
        + (
            f"最大自动加速 {project.settings.chinese_max_auto_speed:g}×。"
            if project.settings.chinese_dubbing_timing_mode == "fit_window"
            else "上一句结束后再播放下一句，不自动加速。"
        )
    )
    if project.settings.subtitle_timeline == "dubbing":
        console.print("旧配音时间轴字幕已失效；请运行 subtitles 重新生成字幕。")


@app.command("doctor")
def doctor_command(
    no_network: Annotated[bool, typer.Option("--no-network", help="跳过翻译服务网络测试")] = False,
) -> None:
    """检查操作系统、设备、FFmpeg、核心依赖和当前选择的后端。"""
    healthy = True
    table = Table(title="ASMR Dubber 环境检查")
    table.add_column("项目")
    table.add_column("状态")
    try:
        require_supported_platform()
        platform_info = current_platform()
        table.add_row(
            "操作系统",
            f"OK · {platform_info.label} · {platform_info.architecture}",
        )
    except AsmrDubberError as exc:
        table.add_row("操作系统", f"失败：{exc}")
        healthy = False
    hardware = detect_hardware()
    cuda = cuda_summary()
    if cuda["available"]:
        gib = int(cuda["memory_bytes"] or 0) / (1024**3)
        table.add_row(
            "CUDA",
            f"OK · {cuda['device']} · {gib:.1f} GiB · "
            f"sm_{str(cuda['capability']).replace('.', '')}",
        )
        table.add_row("PyTorch", f"{cuda['torch']} · CUDA runtime {cuda['cuda_runtime']}")
    else:
        if hardware.gpu:
            vram = f"{hardware.vram_gb:.1f} GiB" if hardware.vram_gb is not None else "未知显存"
            table.add_row("NVIDIA GPU", f"{hardware.gpu} · {vram} · 驱动 {hardware.driver}")
            table.add_row("PyTorch CUDA", "当前核心环境未安装或未启用；GPU 硬件已检测到")
        else:
            table.add_row("GPU", "未检测到 NVIDIA GPU（CPU 和外部 API 后端仍可使用）")
    try:
        table.add_row("FFmpeg", ffmpeg_version())
    except Exception as exc:
        table.add_row("FFmpeg", f"失败：{exc}")
        healthy = False
    for distribution, module in (
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("httpx", "httpx"),
        ("edge-tts", "edge_tts"),
    ):
        try:
            version = importlib.metadata.version(distribution)
            importlib.import_module(module)
            if distribution == "edge-tts":
                from .edge_tts_client import require_compatible_client

                require_compatible_client()
        except importlib.metadata.PackageNotFoundError:
            version = "未安装"
            healthy = False
        except Exception as exc:
            version = f"导入失败：{exc}"
            healthy = False
        table.add_row(distribution, version)
    user_settings = load_user_settings()
    for kind, backend_id, registry in (
        ("ASR（语音识别）", user_settings.asr_backend, ASR_BACKENDS),
        ("TTS（语音合成）", user_settings.tts_backend, TTS_BACKENDS),
    ):
        spec = registry.get(backend_id)
        if spec is None:
            table.add_row(f"当前 {kind}", f"失败：未知后端 {backend_id}")
            healthy = False
            continue
        status = backend_status(spec, settings=user_settings)
        detail = f" · {status.detail}" if status.detail else ""
        table.add_row(f"当前 {kind} · {spec.label}", f"{status.label}{detail}")
        if status.state in {"missing", "broken", "incompatible"}:
            healthy = False
    provider = user_settings.translation_provider
    provider_label = str(PROVIDER_PRESETS.get(provider, {}).get("label", provider))
    try:
        resolve_api_key(provider)
        key_state = "已设置或无需密钥"
    except AsmrDubberError:
        key_state = "未设置（请前往 UI 设置页）"
    table.add_row("翻译服务", provider_label)
    table.add_row("翻译密钥", key_state)
    if no_network:
        table.add_row("翻译服务网络", "已跳过")
    else:
        try:
            import httpx

            base_url = user_settings.translation_base_url.rstrip("/")
            httpx_logger = logging.getLogger("httpx")
            previous_level = httpx_logger.level
            try:
                httpx_logger.setLevel(logging.WARNING)
                with httpx.Client(timeout=20.0) as client:
                    response = client.get(base_url)
            finally:
                httpx_logger.setLevel(previous_level)
            if response.status_code not in {200, 401, 403}:
                raise RuntimeError(f"HTTP {response.status_code}")
            table.add_row("翻译服务网络", f"OK · HTTP {response.status_code}")
        except Exception as exc:
            table.add_row("翻译服务网络", f"失败：{exc}")
            healthy = False
    table.add_row("项目目录", str(default_projects_dir()))
    console.print(table)
    if not healthy:
        raise typer.Exit(code=1)


register_admin_commands(app)


@app.command("ui")
def ui_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 7860,
) -> None:
    """启动本地浏览器界面。"""
    from .ui import launch

    launch(host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
