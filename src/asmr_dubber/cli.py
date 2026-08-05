from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import uuid
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .app_logging import configure_logging
from .asr import transcribe_source
from .audio import make_analysis_copy
from .constants import ASMR_VAD_MODEL, DEFAULT_ALIGNER_MODEL
from .environment import cached_model_path, cuda_summary, ffmpeg_version
from .errors import AsmrDubberError
from .languages import SpeechSourceLanguage
from .model_pack_download import ModelPackDownloadError, prepare_remote_model_pack
from .model_packs import (
    ModelPackError,
    discover_model_packs,
    import_discovered_model_packs,
    import_model_pack,
)
from .model_registry import ASR_BACKENDS, TTS_BACKENDS
from .models import ProjectSettings, save_project
from .pipeline import (
    analyze_project,
    create_project,
    default_projects_dir,
    export_transcript,
    generate_subtitles,
    mix_project,
    reload_project,
    synthesize_project,
    translate_project,
)
from .platforms import current_platform, portable_home, require_supported_platform
from .runtime_manager import (
    backend_status,
    detect_hardware,
    download_backend_models,
    install_backend,
)
from .subtitles import SubtitleLanguage
from .user_settings import PROVIDER_PRESETS, load_user_settings, resolve_api_key


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


@app.command("create")
def create_command(
    input_media: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    projects_root: Annotated[Path | None, typer.Option("--projects-root")] = None,
    offset_ms: Annotated[int | None, typer.Option("--offset-ms")] = None,
    max_speed: Annotated[float | None, typer.Option("--max-speed")] = None,
    source_language: Annotated[str, typer.Option("--source-language")] = "ja",
) -> None:
    """从音频或视频建立项目并保存原始文件副本。"""
    try:
        if source_language not in {"ja", "en"}:
            raise ValueError("--source-language 只能是 ja 或 en。")
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
            source_language=cast(SpeechSourceLanguage, source_language),
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(directory / "project.json")


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
) -> None:
    """使用项目记录的翻译服务；API Key 从本机设置或环境变量读取。"""
    try:
        project, directory = reload_project(project_path)
        translate_project(project, directory, force=force, progress=ConsoleProgress())
    except AsmrDubberError as exc:
        _fail(exc)
    console.print(directory / "exports" / "transcript.json")


@app.command("synthesize")
def synthesize_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    force: Annotated[bool, typer.Option("--force", help="忽略中文音频缓存")] = False,
    sentence: Annotated[
        list[str] | None, typer.Option("--sentence", "-s", help="只重做指定句子 id，可重复")
    ] = None,
) -> None:
    """使用项目设置中选定的 TTS（语音合成）后端克隆全部中文。"""
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
        _fail(exc)
    console.print(directory / "chinese")


@app.command("mix")
def mix_command(project_path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """将中文轨与原轨相加；视频项目同时保留画面并输出视频。"""
    try:
        project, directory = reload_project(project_path)
        output = mix_project(project, directory, progress=ConsoleProgress())
    except AsmrDubberError as exc:
        _fail(exc)
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
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(f"[bold green]{srt}[/bold green]")
    console.print(f"[bold green]{lrc}[/bold green]")
    if video is not None:
        console.print(f"[bold green]{video}[/bold green]")


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
) -> None:
    """修改中文配音偏移和冲突加速上限；重混即可。"""
    try:
        if offset_ms is None and max_speed is None:
            raise ValueError("请至少提供 --offset-ms 或 --max-speed。")
        project, directory = reload_project(project_path)
        settings = project.settings.model_dump()
        if offset_ms is not None:
            settings["chinese_dubbing_offset_ms"] = offset_ms
        if max_speed is not None:
            settings["chinese_max_auto_speed"] = max_speed
        project.settings = ProjectSettings.model_validate(settings)
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_video_file = None
        save_project(project, directory)
        export_transcript(project, directory)
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(
        "中文配音排程已更新："
        f"整体偏移 {project.settings.chinese_dubbing_offset_ms:+d} ms，"
        f"最大自动加速 {project.settings.chinese_max_auto_speed:g}×。"
    )


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
        ("gradio", "gradio"),
    ):
        try:
            version = importlib.metadata.version(distribution)
            importlib.import_module(module)
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


@app.command("download-models")
def download_models_command(
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help=(
                "Kotoba-Whisper v2.2、Faster-Whisper large-v2、Qwen3 ForcedAligner "
                "和 ASMR VAD，或 all；advanced-asr 为旧别名"
            ),
        ),
    ] = "all",
) -> None:
    """下载并校验进阶档的识别、VAD 和时间戳模型。"""

    if backend not in {"进阶语音识别", "advanced-asr", "all"}:
        _fail(ValueError("--backend 必须是“进阶语音识别”或 all。"))
    for backend_id in ("kotoba_whisper", "faster_whisper"):
        console.print(f"[cyan]准备 {ASR_BACKENDS[backend_id].label}[/cyan]")
        try:
            detail = download_backend_models(
                backend_id,
                log_callback=lambda message: console.print(f"[cyan]{message}[/cyan]"),
            )
        except AsmrDubberError as exc:
            _fail(exc)
        console.print(f"[green]{detail}[/green]")
    for pack_id, model_id, label in (
        ("qwen3-forced-aligner", DEFAULT_ALIGNER_MODEL, "Qwen3 ForcedAligner 0.6B"),
        ("whisper-vad-asmr-onnx", ASMR_VAD_MODEL, "日语 ASMR 专用 Whisper VAD"),
    ):
        if cached_model_path(model_id) is not None:
            console.print(f"[green]复用已安装模型：{label}[/green]")
            continue
        console.print(f"[cyan]准备 {label}[/cyan]")
        try:
            archive = prepare_remote_model_pack(
                pack_id,
                log=lambda message: console.print(f"[cyan]{message}[/cyan]"),
            )
            if archive is None:
                raise ModelPackDownloadError(f"没有为 {pack_id} 配置 ModelScope 模型包。")
            import_model_pack(archive, progress=ConsoleProgress())
        except (ModelPackDownloadError, ModelPackError, OSError, ValueError) as exc:
            _fail(exc)
        if cached_model_path(model_id) is None:
            _fail(ModelPackError(f"{label} 导入后未通过完整性检查。"))
        console.print(f"[green]{label} 已通过完整性检查。[/green]")


@app.command("list-model-packs")
def list_model_packs_command() -> None:
    """列出项目根目录 model-packs 中发现的离线模型包。"""
    inspections = discover_model_packs()
    table = Table(title="ASMR Dubber 离线模型包")
    table.add_column("文件")
    table.add_column("模型包")
    table.add_column("版本")
    table.add_column("解压大小")
    table.add_column("状态")
    for inspection in inspections:
        manifest = inspection.manifest
        if manifest is None:
            table.add_row(
                inspection.archive.name,
                "—",
                "—",
                "—",
                f"无效：{inspection.error}",
            )
            continue
        table.add_row(
            inspection.archive.name,
            manifest.display_name,
            manifest.pack_version,
            f"{manifest.uncompressed_bytes / 1024**3:.2f} GiB",
            "可导入" if inspection.compatible else f"不兼容：{inspection.error}",
        )
    if not inspections:
        table.add_row("—", "—", "—", "—", "未发现 ZIP 模型包")
    console.print(table)


@app.command("prepare-model-pack")
def prepare_model_pack_command(
    pack_id: Annotated[str, typer.Argument(help="要下载到 model-packs 的模型包 ID")],
) -> None:
    """从 mirrors.json 配置的远程来源下载并校验模型包。"""
    try:
        archive = prepare_remote_model_pack(
            pack_id,
            log=lambda message: console.print(f"[cyan]{message}[/cyan]"),
        )
    except (ModelPackDownloadError, ModelPackError, OSError, ValueError) as exc:
        _fail(exc)
    if archive is None:
        console.print(f"没有为 {pack_id} 配置远程模型包；继续使用原始下载源。")
        return
    console.print(f"[green]模型包已就绪：[/green]{archive}")


@app.command("import-model-packs")
def import_model_packs_command(
    archives: Annotated[
        list[Path] | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="要导入的模型包 ZIP；可指定多个。",
        ),
    ] = None,
    all_packs: Annotated[
        bool,
        typer.Option("--all", help="导入项目根目录 model-packs 中所有兼容模型包。"),
    ] = False,
    pack_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--pack-id",
            help="只导入指定 pack_id；可重复使用，供安装档位精确选择。",
        ),
    ] = None,
) -> None:
    """校验并原子导入离线模型包。"""
    if not all_packs and not archives:
        _fail(ValueError("请指定模型包 ZIP，或添加 --all。"))
    try:
        if all_packs:
            results = import_discovered_model_packs(
                pack_ids=set(pack_ids or ()) or None,
                log=lambda message: console.print(f"[cyan]{message}[/cyan]"),
                progress=ConsoleProgress(),
            )
        else:
            results = [
                import_model_pack(
                    archive,
                    log=lambda message: console.print(f"[cyan]{message}[/cyan]"),
                    progress=ConsoleProgress(),
                )
                for archive in archives or []
            ]
    except (ModelPackError, OSError) as exc:
        _fail(exc)
    if not results:
        # Setup calls ``--all`` for every profile. An empty inbox is a normal
        # online-install path, while malformed archives are rejected by the
        # importer above and still produce a non-zero exit code.
        console.print("未发现可导入的本地模型包；继续按所选档位准备。")
        return
    installed = sum(result.installed_files for result in results)
    reused = sum(result.reused_files for result in results)
    console.print(
        f"[green]完成：处理 {len(results)} 个模型包，"
        f"新增/更新 {installed} 个文件，复用 {reused} 个文件。[/green]"
    )


@app.command("install-backend")
def install_backend_command(
    backend_id: Annotated[str, typer.Argument(help="设置页显示的后端 ID")],
) -> None:
    """安装或修复一个后端，并预下载该后端的固定推荐模型。"""
    try:
        result = install_backend(backend_id)
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(f"[green]{result}[/green]")


@app.command("download-backend-models", hidden=True)
def download_backend_models_command(
    backend_id: Annotated[str, typer.Argument(help="后端 ID")],
) -> None:
    """Download the pinned models for one backend."""
    try:
        result = download_backend_models(
            backend_id,
            log_callback=lambda message: console.print(f"[cyan]{message}[/cyan]"),
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(f"[green]{result}[/green]")


@app.command("verify-asr")
def verify_asr_command(
    audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    backend: Annotated[str, typer.Option("--backend")],
    model: Annotated[str, typer.Option("--model")],
    device: Annotated[str, typer.Option("--device")] = "cuda",
    compute_type: Annotated[str, typer.Option("--compute-type")] = "float16",
    decoder: Annotated[str, typer.Option("--decoder")] = "tdt",
    vad_mode: Annotated[str, typer.Option("--vad-mode")] = "off",
    chunk_seconds: Annotated[float, typer.Option("--chunk-seconds")] = 120.0,
    kotoba_chunk_seconds: Annotated[
        float,
        typer.Option("--kotoba-chunk-seconds"),
    ] = 30.0,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 1,
    source_language: Annotated[str, typer.Option("--source-language")] = "ja",
) -> None:
    """用短音频真实加载一个 ASR（语音识别）后端，不翻译或写入项目。"""
    token = uuid.uuid4().hex
    temporary = portable_home() / "temp" / f"verify-asr-{token}.wav"
    try:
        if source_language not in {"ja", "en"}:
            raise ValueError("--source-language 只能是 ja 或 en。")
        analysis = make_analysis_copy(audio.resolve(), temporary)
        settings = ProjectSettings.model_validate(
            {
                "asr_backend": backend,
                "asr_model": model,
                "asr_device": device,
                "asr_compute_type": compute_type,
                "asr_parakeet_decoder": decoder,
                "asr_vad_mode": vad_mode,
                "asr_chunk_seconds": chunk_seconds,
                "asr_kotoba_chunk_seconds": kotoba_chunk_seconds,
                "asr_batch_size": batch_size,
            }
        )
        sentences, language = transcribe_source(
            analysis,
            settings,
            source_language=cast(SpeechSourceLanguage, source_language),
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    finally:
        temporary.unlink(missing_ok=True)
    console.print_json(
        data={
            "language": language,
            "sentences": [
                {
                    "start": sentence.start_seconds,
                    "end": sentence.end_seconds,
                    "text": sentence.source_text,
                }
                for sentence in sentences
            ],
        },
        ensure_ascii=False,
    )


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
