from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import logging
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .asr import transcribe_japanese
from .audio import make_analysis_copy
from .constants import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ASR_MODEL,
    DEFAULT_TTS_MODEL,
    MODEL_LFS_SHA256,
    MODEL_REVISIONS,
    OPTIONAL_ASR_MODEL_REVISIONS,
)
from .environment import cached_model_path, cuda_summary, ffmpeg_version
from .errors import AsmrDubberError
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
from .user_settings import PROVIDER_PRESETS, load_user_settings, resolve_api_key

load_dotenv()
app = typer.Typer(
    name="asmr-dubber",
    no_args_is_help=True,
    add_completion=False,
    help="日语 ASMR → 逐句同音色中文复述（Windows / Linux）",
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


def _fail(exc: Exception) -> None:
    console.print(f"[bold red]错误：[/bold red]{exc}")
    raise typer.Exit(code=1) from exc


@app.command("create")
def create_command(
    input_audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    projects_root: Annotated[Path | None, typer.Option("--projects-root")] = None,
    overlap: Annotated[float | None, typer.Option("--overlap")] = None,
) -> None:
    """建立项目并保存原始音频副本。"""
    try:
        settings = load_user_settings().to_project_settings()
        if overlap is not None:
            values = settings.model_dump()
            values["global_overlap_seconds"] = overlap
            settings = ProjectSettings.model_validate(values)
        _, directory = create_project(
            input_audio,
            projects_root,
            settings=settings,
        )
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(directory / "project.json")


@app.command("analyze")
def analyze_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    force: Annotated[bool, typer.Option("--force", help="丢弃原识别结果并重跑")] = False,
) -> None:
    """使用项目设置中选定的 ASR 后端识别、对齐和切句。"""
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
    """使用项目设置中选定的 TTS 后端克隆全部中文。"""
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
    """将中文轨与原轨相加并输出浏览器兼容的 24-bit PCM WAV。"""
    try:
        project, directory = reload_project(project_path)
        output = mix_project(project, directory, progress=ConsoleProgress())
    except AsmrDubberError as exc:
        _fail(exc)
    console.print(f"[bold green]{output}[/bold green]")


@app.command("set-timing")
def set_timing_command(
    project_path: Annotated[Path, typer.Argument(exists=True)],
    overlap: Annotated[
        float | None,
        typer.Option(
            "--overlap",
            help="最长提前秒数；正数=句末前提前，0=句末，负数=句末后等待",
        ),
    ] = None,
    percentage: Annotated[
        float | None,
        typer.Option(
            "--percentage",
            help="正数提前量最多占当前日语句时长的百分比（0–100）",
        ),
    ] = None,
) -> None:
    """修改全局中文开始位置；重混即可，无需重做识别/翻译/配音。"""
    try:
        if overlap is None and percentage is None:
            raise ValueError("请至少提供 --overlap 或 --percentage。")
        project, directory = reload_project(project_path)
        settings = project.settings.model_dump()
        if overlap is not None:
            settings["global_overlap_seconds"] = overlap
        if percentage is not None:
            settings["global_overlap_percentage"] = percentage
        project.settings = ProjectSettings.model_validate(settings)
        project.chinese_stem_file = None
        project.output_file = None
        save_project(project, directory)
        export_transcript(project, directory)
    except (AsmrDubberError, ValueError) as exc:
        _fail(exc)
    console.print(
        "中文提前设置已更新："
        f"{project.settings.global_overlap_seconds:.3f} 秒，"
        f"句长的 {project.settings.global_overlap_percentage:g}%。"
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
        ("ASR", user_settings.asr_backend, ASR_BACKENDS),
        ("TTS", user_settings.tts_backend, TTS_BACKENDS),
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
            help="qwen3_asr、voxcpm2、advanced-asr 或 all",
        ),
    ] = "all",
) -> None:
    """下载并校验固定版本的内置模型。"""
    from .mirrors import snapshot_download_with_fallback

    advanced_asr_models = (
        "kotoba-tech/kotoba-whisper-v2.2",
        "Systran/faster-whisper-large-v2",
    )
    backend_models = {
        "qwen3_asr": (DEFAULT_ASR_MODEL, DEFAULT_ALIGNER_MODEL),
        "voxcpm2": (DEFAULT_TTS_MODEL,),
        "advanced-asr": advanced_asr_models,
        "all": tuple(MODEL_REVISIONS),
    }
    model_ids = backend_models.get(backend)
    if model_ids is None:
        _fail(ValueError("--backend 必须是 qwen3_asr、voxcpm2、advanced-asr 或 all。"))
    if backend == "advanced-asr":
        try:
            imported = import_discovered_model_packs(
                pack_ids={"kotoba-whisper-v2.2", "faster-whisper-large-v2"},
                log=lambda message: console.print(f"[cyan]{message}[/cyan]"),
                progress=ConsoleProgress(),
            )
        except ModelPackError as exc:
            _fail(exc)
        if imported:
            console.print(f"[green]已处理 {len(imported)} 个本地离线模型包。[/green]")
    for index, model_id in enumerate(model_ids, start=1):
        cached = cached_model_path(model_id)
        revision = MODEL_REVISIONS.get(model_id) or OPTIONAL_ASR_MODEL_REVISIONS.get(model_id)
        if revision is None:
            _fail(AsmrDubberError(f"模型没有固定版本：{model_id}"))
        if cached is not None:
            path = cached
            console.print(f"[cyan][{index}/{len(model_ids)}] 复用本地模型 {model_id}[/cyan]")
        else:
            console.print(f"[cyan][{index}/{len(model_ids)}] 下载 {model_id}[/cyan]")
            try:
                path = snapshot_download_with_fallback(
                    repo_id=model_id,
                    revision=revision,
                    max_workers=2,
                )
            except Exception as exc:
                _fail(AsmrDubberError(f"模型下载失败 {model_id}：{exc}"))
            cached = cached_model_path(model_id)
        if cached is None:
            _fail(AsmrDubberError(f"模型快照不完整：{model_id}"))
        known_hashes = MODEL_LFS_SHA256.get(model_id, {})
        if known_hashes:
            console.print(f"[cyan]校验大权重 SHA-256：{model_id}[/cyan]")
        for relative, expected in known_hashes.items():
            digest = hashlib.sha256()
            with (cached / relative).open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != expected:
                _fail(AsmrDubberError(f"模型权重校验失败：{model_id}/{relative}"))
        console.print(f"[green]完成：[/green]{path}")


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
) -> None:
    """用短音频真实加载一个 ASR 后端，不执行翻译或写入项目。"""
    token = uuid.uuid4().hex
    temporary = portable_home() / "temp" / f"verify-asr-{token}.wav"
    try:
        analysis = make_analysis_copy(audio.resolve(), temporary)
        settings = ProjectSettings(
            asr_backend=backend,
            asr_model=model,
            asr_device=device,
            asr_compute_type=compute_type,
            asr_parakeet_decoder=decoder,
            asr_batch_size=1,
        )
        sentences, language = transcribe_japanese(analysis, settings)
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
                    "text": sentence.ja_text,
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
