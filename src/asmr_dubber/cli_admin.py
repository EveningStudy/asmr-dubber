from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.table import Table

from .constants import ASMR_VAD_MODEL, DEFAULT_ALIGNER_MODEL
from .errors import AsmrDubberError
from .languages import SpeechSourceLanguage
from .model_pack_download import ModelPackDownloadError
from .model_packs import ModelPackError
from .model_registry import ASR_BACKENDS, TTS_BACKENDS
from .models import ProjectSettings
from .user_settings import UserSettings

settings_app = typer.Typer(
    name="settings",
    no_args_is_help=True,
    help="读取或修改便携设置和 API 密钥。",
)


def _parse_setting_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@settings_app.command("show")
def settings_show_command() -> None:
    """以 JSON 显示当前全局设置；不会显示密钥。"""
    from . import cli

    cli.console.print_json(
        data=cli.load_user_settings().model_dump(mode="json"), ensure_ascii=False
    )


@settings_app.command("set")
def settings_set_command(
    field: Annotated[str, typer.Argument(help="settings show 中的字段名")],
    value: Annotated[str, typer.Argument(help="新值；布尔值和数字可直接填写 JSON 字面量")],
) -> None:
    """修改一个全局设置，并用与网页相同的模型完整校验。"""
    from . import cli

    try:
        current = cli.load_user_settings()
        if field not in UserSettings.model_fields:
            raise ValueError(f"未知设置项：{field}")
        payload = current.model_dump()
        payload[field] = _parse_setting_value(value)
        settings = UserSettings.model_validate(payload)
        destination = cli.save_user_settings(settings)
    except (AsmrDubberError, ValueError) as exc:
        cli._fail(exc)
    cli.console.print(f"[green]已保存 {field}：[/green]{destination}")


@settings_app.command("set-translation-key")
def settings_set_translation_key_command(
    provider: Annotated[str, typer.Argument(help="deepseek、bailian、doubao 等服务 ID")],
    key: Annotated[
        str | None,
        typer.Option("--key", help="省略时安全提示输入；自动化部署也可使用服务环境变量"),
    ] = None,
) -> None:
    """保存翻译服务密钥到程序目录。"""
    from . import cli

    try:
        secret = key if key is not None else typer.prompt("API Key", hide_input=True)
        destination = cli.save_api_key(provider, secret)
    except (AsmrDubberError, ValueError) as exc:
        cli._fail(exc)
    cli.console.print(f"[green]翻译密钥已保存：[/green]{destination}")


@settings_app.command("clear-translation-key")
def settings_clear_translation_key_command(
    provider: Annotated[str, typer.Argument(help="要清除的翻译服务 ID")],
) -> None:
    """清除一个已保存的翻译服务密钥。"""
    from . import cli

    try:
        cli.clear_api_key(provider)
    except AsmrDubberError as exc:
        cli._fail(exc)
    cli.console.print("[green]翻译密钥已清除。[/green]")


@settings_app.command("set-tts-key")
def settings_set_tts_key_command(
    backend: Annotated[str, typer.Argument(help="mimo_tts、minimax、gpt_sovits 等后端 ID")],
    key: Annotated[
        str | None,
        typer.Option("--key", help="省略时安全提示输入"),
    ] = None,
) -> None:
    """保存外部 TTS 服务密钥到程序目录。"""
    from . import cli

    try:
        if backend not in TTS_BACKENDS:
            raise ValueError(f"未知 TTS 后端：{backend}")
        secret = key if key is not None else typer.prompt("API Key", hide_input=True)
        destination = cli.save_service_key(f"tts:{backend}", secret)
    except (AsmrDubberError, ValueError) as exc:
        cli._fail(exc)
    cli.console.print(f"[green]TTS 密钥已保存：[/green]{destination}")


@settings_app.command("clear-tts-key")
def settings_clear_tts_key_command(
    backend: Annotated[str, typer.Argument(help="要清除的 TTS 后端 ID")],
) -> None:
    """清除一个已保存的外部 TTS 密钥。"""
    from . import cli

    cli.clear_service_key(f"tts:{backend}")
    cli.console.print("[green]TTS 密钥已清除。[/green]")


def batch_command(
    folder: Annotated[
        Path | None, typer.Argument(help="DLsite 作品文件夹；省略则进入交互队列")
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="audio、video-normal 或 video-harmonized"),
    ] = None,
    layout: Annotated[
        str | None,
        typer.Option("--layout", help="merged、separate 或 both"),
    ] = None,
    edition: Annotated[str | None, typer.Option("--edition")] = None,
    include_bonus: Annotated[bool, typer.Option("--include-bonus")] = False,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    background: Annotated[str | None, typer.Option("--background")] = None,
    embed_subtitles: Annotated[
        str | None,
        typer.Option("--embed-subtitles", help="yes 或 no"),
    ] = None,
    rebuild: Annotated[bool, typer.Option("--rebuild")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    self_test: Annotated[bool, typer.Option("--self-test")] = False,
) -> None:
    """运行集成的 AutoFlow 批量音频/静态视频工作流。"""
    from .autoflow.engine import main as autoflow_main

    arguments: list[str] = []
    if folder is not None:
        arguments.append(str(folder))
    for option, value in (
        ("--mode", mode),
        ("--layout", layout),
        ("--edition", edition),
        ("--output-root", str(output_root) if output_root is not None else None),
        ("--background", background),
        ("--embed-subtitles", embed_subtitles),
    ):
        if value is not None:
            arguments.extend((option, value))
    if include_bonus:
        arguments.append("--include-bonus")
    if rebuild:
        arguments.append("--rebuild")
    if force:
        arguments.append("--force")
    if self_test:
        arguments.append("--self-test")
    raise typer.Exit(code=autoflow_main(arguments))


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
    from . import cli

    if backend not in {"进阶语音识别", "advanced-asr", "all"}:
        cli._fail(ValueError("--backend 必须是“进阶语音识别”或 all。"))
    for backend_id in ("kotoba_whisper", "faster_whisper"):
        cli.console.print(f"[cyan]准备 {ASR_BACKENDS[backend_id].label}[/cyan]")
        try:
            detail = cli.download_backend_models(
                backend_id,
                log_callback=lambda message: cli.console.print(f"[cyan]{message}[/cyan]"),
            )
        except AsmrDubberError as exc:
            cli._fail(exc)
        cli.console.print(f"[green]{detail}[/green]")
    for pack_id, model_id, label in (
        ("qwen3-forced-aligner", DEFAULT_ALIGNER_MODEL, "Qwen3 ForcedAligner 0.6B"),
        ("whisper-vad-asmr-onnx", ASMR_VAD_MODEL, "日语 ASMR 专用 Whisper VAD"),
    ):
        if cli.cached_model_path(model_id) is not None:
            cli.console.print(f"[green]复用已安装模型：{label}[/green]")
            continue
        cli.console.print(f"[cyan]准备 {label}[/cyan]")
        try:
            archive = cli.prepare_remote_model_pack(
                pack_id,
                log=lambda message: cli.console.print(f"[cyan]{message}[/cyan]"),
            )
            if archive is None:
                raise ModelPackDownloadError(f"没有为 {pack_id} 配置 ModelScope 模型包。")
            cli.import_model_pack(archive, progress=cli.ConsoleProgress())
        except (ModelPackDownloadError, ModelPackError, OSError, ValueError) as exc:
            cli._fail(exc)
        if cli.cached_model_path(model_id) is None:
            cli._fail(ModelPackError(f"{label} 导入后未通过完整性检查。"))
        cli.console.print(f"[green]{label} 已通过完整性检查。[/green]")


def list_model_packs_command() -> None:
    """列出项目根目录 model-packs 中发现的离线模型包。"""
    from . import cli

    inspections = cli.discover_model_packs()
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
    cli.console.print(table)


def prepare_model_pack_command(
    pack_id: Annotated[str, typer.Argument(help="要下载到 model-packs 的模型包 ID")],
) -> None:
    """从 mirrors.json 配置的远程来源下载并校验模型包。"""
    from . import cli

    try:
        archive = cli.prepare_remote_model_pack(
            pack_id,
            log=lambda message: cli.console.print(f"[cyan]{message}[/cyan]"),
        )
    except (ModelPackDownloadError, ModelPackError, OSError, ValueError) as exc:
        cli._fail(exc)
    if archive is None:
        cli.console.print(f"没有为 {pack_id} 配置远程模型包；继续使用原始下载源。")
        return
    cli.console.print(f"[green]模型包已就绪：[/green]{archive}")


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
    from . import cli

    if not all_packs and not archives:
        cli._fail(ValueError("请指定模型包 ZIP，或添加 --all。"))
    try:
        if all_packs:
            results = cli.import_discovered_model_packs(
                pack_ids=set(pack_ids or ()) or None,
                log=lambda message: cli.console.print(f"[cyan]{message}[/cyan]"),
                progress=cli.ConsoleProgress(),
            )
        else:
            results = [
                cli.import_model_pack(
                    archive,
                    log=lambda message: cli.console.print(f"[cyan]{message}[/cyan]"),
                    progress=cli.ConsoleProgress(),
                )
                for archive in archives or []
            ]
    except (ModelPackError, OSError) as exc:
        cli._fail(exc)
    if not results:
        # Setup calls ``--all`` for every profile. An empty inbox is a normal
        # online-install path, while malformed archives are rejected by the
        # importer above and still produce a non-zero exit code.
        cli.console.print("未发现可导入的本地模型包；继续按所选档位准备。")
        return
    installed = sum(result.installed_files for result in results)
    reused = sum(result.reused_files for result in results)
    cli.console.print(
        f"[green]完成：处理 {len(results)} 个模型包，"
        f"新增/更新 {installed} 个文件，复用 {reused} 个文件。[/green]"
    )


def install_backend_command(
    backend_id: Annotated[str, typer.Argument(help="设置页显示的后端 ID")],
) -> None:
    """安装或修复一个后端，并预下载该后端的固定推荐模型。"""
    from . import cli

    try:
        result = cli.install_backend(backend_id)
    except (AsmrDubberError, ValueError) as exc:
        cli._fail(exc)
    cli.console.print(f"[green]{result}[/green]")


def download_backend_models_command(
    backend_id: Annotated[str, typer.Argument(help="后端 ID")],
) -> None:
    """Download the pinned models for one backend."""
    from . import cli

    try:
        result = cli.download_backend_models(
            backend_id,
            log_callback=lambda message: cli.console.print(f"[cyan]{message}[/cyan]"),
        )
    except (AsmrDubberError, ValueError) as exc:
        cli._fail(exc)
    cli.console.print(f"[green]{result}[/green]")


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
    from . import cli

    token = uuid.uuid4().hex
    temporary = cli.portable_home() / "temp" / f"verify-asr-{token}.wav"
    try:
        if source_language not in {"ja", "en", "zh"}:
            raise ValueError("--source-language 只能是 ja、en 或 zh。")
        analysis = cli.make_analysis_copy(audio.resolve(), temporary)
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
        sentences, language = cli.transcribe_source(
            analysis,
            settings,
            source_language=cast(SpeechSourceLanguage, source_language),
        )
    except (AsmrDubberError, ValueError) as exc:
        cli._fail(exc)
    finally:
        temporary.unlink(missing_ok=True)
    cli.console.print_json(
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


def register_admin_commands(app: typer.Typer) -> None:
    app.add_typer(settings_app, name="settings")
    app.command("batch")(batch_command)
    app.command("download-models")(download_models_command)
    app.command("list-model-packs")(list_model_packs_command)
    app.command("prepare-model-pack")(prepare_model_pack_command)
    app.command("import-model-packs")(import_model_packs_command)
    app.command("install-backend")(install_backend_command)
    app.command("download-backend-models", hidden=True)(download_backend_models_command)
    app.command("verify-asr")(verify_asr_command)
