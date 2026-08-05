from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelKind = Literal["asr", "tts"]
RuntimeKind = Literal["builtin", "optional_python", "http", "command"]
InstallerKind = Literal["python-extra", "isolated"]
SupportLevel = Literal["verified", "supported", "experimental", "community"]
DeviceKind = Literal["cpu", "cuda", "rocm", "mps"]
BatchStrategy = Literal["none", "native_list", "internal_chunks", "request_concurrency"]
ProgressStrategy = Literal["none", "per_item", "streamed_process"]


SUPPORT_LEVEL_LABELS: dict[SupportLevel, str] = {
    "verified": "已验证",
    "supported": "支持",
    "experimental": "实验性",
    "community": "社区适配",
}

CLONE_MODE_LABELS: dict[str, str] = {
    "stable_reference": "统一声纹，仅音色参考（推荐）",
    "reference_only": "逐句参考",
}


@dataclass(frozen=True)
class ExecutionCapabilities:
    """Performance/lifecycle contract used by adapters and the settings UI."""

    batch_strategy: BatchStrategy = "none"
    quality_sensitive_batch: bool = False
    reusable_reference_conditioning: bool = False
    progress_strategy: ProgressStrategy = "per_item"
    persistent_session: bool = False


@dataclass(frozen=True)
class ModelBackend:
    id: str
    label: str
    kind: ModelKind
    runtime: RuntimeKind
    default_model: str
    models: tuple[str, ...]
    help: str
    setup: str
    reference_audio: bool = False
    reference_text: Literal["unused", "optional", "required"] = "unused"
    style_reference: bool = False
    clone_modes: tuple[str, ...] = ("stable_reference", "reference_only")
    api_key: bool = False
    tested_default: bool = False
    support_level: SupportLevel = "experimental"
    platforms: tuple[str, ...] = ("windows", "linux")
    devices: tuple[DeviceKind, ...] = ("cpu", "cuda")
    minimum_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    disk_gb: float | None = None
    installer: InstallerKind | None = None
    python_extra: str | None = None
    homepage: str = ""
    execution: ExecutionCapabilities = ExecutionCapabilities()

    @property
    def support_label(self) -> str:
        return SUPPORT_LEVEL_LABELS[self.support_level]

    @property
    def execution_label(self) -> str:
        batch_labels = {
            "none": "逐项",
            "native_list": "原生批处理",
            "internal_chunks": "内部切块批处理",
            "request_concurrency": "服务请求并发",
        }
        parts = [batch_labels[self.execution.batch_strategy]]
        if self.execution.quality_sensitive_batch:
            parts.append("增大批量可能产生细微结果差异")
        if self.execution.reusable_reference_conditioning:
            parts.append("可复用声纹条件")
        if self.execution.persistent_session:
            parts.append("支持常驻会话扩展")
        return "；".join(parts)


ASR_BACKENDS: dict[str, ModelBackend] = {
    "parakeet_nemo": ModelBackend(
        id="parakeet_nemo",
        label="Parakeet 日语 · CrispASR（推荐）",
        kind="asr",
        runtime="command",
        default_model="grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo",
        models=(
            "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo",
            "nvidia/parakeet-tdt_ctc-0.6b-ja",
        ),
        help="日语专用 FastConformer。默认 1.1B 质量优先，0.6B 适合低资源设备。",
        setup="在“设备与模型”页安装 CrispASR 和两款经过校验的 F16 模型。",
        tested_default=True,
        support_level="verified",
        devices=("cpu", "cuda"),
        recommended_vram_gb=6,
        disk_gb=5,
        installer="isolated",
        homepage="https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja",
        execution=ExecutionCapabilities(
            batch_strategy="native_list",
            progress_strategy="streamed_process",
        ),
    ),
    "kotoba_whisper": ModelBackend(
        id="kotoba_whisper",
        label="Kotoba-Whisper 日语",
        kind="asr",
        runtime="optional_python",
        default_model="kotoba-tech/kotoba-whisper-v2.2",
        models=(
            "kotoba-tech/kotoba-whisper-v2.2",
            "kotoba-tech/kotoba-whisper-v2.1",
            "kotoba-tech/kotoba-whisper-v2.0",
        ),
        help="日本语料蒸馏的 Whisper，适合作为 Parakeet 的质量复核模型。",
        setup="“进阶”档位会安装运行依赖和 v2.2 模型。",
        tested_default=True,
        support_level="verified",
        devices=("cpu", "cuda"),
        minimum_vram_gb=3,
        recommended_vram_gb=6,
        disk_gb=4,
        installer="python-extra",
        python_extra="asr-kotoba-whisper",
        homepage="https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2",
        execution=ExecutionCapabilities(
            batch_strategy="internal_chunks",
            quality_sensitive_batch=True,
        ),
    ),
    "faster_whisper": ModelBackend(
        id="faster_whisper",
        label="Faster-Whisper（日语/英语）",
        kind="asr",
        runtime="optional_python",
        default_model="large-v2",
        models=(
            "large-v2",
            "kotoba-tech/kotoba-whisper-v2.0-faster",
            "distil-large-v2",
            "large-v3",
            "large-v3-turbo",
            "medium",
            "small",
        ),
        help=(
            "CTranslate2 版 Whisper，支持日语/英语、词级时间戳和可选后端 VAD。"
            "英语项目会自动使用该后端并排除日语专用 Kotoba 模型。"
        ),
        setup="“进阶”档位会安装运行依赖和 large-v2 模型。",
        tested_default=True,
        support_level="verified",
        devices=("cpu", "cuda"),
        minimum_vram_gb=2,
        recommended_vram_gb=6,
        disk_gb=4,
        installer="python-extra",
        python_extra="asr-faster-whisper",
        homepage="https://github.com/SYSTRAN/faster-whisper",
        execution=ExecutionCapabilities(
            batch_strategy="internal_chunks",
            quality_sensitive_batch=True,
        ),
    ),
}


TTS_BACKENDS: dict[str, ModelBackend] = {
    "indextts2": ModelBackend(
        id="indextts2",
        label="IndexTTS2 本地音色克隆（推荐）",
        kind="tts",
        runtime="command",
        default_model="IndexTTS2",
        models=("IndexTTS2",),
        help=(
            "零样本音色克隆，可把音色参考和情绪控制分开。音色参考不要求转写；"
            "可用文本情绪或情绪向量，随机采样会降低克隆一致性。"
        ),
        setup=(
            "在“设备与模型”页一键安装；也可在 Windows 运行 "
            "`./scripts/windows/install-indextts2.ps1`，Linux 运行 "
            "`bash scripts/linux/install-indextts2.sh`。程序会自动调用隔离环境；"
            "也可手工准备官方仓库和 checkpoints 后填写模型目录。"
        ),
        reference_audio=True,
        reference_text="unused",
        style_reference=True,
        tested_default=True,
        support_level="verified",
        devices=("cuda",),
        minimum_vram_gb=6,
        recommended_vram_gb=10,
        disk_gb=20,
        installer="isolated",
        homepage="https://github.com/index-tts/index-tts",
        execution=ExecutionCapabilities(
            batch_strategy="native_list",
            reusable_reference_conditioning=True,
            progress_strategy="streamed_process",
            persistent_session=True,
        ),
    ),
    "gpt_sovits": ModelBackend(
        id="gpt_sovits",
        label="GPT-SoVITS 外部 API",
        kind="tts",
        runtime="http",
        default_model="GPT-SoVITS-v4",
        models=("GPT-SoVITS-v4", "GPT-SoVITS-v3", "GPT-SoVITS-v2"),
        help=(
            "连接官方 api_v2.py 的 /tts 接口。参考音频路径必须是服务端可见路径；"
            "同一台机器上的服务可直接使用；Docker/远程服务需挂载或映射参考音频路径。"
        ),
        setup="在 GPT-SoVITS 环境启动 `python api_v2.py`，默认端口 9880。",
        reference_audio=True,
        reference_text="required",
        support_level="supported",
        devices=("cpu",),
        homepage="https://github.com/RVC-Boss/GPT-SoVITS",
        execution=ExecutionCapabilities(
            batch_strategy="request_concurrency",
            reusable_reference_conditioning=True,
        ),
    ),
    "cosyvoice": ModelBackend(
        id="cosyvoice",
        label="CosyVoice 外部 API",
        kind="tts",
        runtime="http",
        default_model="Fun-CosyVoice3-0.5B",
        models=("Fun-CosyVoice3-0.5B-2512", "CosyVoice2-0.5B", "CosyVoice-300M"),
        help=(
            "连接官方 FastAPI runtime。零样本模式使用参考音频+文本；跨语言模式只用参考音频，"
            "更方便但对音色/韵律的约束方式不同。"
        ),
        setup="启动 CosyVoice `runtime/python/fastapi/server.py`，默认端口 50000。",
        reference_audio=True,
        reference_text="optional",
        support_level="supported",
        devices=("cpu",),
        homepage="https://github.com/FunAudioLLM/CosyVoice",
        execution=ExecutionCapabilities(
            batch_strategy="request_concurrency",
            reusable_reference_conditioning=True,
        ),
    ),
    "fish_speech": ModelBackend(
        id="fish_speech",
        label="Fish Speech / Fish Audio 云服务 API",
        kind="tts",
        runtime="http",
        default_model="fish-speech-1.5",
        models=("fish-speech-1.5", "s1", "speech-1.6"),
        help=(
            "面向 Fish Speech 自建 OpenAPI 或 Fish Audio 兼容服务。不同发行版接口变化较快，"
            "本工具使用 /v1/tts 的 references(audio+text) 请求格式。"
        ),
        setup="启动兼容 /v1/tts 服务；云服务请保存 API Key。",
        reference_audio=True,
        reference_text="required",
        api_key=True,
        devices=("cpu",),
        homepage="https://github.com/fishaudio/fish-speech",
        execution=ExecutionCapabilities(
            batch_strategy="request_concurrency",
            reusable_reference_conditioning=True,
        ),
    ),
}


def asr_backend(backend_id: str) -> ModelBackend:
    try:
        return ASR_BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(f"unknown ASR backend: {backend_id}") from exc


def tts_backend(backend_id: str) -> ModelBackend:
    try:
        return TTS_BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(f"unknown TTS backend: {backend_id}") from exc
