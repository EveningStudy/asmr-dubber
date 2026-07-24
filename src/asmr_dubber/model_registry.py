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
    "stable_voice_sentence_style": "统一声纹 + 逐句语气（VoxCPM 实验）",
    "stable_hifi": "统一声纹 Hi-Fi（VoxCPM 实验）",
    "ultimate": "逐句 Hi-Fi（VoxCPM 实验）",
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
        label="NVIDIA Parakeet 日语 · CrispASR（推荐）",
        kind="asr",
        runtime="command",
        default_model=("grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"),
        models=(
            "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo",
            "nvidia/parakeet-tdt_ctc-0.6b-ja",
        ),
        help=(
            "日语专用 FastConformer，通过 MIT 许可的 CrispASR 原生运行时加载 F16 GGUF。"
            "默认使用 1.1B CTC GAL；官方 0.6B TDT/CTC 同步安装，可直接切换。1.1B 文件只由经过"
            "检查的 GAL checkpoint 转换。运行时会针对日语模型的长音频不稳定问题自动做"
            "全局归一化和约 8 秒流式编码，并输出真实词/段时间戳。"
        ),
        setup="在“设备与模型”页安装 CrispASR 和两款 F16 模型。",
        tested_default=True,
        support_level="verified",
        devices=("cpu", "cuda"),
        minimum_vram_gb=None,
        recommended_vram_gb=6,
        disk_gb=5,
        installer="isolated",
        python_extra="asr-parakeet",
        homepage="https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja",
        execution=ExecutionCapabilities(
            batch_strategy="native_list",
            progress_strategy="streamed_process",
        ),
    ),
    "kotoba_whisper": ModelBackend(
        id="kotoba_whisper",
        label="Kotoba-Whisper 日语（推荐）",
        kind="asr",
        runtime="optional_python",
        default_model="kotoba-tech/kotoba-whisper-v2.2",
        models=(
            "kotoba-tech/kotoba-whisper-v2.2",
            "kotoba-tech/kotoba-whisper-v2.1",
            "kotoba-tech/kotoba-whisper-v2.0",
        ),
        help=(
            "日本语料蒸馏的 Whisper。v2.2 是最新推荐项（转录 + 标点，并保留后续"
            "说话人扩展能力）；本工具按单人转录方式使用其 ASR 核心，"
            "不要求 pyannote 账号或 HF Token。若重视速度，可在 Faster-Whisper 中选择"
            " kotoba-whisper-v2.0-faster。"
        ),
        setup="在“设备与模型”页一键安装 Transformers 适配器；模型会下载到项目缓存。",
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
    "qwen3_asr": ModelBackend(
        id="qwen3_asr",
        label="Qwen3-ASR + ForcedAligner（推荐）",
        kind="asr",
        runtime="builtin",
        default_model="Qwen/Qwen3-ASR-1.7B",
        models=("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"),
        help=(
            "先识别完整日文，再由 Qwen 强制对齐器生成词/字时间戳。"
            "1.7B 质量优先，0.6B 显存占用较低。"
        ),
        setup="在“设备与模型”页一键安装，或重新运行当前平台的安装脚本。",
        tested_default=True,
        support_level="verified",
        devices=("cuda",),
        minimum_vram_gb=6,
        recommended_vram_gb=10,
        disk_gb=7,
        installer="python-extra",
        python_extra="local-default",
        homepage="https://github.com/QwenLM/Qwen3-ASR",
        execution=ExecutionCapabilities(
            batch_strategy="native_list",
            quality_sensitive_batch=True,
        ),
    ),
    "faster_whisper": ModelBackend(
        id="faster_whisper",
        label="Faster-Whisper（推荐）",
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
            "distil-large-v3.5",
        ),
        help=(
            "CTranslate2 版 Whisper，支持词级时间戳和 Silero VAD。速度快、生态成熟；"
            "保留 large-v3 系列并加入 large-v2 / distil-large-v2。ASMR 建议先关闭 VAD，"
            "避免把轻声或气声当静音；Kotoba-faster 是日语优先的快速选项。"
        ),
        setup="在独立环境安装 `faster-whisper`，或确认当前环境能够导入该包。",
        tested_default=True,
        support_level="verified",
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
    "openai_whisper": ModelBackend(
        id="openai_whisper",
        label="OpenAI Whisper（原版）",
        kind="asr",
        runtime="optional_python",
        default_model="large-v3",
        models=("large-v3", "turbo", "large-v2", "medium"),
        help="原版 Whisper Python 推理，启用词级时间戳；通常比 Faster-Whisper 慢。",
        setup="在独立环境安装官方 `openai-whisper`。",
        devices=("cpu", "cuda"),
        minimum_vram_gb=5,
        recommended_vram_gb=10,
        disk_gb=4,
        installer="python-extra",
        python_extra="asr-openai-whisper",
        homepage="https://github.com/openai/whisper",
        execution=ExecutionCapabilities(batch_strategy="internal_chunks"),
    ),
    "whisperx": ModelBackend(
        id="whisperx",
        label="WhisperX",
        kind="asr",
        runtime="optional_python",
        default_model="large-v3",
        models=("large-v3", "large-v3-turbo"),
        help=(
            "Whisper + 独立对齐模型，时间戳通常更精细。首次使用还会下载日语对齐模型，"
            "依赖较多，建议独立环境。"
        ),
        setup="在独立环境安装官方 `whisperx`。",
        devices=("cpu", "cuda"),
        minimum_vram_gb=5,
        recommended_vram_gb=10,
        disk_gb=6,
        homepage="https://github.com/m-bain/whisperX",
        execution=ExecutionCapabilities(batch_strategy="internal_chunks"),
    ),
    "funasr": ModelBackend(
        id="funasr",
        label="FunASR / SenseVoice",
        kind="asr",
        runtime="optional_python",
        default_model="FunAudioLLM/Fun-ASR-Nano-2512",
        models=(
            "FunAudioLLM/Fun-ASR-Nano-2512",
            "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
            "iic/SenseVoiceSmall",
        ),
        help=(
            "FunASR 工具链。Nano 支持日/中/英，SenseVoiceSmall 还可识别情绪和事件；"
            "本工具读取 sentence_info/VAD 句段作为时间轴。"
        ),
        setup="在独立环境安装官方 `funasr`；SenseVoice 建议同时配置 FSMN-VAD。",
        support_level="supported",
        disk_gb=4,
        installer="python-extra",
        python_extra="asr-funasr",
        homepage="https://github.com/modelscope/FunASR",
        execution=ExecutionCapabilities(batch_strategy="internal_chunks"),
    ),
    "openai_compatible_asr": ModelBackend(
        id="openai_compatible_asr",
        label="OpenAI-compatible ASR 服务",
        kind="asr",
        runtime="http",
        default_model="whisper-1",
        models=("whisper-1", "large-v3", "large-v3-turbo"),
        help=(
            "调用 /v1/audio/transcriptions 并请求 verbose_json 时间戳。适用于 whisper.cpp、"
            "FunASR Server、LocalAI、兼容云服务或自建服务。"
        ),
        setup="先启动兼容服务，并确保 verbose_json 返回 words 或 segments。",
        api_key=True,
        support_level="supported",
        devices=("cpu",),
        homepage="https://platform.openai.com/docs/api-reference/audio",
        execution=ExecutionCapabilities(
            batch_strategy="request_concurrency",
            progress_strategy="none",
        ),
    ),
}


TTS_BACKENDS: dict[str, ModelBackend] = {
    "voxcpm2": ModelBackend(
        id="voxcpm2",
        label="VoxCPM2 2B（兼容）",
        kind="tts",
        runtime="builtin",
        default_model="openbmb/VoxCPM2",
        models=("openbmb/VoxCPM2",),
        help=(
            "已验证的兼容后端。支持跨语言零样本克隆、仅音色参考和提示音频/文本模式；"
            "旧项目和现有缓存可继续使用。"
        ),
        setup="在“设备与模型”页一键安装，或重新运行当前平台的安装脚本。",
        reference_audio=True,
        reference_text="optional",
        style_reference=True,
        clone_modes=(
            "stable_reference",
            "reference_only",
            "stable_voice_sentence_style",
            "stable_hifi",
            "ultimate",
        ),
        tested_default=True,
        support_level="verified",
        devices=("cuda",),
        minimum_vram_gb=8,
        recommended_vram_gb=12,
        disk_gb=6,
        installer="python-extra",
        python_extra="local-default",
        homepage="https://github.com/OpenBMB/VoxCPM",
        execution=ExecutionCapabilities(
            reusable_reference_conditioning=True,
            persistent_session=True,
        ),
    ),
    "qwen3_tts": ModelBackend(
        id="qwen3_tts",
        label="Qwen3-TTS Voice Clone",
        kind="tts",
        runtime="optional_python",
        default_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        models=(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ),
        help=(
            "官方 Base 模型支持中日文和可复用 voice-clone prompt。完整参考文本质量最佳；"
            "仅声纹向量模式可不填文本，但克隆质量可能下降。"
        ),
        setup="建议在独立 Python 3.12 环境安装官方 `qwen-tts`。",
        reference_audio=True,
        reference_text="required",
        devices=("cuda",),
        minimum_vram_gb=4,
        recommended_vram_gb=8,
        disk_gb=5,
        homepage="https://github.com/QwenLM/Qwen3-TTS",
        execution=ExecutionCapabilities(
            batch_strategy="native_list",
            reusable_reference_conditioning=True,
            persistent_session=True,
        ),
    ),
    "indextts2": ModelBackend(
        id="indextts2",
        label="IndexTTS2（推荐）",
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
        label="GPT-SoVITS API v2",
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
        label="CosyVoice 2/3 FastAPI",
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
    "f5_tts": ModelBackend(
        id="f5_tts",
        label="F5-TTS CLI",
        kind="tts",
        runtime="command",
        default_model="F5TTS_v1_Base",
        models=("F5TTS_v1_Base", "F5TTS_Base"),
        help="调用官方 f5-tts_infer-cli。参考文本建议准确填写，否则 CLI 会额外启动 ASR。",
        setup="在独立环境安装官方 F5-TTS，并填写该环境中的 CLI 绝对路径。",
        reference_audio=True,
        reference_text="required",
        devices=("cpu", "cuda"),
        minimum_vram_gb=4,
        recommended_vram_gb=8,
        disk_gb=5,
        homepage="https://github.com/SWivid/F5-TTS",
        execution=ExecutionCapabilities(reusable_reference_conditioning=True),
    ),
    "fish_speech": ModelBackend(
        id="fish_speech",
        label="Fish Speech / Fish Audio API",
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
    "xtts_v2": ModelBackend(
        id="xtts_v2",
        label="Coqui XTTS-v2",
        kind="tts",
        runtime="optional_python",
        default_model="tts_models/multilingual/multi-dataset/xtts_v2",
        models=("tts_models/multilingual/multi-dataset/xtts_v2",),
        help="成熟的多语言零样本克隆基线。使用 speaker_wav，不要求参考文本。",
        setup="建议在独立兼容环境安装 Coqui `TTS`；其依赖可能与本项目主环境冲突。",
        reference_audio=True,
        reference_text="unused",
        devices=("cpu", "cuda"),
        minimum_vram_gb=4,
        recommended_vram_gb=8,
        disk_gb=4,
        homepage="https://huggingface.co/coqui/XTTS-v2",
        execution=ExecutionCapabilities(
            reusable_reference_conditioning=True,
            persistent_session=True,
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
