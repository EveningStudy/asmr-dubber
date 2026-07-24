# 第三方软件、模型与许可证说明

ASMR Dubber 自身代码采用 MIT License。Git 仓库、源码包和程序包**不包含模型权重**；Release 可以另行提供名称明确的独立离线模型包。安装器会按用户选择导入离线包或从对应上游下载软件和模型。每个离线包保留适用的许可证和归属说明。下载、使用和分发第三方组件时，用户必须遵守其当前许可证、服务条款和适用法律。本文件是信息性清单，不构成法律意见，也不会替代上游许可证全文。

## 支持的下载组件

| 组件 | 上游 | 许可证/说明 |
|---|---|---|
| CrispASR v0.8.21 | <https://github.com/CrispStrobe/CrispASR> | MIT；安装器固定 Release 并校验 SHA-256 |
| Parakeet CTC 1.1B JA GAL | <https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja> | Apache-2.0；只使用 GAL 来源的 F16 GGUF |
| Parakeet TDT/CTC 0.6B JA | <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja> | CC-BY-4.0；F16 GGUF 来自 cstr 转换仓库 |
| Kotoba-Whisper v2.x | <https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2> | 模型卡标记 Apache-2.0 |
| Faster-Whisper large-v2 | <https://huggingface.co/Systran/faster-whisper-large-v2> | 模型卡标记 MIT |
| Qwen3-ASR 1.7B/0.6B | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> | 模型卡标记 Apache-2.0 |
| Qwen3 ForcedAligner 0.6B | <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B> | 以模型仓库当前许可证为准 |
| VoxCPM2 | <https://huggingface.co/openbmb/VoxCPM2> | 模型卡与上游代码标记 Apache-2.0 |
| PyTorch / TorchAudio | <https://pytorch.org/> | BSD-style；以 wheel 内许可证为准 |
| FFmpeg shared build | <https://github.com/BtbN/FFmpeg-Builds> | Windows 安装器选择 LGPL shared build；许可证随运行时保存 |
| uv / managed CPython | <https://github.com/astral-sh/uv> | Apache-2.0/MIT；CPython 使用 PSF License |

安装器固定默认模型 revision，并对必要大权重校验 SHA-256，但许可证和模型输出责任仍由各上游条款决定。

## IndexTTS2

IndexTTS2 来自 <https://github.com/index-tts/index-tts>，代码和模型受 **bilibili Model Use License Agreement** 约束，不是本项目的 MIT License。该许可证包含使用限制、下游义务、合规要求和特定规模组织的额外授权条件。安装器把上游 `LICENSE`、`LICENSE_ZH.txt` 及 checkpoints 许可证保留在
IndexTTS2 独立运行时目录。运行安装器或使用模型前请完整阅读并决定是否接受；不同意时不要安装或使用 IndexTTS2。

ASMR Dubber 没有修改 IndexTTS2 模型权重；Release 可以单独提供保留上游中英文许可证和 README 的 checkpoints 离线包。获取、分发或使用该模型即受上游许可证约束，不代表 bilibili 或 IndexTTS2 权利人对本项目提供背书、保证或认可。

## 可选后端

Qwen3-TTS、Faster-Whisper、OpenAI Whisper、WhisperX、FunASR/SenseVoice、GPT-SoVITS、CosyVoice、F5-TTS、Fish Speech、XTTS-v2 及其模型均从各自上游安装或由用户启动。它们的代码许可证、模型许可、训练数据声明和商用条件可能不同且会变化；请在启用前阅读对应仓库和模型卡。

云端翻译/语音 API（DeepSeek、OpenAI、Anthropic、Google、Microsoft、DeepL、Fish Audio 等）受供应商服务条款、数据处理政策、地区可用性和计费规则约束。本项目不提供这些服务，也不授予其使用权。

## Python 依赖

Python 直接和传递依赖记录在 `pyproject.toml` 与 `uv.lock`。安装后的发行包/wheel 包含各项目自己的metadata 与许可证文件。发布者在重新分发打包后的依赖或独立可执行程序时，应生成完整许可证清单并保留原始 notices；本项目的 Source Release 默认不重新分发这些 wheel。

## 音频和声音权利

开源模型许可证不自动授予参考音频、声优声音、角色、作品或生成内容的权利。用户必须自行取得必要授权，不得使用本工具实施冒充、欺诈、骚扰、诽谤或侵犯隐私/人格权的行为。
