# 第三方软件、模型与服务

ASMR Dubber 自身代码采用[MIT License](../LICENSE)。这个许可证只覆盖本仓库有权许可的代码，不自动覆盖模型权重、第三方运行时、输入作品、参考声音、云服务或生成内容。

项目下载包通常不包含大型模型。Setup 下载或离线包导入的第三方内容仍受各自许可证约束。本页用于指出边界和上游来源，不替代许可证全文，也不构成法律意见。发布、商用或重新分发前，应读取实际制品中随附的许可证和 notices。

## 识别、VAD 和对齐

| 组件 | 上游 | 上游标示的许可证 |
|---|---|---|
| CrispASR | [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) | MIT；模型权重另行适用各模型许可证 |
| Parakeet CTC 1.1B JA | [grider-transwithai/parakeet-ctc-1.1b-ja](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja) | Apache-2.0 |
| Parakeet TDT/CTC 0.6B JA | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | CC BY 4.0 |
| Kotoba-Whisper v2.2 | [kotoba-tech/kotoba-whisper-v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | Apache-2.0 |
| Faster-Whisper large-v2 权重 | [Systran/faster-whisper-large-v2](https://huggingface.co/Systran/faster-whisper-large-v2) | MIT；由 OpenAI Whisper large-v2 转换 |
| Faster-Whisper 代码 | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 以仓库 LICENSE 为准；同时依赖 CTranslate2 |
| 日语 ASMR Whisper VAD ONNX | [TransWithAI/Whisper-Vad-EncDec-ASMR-onnx](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) | MIT；模型卡同时说明其基于 Whisper 表征 |
| Qwen3 ForcedAligner 0.6B | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache-2.0 |

模型包中的格式转换、量化或镜像分发不会把上游许可证改成 ASMR Dubber 的 MIT License。归属、引用、再分发和衍生模型义务应按模型卡及包内文件执行。

## IndexTTS2

IndexTTS2 来自[index-tts/index-tts](https://github.com/index-tts/index-tts)。上游仓库的代码、权重和模型输出受独立的 **bilibili Model Use License Agreement** 约束，不属于 OSI MIT 许可证。

该协议包含使用限制、下游分发义务、合规责任、高风险场景条款，以及针对特定用户规模或营收组织的单独授权条件。上游 README 也要求商业使用与合作方联系作者。不要根据“GitHub 可下载”推断任何用途都自动获准。

安装器固定一份上游源码 revision，并在隔离运行时保留 `LICENSE`、`LICENSE_ZH.txt` 和相关说明。运行或分发前应阅读这些原文；不同意时不要安装或使用 IndexTTS2。

## 基础运行时和媒体组件

安装器可能获取以下软件：

- [uv](https://github.com/astral-sh/uv)；
- [python-build-standalone](https://github.com/astral-sh/python-build-standalone)打包的 CPython；
- [CPython](https://www.python.org/)；
- [FFmpeg](https://ffmpeg.org/)与 BtbN 的 Windows shared 构建；
- [PyTorch](https://pytorch.org/)、TorchAudio、Transformers、ONNX Runtime、CTranslate2；
- [edge-tts](https://github.com/rany2/edge-tts)；
- `pyproject.toml` 和 `uv.lock` 中列出的 Python 直接与传递依赖。

这些组件分别适用自己的许可证。FFmpeg 的义务取决于实际构建配置；本项目选择的 Windows 归档名称标示为 LGPL shared 构建，但重新分发者仍应以归档内许可证、构建信息和实际链接库为准。不要删除 DLL、wheel、Python 发行包或源码归档随附的许可证。

发布含运行时或 wheelhouse 的安装包时，维护者应从最终制品生成依赖清单，并检查 notices，而不是只复制本页。

## 外部 TTS（语音合成）服务

ASMR Dubber 只实现以下服务的客户端适配，不分发或启动它们的服务端代码和模型：

- Microsoft Edge 在线语音服务，通过 [edge-tts](https://github.com/rany2/edge-tts) 调用；
- [小米 MiMo TTS](https://mimo.mi.com/docs/usage-guide/speech-synthesis)；
- [MiniMax TTS](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http)；
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)；
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)；
- [Fish Speech](https://github.com/fishaudio/fish-speech)与兼容的 Fish Audio 云服务。

同一项目名称下的代码、预训练权重、训练数据和托管 API 可能采用不同条款。自建服务请检查实际 checkout 和模型卡；云服务请检查账户对应的服务协议、价格、内容政策和数据处理说明。

## 翻译和云 API

DeepSeek、阿里云百炼、豆包/火山方舟、商汤 SenseNova、OpenAI、Anthropic、Google、Microsoft、DeepL、小米 MiMo、MiniMax 和 Fish Audio 等 API 由相应供应商提供。ASMR Dubber 不转售这些服务，也不授予调用权限。用户自行承担：

- 账户、地区、配额和费用；
- 输入文字或音频是否允许发送给该供应商；
- 供应商的数据保留、训练、隐私和内容政策；
- 服务输出的使用限制和准确性复核。

OpenAI-compatible 只是请求格式，不说明服务端采用何种代码、模型或许可证。

## 输入作品、声音和生成内容

模型许可证不会自动授予作品版权、表演者权、声音人格权、角色权利或隐私权。使用者必须确认：

- 有权复制和处理输入媒体；
- 有权把参考声音用于克隆或合成；
- 有权把所需文字和音频发送给所选外部服务；
- 发布时遵守适用的合成内容标识、平台规则和当地法律。

不得使用本工具实施冒充、欺诈、骚扰、诽谤、未经许可的声音克隆，或其它侵犯知识产权、隐私与人格权益的行为。

## 发布者检查清单

重新分发 ASMR Dubber、便携运行时或模型包时，至少完成：

1. 锁定每个第三方制品的准确版本或 revision；
2. 保留原始 `LICENSE`、`NOTICE`、模型卡和归属文件；
3. 确认转换/量化模型的来源和许可链；
4. 检查 wheelhouse 和二进制的传递依赖；
5. 在最终归档上核对许可证文件确实存在；
6. 对不能再分发的组件只提供用户自行获取的安装路径。

发现本清单与上游制品不一致时，应以实际许可证原文为准，并提交修正文档的 Issue 或补丁。
