# ASMR Dubber 0.7.2

ASMR Dubber 面向 64 位 Windows 10/11 和 x86_64 Linux。它处理日语或英语音频/视频，提供可校对的
源语言识别、中文翻译、音色克隆、混音和字幕工作流，也支持导入中文台本后直接配音。英语项目
使用现有 Faster-Whisper，不增加新的识别模型。

主界面可以直接在系统文件管理器中打开当前项目目录。日语内置翻译 Prompt 会把
「しこしこ」「シコシコ」及其罗马字转写按拟声处理，避免它们进入中文配音。

无时间轴台本现在可以先按项目设置完成 ASR 和翻译，再由大模型参照原文或中文台本校正句子文字。
识别得到的时间轴保持不变；SRT、VTT、ASS/SSA 和 LRC 等带时间轴文件仍然直接导入。

新项目的音频语言放在 ASR 设置中。选择英语时，界面会同时过滤不兼容的识别后端、VAD 和交叉
校对模型。已有原文台本沿用当前项目语言，导入时只需区分原文台本和中文配音稿。日语、英语
翻译 Prompt 分开保存，切换语言时会显示对应的内置或自定义内容。

中文配音默认从原字幕开始时间向后偏移 500 ms。当前句超过下一条有效中文配音的开始时间时，
混音阶段会在默认 1.8×、最高可选 4× 的上限内做保持音调的自动加速；达到上限仍放不下时
允许重叠。逐句 TTS 缓存不会因此被改写。

响度设置改为“跟随对应原声”“所有中文保持统一音量”和“保留 TTS 原始音量”三种互斥方式，
只显示当前方式需要的主要参数。高级响度参数默认折叠，并逐项说明用途。

新项目沿用本次验证使用的默认组合：Parakeet 与 Kotoba-Whisper 作为交叉校对候选、Qwen3
ForcedAligner 作为时间戳优先来源、IndexTTS2 情绪权重 0.5、中文相对原声 -8 dB，并保留
中文克隆音轨可以单独导出，之后重新加入原音轨；交叉校对仍默认关闭，未安装的模型不会显示为可选项。

DeepSeek Flash 翻译使用随包提供的结构约束 Prompt，并在返回格式校验失败时按相同 ID 和顺序
重试。Prompt 位于 `src/asmr_dubber/prompts`，可以直接查看。

## 下载包包含什么

- ASMR Dubber 源码和命令行工具；
- Windows 安装器与启动器；
- Windows/Linux 安装、启动和模型管理脚本；
- ModelScope 镜像与制品校验清单；
- 完整用户、配置、后端、排障和维护文档。

下载包不包含大型模型、用户项目、API Key 或已经建立的运行环境。首次安装会按用户选择准备
便携 Python、依赖和固定模型，所有内容默认放在程序目录的 `.asmr-dubber`。

Windows 用户从 [GitHub Releases](https://github.com/EveningStudy/ASMR-Dubber/releases/latest)
下载 `ASMR-Dubber-windows-portable.zip`，完整解压后运行 `ASMR-Dubber-Setup.exe`。

## 安装方案

| 方案 | 内容 | 建议可用空间 |
|---|---|---:|
| 基础 | 程序、网页、音频工具和外部 API 客户端，不含本地识别模型 | 5 GB |
| 推荐 | 基础 + Parakeet CTC 1.1B JA GAL + Parakeet TDT/CTC 0.6B JA；NVIDIA 电脑再装 IndexTTS2 | 35 GB |
| 进阶 | 推荐模型 + Kotoba-Whisper v2.2 + Faster-Whisper large-v2 + ASMR Whisper VAD ONNX + Qwen3 ForcedAligner 0.6B；NVIDIA 电脑再装 IndexTTS2 | 50 GB |

Windows 用户完整解压后运行 `ASMR-Dubber-Setup.exe`。不要求预装 Python、Git、FFmpeg、
CUDA Toolkit 或 PowerShell 7；系统自带 Windows PowerShell 5.1 可以运行安装脚本。

Linux 用户运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

安装要求、硬件选择、下载策略和离线模型包见[安装指南](INSTALLATION.md)。

## 下载和完整性

安装器默认优先使用 ModelScope。固定制品在导入前检查字节数和 SHA-256，模型包还检查内部
manifest。GitHub、Hugging Face 和海外官方软件源默认不作为自动回退。

下载中断后重新运行 Setup 即可继续。已经完整且哈希正确的大文件会直接复用，不需要删除缓存
或重新下载。

## 数据和密钥

项目、模型、缓存、设置和密钥默认保存在程序目录内。API Key 以明文位于
`.asmr-dubber/config/secrets.json`，请用操作系统账户权限保护该目录，不要把它加入压缩包、
Git 仓库或公开网盘。

停止程序后删除整个程序文件夹即可卸载。删除前请先备份需要保留的项目和密钥。

## 许可证

项目代码采用 MIT License。模型、运行时和外部服务适用各自条款，IndexTTS2 使用独立的
bilibili Model Use License。请阅读[第三方软件、模型与服务](THIRD_PARTY_NOTICES.md)，并确认
你有权处理输入作品和参考声音。
