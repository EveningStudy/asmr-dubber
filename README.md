# ASMR Dubber

ASMR Dubber 是一套日语音视频中文配音工具。它把一段完整媒体拆成可校对的句子，可以依次完成
日语识别、中文翻译、音色克隆和混音，也可以单独导出日中字幕。程序通过本机网页操作，项目、
模型、设置和缓存默认都留在程序文件夹中。


## 能做什么

- 从音频或视频创建独立项目，并保留输入文件的原始副本；
- 使用 Parakeet、Kotoba-Whisper 或 Faster-Whisper 做 ASR（语音识别）；
- 可导入日语或中文的 SRT、VTT、ASS/SSA、LRC 和纯文本台本；中文台本可跳过 ASR 与翻译，
  直接进入校对和配音；
- 可选日语 ASMR 专用 VAD（语音活动检测）和 Qwen3 ForcedAligner 时间戳对齐；
- 使用多个已安装的识别模型交叉校对日文；
- 通过 DeepSeek、OpenAI、Claude、Gemini、DeepL、Google 或 Microsoft 等服务翻译；
- 使用本地 IndexTTS2，或连接 GPT-SoVITS、CosyVoice、Fish Speech/Fish Audio API 做
  TTS（语音合成）；
- 逐句调整日文、中文、起止时间和是否配音；
- 输出混音后的 WAV、视频，以及双语/中文/日文 SRT 和 LRC 字幕；
- 中断后继续工作，只重做缺失或设置已经失效的部分。
- 长任务可在网页中取消；程序日志可直接查看和下载。

目前的识别流程按日语设计，目标翻译和配音语言为简体中文。目前不提供其它语种的完整工作流。

## 演示

### 素材 1

https://github.com/user-attachments/assets/70d1af0d-a165-410e-a28d-2a5dbaa203dd

### 素材 2

https://github.com/user-attachments/assets/ca7884c7-9272-4b07-a527-5e7c0351758b

### 素材 3

https://github.com/user-attachments/assets/d7106c36-8a5d-4aab-96b3-0f17d027d0d3

### 素材 4（18+）

https://github.com/user-attachments/assets/f938ef87-ebf4-4597-87ec-50313ff4a20c

[![B 站演示视频封面](assets/demos/bilibili-preview.jpg)](https://www.bilibili.com/video/BV1f43G6YEov/)

[▶ 在 B 站观看完整演示](https://www.bilibili.com/video/BV1f43G6YEov/)

以上素材仅用于功能演示，如有侵权请联系删除。

## 下载与安装（Windows）

从 [GitHub Releases](https://github.com/EveningStudy/ASMR-Dubber/releases/latest) 下载
`ASMR-Dubber-windows-portable.zip`，完整解压到最终使用位置。不要直接在压缩软件里运行。

> **使用前请开启 Windows 长路径。以避免一些问题** 进入“设置 → 系统 → 高级 → 文件资源管理器”，打开
> “启用长路径”，然后重新启动 ASMR Dubber。未开启时，IndexTTS2 等第三方运行环境中的深层
> 文件可能超过 Windows 的传统路径限制，出现“文件明明存在但程序报告找不到”的错误。如果
> 当前 Windows 没有这个开关，请把程序解压到 `D:\ASMR-Dubber` 这类短路径。

![Windows“启用长路径”开关位置](assets/windows-enable-long-paths.png)

Windows 包要让进阶组件全部使用 GPU，需要 NVIDIA Turing 或更新架构。主环境使用 CUDA 13，
不支持 Maxwell、Pascal 和 Volta；这些旧卡即使显存较大，也不属于完整本地 GPU 支持范围。


## 快速开始

### Windows

适用于 64 位 Windows 10/11。

1. 下载项目压缩包并完整解压。不要直接在压缩软件里运行。
2. 双击 `ASMR-Dubber-Setup.exe`，按提示选择安装方案。
3. 安装结束后双击 `ASMR-Dubber.exe`。
4. 浏览器打开后，先到“设置”确认识别、翻译和合成方式，再回到“项目工作台”。
5. 保留启动终端。要停止程序，按 `Ctrl+C` 或关闭终端。

IndexTTS2 第一次生成语音时需要加载模型并初始化 CUDA，等待时间可能明显长于后续任务。

不需要预装 Python、uv、Git、FFmpeg 或 CUDA Toolkit。启动器优先使用 PowerShell 7；电脑上
没有 PowerShell 7 时会使用 Windows 自带的 PowerShell 5.1。下载阶段需要系统中的
`curl.exe`。

建议把程序解压到短且可写的路径，例如 `D:\Apps\ASMR-Dubber`。不要放在
`C:\Program Files`，也不要让云盘同步程序运行时目录。

### Linux

适用于 64 位 x86_64 Linux。安装前确认系统有 `bash`、`curl`、`tar` 和 `getconf`：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

本地 NVIDIA 模型还需要可用的显卡驱动。程序会准备自己的 Python 和依赖，不修改系统 Python。

更完整的系统要求、磁盘估算、下载策略和离线安装方法见
[安装指南](docs/INSTALLATION.md)。

## 选择安装方案

| 方案 | 安装内容 | 安装后约占用 | 安装前建议可用空间 |
|---|---|---:|---:|
| 基础 | 程序、网页、音频工具和外部 API 客户端；不含本地识别模型 | 约 2 GB | 至少 5 GB |
| 推荐 | 基础 + 两个 Parakeet 日语模型；NVIDIA 电脑再装 IndexTTS2 | 约 24–28 GB | 至少 35 GB |
| 进阶 | 基础 + 下列固定分析模型；NVIDIA 电脑再装 IndexTTS2 | 约 33–39 GB | 至少 50 GB |

“进阶”会准备以下模型，不代表下载某个模型仓库中的所有内容：

1. Parakeet CTC 1.1B JA GAL；
2. Parakeet TDT/CTC 0.6B JA；
3. Kotoba-Whisper v2.2；
4. Faster-Whisper large-v2；
5. 日语 ASMR 专用 Whisper VAD ONNX；
6. Qwen3 ForcedAligner 0.6B；
7. IndexTTS2 checkpoints，仅在检测到 NVIDIA GPU 时安装。

没有 NVIDIA GPU 的电脑仍可使用 CPU 识别和外部 TTS（语音合成）API。CPU 处理长音频会慢
很多；IndexTTS2 不提供 CPU 模式。硬件选择可参考[后端指南](docs/BACKENDS.md)。

## 做完第一个项目

1. 在“设置 → 设备与模型”确认所选后端显示为“可用”。
2. 在“设置 → 翻译”选择服务并保存 API Key；本地翻译接口可按服务实际情况留空密钥。
3. 如已打开项目，点击“保存并应用到当前项目”。“仅保存为以后新项目默认值”不会改动当前项目。
4. 回到“项目工作台”，选择日语音频或视频并点击“新建项目”。
5. 点击“运行 ASR（语音识别）”，检查句子表里的文字和时间。已有台本或字幕时，也可以展开
   “已有台本或字幕”直接导入。导入前选择台本语言；中文台本会跳过 ASR 和翻译。
6. 日语台本或识别结果需要点击“翻译日文”；中文台本可直接编辑中文并保存校对表格。
7. 在“统一音色参考”中选择一条清晰的 5–15 秒台词。
8. 点击“TTS（语音合成）并混音”。成品会显示在页面中，也会保存在项目的 `output` 目录。

每句中文默认从原字幕开始时间向后偏移 500 ms。需要调整时，可在“混音与字幕”中填写其它
毫秒偏移；中文音频挤到下一句时，程序会在默认 1.8×、最高可选 4× 的上限内自动加速，仍放
不下则保留上限并允许重叠。这个过程只影响混音，不会修改逐句 TTS 缓存。

字幕不依赖完整配音流程。识别完成后即可选择字幕内容并点击“生成字幕”。详细操作和常见参数
取舍见[使用指南](docs/USER_GUIDE.md)。

## 支持的后端

| 环节 | 可选后端 |
|---|---|
| ASR（语音识别） | Parakeet 日语、Kotoba-Whisper、Faster-Whisper |
| 时间戳 | 识别模型自带时间戳、Qwen3 ForcedAligner 0.6B |
| TTS（语音合成） | IndexTTS2、GPT-SoVITS API、CosyVoice API、Fish Speech/Fish Audio API |
| 翻译 | DeepSeek、OpenAI、Anthropic Claude、Google Gemini、OpenAI-compatible、DeepL、Google Cloud Translation、Microsoft Azure Translator |

程序不会在任务开始后悄悄换用另一个模型。网页中的模型安装状态来自本地文件和运行环境检测；
多模型交叉校对也只列出本机完整可用的识别模型。

## 数据放在哪里

默认情况下，所有可变数据都在程序根目录的 `.asmr-dubber`：

```text
<程序目录>/
├── .asmr-dubber/
│   ├── bootstrap/       # uv
│   ├── runtimes/        # Python、FFmpeg 和隔离模型运行时
│   ├── venv/            # 主程序环境
│   ├── models/          # 本地模型
│   ├── cache/           # 下载与计算缓存
│   ├── config/          # 设置、密钥和外部参考音频
│   ├── projects/        # 项目
│   ├── logs/            # Setup 与程序运行日志
│   └── temp/            # 可清理的临时文件
├── model-packs/         # 可选的离线模型包入口
├── ASMR-Dubber.exe
└── ASMR-Dubber-Setup.exe
```

API Key 按便携设计以**明文**保存在 `.asmr-dubber/config/secrets.json`。它不会写入项目文件、
性能记录或字幕，但任何能读取程序目录的账户都能读取它。不要共享 `.asmr-dubber`，也不要把它
提交到 Git 或公开网盘。

停止程序后，可以复制整个文件夹来备份或搬到另一块磁盘；移动后重新运行 Setup，让脚本修复
运行环境里的绝对路径。要卸载，先关闭程序和相关模型服务，再删除整个程序文件夹。

## 项目目录

每个项目都有自己的设置快照和中间结果：

```text
<项目>/
├── project.json         # 项目状态和设置
├── source.<扩展名>      # 输入媒体副本
├── analysis/            # 分析音频、识别候选和校对记录
├── references/          # 项目参考片段
├── chinese/             # 逐句中文音频缓存
├── mix/                 # 可选中文中间轨
├── output/              # 最终音频和视频
├── subtitles/           # SRT、LRC 和字幕视频
├── exports/             # JSON、CSV 时间轴
└── performance.json     # 不含密钥和 Prompt 的阶段指标
```

不要只复制 `project.json`。需要备份项目时，请复制整个项目目录。

## 命令行

大多数用户只需要网页。脚本、自动化和排障可以使用命令行：

```powershell
.\scripts\windows\run-cli.ps1 doctor --no-network
.\scripts\windows\run-cli.ps1 --help
```

```bash
bash scripts/linux/run-cli.sh doctor --no-network
bash scripts/linux/run-cli.sh --help
```

常用命令有 `create`、`analyze`、`translate`、`synthesize`、`mix`、`subtitles`、
`set-timing`、`install-backend` 和 `verify-asr`。

## 文档

- [安装指南](docs/INSTALLATION.md)：系统要求、安装方案、网络、离线包和卸载
- [使用指南](docs/USER_GUIDE.md)：从建项目到配音、字幕和恢复任务
- [命令行参考](docs/CLI.md)：项目阶段、模型包、后端安装和自动化命令
- [配置参考](docs/CONFIGURATION.md)：设置作用域和各项参数
- [后端指南](docs/BACKENDS.md)：模型、API、硬件需求和组合方式
- [排障指南](docs/TROUBLESHOOTING.md)：安装、下载、模型、API 和输出问题
- [架构说明](docs/ARCHITECTURE.md)：项目结构、数据流和扩展边界
- [ModelScope 制品维护](docs/MODELSCOPE_UPLOADS.md)：镜像仓库与发布校验
- [第三方软件与模型说明](docs/THIRD_PARTY_NOTICES.md)
- [贡献指南](CONTRIBUTING.md)与[安全策略](SECURITY.md)

## 许可与责任

项目代码采用 [MIT License](LICENSE)。模型、运行时和云服务适用各自的许可证或服务条款，
其中 IndexTTS2 使用独立的 bilibili Model Use License。请确认你有权处理输入作品和参考声音，
并按适用规则标记合成内容。详见[第三方软件与模型说明](docs/THIRD_PARTY_NOTICES.md)。

## 配套工具

[ASMR-Dubber AutoFlow](https://github.com/EveningStudy/asmr-dubber-autoflow) 面向DLsite音声
作品，可以按顺序快速合并分轨音频，并衔接 ASMR Dubber 的处理流程，方便整理成双语音声或
视频后发布到某些视频网站。
