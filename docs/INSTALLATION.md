# 安装指南

这份指南面向第一次安装 ASMR Dubber 的用户。程序采用便携目录，不需要管理员权限；只要当前
账户能写入解压目录，运行环境、模型、缓存和项目都会放在该目录下。

## 安装前检查

### Windows

- 64 位 Windows 10 或 Windows 11；
- x86_64 处理器；
- 可写的 NTFS 目录；
- 可用的 `curl.exe`；
- 安装期间保持网络稳定，并按所选方案预留磁盘空间。

Windows 版不要求预装 Python、uv、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。启动器先找
`C:\Program Files\PowerShell\7\pwsh.exe`，找不到时使用系统自带的
`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`。因此只有 Windows
PowerShell 5.1 的电脑也能安装和启动。

系统必须能找到 `curl.exe`。可在命令提示符中检查：

```powershell
curl.exe --version
```

建议使用 `D:\Apps\ASMR-Dubber` 这类短路径。不要从 ZIP 内直接运行，不要安装到
`C:\Program Files`，也不要把 `.asmr-dubber` 放进 OneDrive 等实时同步目录。杀毒软件首次
扫描数万个 Python 和模型文件时可能让导入阶段明显变慢。

### Linux

- 64 位 x86_64 Linux；
- `bash`、`curl`、`tar` 和 `getconf`；
- 解压目录可写；
- 使用本地 NVIDIA 模型时，显卡驱动必须能正常提供 CUDA。

可以先检查基本工具：

```bash
uname -m
getconf LONG_BIT
command -v bash curl tar getconf
```

预期架构是 `x86_64`，位数是 `64`。macOS、ARM64 和 32 位系统不在支持范围内。WSL 会按
Linux 环境运行，但桌面浏览器、显卡直通和跨文件系统性能取决于 WSL 配置；原生 Windows
用户应优先使用 Windows 版。

## 硬件怎么选

本地识别既能用 CPU，也能用 NVIDIA CUDA。CPU 适合短音频、验证流程或没有独立显卡的电脑；
处理长音频时通常慢得多。本地 IndexTTS2 只支持 NVIDIA GPU。

| 组件 | 最低条件 | 更合适的配置 |
|---|---|---|
| Parakeet 日语 | CPU 或 NVIDIA CUDA | 约 6 GB 以上显存 |
| Kotoba-Whisper | CPU；GPU 约 3 GB 显存起 | 约 6 GB 以上显存 |
| Faster-Whisper | CPU；GPU 约 2 GB 显存起 | 约 6 GB 以上显存 |
| ASMR 专用 VAD | CPU，ONNX Runtime | 无需独立显卡 |
| Qwen3 ForcedAligner | CPU 或 CUDA，实际速度取决于 PyTorch | 与识别模型共用时留足显存 |
| IndexTTS2 | NVIDIA CUDA，约 6 GB 显存起 | 10 GB 以上显存 |

显存下限表示模型有机会装入，不代表长任务一定稳定。显示器、浏览器和其它程序也会占显存。
6 GB 显存的电脑建议一次只运行一个模型、保持批大小为 1，并关闭其它 GPU 程序。AMD/Intel
显卡不会用于这些 CUDA 后端，可改用 CPU 识别和外部 TTS（语音合成）API。

## Windows 免安装包

不想等待 Setup 下载时，可以直接使用已经装好运行环境和模型的完整压缩包：

[在 ModelScope 下载 Windows 免安装包](https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Windows-Portable/files)

| 压缩包 | 已包含内容 |
|---|---|
| `ASMR-Dubber-Windows-Core-Portable-v0.4.0.zip` | 主程序、网页、便携 Python、FFmpeg、翻译和外部 TTS API 客户端；不含本地 ASR（语音识别）模型和 IndexTTS2 checkpoints |
| `ASMR-Dubber-Windows-Recommended-Portable-v0.4.0.zip` | 核心包的全部内容；Parakeet CTC 1.1B JA GAL；Parakeet TDT/CTC 0.6B JA；IndexTTS2 完整隔离环境和 checkpoints |
| `ASMR-Dubber-Windows-Advanced-Portable-v0.4.0.zip` | 推荐包的全部内容；Kotoba-Whisper v2.2；Faster-Whisper large-v2；`TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`；`Qwen/Qwen3-ForcedAligner-0.6B`；这些模型需要的固定 Windows 运行依赖 |

三个 ZIP 都是完整包，互相不依赖。选择其中一个下载，完整解压到最终使用位置，然后双击
`ASMR-Dubber.exe`；首次使用也不需要运行 `ASMR-Dubber-Setup.exe`。同名 `.sha256` 文件
用于检查下载是否完整。

IndexTTS2 只支持 NVIDIA GPU。把推荐包或进阶包解压到没有 NVIDIA GPU 的电脑不会妨碍其它
功能；本地配音请改用外部 TTS API。免安装包已经包含大量小文件，首次解压和杀毒软件扫描可能
需要较长时间。不要直接在压缩软件里启动程序。

## 三种安装方案

### 基础

安装后约占 2 GB，建议安装前至少留出 5 GB。

包含程序、网页界面、便携 Python、音频处理依赖以及翻译/TTS 外部 API 客户端，不下载大型
识别或语音合成模型。基础方案可以打开网页、配置云服务和管理项目，但在安装至少一个 ASR
（语音识别）模型前不能完成本地识别。

适合以下情况：

- 先检查界面和服务配置；
- 只准备基础环境，稍后从“设备与模型”按需安装；
- 使用外部 TTS 服务，不需要本地 IndexTTS2。

### 推荐

安装后约占 24–28 GB，建议安装前至少留出 35 GB。

固定安装两个 Parakeet 日语模型：

- Parakeet CTC 1.1B JA GAL，默认质量优先；
- Parakeet TDT/CTC 0.6B JA，资源占用较低，也可用于对照。

检测到 NVIDIA GPU 时还会安装 IndexTTS2 的隔离运行环境和 checkpoints。没有 NVIDIA GPU
时会跳过 IndexTTS2，实际占用较小；此时可用 Parakeet CPU 识别和外部 TTS API。

推荐方案适合只需要一套稳定识别模型、不开多模型交叉校对的用户。

### 进阶

安装后约占 33–39 GB，建议安装前至少留出 50 GB。安装内容是固定清单：

| 环节 | 模型 |
|---|---|
| ASR（语音识别） | Parakeet CTC 1.1B JA GAL |
| ASR（语音识别） | Parakeet TDT/CTC 0.6B JA |
| ASR（语音识别） | `kotoba-tech/kotoba-whisper-v2.2` |
| ASR（语音识别） | `Systran/faster-whisper-large-v2` |
| VAD（语音活动检测） | `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx` |
| 时间戳对齐 | `Qwen/Qwen3-ForcedAligner-0.6B` |
| TTS（语音合成） | IndexTTS2 checkpoints，仅 NVIDIA GPU |

Windows NVIDIA 环境还会准备这些模型需要的 PyTorch、Transformers、Accelerate、ONNX
Runtime、`qwen-asr` 和 Faster-Whisper/CTranslate2。它们是运行库，不是额外模型。

进阶方案适合需要切换识别器、多模型交叉校对、ASMR 专用 VAD 或独立时间戳对齐的用户。它
不会顺带下载 Kotoba v2.0/v2.1、Faster-Whisper large-v3 或其它注册表模型。

## Windows 安装

1. 把下载包完整解压到最终使用位置。
2. 确认根目录有 `ASMR-Dubber-Setup.exe`、`ASMR-Dubber.exe`、`mirrors.json` 和
   `scripts` 文件夹。
3. 双击 `ASMR-Dubber-Setup.exe`。
4. 输入 `1`、`2` 或 `3`；直接回车选择“推荐”。
5. 等待环境检查结束。下载大型模型时不要关闭窗口或让电脑休眠。
6. 看到“安装或修复完成”后，双击 `ASMR-Dubber.exe`。

安装日志保存在：

```text
.asmr-dubber/logs/setup-年月日-时分秒-毫秒.log
```

安装失败时可以直接再次运行 Setup。完整且 SHA-256 校验通过的下载会复用；中断文件会保留
断点。反复运行不会重复建立另一套系统环境。

也可以从 PowerShell 运行脚本：

```powershell
.\scripts\windows\setup.ps1 -Profile 基础
.\scripts\windows\setup.ps1 -Profile 推荐
.\scripts\windows\setup.ps1 -Profile 进阶
```

NVIDIA 电脑如果不打算使用本地 IndexTTS2，可以跳过它：

```powershell
.\scripts\windows\setup.ps1 -Profile 推荐 -SkipRecommendedTTS
```

## Linux 安装

在项目根目录执行：

```bash
bash scripts/linux/setup.sh 基础
bash scripts/linux/setup.sh 推荐
bash scripts/linux/setup.sh 进阶
```

安装完成后启动网页：

```bash
bash scripts/linux/run-ui.sh
```

不安装 IndexTTS2：

```bash
ASMR_DUBBER_SKIP_RECOMMENDED_TTS=1 bash scripts/linux/setup.sh 推荐
```

脚本只为当前项目准备运行环境，不向系统 Python 安装包，也不修改全局 `PATH`。

## 下载来源

下载策略由根目录的 `mirrors.json` 统一管理：

1. 运行时、依赖包和模型包优先从 ModelScope 获取；
2. ModelScope wheelhouse 没有发布时，小型 Python 依赖可使用配置中的国内 PyPI 镜像；
3. GitHub、Hugging Face、hf-mirror、官方 PyPI 和 PyTorch 海外源默认关闭；
4. 所有固定制品在使用前检查大小和 SHA-256，模型包还会检查内部 manifest。

公开 ModelScope 仓库通常不需要 Token。私有仓库才需要在当前终端临时设置
`MODELSCOPE_API_TOKEN`，不要把 Token 写进文档、脚本或问题附件。

只有你明确愿意使用海外备用源时，才为当前进程设置：

```powershell
$env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS = '1'
.\ASMR-Dubber-Setup.exe
```

```bash
ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1 bash scripts/linux/setup.sh 推荐
```

这个开关不会写入永久设置。未开启时，某个 ModelScope 大文件失败会停止并保留断点，不会在
后台转向海外站点消耗流量。

## 离线模型包

把 ASMR Dubber 格式的模型 ZIP 原样放入根目录 `model-packs`。不要解压，不要改名，也不要
修改包内 manifest。随后有三种导入方式：

- 重新运行 Setup，让当前安装方案扫描对应模型包；
- 在网页“设置 → 设备与模型”点击“扫描并导入本地模型包”；
- 使用命令行：

```powershell
.\scripts\windows\run-cli.ps1 list-model-packs
.\scripts\windows\run-cli.ps1 import-model-packs --all
```

```bash
bash scripts/linux/run-cli.sh list-model-packs
bash scripts/linux/run-cli.sh import-model-packs --all
```

导入成功后可以删除 ZIP；已安装文件在 `.asmr-dubber/models` 或相应隔离运行时中。如果下载页
提供 `.part*.rar` 分卷，先用 7-Zip 或 WinRAR 从 `part1.rar` 解出完整 ZIP，再把 ZIP 放入
`model-packs`。程序不直接读取 RAR。

## 验证安装

启动网页后打开“设置 → 设备与模型”，点击“重新检测硬件与后端”。“可用”表示运行依赖和
所需模型都完整；“模型不完整”与“未安装”是两种不同状态。

命令行检查不会加载大型模型：

```powershell
.\scripts\windows\run-cli.ps1 doctor --no-network
```

```bash
bash scripts/linux/run-cli.sh doctor --no-network
```

需要真实验证某个识别后端时，可以用几秒钟的日语音频执行 `verify-asr`。完整参数见：

```powershell
.\scripts\windows\run-cli.ps1 verify-asr --help
```

## 移动、备份和卸载

移动或备份前先关闭启动终端，确保没有安装、识别或合成任务正在运行。复制整个程序目录即可
保留项目、设置、密钥、模型和缓存。移动到新路径后运行一次 Setup，让便携虚拟环境修复路径。

如果只想重新安装运行环境，可以在做好项目和密钥备份后删除 `.asmr-dubber`，再运行 Setup。
这会同时删除模型、缓存、配置和默认项目目录，不能撤销。

完整卸载步骤：

1. 关闭 ASMR Dubber 和你自行启动的外部模型服务；
2. 确认需要保留的项目已经复制到别处；
3. 删除整个 ASMR Dubber 程序文件夹。

程序不依赖系统级 Python 环境，也不需要保留卸载器。API Key 是明文文件，删除前后都应按你的
密钥管理要求处理备份和撤销。

遇到错误时先查看[排障指南](TROUBLESHOOTING.md)，并附上去除隐私信息后的最新安装日志。
