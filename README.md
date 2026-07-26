# ASMR Dubber

ASMR Dubber 在日语原声中加入逐句中文复述。日语时间轴和内容保持不变；中文由所选 TTS 后端使用日语声纹参考生成，并按对应原句的局部响度混入。

## 演示

- [素材 1](assets/demos/demo-1.wav)
- [素材 2](assets/demos/demo-2.wav)
- [素材 3](assets/demos/demo-3.wav)
- [素材 4（18+，请谨慎播放）](assets/demos/demo-4.wav)
- [B 站演示视频](https://www.bilibili.com/video/BV1f43G6YEov/)

以上素材仅用于功能演示，如有侵权请联系删除。

支持 Windows 10/11 和 64 位 Linux。

## 安装

代码包不包含模型权重。模型可由安装器下载，也可通过离线模型包导入。

### 获取程序

- Windows：从 [Releases](https://github.com/EveningStudy/ASMR-Dubber/releases) 下载 `ASMR-Dubber-windows-portable.zip` 并完整解压。
- Windows NVIDIA 用户也可从 [ModelScope](https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Windows-Recommended-Portable) 下载 `ASMR-Dubber-Windows-Recommended-v0.3.4.zip`。完整解压后直接运行 `ASMR-Dubber.exe`，无需先运行 Setup。
- Linux：下载 Source code 压缩包并解压，或运行 `git clone https://github.com/EveningStudy/ASMR-Dubber.git`。
- 已安装旧版本：备份重要项目后更新代码文件，保留原有 `.asmr-dubber`，再运行 Setup 修复依赖。

### Windows

首次使用时运行项目根目录中的 `ASMR-Dubber-Setup.exe`，在终端中选择 Core、Recommended、Advanced 或 Full。安装中断或依赖损坏时，再次运行它即可检查状态、复用已完成文件并继续安装。

依赖完成后运行 `ASMR-Dubber.exe`。它只负责启动服务并打开浏览器；如果 Core 环境未安装或不完整，会提示改用安装器修复。运行期间请保留终端；按 `Ctrl+C` 或关闭终端即可停止服务。

### Linux

要求 64 位 Linux、`bash` 和 `curl`。NVIDIA 后端需要可用的驱动。

```bash
bash scripts/linux/setup.sh Recommended
bash scripts/linux/run-ui.sh
```

打开终端显示的 `http://127.0.0.1:7860`。

### 下载镜像

根目录的 `mirrors.json` 按顺序列出 PyPI、Hugging Face、PyTorch、GitHub 下载代理和 Python 下载源。Windows 和 Linux 默认优先尝试国内可用地址，失败后自动切换，最后回退官方源；可直接编辑这个文件调整、删除或添加 HTTPS 地址。

### 安装配置

| 配置 | 内容 | 安装后约占 | 建议预留 |
|---|---|---:|---:|
| `Core` | 应用、UI 和基础音频依赖；不下载大型 ASR/TTS 权重 | 2 GB | 5 GB |
| `Recommended` | Core、两款 Parakeet；NVIDIA 设备安装 IndexTTS2 | 24–28 GB | 35 GB |
| `Advanced` | Recommended，另加 Kotoba-Whisper v2.2 和 Faster-Whisper large-v2 | 30–35 GB | 45 GB |
| `Full` | Advanced，另加其余已集成且可自动安装的本地后端 | 42–48 GB | 60 GB |

配置只决定首次批量准备的内容。安装后仍需在“设置 → 设备与模型”确认状态，并按需下载、安装或修复没有包含在当前配置中的模型。无 NVIDIA GPU 时，安装器跳过 CUDA 模型，实际占用也会减少；可使用 Faster-Whisper CPU `int8` 和外部 TTS 服务。

也可以先安装 Core，再从“设置 → 设备与模型”逐个安装后端。网页安装会先检查根目录的 `model-packs`，存在匹配包时优先导入；下载过程中可点击“暂停当前下载”，之后再次安装会复用已完成的缓存。

### 离线模型包

从项目提供的网盘下载模型包后，保持 ZIP 原名并放入项目根目录的 `model-packs` 文件夹，再运行 `ASMR-Dubber-Setup.exe`。安装器会在联网下载前校验并导入；已安装的文件会直接复用。若 ZIP 损坏或校验不符，安装会停止并指明文件，不会静默重新下载。

GitHub Release 中的大模型使用 WinRAR 分卷。下载同一模型的全部 `.part*.rar` 到同一目录，用 WinRAR 或 7-Zip 打开 `part1.rar` 并解压；把得到的完整模型 ZIP 放进项目根目录的 `model-packs`。Setup 和网页只导入解压后的 ZIP，不直接读取 RAR 分卷。

Advanced 的四个独立包为 Parakeet 日语（含 Windows CrispASR）、IndexTTS2 checkpoints、Kotoba-Whisper v2.2 和 Faster-Whisper large-v2。

高级用户如需从命令行跳过 Recommended 或更高档位中的 IndexTTS2：

```powershell
./scripts/windows/setup.ps1 -Profile Recommended -SkipRecommendedTTS
```

```bash
ASMR_DUBBER_SKIP_RECOMMENDED_TTS=1 bash scripts/linux/setup.sh Recommended
```

## 使用

1. 在“设置 → 设备与模型”检查后端状态。需要 Parakeet 时，在此安装或运行平台安装脚本。
2. 在“设置 → 翻译设置”选择服务、填写 API Key 并保存。
3. 上传日语音频，点击“① 新建并识别 + 翻译”。
4. 检查日文、中文、启用状态和时间轴，点击“② 保存表格”。
5. 试听并保存一条清晰的项目参考句；IndexTTS2 可在 TTS 设置中分别选择音色和情绪来源。
6. 点击“③ 逐句克隆 + 混音”。

## 推荐后端

| 类型 | 后端 | 适用情况 |
|---|---|---|
| ASR | Parakeet CTC 1.1B JA GAL | 默认；日语专用；CPU/CUDA |
| ASR | Parakeet TDT/CTC 0.6B JA | 同步安装；低资源或对照选项 |
| ASR | Kotoba-Whisper v2.2 | 日语专用 Whisper |
| ASR | Qwen3-ASR 1.7B/0.6B | CUDA；ForcedAligner 时间戳 |
| ASR | Faster-Whisper large-v2 | 日语默认；CPU/CUDA |
| TTS | IndexTTS2 | 推荐本地零样本音色克隆；NVIDIA CUDA |

在“设置 → 设备与模型”中可逐项安装。命令行等价命令：

```powershell
./scripts/windows/install-parakeet.ps1 -Variant Auto
./scripts/windows/install-indextts2.ps1
./scripts/windows/run-cli.ps1 install-backend kotoba_whisper
./scripts/windows/run-cli.ps1 install-backend faster_whisper
```

```bash
bash scripts/linux/install-parakeet.sh
bash scripts/linux/install-indextts2.sh
bash scripts/linux/run-cli.sh install-backend kotoba_whisper
bash scripts/linux/run-cli.sh install-backend faster_whisper
```

其他 ASR/TTS 作为兼容或扩展接口保留。能力、依赖和服务协议见[后端指南](docs/BACKENDS.md)。

## 项目与输出

输入文件会按原字节保存，并记录 SHA-256：

```text
<project>/
├── project.json
├── source.<ext>
├── analysis/
│   ├── asr_16k_mono.wav
│   ├── asr_candidates.json
│   └── asr_review.json
├── references/
├── chinese/
├── mix/
├── output/
│   └── <source>__ja-zh__<tts-backend>-<model>-<reference-mode>.wav
├── exports/
└── performance.json
```

最终文件为浏览器兼容的 24-bit PCM WAV。混音会解码原轨并与中文轨逐样本相加；逐句生成结果和已经完成的阶段会被缓存，重新运行只处理失效部分。

## 便携目录

默认持久文件均位于仓库中的 `.asmr-dubber`：

```text
.asmr-dubber/
├── bootstrap/   # uv
├── runtimes/    # Python、FFmpeg、隔离后端
├── venv/        # 主应用环境
├── models/
├── cache/
├── config/
├── projects/
└── temp/
```

停止程序后删除整个仓库目录即可卸载。

## 命令行

```powershell
./scripts/windows/run-cli.ps1 doctor --no-network
./scripts/windows/run-cli.ps1 --help
```

```bash
bash scripts/linux/run-cli.sh doctor --no-network
bash scripts/linux/run-cli.sh --help
```

命令行提供 `create`、`analyze`、`translate`、`synthesize`、`mix`、`set-timing`、
`install-backend` 和 `verify-asr`。

进阶内容见[配置指南](docs/CONFIGURATION.md)、[后端指南](docs/BACKENDS.md)和[排障指南](docs/TROUBLESHOOTING.md)。

## 许可与使用

项目代码采用 [MIT License](LICENSE)。模型和外部服务适用各自条款；IndexTTS2 使用 bilibiliModel Use License。详见[第三方软件、模型与许可证说明](docs/THIRD_PARTY_NOTICES.md)。

仅克隆已获授权的声音，并明确标记合成内容。
