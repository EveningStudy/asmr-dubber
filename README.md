# ASMR Dubber

ASMR Dubber 在日语原声中加入逐句中文复述。日语时间轴和内容保持不变；中文由所选 TTS 后端使用日语声纹参考生成，并按对应原句的局部响度混入。

支持 Windows 10/11 和 64 位 Linux。

## 安装

项目不包含模型权重。`Recommended` 在 NVIDIA 设备上约需 30 GB 可用空间，具体取决于缓存和后端。

### Windows

双击项目根目录中的 `ASMR-Dubber.exe`。首次运行会打开一个终端，列出 Core、Recommended、Full的详细说明。
以后再次双击同一个 EXE 会直接启动服务并打开浏览器。运行期间终端用于显示；按 `Ctrl+C` 或关闭终端即可停止由它启动的服务。

安装器会在 `.asmr-dubber` 中准备 uv、Python 3.12、FFmpeg、依赖和所选推荐模型，启动器未经代码签名，Windows 首次运行下载的 Release 时可能显示 SmartScreen 来源提示。

### Linux

要求 64 位 Linux、`bash` 和 `curl`。NVIDIA 后端需要可用的驱动。

```bash
bash scripts/linux/setup.sh Recommended
bash scripts/linux/run-ui.sh
```

打开终端显示的 `http://127.0.0.1:7860`。

### 安装配置

| 配置 | 内容 |
|---|---|
| `Core` | 应用、UI 和基础音频依赖；不下载大型 ASR/TTS 权重 |
| `Recommended` | 两款 Parakeet 模型、Kotoba/Faster-Whisper 运行库；NVIDIA 设备安装 IndexTTS2 |
| `Full` | Recommended，另加 Qwen3-ASR/ForcedAligner、VoxCPM2 权重和更多 ASR 运行库 |

配置只决定首次批量准备的内容。安装后仍需在“设置 → 设备与模型”确认状态，并按需下载、安装或修复没有包含在当前配置中的模型。无 NVIDIA GPU 时，安装器跳过 CUDA 模型；可使用Faster-Whisper CPU `int8` 和外部 TTS 服务。

高级用户如需从命令行跳过 Recommended 中的 IndexTTS2：

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
5. 试听并保存一条清晰的统一声纹参考句，或在 TTS 设置中选择外部参考。
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

其他 ASR/TTS 作为兼容或扩展接口保留。能力、依赖和服务协议见
[后端指南](docs/BACKENDS.md)。


## 项目与输出

输入文件会按原字节保存，并记录 SHA-256；

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
