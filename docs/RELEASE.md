# ASMR Dubber 1.2.1

## 相比 1.2.0

- 增加 IndexTTS2 云端/自建 API，可上传音色参考和可选情绪参考；
- 增加兼容 OpenAI `/v1/audio/transcriptions` 的通用 ASR API；
- 增加兼容 OpenAI `/v1/audio/speech` 的通用 TTS API；
- 翻译增加商汤 SenseNova，并补齐百炼、豆包、OpenAI 和 Gemini 的思考模式控制；
- ASR、翻译和 TTS 的 API 设置可以填写服务商要求的附加 JSON 参数。

## 修复

- 豆包 Seed 2.0 翻译使用 `reasoning_effort=minimal`，避免结构化翻译默认进入深度思考；
- 商汤结构化请求显式使用 `reasoning_effort=none`，并兼容不接受 JSON Mode 的接口；
- API 地址、附加参数、鉴权、音频响应和错误信息使用统一校验；
- 通用 API 和云端 IndexTTS2 的设置会参与缓存键，切换接口或参数后不会误用旧音频。

本版本没有修改本地模型和依赖的下载流程。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从 1.2.0 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.2.1 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。本版本没有增加本地运行依赖，原本运行正常的环境不需要重新执行 Setup。

### Linux

支持 64 位 x86_64 Linux；Ubuntu 24.04 和 WSL2 是当前验证过的环境。安装脚本、网页界面和无头命令行都使用程序目录内的运行时；ARM64、macOS 和其它架构不在支持范围内。

下载本 Release 的源码压缩包并解压，然后运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

无桌面服务器可使用：

```bash
bash scripts/linux/run-cli.sh doctor --no-network
bash scripts/linux/run-cli.sh --help
```

项目、模型、缓存、设置和密钥默认保存在程序目录中。API Key 以明文位于 `.asmr-dubber/config/secrets.json`。
