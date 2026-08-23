# ASMR Dubber 1.2.0

## 相比 1.1.3

- Linux/WSL 的安装、CLI 和 AutoFlow 流程已补齐，并覆盖无头服务器使用方式；
- Linux Parakeet 所需的 OpenBLAS、CrispASR 标点模型和 VAD 模型已加入 Portable Mirror；
- 没有打开项目时，网页操作会直接提示“请先新建或打开项目”；
- 测试改为以真实项目流程、媒体输出、CLI 和 Linux 脚本为主要验收内容。

## 修复

- Windows 下载器会先检测系统 `curl.exe` 是否支持 `--retry-all-errors`，旧版 curl 不再因为无法识别该参数而中止下载。
- IndexTTS2 单独安装或修复时优先复用推荐档依赖包，只导入所需运行环境，不覆盖正在使用的主程序环境；Python 3.11、FFmpeg 和 checkpoints 的 ModelScope 路径已补齐并重新校验。
- 基础档明确安装并检查 Edge TTS 与 `httpx`，避免安装结束后才报告在线客户端不完整。
- 切回 IndexTTS2 时会在检测到 NVIDIA GPU 后恢复 CUDA；CPU 仍可手动选择，并明确提示速度较慢。IndexTTS2 任务日志会记录实际使用的设备和精度。
- 移除未发布 wheelhouse 的无效下载地址，安装器不再先请求不存在的文件；CrispASR、Parakeet 原始模型和 FFmpeg 镜像文件已按固定大小与 SHA-256 配置。

## Windows SmartScreen

首次运行未签名的启动器时，Windows 可能显示 SmartScreen 提示。确认文件来自本项目的 GitHub Release 后，点击“更多信息”，再点击“仍要运行”。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从 1.1.3 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.2.0 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。原本运行正常的环境通常不需要重新安装；Linux 用户建议重新运行一次对应档位的安装脚本，让新增的 OpenBLAS 和 CrispASR 文件完成检查。

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

Parakeet 和 IndexTTS2 已在 Ubuntu 24.04/WSL2 的 NVIDIA 环境验证；其它模型请以安装后的后端检测结果为准。

项目、模型、缓存、设置和密钥默认保存在程序目录中。API Key 以明文位于 `.asmr-dubber/config/secrets.json`。
