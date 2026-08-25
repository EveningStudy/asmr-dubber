# ASMR Dubber 1.2.2

## 相比 1.2.1

- 批量处理增加“仅生成字幕”，不会进入参考音频、TTS 或混音阶段；
- 仅字幕的视频任务可以选择是否生成带字幕的原声视频；
- 任务队列会保存并显示处理内容，重新编辑时恢复原选择。

本版本没有修改模型、依赖或下载流程。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从 1.2.1 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.2.2 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。本版本没有增加本地运行依赖，原本运行正常的环境不需要重新执行 Setup。

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
