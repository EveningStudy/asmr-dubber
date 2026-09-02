# ASMR Dubber 1.3.1

## 相比 1.3.0

- 批量页会扫描整个作品文件夹，并对只有视频、没有音频的目录给出专门提示；
- 音轨可以绑定 TXT、SRT、VTT、ASS/SSA 或 LRC，并选择沿用字幕时间轴或运行 ASR 重新定时；
- 台本重新定时按顺序使用原始台本文字，同一片段不会重复映射；中文台本只对未匹配句子继续翻译；
- 批量任务完成后直接列出 SRT、LRC 路径，并可打开输出目录；
- IndexTTS2 的当前句参考明显无效时回退到项目统一参考，过长中文台本会在日志中提示检查；
- 设置页统一为一个保存按钮，API 后端增加当前使用状态、连接测试和请求日志；
- Setup 的安装后导入检查不再把已经成功完成的安装误判为失败。

本版本没有修改模型文件、依赖版本或下载源。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable-v1.3.1.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从旧版本升级

关闭 ASMR Dubber 和正在执行的任务，把 1.3.1 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。本版本没有增加本地运行依赖，原本运行正常的环境不需要重新执行 Setup。

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
