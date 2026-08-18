# ASMR Dubber 1.1.1

## 修复

- 修复 Windows 全新安装时，在主程序依赖尚未安装前提前检查 `httpx`，导致基础、推荐或进阶安装中止的问题。
- Windows 便携包内置基础应用、Web UI 和在线 API 客户端的锁定依赖。首次运行 Setup 时会优先离线安装这些小型依赖，Python、本地模型和大型运行环境仍按原有策略优先从 ModelScope 获取。
- 调整依赖检查顺序：先完成主程序安装，再统一检查 Edge TTS 与在线服务客户端；重复运行 Setup 会复用已经完整的环境。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从 1.1.0 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.1.1 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。若 1.1.0 的 Setup 曾在“在线/API 客户端安装后仍不完整”处失败，覆盖后重新运行 Setup 即可，已下载内容会继续复用。

### Linux

下载本 Release 的源码压缩包并解压，然后运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

项目、模型、缓存、设置和密钥默认保存在程序目录中。API Key 以明文位于 `.asmr-dubber/config/secrets.json`。
