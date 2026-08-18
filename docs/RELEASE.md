# ASMR Dubber 1.1.2

## 修复

- IndexTTS2 的逐句参考过短时，会在源音频范围内自动扩展到安全长度；旧的过短参考缓存会自动重建，外部参考不足 1 秒时会直接给出明确提示。
- 修复“单个作品”完成后，大型 WAV 或 MP4 无法在输出区域播放的问题。音频改用浏览器原生播放器，长音频不再整份载入后才开始播放。
- Web UI 的上传与预览缓存固定保存在程序目录的 `.asmr-dubber/temp`，不再把数 GB 的成品重复复制到系统临时目录。

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

### 从 1.1.1 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.1.2 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。本次没有调整依赖或模型下载，通常不需要重新运行 Setup。

### Linux

下载本 Release 的源码压缩包并解压，然后运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

项目、模型、缓存、设置和密钥默认保存在程序目录中。API Key 以明文位于 `.asmr-dubber/config/secrets.json`。
