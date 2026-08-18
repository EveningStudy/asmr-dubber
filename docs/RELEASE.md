# ASMR Dubber 1.1.0

## 相比 1.0.1 的变化

- 新增 Edge TTS、小米 MiMo TTS 和 MiniMax TTS。IndexTTS2 未安装或不完整时，新项目默认使用 Edge TTS。
- Edge TTS 无需 API Key，设置页可以直接试听所选音色。
- 翻译新增阿里云百炼和豆包（火山方舟），默认翻译模型调整为 `deepseek-v4-flash`。
- 基础、推荐和进阶安装方案都会安装并检查在线翻译与 TTS 所需客户端；三种方案仍只在本地模型数量上有差异。
- TTS 设置只显示当前后端和模型需要的参数，MiMo 音色克隆、预置音色和文字设计音色分别使用对应配置。
- 修复单个作品完成混音后，文件名包含 `#` 等特殊字符时音频或视频无法在网页中预览的问题。真实输出文件名不会改变。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从 1.0.1 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.1.0 压缩包解压并覆盖旧程序文件，保留原有 `.asmr-dubber` 文件夹。覆盖后运行一次 Setup 补齐 Edge TTS 等基础客户端；已有本地模型会直接复用，不需要重新下载。

### Linux

下载本 Release 的源码压缩包并解压，然后运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

项目、模型、缓存、设置和密钥默认保存在程序目录中。API Key 以明文位于 `.asmr-dubber/config/secrets.json`。
