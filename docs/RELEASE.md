# ASMR Dubber 1.0.1

本版本改进批量处理中的参考音频选择，不改变模型、依赖包或下载来源。

## 相比 1.0.0 的变化

- 批量任务完成识别和翻译、准备进入配音时，会在对应队列任务中提示选择参考音频。
- 参考音频选择器默认折叠，可以试听并选择项目内片段，也可以为当前作品导入外部音频。
- 不手动选择时仍使用原有的自动推荐逻辑。默认等待 60 秒，可在“设置 → 自动处理”中修改或关闭等待。
- 参考音频可选时会显示页面内提醒。浏览器阻止提醒、页面关闭或通知失败不会中断后台任务；等待结束后程序会自行继续。
- 已经保存参考音频的项目会直接复用；分轨任务继续共用同一份作品参考音频。

## 下载

### Windows

从本 Release 的 Assets 下载 `ASMR-Dubber-windows-portable.zip`：

1. 在 Windows 设置中启用长路径；
2. 把压缩包完整解压到短且可写的目录，例如 `D:\Apps\ASMR-Dubber`；
3. 首次使用运行 `ASMR-Dubber-Setup.exe`，选择基础、推荐或进阶；
4. 安装完成后运行 `ASMR-Dubber.exe`。

Windows 10/11 不要求预装 Python、Git、FFmpeg、CUDA Toolkit 或 PowerShell 7。

### 从 1.0.0 升级

关闭 ASMR Dubber 和正在执行的任务，把 1.0.1 压缩包解压并覆盖旧程序文件，同时保留原有 `.asmr-dubber` 文件夹。本版本没有更新模型或依赖，覆盖后可以直接启动；如果运行环境曾被手工修改，再运行一次 Setup 修复即可。

### Linux

下载本 Release 的源码压缩包并解压，然后运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

项目、模型、缓存、设置和密钥仍默认保存在程序目录中。API Key 以明文位于 `.asmr-dubber/config/secrets.json`。
