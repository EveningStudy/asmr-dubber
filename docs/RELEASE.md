中文 | [English](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.0/docs/en/RELEASE.md)

# ASMR Dubber 2.0.0

界面整个重做了，安装方式也变了。

## 新的界面

- 左侧四个入口：项目、批量、模型、设置。
- 项目里按四步走：识别、翻译、配音、导出。左边是句子表格，右边只显示当前这一步的设置。
- 所有参数都还在，平时折叠着，需要时展开。
- 在项目里改的设置只影响这个项目；“设置 → 新项目默认值”只影响以后新建的项目。不再有“设置保存范围”。
- 改了就自动保存，没有保存按钮。

## 新的安装方式

- 不再需要 `ASMR-Dubber-Setup.exe`，也不用选安装方案。解压后直接运行 `ASMR-Dubber.exe`。
- 模型在界面的“模型”页按需下载，支持暂停和断点续传。
- 界面左下角会提示有没有新版本，点一下可以自动下载并安装。

## 其他

- 上传的源文件用完后会自动清理，不再占用磁盘。
- “全部重新翻译”会先确认，避免误覆盖手动改过的译文。
- 文档全部重写，配了新界面的截图。

## 下载与升级

Windows 下载 `ASMR-Dubber-windows-portable-v2.0.0.zip`，完整解压到一个路径短的文件夹，运行 `ASMR-Dubber.exe`。

从 1.x 升级：先关掉程序，删除旧文件夹里的 `src` 文件夹和 `ASMR-Dubber-Setup.exe`，再把新版本的文件全部覆盖进去。`.asmr-dubber` 文件夹要保留，里面是你的项目、模型和设置。升级前建议先备份它。

Linux x86_64 在源码根目录运行 `bash scripts/linux/setup.sh 基础`，之后运行 `bash scripts/linux/run-ui.sh`。

[使用手册](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.0/docs/USER_GUIDE.md) · [安装与模型](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.0/docs/INSTALLATION.md)
