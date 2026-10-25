中文 | [English](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/en/RELEASE.md)

# ASMR Dubber 2.1.0

## 新增与改进

- 加入 MOSS-Transcribe-Diarize 0.9B，支持日语、英语、中文转写和分块内说话人标记；提供完整推理参数。使用独立运行环境，模型优先从项目维护者的 ModelScope 镜像下载，失败后回退官方源。默认识别模型仍为 Parakeet。
- 批量添加、编辑作品时恢复封面预览；可保存默认成品，默认输出“完全配音版 + 双语版”。完全配音版会为该任务自动启用人声分离，需要先安装分离模型。
- 批量支持整队共用外部参考音频，运行中也能应用到当前及剩余等待作品；参考副本和队列选择会保存，方便无人值守处理。
- 最大自动加速默认从 1.8× 调整为 1.4×。完全配音版音量控制直接显示，默认跟随对应原声响度、相对 0 dB、额外增益 0 dB，仍受峰值保护和响度范围限制。已有项目与已保存的参数保留原值。
- 修复 DeepSeek 密钥保存后的首页勾选状态；提示支持手动关闭并自动收起，避免重复弹出；语言切换按钮改为等宽。
- 新增应用、浏览器及 Windows 启动器图标，更新中英文 README 和参数文档。

## 下载与升级

Windows 下载 `ASMR-Dubber-windows-portable-v2.1.0.zip`，完整解压到一个路径短的文件夹，运行 `ASMR-Dubber.exe`。

从 2.0.x 升级：先关闭程序并备份，将新版本文件覆盖到原程序目录。保留 `.asmr-dubber` 文件夹，其中包含项目、模型和设置。

从 1.x 升级：先关掉程序，删除旧文件夹里的 `src` 文件夹和 `ASMR-Dubber-Setup.exe`，再覆盖新版本文件。保留并备份 `.asmr-dubber` 文件夹。

[批量处理](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/BATCH.md) · [MOSS 使用说明](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/MOSS_TRANSCRIBE_DIARIZE.md) · [完整参数](https://github.com/EveningStudy/asmr-dubber/blob/v2.1.0/docs/PARAMETERS.md)
