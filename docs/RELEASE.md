中文 | [English](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/en/RELEASE.md)

# ASMR Dubber 2.2.0

## 修复与改进

- 修正批量字幕选项：“只要字幕文件”只导出 SRT/LRC，“原声 + 字幕，不配音”保留原声成品。仅译文、沿用音频原名和标题翻译按设置生效。
- 旧字幕队列按界面选项升级，并隔离旧执行缓存。已有输出会保留，明确重新处理或更换输出目录后才能替换。
- 有实义的句子返回空译文时自动单句重试，仍失败则保留启用并显示错误，保存其他成功译文。可恢复旧版误跳过的句子；勾选后立即更新行颜色和剩余数。
- 批量逐作品显示等待、正在处理、等待参考、完成或失败，保留混合批次的成功状态。历史记录显示运行时间并可隐藏，空队列不再创建失败任务。
- 新批量项目目录带作品名、音轨和任务编号，存储及清理预览显示归属；旧项目从可用检查点恢复显示，保持原路径可续跑。

## 下载与升级

Windows 下载 `ASMR-Dubber-windows-portable-v2.2.0.zip`，完整解压到一个路径短的文件夹，运行 `ASMR-Dubber.exe`。

从 2.x 升级：先关闭程序并备份，将新版本文件覆盖到原程序目录。保留 `.asmr-dubber` 文件夹，其中包含项目、模型和设置。

从 1.x 升级：先关掉程序，删除旧文件夹里的 `src` 文件夹和 `ASMR-Dubber-Setup.exe`，再覆盖新版本文件。保留并备份 `.asmr-dubber` 文件夹。

[批量处理](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/BATCH.md) · [MOSS 使用说明](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/MOSS_TRANSCRIBE_DIARIZE.md) · [完整参数](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.0/docs/PARAMETERS.md)

如果之前通过反选字幕选项绕过 2.1 的映射问题，升级后请重新确认队列的成品选择。
