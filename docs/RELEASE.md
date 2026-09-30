中文 | [English](https://github.com/EveningStudy/asmr-dubber/blob/v1.6.2/docs/en/RELEASE.md)

# ASMR Dubber 1.6.2

## 更新

- 修复批量人声分离时请求文件短暂被占用导致 worker 退出的问题：请求读取与删除增加有限重试，避免重复处理分块，并校验返回的音频文件。
- 保留已完成的分离分块；升级后重启程序，再重试失败任务。

## 1.6.1 功能回顾

- 人声分离在一次任务内只加载一次模型，保留分块、取消和断点恢复。
- 双语版与替换配音版复用中文 RTF 音轨；音频、时间轴和相关参数变化时重新计算。各版本直接写入独立目录，减少中转副本。
- 新增 **设置 → 存储与清理**：先扫描、再选择项目并确认。区分安全缓存、可重建缓存和分离结果；保护项目、原素材、逐句配音与成品，跳过运行中或扫描后发生变化的文件。
- 支持日语、英语、中文 ASR 输入及中文／英文配音目标；具体语言能力以所选后端为准。
- 批量处理支持双语成品、替换配音或两者，同时保留原声与纯字幕选项，沿用 RTF、分离和混音设置。
- 修复批量 IndexTTS 版本与外部音色参考设置传递；等待参考选择时可修改时间和原文。
- 同步中英文说明，补充 README 演示流程与缓存清理指南。

## 下载与升级

Windows 下载 `ASMR-Dubber-windows-portable-v1.6.2.zip`，完整解压后运行 `ASMR-Dubber-Setup.exe`，再运行 `ASMR-Dubber.exe`。请启用长路径并使用短且可写的目录。

Linux x86_64 在源码根目录运行 `bash scripts/linux/setup.sh 推荐`，之后运行 `bash scripts/linux/run-ui.sh`。

升级前停止任务并备份项目、配置与成品，保留 `.asmr-dubber` 和外部项目目录。更新后重启程序，仅刷新网页不会加载新代码。旧项目缓存不会自动删除；清理需手动确认。

[使用指南](https://github.com/EveningStudy/asmr-dubber/blob/v1.6.2/docs/USER_GUIDE.md) · [安装指南](https://github.com/EveningStudy/asmr-dubber/blob/v1.6.2/docs/INSTALLATION.md)

## 验证范围

已验证本机分离模型的多分块进程复用、缓存命中、短样本双版本混音及字幕视频，以及浏览器扫描／确认清理流程。未据此承诺长任务提速倍数、所有云端服务或所有硬件的效果。
