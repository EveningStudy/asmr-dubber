# 更新记录

本项目使用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式和语义化版本。

## [Unreleased]

### Fixed

- 设置保存后直接更新新页面默认值，不再在刷新时排队重放全部设置控件。
- Parakeet 固定使用兼容的 CrispASR 参数、Silero VAD、便携缓存和可诊断错误输出。
- 隐藏 Gradio/Starlette 已知且无操作意义的 422 常量弃用警告。

### Removed

- 精简 README 中重复的开发和网络说明。

## [0.2.1] - 2026-07-24

### Changed

- 安装配置调整为 Core、Recommended、Advanced 和 Full 四档；Advanced 增加
  Kotoba-Whisper v2.2 与 Faster-Whisper large-v2。
- Recommended 聚焦 Parakeet 与 IndexTTS2；安装器同时显示预计占用和建议预留空间。

### Added

- 新增带文件级 SHA-256 校验的离线模型包；Setup 可按档位自动导入，
  设备与模型页也可手动扫描。

### Fixed

- ASR/TTS 后端安装日志改为实时流式输出，并使用独立队列，不再长时间停在转圈状态。
- 已完整安装的 Parakeet 会在本地校验后直接复用，不再重复联网下载。

## [0.4.0] - 2026-07-23

### Added

- Parakeet CTC 1.1B JA GAL 成为默认 ASR；安装器同时准备可切换的 0.6B TDT/CTC。
- 新增 Kotoba-Whisper v2.2、Faster-Whisper large-v2 和多 ASR 证据校对。
- 新增可选温和降噪和实验性人声分离；处理范围限于 ASR 分析副本。
- IndexTTS2 成为新项目默认 TTS，并提供 Windows/Linux 隔离安装器。
- 完成音频采用可播放的 24-bit PCM WAV，并按 TTS 后端、模型和参考模式命名。

### Changed

- 中文默认提前量改为最多 5 秒且不超过对应日语句长的 50%。
- 翻译默认过滤纯笑声、呼吸、喘息、呻吟、亲吻声、拟声和无意义重复音。
- 源文件复制与 SHA-256 合并为一次顺序读取；混音不再生成重复的完整临时轨。
- 设置页只显示当前供应商或后端支持的参数。

### Fixed

- 长时间 ASR、TTS 和混音阶段持续报告确定进度。
- Parakeet CTC 1.1B 使用 token 时间戳切句，不再把完整录音合并为一行。
- 项目位于 Downloads 等自定义目录时，参考句仍可试听。
- Kotoba-Whisper v2.2 使用便携 FFmpeg；v2.0-faster 改用原生段级时间戳。
- DeepSeek 输出过长时自动缩小批次并保留成功结果。
- 修复 Gradio 完成音频控件的播放和布局问题。

### Security

- Gradio 文件白名单不再包含配置和密钥目录。
- 项目源文件、TTS 缓存和混音输入统一拒绝路径越界。

## [0.3.1] - 2026-07-23

### Fixed

- Qwen3-ASR 按低能量边界分段识别，并在每段完成后更新进度。

## [0.3.0] - 2026-07-23

### Added

- Windows 和 Linux 便携运行模式；Python、FFmpeg、模型、缓存、配置、密钥、项目和临时文件位于
  `.asmr-dubber`。

## [0.2.0] - 2026-07-23

### Added

- Windows 10/11 和 64 位 Linux 安装、启动及私有运行时管理。
- 分层设置、后端注册表、多翻译供应商、外部参考和可恢复逐句缓存。
- Qwen/Whisper/FunASR、VoxCPM2/IndexTTS2 及多种 HTTP TTS 适配器。

### Fixed

- 短句中文开始时间不会早于对应日语句开始。
- 固定统一参考和随机种子，降低逐句音色漂移。

## [0.1.0]

### Added

- WSL 原型：Qwen3-ASR、DeepSeek、VoxCPM 和逐句混音。
