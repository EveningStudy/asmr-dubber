# 更新记录

## [0.3.0] - 2026-07-24

### Added

- “设备与模型”支持暂停当前依赖或模型下载，重新安装时复用已有缓存。
- 网页安装后端前自动扫描并导入 `model-packs` 中匹配的离线模型包。

### Changed

- Windows Advanced/Full 在 NVIDIA 设备上安装 CUDA PyTorch，Kotoba-Whisper 可直接使用 GPU。
- 统一设置页的中英文字体，修复大写 A 字号不一致。
- 完善 Windows、Linux、分档安装和离线模型包说明。

### Fixed

- 修复 Advanced 安装可能保留 CPU 版 PyTorch，导致 Kotoba-Whisper 报告 CUDA 不可用的问题。
- Kotoba-Whisper 使用已解码音频时不再触发可选 TorchCodec，避免残留的不兼容 DLL 阻断识别。
- 网页安装进程现在可以终止完整子进程树，不再在暂停后继续占用下载或 GPU 资源。

## [0.2.1] - 2026-07-24

### Added

- 新增 Core、Recommended、Advanced 和 Full 四档安装方案。
- 新增带文件级 SHA-256 校验的离线模型包，并支持 Setup 自动导入。

### Fixed

- ASR/TTS 后端安装日志实时输出，已完整安装的 Parakeet 不再重复下载。

## [0.2.0] - 2026-07-23

### Added

- Windows 10/11 和 64 位 Linux 安装与启动。
- 分层设置、后端注册表、多翻译供应商、外部参考和可恢复逐句缓存。
- Parakeet、Kotoba-Whisper、Qwen3-ASR、Faster-Whisper、IndexTTS2、VoxCPM2 等后端适配。
- 多 ASR 交叉校对、逐句响度跟随、离线密钥存储和项目缓存。

### Fixed

- Parakeet 1.1B 按时间戳分句，Kotoba-Whisper 使用段级时间戳。
- DeepSeek 输出过长时自动缩小批次并保留成功结果。
- 修复参考句试听、设置持久化、播放器和长流程进度问题。

## [0.1.0] - 2026-07-22

### Added

- 首个可用版本：日语识别、中文翻译、音色克隆和原轨混音。
