# 贡献指南

感谢你愿意改进 ASMR Dubber。开始写代码前，请先阅读[架构说明](docs/ARCHITECTURE.md)和与你
准备修改的[后端说明](docs/BACKENDS.md)。涉及新模型、安装体积、数据格式或用户流程的工作，
建议先开 Issue 说明目的和维护成本。

一般使用问题请参阅[支持说明](SUPPORT.md)，安全问题按[安全策略](SECURITY.md)私密报告。参与
讨论和贡献即表示同意遵守[社区行为准则](CODE_OF_CONDUCT.md)。

## 开发环境

项目使用 Python 3.12 和 uv。开发环境可以放在仓库内，但不要复用用户的便携运行环境：

```bash
uv sync --locked --extra ui --extra dev
```

确认基本质量门槛：

```bash
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pyright
uv run --no-sync python scripts/verify_modelscope_artifacts.py
uv run --no-sync pytest
```

构建一次 UI 图，能提前发现事件输入/输出数量不匹配：

```bash
uv run --no-sync python -c "from asmr_dubber.ui import build_app; app = build_app(); assert app.blocks"
```

测试如果需要便携目录，使用单独的临时路径，不要读取开发者真实的 `.asmr-dubber`：

```bash
ASMR_DUBBER_HOME="$PWD/.test-home" uv run --no-sync pytest
```

PowerShell 可以写成：

```powershell
$env:ASMR_DUBBER_HOME = Join-Path $PWD '.test-home'
uv run --no-sync pytest
```

## 代码边界

- UI 和 CLI 都通过 `pipeline.py` 执行业务流程；不要在事件处理器里复制模型逻辑；
- 后端能力只在 `model_registry.py` 声明，项目 schema 和适配器引用同一 ID；
- 所有用户文件路径都要验证在允许的目录内，不能直接信任 manifest 字符串；
- 项目、设置、密钥和索引写入使用 `storage.py` 的锁与原子替换；
- 失败时报告选定后端，不要静默换模型、改参数或联网下载；
- 长音频必须分块，批量请求必须有明确并发上限；
- 日志不得包含 API Key、Prompt 正文、私人媒体内容或不必要的完整路径；
- 默认持久状态应能留在程序目录，除非用户明确配置外部路径。

## 支持范围

ASR（语音识别）范围：

- 通用 ASR API（OpenAI-compatible）；
- Parakeet；
- Kotoba-Whisper；
- Faster-Whisper。

TTS（语音合成）范围：

- IndexTTS2 本地后端和 IndexTTS2 API；
- 通用 TTS API（OpenAI-compatible）；
- GPT-SoVITS API；
- CosyVoice API；
- Fish Speech/Fish Audio API；
- Edge TTS；
- 小米 MiMo TTS API；
- MiniMax TTS API。

引入另一个模型系列会增加运行时冲突、模型镜像、硬件验证、缓存语义、许可证和 UI 复杂度。不要只添加
一个能在开发机运行的下拉项。需要扩展范围时，提案应覆盖长期维护和删除条件。

## 后端接入要求

一个完整后端至少应提供：

1. 注册表中的平台、设备、显存、模型、安装方式和执行能力；
2. `ProjectSettings` 中受约束的参数；
3. 统一输入输出的适配器；
4. 不加载模型的可用性检测；
5. 安装、修复、取消和重试行为；
6. 固定 revision、模型完整性和离线包路径；
7. 覆盖全部影响因素的缓存键；
8. 只显示相关字段的 UI；
9. 单元测试和真实短音频烟雾测试；
10. 第三方许可证、模型卡和隐私边界说明。

## 安装器和下载

安装器必须同时支持 Windows PowerShell 5.1 和 PowerShell 7。修改 `.ps1` 后，用 Windows
PowerShell 5.1 解析所有脚本；现有需要兼容 5.1 的文件应保留 UTF-8 BOM。

远程制品遵循不可变合同：文件名、字节数、SHA-256 和包内 manifest 必须匹配。任何制品调整
都要同步检查：

- `mirrors.json`；
- `modelscope-artifacts.lock.json`；
- `model_pack_download.py`；
- Windows 依赖包常量；
- 安装测试和 [ModelScope 制品维护](docs/MODELSCOPE_UPLOADS.md)。

单元测试和 CI 不下载大型模型。下载器测试使用本地 HTTP fixture、短字节串和已知哈希，覆盖
完整文件复用、断点、Range 校验、取消、错误长度和错误摘要。

## 测试重点

按风险选择测试，不要求每个补丁机械增加同一类用例。

### 项目和存储

- schema 验证与受支持 manifest 的兼容性读取；
- revision 冲突、线程/进程锁和超时；
- 写入失败后的正式文件与临时文件状态；
- 路径逃逸、符号链接和损坏 JSON。

### 媒体流程

- VAD 时间映射、分段和时间戳边界；
- 声道布局、响度、峰值和限幅器时间对齐；
- 字幕换行、最短时长、阅读速度和语言 metadata；
- 视频封装回退和输出扩展名。

### 网络和模型

- 超时、4xx/5xx、非音频响应、重试和并发上限；
- ModelScope 路径、大小、哈希和海外源显式开关；
- 只有本地完整模型才进入多 ASR 校对列表；
- 安装和推理的运行时互斥。

### 设置和 UI

- “仅保存默认值”和“保存并应用到当前项目”的不同结果；
- ASR 设置变化后当前结果被标记为待更新；
- 切换服务或后端时只显示相关参数；
- 中文界面的首次缩写写成 ASR（语音识别）、TTS（语音合成）、VAD（语音活动检测）；
- Gradio 事件输入、输出和动态 choices 保持一致。

## 真实模型验证

真实验证只使用开发者已经安装的模型，不在测试时自动下载：

```bash
uv run --no-sync python scripts/smoke_models.py --help
```

记录测试平台、GPU、显存、驱动、模型 revision、计算精度、短音频来源和结果。测试媒体必须有
合法使用权，且不能提交到仓库，除非它明确采用允许再分发的许可证。

## 文档

文档首先写给第一次使用的人：先说明条件和结果，再给命令。避免宣传口号、版本对比和只有作者
才懂的缩写。

- 面向用户的路径使用占位符，不写开发者个人目录；
- 安装方案必须列出实际模型，不用“完整”“高级模型”等含糊词；
- 参数文档应写默认值、适用后端和误调后果；
- 同一事实尽量链接到一个权威页面，减少多处数值漂移；
- 所有相对 Markdown 链接在提交前检查；
- 第三方许可证只引用上游原文，不自行扩大授权范围。

## 提交内容

补丁或 PR 说明应让评审者直接回答：

- 用户遇到什么问题，或要完成什么任务；
- 为什么选择这个实现；
- 哪些数据、缓存、安装或网络行为会受影响；
- 做了哪些自动和手工测试；
- 是否改变第三方制品、许可证或隐私边界。

不要提交 `.asmr-dubber`、模型、媒体、下载缓存、构建目录、日志、API Key 或个人配置。改动
依赖时修改 `pyproject.toml` 后运行 `uv lock`，不要手工编辑生成的锁文件。

## 发布记录

实现和验收完成后再更新 `docs/RELEASE.md`。每个版本都要保留 Linux 长期未维护、当前发布不保证可用的提示；没有重新完成 Linux 安装与功能验证时，不得删除这条提示。
