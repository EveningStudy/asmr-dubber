# 架构说明

本文面向维护者和准备扩展代码的开发者。用户操作请从[使用指南](USER_GUIDE.md)开始。

## 设计约束

ASMR Dubber 的实现围绕几条约束展开：

- **便携目录**：默认持久状态只写入 `<program>/.asmr-dubber`；
- **项目可恢复**：长流程分阶段保存，逐句工作按输入签名缓存；
- **结果可解释**：项目明确记录后端、模型和参数，不做静默模型回退；
- **资源有上限**：长音频分块、翻译上下文有界、HTTP 并发有界；
- **写入可恢复**：项目 revision、跨进程锁、唯一临时文件和原子替换；
- **运行时隔离**：主程序、CrispASR、IndexTTS2 和下载工具不混用系统环境。

这些约束比“少写几行适配代码”优先级更高。新的后端或流程必须保持相同的数据和失败语义。

## 分层

```text
Windows 启动器 / Shell 脚本
                 │
          Web UI / CLI
                 │
   ui_services.py / autoflow/
                 │
           pipeline.py
       ┌─────────┼──────────┐
       │         │          │
   分析/识别   翻译/合成   混音/字幕
       │         │          │
       └─────────┼──────────┘
                 │
        models.py + storage.py
                 │
       项目目录 / 便携数据目录
```

UI 不直接加载模型。`ui_services.py` 负责把网页输入转换成领域对象、刷新项目视图和限制输出路径；实际任务进入 `pipeline.py`。CLI 调用同一套 pipeline，因此网页和脚本具有相同的缓存、持久化和错误行为。

## 模块职责

### 入口和编排

| 模块 | 职责 |
|---|---|
| `ui.py` | 构建 Gradio 组件、动态显示相关设置、注册事件和远程访问认证 |
| `ui_services.py` | 校验表格/上传、创建项目视图、应用全局设置、网页文件暂存 |
| `cli.py` | Typer 命令、终端进度、环境检查和安装入口 |
| `pipeline.py` | 创建、识别、翻译、合成、混音、字幕、导出和性能记录的事务边界 |
| `autoflow/catalog.py` | 扫描作品目录，识别音频版本、附加音轨、带时间轴字幕和背景图 |
| `autoflow/engine.py` | 规范化分轨、建立批量任务、断点续跑并整理音频、视频和字幕成品 |
| `autoflow/ui_services.py` | 把批量页选项转换成不可变任务计划，维护队列和日志视图 |
| `autoflow/ui_components.py` | 音轨和队列卡片、拖动排序及逐项编辑事件 |

### 媒体和模型

| 模块 | 职责 |
|---|---|
| `audio.py` | 媒体探测、分析副本、参考截取、响度处理、混音和视频封装 |
| `asr.py` | 三个识别系列的适配和统一句子输出 |
| `segmentation.py` | token/segment 时间戳整理、标点与停顿切句 |
| `vad.py` | ASMR ONNX VAD、区间压缩、缓存和原时间映射 |
| `forced_alignment.py` | Qwen3 ForcedAligner 句子边界计算 |
| `asr_review.py` | 多模型候选窗口、证据合同和 LLM 裁决 |
| `translation.py` | LLM 与机器翻译请求、分批、上下文和翻译记忆 |
| `tts.py` | 逐句缓存键、参考解析和合成调度 |
| `tts_backends.py` | IndexTTS2 子进程及外部 HTTP API 适配 |
| `voice_reference.py` | 项目统一参考和逐句参考选择 |
| `timing.py` | 中文落点、冲突检测、自动加速倍速和剩余重叠计算 |
| `subtitles.py` | SRT/LRC 文本、时间轴和可读性限制 |
| `filtering.py` | 日语语气词和非实义文本判断 |

### 状态、运行时和下载

| 模块 | 职责 |
|---|---|
| `models.py` | Pydantic 项目 schema、设置验证、加载、revision 保存和兼容性规范化 |
| `user_settings.py` | 新项目默认值、便携路径、明文密钥和外部参考音频 |
| `storage.py` | 线程/进程文件锁、持久化临时文件和原子替换 |
| `model_registry.py` | 后端白名单、设备、模型、安装方式和能力声明 |
| `runtime_manager.py` | 硬件探测、后端状态、安装互斥和模型准备 |
| `model_packs.py` | 离线模型包 manifest、路径安全、文件校验和导入 |
| `model_pack_download.py` | 固定远程模型包、分段/断点下载和 SHA-256 |
| `mirrors.py` | Python 侧镜像策略和固定 snapshot 下载 |
| `platforms.py` | 平台检查、便携路径和第三方子进程私有环境 |
| `environment.py` | FFmpeg、CUDA 和本地模型缓存定位 |
| `performance.py` | 阶段耗时、资源信息和缓存统计 |
| `hashing.py` | 大文件摘要及基于 path/size/mtime 的安全缓存 |
| `constants.py` | schema 版本、默认后端、固定模型 revision 和必需文件合同 |
| `errors.py` | 可展示给 UI/CLI 的领域错误类型 |

PowerShell/Bash 脚本负责在 Python 可用之前引导 uv、managed CPython、依赖和隔离运行时。`mirrors.json` 与 `modelscope-artifacts.lock.json` 是引导阶段的外部制品合同。

Windows 的两个 C# 启动器只负责引导：Setup 建立日志并调用安装脚本；应用启动器检查核心环境、选择空闲端口、启动 PowerShell 子进程并验证产品页面。业务逻辑仍在 Python 中，启动器不会维护另一份后端或设置实现。

## 项目生命周期

### 批量编排

AutoFlow 先把扫描结果固化为带指纹的任务计划，再为合并成品或每条分轨建立普通 ASMR Dubber 项目。每个计划记录源文件大小、修改时间、字幕选择与语言、音轨顺序、输出模式和背景图；源文件变化后不会静默复用旧结果。

完整的带时间轴字幕直接替换识别时间轴。中文字幕使项目进入中文配音稿状态，跳过 ASR 与翻译；日语或英语字幕跳过 ASR，之后仍进入翻译。只有字幕未覆盖任务时才运行识别兜底。分轨加合并模式先完成各分轨项目，再从它们的成品和时间轴生成合并版本，不会为合并版再次运行 ASR 或 TTS。

批量状态、失败记录、共享参考音频和工作目录位于 `.asmr-dubber/autoflow`，源作品目录只接收用户选择的输出子目录。

### 创建

`create_project` 建立唯一项目目录，复制输入媒体，计算 SHA-256，探测音视频流，并把当前全局默认值转换成 `ProjectSettings` 快照。项目从此不依赖上传临时文件。

### 分析

`analyze_project` 的顺序是：

```text
验证源文件
  → 建立 16 kHz 单声道分析副本
  → 可选 VAD 和压缩时间轴
  → 运行主 ASR 或多个 ASR 候选
  → 可选 LLM 交叉校对
  → 可选 Qwen3 时间戳对齐
  → 写句子、审计文件和导出表
```

VAD 的压缩音频只影响识别输入。每个识别边界在进入项目之前映射回原媒体时间；后续参考截取、字幕和混音始终使用原时间轴。

### 翻译

`translate_project` 只提交已启用且中文为空的句子，除非调用方要求强制刷新。成功结果按批写回项目；后续请求失败不会撤销前面已经保存的中文。

翻译上下文是围绕当前批次的滑动窗口，记忆只保留固定数量的已确认对照。这样请求体和进程内存不会随项目长度无限增长。

### 合成

`synthesize_project` 为每句计算缓存键。键覆盖中文、后端、模型、参考音频内容摘要和会改变声音的参数。只有键相同且缓存音频可读时才复用。

外部 API 使用有界线程池；IndexTTS2 和 CrispASR 通过隔离子进程运行。每个音频先写到唯一临时路径，验证时长和格式后再进入正式缓存。

### 混音和字幕

`mix_project` 一次只把一个中文句子波形放进内存，按原媒体采样率构建 RF64 浮点中文克隆音轨。根据 `mix_output_mode`，它可以只保留中文轨、只生成混音成品，或同时保留两者；切换输出模式不需要重新生成逐句 TTS。响度规范化、逐句峰值、stem 峰值和最终峰值分别处理。最终输出为 24-bit WAV；视频封装在可行时复制原视频流和其它媒体流。

时间窗口模式只在当前句超过下一句开始时间时自动加速，并受最大倍速约束；顺延模式不改变语速，而是把发生冲突的下一句移动到上一句结尾。两种方式都由同一个时间计划同时驱动混音、中文轨、字幕和导出表，避免各输出使用不同落点。

`generate_subtitles` 独立于合成阶段。字幕边界可以取原句或中文配音，文本先经过换行、最短时长和阅读速度约束，再原子写入 SRT/LRC。视频字幕优先烧录，失败时尝试软字幕封装。

## 项目数据模型

`project.json` 是项目的权威状态，包含：

- schema 和应用版本；
- revision、创建时间和更新时间；
- 源媒体摘要与流信息；
- 项目设置快照；
- 句子、启用状态、源文/中文正文、时间和逐句缓存引用；
- 当前音频、视频和字幕输出路径。

项目 manifest 使用 `extra="forbid"`，不接受未知顶层字段。设置模型使用受控枚举和范围验证，避免无效后端或极端数值进入执行层。加载时可以在内存中规范化受支持的历史 schema；需要重写 manifest 时，保存路径先在 `backups` 留原始副本。

所有 manifest 路径在访问前都解析为项目内安全路径，拒绝 `..`、绝对路径逃逸和指向项目外的符号链接目标。

## 保存和并发

保存项目时：

1. 获取 `<project>/.project.lock` 的线程内和跨进程独占锁；
2. 读取磁盘 revision；
3. 与内存 revision 不同则抛出 `ProjectConflictError`；
4. 增加 revision 和更新时间；
5. 写入带随机名的同目录临时文件并 `fsync`；
6. 使用 `os.replace` 原子替换；POSIX 上再同步目录项。

失败时恢复内存中的 revision。不同任务不会共享固定 `.tmp` 文件名，因此并发和崩溃不会让半成品看起来像完整输出。

安装和推理另有 `.asmr-dubber/.runtime-install.lock`。网页事件使用同一个运行时任务队列，进程锁提供最终互斥；这防止安装器在模型已经加载时替换 DLL 或 Python 包。

## 便携存储边界

`platforms.portable_home()` 默认定位仓库旁的 `.asmr-dubber`。启动器和运行脚本会设置明确的 `ASMR_DUBBER_HOME`，避免当前工作目录影响数据位置。

第三方子进程不一定遵守本项目目录约定，因此 `isolated_runtime_environment()` 为它们单独设置 `APPDATA`、`LOCALAPPDATA`、XDG 配置、状态和缓存目录。修改只存在于子进程环境，不写系统变量。

API Key 在 `.asmr-dubber/config/secrets.json` 明文保存，这是产品的便携性选择。文件写入与普通设置分离，POSIX 上使用私有权限；它不会进入项目、性能记录或 UI 输出。

## 网页文件边界

项目目录不直接加入 Gradio `allowed_paths`。需要播放或下载的文件由 `ui_services.stage_for_ui` 硬链接或复制到 `.asmr-dubber/temp/ui`，网页只允许读取该目录。暂存名包含源路径、大小和 mtime 摘要，过期文件定期清理。

默认只监听 loopback。绑定非 loopback 地址时强制认证；未设置密码则为当前进程生成随机值。上传大小有明确上限，默认 20 GB。

## 下载和供应链边界

引导制品遵循四层验证：

1. `mirrors.json` 决定允许的源及顺序；
2. `modelscope-artifacts.lock.json` 固定引导制品路径、大小和 SHA-256；
3. `model_pack_download.py` 固定大型模型包文件合同；
4. `model_packs.py` 再验证包内 manifest、相对路径、每个文件大小和哈希。

外部源默认关闭，只有显式环境变量才能进入候选列表。下载器支持 Range、分段状态和中断恢复；正式文件必须经过完整哈希后才交给导入器。

## 性能策略

- Parakeet、Kotoba-Whisper 和 VAD 按块读长音频；
- 混音一次只持有一个中文句子波形；
- 外部文件 SHA-256 按 path/size/mtime 缓存；
- 外部 TTS 使用 1–8 个工作线程的有界池；
- 翻译上下文和记忆有固定句数上限；
- 逐句 TTS 使用内容完整的缓存键；
- 安装和推理互斥，避免峰值资源叠加；
- `performance.json` 记录阶段时间和缓存命中，便于定位瓶颈。

## 扩展后端

增加后端不能只在网页添加一个下拉项。完整接入至少包括：

1. `model_registry.py` 的平台、设备、模型、安装方式和执行能力声明；
2. `ProjectSettings` 白名单和参数验证；
3. ASR 或 TTS 适配器，并返回统一领域对象；
4. `runtime_manager.py` 的可用性检测和安装语义；
5. 模型 revision、缓存键和失败行为；
6. 动态 UI，只显示当前后端使用的参数；
7. 单元测试、无网络安装测试和真实短音频烟雾测试；
8. 第三方许可证、模型卡和服务数据边界说明。

当前产品范围只接受 Parakeet、Kotoba-Whisper、Faster-Whisper 系列，以及注册表中已有的 TTS 适配器。扩大范围前应先讨论持续维护、安装体积、硬件验证和用户界面成本。
