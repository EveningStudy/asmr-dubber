# 命令行参考

网页和命令行调用同一套项目流程。需要批处理、远程终端或精确重跑某个阶段时使用 CLI；日常制作仍建议用网页校对表格。

## 启动方式

Windows：

```powershell
.\scripts\windows\run-cli.ps1 --help
```

Linux：

```bash
bash scripts/linux/run-cli.sh --help
```

使用这些脚本能保证便携 Python、FFmpeg、模型目录和隔离 DLL 路径正确。不要直接调用系统 `python -m asmr_dubber`，除非你正在开发并明确配置了环境。

下面示例用 `<project>` 表示项目目录或其中的 `project.json`。

## 环境检查

```powershell
.\scripts\windows\run-cli.ps1 doctor --no-network
```

`doctor` 检查操作系统、架构、GPU、PyTorch CUDA、FFmpeg、核心包、当前 ASR（语音识别）后端、当前 TTS（语音合成）后端和翻译密钥。去掉 `--no-network` 后还会请求翻译服务基础地址。

某个当前后端未安装时命令返回非零，即使网页核心环境已经可以启动。脚本自动化应检查退出码，不要只搜索输出中的 `OK`。

## 创建项目

```powershell
.\scripts\windows\run-cli.ps1 create 'D:\Media\input.mp4'
```

可选参数：

```text
--projects-root PATH   把项目放到指定目录
--source-language ja|en  指定源语言；英语项目使用现有 Faster-Whisper
--offset-ms INTEGER    设置中文配音整体偏移（毫秒）
--max-speed FLOAT      设置冲突时最大自动加速倍速（1.0–4.0）
```

命令输出新项目的 `project.json` 路径。项目会复制输入媒体，之后可以移动或删除原输入文件。

Linux 示例：

```bash
bash scripts/linux/run-cli.sh create /data/input.wav --projects-root /data/projects
```

## ASR（语音识别）

```powershell
.\scripts\windows\run-cli.ps1 analyze '<project>'
```

项目已有句子时，默认直接保留。强制按项目当前设置重新识别：

```powershell
.\scripts\windows\run-cli.ps1 analyze '<project>' --force
```

VAD（语音活动检测）、主识别器、多模型校对和 Qwen3 时间戳对齐都从项目设置读取。英语项目会自动使用 Faster-Whisper，并把识别语言固定为英语；Parakeet、Kotoba-Whisper 和日语 ASMR VAD 只用于日语。CLI 不接受一组临时后端参数，以免运行结果与 `project.json` 记录不一致；先在网页应用设置，或通过受验证的代码修改项目设置。

识别完成后输出 `exports/transcript.json` 路径。

## 翻译

```powershell
.\scripts\windows\run-cli.ps1 translate '<project>'
```

默认只翻译已启用且中文为空的句子。重新翻译全部已启用句子：

```powershell
.\scripts\windows\run-cli.ps1 translate '<project>' --force
```

服务、模型、基础地址和 Prompt 来自项目设置。API Key 按以下顺序读取：调用方显式提供的值、便携密钥文件、对应环境变量。普通 CLI 没有命令行密钥参数，建议先在网页保存或设置环境变量。

## TTS（语音合成）

```powershell
.\scripts\windows\run-cli.ps1 synthesize '<project>'
```

默认复用缓存键完全匹配的逐句音频。忽略全部中文缓存：

```powershell
.\scripts\windows\run-cli.ps1 synthesize '<project>' --force
```

只重做指定句子，`--sentence` / `-s` 可以重复：

```powershell
.\scripts\windows\run-cli.ps1 synthesize '<project>' `
    --sentence s000001 `
    --sentence s000004
```

句子 ID 以 `project.json` 或 `exports/transcript.csv` 为准。

## 混音

```powershell
.\scripts\windows\run-cli.ps1 mix '<project>'
```

混音要求所有启用且有中文的句子已经具备有效 TTS 文件。命令按照项目设置的 `mix_output_mode` 输出混音成品、中文克隆音轨，或两者。只输出中文轨后再切回包含混音的模式，仍可直接加入原音轨，不必重做 TTS。视频项目只有在生成混音成品时才会输出封装视频。修改中文整体偏移、自动加速上限、响度、声道路由或峰值设置后，只需重新 `mix`，不必重做识别、翻译或合成。

## 修改中文配音排程

```powershell
.\scripts\windows\run-cli.ps1 set-timing '<project>' `
    --offset-ms 500 `
    --mode fit-window `
    --max-speed 1.8
```

至少提供一个参数：

- `--offset-ms`：相对原字幕开始时间的整体偏移，负数提前、正数延后；
- `--mode`：`fit-window` 在冲突时自动加速；`sequential` 等上一句结束后再播放下一句；
- `--max-speed`：`fit-window` 模式下的最大自动加速倍速，范围 1.0–4.0。

命令会使现有混音和字幕视频失效，但保留逐句 TTS 缓存。`fit-window` 达到速度上限后仍放不下的部分允许重叠；`sequential` 不加速，冲突的下一句顺延，输出可能长于原媒体。

## 字幕

```powershell
.\scripts\windows\run-cli.ps1 subtitles '<project>' --language bilingual
```

`--language` 可用值：

| 值 | 输出内容 |
|---|---|
| `bilingual` | 源文 + 中文双语 |
| `zh` | 仅中文 |
| `source` | 仅源文（`ja` 仍可作为兼容别名）|

命令生成 SRT 和 LRC；视频项目还尝试生成字幕视频。字幕时间轴和可读性参数从项目设置读取。

## 安装或修复后端

```powershell
.\scripts\windows\run-cli.ps1 install-backend parakeet_nemo
.\scripts\windows\run-cli.ps1 install-backend kotoba_whisper
.\scripts\windows\run-cli.ps1 install-backend faster_whisper
.\scripts\windows\run-cli.ps1 install-backend indextts2
```

后端 ID 是稳定的项目字段，不是网页显示名称。Edge TTS 随基础依赖安装；MiMo、MiniMax、GPT-SoVITS、CosyVoice 和 Fish API 不通过此命令安装。

安装和本地推理互斥。另一个任务占用运行时锁时，命令会等待或明确超时，不应并行启动多个安装进程。

## 离线模型包

列出根目录 `model-packs` 中发现的 ZIP：

```powershell
.\scripts\windows\run-cli.ps1 list-model-packs
```

导入全部兼容包：

```powershell
.\scripts\windows\run-cli.ps1 import-model-packs --all
```

只导入指定 pack ID：

```powershell
.\scripts\windows\run-cli.ps1 import-model-packs --all `
    --pack-id qwen3-forced-aligner `
    --pack-id whisper-vad-asmr-onnx
```

也可以把一个或多个 ZIP 作为位置参数直接导入。所有方式都会验证内部 manifest 和文件哈希。

从 `mirrors.json` 配置的来源准备一个模型包到本地 inbox：

```powershell
.\scripts\windows\run-cli.ps1 prepare-model-pack qwen3-forced-aligner
```

有效 pack ID：

```text
parakeet-ja-windows
indextts2-checkpoints
kotoba-whisper-v2.2
faster-whisper-large-v2
qwen3-forced-aligner
whisper-vad-asmr-onnx
```

`download-models --backend 进阶语音识别` 是分档安装使用的组合命令，会准备 Kotoba v2.2、Faster-Whisper large-v2、Qwen3 ForcedAligner 和 ASMR VAD。

## 真实验证一个识别器

`verify-asr` 读取一段短音频，真实加载指定模型并打印 JSON，不创建项目、不翻译：

```powershell
.\scripts\windows\run-cli.ps1 verify-asr '.\sample.wav' `
    --backend faster_whisper `
    --model large-v2 `
    --device cuda `
    --compute-type float16
```

Parakeet 0.6B 可额外指定 `--decoder tdt` 或 `--decoder ctc`。CPU Faster-Whisper 通常使用：

```powershell
.\scripts\windows\run-cli.ps1 verify-asr '.\sample.wav' `
    --backend faster_whisper `
    --model large-v2 `
    --device cpu `
    --compute-type int8
```

先用几秒到几十秒、来源明确的日语或英语测试音频；英语项目指定 `--source-language en`。该命令不会自动把未安装模型下载下来。

## 无界面服务器流程

Linux 和 WSL 使用同一套 CLI，不依赖 PowerShell：

```bash
bash scripts/linux/run-cli.sh doctor --no-network
bash scripts/linux/run-cli.sh create /data/input.wav --projects-root /data/projects
bash scripts/linux/run-cli.sh import-transcript /data/projects/<project>/project.json /data/script.srt --kind zh
bash scripts/linux/run-cli.sh run /data/projects/<project>/project.json \
  --start synthesize --stop subtitles
```

`run` 会按顺序执行指定阶段；中文台本项目会自动跳过 ASR（语音识别）和翻译。可用阶段为 `analyze`、`translate`、`synthesize`、`mix` 和 `subtitles`。

Linux 还提供设置命令，适合远程部署时不打开网页：

```bash
bash scripts/linux/run-cli.sh settings show
bash scripts/linux/run-cli.sh settings set tts_backend edge_tts
bash scripts/linux/run-cli.sh settings set-translation-key deepseek --key 'YOUR_KEY'
```

不要把 API Key 写进 shell 历史或提交到服务器镜像；推荐通过交互式输入或服务器密钥管理器注入。

## 启动网页

```powershell
.\scripts\windows\run-cli.ps1 ui --host 127.0.0.1 --port 7860
```

绑定非本机地址会强制认证。可在启动前设置：

```powershell
$env:ASMR_DUBBER_UI_USERNAME = 'asmr'
$env:ASMR_DUBBER_UI_PASSWORD = '使用你自己的强密码'
.\scripts\windows\run-cli.ps1 ui --host 0.0.0.0 --port 7860
```

这仍是开发型本地网页，不建议直接暴露到公网。详细边界见[安全策略](../SECURITY.md)。

Linux/WSL 远程服务器：

```bash
export ASMR_DUBBER_UI_USERNAME=asmr
export ASMR_DUBBER_UI_PASSWORD='使用你自己的强密码'
bash scripts/linux/run-ui.sh --host 0.0.0.0 --port 7860
```

服务器部署建议优先使用 CLI；网页只监听内网地址，并通过 SSH 隧道、反向代理或 VPN 访问。程序不会自动把 WebUI 暴露到公网。

## 退出码和错误

- 成功命令返回 `0`；
- 项目、环境、模型、网络或参数错误返回非零；
- 错误正文写到终端，不把失败伪装成空输出；
- 输出路径只有在相应文件完整写入后才打印。

自动化脚本应检查退出码和输出文件是否存在。不要根据中文进度文字推断任务成功。
