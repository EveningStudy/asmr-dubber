# 后端指南

ASMR Dubber 的本地语音范围刻意保持精简：ASR（语音识别）只接入 Parakeet、Kotoba-Whisper 和 Faster-Whisper 三个系列；TTS（语音合成）只接入 IndexTTS2 以及三类外部 API。界面、项目校验、安装器和任务分发使用同一份白名单。

任务失败时程序会报告当前后端，不会悄悄改用另一个模型。这样同一项目的结果和资源需求才可复现。

## 一览

### ASR（语音识别）

| 后端 | 设备 | 建议显存 | 安装方案中的模型 | 适合场景 |
|---|---|---:|---|---|
| Parakeet（日语）/ CrispASR | CPU、NVIDIA CUDA | 6 GB | 推荐、进阶 | 默认主识别，日语质量优先 |
| Kotoba-Whisper（日语）| CPU、NVIDIA CUDA | 6 GB | 进阶安装 v2.2 | 日语 Whisper 对照与复核 |
| Faster-Whisper（日语/英语）| CPU、NVIDIA CUDA | 6 GB | 进阶安装 large-v2 | 英语项目唯一的本地 ASR；CPU `int8`、词级时间戳 |

Kotoba-Whisper 约 3 GB 显存、Faster-Whisper 约 2 GB 显存可能装入较小任务，但还要给驱动、音频和中间张量留空间。显存接近下限时保持批大小 1。

### TTS（语音合成）

| 后端 | 运行位置 | 参考文字 | API Key |
|---|---|---|---|
| IndexTTS2 | 本机 NVIDIA CUDA | 不需要 | 不需要 |
| GPT-SoVITS API | 用户管理的本机、容器或远程服务 | 需要准确源文 | 取决于服务端 |
| CosyVoice API | 用户管理的 FastAPI 服务 | 零样本需要；跨语言不需要 | 取决于服务端 |
| Fish Speech / Fish Audio API | 自建或云端服务 | 需要 | 云服务通常需要 |

IndexTTS2 约 6 GB 显存起，10 GB 以上更合适。其它三种后端只在本程序中运行 HTTP 客户端，服务端硬件和模型由用户自行管理。

## 安装方案中的固定模型

“推荐”安装两个 Parakeet 模型；“进阶”安装以下完整组合：

1. Parakeet CTC 1.1B JA GAL；
2. Parakeet TDT/CTC 0.6B JA；
3. `kotoba-tech/kotoba-whisper-v2.2`；
4. `Systran/faster-whisper-large-v2`；
5. `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`；
6. `Qwen/Qwen3-ForcedAligner-0.6B`；
7. IndexTTS2 checkpoints，仅 NVIDIA GPU。

注册表可以识别同系列的若干其它模型 ID，但分档安装不会下载它们。详见[安装指南](INSTALLATION.md)。

## Parakeet 日语

Parakeet 通过固定的 CrispASR F16 运行时执行，不在主 Python 环境中安装 NVIDIA NeMo。

| 模型 ID | 本地文件 | 用途 |
|---|---|---|
| `grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | `parakeet-ctc-1.1b-ja-f16.gguf` | 默认，质量优先 |
| `nvidia/parakeet-tdt_ctc-0.6b-ja` | `parakeet-tdt-0.6b-ja.gguf` | 更省资源，可选 TDT/CTC 解码头 |

长音频默认由主程序先按安静边界切成 120 秒左右的临时片段，范围是 15–600 秒，再一次性交给同一个 CrispASR 模型进程处理；不会为每个片段重新加载模型。1.1B 输出的 token 时间戳会再按标点、停顿和单句最长时间整理成句子。“连续无响应超时”只在进程长期没有任何输出时停止任务，持续收到进度的长音频不会因为总耗时超过该值而中断。任务结束或用户取消后会清理子进程和临时目录。

独立安装或修复：

```powershell
.\scripts\windows\install-parakeet.ps1 -Variant Auto
```

```bash
bash scripts/linux/install-parakeet.sh
```

## Kotoba-Whisper

Kotoba-Whisper 使用 Transformers 和 PyTorch。分档安装准备经过固定 revision 校验的 v2.2；注册表还允许选择同系列的 v2.1 和 v2.0，但这些变体必须先完整下载到本地缓存。

音频默认按 30 秒分块，范围是 5–120 秒。较小分块降低峰值内存，较大分块保留更多上下文。它适合作为 Parakeet 的第二意见，也可以独立作为主识别器。

Kotoba-Whisper 没有在本项目中暴露“后端自带 VAD”选项。需要预处理时使用独立 ASMR VAD，或直接保留完整音频。

## Faster-Whisper

Faster-Whisper 使用 CTranslate2，支持词级时间戳。分档安装固定准备 `Systran/faster-whisper-large-v2`。其它 Faster-Whisper 模型可以从本地目录加载。

常用计算方式：

- NVIDIA GPU：`float16`，显存紧张时尝试 `int8_float16`；
- CPU：`int8`；
- 后端 VAD 默认不启用，只有在设置中明确选择后才处理静音区间。

Windows 的 CTranslate2 CUDA 构建需要其对应的 CUDA 12 BLAS 运行库。安装器把这些 DLL 放在主程序私有环境并只修改当前进程的搜索路径，不要求安装系统级 CUDA Toolkit。

### 使用 large-v3

先在“设置 → 设备与模型”确认 Faster-Whisper 运行环境可用，再把完整的 CTranslate2 模型放到：

```text
.asmr-dubber\models\faster-whisper-large-v3
```

模型可以从[ModelScope](https://modelscope.cn/models/keepitsimple/faster-whisper-large-v3)手动下载，也可以在程序根目录运行：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
& ".\.asmr-dubber\venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir=r'.asmr-dubber\models\faster-whisper-large-v3')"
```

随后在 ASR 设置中选择 Faster-Whisper，并把模型填写为 `.asmr-dubber\models\faster-whisper-large-v3`。NVIDIA GPU 通常使用 `float16`，显存紧张时使用 `int8_float16`；CPU 使用 `int8`。已经打开项目时，保存后还需应用到当前项目。

## VAD、识别和时间戳如何组合

一次识别任务可以看成三段：

```text
原媒体 → 可选 VAD → ASR 文字与初始边界 → 可选 Qwen3 对齐 → 句子表
```

### 后端 VAD

Parakeet/CrispASR 和 Faster-Whisper 可以使用各自的 Silero VAD。它跟随后端运行，设置较少，适合普通语音。ASMR 中的耳语和低响度发声容易靠近阈值，发现漏句时应关闭做对照。

### 日语 ASMR 专用 VAD

`TransWithAI/Whisper-Vad-EncDec-ASMR-onnx` 是独立预处理模型，通过 ONNX Runtime 在 CPU 运行。它读取程序生成的 16 kHz 单声道分析副本，按 30 秒块输出 20 ms 帧级概率，再把保留区间映射回原媒体时间。它可以放在日语 Parakeet、Kotoba-Whisper 或 Faster-Whisper 前面，原文件不会被裁剪或改写。

### Qwen3 ForcedAligner

`Qwen/Qwen3-ForcedAligner-0.6B` 接收任一支持识别器得到的日语或英语，只重算句子起止边界。它不是识别后端，也不参与修改文字。英语项目同样可以使用它，不需要额外的英文模型。

单模型识别可以直接启用；多模型校对也可以把它选为最终时间戳来源。单句对齐失败时保留 ASR 原边界，并把原因写进 `analysis/asr_forced_alignment.json` 或 `analysis/asr_review.json`。

## 多模型交叉校对

多模型模式让已安装的 Parakeet、Kotoba-Whisper 和 Faster-Whisper 依次识别，再由 LLM 在时间窗口内选择证据。它不是简单多数投票：日语专用模型一致、上下文、重复幻觉和候选置信度都会影响判断。

界面只列出本地模型和运行依赖都完整的组合。文字优先来源必须是已安装识别器；最终时间戳可来自某个候选，也可由 Qwen3 ForcedAligner 重算。LLM 只能引用真实候选的 `evidence_ids`，程序再从这些候选计算边界，不接受模型自行编造时间。

审计文件：

```text
analysis/asr_candidates.json
analysis/asr_review.json
```

DeepSeek、OpenAI、Anthropic Claude、Google Gemini 和 OpenAI-compatible LLM 可以校对。DeepL、Google Cloud Translation 和 Microsoft Azure Translator 只能翻译，不能做这一步。

## IndexTTS2

IndexTTS2 安装在 `.asmr-dubber/runtimes/index-tts` 的隔离环境中，避免它的固定依赖与主程序冲突。模型 checkpoints 默认在该目录下，由 Setup 或“设备与模型”准备。

```powershell
.\scripts\windows\install-indextts2.ps1
```

```bash
bash scripts/linux/install-indextts2.sh
```

音色和情绪使用不同参考：

- 音色默认取项目统一参考句；
- 情绪默认取当前源语言句；
- 音色也可取当前句或外部音频；
- 情绪也可取项目参考、音色参考、外部音频或文字描述。

统一音色参考更适合单角色长项目。逐句参考会跟随场景变化，但短句、气声、音效和背景音乐也更容易造成音色漂移。推荐选 5–15 秒、单一说话人、清晰且包含实义语音的参考。

IndexTTS2 使用独立的 bilibili Model Use License，不属于本项目 MIT License。安装和使用前请阅读上游条款。

## GPT-SoVITS API

适配官方 `api_v2.py` 的 `/tts`：

```text
默认地址：http://127.0.0.1:9880
```

高质量克隆需要准确的参考原文。英语项目填写英语原文。请求中的 `ref_audio_path` 是文件路径，不是上传字节；同机服务可以直接读取，Docker 需要把参考目录挂载到一致或可映射的位置，远程服务则需要双方约定可见路径。

程序不会安装或启动 GPT-SoVITS 服务端，也不会判断服务端实际加载了哪个权重。

## CosyVoice API

适配官方 FastAPI runtime：

```text
默认地址：http://127.0.0.1:50000
```

- `zero_shot`：发送参考音频和对应文字；
- `cross_lingual`：只发送参考音频，网页会隐藏无用的参考文字字段。

不同 CosyVoice 发行版的模型名可能不同，模型输入框可以填写服务端实际接受的 ID。

## Fish Speech / Fish Audio API

适配兼容 `/v1/tts` 的自建或云端接口，请求使用 `references(audio + text)` 格式。参考音频按 base64 发送，因此远程服务不需要访问本地路径。云服务通常需要 API Key，保存在便携密钥文件的 `tts:fish_speech` 项下。

Fish API 版本变化较快。出现 4xx 或响应格式错误时，先对照服务端 OpenAPI，确认当前接口仍接受该请求结构。

## 外部 API 并发和缓存

外部 TTS 请求并发范围为 1–8，默认 2。提高并发只会让独立句子同时请求，不会并行修改本地运行环境。服务限流、显存不足或返回不稳定时先降到 1。

逐句缓存键包含后端、模型、中文、参考音频摘要和相关参数。缓存只有在输入完全匹配时复用；程序不会因为文件名相同就把旧声音当成当前结果。

## 翻译服务

LLM 服务使用有界滑动上下文和翻译记忆，并要求每个输入句子 ID 恰好返回一项。普通机器翻译服务按句请求，不使用 LLM Prompt。

| 服务 | 典型用途 |
|---|---|
| DeepSeek、OpenAI、Claude、Gemini | 上下文翻译和多 ASR 校对 |
| OpenAI-compatible | Ollama、LM Studio、vLLM 或自建兼容接口 |
| DeepL | 专业机器翻译 API |
| Google Cloud Translation | Basic v2 逐句翻译 |
| Microsoft Azure Translator | Translator Text v3 逐句翻译 |

云端会收到完成任务所需的文字；外部 TTS 还可能收到参考音频。是否适合发送由用户根据作品、隐私和供应商条款判断。许可证和服务边界见[第三方软件与模型说明](THIRD_PARTY_NOTICES.md)。
