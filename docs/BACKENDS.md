# 后端指南

后端注册表定义模型、设备、运行方式、参考要求、安装器和支持等级。后端失败时不会静默切换模型。

支持等级：

- **已验证**：完成真实模型加载和输出验收。
- **支持**：适配器和自动化测试完整，硬件组合仍需用户验证。
- **实验性**：上游接口或依赖变化较快。
- **社区适配**：提供通用接口，依赖社区维护。

## 推荐 ASR

### Parakeet 日语

默认使用 `parakeet-ctc-1.1b-ja` 的 GAL checkpoint。安装器同时准备 1.1B 和
`parakeet-tdt_ctc-0.6b-ja`，设置中可直接切换。

| 模型 | 用途 |
|---|---|
| Parakeet CTC 1.1B JA GAL | 默认；质量优先 |
| Parakeet TDT/CTC 0.6B JA | 低资源或对照；默认使用 TDT 解码 |

两者通过固定版本 CrispASR F16 运行时执行，不向主 Python 安装 NVIDIA NeMo。1.1B 的 token时间戳会按标点、停顿和最长句长切分，不会把完整录音合并为一行。长音频默认使用运行时的流式编码；“分块上限”保持 `0`。

```powershell
./scripts/windows/install-parakeet.ps1 -Variant Auto
```

```bash
bash scripts/linux/install-parakeet.sh
```

### Kotoba-Whisper

- `kotoba-whisper-v2.2`：默认 Kotoba 模型，15 秒分块。
- `kotoba-whisper-v2.1`：兼容旧项目。
- `kotoba-whisper-v2.0-faster`：在 Faster-Whisper 后端中使用。

应用只使用 ASR 核心，不启用需要 gated pyannote 模型的说话人分离。
内置 Transformers 模型使用固定 Hub revision。自定义模型需先下载到本地目录再选择，应用不会让兼容运行时直接加载任意远程仓库。

### Faster-Whisper

日语默认 `large-v2`。GPU 使用 `float16` 或 `int8_float16`；CPU 使用 `int8`。VAD 默认关闭。`large-v3` 系列保留为可选模型。


### Qwen3-ASR

Qwen3-ASR 先生成完整转写，再由 ForcedAligner 计算词级时间戳。提供 1.7B 和 0.6B，当前适配面向CUDA。

## 多 ASR 校对

开启后，各 ASR 串行执行并在下一模型加载前释放。候选按时间窗口对齐，LLM 必须引用候选`evidence_；程序只从被引用的真实时间戳计算最终边界。背景信息只用于专名和上下文消歧。

输出：

```text
analysis/asr_candidates.json
analysis/asr_review.json
```

校对使用“翻译设置”中的 DeepSeek、OpenAI、Claude、Gemini 或 OpenAI-compatible 模型。DeepL、Google Cloud Translation 和 Microsoft Translator 不能执行此步骤。

## 其他 ASR

| 后端 | 接口 |
|---|---|
| OpenAI Whisper | 官方 Python API，词级时间戳 |
| WhisperX | Whisper + 日语对齐模型 |
| FunASR / SenseVoice | `sentence_info`、VAD 和标点模型 |
| OpenAI-compatible ASR | `/v1/audio/transcriptions`，要求 `verbose_json` words/segments |

## 推荐 TTS

### IndexTTS2

IndexTTS2 是默认 TTS。

```powershell
./scripts/windows/install-indextts2.ps1
```

```bash
bash scripts/linux/install-indextts2.sh
```

参考音频不需要转写。默认使用 FP16 和确定性采样。IndexTTS2 可分别选择音色参考和
情绪参考：默认用用户选中的项目参考句保持音色一致，并用每句话对应的日文原句提供情绪。
两者也可改为项目参考句、逐句原句、外部音频等来源；文本情绪仅在选择该来源时启用。

## 其他 TTS

| 后端 | 接口与限制 |
|---|---|
| VoxCPM2 | 已验证兼容后端；CUDA；支持 VoxCPM 专属参考模式 |
| Qwen3-TTS | 官方 Voice Clone API；高质量模式需要准确参考文本 |
| GPT-SoVITS | 官方 `api_v2.py` `/tts`；参考路径必须对服务端可见 |
| CosyVoice | FastAPI；zero-shot 或 cross-lingual |
| F5-TTS | `f5-tts_infer-cli` |
| Fish Speech | `/v1/tts` 兼容接口 |
| XTTS-v2 | Coqui Python API；依赖兼容范围较窄 |


## 参考模式

- **统一声纹，仅音色参考**：所有中文使用同一参考。
- **逐句参考**：每句使用对应日语片段；短句、气声和音效会增加声纹漂移。
- **统一声纹 + 逐句语气**、**Hi-Fi**：仅 VoxCPM2 的实验模式。

这些通用参考模式不用于 IndexTTS2；它在自己的设置中分别管理音色和情绪来源。选择外部参考后，
参考模式不再改变音色来源。需要参考文本的后端必须提供与音频准确对应的文本。

## 翻译

LLM 翻译使用严格 JSON：每个输入 ID 恰好返回一项并保持顺序。纯非语言内容返回空 `zh`，该句随即停用配音。DeepSeek 长输出会自动缩小批次并保存已经成功的结果。

支持：

- DeepSeek、OpenAI、Anthropic Claude、Google Gemini；
- 本地或自定义 OpenAI-compatible 服务；
- DeepL、Google Cloud Translation、Microsoft Translator。

机器翻译 API 不使用自定义 Prompt，也不支持多 ASR 校对。
