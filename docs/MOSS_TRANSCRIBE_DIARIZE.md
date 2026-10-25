中文 | [English](en/MOSS_TRANSCRIBE_DIARIZE.md)

# MOSS-Transcribe-Diarize

## 使用

1. 重启软件，在「模型」页下载 **MOSS-Transcribe-Diarize 0.9B**。本机已经安装并做过 GPU 实测。
2. 在项目「识别」中选择 MOSS；也可以在「设置 → 新项目默认值 → 识别」中选择。
3. 展开「MOSS 参数」。设置自动保存；全局默认值只影响后续新项目。

MOSS 支持日语、英语、中文。默认 ASR 仍为 Parakeet；本次没有用未经日语耳语对比的结果替换全局默认模型。

## 默认配置

这是一组以官方推理方式为基础、兼顾便携设备内存的推荐起点。没有一组参数能保证所有 ASMR 素材都获得最高准确率。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `asr_moss_chunk_seconds` | 300 秒 | 静音附近分块；测试过约 320 秒音频的两块推理。更长分块能保留更多说话人上下文，但增加显存与内存需求。 |
| `asr_moss_dtype` | auto | 支持 BF16 的 CUDA 设备用 BF16，否则 CUDA 用 FP16，CPU 用 FP32。 |
| `asr_moss_attention` | sdpa | Windows 可直接运行；Flash Attention 2 需自行安装兼容的 `flash-attn`，eager 长音频更容易耗尽显存。 |
| `asr_moss_max_new_tokens` | 16384 | 比官方模型配置的 5120 更充足，避免长转写被截断。生成 EOS 后提前结束，不会强制生成到上限。 |
| `asr_moss_max_context` | 131072 | 输入与预留输出的总 Token 预算，不能超过模型上下文。 |
| `asr_moss_do_sample` | false | 官方确定性推理方式；随机采样没有被证明能提高本项目 ASR 准确率。 |
| `asr_moss_temperature` | 1.0 | 仅采样开启时生效。 |
| `asr_moss_top_p` | 1.0 | 仅采样开启时生效。 |
| `asr_moss_top_k` | 50 | 仅采样开启时生效；0 表示关闭 Top K 过滤。 |
| `asr_moss_num_beams` | 1 | 官方贪心方式；增大不保证更准确，会增加资源消耗。 |
| `asr_moss_repetition_penalty` | 1.0 | 不额外惩罚重复，保留音声中的真实重复表达。 |
| `asr_moss_use_cache` | true | 开启 KV 缓存，避免逐 Token 重算。 |
| `asr_moss_seed` | 0 | 控制采样随机性；跨设备、内核不保证逐位一致。 |
| `asr_moss_prompt` | 官方转写指令 | 包含时间戳与说话人格式要求。清空后恢复内置指令。不要随意删除输出格式要求。 |
| `asr_moss_hotwords` | 空 | 追加专名、角色名；只填素材中实际出现的词。 |
| `asr_moss_generation_params` | `{}` | 其他 Transformers `generate` 参数；专用控件对应字段及输入、输出结构保留字段不可在这里覆盖。无效参数会报错。 |
| `asr_moss_timeout_seconds` | 7200 秒 | 整次识别的总超时，包括加载、所有分块与生成。长作品可提高到 86400 秒。 |

分块上限为 5400 秒，代表可配置范围，不代表本机能在该长度完成推理。12 GB 显存建议先保持 300 秒。显存不足时减至 120 或 60 秒。

## 时间轴与说话人

- 使用模型原生的段级时间戳，不伪造词级对齐，也不把整段时长均摊到单词。
- 「所选句子」显示 `c0001:S01` 这样的说话人标签，并随项目保存。
- `c0001` 是分块编号。不同分块的 `S01` **不代表同一个人**；没有实现跨块身份合并。
- MOSS 段级结果不会强行套用 Whisper 的停顿断句/单句最长设置。长句可在句子表中编辑，或单独使用时间对齐。
- 本次没有添加按说话人自动分配不同 TTS 音色，也没有修改现有字幕格式。

## 下载与隔离

权重及配置固定为 `OpenMOSS-Team/MOSS-Transcribe-Diarize@704aa4a9c304e8520be88901e0d1960158ef5b15`。

每个文件按以下顺序下载：

1. [你的 ModelScope 镜像](https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Portable-Mirror/files)：`artifacts/models/moss-transcribe-diarize/<revision>/`。
2. 官方 Hugging Face 固定快照；Apache-2.0 许可证回退到官方源码的固定版本。

下载逐文件验证字节数和 SHA-256；使用 `.partial` 续传，验证后原子替换。损坏文件不会作为完整模型使用。此 MOSS 路径按上述顺序回退，即使旧的全局下载策略选择了「国内镜像」。

MOSS 使用 `.asmr-dubber/runtimes/moss-transcribe-diarize/.venv`，固定 Torch 2.11.0、Transformers 5.6.0，不修改现有 Transformers 4.x 环境。首次安装运行库会使用清华 PyPI 镜像/PyPI 和 uv 自动选择的 Torch 源；**这些运行依赖未上传为 ModelScope 离线依赖包**。权重、分词器、配置和模型代码已上传。

推理仅从本地校验快照加载，取消或超时会结束工作进程，失败结果不会覆盖原句子表。模型页删除功能只删除 MOSS 权重快照，保留运行库便于重装。

## 验证范围

- Windows / RTX 5070 Ti Laptop 12 GB / CUDA 13 / BF16 / SDPA。
- 10.435 秒公开英语语音：成功生成段级转写和说话人标签。
- 320.18 秒重复公开语音：两块、56 段、约 90.11 秒完成，末段结束于 318.87 秒；此项验证分块与时间偏移，不能用来估计自然长音频准确率。
- ModelScope 上传后匿名读回全部 18 个运行文件，逐字节计算 SHA-256，全部匹配。
- 尚未完成日语耳语与现有 Parakeet/Kotoba 的人工标注对比、真实多人说话人评测、Linux/CPU 推理验证。

相关：[全部默认值审查](DEFAULTS_AUDIT_2026-10-16.md) · [验证报告](MOSS_VALIDATION_2026-10-16.md)

上游：[官方源码与推理说明](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) · [模型](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)。
