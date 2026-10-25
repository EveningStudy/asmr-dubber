中文 | [English](en/DEFAULTS_AUDIT_2026-10-16.md)

# 默认设置审查 · 2026-10-16

覆盖当前设置目录的 **218 个字段**，同时读取当前全局保存值。没有修改你的全局设置或现有项目。

## 结论

目前默认值是一组可用、偏稳妥的起点，不能称为所有素材上的“最好”。本次只对 MOSS 做了实际 GPU 推理；其余属于代码/参数合同审查与条件性建议，没有对所有模型完成主观音质盲测。

### 最值得先改或试听的项目

| 优先项 | 当前情况 | 建议 |
|---|---|---|
| 翻译模型名 | schema 与保存值都是 `deepseek-v4-flash` | 改为当前官方名称 `deepseek-flash`。旧别名已路由到 V4.1，同一次改名不意味着质量升级。 |
| 空间跟随 | schema 关闭，你保存的已开启 | 双耳 ASMR 保持你当前开启；若调整发行版默认，可考虑按双耳项目启用。 |
| 自动加速 | schema 1.8，你保存 1.7 | 为自然耳语先对比 1.25–1.4；过长译文需缩写或改排期，不保证所有句子仍塞入原窗。 |
| 配音排期 | schema fit_window，你保存 sequential | 学习双语可保留 sequential；替换版先用 fit_window、偏移 0 ms，并抽查重叠。 |
| 人工声纹参考 | schema 等待人工选择，你保存关闭 | 正式作品建议开启；选择干净、连续、同一角色的参考，通常比盲调采样参数更有价值。 |
| 多模型复核 | 功能关闭；你保存的模式为 conservative | 首次开启先改 suggest；可加入 MOSS 形成不同家族证据，先抽查再自动采纳。 |
| Whisper 前文条件 | 默认开启 | 长静音/循环幻觉时对比关闭；关闭也可能降低跨窗文本一致性。 |
| 情绪来源 | 逐句情绪参考；你保存强度 0.55 | 需要表达变化可保留；耳语参考过轻/噪声重时比较统一情绪参考，强度先试 0.4–0.55。 |
| 分离窗口 | MDXC 强制 256 | 和模型原生窗口配置比较后决定；分离不是双语版必开增强。 |
| MOSS 替换默认识别 | 已接入，默认仍为 Parakeet | 先在真实日语耳语上比较错字、漏词、静音幻觉和边界，再决定长期默认。 |

## 当前保存值与 schema 的差异

JSON 仅空格差异不计；路径做了概括。此表包含当前机器偏好，不作为软件发布配置。

| 字段 | schema 默认 | 当前保存值 |
|---|---|---|
| `separation_keep_original` | `none` | `unvoiced` |
| `spatial_rtf_enabled` | `false` | `true` |
| `asr_review_mode` | `suggest` | `conservative` |
| `asr_review_models` | `["parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo","kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2"]` | `["parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo","parakeet_nemo&#124;nvidia/parakeet-tdt_ctc-0.6b-ja","kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2","faster_whisper&#124;large-v2"]` |
| `asr_review_text_priority_model` | `parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | `faster_whisper&#124;large-v2` |
| `asr_review_timestamp_priority_model` | `qwen_forced_aligner&#124;Qwen/Qwen3-ForcedAligner-0.6B` | `faster_whisper&#124;large-v2` |
| `translation_base_url` | 空 | `https://api.deepseek.com` |
| `tts_api_base_url` | `http://127.0.0.1:9880` | 空 |
| `tts_index_emo_alpha` | `0.5` | `0.55` |
| `chinese_dubbing_offset_ms` | `500` | `400` |
| `chinese_max_auto_speed` | `1.8` | `1.7` |
| `chinese_dubbing_timing_mode` | `fit_window` | `sequential` |
| `projects_root` | 空 | 本机资源路径 |
| `tts_device_selection_version` | `0` | `1` |
| `autoflow_timestamp_footer_position` | `after` | `before` |
| `autoflow_reference_wait_enabled` | `true` | `false` |
| `autoflow_timestamp_footer` | `双语音声制作器：BV1f43G6YEov<br>内嵌字幕和配音为本地 AI 生成，内容仅供参考。<br>仅供日语学习，有能力请购买正版支持。` | `双语音声制作器：BV1f43G6YEov` |

## 全字段检查

下表逐项给出默认值和判断。只作用于其他后端的参数，在当前模型中可能不生效；数值更大不等于效果更好。每个字段的类型、范围、可见条件见 [完整参数参考](PARAMETERS.md)。

### 人声分离

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `separation_enabled` | `false` | 双语版默认关闭合理，可保留环境与非语言声；替换版才开启并试听残留。 |
| `separation_backend` | `local` | local 不发送原音频，保持；云端方式需服务可用与用户同意。 |
| `separation_model` | `vocals_mel_band_roformer.ckpt` | Mel-Band RoFormer 是现有起点，不能保证对每种耳语/环境音最佳。 |
| `separation_device` | `cuda` | 本机 CUDA 合理；CPU 主要影响速度。 |
| `separation_chunk_seconds` | `30.0` | 30 秒控制内存；边缘音色不连续时可比较 60 秒，显存成本增加。 |
| `separation_overlap_seconds` | `1.0` | 1 秒是外层前后保留，不等同于模型内部 overlap；边界伪影再试 2 秒。 |
| `separation_timeout_seconds` | `1800.0` | 1800 秒保护上限，按硬件耗时调整。 |
| `separation_use_autocast` | `true` | 保持 true 节省显存；只有实测数值问题再禁用。 |
| `separation_normalization` | `1.0` | 1 为峰值归一化强度，不保证无最终混音削波；保留总峰值保护。 |
| `separation_vocal_stem` | `Vocals` | Vocals 必须与所选模型输出名匹配。 |
| `separation_common_params` | 内置长文本/JSON，见参数表 | 保持不编译、不原生 FP16、不反相；无证据时避免开启会改变音频的处理。 |
| `separation_mdx_params` | `{"hop_length":1024,"segment_size":256,"overlap":0.25,"batch_size":1,"enable_denoise":false}` | batch=1/overlap=0.25 是资源折中；模型不适用时不生效，不能与 MDXC 的 overlap=4 混为同一量纲。 |
| `separation_vr_params` | 内置长文本/JSON，见参数表 | TTA 默认关；可提高计算量做对比，aggression 不应为去残留盲目增大。 |
| `separation_demucs_params` | `{"segment_size":"Default","shifts":2,"overlap":0.25,"segments_enabled":true}` | shifts=2/overlap=0.25 是折中；可试 shifts=4、overlap=0.5，但仅在 Demucs 生效且不保证更好。 |
| `separation_mdxc_params` | `{"segment_size":256,"override_model_segment_size":true,"batch_size":1,"overlap":4,"pitch_shift":0}` | 当前强制 segment_size=256；建议和 override_model_segment_size=false 的模型原生设置做配对试听，再决定默认。 |
| `separation_api_url` | 空 | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `separation_api_model` | 空 | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `separation_api_audio_field` | `audio` | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `separation_api_output_field` | `vocals` | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `separation_api_params` | `{}` | 默认空对象合理；仅按该服务真实 API 添加字段，不能宣称通用最优。 |
| `separation_cloud_consent` | `false` | 默认 false 合理；授权影响数据传输而非音质。 |

### 识别与复核

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `asr_backend` | `parakeet_nemo` | 保留 Parakeet 日语默认；MOSS 可选，尚无本作品集对比证据支持直接替换。 |
| `asr_model` | `grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | GAL 1.1B 日语模型维持原推荐；英语/中文会使用兼容后端。 |
| `asr_batch_size` | `1` | 1 更省显存且减少批处理差异；提高主要改善吞吐，不代表识别更好。 |
| `asr_device` | `cuda` | 本机有 12 GB NVIDIA 显卡，保持 CUDA；无兼容显卡才选 CPU。 |
| `asr_compute_type` | `float16` | 保持 FP16；内存不足才用低精度量化，不把量化当准确率升级。 |
| `asr_beam_size` | `5` | 5 是合理起点，不能保证更大的束搜索更好。 |
| `asr_vad_filter` | `false` | 兼容字段，优先通过 asr_vad_mode 选择；MOSS 不用后端 VAD。 |
| `asr_vad_mode` | `off` | 耳语优先保留 off，避免漏掉轻声；长静音幻觉明显再对比 ASMR VAD。 |
| `asr_vad_min_silence_ms` | `500` | 500 ms 合理；减少时可能切断慢速音声中的停顿。 |
| `asr_asmr_vad_threshold` | `0.5` | 0.5 仅为起点；漏掉耳语可试 0.3–0.4 并核对时间轴。 |
| `asr_asmr_vad_min_speech_ms` | `250` | 250 ms 可能滤掉极短词，必要时试 100–150 ms；VAD 默认关闭。 |
| `asr_asmr_vad_min_silence_ms` | `100` | 100 ms 允许较细分段；结合保留边界试听。 |
| `asr_asmr_vad_speech_pad_ms` | `200` | 200 ms 合理；漏字可试 300 ms，更多保留也会带入噪声。 |
| `aligner_model` | `Qwen/Qwen3-ForcedAligner-0.6B` | 保持固定 Qwen3-ForcedAligner；模型仅对已有文本做对齐，不纠正原文本。 |
| `asr_forced_alignment_enabled` | `false` | 默认关闭合理；对已校对文字但边界不准的句子按需开启，不能普遍保证更准。 |
| `asr_condition_on_previous_text` | `true` | 长静音/重复幻觉场景建议试 false；官方说明关闭可减少循环，但跨窗文字一致性可能下降。 |
| `asr_initial_prompt` | 空 | 默认空合理；仅添加实际专名或短语，不填无关长提示。 |
| `asr_timeout_seconds` | `600.0` | 超时保护，不提升质量；慢 CPU/复杂音频按运行时间增大。 |
| `asr_api_base_url` | `http://127.0.0.1:8000/v1` | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `asr_api_extra_body` | `{}` | 默认空对象合理；仅按该服务真实 API 添加字段，不能宣称通用最优。 |
| `asr_parakeet_decoder` | `tdt` | tdt 只影响官方 0.6B TDT/CTC 模型；默认 GAL 1.1B 不使用这个选项。 |
| `asr_chunk_seconds` | `120.0` | 120 秒是资源与上下文折中；较短能省显存，不能假定越长越准。 |
| `asr_kotoba_chunk_seconds` | `30.0` | 30 秒符合 Whisper 前端窗口，保持。 |
| `asr_moss_chunk_seconds` | `300.0` | 300 秒已通过约 320 秒双块实测；更长上下文可能利于说话人一致性，但需要更多显存，跨块身份不会自动合并。 |
| `asr_moss_dtype` | `auto` | auto：本机 BF16、旧 CUDA 卡 FP16、CPU FP32；这是兼容与内存选择，不声称精度排名。 |
| `asr_moss_attention` | `sdpa` | SDPA 已在本机验证；Flash Attention 2 需额外兼容安装，主要改善效率。 |
| `asr_moss_max_new_tokens` | `16384` | 16384 避免长转写截断；达到上限报错保留原结果，不静默接受半段。 |
| `asr_moss_max_context` | `131072` | 131072 是模型上下文上限；输入加预留输出超出时明确报错。 |
| `asr_moss_do_sample` | `false` | false 为官方贪心推理起点，优先文字确定性。 |
| `asr_moss_temperature` | `1.0` | 默认关闭采样，此参数不生效；仅实验时开启采样并调节。 |
| `asr_moss_top_p` | `1.0` | 默认关闭采样，此参数不生效；仅实验时开启采样并调节。 |
| `asr_moss_top_k` | `50` | 默认关闭采样，此参数不生效；仅实验时开启采样并调节。 |
| `asr_moss_num_beams` | `1` | 1 为官方方式；更多束并无本项目准确率提升证据。 |
| `asr_moss_repetition_penalty` | `1.0` | 1 不额外抑制真实重复词。 |
| `asr_moss_use_cache` | `true` | true 提高生成效率；不是音质增强。 |
| `asr_moss_seed` | `0` | 0 固定实验随机性；确定性模式下不代表最优种子。 |
| `asr_moss_prompt` | `请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。` | 官方含时间戳/说话人格式的指令；清空恢复内置，避免破坏解析。 |
| `asr_moss_hotwords` | 空 | 默认空；真实角色名、专名可按作品追加。 |
| `asr_moss_generation_params` | `{}` | 空对象合理；支持额外 generate 参数，保留输入/输出合同与专用控件不允许覆盖。 |
| `asr_moss_timeout_seconds` | `7200.0` | 7200 秒为整次总超时；特别长作品可提高到 86400 秒，不改变准确率。 |
| `asr_review_enabled` | `false` | 默认关闭节省成本；最终稿可先建议模式跑多模型复核并人工抽查。 |
| `asr_review_mode` | `suggest` | 建议 suggest；你保存的是 conservative，启用后会自动采纳部分结果，最好先看建议。 |
| `asr_review_window_seconds` | `30.0` | 30 秒上下文合理。 |
| `asr_review_context_seconds` | `0.5` | 0.5 秒合理；切断词尾时可增到 1 秒。 |
| `asr_review_models` | `["parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo","kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2"]` | 用不同模型家族交叉对比；可把 MOSS 加入。多个 Whisper 版本不是独立家族证据。 |
| `asr_review_text_priority_model` | `parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | 保留 Parakeet 起点；你保存的是 Faster-Whisper，建议日语素材先对比后决定。 |
| `asr_review_timestamp_priority_model` | `qwen_forced_aligner&#124;Qwen/Qwen3-ForcedAligner-0.6B` | schema 选 Qwen 对齐；你保存的是 Faster-Whisper。先校对文本，再抽查对齐，勿把对齐等同于文字纠错。 |
| `asr_review_background` | 空 | 作品、角色与实际专名有助于复核；默认空不产生编造背景。 |
| `asr_review_prompt` | 内置长文本/JSON，见参数表 | 保持限制候选选择的内置提示；不鼓励模型在证据外补写。 |
| `asr_review_max_drift_seconds` | `1.5` | 1.5 秒是容许阈值，增大只是放宽保护，不提高时间准确率。 |
| `pause_split_seconds` | `0.55` | 0.55 秒合理；MOSS 保留原段级边界，此项不用于强行拆 MOSS 结果。 |
| `max_sentence_seconds` | `15.0` | 15 秒有利于逐句编辑；MOSS 不伪造词级边界来满足该上限。 |
| `default_source_language` | `ja` | 按实际源语言选，日语作品保持 ja。 |

### 翻译

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `translation_provider` | `deepseek` | DeepSeek 是现有可用选择；没有对你的音声做供应商盲测，不能称为质量第一。 |
| `translation_model` | `deepseek-v4-flash` | 建议模型名更新为 deepseek-flash。官方已把旧 deepseek-v4-flash 路由到 V4.1；改名本身不提升质量。 |
| `translation_base_url` | 空 | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `translation_prompt` | 空 | 空值使用内置源语言模板；最值得加的是角色/称呼/术语约定。 |
| `translation_temperature` | `0.1` | 0.1 是保守翻译起点；不建议仅为“更生动”而大幅增温。 |
| `translation_top_p` | `1.0` | 保持 1，与低 temperature 配合；不要同时随意收紧两者。 |
| `translation_max_output_tokens` | `16384` | 16384 对逐批翻译较充足；提高只避免截断，不自动提升翻译质量。 |
| `translation_send_context` | `true` | 保持 true，有助于主语、省略与角色称呼的一致性。 |
| `translation_context_sentences` | `24` | 前后各 24 句偏重上下文但合理；极长台词可降到 12 以减少输入。 |
| `translation_memory_sentences` | `50` | 50 句有助于术语连贯，但会传递先前误译，需要校对。 |
| `translation_deepl_formality` | `default` | 只影响 DeepL；由目标文风决定，无通用最佳。 |
| `translation_microsoft_region` | 空 | 只影响微软服务的可用性，必须与账号资源匹配。 |
| `translation_extra_body` | `{}` | 默认空对象合理；仅按该服务真实 API 添加字段，不能宣称通用最优。 |
| `translation_prompt_ja` | 空 | 空值回落内置语言模板；角色、术语表比盲目增加采样温度更值得定制。 |
| `translation_prompt_en` | 空 | 空值回落内置语言模板；角色、术语表比盲目增加采样温度更值得定制。 |
| `translation_prompt_zh` | 空 | 空值回落内置语言模板；角色、术语表比盲目增加采样温度更值得定制。 |

### 配音

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `tts_target_language` | `zh` | 选择真实目标语言；中文配音保持 zh。 |
| `tts_backend` | `indextts2` | 保留 IndexTTS2；IndexTTS2.5 不是仅凭版本号就能宣称更适合耳语。 |
| `tts_model` | `IndexTTS2` | 与所选后端匹配；无跨模型人工盲测排名。 |
| `tts_device` | `cuda` | 本机保持 CUDA；CPU 改变速度，不是升级音色。 |
| `tts_reference_source` | `project_sentence` | 项目内干净语音参考合理；外部参考需匹配人物、录音条件。 |
| `tts_external_reference_audio` | 空 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_external_reference_text` | 空 | 依赖所选后端；填写参考音频的准确原文，而非待合成译文。 |
| `tts_external_reference_language` | `auto` | auto 合理；后端不能自动判断时填写真实源语言。 |
| `tts_api_base_url` | `http://127.0.0.1:9880` | 连接/请求合同字段，须与实际服务匹配，不存在通用最高效果值。 |
| `tts_api_extra_body` | `{}` | 默认空对象合理；仅按该服务真实 API 添加字段，不能宣称通用最优。 |
| `tts_timeout_seconds` | `600.0` | 600 秒是保护；按设备与后端耗时调整。 |
| `tts_request_concurrency` | `2` | 2 是服务吞吐折中；增大不改善音色，可能触发限流。 |
| `tts_model_path` | 本机资源路径 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_config_path` | 本机资源路径 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_executable` | 空 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_speed` | `1.0` | 保持 1；通过翻译长度和排期处理时长，优先保留自然语速。 |
| `tts_voice` | 空 | 外部服务依赖具体音色；本地克隆不靠此字段选声纹。 |
| `tts_volume` | `1.0` | 保持 1，在混音层控制相对响度，避免多层增益叠加。 |
| `tts_pitch` | `0` | 保持 0，避免无根据地改变音高。 |
| `tts_emotion` | `auto` | auto 可作外部服务起点；不是所有后端都会使用。 |
| `tts_style_prompt` | 空 | 按人物与场景填写短风格说明；默认空合理。 |
| `tts_temperature` | `0.8` | IndexTTS2 0.8 是现有起点；需要听辨稳定性时可对比 0.7，不能保证更好。 |
| `tts_top_p` | `0.9` | 0.9 保留自然变化；与 temperature 联合对比，不能单向判断最佳。 |
| `tts_index_use_fp16` | `true` | 本机保持 true；FP32 主要增加内存，未证实提高听感。 |
| `tts_index_emo_alpha` | `0.5` | schema 0.5，你保存 0.55；耳语先保持 0.4–0.55 区间，过高可能夸张，需试听。 |
| `tts_index_speaker_source` | `project_reference` | project_reference 有利于统一声纹，保持。 |
| `tts_index_emotion_source` | `sentence_reference` | 逐句参考保留表演变化；轻声参考噪声重时可改 project_reference 或 speaker_reference 对比，不能一概改。 |
| `tts_index_external_emotion_audio` | 空 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_index_emo_text` | 空 | 仅文字情绪模式生效；不应与真实场景冲突。 |
| `tts_index25_model_path` | 本机资源路径 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_index25_config_path` | 本机资源路径 | 资源定位字段；必须指向正确模型或干净参考，路径本身不决定质量。 |
| `tts_index25_language` | `zh` | 兼容字段；本地 2.5 实际请求语言来自 tts_target_language，不把此字段视作独立生效参数。 |
| `tts_index25_use_bf16` | `true` | 本机支持 BF16，保持；旧卡按硬件调整。 |
| `tts_index25_use_cuda_kernel` | `false` | 默认关闭合理，减少安装/兼容风险；没有质量提升证据。 |
| `tts_index25_use_deepspeed` | `false` | 默认关闭合理，主要影响加速与兼容。 |
| `tts_index25_use_accel` | `false` | 默认关闭合理，加速不等于质量提升。 |
| `tts_index25_use_torch_compile` | `false` | 默认关闭合理，避免首次编译和便携部署兼容成本。 |
| `tts_index25_emotion_vector` | `[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]` | 全零只在向量模式使用；根据实际情绪填写，不应全部拉满。 |
| `tts_index25_use_random` | `false` | 保持 false，减少随机参考变化。 |
| `tts_index25_interval_silence_ms` | `200` | 200 ms 合理，较多会让内部拼接停顿明显。 |
| `tts_index25_max_text_tokens` | `120` | 120 合理；增大可能减少切段但增加不稳定性，需试听。 |
| `tts_index25_duration_factor` | `1.0` | 保持 1；调时长需要试听自然度。 |
| `tts_index25_text_normalization` | `true` | 保持 true；专名/字母的规范化错误需人工校正。 |
| `tts_index25_do_sample` | `true` | TTS 保留采样有助于自然韵律；ASR 的关闭采样结论不直接用于 TTS。 |
| `tts_index25_temperature` | `0.8` | 保持 0.8 作为起点，只有配对试听才能选更佳值。 |
| `tts_index25_top_p` | `0.8` | 保持 0.8 作为现有起点。 |
| `tts_index25_top_k` | `30` | 保持 30 作为现有起点。 |
| `tts_index25_num_beams` | `3` | 保持 3；增大不保证音质更好。 |
| `tts_index25_repetition_penalty` | `10.0` | 10 是此 TTS 的现有配置；不要照搬到 MOSS ASR。 |
| `tts_index25_length_penalty` | `0.0` | 保持 0；不把生成长度偏置当时长对齐。 |
| `tts_index25_max_mel_tokens` | `1500` | 1500 是上限保护；截断时先缩短译文/拆句。 |
| `tts_gpt_top_k` | `15` | GPT-SoVITS 专用，15 是起点，需该模型试听。 |
| `tts_gpt_text_split_method` | `cut5` | cut5 是 GPT-SoVITS 切句策略，按文本标点调整。 |
| `tts_gpt_sample_steps` | `32` | 32 仅影响支持该参数的 GPT-SoVITS 版本，增大主要增加耗时。 |
| `tts_cosyvoice_mode` | `zero_shot` | zero_shot 适用于相应参考方式；跨语言需求选择兼容模式并核对参考文本。 |
| `tts_clone_mode` | `stable_reference` | 保持 stable_reference，先确保整条作品声纹一致。 |
| `tts_reference_sentence_id` | `null` | 默认不固定句号；正式作品建议人工选干净、连续、角色一致的参考句。 |
| `skip_japanese_fillers` | `true` | true 会跳过被识别为无实义语气词的配音；需要完整表演时可关闭并人工筛选。 |
| `reference_padding_seconds` | `0.0` | 0 避免带入相邻声；有辅音切头时可试 0.08–0.15 秒并检查参考，不能盲目扩大。 |
| `random_seed` | `20260722` | 固定种子便于对比，不代表此数字音质最好。 |
| `tts_device_selection_version` | `0` | 迁移计数器，不能当质量参数调整。 |

### 混音与空间跟随

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `separation_mix_mode` | `bilingual` | bilingual 保留原声，适合当前学习用途；replace 会受分离质量限制。 |
| `separation_keep_original` | `none` | schema none，你保存 unvoiced；替换版想保留喘息等可用 unvoiced，但应抽查是否回填原语言。 |
| `separation_keep_ids` | 空 | manual 模式下逐句选择；空值合理。 |
| `separation_keep_padding_ms` | `40.0` | 40 ms 合理；扩大可能带回相邻原语言。 |
| `spatial_rtf_enabled` | `false` | 双耳 ASMR 建议开启；schema 默认 false，你保存的 true 已符合空间跟随目标。单声道/不需要迁移可关闭。 |
| `spatial_rtf_strength` | `1.0` | 1 是完整迁移起点；耳位效果过强时降到 0.8 对比。 |
| `spatial_rtf_level_strength` | `0.7` | 0.7 是折中；近远变化过大时降低。 |
| `spatial_rtf_color_strength` | `0.35` | 0.35 是温和染色；发闷/变薄时试 0.2，不能保证所有素材都改善。 |
| `spatial_rtf_fft_size` | `0` | 0 自动选择合理。 |
| `spatial_rtf_hop_divisor` | `8` | 8 合理；更高主要增加计算，不是必然更好。 |
| `spatial_rtf_block_seconds` | `10.0` | 10 秒为内存控制；不要为质量直接拉满。 |
| `chinese_dubbing_offset_ms` | `500` | 双语学习版可保留 500 ms；替换版应从 0 ms 起步。你当前保存 400 ms。 |
| `chinese_max_auto_speed` | `1.8` | 自然耳语建议对比 1.25–1.4，schema 1.8 容易使较长译文加速明显；代价是要缩译文或改排期。 |
| `chinese_dubbing_timing_mode` | `fit_window` | fit_window 更守原时间窗；你保存 sequential，适合自然语速但可能逐句推迟，无通用最好。 |
| `chinese_gain_db` | `0.0` | 保持 0，先调相对响度而不是叠加固定增益。 |
| `normalize_chinese_loudness` | `true` | 保持 true，避免句间音量跳变。 |
| `match_source_loudness` | `true` | 保持 true，保留源句强弱；极端底噪源需抽查。 |
| `chinese_relative_loudness_db` | `-8.0` | 双语版 -8 dB 保留原声主导；想更听清中文可试 -6 dB，属于听感选择。 |
| `chinese_min_active_rms_dbfs` | `-42.0` | -42 dBFS 作为保护下限，过度抬升会放大耳语底噪。 |
| `chinese_target_active_rms_dbfs` | `-30.0` | -30 dBFS 较轻，适用于双语底下的配音；替换版应使用独立响度组。 |
| `chinese_max_loudness_boost_db` | `12.0` | 12 dB 是保护上限，不是要求把所有句子抬满。 |
| `chinese_line_peak_dbfs` | `-9.0` | -9 dBFS 保守留余量，保持。 |
| `chinese_stem_peak_dbfs` | `-3.0` | -3 dBFS 保留混音余量，保持。 |
| `chinese_fade_ms` | `8.0` | 8 ms 有助于减少拼接爆点；过大会吞辅音。 |
| `replacement_chinese_gain_db` | `0.0` | 保持 0，优先使用响度匹配。 |
| `replacement_normalize_chinese_loudness` | `true` | 保持 true，控制句间响度。 |
| `replacement_match_source_loudness` | `true` | 保持 true，沿用源句强弱。 |
| `replacement_chinese_relative_loudness_db` | `0.0` | 0 dB 相对原人声是替换版合理起点。 |
| `replacement_chinese_min_active_rms_dbfs` | `-42.0` | -42 dBFS 的保护下限合理。 |
| `replacement_chinese_target_active_rms_dbfs` | `-20.0` | -20 dBFS 比双语更突出；耳语可能偏响，建议与源声匹配并试听。 |
| `replacement_chinese_max_loudness_boost_db` | `12.0` | 12 dB 保守限制额外抬升。 |
| `replacement_chinese_line_peak_dbfs` | `-6.0` | -6 dBFS 预留峰值空间，保持。 |
| `replacement_chinese_stem_peak_dbfs` | `-3.0` | -3 dBFS 预留总混音空间，保持。 |
| `replacement_chinese_fade_ms` | `8.0` | 8 ms 保留短淡入淡出。 |
| `chinese_channel_routing` | `auto` | auto 维持声道布局；不要无条件把配音扩散到所有声道。 |
| `mix_peak_protection` | `true` | 保持 true，避免叠加削波。 |
| `mix_peak_limit_dbfs` | `-1.0` | -1 dBFS 是采样峰值保护，不等于测量过的 True Peak 保证。 |
| `mix_output_mode` | `both` | both 保留成品和配音轨便于复核，只增加文件量。 |

### 字幕

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `subtitle_timeline` | `source` | source 保持原声字幕；替换版如需跟随延后的配音可选 dubbing。 |
| `subtitle_max_chars_per_line` | `22` | 22 适合中文短行；屏幕与字体不同可试 18–24。 |
| `subtitle_min_duration_seconds` | `1.0` | 1 秒是可读性起点；密集对白会与时间窗冲突，需要抽查。 |
| `subtitle_max_cps` | `18.0` | 18 字/秒是阅读速度上限，不提高识别或配音质量。 |

### 通用

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `projects_root` | 空 | 存储位置，不影响质量；保留自己的项目目录。 |
| `huggingface_endpoint` | 空 | 下载端点，不影响同一 SHA-256 模型的质量。 |
| `pypi_index_url` | 空 | 安装源，不影响固定依赖的算法质量；MOSS 目前使用独立安装源。 |
| `ui_language` | `zh` | 界面语言，不影响用户文本、模型与输出语言。 |
| `open_browser` | `true` | 启动偏好，不影响推理。 |
| `download_source` | `modelscope` | 旧下载策略默认国内镜像；MOSS 权重路径始终先镜像、失败后官方固定源。 |

### 批量

| 字段 | 默认值 | 判断 / 建议 |
|---|---|---|
| `autoflow_output_folder_name` | `AutoFlow输出` | 存储命名偏好，不影响质量。 |
| `autoflow_default_mode` | `video_normal` | 视频/音频容器与输出用途选择，不代表语音质量等级。 |
| `autoflow_default_layout` | `merged` | merged 决定合并布局；需要逐轨抽查可用 both，增加磁盘占用。 |
| `autoflow_include_bonus` | `false` | 是否纳入 bonus 轨道，内容范围偏好。 |
| `autoflow_background_policy` | `auto` | auto 决定视频画面，不影响配音质量。 |
| `autoflow_embed_subtitles` | `true` | true 增加可用性；不影响 ASR 准确率。 |
| `autoflow_subtitle_language` | `bilingual` | bilingual 适合学习用途，按观众选择。 |
| `autoflow_subtitle_naming` | `original` | original 保持关联；命名不影响字幕时间。 |
| `autoflow_subtitle_custom_name` | `字幕` | 仅自定义命名生效。 |
| `autoflow_original_hard_subtitles` | `false` | 按原素材是否已有硬字幕选择，避免重复画面。 |
| `autoflow_timestamp_footer_position` | `after` | 排版偏好，不影响模型质量。 |
| `autoflow_translate_work_title` | `true` | true 便于中文查找；专名仍需校对。 |
| `autoflow_translate_track_titles` | `true` | true 便于中文查找；专名仍需校对。 |
| `autoflow_reference_wait_enabled` | `true` | 正式作品建议开启人工参考选择；你保存 false，全自动可能选到不够干净的参考。 |
| `autoflow_reference_wait_seconds` | `60` | 60 秒等待是交互超时，不是参考音频长度；按操作习惯增加。 |
| `autoflow_preferred_audio_formats` | `wav,flac,ape,m4a,mp3` | 现有顺序优先 WAV/FLAC 合理；若 WAV 为有损转码而 FLAC 为原盘，应人工选真实优质源。 |
| `autoflow_harmonized_volume_reduction_db` | `10.0` | 10 dB 是成品偏好，无通用音质最佳。 |
| `autoflow_harmonized_delay_minutes` | `20.0` | 20 分钟是特殊编排模式，不是日常配音延迟参数。 |
| `autoflow_timestamp_footer` | `双语音声制作器：BV1f43G6YEov<br>内嵌字幕和配音为本地 AI 生成，内容仅供参考。<br>仅供日语学习，有能力请购买正版支持。` | 成品说明文字，不影响模型质量。 |

## 依据与边界

- 代码：`models.py`、`user_settings.py`、参数目录、ASR/TTS 请求构造、混音/分离与任务生命周期。
- [MOSS 官方推理说明](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)：确定性生成、官方指令与注意力建议。
- [DeepSeek 当前模型说明](https://api-docs.deepseek.com/quick_start/pricing/)：当前名称 `deepseek-flash`，旧别名路由说明；本项目已显式关闭 thinking，低温度用于非思考翻译请求。
- [Faster-Whisper 参数定义](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)：关闭前文条件可减少重复循环，同时可能降低跨窗一致性。
- [IndexTTS2 官方模型](https://huggingface.co/IndexTeam/IndexTTS-2)：未把“更大采样值/更新版本”作为本项目音质结论。
- 当前审查不包含你每个旧项目的独立覆盖值；它们可能与全局默认不同。没有调用付费翻译/TTS API，也没有修改其模型选择。
