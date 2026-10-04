中文 | [English](en/PARAMETERS.md)

[文档索引](INDEX.md) · [配置讲解](CONFIGURATION.md)

# 完整参数参考

按当前 `services/parameters.py` 清单生成，**201 个字段全部列出**。这是 schema 默认值，不是已保存值。后端/语言可能带入不同模型、Prompt、设备或音色；操作讲解见对应功能章节。

项目字段出现在项目/新默认中；全局字段作用于全局/新项目/新队列。Always 表示无静态显示条件，高级分组仍需展开。选项可能随后端变化；校验还包括跨字段依赖、语言能力与 JSON 合同。

路径用 `<program>` 代替本机路径。空 Prompt 使用内置语言模板。迁移/旧复核字段也列出，但普通操作不应修改兼容计数器或旧复核 Prompt。

## 人声分离

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `separation_enabled`<br>人声分离 | boolean<br>project | `false` | — | Always |
| `separation_backend`<br>分离方式 | string<br>project | `local` | local, replicate, http | Always |
| `separation_model`<br>模型 | string<br>project | `vocals_mel_band_roformer.ckpt` | — | Always |
| `separation_device`<br>运行在 | string<br>project | `cuda` | cuda, cpu | Always |
| `separation_chunk_seconds`<br>每块长度 | number<br>project | `30.0` | ≥5; ≤300 | Always |
| `separation_overlap_seconds`<br>前后多留 | number<br>project | `1.0` | ≥0; ≤3 | Always |
| `separation_timeout_seconds`<br>每块超时 | number<br>project | `1800.0` | ≥30; ≤14400 | Always |
| `separation_use_autocast`<br>自动精度 | boolean<br>project | `true` | — | Always |
| `separation_normalization`<br>归一化强度 | number<br>project | `1.0` | ≤1; >0 | Always |
| `separation_vocal_stem`<br>人声轨名称 | string<br>project | `Vocals` | — | Always |
| `separation_common_params`<br>通用高级参数（JSON） | string<br>project | `{"invert_using_spec":false,"use_torch_compile":false,"use_native_fp16":false,"amplification_threshold":0.0}` | — | Always |
| `separation_mdx_params`<br>MDX 参数 | string<br>project | `{"hop_length":1024,"segment_size":256,"overlap":0.25,"batch_size":1,"enable_denoise":false}` | — | Always |
| `separation_vr_params`<br>VR 参数 | string<br>project | `{"batch_size":1,"window_size":512,"aggression":5,"enable_tta":false,"enable_post_process":false,"post_process_threshold":0.2,"high_end_process":false}` | — | Always |
| `separation_demucs_params`<br>Demucs 参数 | string<br>project | `{"segment_size":"Default","shifts":2,"overlap":0.25,"segments_enabled":true}` | — | Always |
| `separation_mdxc_params`<br>Mel/BS-RoFormer 与 MDX23C 参数 | string<br>project | `{"segment_size":256,"override_model_segment_size":true,"batch_size":1,"overlap":4,"pitch_shift":0}` | — | Always |
| `separation_api_url`<br>接口地址 | string<br>project | 空 | — | Always |
| `separation_api_model`<br>模型版本 | string<br>project | 空 | — | Always |
| `separation_api_audio_field`<br>音频字段 | string<br>project | `audio` | — | Always |
| `separation_api_output_field`<br>人声字段 | string<br>project | `vocals` | — | Always |
| `separation_api_params`<br>附加请求参数 | string<br>project | `{}` | — | Always |
| `separation_cloud_consent`<br>同意上传音频 | boolean<br>project | `false` | — | Always |

## 识别与复核

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `asr_backend`<br>识别模型 | string<br>project | `parakeet_nemo` | parakeet_nemo, kotoba_whisper, faster_whisper, generic_asr_api | Always |
| `asr_model`<br>版本 | string<br>project | `grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | 随所选后端 | Always |
| `asr_batch_size`<br>批量大小 | integer<br>project | `1` | ≥1; ≤32 | Always |
| `asr_device`<br>运行在 | string<br>project | `cuda` | cuda, cpu | Always |
| `asr_compute_type`<br>精度 | string<br>project | `float16` | — | asr_backend in ["faster_whisper","kotoba_whisper"] |
| `asr_beam_size`<br>束搜索大小 | integer<br>project | `5` | ≥1; ≤100 | asr_backend in ["faster_whisper"] |
| `asr_vad_filter`<br>使用模型自带 VAD | boolean<br>project | `false` | — | Always |
| `asr_vad_mode`<br>方式 | string<br>project | `off` | off, backend, asmr | Always |
| `asr_vad_min_silence_ms`<br>VAD 最短静音毫秒 | integer<br>project | `500` | ≥50; ≤10000 | asr_vad_mode in ["backend"] |
| `asr_asmr_vad_threshold`<br>ASMR VAD 语音阈值 | number<br>project | `0.5` | ≥0.05; ≤0.95 | asr_vad_mode in ["asmr"] |
| `asr_asmr_vad_min_speech_ms`<br>ASMR VAD 最短语音毫秒 | integer<br>project | `250` | ≥20; ≤10000 | asr_vad_mode in ["asmr"] |
| `asr_asmr_vad_min_silence_ms`<br>ASMR VAD 最短静音毫秒 | integer<br>project | `100` | ≥20; ≤10000 | asr_vad_mode in ["asmr"] |
| `asr_asmr_vad_speech_pad_ms`<br>ASMR VAD 边界保留毫秒 | integer<br>project | `200` | ≥0; ≤5000 | asr_vad_mode in ["asmr"] |
| `aligner_model`<br>时间对齐模型 | string<br>project | `Qwen/Qwen3-ForcedAligner-0.6B` | — | Always |
| `asr_forced_alignment_enabled`<br>Qwen3 时间对齐 | boolean<br>project | `false` | — | Always |
| `asr_condition_on_previous_text`<br>参考前面的文字 | boolean<br>project | `true` | — | asr_backend in ["faster_whisper"] |
| `asr_initial_prompt`<br>提示词 | string<br>project | 空 | — | asr_backend in ["faster_whisper"] |
| `asr_timeout_seconds`<br>Parakeet 连续无响应超时（秒） | number<br>project | `600.0` | ≥10.0; ≤7200.0 | asr_backend in ["parakeet_nemo"] |
| `asr_api_base_url`<br>通用 ASR API（接口）基础地址 | string<br>project | `http://127.0.0.1:8000/v1` | — | asr_backend in ["generic_asr_api"] |
| `asr_api_extra_body`<br>通用 ASR API 附加请求参数（JSON） | string<br>project | `{}` | — | asr_backend in ["generic_asr_api"] |
| `asr_parakeet_decoder`<br>解码方式 | string<br>project | `tdt` | tdt, ctc | asr_backend in ["parakeet_nemo"] |
| `asr_chunk_seconds`<br>分块长度 | number<br>project | `120.0` | ≥15.0; ≤600.0 | asr_backend in ["parakeet_nemo"] |
| `asr_kotoba_chunk_seconds`<br>分块长度 | number<br>project | `30.0` | ≥5.0; ≤120.0 | asr_backend in ["kotoba_whisper"] |
| `asr_review_enabled`<br>多模型复核 | boolean<br>project | `false` | — | Always |
| `asr_review_mode`<br>复核方式 | string<br>project | `suggest` | suggest, conservative | asr_review_enabled in [true] |
| `asr_review_window_seconds`<br>每段长度 | number<br>project | `30.0` | ≥10.0; ≤90.0 | asr_review_enabled in [true] |
| `asr_review_context_seconds`<br>前后多听 | number<br>project | `0.5` | ≥0.0; ≤3.0 | asr_review_enabled in [true] |
| `asr_review_models`<br>复核模型 | array<br>project | `["parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo","kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2"]` | max items 6; parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo, parakeet_nemo&#124;nvidia/parakeet-tdt_ctc-0.6b-ja, kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.2, kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.1, kotoba_whisper&#124;kotoba-tech/kotoba-whisper-v2.0, faster_whisper&#124;large-v2, faster_whisper&#124;kotoba-tech/kotoba-whisper-v2.0-faster, faster_whisper&#124;distil-large-v2, faster_whisper&#124;large-v3, faster_whisper&#124;large-v3-turbo, faster_whisper&#124;medium, faster_whisper&#124;small | asr_review_enabled in [true] |
| `asr_review_text_priority_model`<br>文字优先 | string<br>project | `parakeet_nemo&#124;grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo` | — | asr_review_enabled in [true] |
| `asr_review_timestamp_priority_model`<br>时间优先 | string<br>project | `qwen_forced_aligner&#124;Qwen/Qwen3-ForcedAligner-0.6B` | — | asr_review_enabled in [true] |
| `asr_review_background`<br>背景说明 | string<br>project | 空 | — | asr_review_enabled in [true] |
| `asr_review_prompt`<br>复核 Prompt | string<br>project | 内置模板，见 [Prompt](PROMPTS.md) | — | asr_review_enabled in [true] |
| `asr_review_max_drift_seconds`<br>最大时间偏差 | number<br>project | `1.5` | ≥0.1; ≤10.0 | asr_review_enabled in [true] |
| `pause_split_seconds`<br>停顿多久算一句结束 | number<br>project | `0.55` | ≥0.1; ≤5.0 | Always |
| `max_sentence_seconds`<br>单句最长 | number<br>project | `15.0` | ≥2.0; ≤60.0 | Always |
| `default_source_language`<br>音频里说的是 | string<br>global | `ja` | ja, en, zh | Always |

## 翻译

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `translation_provider`<br>翻译服务 | string<br>project | `deepseek` | deepseek, bailian, doubao, openai, anthropic, gemini, openai_compatible, sensenova, deepl, google_translate, microsoft_translate | Always |
| `translation_model`<br>模型 | string<br>project | `deepseek-v4-flash` | 随所选后端 | Always |
| `translation_base_url`<br>接口地址 | string<br>project | 空 | — | Always |
| `translation_prompt`<br>翻译 Prompt | string<br>project | 空 | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_temperature`<br>Temperature | number<br>project | `0.1` | ≥0.0; ≤2.0 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_top_p`<br>Top P | number<br>project | `1.0` | ≤1.0; >0.0 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_max_output_tokens`<br>单次最多输出 | integer<br>project | `16384` | ≥1024; ≤131072 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_send_context`<br>带上前后文 | boolean<br>project | `true` | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_context_sentences`<br>前后各带 | integer<br>project | `24` | ≥0; ≤200 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_memory_sentences`<br>记住前面已翻译的 | integer<br>project | `50` | ≥0; ≤500 | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_deepl_formality`<br>正式程度 | string<br>project | `default` | — | translation_provider in ["deepl"] |
| `translation_microsoft_region`<br>区域 | string<br>project | 空 | — | translation_provider in ["microsoft_translate"] |
| `translation_extra_body`<br>附加请求参数 | string<br>project | `{}` | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_prompt_ja`<br>translation_prompt_ja | string<br>global | 空 | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_prompt_en`<br>translation_prompt_en | string<br>global | 空 | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |
| `translation_prompt_zh`<br>translation_prompt_zh | string<br>global | 空 | — | translation_provider in ["deepseek","bailian","doubao","openai","anthropic","gemini","openai_compatible","sensenova"] |

## 配音

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `tts_target_language`<br>配音成 | string<br>project | `zh` | zh, en | Always |
| `tts_backend`<br>配音模型 | string<br>project | `indextts2` | indextts2, indextts2_5, edge_tts, indextts2_api, gpt_sovits, cosyvoice, mimo_tts, minimax, fish_speech, generic_tts_api | Always |
| `tts_model`<br>模型 | string<br>project | `IndexTTS2` | 随所选后端 | Always |
| `tts_device`<br>运行在 | string<br>project | `cuda` | cuda, cpu | tts_backend in ["indextts2","indextts2_5"] |
| `tts_reference_source`<br>参考来自 | string<br>project | `project_sentence` | project_sentence, external | Always |
| `tts_external_reference_audio`<br>外部参考音频 | string<br>project | 空 | — | Always AND (tts_reference_source in ["external"] AND tts_backend in ["gpt_sovits","cosyvoice","fish_speech","mimo_tts","generic_tts_api"] OR tts_index_speaker_source in ["external"] AND tts_backend in ["indextts2","indextts2_api","indextts2_5"]) |
| `tts_external_reference_text`<br>参考文字 | string<br>project | 空 | — | tts_backend in ["gpt_sovits","fish_speech"] AND tts_reference_source in ["external"] |
| `tts_external_reference_language`<br>参考语言 | string<br>project | `auto` | auto, ja, en, zh | tts_backend in ["gpt_sovits"] AND tts_reference_source in ["external"] |
| `tts_api_base_url`<br>接口地址 | string<br>project | `http://127.0.0.1:9880` | — | tts_backend in ["indextts2_api","gpt_sovits","cosyvoice","mimo_tts","minimax","fish_speech"] |
| `tts_api_extra_body`<br>附加请求参数 | string<br>project | `{}` | — | tts_backend in ["indextts2_api","gpt_sovits","cosyvoice","mimo_tts","minimax","fish_speech"] |
| `tts_timeout_seconds`<br>单句超时 | number<br>project | `600.0` | ≥10.0; ≤7200.0 | Always |
| `tts_request_concurrency`<br>同时请求 | integer<br>project | `2` | ≥1; ≤8 | tts_backend in ["indextts2_api","gpt_sovits","cosyvoice","mimo_tts","minimax","fish_speech"] |
| `tts_model_path`<br>模型权重目录 | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts\checkpoints` | — | tts_backend in ["indextts2"] |
| `tts_config_path`<br>配置文件 | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts\checkpoints\config.yaml` | — | tts_backend in ["indextts2"] |
| `tts_executable`<br>tts_executable | string<br>project | 空 | — | tts_backend in ["indextts2","indextts2_5"] |
| `tts_speed`<br>语速 | number<br>project | `1.0` | ≥0.25; ≤4.0 | Always |
| `tts_voice`<br>音色 | string<br>project | 空 | 随所选后端 | tts_backend in ["edge_tts","mimo_tts","minimax","fish_speech","generic_tts_api"] |
| `tts_volume`<br>音量 | number<br>project | `1.0` | ≥0.1; ≤10.0 | tts_backend in ["edge_tts","minimax"] |
| `tts_pitch`<br>音调 | integer<br>project | `0` | ≥-12; ≤12 | tts_backend in ["minimax"] |
| `tts_emotion`<br>情绪 | string<br>project | `auto` | — | tts_backend in ["minimax"] |
| `tts_style_prompt`<br>语气与风格说明 | string<br>project | 空 | — | tts_backend in ["mimo_tts"] |
| `tts_temperature`<br>Temperature | number<br>project | `0.8` | ≥0.0; ≤2.0 | Always |
| `tts_top_p`<br>Top P | number<br>project | `0.9` | ≤1.0; >0.0 | Always |
| `tts_index_use_fp16`<br>FP16 | boolean<br>project | `true` | — | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_emo_alpha`<br>情绪强度 | number<br>project | `0.5` | ≥0.0; ≤1.0 | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_speaker_source`<br>音色来自 | string<br>project | `project_reference` | project_reference, sentence_reference, external | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_emotion_source`<br>情绪来自 | string<br>project | `sentence_reference` | sentence_reference, project_reference, speaker_reference, external, text, vector | tts_backend in ["indextts2","indextts2_api","indextts2_5"] |
| `tts_index_external_emotion_audio`<br>情绪参考音频 | string<br>project | 空 | — | tts_backend in ["indextts2","indextts2_api","indextts2_5"] AND tts_index_emotion_source in ["external"] |
| `tts_index_emo_text`<br>情绪文字 | string<br>project | 空 | — | tts_backend in ["indextts2","indextts2_api","indextts2_5"] AND tts_index_emotion_source in ["text"] |
| `tts_index25_model_path`<br>模型权重目录 | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts-2.5\checkpoints` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_config_path`<br>配置文件 | string<br>project | `<program>\.asmr-dubber\runtimes\index-tts-2.5\checkpoints\config.yaml` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_language`<br>语言 | string<br>project | `zh` | zh, en, ja, es, ar | tts_backend in ["indextts2_5"] |
| `tts_index25_use_bf16`<br>BF16 | boolean<br>project | `true` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_cuda_kernel`<br>CUDA Kernel | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_deepspeed`<br>DeepSpeed | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_accel`<br>加速 | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_use_torch_compile`<br>torch.compile | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_emotion_vector`<br>情绪向量 | array<br>project | `[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]` | min items 8; max items 8 | tts_backend in ["indextts2_5"] |
| `tts_index25_use_random`<br>随机选择情绪和音色条件 | boolean<br>project | `false` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_interval_silence_ms`<br>内部切段间隔 | integer<br>project | `200` | ≥0; ≤2000 | tts_backend in ["indextts2_5"] |
| `tts_index25_max_text_tokens`<br>单段最大文本 | integer<br>project | `120` | ≥20; ≤600 | tts_backend in ["indextts2_5"] |
| `tts_index25_duration_factor`<br>时长系数 | number<br>project | `1.0` | ≥0.5; ≤2.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_text_normalization`<br>文本规范化 | boolean<br>project | `true` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_do_sample`<br>启用采样 | boolean<br>project | `true` | — | tts_backend in ["indextts2_5"] |
| `tts_index25_temperature`<br>Temperature | number<br>project | `0.8` | ≤2.0; >0.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_top_p`<br>Top P | number<br>project | `0.8` | ≤1.0; >0.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_top_k`<br>Top K | integer<br>project | `30` | ≥0; ≤100 | tts_backend in ["indextts2_5"] |
| `tts_index25_num_beams`<br>束搜索数量 | integer<br>project | `3` | ≥1; ≤10 | tts_backend in ["indextts2_5"] |
| `tts_index25_repetition_penalty`<br>重复惩罚 | number<br>project | `10.0` | ≥0.1; ≤20.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_length_penalty`<br>长度惩罚 | number<br>project | `0.0` | ≥-2.0; ≤2.0 | tts_backend in ["indextts2_5"] |
| `tts_index25_max_mel_tokens`<br>最大声学 Token | integer<br>project | `1500` | ≥100; ≤1815 | tts_backend in ["indextts2_5"] |
| `tts_gpt_top_k`<br>Top K | integer<br>project | `15` | ≥1; ≤100 | tts_backend in ["gpt_sovits"] |
| `tts_gpt_text_split_method`<br>文本切分方法 | string<br>project | `cut5` | — | tts_backend in ["gpt_sovits"] |
| `tts_gpt_sample_steps`<br>采样步数 | integer<br>project | `32` | ≥1; ≤64 | tts_backend in ["gpt_sovits"] |
| `tts_cosyvoice_mode`<br>模式 | string<br>project | `zero_shot` | zero_shot, cross_lingual | tts_backend in ["cosyvoice"] |
| `tts_clone_mode`<br>模仿方式 | string<br>project | `stable_reference` | stable_reference, reference_only | Always |
| `tts_reference_sentence_id`<br>项目统一参考句 ID | string / nullable<br>project | `null` | — | Always |
| `skip_japanese_fillers`<br>跳过纯语气词 | boolean<br>project | `true` | — | Always |
| `reference_padding_seconds`<br>参考音频边缘扩展秒数 | number<br>project | `0.0` | ≥0.0; ≤2.0 | Always |
| `random_seed`<br>随机种子 | integer<br>project | `20260722` | ≥0 | Always |
| `tts_device_selection_version`<br>tts_device_selection_version | integer<br>global | `0` | ≥0 | Always |

## 混音与空间跟随

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `separation_mix_mode`<br>成品 | string<br>project | `bilingual` | bilingual, replace | Always |
| `separation_keep_original`<br>回填哪些 | string<br>project | `none` | none, unvoiced, manual | separation_mix_mode in ["replace"] |
| `separation_keep_ids`<br>句子编号 | string<br>project | 空 | — | separation_mix_mode in ["replace"] |
| `separation_keep_padding_ms`<br>前后多留 | number<br>project | `40.0` | ≥0; ≤500 | separation_mix_mode in ["replace"] |
| `spatial_rtf_enabled`<br>空间跟随 | boolean<br>project | `false` | — | Always |
| `spatial_rtf_strength`<br>强度 | number<br>project | `1.0` | ≥0; ≤1 | spatial_rtf_enabled in [true] |
| `spatial_rtf_level_strength`<br>远近变化强度 | number<br>project | `0.7` | ≥0; ≤1 | spatial_rtf_enabled in [true] |
| `spatial_rtf_color_strength`<br>音色染色强度 | number<br>project | `0.35` | ≥0; ≤1 | spatial_rtf_enabled in [true] |
| `spatial_rtf_fft_size`<br>FFT 窗口 | integer<br>project | `0` | 0, 1024, 2048, 4096, 8192 | spatial_rtf_enabled in [true] |
| `spatial_rtf_hop_divisor`<br>FFT 步长除数 | integer<br>project | `8` | 4, 8, 16 | spatial_rtf_enabled in [true] |
| `spatial_rtf_block_seconds`<br>内存分块 | number<br>project | `10.0` | ≥1; ≤30 | spatial_rtf_enabled in [true] |
| `chinese_dubbing_offset_ms`<br>整体偏移 | integer<br>project | `500` | ≥-30000; ≤30000 | Always |
| `chinese_max_auto_speed`<br>最多加速到 | number<br>project | `1.8` | ≥1.0; ≤4.0 | Always |
| `chinese_dubbing_timing_mode`<br>配音比原句长时 | string<br>project | `fit_window` | fit_window, sequential | Always |
| `chinese_gain_db`<br>处理后整体微调 | number<br>project | `0.0` | ≥-40.0; ≤20.0 | Always |
| `normalize_chinese_loudness`<br>音量处理方式 | boolean<br>project | `true` | — | Always |
| `match_source_loudness`<br>跟随对应原声 | boolean<br>project | `true` | — | Always |
| `chinese_relative_loudness_db`<br>相对原声 | number<br>project | `-8.0` | ≥-24.0; ≤24.0 | Always |
| `chinese_min_active_rms_dbfs`<br>最安静不低于 | number<br>project | `-42.0` | ≥-60.0; ≤-20.0 | Always |
| `chinese_target_active_rms_dbfs`<br>最响不超过 / 统一到 | number<br>project | `-30.0` | ≥-50.0; ≤-16.0 | Always |
| `chinese_max_loudness_boost_db`<br>每句最多提升 | number<br>project | `12.0` | ≥0.0; ≤30.0 | Always |
| `chinese_line_peak_dbfs`<br>单句峰值上限 | number<br>project | `-9.0` | ≥-20.0; ≤-1.0 | Always |
| `chinese_stem_peak_dbfs`<br>配音轨峰值上限 | number<br>project | `-3.0` | ≥-12.0; ≤-0.1 | Always |
| `chinese_fade_ms`<br>句首句尾淡入淡出 | number<br>project | `8.0` | ≥0.0; ≤100.0 | Always |
| `replacement_chinese_gain_db`<br>处理后整体微调 | number<br>project | `0.0` | ≥-40.0; ≤20.0 | separation_mix_mode in ["replace"] |
| `replacement_normalize_chinese_loudness`<br>音量处理方式 | boolean<br>project | `true` | — | separation_mix_mode in ["replace"] |
| `replacement_match_source_loudness`<br>跟随对应原声 | boolean<br>project | `true` | — | separation_mix_mode in ["replace"] |
| `replacement_chinese_relative_loudness_db`<br>相对原声 | number<br>project | `0.0` | ≥-24.0; ≤24.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_min_active_rms_dbfs`<br>最安静不低于 | number<br>project | `-42.0` | ≥-60.0; ≤-20.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_target_active_rms_dbfs`<br>最响不超过 / 统一到 | number<br>project | `-20.0` | ≥-50.0; ≤-16.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_max_loudness_boost_db`<br>每句最多提升 | number<br>project | `12.0` | ≥0.0; ≤30.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_line_peak_dbfs`<br>单句峰值上限 | number<br>project | `-6.0` | ≥-20.0; ≤-1.0 | separation_mix_mode in ["replace"] |
| `replacement_chinese_stem_peak_dbfs`<br>配音轨峰值上限 | number<br>project | `-3.0` | ≥-12.0; ≤-0.1 | separation_mix_mode in ["replace"] |
| `replacement_chinese_fade_ms`<br>句首句尾淡入淡出 | number<br>project | `8.0` | ≥0.0; ≤100.0 | separation_mix_mode in ["replace"] |
| `chinese_channel_routing`<br>多声道时配音放在 | string<br>project | `auto` | auto, all | Always |
| `mix_peak_protection`<br>成品峰值保护 | boolean<br>project | `true` | — | Always |
| `mix_peak_limit_dbfs`<br>峰值上限 | number<br>project | `-1.0` | ≥-6.0; ≤-0.1 | Always |
| `mix_output_mode`<br>保存哪些音频 | string<br>project | `both` | both, mixed, stem | Always |

## 字幕

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `subtitle_timeline`<br>时间 | string<br>project | `source` | source, dubbing | Always |
| `subtitle_max_chars_per_line`<br>每行最多 | integer<br>project | `22` | ≥8; ≤500 | Always |
| `subtitle_min_duration_seconds`<br>最短显示 | number<br>project | `1.0` | ≥0.2; ≤10.0 | Always |
| `subtitle_max_cps`<br>每秒最多 | number<br>project | `18.0` | ≥5.0; ≤40.0 | Always |

## 通用

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `projects_root`<br>项目保存位置 | string<br>global | 空 | — | Always |
| `huggingface_endpoint`<br>Hugging Face 端点 | string<br>global | 空 | — | Always |
| `pypi_index_url`<br>Python 软件源 | string<br>global | 空 | — | Always |
| `ui_language`<br>界面语言 | string<br>global | `zh` | ^(zh&#124;en)$; zh, en | Always |
| `open_browser`<br>启动时自动打开浏览器 | boolean<br>global | `true` | — | Always |
| `download_source`<br>下载来源 | string<br>global | `modelscope` | ^(modelscope&#124;original)$; modelscope, original | Always |

## 批量

| 键 / 界面名称 | 类型 / 范围归属 | 默认值 | 范围 / 可选值 | 显示条件 |
|---|---|---|---|---|
| `autoflow_output_folder_name`<br>成品文件夹名称 | string<br>global | `AutoFlow输出` | — | Always |
| `autoflow_default_mode`<br>默认格式 | string<br>global | `video_normal` | audio, video_normal, video_harmonized | Always |
| `autoflow_default_layout`<br>默认多轨方式 | string<br>global | `merged` | merged, separate, both | Always |
| `autoflow_include_bonus`<br>包含特典、样本和 Free Talk | boolean<br>global | `false` | — | Always |
| `autoflow_background_policy`<br>默认视频画面 | string<br>global | `auto` | auto, black | Always |
| `autoflow_embed_subtitles`<br>默认把字幕烧进视频 | boolean<br>global | `true` | — | Always |
| `autoflow_subtitle_language`<br>默认内容 | string<br>global | `bilingual` | bilingual, source, zh | Always |
| `autoflow_subtitle_naming`<br>文件命名 | string<br>global | `original` | original, standard, custom | Always |
| `autoflow_subtitle_custom_name`<br>自定义名称 | string<br>global | `字幕` | — | Always |
| `autoflow_original_hard_subtitles`<br>原声视频也烧字幕 | boolean<br>global | `false` | — | Always |
| `autoflow_timestamp_footer_position`<br>附加文字位置 | string<br>global | `after` | before, after | Always |
| `autoflow_translate_work_title`<br>翻译作品文件夹名称 | boolean<br>global | `true` | — | Always |
| `autoflow_translate_track_titles`<br>翻译音轨标题 | boolean<br>global | `true` | — | Always |
| `autoflow_reference_wait_enabled`<br>选音色参考时等我确认 | boolean<br>global | `true` | — | Always |
| `autoflow_reference_wait_seconds`<br>最多等 | integer<br>global | `60` | ≥1; ≤3600 | Always |
| `autoflow_preferred_audio_formats`<br>有多种音频格式时 | string<br>global | `wav,flac,ape,m4a,mp3` | — | Always |
| `autoflow_harmonized_volume_reduction_db`<br>原声降低 | number<br>global | `10.0` | ≥0.0; ≤60.0 | Always |
| `autoflow_harmonized_delay_minutes`<br>原声、配音和字幕整体延后 | number<br>global | `20.0` | ≥0.0; ≤1440.0 | Always |
| `autoflow_timestamp_footer`<br>附加文字 | string<br>global | `双语音声制作器：BV1f43G6YEov<br>内嵌字幕和配音为本地 AI 生成，内容仅供参考。<br>仅供日语学习，有能力请购买正版支持。` | — | Always |

