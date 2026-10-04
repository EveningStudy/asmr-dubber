# 界面重构实施与覆盖计划

## 约束与基线
- 唯一设计依据：design/mockup.html。保留布局、文案、四步工作区及修改即保存交互。
- 参数类型、默认值、范围来自现有 ProjectSettings / UserSettings；模型数字来自 model_registry / runtime_manager，未知数字不编造。
- 核心 pipeline / ASR / TTS / translation / audio 不改变行为。服务层不导入界面库；接口层只校验、路由并调用服务。
- 当前目录实施，不推送，不改变用户文档或版本号。已有脏文件的完整补丁与源码快照已保存在系统临时目录。
- 基线全量测试：646 passed, 3 skipped（2026-10-04）。每步全量通过后单独提交。
- 新文件不超过 500 行。现有超过 500 行的核心文件保持原状；界面与抽出的服务按业务职责拆分。

## 1. mockup 控件到现有功能
| 页面 / 控件 | 现有字段或函数 | 新接口 / 服务 |
| --- | --- | --- |
| 项目：拖入/选择媒体、新建、语言、配音成、文字来源 | ui_services.create_project; source_language; target_language; import_transcript_data | POST projects/create / uploads; 统一任务 |
| 最近项目、继续制作、打开文件夹 | recent_projects; load_view; open_project_directory | projects/list, projects/get, projects/open |
| 四步：识别、翻译剩余/全部、配音剩余、导出 | analyze; translate; synthesize; mix; subtitles; pipeline.translate_project(force) | tasks/start；服务动作 analyze/translate/synthesize/export |
| 搜索、全部/未翻译/未配音/待复核 | Sentence.source_text/zh_text/status/script_review_note | 前端按真实项目状态过滤 |
| 句子启用、时间、原文/译文、自动保存 | project_rows; apply_table; save_table; browser_revision_scope | projects/table；revision 冲突保留编辑 |
| 新增/删除句子、逐句原声/配音开关和增益 | sentence_editor; Sentence.original_audio_enabled/gain, chinese_audio_enabled/gain | 同一表格接口，补选中句子编辑区 |
| 原声/配音/一起试听与播放定位 | source media; Sentence.tts_file; output_file; chinese_stem_file | 注册媒体凭据 + Range 音频响应 |
| 导入字幕/台本、粘贴、原文/配音稿、纯文字时序 | import_transcript_data(script_kind,plain_timing) | tasks/start(import) |
| 音色参考试听/更换、外部音频、参考文字/语言、情绪音频 | reference_picker/preview/select_reference; store_reference_audio; tts_*reference* | projects/reference / tasks/start(preview) |
| 输出音频、配音轨、SRT/LRC、视频、烧录字幕 | ProjectView; subtitles; mix_output_mode; video settings | projects/get媒体；tasks/start(export/subtitles) |
| 多模型复核试听、采纳、保留、撤销、解除锁定、重试、仅对齐 | review_overview/candidate_details/apply_review/undo_review/unlock_review/retry_review/realign_review | review/get/details；统一项目动作 |
| 批量添加作品/扫描/版本/特典/格式/轨道顺序 | autoflow.ui_services.scan_for_ui/preview_edition_for_ui/reorder_tracks_for_ui | batch/scan/edition/tracks |
| 每轨字幕文件/语言/直接导入或识别后校对 | set_track_subtitle_for_ui | batch/subtitle |
| 批量成品/格式/多轨/画面/烧录字幕/重新处理 | build_plan_for_ui; SmartTaskPlan; output_policy | batch/plan |
| 队列编辑/删除/调整顺序/重试 | edit_plan_for_ui/replace_plan_in_queue/remove_plan_from_queue/reorder_queue_for_ui/toggle_plan_rebuild | batch/edit/save/remove/reorder |
| 顺序执行/暂停/恢复/完成/失败 | run_queue; CancellationToken; core checkpoint | tasks/start(batch), cancel/resume；持久队列 |
| 等你确认：试听并选择项目句/编辑时序文字/外部参考 | engine reference events; select_autoflow_project_reference/external_reference | batch/reference；保留等待超时行为 |
| 模型组、硬件、磁盘、已占用、下载/暂停/恢复 | model_registry; detect_hardware; backend_model_status; install_backend | models/list；tasks/start(download) |
| ASMR VAD、Qwen3、分离 MelBand/Demucs | download_optional_asr_models; separation.prepare_local_model | 同一下载任务 |
| 导入离线模型包 | discover_model_packs/import_discovered_model_packs | tasks/start(import_models) |
| 删除模型 | 实际托管模型目录/cache路径 | models/remove，限定目标且禁止运行中删除 |
| 国内镜像/海外源、HF/Python源 | mirrors; huggingface_endpoint; pypi_index_url | settings/update；下载任务捕获来源 |
| 界面语言、项目保存位置、启动自动打开浏览器 | locales/en.json；projects_root；新增界面偏好 | settings/get/update；现有 locales 机制 |
| 云端服务填写/更换/清除Key/接口测试 | PROVIDER_PRESETS; save/clear_api_key; save/clear_service_key; backend_diagnostics | settings/key；tasks/start(diagnostic) |
| 新项目默认值、项目面板、条件显示 | ProjectSettings/UserSettings 全字段，见下表 | parameters：同一清单生成两种表单与校验 |
| 存储占用、缓存预览/勾选/确认清理 | cache_cleanup.scan_caches/clean_caches | storage/scan/clean，保留原保护规则 |
| 环境检查、VC运行库/后端修复、日志导出 | check/repair_runtime_health; recent_log_text/application_log_path | tasks/start(health/repair)；logs/get |

## 2. 接口与数据契约
- GET /api/bootstrap：参数清单、全局设置、模型/服务选项、版本、文案、任务、队列。
- POST /api/<域>/<动作>：JSON object；错误返回 {error,detail}，成功返回 JSON。
- POST /api/uploads：媒体/字幕/参考文件原始字节，原文件不修改。
- GET /media/<随机凭据>：只访问服务注册的媒体，支持 Range；不暴露任意本机路径。
- 项目写操作必须带 revision；沿用 lifecycle 的乐观锁，失败后前端保留本地草稿。
- 长任务统一 {id,kind,resource,status,current,total,message,result,error,request}；start/list/get/cancel/resume。
- 同一项目写入串行；下载全局串行；重启后 running 标成 interrupted；resume 重放保存的请求，依赖核心检查点保留已完成结果。
- 批量队列/任务快照放 portable_home/config 和 runtime；密钥只走现有 secrets.json，不回显或写任务日志。

## 3. 参数清单的数据结构
- ProjectSettings / UserSettings 是类型、默认值、范围和核心校验的唯一来源。
- 服务 parameters 将 model_json_schema 的每个字段与一份展示清单合并：
  {key,type,default,minimum,maximum,choices,scope,panel,group,label,hint,visible_when,basic}。
- 展示清单只定义分组、mockup 标签和条件，数值不复制；label/hint/choices 从 locales/en.json 生成英文。
- 前端按服务下发清单生成项目表单和默认值表单，不重新列字段或默认值。
- source_language 属于项目；default_source_language 属于全局；target_language 沿用核心现有字段。
- 非设置的交互值（导入种类、任务参数、字幕导出选择等）随请求校验并记录在任务中。

## 4. 全参数覆盖清单
下表包含现有所有 UserSettings 字段。项目共有字段同时进入项目面板与新项目默认值；全局字段仅进入设置/批量默认值。
| 键 | 现有控件文案 / 来源 | 范围 |
| --- | --- | --- |
| separation_enabled | 启用人声分离（实验性，不推荐） | 项目及默认值 |
| separation_backend | 分离后端 | 项目及默认值 |
| separation_model | 选择分离模型 | 项目及默认值 |
| separation_device | 分离设备 | 项目及默认值 |
| separation_chunk_seconds | 外层分块秒数 | 项目及默认值 |
| separation_overlap_seconds | 分块上下文秒数 | 项目及默认值 |
| separation_timeout_seconds | 每块超时秒数 | 项目及默认值 |
| separation_use_autocast | 自动混合精度 | 项目及默认值 |
| separation_normalization | 分离归一化峰值 | 项目及默认值 |
| separation_vocal_stem | 作为人声的输出轨名称 | 项目及默认值 |
| separation_common_params | 通用高级参数（JSON） | 项目及默认值 |
| separation_mdx_params | MDX 参数 | 项目及默认值 |
| separation_vr_params | VR 参数 | 项目及默认值 |
| separation_demucs_params | Demucs 参数 | 项目及默认值 |
| separation_mdxc_params | Mel/BS-RoFormer 与 MDX23C 参数 | 项目及默认值 |
| separation_api_url | 自建 API 完整地址 | 项目及默认值 |
| separation_api_model | 云端模型／固定版本 ID | 项目及默认值 |
| separation_api_audio_field | 输入音频字段名 | 项目及默认值 |
| separation_api_output_field | 人声 URL 的 JSON 字段路径 | 项目及默认值 |
| separation_api_params | 云模型其余参数（JSON） | 项目及默认值 |
| separation_cloud_consent | 同意上传音频到所选分离服务（实验性，不推荐） | 项目及默认值 |
| separation_mix_mode | 混音方式 | 项目及默认值 |
| separation_keep_original | 替换模式的原声回填 | 项目及默认值 |
| separation_keep_ids | 手动保留原声的句子 ID | 项目及默认值 |
| separation_keep_padding_ms | 回填前后保留毫秒 | 项目及默认值 |
| spatial_rtf_enabled | 启用 RTF（原声空间线索迁移） | 项目及默认值 |
| spatial_rtf_strength | RTF 干湿比例 | 项目及默认值 |
| spatial_rtf_level_strength | 相对远近电平变化强度 | 项目及默认值 |
| spatial_rtf_color_strength | 相对频谱染色强度 | 项目及默认值 |
| spatial_rtf_fft_size | FFT 窗口 | 项目及默认值 |
| spatial_rtf_hop_divisor | FFT 步长除数 | 项目及默认值 |
| spatial_rtf_block_seconds | RTF 内存分块秒数 | 项目及默认值 |
| asr_backend | ASR（语音识别）后端 | 项目及默认值 |
| asr_model | ASR（语音识别）模型 | 项目及默认值 |
| asr_batch_size | 批大小 | 项目及默认值 |
| asr_device | 识别设备 | 项目及默认值 |
| asr_compute_type | 计算精度 | 项目及默认值 |
| asr_beam_size | 束搜索宽度（Beam Size） | 项目及默认值 |
| asr_vad_filter | asr_vad_filter | 项目及默认值 |
| asr_vad_mode | VAD（语音活动检测）预处理 | 项目及默认值 |
| asr_vad_min_silence_ms | VAD 最短静音毫秒 | 项目及默认值 |
| asr_asmr_vad_threshold | ASMR VAD 语音阈值 | 项目及默认值 |
| asr_asmr_vad_min_speech_ms | ASMR VAD 最短语音毫秒 | 项目及默认值 |
| asr_asmr_vad_min_silence_ms | ASMR VAD 最短静音毫秒 | 项目及默认值 |
| asr_asmr_vad_speech_pad_ms | ASMR VAD 边界保留毫秒 | 项目及默认值 |
| aligner_model | aligner_model | 项目及默认值 |
| asr_forced_alignment_enabled | 识别后使用 Qwen3 ForcedAligner 0.6B（阿里）重新计算时间戳 | 项目及默认值 |
| asr_condition_on_previous_text | 使用上一段文字作为识别条件 | 项目及默认值 |
| asr_initial_prompt | 识别提示词（人名、作品名或特殊读法） | 项目及默认值 |
| asr_timeout_seconds | Parakeet 连续无响应超时（秒） | 项目及默认值 |
| asr_api_base_url | 通用 ASR API（接口）基础地址 | 项目及默认值 |
| asr_api_extra_body | 通用 ASR API 附加请求参数（JSON） | 项目及默认值 |
| asr_parakeet_decoder | Parakeet 解码头 | 项目及默认值 |
| asr_chunk_seconds | Parakeet 分块秒数（15–600） | 项目及默认值 |
| asr_kotoba_chunk_seconds | Kotoba-Whisper 分块秒数（5–120） | 项目及默认值 |
| asr_review_enabled | 启用统一音频片段复核 | 项目及默认值 |
| asr_review_mode | 复核结果处理方式 | 项目及默认值 |
| asr_review_window_seconds | 统一音频片段目标秒数 | 项目及默认值 |
| asr_review_context_seconds | 复核音频上下文秒数 | 项目及默认值 |
| asr_review_models | 复核模型 | 项目及默认值 |
| asr_review_text_priority_model | 主文字来源 | 项目及默认值 |
| asr_review_timestamp_priority_model | 最终时间戳来源 | 项目及默认值 |
| asr_review_background | 作品、人物与场景背景 | 项目及默认值 |
| asr_review_prompt | ASR（语音识别）校对提示词（Prompt） | 项目及默认值 |
| asr_review_max_drift_seconds | 允许时间漂移秒数 | 项目及默认值 |
| pause_split_seconds | 停顿切句秒数 | 项目及默认值 |
| max_sentence_seconds | 单句最长秒数 | 项目及默认值 |
| translation_provider | 翻译服务 | 项目及默认值 |
| translation_model | 翻译模型 | 项目及默认值 |
| translation_base_url | 翻译 API（接口）基础地址 | 项目及默认值 |
| translation_prompt | translation_prompt | 项目及默认值 |
| tts_target_language | 配音目标语言 | 项目及默认值 |
| translation_temperature | 随机度（Temperature） | 项目及默认值 |
| translation_top_p | 核采样概率（Top P） | 项目及默认值 |
| translation_max_output_tokens | 最大输出词元数（Token） | 项目及默认值 |
| translation_send_context | 发送相邻句上下文 | 项目及默认值 |
| translation_context_sentences | 上下文句数 | 项目及默认值 |
| translation_memory_sentences | 翻译记忆句数 | 项目及默认值 |
| translation_deepl_formality | DeepL 正式程度 | 项目及默认值 |
| translation_microsoft_region | Azure Translator 区域 | 项目及默认值 |
| translation_extra_body | 翻译 API 附加请求参数（JSON，可选） | 项目及默认值 |
| tts_backend | TTS（语音合成）后端 | 项目及默认值 |
| tts_model | TTS（语音合成）模型 | 项目及默认值 |
| tts_device | 合成设备 | 项目及默认值 |
| tts_reference_source | 参考音频来源 | 项目及默认值 |
| tts_external_reference_audio | tts_external_reference_audio | 项目及默认值 |
| tts_external_reference_text | 外部参考音频对应原文 | 项目及默认值 |
| tts_external_reference_language | 外部参考音频语言 | 项目及默认值 |
| tts_api_base_url | TTS（语音合成）API（接口）基础地址 | 项目及默认值 |
| tts_api_extra_body | TTS（语音合成）API 附加请求参数（JSON，可选） | 项目及默认值 |
| tts_timeout_seconds | 单句超时秒数 | 项目及默认值 |
| tts_request_concurrency | 外部 API（接口）并发数 | 项目及默认值 |
| tts_model_path | 模型权重目录（checkpoints） | 项目及默认值 |
| tts_config_path | 配置文件（config.yaml） | 项目及默认值 |
| tts_executable | tts_executable | 项目及默认值 |
| tts_speed | 语速 | 项目及默认值 |
| tts_voice | 音色 ID | 项目及默认值 |
| tts_volume | 音量 | 项目及默认值 |
| tts_pitch | 音调 | 项目及默认值 |
| tts_emotion | 情绪 | 项目及默认值 |
| tts_style_prompt | MiMo 语气与风格说明（可选） | 项目及默认值 |
| tts_temperature | 随机度（Temperature） | 项目及默认值 |
| tts_top_p | 核采样概率（Top P） | 项目及默认值 |
| tts_index_use_fp16 | 使用半精度计算（FP16） | 项目及默认值 |
| tts_index_emo_alpha | 情绪权重 | 项目及默认值 |
| tts_index_speaker_source | 音色参考来源 | 项目及默认值 |
| tts_index_emotion_source | 情绪参考来源 | 项目及默认值 |
| tts_index_external_emotion_audio | tts_index_external_emotion_audio | 项目及默认值 |
| tts_index_emo_text | 情绪文字描述 | 项目及默认值 |
| tts_index25_model_path | 模型权重目录（checkpoints） | 项目及默认值 |
| tts_index25_config_path | 配置文件（config.yaml） | 项目及默认值 |
| tts_index25_language | 合成语言 | 项目及默认值 |
| tts_index25_use_bf16 | BF16 半精度 | 项目及默认值 |
| tts_index25_use_cuda_kernel | BigVGAN CUDA 内核 | 项目及默认值 |
| tts_index25_use_deepspeed | DeepSpeed | 项目及默认值 |
| tts_index25_use_accel | GPT 加速引擎 | 项目及默认值 |
| tts_index25_use_torch_compile | torch.compile | 项目及默认值 |
| tts_index25_emotion_vector | tts_index25_emotion_vector | 项目及默认值 |
| tts_index25_use_random | 随机选择情绪/音色条件 | 项目及默认值 |
| tts_index25_interval_silence_ms | 内部切段间隔（毫秒） | 项目及默认值 |
| tts_index25_max_text_tokens | 单段最大文本 Token | 项目及默认值 |
| tts_index25_duration_factor | 时长倍率 | 项目及默认值 |
| tts_index25_text_normalization | 文本规范化 | 项目及默认值 |
| tts_index25_do_sample | 启用采样 | 项目及默认值 |
| tts_index25_temperature | 随机度（Temperature） | 项目及默认值 |
| tts_index25_top_p | 核采样概率（Top P） | 项目及默认值 |
| tts_index25_top_k | 候选数（Top K） | 项目及默认值 |
| tts_index25_num_beams | 束搜索数量（Beams） | 项目及默认值 |
| tts_index25_repetition_penalty | 重复惩罚 | 项目及默认值 |
| tts_index25_length_penalty | 长度惩罚 | 项目及默认值 |
| tts_index25_max_mel_tokens | 最大声学 Token | 项目及默认值 |
| tts_gpt_top_k | 候选数（Top K） | 项目及默认值 |
| tts_gpt_text_split_method | 文本切分方法 | 项目及默认值 |
| tts_gpt_sample_steps | 采样步数 | 项目及默认值 |
| tts_cosyvoice_mode | CosyVoice 模式 | 项目及默认值 |
| tts_clone_mode | 参考策略 | 项目及默认值 |
| tts_reference_sentence_id | tts_reference_sentence_id | 项目及默认值 |
| chinese_dubbing_offset_ms | 中文配音整体偏移（毫秒） | 项目及默认值 |
| chinese_max_auto_speed | 冲突时最大自动加速倍速 | 项目及默认值 |
| chinese_dubbing_timing_mode | 中文配音排程方式 | 项目及默认值 |
| chinese_gain_db | chinese_gain_db | 项目及默认值 |
| normalize_chinese_loudness | normalize_chinese_loudness | 项目及默认值 |
| match_source_loudness | match_source_loudness | 项目及默认值 |
| chinese_relative_loudness_db | chinese_relative_loudness_db | 项目及默认值 |
| chinese_min_active_rms_dbfs | chinese_min_active_rms_dbfs | 项目及默认值 |
| chinese_target_active_rms_dbfs | chinese_target_active_rms_dbfs | 项目及默认值 |
| chinese_max_loudness_boost_db | chinese_max_loudness_boost_db | 项目及默认值 |
| chinese_line_peak_dbfs | chinese_line_peak_dbfs | 项目及默认值 |
| chinese_stem_peak_dbfs | chinese_stem_peak_dbfs | 项目及默认值 |
| chinese_fade_ms | chinese_fade_ms | 项目及默认值 |
| replacement_chinese_gain_db | replacement_chinese_gain_db | 项目及默认值 |
| replacement_normalize_chinese_loudness | replacement_normalize_chinese_loudness | 项目及默认值 |
| replacement_match_source_loudness | replacement_match_source_loudness | 项目及默认值 |
| replacement_chinese_relative_loudness_db | replacement_chinese_relative_loudness_db | 项目及默认值 |
| replacement_chinese_min_active_rms_dbfs | replacement_chinese_min_active_rms_dbfs | 项目及默认值 |
| replacement_chinese_target_active_rms_dbfs | replacement_chinese_target_active_rms_dbfs | 项目及默认值 |
| replacement_chinese_max_loudness_boost_db | replacement_chinese_max_loudness_boost_db | 项目及默认值 |
| replacement_chinese_line_peak_dbfs | replacement_chinese_line_peak_dbfs | 项目及默认值 |
| replacement_chinese_stem_peak_dbfs | replacement_chinese_stem_peak_dbfs | 项目及默认值 |
| replacement_chinese_fade_ms | replacement_chinese_fade_ms | 项目及默认值 |
| chinese_channel_routing | 多声道路由 | 项目及默认值 |
| mix_peak_protection | 最终混音峰值保护 | 项目及默认值 |
| mix_peak_limit_dbfs | 最终峰值上限（dBFS） | 项目及默认值 |
| mix_output_mode | 音频输出方式 | 项目及默认值 |
| skip_japanese_fillers | 日语项目跳过纯语气词 | 项目及默认值 |
| reference_padding_seconds | 参考音频边缘扩展秒数 | 项目及默认值 |
| random_seed | 随机种子 | 项目及默认值 |
| subtitle_timeline | 字幕时间轴 | 项目及默认值 |
| subtitle_max_chars_per_line | 每行最多字符（8–500） | 项目及默认值 |
| subtitle_min_duration_seconds | 最短显示秒数 | 项目及默认值 |
| subtitle_max_cps | 最大每秒字符数 | 项目及默认值 |
| projects_root | 项目保存目录 | 全局 |
| huggingface_endpoint | Hugging Face 下载端点 | 全局 |
| pypi_index_url | Python 软件源 | 全局 |
| default_source_language | 新建媒体项目的音频语言 | 全局 |
| tts_device_selection_version | tts_device_selection_version | 全局 |
| translation_prompt_ja | translation_prompt_ja | 全局 |
| translation_prompt_en | translation_prompt_en | 全局 |
| translation_prompt_zh | translation_prompt_zh | 全局 |
| autoflow_output_folder_name | 成品输出文件夹名称 | 全局 |
| autoflow_default_mode | 默认输出类型 | 全局 |
| autoflow_default_layout | 默认成品组织 | 全局 |
| autoflow_include_bonus | 默认包含附加音轨 | 全局 |
| autoflow_background_policy | 默认视频画面 | 全局 |
| autoflow_embed_subtitles | 默认在视频中内嵌字幕 | 全局 |
| autoflow_subtitle_language | 仅字幕文件的默认内容 | 全局 |
| autoflow_subtitle_naming | 字幕文件命名（仅字幕文件模式） | 全局 |
| autoflow_subtitle_custom_name | 自定义字幕名称（不含扩展名） | 全局 |
| autoflow_original_hard_subtitles | 原声视频也编码硬字幕 | 全局 |
| autoflow_timestamp_footer_position | 时间戳文档附加文字位置 | 全局 |
| autoflow_translate_work_title | 翻译作品文件夹名称 | 全局 |
| autoflow_translate_track_titles | 翻译音轨标题 | 全局 |
| autoflow_reference_wait_enabled | 处理到参考音频时等待手动选择 | 全局 |
| autoflow_reference_wait_seconds | 每个作品最多等待（秒） | 全局 |
| autoflow_preferred_audio_formats | 同一作品多种音频格式时的选择顺序 | 全局 |
| autoflow_harmonized_volume_reduction_db | 原声降低音量（dB） | 全局 |
| autoflow_harmonized_delay_minutes | 原声、配音和字幕整体延后（分钟） | 全局 |
| autoflow_timestamp_footer | 时间戳文档页脚 | 全局 |

## 5. 启动与打包
- ui.py 最终仅启动标准库 HTTP 服务；保留 CLI ui 命令及 7860 默认端口、产品标记、端口复用与子进程生命周期。
- Windows launcher 自动准备基础 Python/依赖，不再要求用户先启动 Setup 或选安装方案；可用基础环境立即打开前端。
- 便携包包含基础 Python、基础运行依赖、前端静态文件、FFmpeg；模型在模型页按需下载。
- 更新 dependency inventory / setup / CI / release 中 Gradio 依赖检查及锁文件；移除仅服务旧界面的资源/脚本。
- 不删除仍供模型安装使用的底层安装脚本，不发布压缩包或推送。

## 6. 分步验收与提交
1. 计划与覆盖表：全量测试通过，提交 PLAN。
2. 抽取服务、参数与 JSON 接口：新增服务/接口/任务/文件访问测试，全量通过，提交。
3. 原生前端六个页面及补充面板：浏览器检查中文/英文、表单、任务、项目流；全量通过，提交。
4. 解压即用启动/打包：启动器编译、自检和真实启动；全量通过，提交。
5. 覆盖审计完成后移除 Gradio 代码/依赖；旧测试迁移到对应服务，不删测试、不削弱行为断言；全量通过，提交。

## 待确认
- mockup 两种配音语言以核心现有 target_language 支持集合为准，不新造翻译行为。
- mockup 未画到的高级字段统一保留在对应步骤的“更多设置”，不删除。
- 模型 registry 没有精确权重大小的项显示实际已用空间或未知，不使用 mockup 示意数字。
- “暂停”使用已有取消及检查点，恢复重新启动同一动作；不承诺所有第三方网络请求都能瞬时结束。
- mockup 缓存确认属于产品交互，仍须展示真实删除预览；本次实施无需中途等待开发确认。
- 现有超过 500 行的核心模块与已有用户文档不在重构范围，不为行数要求改变核心行为。

## 发现的现有 bug
- 暂无确认的新 bug；基线全部测试通过。发现后只记录，不顺手修改核心。

## 实际验证记录
- 基线及步骤 1 pytest：646 passed, 3 skipped；步骤 1 提交 7686ecf。
- 步骤 2：新增 20 项服务/HTTP 验证，首次完整验证 666 passed, 3 skipped；服务/接口 Pyright 0 errors，Ruff 通过。
- 模型安装沿用原有后端粒度：Parakeet 运行环境包含两款权重；可选 VAD/对齐复用已有进阶依赖安装路径。
- 原始 Hugging Face 快照下载和部分离线解包没有细粒度取消回调：取消信号保留，第三方调用返回后结束任务，不改变核心。
