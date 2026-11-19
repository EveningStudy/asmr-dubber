中文 | [English](CHANGELOG.en.md)

# 更新日志

ASMR Dubber 各版本的重要变更记录在此文件中。

## [3.0.0] - 2026-11-18

### 完全配音版 3.0 与音频引擎
- **完全配音版推荐 IndexTTS-2.5，自有 IndexTTS2 Worker**：“完全配音版”预设在已安装 IndexTTS-2.5 时自动切换到它（未安装则提示）。新增独立 `indextts2_worker.py` 并升级 `indextts25_worker.py`，支持原生 S2Mel 目标时长控制（`tts_fit_source_duration`）、`--daemon` 常驻 JSONL 协议与多候选可复现种子。在 7 段评测素材上，起点偏差中位数降至 `38.0 ms`（`p95 = 107.6 ms`），需加速超过 `1.1×` 的句子降至 `8%`。
- **变速仍用 FFmpeg `atempo`，并支持放慢**：生成时贴合原句时长后，需要变速的句子已很少；剩下的用 `atempo` 处理（`0.90×` 放慢至 `1.25×` 加速）。盲听中它比内置相位锁定声码器更干净。
- **正常/耳语双音色锚点与声学贴合**：项目自动分别挑选正常发声和耳语参考锚点（`tts_dual_voice_anchor`）；音色匹配（`voice_match.py`）合并为单次 STFT 处理，含仅针对有声谐波的气声化转换（`replacement_whisper_match = 0.65`）与 `60–260 Hz` 近讲低频贴合。
- **读音复听与残留识别质检闭环**：新增可选的目标语语音识别复听检查（`tts_asr_quality_check`，含普通话音节近似容错与自动重做退避；两次重试仍不达标时保留字错率最低的一版），以及混音后替换窗口内的日语残留识别（`replacement_residual_asr_check`）与人声频段维纳残差抑制（`replacement_residual_suppression`）。
- **双语版空间一致性提升**：多片段空间参考估计与双耳功率响度对齐使双语版耳间声级误差（`ild_error_db`）从 `15.53 dB` 降至 `3.39 dB`，双耳时间差误差从 `0.72 ms` 降至 `0.22 ms`。
- **音色克隆参考改回 16 kHz，消除破音**：全带宽参考会把原声里的气声和口水音带进 IndexTTS，盲听中出现破音、电音；新项目默认 `reference_audio_mode = "legacy"`（16 kHz 单声道）。这样合成的配音 8 kHz 以上是空的，“补全高频”的重建增益提高 2 dB（抵消双耳渲染削掉的高频），评测中 8–12 kHz 与原声相差从 −2.1 dB 缩小到 −0.1 dB。已有项目保留原设置。
- **完全配音版响度按单声道功率对齐**：此前按左右声道功率之和计算原句响度，替换后的配音整体比原声响约 3 dB；现在逐句与原声持平。

### 多说话人区分与分角色配音
- **本地多说话人分割（`asr_diarize_speakers`）**：新增 CPU 端纯 PyTorch `CAMPPlus` 说话人嵌入提取（`campplus.py`，不占显卡显存）与轮廓系数引导的层次余弦聚类（`diarization.py`），支持 MOSS 跨分块说话人拼接。
- **分角色音色克隆与界面微调**：多人项目自动为每位说话人（`S01`、`S02` 等）分别挑选正常与耳语锚点，并在翻译上下文中附带说话人标记；表格内可逐句修改说话人编号。

### 句子编辑器、翻译预算与桌面体验
- **虚拟滚动句子表与交互式迷你波形**：用虚拟滚动取代每页 50 行翻页（上千句流畅滚动），新增逐行独立的原声片段（`▶`）与配音（`▶`）一键试听、带起止拖拽手柄的迷你波形、键盘工作流（`↑/↓` 选句、空格试听、回车编辑）以及 `Ctrl+Z` / `Ctrl+Y` 撤销重做。
- **时长预算翻译、压缩、按要求改译与读音词典**：新增按句时长预算（`translation_length_budget`）、自动与一键压缩过长译文、按要求改译提议审阅、单句多候选配音（`+3`），以及项目级读音词典与中文数字/百分比读法展开（`tts_pronunciation_glossary`、`tts_spoken_normalization`）。
- **本机大文件免复制与程序生命周期**：新增仅限本机访问的原生文件选择（`files/media`）与同盘视频文件的硬链接建项（音频仍复制一份，原文件被标签软件改动也不影响项目）、增量任务轮询（`tasks/poll`）、单行保存 I/O 加速、首次启动向导、质量与成品类型预设、全局任务面板，以及左下角“退出程序”按钮。

### 批量引擎、数据模型 v4 与国内镜像
- **批量进程内执行与常驻模型池**：AutoFlow 批量队列在进程内直接调用流水线函数（`inprocess_pipeline.py`），配合常驻识别会话与常驻 IndexTTS 守护进程池（`tts_daemon.py`），识别和配音模型在整个队列只加载一次（实测 10 个作品时加载约占总时长 10%–14%）；移除 `autoflow/` 中所有交互式 `input()` 与裸 `print()`。批量输出文件夹统一命名为 `完全配音版`（已有的旧 `替换配音版` 成品仍兼容续跑与复用）。
- **项目文件 Schema v4 与代码模块化**：项目清单升级为 `PROJECT_SCHEMA_VERSION = 4`（`Sentence.target_text` 保留 `zh_text` 兼容属性、`SentenceStatus` 枚举、结构化 `error_code` 与解决提示、`ProjectSettings` 分组子模型）；全部源码文件拆分至 `<= 800` 行（`frontend/*.js` 与 `services/*.py` `<= 500` 行）。
- **升级识别模型提醒**：从 2.x 升级且全局日语识别模型仍为 Faster-Whisper / Parakeet / Kotoba-Whisper 时，启动会弹出一次提示，可一键切换到 `anime-whisper`（未安装时自动跳转下载）或保持原设置不再提示。
- **ModelScope 国内镜像扩充**：新增 `anime-whisper`、`qwen3-asr-1.7b` 与 `qwen3-asr-1.7b-ja-anime-galgame` 离线模型包国内镜像。

### 处理速度参考（RTX 5070 Ti Laptop GPU，12 GB 显存）
实测“处理耗时 / 源音频时长”（越小越快）：
- **10 个作品无人值守批量**（共约 9 分钟音频，完全配音版，IndexTTS2）：约 19 分钟，整体约 `2.1×`；10 个作品全部完成。识别和配音模型在整个队列只加载一次，模型加载约占总时长 `10%–14%`（人声分离仍按作品各加载一次）。
- **人声分离**（`Mel-Band RoFormer`）：首次约 `0.25×`；同一项目再次导出时复用。
- **语音识别**（`Faster-Whisper large-v2`）：约 `0.25×`。
- **音色克隆配音**：约为配音句总时长的 `2×`，IndexTTS2 与 IndexTTS-2.5 相近；常驻进程首次加载约 30 秒。
- 翻译取决于所选服务商，混音与音色贴合只占几秒。

### 从 2.2.x / 2.3 升级与旧项目兼容
- `2.2.x` 与 `2.3` 生成的项目（`schema_version = 3`）可直接在 `3.0.0` 中打开、继续翻译、混音和导出；回归测试用的是 2.2.1 原版代码生成的项目。
- 首次保存时自动在项目 `backups/` 下写入原文件备份（`project-schema-v3-*.json`），并升级为 `schema_version = 4`（句子的 `zh_text` 改名为 `target_text`）。升级后的项目不能再用 2.x 打开；要退回旧版，请用这份备份。
- 旧项目自动保留原有的参考音频与停顿/数字处理设置（`reference_audio_mode = "legacy"`、`tts_normalize_pauses = false`、`tts_fit_source_duration = false`、`tts_dual_voice_anchor = false`、`tts_spoken_normalization = false`），**已有配音缓存 100% 继续有效，不会被迫重新合成**。

## [2.2.1] - 上一正式版
- 见 [docs/RELEASE.md](docs/RELEASE.md) 与 `v2.2.1` 标签历史记录。
