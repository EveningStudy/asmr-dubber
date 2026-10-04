# 真实环境验收报告

日期：2026-10-04。未推送，未修改版本号或用户文档。

## 环境与证据边界

- 原始测试提交：`d7c00e1029ee6111179551c90860e41ffeada246`；修复后源码：`d336ac6`。
- 从已提交源码打便携包，解压至全新短路径 `D:\AT01\ASMR-Dubber`，直接运行 EXE；使用独立 `.asmr-dubber`，未复用开发目录数据。
- 修复版重新打包：`D:\AT01\fixed-packages\ASMR-Dubber-Windows-v1.6.2.zip`；再次解压到全新 `D:\AT02\ASMR-Dubber`，EXE 首次启动验证通过。SHA256 见 `final-package-hash.json`。
- Windows，RTX 5070 Ti Laptop，11.9 GiB 显存；FFmpeg 7.1。测试期间可用磁盘逐渐降至约 3 GB。
- 素材：`C:\Users\Chris\Downloads\#1.叫来自宅的上门服务JK.wav`，159 秒、48 kHz、立体声。DeepSeek 使用已保存 Key，经设置 JSON 接口写入测试环境；证据不记录 Key。
- 产品操作全部经浏览器或同一 HTTP JSON 接口。外部测试脚本未导入产品内部 Python 函数。模型、识别、翻译、配音、分离均真实运行。
- 以下证据文件未写绝对路径时，均位于 `D:\AT01\evidence`。完整任务 JSON、截图、FFmpeg 日志及测试脚本保留在那里。
- 测试阶段先记录问题，结束后逐项编写失败回归、修复、全量测试、独立提交，再重跑受影响流程。

## 12 个流程结果

“通过”表示相关修复后的实际复跑结果；初次失败保留在问题清单。主项目目录为 `D:\AT01\ASMR-Dubber\.asmr-dubber\projects\#1.叫来自宅的上门服务JK_20261004T181426Z`。

| # | 流程 | 结果 | 实际验证与证据 |
|---|---|---|---|
| 1 | 首次启动 | 通过 | AT01 原包与 AT02 修复包均直接运行 EXE，首页“开始之前”出现；AT02 没有历史项目或 Key。`01-first-start.png`、`first-browser-log.json`、`final-fresh-start.png/json`、`final-fresh-process.json`；浏览器错误数组为空。 |
| 2 | 下载、取消、续传、重启 | 通过 | Parakeet 1.1B 与 IndexTTS2 真实安装。IndexTTS2 在 35.6% 取消，保留 4,022,337,536 字节；续传从 36.0% 开始，保留片段前缀哈希一致。重启仍已安装。`indextts-partial-before.json`、`indextts-partial-cancelled.json`、`indextts-cancelled.json`、`indextts-resume-proof.json`。数字进度修复后用真实 Roformer 下载验证 12.5/100 与日志一致，未再次完整下载 11 GB IndexTTS2。Parakeet 首次取消到达校验阶段，不作为半途下载证据。 |
| 3 | 单项目全流程、空间跟随 | 通过 | 159 秒原音频→Parakeet 54 句→DeepSeek 三批翻译→41 句 IndexTTS2 CUDA 配音→RTF 双语导出。`main-translated.json`、`main-export-manifest.json`、`final-bilingual-manifest.json`。完整产物保留于 `D:\AT01\evidence\artifacts\bilingual`；159 秒，峰值 -7.659417 dBFS、RMS -39.639388 dBFS，无静音/削波。RTF 一句参考过短发生保守降级，详见限制。 |
| 4 | 配音取消、关闭、恢复 | 通过 | 已生成 27/41 句后取消，关 EXE、重开、恢复相同任务，只生成余下 14 句。`resume-proof.json`：27 句原文件哈希不变，最终 41 句。任务 `a95cbde21f0e42379f57c6e043981cfb`。 |
| 5 | 编辑一句、禁用一句 | 通过 | 修改 s000002 后仅这一句文件哈希变化，任务生成 1 句；禁用 s000003 后字幕不含该译文，中文轨 7.5–8.9 秒峰值 -inf。`edit-proof.json`、`disabled-proof.json`、`disabled-output-verification.json`、`disabled-segment-ffmpeg.txt`。双语版仍保留日语原声，属预期。 |
| 6 | 参数持久化、默认值隔离、输出影响 | 通过 | 修改识别分块 35、翻译温度 0.2、TTS top_p 0.85、RTF 强度 0.8、字幕每行 8 字，重新打开与磁盘一致。新默认 30 字只作用新项目，原项目仍 8。偏移改变 0.5 秒使中文起音 4.197188→4.697188；字幕最大行长 22→8，行数 91→153。`advanced-persistence.json`、`defaults-isolation.json`、`parameter-effects.json`、`offset-before/after.wav`、`subtitle-before/after.srt`。 |
| 7 | 分离、替换版 | 通过 | 修复 FFmpeg 后真实下载校验 870.8 MB Roformer；完整 159 秒、6 分块 CUDA 分离并导出替换版。产物 `D:\AT01\evidence\artifacts\replacement`，159 秒、峰值 -11.237365、RMS -39.952950 dBFS。`replacement-final-task.json`、`replacement-final-manifest.json`、对应 FFmpeg 日志。 |
| 8 | 三种字幕导入 | 通过 | 从真实素材截取 18 秒：带时间日语 SRT→翻译→配音→导出；中文 SRT→配音→导出；无时间日语台本→真实 ASR/LLM 复核→翻译→配音→导出。项目分别 `01-original_20261004T182256Z`、`182346Z`、`182431Z`，均在 AT01 私有 projects。峰值分别 -21.174006/-20.994535/-21.131758，RMS -40.090919/-40.292198/-39.996003 dBFS。`import-source.srt`、`import-zh.srt`、`import-plain.txt`、`verified-import-*-ffmpeg.txt`。 |
| 9 | 英语/Faster-Whisper | 有问题 | 没有英语素材，按允许方式用日语音频验证英文设置。large-v2 安装首先暴露运行环境覆盖错误，已修；真实重试之后受磁盘空间不足阻断，完整推荐安装与 large-v2 GPU 识别未跑完。另经接口选择 small/CPU，真实下载并成功识别 18 秒得到 7 句，int8；任务 `0fbb3eb4f7ce4ffba6316e4830c5e7b2`。`english-small-settings.json`、`english-small-recognition.json`、`final-english-project.json`、`english-project-*.png`。不代表英语识别准确率验证。 |
| 10 | 批量、合并、换参考 | 通过 | 自建 `D:\AT01\work`，两条真实 18 秒音轨。参考等待期间经接口换成 s000012，继续实际配音并完整跑完队列。任务 `92880c8b559a4f1ca0241ad650643551` exit 0。输出 `D:\AT01\work\AutoFlow输出\合并版\双语版\双语版.wav`，36.53 秒，峰值 -14.135329、RMS -40.200446 dBFS。`batch-rerun-task.json`、`batch-reference-switched-0.json`、`final-batch-queue.json`、`batch-双语版-ffmpeg.txt`。 |
| 11 | 存储清理 | 通过 | 经真实浏览器扫描、确认、清理，最终清理 71,232,220 字节，显示“Cleared 0.07 GB”；128 个项目/源音频/配音/字幕/成品/批量及保留产物哈希全部不变。`cleanup-ui-final.txt`、`ui-cleanup-protected-before.json`、`ui-cleanup-preservation.json`：changed=[]。 |
| 12 | English 全页面、布局 | 通过 | 项目四步、首页、模型、批量、设置九分组及四个移动页面，共 21 个截图；浏览器错误为空。1440/900/390 宽度回归通过；最终截图桌面文档宽 1440、移动 390，无整页横向溢出。`english-qa.json`、`english-final-console.txt`、`qa-en-*.png/txt`。静态 mockup 对照见 `mockup-home.png`、`01-first-start.png`、`pages-project-3.png`，侧栏/卡片/步骤结构和文案检查完成，未做所有状态逐像素比较。 |

## 问题、修复与提交

| 问题 | 现象与复现 | 原因 | 回归测试 / 修复提交 | 实际复跑 |
|---|---|---|---|---|
| A1 下载数字进度失真 | 下载日志已有百分比，任务仍显示 0/0；无最终回调的成功任务进度未完成 | 未将下载日志解析为任务数字进度 | `test_acceptance_download_progress.py`；`6cc1121` | 真实 Roformer 下载数字进度匹配日志，完成状态正确 |
| A2 分离模型安装失败 | 安装后检查 `ffmpeg -version` 返回 3221225781 | 将 FFmpeg EXE 单独复制，离开其 DLL 目录 | `test_acceptance_separation_ffmpeg.py`；`d5fb349` | 真模型下载、6 块分离、替换版导出成功 |
| A3 进阶依赖损坏运行环境 | 下载 Faster-Whisper 安装覆盖 `_cffi_backend.cp312-win_amd64.pyd` 报 WinError 5，重启核心环境损坏 | 进阶包合并覆盖已加载的基础分发及 ABI 文件 | `test_windows_dependency_pack.py` 新增已加载文件/完整分发保留回归；`cdb4b55` | 保留破损 venv 后恢复原包 core，真实重试越过原错误；small 真实识别成功。完整大包仍受磁盘限制 |
| A4 英文动态文案缺漏 | 英文模式仍出现中文模型/任务/Key 状态、句子音频控件、诊断和队列状态 | 动态模板缺少本地化，泛化规则遮挡具体模板 | `test_acceptance_english.py` 运行真实前端 JS；`f8c6132`、`6bf3dee` | 21 页面截图、浏览器日志检查通过 |
| A5 批量英文文件名阻断 | `01-original.wav` 等名称进入 LLM，返回空译文，队列失败 | 将纯拉丁标题按语音正文空翻译规则处理 | `test_acceptance_batch_titles.py`；`dd78092` | 真队列换参考并完成合并。仅纯拉丁标题空译文保留原名并警告，日语空译文仍报错 |
| A6 窄窗口溢出 | 英文页面最小 1180 宽，移动端语言切换不可达 | 固定最小宽度，长控件不换行 | `tests/native_viewport.cjs` 浏览器先复现后验证；`861e143` | 1440/900/390，所有设置分组无整页溢出 |
| A7 浏览器清理无效 | 扫描后立即确认却提示文件已更新，清理 0 字节 | JS JSON 将 >2^53 的时间戳/inode 整数舍入 | `test_acceptance_cleanup_precision.py`、真实浏览器 `native_cleanup.cjs`；`881ae80` | 真实浏览器清理成功，128 个保留文件哈希不变 |
| A8 清理单位错误 | 显示“Cleared 101389729 files” | 将清理字节数作为文件数展示 | `test_acceptance_english.py`、`native_cleanup.cjs`；`5af1b02` | 实际 71,232,220 字节显示 0.07 GB |
| A9 磁盘容量阻断 | large-v2 进阶包重试解压/合并报 No space left on device | 完整约 2.7 GB 进阶依赖包及大量已保留模型/产物需要额外空间 | 未改程序；未删除保留证据或伪造安装完成 | small 识别已跑；完整 large-v2 安装/GPU 识别未跑完 |

另提交 `d336ac6` 修正本轮代码的类型收窄和格式问题，Pyright/Ruff 通过。每项代码修复均先有失败回归，全量测试通过后提交；问题未与先前界面重构或旧改动混合。

### 环境维护与测试操作说明

A3 初始安装损坏后，将破损环境保留为 `D:\AT01\broken-core-venv`，仅用原始新包恢复核心 venv，保持测试数据/模型/项目；将已提交修复代码更新进 AT01。所有后续产品动作仍走 UI/JSON。AT01 的完整模型复跑因此包含这一维护步骤；AT02 是最终完整重打便携包的全新启动验证，未在 AT02 再下载全部模型和跑全长推理。

以下是测试顺序或预期状态，不列为产品 bug：修改 TTS 采样参数会使旧配音失效，必须重新配音或恢复参数再导出；任务运行时清理接口拒绝操作；语言/文本导入产生相应阶段失效。主项目保留参数实验引起的识别失效提示，符合设置变化逻辑。

## 自动化及文件质量验证

- 每项修复的全量测试证据：`fix01-full-tests.txt` 至 `fix09-full-tests.txt`，最终 `final-full-tests.txt`。
- 最终 `python -m pytest -q`：**687 passed, 3 skipped**。既有测试没有删掉或为了通过改变断言；增加回归覆盖。跳过项是环境权限/可选真实模型测试，不计作已通过的验收流程。
- 从本地已提交 Git 内容独立克隆到 `D:\Temp\asmr-committed-verification-20261004-1854`，更新到 `d336ac6` 后运行全量：**687 passed, 3 skipped，32.16 秒**；工作树为空。`committed-clone-tests.txt`。
- `python -m ruff check .`、格式检查：全部通过，162 文件已格式化；`python -m pyright`：0 errors。干净克隆亦通过，见 `committed-clone-pyright.txt`。
- 浏览器回归：设置 `ASMR_TEST_PLAYWRIGHT` 指向本机 Playwright，运行 `node tests/native_viewport.cjs`、`node tests/native_cleanup.cjs`，使用真实 Edge 和实际启动产品。
- `quality.py` 使用包内 FFmpeg 的 astats、silencedetect 检查时长、峰值、RMS、起音；`quality.jsonl` 保留各输出路径及数值，不只检查文件存在。
- 61 个实际逐句 TTS WAV 检测：silent=[]、clipping=[]，最大峰值 -16.345247 dBFS；`sentence-quality-summary.json`。全长及批量输出时长合理，峰值 <0 dBFS。
- 每步检查接口状态/任务日志与磁盘 manifest/文件；浏览器截图和报错监听覆盖实际经过的页面。原始失败日志保留，修复后相关报错已排除，容量失败仍明确保留。

## 未验证、需要亲自确认

1. 完整 Faster-Whisper large-v2 推荐安装及 GPU 识别：需增加可用磁盘空间后再跑。本轮未删除测试模型/输出以腾空间；不将 small 成功等同于完整推荐安装成功。
2. 无英语素材，未验证英语内容的识别准确率。已验证英文模型选项/参数显示及 Faster-Whisper small 真实运行。
3. 未进行主观试听：请确认音色相似度、译文准确性、节奏、双语叠加和替换版听感。FFmpeg 非静音/无削波不能替代试听。
4. RTF 已开启并真实处理；s000051 参考过短/信号不足，日志显示降级到受限双声道电平差而非相位估计。这是现有保守行为，未修改；请试听空间效果。
5. English 的诊断原始日志、源字幕、译文、路径及已配置 Prompt 原文保留其内容语言；未强行翻译用户内容。未对所有 mockup 状态做逐像素比较。
6. `design/review-v2` 是不完整替代 UI 草稿（缺 app.js），不是程序运行依赖；保留未提交，建议暂不提交，也不自动忽略设计源文件。`.test-batch-reference` 是旧 Gradio QA 临时产物，已有 `.test-*/` 忽略规则，不提交。

## 保留目录

- 可继续打开的完整测试程序：`D:\AT01\ASMR-Dubber\ASMR-Dubber.exe`，已重启，地址 `http://127.0.0.1:7860/`。
- 完整项目、模型、句子配音：`D:\AT01\ASMR-Dubber\.asmr-dubber`。
- 所有证据及另存双语/替换产物：`D:\AT01\evidence`；批量音轨及成品：`D:\AT01\work`。
- 原包：`D:\AT01\packages`；修复版便携包：`D:\AT01\fixed-packages`。
- 修复版全新解压首次启动目录：`D:\AT02\ASMR-Dubber`（已停止，保留数据）。
- 初始破损环境：`D:\AT01\broken-core-venv`；只靠已提交内容的干净测试克隆：`D:\Temp\asmr-committed-verification-20261004-1854`。

本报告为本轮最终验收记录；未推送任何提交。
