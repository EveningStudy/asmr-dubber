# 排障指南

先不要删除 `.asmr-dubber` 或几个 GB 的下载文件。大多数安装和模型下载可以继续，直接清缓存
反而会失去断点。

## 先收集这三项信息

1. 运行环境检查；
2. 最新安装日志或页面错误全文；
3. 当前选择的安装方案、后端、模型、CPU/GPU 和可用磁盘空间。

Windows：

```powershell
.\scripts\windows\run-cli.ps1 doctor --no-network
```

Linux：

```bash
bash scripts/linux/run-cli.sh doctor --no-network
```

Windows Setup 日志位于 `.asmr-dubber/logs/setup-*.log`。找最新一份：

```powershell
Get-ChildItem .\.asmr-dubber\logs\setup-*.log |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
```

提交问题前删掉 API Key、用户名、私人路径、作品名和媒体内容。不要上传整个
`.asmr-dubber`、`secrets.json` 或私人项目。

## Setup 无法启动

确认压缩包已经完整解压，且根目录至少有：

```text
ASMR-Dubber-Setup.exe
ASMR-Dubber.exe
mirrors.json
scripts/
src/
pyproject.toml
uv.lock
```

不要从压缩软件预览窗口运行 EXE。把项目移动到可写的短路径，例如
`D:\Apps\ASMR-Dubber`，再重试。

启动器不要求 PowerShell 7；找不到它时会使用 Windows PowerShell 5.1。如果日志提示找不到
PowerShell，检查下面至少一个文件存在：

```text
C:\Program Files\PowerShell\7\pwsh.exe
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe
```

下载还需要 `curl.exe`：

```powershell
curl.exe --version
```

## Setup 中途失败

直接再次运行 `ASMR-Dubber-Setup.exe`，选择同一方案。安装器会重新检查环境：

- 完整文件先核对字节数和 SHA-256，通过后直接复用；
- 完整文件存在时，会清理对应的无效 `.partial` 标记；
- 未完成文件保留断点，下载源支持 Range 时从已有字节继续；
- 已导入且 manifest 完整的模型不会重新写入；
- 虚拟环境不完整时只修复缺失部分。

不要因为进度重新从“校验”开始就判断它在重新下载。多 GB 文件的 SHA-256 本身需要时间和
磁盘读取，日志出现真正的下载速度和网络地址后才表示正在传输。

常见原因：

- 磁盘剩余空间低于所选方案建议值；
- 杀毒软件占用正在导入的 DLL、wheel 或模型；
- 电脑休眠或网络连接被重置；
- 程序目录没有写权限；
- ModelScope 制品路径、大小或哈希与程序固定清单不一致。

Windows 依赖包包含很深的第三方路径，导入器使用扩展长路径处理。仍出现路径过长时，先把整个
程序移到盘符下的短目录，再运行 Setup；不要手工拆开依赖 ZIP。

## ModelScope 下载失败

默认策略只使用 ModelScope 大文件和配置允许的国内软件源，不会失败后自动转到 GitHub 或
Hugging Face。日志会明确写出当前 URL、期望大小、断点位置和校验结果。

按顺序检查：

1. 在浏览器打开日志中的 ModelScope 文件页面，确认仓库和文件存在；
2. 确认系统日期和 HTTPS 证书正常；
3. 检查代理、防火墙或运营商是否会中断大文件 Range 请求；
4. 保留 `.partial` / `.partial.parts`，重新运行；
5. 如果是维护者，按[ModelScope 制品维护](MODELSCOPE_UPLOADS.md)核对远端路径、大小和
   SHA-256。

公共仓库通常不需要 `MODELSCOPE_API_TOKEN`。只有远端仓库确实要求认证时才在当前终端临时
设置，Token 不应写进 `mirrors.json` 或日志。

如果你明确接受海外流量，可以只为当前进程开启备用源：

```powershell
$env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS = '1'
.\ASMR-Dubber-Setup.exe
```

```bash
ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1 bash scripts/linux/setup.sh 推荐
```

使用代理时设置标准 `HTTP_PROXY` / `HTTPS_PROXY`。不要通过修改 manifest 或删除哈希校验来
绕过损坏文件。

## 有现成下载，Setup 仍想联网

安装器只复用**文件名、大小和 SHA-256 都匹配**的制品。仅有同名文件不够。

可采用以下方式：

- 模型 ZIP 原样放进根目录 `model-packs`；
- Windows 推荐/进阶依赖包按固定文件名放进 `model-packs`；
- 用 `ASMR_DUBBER_LOCAL_CACHE_ROOTS` 指向包含有效下载缓存的只读 ASMR Dubber 目录。

Windows 示例：

```powershell
$env:ASMR_DUBBER_LOCAL_CACHE_ROOTS = 'E:\ASMR-Dubber-Cache'
.\ASMR-Dubber-Setup.exe
```

多个目录在 Windows 用分号分隔，Linux 用冒号分隔。只读缓存不会被安装器修改；校验不通过的
文件会跳过。

## 安装完成，但环境检查仍有警告

Setup 最后会运行 `doctor --no-network`。核心网页能启动而当前后端不完整时，Setup 可能以提示
结束。打开“设置 → 设备与模型”，看具体是哪一项：

- **未安装**：点击该后端的“安装/修复”；
- **模型不完整**：重新下载或导入对应模型包；
- **不兼容**：换 CPU/外部 API，或选择符合硬件的后端；
- **外部服务**：先启动服务端，程序不会替你启动。

安装或导入后点击“重新检测硬件与后端”，Windows 下安装新的本地运行时后建议重启
ASMR Dubber。

## 浏览器没有打开

Windows 启动器从 7860 到 7959 选择空闲端口，并验证返回页面确实属于 ASMR Dubber。查看
启动终端中的实际地址，手工复制到浏览器。

手工指定端口：

```powershell
.\scripts\windows\run-cli.ps1 ui --port 7861
```

```bash
bash scripts/linux/run-cli.sh ui --port 7861
```

如果端口被其它网页占用，关闭对应进程或换端口。防火墙拦截本机 `127.0.0.1` 时，应只放行
本机访问，不要为了省事把页面公开到互联网。

## 移动程序后不能启动

停止程序，确认复制的是整个目录，然后在新位置运行一次 Setup。它会修复主虚拟环境和
IndexTTS2 隔离环境中的绝对路径。

不要单独移动 `.asmr-dubber/venv`、`.asmr-dubber/runtimes/index-tts/.venv` 或模型缓存的
某一层。项目如果配置到外部目录，也要单独搬运并更新“项目保存目录”。

## NVIDIA 显卡存在，但 CUDA 不可用

先检查驱动：

```powershell
nvidia-smi
```

然后运行 `doctor --no-network`，看硬件检测和 PyTorch CUDA 是否都正常。常见处理：

1. 关闭占用显存的游戏、浏览器推理和其它模型服务；
2. 在“设备与模型”重新安装当前后端；
3. 重启 ASMR Dubber，让新的 DLL 和 Python 包进入进程；
4. 批大小改为 1，必要时改用更小模型或 CPU。

不需要安装系统 CUDA Toolkit。驱动必须足够新，PyTorch/CTranslate2 所需用户态运行库由程序
私有环境提供。

## Parakeet 不可用

“推荐”和“进阶”都会准备两个 Parakeet 模型。也可单独修复：

```powershell
.\scripts\windows\install-parakeet.ps1 -Variant Auto
```

```bash
bash scripts/linux/install-parakeet.sh
```

检查 `.asmr-dubber/runtimes/crispasr` 和 `.asmr-dubber/models/parakeet` 是否完整。模型文件
很大，不要用文本编辑器或同步软件改写。修复完成后重启网页并重新检测。

## Kotoba/Faster-Whisper 显示模型不完整

进阶方案固定准备 Kotoba-Whisper v2.2 和 Faster-Whisper large-v2。选择同系列其它变体时，
用户必须自行把完整模型放入程序识别的本地缓存；程序不会在点击识别时静默下载大型模型。

如果当前选择正是 v2.2/large-v2：

1. 在“设备与模型”点击对应后端“安装/修复”；
2. 查看安装日志是依赖缺失还是模型校验失败；
3. 有离线包时放入 `model-packs` 并扫描导入；
4. 完成后重启页面进程。

## ASMR 专用 VAD 或 Qwen 时间戳不显示

网页只显示本机能完整运行的分析方式。只有模型目录或只有 Python 包都不算就绪。

- ASMR VAD 需要固定的 Whisper VAD ONNX 快照、`onnxruntime` 和 `transformers`；
- Qwen 时间戳需要固定的 Qwen3 ForcedAligner 0.6B 快照和 `qwen-asr`；
- 两者都包含在进阶方案，也可通过相应模型 ZIP 导入。

导入后点击“重新检测硬件与后端”。VAD 漏掉耳语时降低语音阈值、增加边界保留，或先选
“不做 VAD 预处理”。Qwen 对齐失败会保留原时间，可在 `analysis` 报告中查原因。

## ASR 速度很慢或显存不足

- 批大小改为 1；
- 关闭多模型交叉校对，只验证一个模型；
- Faster-Whisper CPU 使用 `int8`；
- GPU 显存紧张时用 `int8_float16` 或更小的同系列模型；
- 缩小 Kotoba 分块或保持 Parakeet 默认分块；
- 关闭其它 GPU 程序；
- 长文件先用几十秒片段验证设置。

多模型校对是串行的，总耗时接近各模型耗时之和。它不会让多个大型识别器同时常驻显存。

## 多模型交叉校对列表为空

参与列表只显示本地已完整下载且运行环境可用的模型。这不是网络搜索框。

1. 在“设备与模型”检查至少一个识别后端为“可用”；
2. 安装第二个模型后重新检测；
3. 回到 ASR 设置重新开启多模型校对；
4. 选择支持 LLM 的翻译服务。

DeepL、Google Cloud Translation 和 Microsoft Translator 不会出现在校对能力中。候选失败
时查看 `analysis/asr_candidates.json` 和 `analysis/asr_review.json`。

## 改了设置，当前项目仍按原设置运行

只点“仅保存为以后新项目默认值”不会修改已打开项目。请：

1. 保持目标项目处于打开状态；
2. 在设置页完成修改；
3. 点击“保存并应用到当前项目”；
4. 查看设置状态和项目状态是否都确认新配置；
5. 影响识别的修改需要再次点击“运行 ASR（语音识别）”。

如果另一个浏览器窗口也打开同一项目，先关闭旧窗口并重新打开项目，避免 revision 冲突。

## 翻译失败或输出不完整

按顺序检查：

1. 当前服务的密钥状态、余额、配额和地区可用性；
2. 模型 ID 与基础地址是否属于同一服务；
3. 自定义 Prompt 是否仍要求逐句 ID 和有效 JSON；
4. 上下文句数和最大输出 Token 是否过大；
5. 代理或服务端是否返回 HTML 错误页。

恢复默认 Prompt、降低上下文句数后重试。已经保存的成功中文不会因为后续批次失败而丢失。

## 外部 TTS API 连接失败

- GPT-SoVITS 常用端口 9880；
- CosyVoice 常用端口 50000；
- Fish 接口必须兼容 `/v1/tts` 的 references 请求；
- Docker/远程 GPT-SoVITS 必须能读取请求中的参考音频路径；
- Fish 云服务通常需要保存 API Key；
- 把并发数设为 1，排除服务端限流或显存并发问题；
- 确认响应是有效音频，而不是 JSON/HTML 错误正文。

先用服务端自己的示例请求验证接口，再从 ASMR Dubber 调用。不同上游发行版的 API 可能不
兼容，仅端口能连接不代表请求格式匹配。

## IndexTTS2 未就绪或合成失败

```powershell
.\scripts\windows\install-indextts2.ps1
```

```bash
bash scripts/linux/install-indextts2.sh
```

IndexTTS2 必须在 `.asmr-dubber/runtimes/index-tts` 的隔离环境中，并具有完整 checkpoints 和
配套 `config.yaml`。不要把它装到主 venv。

合成显存不足时关闭其它 GPU 程序、保持 FP16、减少同时运行的服务。6 GB 显存接近最低条件，
10 GB 以上更稳妥。

## 中文音色不稳定

- 使用“统一声纹，仅音色参考”；
- 选择 5–15 秒、单人、清晰且有实义语音的参考；
- 避开纯气声、笑声、背景音乐、多人重叠和强音效；
- 需要参考文字的后端填写逐字对应日文；
- 固定随机种子，不要同时大幅改变多个采样参数；
- 多角色项目再考虑逐句参考。

先试听项目统一参考片段。如果参考本身已截断或包含下一句，回到表格修正时间后重新设置参考。

## 混音削波、延迟或声道不对

- 保持最终峰值保护开启；
- 先保持中文相对响度 -4 dB 和额外增益 0 dB；
- 检查单句 TTS 是否已经失真；
- 多声道选择“自动（优先中置）”；
- 只有明确需要时才选“复制到全部声道”；
- 时间不对时检查逐句“提前开始”和全局提前百分比；
- 查看 `exports/transcript.json` 中的实际开始时间和配音时长。

立体声的自动模式会把中文送到左右声道；常见 3.0/5.1/7.1 布局优先中置。最终限幅器不应
移动时间轴。

## 视频或字幕输出失败

先确认纯音频混音能完成，再单独点“生成字幕”。视频输出受输入编码、FFmpeg 构建和本机编码器
影响：

- MP4 无法无损封装时可能输出 MKV；
- 无可用字幕烧录滤镜时会尝试可选择字幕轨；
- NVENC 不可用时会使用软件编码；
- 损坏或非常规媒体先用 FFmpeg 检查流信息。

查看状态框给出的实际输出扩展名，不要只寻找预期的 `.mp4`。

## 项目提示被另一个窗口修改

另一个网页窗口或命令行已经写入更高 revision。当前内存副本会被拒绝，以免覆盖磁盘内容。
重新打开 `project.json`，确认最新表格后再编辑。不要手工把 revision 改小。

## 仍然无法解决

准备一份最小报告：

- 操作系统和架构；
- CPU、GPU、显存和驱动；
- 安装方案；
- 后端、模型和关键设置；
- 可复现的最短步骤；
- `doctor --no-network` 输出；
- 去除隐私后的最新 Setup 日志或错误全文。

能用公开的几秒测试音频复现时，不要附原作品。安全问题请按[安全策略](../SECURITY.md)私下
报告。
