# ModelScope 制品维护

本文面向发布维护者。普通用户不需要上传任何文件，只需运行 Setup 或使用离线模型包。

ASMR Dubber 默认从 ModelScope 获取引导程序、Python、依赖包、模型包和部分第三方运行时。GitHub、Hugging Face、hf-mirror 与海外官方软件源只有在用户显式开启时才进入候选列表。

## 仓库布局

| ModelScope 仓库 | 内容 |
|---|---|
| `EveningStudyW/ASMR-Dubber-Portable-Mirror` | uv、managed CPython、CrispASR、FFmpeg、Parakeet 原始文件和依赖 wheelhouse |
| `EveningStudyW/ASMR-Dubber-Parakeet` | Windows Parakeet 离线模型包 |
| `EveningStudyW/ASMR-Dubber-IndexTTS2` | IndexTTS2 checkpoints 模型包和固定源码 ZIP |
| `EveningStudyW/ASMR-Dubber-Windows-Recommended` | Windows 推荐方案依赖包 |
| `EveningStudyW/ASMR-Dubber-Windows-Advanced` | Windows 进阶依赖包、Kotoba、Faster-Whisper、Qwen 对齐和 ASMR VAD 模型包 |
| `EveningStudyW/ASMR-Dubber-Windows-Portable` | 已装好依赖和模型、解压即可运行的 Windows 核心/推荐/进阶完整包 |

默认 revision 是 `master`。由于 URL 使用可变分支名，每一个已经发布并被代码固定哈希的路径都必须视为不可变制品：**不能用不同内容覆盖同名文件**。需要重建时使用新文件名或新目录，更新所有合同并发布新的程序版本。

API Token 只用于上传或访问私有仓库。不要把 Token 写进仓库、`mirrors.json`、日志、Issue、测试夹具或本文。

## 真相源

发布前同时检查以下文件：

1. `modelscope-artifacts.lock.json`：Portable Mirror 中固定大小和 SHA-256 的引导制品；
2. `mirrors.json`：仓库、路径、下载优先级和海外源开关；
3. `src/asmr_dubber/model_pack_download.py`：六个大型模型包的文件名、字节数和 SHA-256；
4. `scripts/windows/recommended-dependencies.ps1`：Windows 推荐/进阶依赖包合同；
5. IndexTTS2 安装脚本：固定源码 revision 和源码 ZIP SHA-256。

不能只改其中一个。CI 会检查 lock 与镜像配置的一致性，但只有端到端干净安装能证明远端文件、依赖包内部结构和安装脚本仍然匹配。

## Portable Mirror 固定制品

仓库：[EveningStudyW/ASMR-Dubber-Portable-Mirror](https://www.modelscope.cn/models/EveningStudyW/ASMR-Dubber-Portable-Mirror)

下表由 `modelscope-artifacts.lock.json` 约束。路径区分大小写，`+` 是文件名的一部分；网页可能把它显示成 `%2B`，不要因此把实际文件改名。

| 仓库内路径 | 字节数 | SHA-256 |
|---|---:|---|
| `uv-x86_64-pc-windows-msvc.zip` | 25,710,044 | `be8d78c992312212e5cc05e9f9de3fa996db73b7c86a186dfb9231eb9f91d33e` |
| `artifacts/bootstrap/uv/0.11.30/uv-x86_64-unknown-linux-gnu.tar.gz` | 26,274,137 | `04bc7d180d6138bf6dc08387acf507a823f397a98fea55da36b0ccc7fbce3b68` |
| `cpython-3.12.13+20260718-x86_64-pc-windows-msvc-install_only_stripped.tar.gz` | 21,932,298 | `0d422a1439ec308e03f47df551bc30f5994727c456e414b026d202bcda9b7c1c` |
| `artifacts/python-build-standalone/releases/download/20260718/cpython-3.12.13+20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz` | 34,199,823 | `5854aa6ec71cad00334d5065633c210b2e7feb40956767a59a91791cadcf0b79` |
| `artifacts/python-build-standalone/releases/download/20251007/cpython-3.11.13+20251007-x86_64-pc-windows-msvc-install_only_stripped.tar.gz` | 25,990,147 | `cde5153f59a67d9e108f2ed964526e9aed100eba180f54bee0496b4fd73a8b29` |
| `artifacts/python-build-standalone/releases/download/20251007/cpython-3.11.13+20251007-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz` | 30,157,215 | `43bfc42529843ecd1d9c08c4a239ede348f96ff0acaef2ec24b28dc059f4f0c3` |
| `artifacts/runtimes/crispasr/v0.8.21/crispasr-windows-x86_64-cpu.zip` | 6,077,810 | `c16ae6a69bad1c077c9bc01821fbbd6d3671a6ad114239eb0807cf3601e3b6f2` |
| `artifacts/runtimes/crispasr/v0.8.21/crispasr-windows-x86_64-cuda.zip` | 691,366,170 | `d7db946f4b73fa0fbf3a3e27d63a994eee51a90138384813d35f7863a59aeba3` |
| `artifacts/runtimes/crispasr/v0.8.21/crispasr-linux-x86_64.tar.gz` | 23,042,489 | `55d48357052c6d9376ad6548c877f9fb1e0a79728ae5e24dfc84b29405d22434` |
| `artifacts/runtimes/crispasr/v0.8.21/crispasr-linux-x86_64-cuda13.tar.gz` | 212,863,648 | `883edc02ed3666af9e76b26ca29e2fc5db0ce48f97e4b0ef575d482a3619c74d` |
| `artifacts/runtimes/ffmpeg/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip` | 70,830,663 | `96d669b9e33133fba1365c74a8b1d79b26b7245e88fe7d6d3ec198dfab649b4a` |
| `artifacts/runtimes/ffmpeg/btbn-checksums.sha256` | 5,296 | `1305cfe1375c3a54e3bff0383db0a305373aa89771ca1cb74db22f4712d68a9e` |
| `artifacts/models/parakeet/parakeet-ctc-1.1b-ja-f16.gguf` | 2,134,533,952 | `34dd3128275c9bca2b4296f53c5f831feb258fcf3fdd28c29c0dc2d2f7d5ede7` |
| `artifacts/models/parakeet/parakeet-tdt-0.6b-ja.gguf` | 1,246,932,800 | `374eb0132eebaec4df77a9631cbbeb03790be48a4a517f6cc8e8bdb38fe9a584` |
| `artifacts/models/crispasr/fireredpunc-q4_k.gguf` | 57,886,944 | `faf4a43e3135bc307a66194685af00f756e6f4c28c7d9e2dd8f3517cddca5c45` |
| `artifacts/models/crispasr/ggml-silero-v6.2.0.bin` | 885,098 | `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987` |
| `artifacts/runtimes/openblas/0.3.26/libopenblas0-pthread_0.3.26+ds-1ubuntu0.1_amd64.deb` | 7,183,128 | `7dc3b4384c02aecb87eb8b70fa26c5843a08af242f4638aa4b36922bdc4f5b04` |

Linux Parakeet 还需要上述 OpenBLAS、标点和 VAD 文件。它们已经上传到同一个 Portable Mirror 仓库；安装器会在下载后按大小和 SHA-256 校验，不会静默使用系统目录或临时下载到用户目录。

Windows 全新安装首先依赖根目录的 uv 和 Python 3.12 文件。把它们放进子目录或把 `+` 改成空格都会导致 Setup 找不到。

## Windows 依赖包

### 推荐方案

仓库内文件：

```text
EveningStudyW/ASMR-Dubber-Windows-Recommended/
└── ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip
```

- 字节数：`4,060,845,976`
- SHA-256：`a026ea897a36fa7cf22b2c1b5f8069d9b353c02a1e5285e00d0ea984f9a1472b`

它包含 Windows 推荐方案的主应用环境、IndexTTS2 Python/CUDA 环境和共享 FFmpeg 运行文件。导入器支持第三方包的深层路径；压缩包内部布局由 `scripts/import_windows_dependency_pack.py` 校验。

### 进阶方案

仓库内文件：

```text
EveningStudyW/ASMR-Dubber-Windows-Advanced/
└── ASMR-Dubber-Windows-Advanced-Dependencies-v1.0.0.zip
```

- 字节数：`2,905,762,138`
- SHA-256：`bafd2268de9a83bbf391ba8918d1798d24f703b023af70e8f623b2dbffc9a178`

它提供进阶识别、VAD 和对齐所需的主环境依赖，内部布局由 `scripts/import_windows_advanced_dependency_pack.py` 校验。

推荐/进阶依赖 ZIP 也可以原样放入用户的 `model-packs` 目录。Setup 只按固定名称、大小和哈希识别，不会接受“内容看起来相同”的重新压缩文件。

## 大型模型包

下表是 `model_pack_download.py` 的远程合同：

| 仓库 | 文件 | 字节数 | SHA-256 |
|---|---|---:|---|
| `ASMR-Dubber-Parakeet` | `ASMR-Dubber-ModelPack-parakeet-ja-windows-v0.2.1.zip` | 4,070,471,378 | `3a9e95e02df01a40533d5f73893d62fe2bf0bb897b98d2b8e494faa2ed139790` |
| `ASMR-Dubber-IndexTTS2` | `ASMR-Dubber-ModelPack-indextts2-checkpoints-v0.2.1.zip` | 11,189,524,132 | `144aa91c4de24faf8d415df4fa4324b831609c4bbcef4406a5db4f2a952e108e` |
| `ASMR-Dubber-Windows-Advanced` | `ASMR-Dubber-ModelPack-kotoba-whisper-v2.2-v1.0.0.zip` | 3,027,748,160 | `a5da2f63fd2c4972dad4cc53db89e0d0250af9d4431905b8c558d55169734c46` |
| `ASMR-Dubber-Windows-Advanced` | `ASMR-Dubber-ModelPack-faster-whisper-large-v2-v1.0.0.zip` | 3,087,767,076 | `4a4a213561d327e82d5dc5a8e8c071313bd948ad90f7b4c51e650044fd3bc949` |
| `ASMR-Dubber-Windows-Advanced` | `ASMR-Dubber-ModelPack-qwen3-forced-aligner-v1.0.0.zip` | 1,837,358,823 | `6697b80bfba3a182a86290ba0f7b8adc958d7112bfe6cc9caa73bc7207b74242` |
| `ASMR-Dubber-Windows-Advanced` | `ASMR-Dubber-ModelPack-whisper-vad-asmr-onnx-v1.0.0.zip` | 54,692,316 | `f7d4c6ec7c9576d325685ffeaf7a39e5160fa1d3e6fe94ae60ed7dc866e5eaa9` |

文件级哈希只是第一层。导入器还检查 ZIP 内的 `manifest.json`、平台、pack ID、相对路径、解压后字节数和每个文件 SHA-256。不能用普通模型目录随手压缩后覆盖这些文件。

需要制作模型包时先查看脚本参数：

```powershell
.\.asmr-dubber\venv\Scripts\python.exe .\scripts\create-model-packs.py --help
```

模型来源 revision 和许可证文件必须随 manifest 一起固定。

## IndexTTS2 源码

IndexTTS2 源码固定到 revision：

```text
13495845e3028f0bb6ca1462ad22aa0e76349e40
```

ModelScope 文件名：

```text
index-tts-13495845e3028f0bb6ca1462ad22aa0e76349e40.zip
```

固定 SHA-256：

```text
7ed8bc742e2eeeb83f922247ef0e27f96327f418acacb6c63f182cafd66887ba
```

源码 ZIP 与 checkpoints 模型包是两件制品，缺一不可。源码中的 `LICENSE`、`LICENSE_ZH.txt` 和其它 notices 必须保留。

## Wheelhouse

如果要发布完全离线的依赖包，可使用下列路径。当前发行配置没有把这些可选 wheelhouse 声明为下载源；推荐/进阶档直接使用已验证的依赖包，基础档使用配置中的国内 PyPI 镜像补齐小型依赖。每个发布的归档必须有同路径、同文件名再加 `.sha256` 的旁车文件：

| 仓库内路径 | 内容 |
|---|---|
| `artifacts/dependencies/windows/ASMR-Dubber-Windows-Wheelhouse-v0.4.0.zip` | Windows 主程序和 UI/ASR extras |
| `artifacts/dependencies/windows/ASMR-Dubber-Windows-CUDA130-Wheelhouse-v0.4.0.zip` | `torch==2.11.0+cu130`、`torchaudio==2.11.0+cu130` 及解析依赖 |
| `artifacts/dependencies/windows/ASMR-Dubber-IndexTTS2-Wheelhouse-v0.4.0.zip` | 固定 IndexTTS2 `uv.lock` 的 Windows wheels |
| `artifacts/dependencies/linux/ASMR-Dubber-Linux-Wheelhouse-v0.4.0.tar.gz` | Linux 主程序和构建依赖 |
| `artifacts/dependencies/linux/ASMR-Dubber-IndexTTS2-Wheelhouse-v0.4.0.tar.gz` | 固定 IndexTTS2 `uv.lock` 的 Linux wheels |

旁车文件只包含一行：

```text
<64 位小写 SHA-256>  <归档文件名>
```

安装器先取旁车再决定是否下载大包：

- 旁车不存在：认为该 wheelhouse 没有发布，可以使用配置中的国内 PyPI 镜像；
- 旁车存在但归档缺失、大小不完整或哈希错误：立即停止，不静默切换；
- 归档与旁车一致：离线安装，并拒绝从网络补缺 wheel。

因此不要先上传 `.sha256` 再慢慢上传大包。应先让大包完整可见，最后发布旁车文件。

## 本地校验

准备一个与仓库路径完全相同的上传根目录，例如：

```text
E:\modelscope-upload\
├── uv-x86_64-pc-windows-msvc.zip
├── cpython-3.12.13+20260718-x86_64-pc-windows-msvc-install_only_stripped.tar.gz
└── artifacts\...
```

校验 Portable Mirror 全部固定制品：

```powershell
.\.asmr-dubber\venv\Scripts\python.exe `
    .\scripts\verify_modelscope_artifacts.py `
    --local-root E:\modelscope-upload `
    --require-all
```

只检查代码合同，不读取本地大文件：

```powershell
.\.asmr-dubber\venv\Scripts\python.exe .\scripts\verify_modelscope_artifacts.py
```

单个文件可用 PowerShell 核对：

```powershell
$file = Get-Item -LiteralPath 'E:\modelscope-upload\要上传的文件.zip'
$file.Length
(Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
```

哈希计算期间磁盘会持续读取，但不会产生网络流量。

## 发布流程

1. 固定上游 revision、依赖锁和许可证；
2. 用项目提供的打包脚本生成制品；
3. 在最终文件上计算字节数和 SHA-256；
4. 为变更后的制品选择新的不可变路径；
5. 同步修改 lock、镜像配置、下载合同、安装脚本和测试；
6. 运行本地制品校验、Ruff、Pyright 和完整测试；
7. 上传大文件，等待 ModelScope 显示完整大小；
8. 最后上传 wheelhouse 的 `.sha256` 旁车；
9. 用远端 metadata 或小范围请求核对 URL、Content-Length 和 Range 支持；
10. 从不含 `.asmr-dubber` 的发行包执行一次全新安装。

最后一步必须覆盖至少：Windows PowerShell 5.1、无预装 Python、全空缓存、推荐方案和进阶方案。不能用开发机已经存在的模型或依赖缓存代替这个测试。

## 本地只读缓存测试

测试安装器复用另一份有效制品，而不复制整个运行环境：

```powershell
$env:ASMR_DUBBER_LOCAL_CACHE_ROOTS = 'E:\ASMR-Dubber-Artifact-Cache'
.\ASMR-Dubber-Setup.exe
```

也可以传脚本参数：

```powershell
.\scripts\windows\setup.ps1 `
    -Profile 推荐 `
    -LocalCacheRoot 'E:\ASMR-Dubber-Artifact-Cache'
```

缓存根目录只读使用。测试结束后检查源目录时间戳和内容未变化，并确认安装日志明确写出“复用”而不是网络下载。
