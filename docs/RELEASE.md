# 发行包说明

ASMR Dubber 0.5.0 面向 64 位 Windows 10/11 和 x86_64 Linux。它处理日语音频或视频，提供
可校对的日语识别、中文翻译、音色克隆、混音和字幕工作流。

## 下载包包含什么

- ASMR Dubber 源码和命令行工具；
- Windows 安装器与启动器；
- Windows/Linux 安装、启动和模型管理脚本；
- ModelScope 镜像与制品校验清单；
- 完整用户、配置、后端、排障和维护文档。

下载包不包含大型模型、用户项目、API Key 或已经建立的运行环境。首次安装会按用户选择准备
便携 Python、依赖和固定模型，所有内容默认放在程序目录的 `.asmr-dubber`。

## Windows 免安装包

[ModelScope 推荐版免安装包仓库](https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Windows-Recommended-Portable-v0.5.0/files)
另行提供一个可以直接解压使用的完整包：

| 文件 | 内容 |
|---|---|
| `ASMR-Dubber-Windows-Recommended-Portable-v0.5.0.zip` | 程序和完整运行环境 + 两个 Parakeet 日语模型 + IndexTTS2 环境和 checkpoints |

完整解压后直接运行 `ASMR-Dubber.exe`，不需要先运行 Setup。IndexTTS2 只支持 NVIDIA GPU；
第一次生成语音时可能需要较长时间。其它模型可在网页中按需安装。

## 安装方案

| 方案 | 内容 | 建议可用空间 |
|---|---|---:|
| 基础 | 程序、网页、音频工具和外部 API 客户端，不含本地识别模型 | 5 GB |
| 推荐 | 基础 + Parakeet CTC 1.1B JA GAL + Parakeet TDT/CTC 0.6B JA；NVIDIA 电脑再装 IndexTTS2 | 35 GB |
| 进阶 | 推荐模型 + Kotoba-Whisper v2.2 + Faster-Whisper large-v2 + ASMR Whisper VAD ONNX + Qwen3 ForcedAligner 0.6B；NVIDIA 电脑再装 IndexTTS2 | 50 GB |

Windows 用户完整解压后运行 `ASMR-Dubber-Setup.exe`。不要求预装 Python、Git、FFmpeg、
CUDA Toolkit 或 PowerShell 7；系统自带 Windows PowerShell 5.1 可以运行安装脚本。

Linux 用户运行：

```bash
bash scripts/linux/setup.sh 推荐
bash scripts/linux/run-ui.sh
```

安装要求、硬件选择、下载策略和离线模型包见[安装指南](INSTALLATION.md)。

## 下载和完整性

安装器默认优先使用 ModelScope。固定制品在导入前检查字节数和 SHA-256，模型包还检查内部
manifest。GitHub、Hugging Face 和海外官方软件源默认不作为自动回退。

下载中断后重新运行 Setup 即可继续。已经完整且哈希正确的大文件会直接复用，不需要删除缓存
或重新下载。

## 数据和密钥

项目、模型、缓存、设置和密钥默认保存在程序目录内。API Key 以明文位于
`.asmr-dubber/config/secrets.json`，请用操作系统账户权限保护该目录，不要把它加入压缩包、
Git 仓库或公开网盘。

停止程序后删除整个程序文件夹即可卸载。删除前请先备份需要保留的项目和密钥。

## 许可证

项目代码采用 MIT License。模型、运行时和外部服务适用各自条款，IndexTTS2 使用独立的
bilibili Model Use License。请阅读[第三方软件、模型与服务](THIRD_PARTY_NOTICES.md)，并确认
你有权处理输入作品和参考声音。
