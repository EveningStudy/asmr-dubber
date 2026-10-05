中文 | [English](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.1/docs/en/RELEASE.md)

# ASMR Dubber 2.0.1

## 修复

- 修复 2.0.0 解压后无法启动的问题（启动时提示“内置基础应用 wheelhouse 不完整”，随后安装失败）。
- 修复移动程序文件夹后无法启动的问题。
- IndexTTS-2.5 的加速选项（DeepSpeed、CUDA Kernel 等）在环境不支持时会自动跳过，不再让每一句都报错。

## 已经下载了 2.0.0

重新下载 2.0.1 即可。不想重新下载的话，删除程序目录下的 `.asmr-dubber\venv` 文件夹，再运行 `ASMR-Dubber.exe`，程序会联网重建运行环境。

## 下载与升级

Windows 下载 `ASMR-Dubber-windows-portable-v2.0.1.zip`，完整解压到一个路径短的文件夹，运行 `ASMR-Dubber.exe`。

从 1.x 升级：先关掉程序，删除旧文件夹里的 `src` 文件夹和 `ASMR-Dubber-Setup.exe`，再把新版本的文件全部覆盖进去。`.asmr-dubber` 文件夹要保留，里面是你的项目、模型和设置。升级前建议先备份它。

2.0.0 的全部变化见 [v2.0.0](https://github.com/EveningStudy/asmr-dubber/releases/tag/v2.0.0)。

[使用手册](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.1/docs/USER_GUIDE.md) · [安装与模型](https://github.com/EveningStudy/asmr-dubber/blob/v2.0.1/docs/INSTALLATION.md)
