中文 | [English](https://github.com/EveningStudy/asmr-dubber/blob/v2.2.1/docs/en/RELEASE.md)

# ASMR Dubber 2.2.1

## 修复与改进

- 更新 Edge TTS 到 7.3.1，支持 Microsoft 新版鉴权，修复旧客户端合成和试听返回 HTTP 403。
- Windows 启动器和启动脚本检查客户端版本，覆盖升级后自动修复旧基础环境；便携包内置兼容依赖，优先离线更新。Linux 启动脚本也会检测并修复旧版本。
- 配音与音色试听共用自动重试、超时与取消处理。配音遵循“单句超时”，试听最多等待 30 秒；服务不返回音频时可停止等待。
- 区分鉴权拒绝、限流、DNS、超时和未收到音频，失败只清理临时文件，保留已成功的配音缓存。

## 下载与升级

Windows 下载 `ASMR-Dubber-windows-portable-v2.2.1.zip`，完整解压到一个路径短的文件夹，运行 `ASMR-Dubber.exe`。

从 2.x 升级：先关闭程序并备份，将新版本文件覆盖到原程序目录。保留 `.asmr-dubber` 文件夹，其中包含项目、模型和设置。首次启动会检查并更新基础客户端，无需重新下载语音模型。

从 1.x 升级：先关掉程序，删除旧文件夹里的 `src` 文件夹和 `ASMR-Dubber-Setup.exe`，再覆盖新版本文件。保留并备份 `.asmr-dubber` 文件夹。

Edge TTS 依赖 Microsoft 在线服务。服务波动、音色临时异常或限流仍可能发生，可以只重试失败句或换音色。
