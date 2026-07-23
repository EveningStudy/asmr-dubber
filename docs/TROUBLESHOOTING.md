# 排障指南

先运行：

```powershell
./scripts/windows/run-cli.ps1 doctor --no-network
```

```bash
bash scripts/linux/run-cli.sh doctor --no-network
```

提交 Issue 前删除用户名、私人路径和其他敏感信息。

## Windows 启动器无法运行

确认 `ASMR-Dubber.exe` 位于项目根目录，且 `scripts/windows/setup.ps1` 和 `scripts/windows/run-ui.ps1` 存在。下载的未签名 Release 可能触发 SmartScreen 来源提示；请只从可信 Release 获取启动器。高级用户仍可运行 `./scripts/windows/setup.ps1 -Profile Recommended`。

## 找不到 Python 或 Conda

项目使用 `.asmr-dubber` 内的托管 Python，不依赖全局 Python、Conda 或系统 PATH。始终通过`ASMR-Dubber.exe` 或 `scripts/linux/run-ui.sh` 启动。移动项目目录后重新运行安装器；已有模型缓存会复用。

## NVIDIA 驱动正常，但 CUDA 不可用

1. 确认 `nvidia-smi` 正常。
2. 运行 `doctor --no-network`。
3. 重新运行相应后端安装。
4. 关闭占用显存的程序。

WSL 只需要 Windows NVIDIA 驱动，不要在 WSL 内安装内核驱动。

## 下载失败

- 直接重跑安装器；uv、Hugging Face、Parakeet 和 IndexTTS2 均会复用已完成文件。
- 使用可信的 PyPI `/simple` 索引或 Hugging Face endpoint。
- 代理使用标准 `HTTP_PROXY` / `HTTPS_PROXY`。
- WSL 网络正常时可设置 `ASMR_DUBBER_WINDOWS_BRIDGE=0` 禁用临时网络桥。
- 镜像文件不完整时切回官方源。

安装器会固定源码或模型 revision，并对关键文件执行完整性检查。

## Parakeet 未安装或模型不可切换

`Recommended` 会安装 1.1B CTC GAL 和 0.6B TDT/CTC。也可单独执行：

```powershell
./scripts/windows/install-parakeet.ps1 -Variant Auto
```

```bash
bash scripts/linux/install-parakeet.sh
```

完成后重启 UI。默认选择 1.1B；0.6B 在同一模型下拉框中。



## 多 ASR 校对失败

1. 分别用短音频验证每个 ASR。
2. 选择 DeepSeek、OpenAI、Claude、Gemini 或 OpenAI-compatible 翻译供应商。
3. 恢复默认校对 Prompt。
4. 查看 `analysis/asr_candidates.json` 和 `analysis/asr_review.json`。

模型串行执行，但单个模型仍必须满足显存要求。

## DeepSeek 输出达到长度上限

当前实现会分批、自动二分并保存成功结果。仍失败时：

1. 恢复默认翻译 Prompt；
2. 确认模型支持 JSON 输出；
3. 不要求模型输出思考过程；
4. 检查余额、配额和模型权限；
5. 重新加载项目并点击“补译空白中文”。

## 纯笑声或语气词仍被配音

过滤分两层执行：本地纯语气词检测和 LLM 空翻译。确认使用当前默认翻译 Prompt。混有实义内容、答复、否定或明确惊讶的句子会保留。最终结果可在表格中手动启用或停用。

## 中文音色不稳定

- 使用“统一声纹，仅音色参考”；
- 选择 4–15 秒、清晰、单一说话人、非纯气声的参考；
- 保持固定随机种子和确定性采样；
- 单角色项目不要使用逐句参考；
- 必要时使用外部参考音频。




## IndexTTS2 未就绪

重新运行：

```powershell
./scripts/windows/install-indextts2.ps1
```

```bash
bash scripts/linux/install-indextts2.sh
```

不要把 IndexTTS2 安装进主 `.asmr-dubber/venv`。它使用
`.asmr-dubber/runtimes/index-tts/.venv`。

## 端口 7860 被占用
用别的端口

```powershell
./scripts/windows/run-cli.ps1 ui --port 7861
```

```bash
bash scripts/linux/run-cli.sh ui --port 7861
```


## 卸载

停止程序并删除仓库目录。
