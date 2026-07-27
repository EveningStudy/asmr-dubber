ASMR Dubber 离线模型包目录
==========================

把 ASMR Dubber 格式的模型 ZIP 原样放在这里。不要解压、改名或修改包内 manifest。

导入方法：

1. 重新运行 ASMR-Dubber-Setup.exe，Setup 会导入所选安装方案需要的模型；或
2. 在网页“设置 → 设备与模型”点击“扫描并导入本地模型包”；或
3. 运行 scripts/windows/run-cli.ps1 import-model-packs --all。

程序会检查 pack ID、目标平台、相对路径、文件大小和 SHA-256。损坏、不完整、路径不安全或
内容被重新压缩的模型包会被拒绝，不会通过关闭校验继续安装。

Windows“进阶”方案使用 6 个 ZIP，包含 7 个模型：

1. Parakeet 模型包
   - Parakeet CTC 1.1B JA GAL
   - Parakeet TDT/CTC 0.6B JA
2. Kotoba-Whisper v2.2
3. Faster-Whisper large-v2
4. Qwen3 ForcedAligner 0.6B
5. 日语 ASMR 专用 Whisper VAD ONNX
6. IndexTTS2 checkpoints（只在 NVIDIA GPU 电脑上安装）

依赖包也可以放在这里，但依赖包不是模型包。Setup 只接受程序固定的文件名、字节数和
SHA-256。不要用自己重新压缩的同名文件替换。

导入成功后可以删除 ZIP，模型已经写入 .asmr-dubber/models 或对应隔离运行时。需要备份或
在另一台电脑继续安装时，也可以保留这些 ZIP。

如果下载提供 .part*.rar 分卷，先下载同一制品的全部分卷，用 7-Zip 或 WinRAR 打开
part1.rar，解出一个完整 ZIP，再把 ZIP 放到本目录。Setup 不直接读取 RAR。

可先列出并检查本目录中的模型包：

scripts/windows/run-cli.ps1 list-model-packs

Linux 使用：

bash scripts/linux/run-cli.sh list-model-packs

Parakeet 离线包是 Windows 专用；Linux Setup 会按固定文件和哈希准备 Linux CrispASR 与模型。
