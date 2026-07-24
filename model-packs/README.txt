把 ASMR Dubber 离线模型包（.zip）放在这个目录。

ASMR-Dubber-Setup.exe 会在联网下载前自动校验并导入兼容的模型包。
压缩包损坏或被篡改时，安装器会明确停止，不会静默改为重新下载。
导入完成后可以删除这里的 ZIP；已安装模型位于 .asmr-dubber 目录。

Advanced 对应四个独立包：Parakeet、IndexTTS2 checkpoints、
Kotoba-Whisper v2.2 和 Faster-Whisper large-v2。

GitHub Release 的大模型采用 .part*.rar 分卷。请下载同一模型的全部分卷，
用 WinRAR 或 7-Zip 打开 part1.rar，解压出完整 ZIP 后再放入本目录。
Setup 和网页不直接读取 RAR 分卷。
