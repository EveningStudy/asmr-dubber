function Write-SetupHost {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)][object]$Object = '',
        [ConsoleColor]$ForegroundColor = [ConsoleColor]::Gray,
        [switch]$NoNewline
    )
    $Message = [string]$Object
    if ($env:ASMR_DUBBER_UI_LANGUAGE -eq 'en') {
        $Messages = [ordered]@{
            '正在检查 Windows 原生运行库（仅报告，不会中止安装）...' = 'Checking Windows native runtimes (report only; installation continues)...'
            'Microsoft Visual C++ x64 运行库检查通过。' = 'Microsoft Visual C++ x64 runtime check passed.'
            '现有 CrispASR 原生运行时可以启动。' = 'The installed CrispASR runtime starts successfully.'
            'ASMR Dubber · Windows 安装' = 'ASMR Dubber · Windows setup'
            '项目目录：' = 'Application directory: '
            '数据目录：' = 'Data directory: '
            '安装配置：' = 'Installation profile: '
            '下载策略：ModelScope 优先；已显式允许海外备用源。' = 'Downloads: ModelScope first; external fallback explicitly enabled.'
            '下载策略：ModelScope 优先；GitHub/Hugging Face/官方海外源已关闭。' = 'Downloads: ModelScope first; external sources disabled.'
            '只读本地缓存：' = 'Read-only local cache: '
            '预计安装后占用：' = 'Estimated installed size: '
            '建议安装前可用空间：' = 'Suggested free space: '
            '进阶档位会安装以下固定模型：' = 'Advanced installs these pinned models:'
            'ASR（语音识别）' = 'ASR '
            'VAD（语音活动检测）日语 ASMR 专用' = 'Japanese ASMR VAD '
            '时间戳对齐：' = 'Timestamp alignment: '
            '（阿里 Qwen）' = '(Alibaba Qwen)'
            'TTS（语音合成）' = 'TTS '
            '（仅 NVIDIA GPU）' = '(NVIDIA GPU only)'
            '不会自动安装 Kotoba v2.0/v2.1、Faster-Whisper large-v3 或其它识别模型。' = 'Does not install Kotoba v2.0/v2.1, Faster-Whisper large-v3 or other ASR models.'
            '项目未包含可用的 uv，正在下载修复副本...' = 'No working bundled uv; downloading a repair copy...'
            'uv 已就绪。' = 'uv is ready.'
            '正在准备 Python 3.12...' = 'Preparing Python 3.12...'
            '检测到 NVIDIA GPU，正在安装官方 CUDA 13.0 PyTorch...' = 'NVIDIA GPU detected; installing CUDA 13.0 PyTorch...'
            '使用 ModelScope CUDA wheelhouse。' = 'Using the ModelScope CUDA wheelhouse.'
            '正在安装应用依赖：' = 'Installing application dependencies: '
            '使用便携包内置的基础应用 wheelhouse。' = 'Using the bundled core wheelhouse.'
            '使用 ModelScope 应用依赖 wheelhouse。' = 'Using the ModelScope application wheelhouse.'
            '应用依赖已由 Windows 依赖包提供。' = 'Application dependencies supplied by the Windows dependency package.'
            '使用便携包内置的在线/API 客户端 wheelhouse。' = 'Using the bundled online/API client wheelhouse.'
            '正在检测并导入当前档位的本地模型包...' = 'Checking and importing local model packages for this profile...'
            '正在安装推荐 ASR（语音识别）：Parakeet 日语...' = 'Installing recommended ASR: Japanese Parakeet...'
            '正在安装推荐 TTS（语音合成）：IndexTTS2（约需 20 GB）...' = 'Installing recommended TTS: IndexTTS2 (about 20 GB)...'
            '正在执行环境检查...' = 'Checking the environment...'
            '安装完成。运行项目根目录的 ASMR-Dubber.exe 启动界面。' = 'Installation complete. Run ASMR-Dubber.exe from the application directory.'
        }
        # Longest first: complete messages before embedded technical labels.
        foreach ($Entry in ($Messages.GetEnumerator() | Sort-Object { $_.Key.Length } -Descending)) {
            $Message = $Message.Replace([string]$Entry.Key, [string]$Entry.Value)
        }
    }
    Microsoft.PowerShell.Utility\Write-Host $Message -ForegroundColor $ForegroundColor -NoNewline:$NoNewline
}
