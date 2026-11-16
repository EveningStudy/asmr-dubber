$LocalPackIds = switch ($Profile) {
    "基础" { @() }
    "推荐" { @("parakeet-ja-windows", "indextts2-checkpoints") }
    "进阶" {
        @(
            "parakeet-ja-windows",
            "indextts2-checkpoints",
            "kotoba-whisper-v2.2",
            "faster-whisper-large-v2",
            "anime-whisper",
            "qwen3-forced-aligner",
            "whisper-vad-asmr-onnx"
        )
    }
}
if ($Profile -ne "基础") {
    Write-SetupHost "正在检测并导入当前档位的本地模型包..." -ForegroundColor Cyan
    $ImportArguments = @("-m", "asmr_dubber.cli", "import-model-packs", "--all")
    foreach ($PackId in @($LocalPackIds)) {
        $ImportArguments += @("--pack-id", $PackId)
    }
    Invoke-Checked -FilePath $Python `
        -ArgumentList $ImportArguments `
        -FailureMessage "本地模型包扫描或导入失败；请检查 model-packs 目录中的压缩包"
}

if ($InstallAdvancedModels) {
    Write-SetupHost (
        "正在准备进阶分析模型：Kotoba-Whisper v2.2、Faster-Whisper large-v2、" +
        "Qwen3 ForcedAligner 与日语 ASMR 专用 VAD..."
    ) -ForegroundColor Cyan
    Invoke-Checked -FilePath $Python `
        -ArgumentList @(
            "-m", "asmr_dubber.cli", "download-models", "--backend", "进阶语音识别"
        ) `
        -FailureMessage "进阶识别、VAD 与时间戳模型下载或校验失败"
}

if ($InstallAdvancedModels) {
    # Recommended recognizer. It has no ModelScope copy yet, so a failure here
    # must not abort Setup: Parakeet (the fallback) is already installed or
    # being installed, and the Models page can retry the download.
    Write-SetupHost "正在准备 anime-whisper 日语识别模型（约 3 GB）..." -ForegroundColor Cyan
    try {
        Invoke-Checked -FilePath $Python `
            -ArgumentList @("-m", "asmr_dubber.cli", "download-backend-models", "anime_whisper") `
            -FailureMessage "anime-whisper 下载失败"
    } catch {
        Write-Warning ("anime-whisper 未能下载，识别将先使用 Parakeet。" +
            "需要海外源时，请在设置中把“下载来源”改为“海外源”后，到“设备与模型”页重试。")
    }
}

if ($InstallParakeet) {
    Write-SetupHost "正在安装推荐 ASR（语音识别）：Parakeet 日语..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "install-parakeet.ps1") -Variant Auto
}

if ($InstallRecommendedTTS) {
    Write-SetupHost "正在安装推荐 TTS（语音合成）：IndexTTS2（约需 20 GB）..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "install-indextts2.ps1") `
        -IndexUrl $PreferredIndex -HuggingFaceEndpoint $PreferredHuggingFace
}

Write-SetupHost "正在执行环境检查..." -ForegroundColor Cyan
try {
    Invoke-Checked -FilePath $Python `
        -ArgumentList @("-m", "asmr_dubber.cli", "doctor", "--no-network") `
        -FailureMessage "环境检查未完全通过"
} catch {
    Write-Warning "核心程序已经安装，但当前选择的本地模型尚未全部可用。请在设置 → 设备与模型中查看。"
}

Write-SetupHost ""
Write-SetupHost "安装完成。运行项目根目录的 ASMR-Dubber.exe 启动界面。" -ForegroundColor Green
