Set-StrictMode -Version 2.0

$script:RecommendedDependencyPackName = `
    "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip"
$script:RecommendedDependencyPackSha256 = `
    "a026ea897a36fa7cf22b2c1b5f8069d9b353c02a1e5285e00d0ea984f9a1472b"
$script:RecommendedDependencyPackSize = 4060845976
$script:AdvancedDependencyPackName = `
    "ASMR-Dubber-Windows-Advanced-Dependencies-v1.0.0.zip"
$script:AdvancedDependencyPackSha256 = `
    "bafd2268de9a83bbf391ba8918d1798d24f703b023af70e8f623b2dbffc9a178"
$script:AdvancedDependencyPackSize = 2905762138

function Test-ASMRDubberCoreRuntime {
    param([Parameter(Mandatory = $true)][string]$PortableRoot)

    $Python = Join-Path $PortableRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) { return $false }
    $Code = @"
import sys
try:
    import asmr_dubber.ui, av, gradio, soundfile, setuptools
except Exception:
    sys.exit(1)
"@
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @("-c", $Code) -WorkingDirectory $PortableRoot
    return $ExitCode -eq 0
}

function Test-ASMRDubberIndexRuntimeDependencies {
    param([Parameter(Mandatory = $true)][string]$PortableRoot)

    $BasePython = Join-Path $PortableRoot `
        "runtimes\python\cpython-3.11.13-windows-x86_64-none\python.exe"
    $Python = Join-Path $PortableRoot "runtimes\index-tts\.venv\Scripts\python.exe"
    if (-not (Test-Path $BasePython) -or -not (Test-Path $Python)) { return $false }
    $Code = @"
import sys
try:
    import modelscope, torch, torchaudio, transformers
except Exception:
    sys.exit(1)
"@
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @("-c", $Code) -WorkingDirectory $PortableRoot
    return $ExitCode -eq 0
}

function Test-ASMRDubberRecommendedDependencies {
    param([Parameter(Mandatory = $true)][string]$PortableRoot)

    if (-not (Test-ASMRDubberCoreRuntime -PortableRoot $PortableRoot)) { return $false }
    if (-not (Test-ASMRDubberIndexRuntimeDependencies -PortableRoot $PortableRoot)) {
        return $false
    }
    return $null -ne (Get-ASMRDubberSharedFFmpegBin -DataRoot $PortableRoot)
}

function Import-ASMRDubberRecommendedDependencies {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PortableRoot,
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][object]$MirrorConfiguration
    )

    if (Test-ASMRDubberRecommendedDependencies -PortableRoot $PortableRoot) {
        Write-Host "Windows 推荐档运行依赖已经完整，无需下载依赖包。" `
            -ForegroundColor Green
        return $true
    }
    if ($script:RecommendedDependencyPackSize -le 0 -or `
        $script:RecommendedDependencyPackSha256 -notmatch "^[0-9a-f]{64}$") {
        Write-Warning "Windows 推荐档依赖包元数据尚未发布，继续在线安装。"
        return $false
    }

    $Inbox = Join-Path $Root "model-packs"
    $LocalArchive = Join-Path $Inbox $script:RecommendedDependencyPackName
    $DownloadRoot = Join-Path $PortableRoot "cache\downloads"
    $DownloadedArchive = Join-Path $DownloadRoot $script:RecommendedDependencyPackName
    $Archive = $null
    New-Item -ItemType Directory -Force -Path $Inbox, $DownloadRoot | Out-Null

    if (Test-Path $LocalArchive) {
        $Archive = $LocalArchive
        Write-Host "使用项目 model-packs 中的 Windows 推荐档依赖包。" `
            -ForegroundColor Cyan
    } else {
        foreach ($Url in Get-ASMRDubberMirrorList `
            -Configuration $MirrorConfiguration `
            -Name "windows_recommended_dependency_archives") {
            try {
                Write-Host "正在下载 Windows 推荐档依赖包；支持中断后续传..." `
                    -ForegroundColor Cyan
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Url -Destination $DownloadedArchive `
                    -Sha256 $script:RecommendedDependencyPackSha256 -Resume | Out-Null
                $Archive = $DownloadedArchive
                break
            } catch {
                Write-Warning "当前 Windows 推荐档依赖包来源失败：$($_.Exception.Message)"
            }
        }
    }
    if (-not $Archive) {
        Write-Warning (
            "Windows 推荐档依赖包不可用；将仅尝试配置中允许的软件源。" +
            "不会自动切换到 GitHub、Hugging Face 或官方 PyPI。"
        )
        return $false
    }

    try {
        $File = Get-Item $Archive
        if ($File.Length -ne $script:RecommendedDependencyPackSize) {
            throw "依赖包大小不符：$($File.Length)"
        }
        $ActualHash = Get-ASMRDubberFileSha256 -Path $Archive
        if ($ActualHash -ne $script:RecommendedDependencyPackSha256) {
            throw "依赖包 SHA-256 校验失败：$ActualHash"
        }
        Write-Host "正在导入 Windows 推荐档依赖包..." -ForegroundColor Cyan
        $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
            -ArgumentList @(
                (Join-Path $Root "scripts\import_windows_dependency_pack.py"),
                $Archive,
                $PortableRoot,
                "--sha256",
                $script:RecommendedDependencyPackSha256
            ) -WorkingDirectory $Root
        if ($ExitCode -ne 0) {
            throw "依赖包导入进程退出码 $ExitCode"
        }
        Repair-ASMRDubberPortablePythonPaths `
            -Root $Root -PortableRoot $PortableRoot `
            -RuntimeRoot (Join-Path $PortableRoot "runtimes") `
            -Venv (Join-Path $PortableRoot "venv")
        if (-not (Test-ASMRDubberRecommendedDependencies -PortableRoot $PortableRoot)) {
            throw "依赖包导入后的运行环境检查失败"
        }
        Write-Host "Windows 推荐档依赖包已导入。" -ForegroundColor Green
        return $true
    } catch {
        throw (
            "Windows 推荐档依赖包已取得但无法安全导入：$($_.Exception.Message)。" +
            "文件已保留，便于修复后直接重试。"
        )
    }
}

function Test-ASMRDubberAdvancedDependencies {
    param([Parameter(Mandatory = $true)][string]$PortableRoot)

    $Python = Join-Path $PortableRoot "venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { return $false }
    $Code = @"
import sys
try:
    import accelerate, asmr_dubber.ui, faster_whisper, onnxruntime, qwen_asr, torch, torchaudio, transformers
    assert torch.__version__ == "2.11.0+cu130"
    assert torch.version.cuda == "13.0"
except Exception:
    sys.exit(1)
"@
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @("-c", $Code) -WorkingDirectory $PortableRoot
    return $ExitCode -eq 0
}

function Import-ASMRDubberAdvancedDependencies {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PortableRoot,
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][object]$MirrorConfiguration
    )

    if (Test-ASMRDubberAdvancedDependencies -PortableRoot $PortableRoot) {
        Write-Host (
            "Windows 进阶运行依赖已经完整：PyTorch 2.11 CUDA 13、" +
            "Kotoba/Faster-Whisper、Qwen3 ForcedAligner 与 ASMR VAD。"
        ) -ForegroundColor Green
        return $true
    }

    $Inbox = Join-Path $Root "model-packs"
    $DownloadRoot = Join-Path $PortableRoot "cache\downloads"
    $LocalArchive = Join-Path $Inbox $script:AdvancedDependencyPackName
    $DownloadedArchive = Join-Path $DownloadRoot $script:AdvancedDependencyPackName
    $Archive = $null
    New-Item -ItemType Directory -Force -Path $Inbox, $DownloadRoot | Out-Null

    if (Test-Path -LiteralPath $LocalArchive -PathType Leaf) {
        $Archive = $LocalArchive
        Write-Host "使用项目 model-packs 中的 Windows 进阶依赖包。" `
            -ForegroundColor Cyan
    } else {
        foreach ($Url in Get-ASMRDubberMirrorList `
            -Configuration $MirrorConfiguration `
            -Name "windows_advanced_dependency_archives") {
            try {
                Write-Host (
                    "正在从 ModelScope 下载 Windows 进阶依赖包：" +
                    "PyTorch 2.11 CUDA 13、Kotoba/Faster-Whisper 运行依赖；支持断点续传..."
                ) -ForegroundColor Cyan
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Url -Destination $DownloadedArchive `
                    -Sha256 $script:AdvancedDependencyPackSha256 -Resume | Out-Null
                $Archive = $DownloadedArchive
                break
            } catch {
                Write-Warning "Windows 进阶依赖包来源失败：$($_.Exception.Message)"
            }
        }
    }
    if (-not $Archive) {
        Write-Warning "Windows 进阶依赖包不可用，将尝试已配置的 wheelhouse。"
        return $false
    }

    $File = Get-Item -LiteralPath $Archive
    if ($File.Length -ne $script:AdvancedDependencyPackSize) {
        throw "Windows 进阶依赖包大小不符：$($File.Length)"
    }
    $ActualHash = Get-ASMRDubberFileSha256 -Path $Archive
    if ($ActualHash -ne $script:AdvancedDependencyPackSha256) {
        throw "Windows 进阶依赖包 SHA-256 校验失败：$ActualHash"
    }
    Write-Host "正在导入 Windows 进阶依赖包..." -ForegroundColor Cyan
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @(
            (Join-Path $Root "scripts\import_windows_advanced_dependency_pack.py"),
            $Archive,
            $PortableRoot,
            "--sha256",
            $script:AdvancedDependencyPackSha256
        ) -WorkingDirectory $Root
    if ($ExitCode -ne 0) {
        throw "Windows 进阶依赖包导入进程退出码 $ExitCode"
    }
    Repair-ASMRDubberPortablePythonPaths `
        -Root $Root -PortableRoot $PortableRoot `
        -RuntimeRoot (Join-Path $PortableRoot "runtimes") `
        -Venv (Join-Path $PortableRoot "venv")
    if (-not (Test-ASMRDubberAdvancedDependencies -PortableRoot $PortableRoot)) {
        throw "Windows 进阶依赖包导入后的运行环境检查失败"
    }
    Write-Host "Windows 进阶依赖包已导入并校验。" -ForegroundColor Green
    return $true
}
