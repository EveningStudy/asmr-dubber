Set-StrictMode -Version 2.0

$script:RecommendedDependencyPackName = `
    "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip"
$script:RecommendedDependencyPackSha256 = `
    "a026ea897a36fa7cf22b2c1b5f8069d9b353c02a1e5285e00d0ea984f9a1472b"
$script:RecommendedDependencyPackSize = 4060845976

function Test-ASMRDubberCoreRuntime {
    param([Parameter(Mandatory = $true)][string]$PortableRoot)

    $Python = Join-Path $PortableRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) { return $false }
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @(
            "-c",
            "import asmr_dubber.ui, av, gradio, soundfile, setuptools"
        ) -WorkingDirectory $PortableRoot
    return $ExitCode -eq 0
}

function Test-ASMRDubberIndexRuntimeDependencies {
    param([Parameter(Mandatory = $true)][string]$PortableRoot)

    $BasePython = Join-Path $PortableRoot `
        "runtimes\python\cpython-3.11.13-windows-x86_64-none\python.exe"
    $Python = Join-Path $PortableRoot "runtimes\index-tts\.venv\Scripts\python.exe"
    if (-not (Test-Path $BasePython) -or -not (Test-Path $Python)) { return $false }
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @("-c", "import modelscope, torch, torchaudio, transformers") `
        -WorkingDirectory $PortableRoot
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
        Write-Host "Windows Recommended 运行依赖已经完整，无需下载依赖包。" `
            -ForegroundColor Green
        return $true
    }
    if ($script:RecommendedDependencyPackSize -le 0 -or `
        $script:RecommendedDependencyPackSha256 -notmatch "^[0-9a-f]{64}$") {
        Write-Warning "Windows Recommended 依赖包元数据尚未发布，继续在线安装。"
        return $false
    }

    $Inbox = Join-Path $Root "model-packs"
    $LocalArchive = Join-Path $Inbox $script:RecommendedDependencyPackName
    $DownloadRoot = Join-Path $PortableRoot "cache\downloads"
    $DownloadedArchive = Join-Path $DownloadRoot $script:RecommendedDependencyPackName
    $Archive = $null
    $DeleteAfterImport = $false
    New-Item -ItemType Directory -Force -Path $Inbox, $DownloadRoot | Out-Null

    if (Test-Path $LocalArchive) {
        $Archive = $LocalArchive
        Write-Host "使用项目 model-packs 中的 Windows Recommended 依赖包。" `
            -ForegroundColor Cyan
    } else {
        foreach ($Url in Get-ASMRDubberMirrorList `
            -Configuration $MirrorConfiguration `
            -Name "windows_recommended_dependency_archives") {
            try {
                Write-Host "正在下载 Windows Recommended 依赖包；支持中断后续传..." `
                    -ForegroundColor Cyan
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Url -Destination $DownloadedArchive `
                    -Sha256 $script:RecommendedDependencyPackSha256 -Resume | Out-Null
                $Archive = $DownloadedArchive
                $DeleteAfterImport = $true
                break
            } catch {
                Write-Warning "当前 Windows Recommended 依赖包来源失败：$($_.Exception.Message)"
            }
        }
    }
    if (-not $Archive) {
        Write-Warning "Windows Recommended 依赖包不可用，继续使用原有在线安装方式。"
        return $false
    }

    try {
        $File = Get-Item $Archive
        if ($File.Length -ne $script:RecommendedDependencyPackSize) {
            throw "依赖包大小不符：$($File.Length)"
        }
        $ActualHash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $script:RecommendedDependencyPackSha256) {
            throw "依赖包 SHA-256 校验失败：$ActualHash"
        }
        Write-Host "正在导入 Windows Recommended 依赖包..." -ForegroundColor Cyan
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
        if ($DeleteAfterImport -and (Test-Path $Archive)) {
            Remove-Item -Force $Archive
        }
        Write-Host "Windows Recommended 依赖包已导入。" -ForegroundColor Green
        return $true
    } catch {
        Write-Warning "Windows Recommended 依赖包无法使用，将回退在线安装：$($_.Exception.Message)"
        return $false
    }
}
