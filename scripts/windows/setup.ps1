[CmdletBinding()]
param(
    [ValidateSet("Core", "Recommended", "Full")]
    [string]$Profile = "Recommended",
    [string]$IndexUrl = "",
    [string]$PythonMirror = "",
    [string]$HuggingFaceEndpoint = "",
    [string]$TorchIndexUrl = "",
    [switch]$SkipRecommendedTTS
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
. (Join-Path $Root "scripts\mirrors.ps1")
$MirrorConfiguration = Get-ASMRDubberMirrorConfiguration -Root $Root
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
$DataRoot = $Paths.Home
$Bootstrap = $Paths.Bootstrap
$UvDir = $Paths.UvDir
$Uv = $Paths.Uv
$Venv = $Paths.Venv
$Python = $Paths.Python
. (Join-Path $Root "scripts\windows-runtime.ps1")

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $StartInfo.FileName = $FilePath
        $StartInfo.WorkingDirectory = $Root
        $StartInfo.UseShellExecute = $false
        foreach ($Argument in $ArgumentList) {
            [void]$StartInfo.ArgumentList.Add($Argument)
        }
        $Process = [System.Diagnostics.Process]::Start($StartInfo)
        $Process.WaitForExit()
        $ExitCode = $Process.ExitCode
    } else {
        # Windows PowerShell 5.1 does not expose ProcessStartInfo.ArgumentList.
        $Process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
            -WorkingDirectory $Root -NoNewWindow -Wait -PassThru
        $ExitCode = $Process.ExitCode
    }
    if ($ExitCode -ne 0) {
        throw "$FailureMessage（退出码 $ExitCode）"
    }
}

New-Item -ItemType Directory -Force -Path $Bootstrap, $UvDir, $DataRoot | Out-Null
$env:UV_NO_MODIFY_PATH = "1"
$env:UV_UNMANAGED_INSTALL = $UvDir
$env:UV_LINK_MODE = "copy"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$PreferredIndex = if ($IndexUrl) {
    $IndexUrl
} elseif ($env:ASMR_DUBBER_PYPI_MIRROR) {
    $env:ASMR_DUBBER_PYPI_MIRROR
} else { "" }
$PreferredHuggingFace = if ($HuggingFaceEndpoint) {
    $HuggingFaceEndpoint
} elseif ($env:ASMR_DUBBER_HF_ENDPOINT) {
    $env:ASMR_DUBBER_HF_ENDPOINT
} else { "" }
$HuggingFaceEndpoints = @(Get-ASMRDubberMirrorList `
    -Configuration $MirrorConfiguration -Name "huggingface_endpoints" `
    -Preferred $PreferredHuggingFace)
$env:ASMR_DUBBER_HF_ENDPOINTS = $HuggingFaceEndpoints -join ";"
$env:HF_ENDPOINT = $HuggingFaceEndpoints[0]

Write-Host "ASMR Dubber · Windows 安装" -ForegroundColor Cyan
Write-Host "项目目录：$Root"
Write-Host "便携目录：$DataRoot"
Write-Host "安装配置：$Profile"

function Test-PortableUv {
    if (-not (Test-Path $Uv)) {
        return $false
    }
    try {
        $ExitCode = Invoke-ASMRDubberProcess -FilePath $Uv `
            -ArgumentList @("--version") -WorkingDirectory $Root
        return $ExitCode -eq 0
    } catch {
        return $false
    }
}

if (-not (Test-PortableUv)) {
    if (Test-Path $Uv) {
        Write-Warning "项目内置 uv 已损坏，将重新下载并修复。"
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $UvDir
        New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    } else {
        Write-Host "项目未包含可用的 uv，正在下载修复副本..." -ForegroundColor Cyan
    }

    $UvArchive = Join-Path $Bootstrap "uv-x86_64-pc-windows-msvc.zip"
    $UvStaging = Join-Path $Bootstrap "uv-extract"
    $UvSha256 = "be8d78c992312212e5cc05e9f9de3fa996db73b7c86a186dfb9231eb9f91d33e"
    $UvInstalled = $false
    foreach ($ArchiveUrl in Get-ASMRDubberMirrorList `
        -Configuration $MirrorConfiguration -Name "uv_archives_windows") {
        try {
            Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                -Url $ArchiveUrl -Destination $UvArchive -Sha256 $UvSha256 -Resume |
                Out-Null
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $UvStaging
            New-Item -ItemType Directory -Force -Path $UvStaging | Out-Null
            Expand-Archive -Path $UvArchive -DestinationPath $UvStaging -Force
            $DownloadedUv = Get-ChildItem $UvStaging -Filter "uv.exe" -File -Recurse |
                Select-Object -First 1
            if (-not $DownloadedUv) {
                throw "uv 压缩包中找不到 uv.exe。"
            }
            New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
            Copy-Item -Force (Join-Path $DownloadedUv.Directory.FullName "*.exe") $UvDir
            if (Test-PortableUv) {
                $UvInstalled = $true
                break
            }
            throw "下载的 uv 无法运行。"
        } catch {
            Write-Warning "uv 下载源失败：$($_.Exception.Message)"
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $UvStaging
        }
    }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $UvStaging
    if (-not $UvInstalled) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $UvDir
    }
}
if (-not (Test-PortableUv)) {
    throw "uv 安装失败：$Uv"
}
Write-Host "项目私有 uv 已就绪。" -ForegroundColor Green

Write-Host "正在准备项目私有 Python 3.12..." -ForegroundColor Cyan
$ManagedPython = Get-ChildItem `
    (Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.12.*-windows-x86_64-none\python.exe") `
    -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $ManagedPython) {
    $PythonMirrors = Get-ASMRDubberMirrorList `
        -Configuration $MirrorConfiguration -Name "python_install_mirrors" `
        -Preferred $PythonMirror
    $PythonInstalled = $false
    foreach ($Candidate in $PythonMirrors) {
        Write-Host "使用 Python 下载源：$Candidate" -ForegroundColor DarkGray
        $env:UV_PYTHON_INSTALL_MIRROR = $Candidate
        $ExitCode = Invoke-ASMRDubberProcess -FilePath $Uv `
            -ArgumentList @("python", "install", "3.12", "--managed-python", "--no-bin") `
            -WorkingDirectory $Root
        if ($ExitCode -eq 0) {
            $PythonInstalled = $true
            break
        }
        Write-Warning "当前 Python 下载源失败，自动切换。"
    }
    if (-not $PythonInstalled) {
        throw "所有 Python 下载源均失败。"
    }
    $ManagedPython = Get-ChildItem `
        (Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.12.*-windows-x86_64-none\python.exe") `
        -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
}
if (-not $ManagedPython) {
    throw "Python 3.12 安装命令已完成，但没有找到解释器。"
}

if (-not (Test-Path $Python)) {
    Invoke-Checked -FilePath $Uv `
        -ArgumentList @("venv", "--python", $ManagedPython.FullName, $Venv) `
        -FailureMessage "虚拟环境创建失败"
}

$NvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
if (-not $NvidiaSmi) {
    $SystemNvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    if (Test-Path $SystemNvidiaSmi) {
        $NvidiaSmi = $SystemNvidiaSmi
    }
}
$InstallDefaultModels = $false
$InstallRecommendedTTS = $false
$InstallParakeet = $false
$Extra = switch ($Profile) {
    "Core" { ".[ui]" }
    "Recommended" {
        $InstallParakeet = $true
        if ($NvidiaSmi) {
            $InstallRecommendedTTS = -not $SkipRecommendedTTS
        }
        ".[ui,asr-faster-whisper,asr-kotoba-whisper]"
    }
    "Full" {
        $InstallParakeet = $true
        if ($NvidiaSmi) {
            $InstallDefaultModels = $true
            $InstallRecommendedTTS = -not $SkipRecommendedTTS
            ".[ui,local-default,asr-faster-whisper,asr-kotoba-whisper,asr-openai-whisper,asr-funasr]"
        } else {
            Write-Warning "未检测到 NVIDIA GPU；跳过 CUDA 专用 Qwen3-ASR/VoxCPM2。"
            ".[ui,asr-faster-whisper,asr-kotoba-whisper,asr-openai-whisper,asr-funasr]"
        }
    }
}

if ($InstallDefaultModels -and $NvidiaSmi) {
    Write-Host "检测到 NVIDIA GPU，正在安装官方 CUDA 13.0 PyTorch..." -ForegroundColor Cyan
    Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
        -Uv $Uv -Root $Root -MirrorName "pytorch_indexes" -Preferred $TorchIndexUrl `
        -IndexOption "--index" -Arguments @(
            "pip", "install", "--python", $Python, "--reinstall",
            "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"
        )
}

Write-Host "正在安装应用依赖：$Extra" -ForegroundColor Cyan
Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
    -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
    -Arguments @("pip", "install", "--python", $Python, "--editable", $Extra)
# PyTorch 2.11 requires setuptools <82. Upgrade existing portable environments
# to the newest compatible release instead of retaining an older installer.
Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
    -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
    -Arguments @(
        "pip", "install", "--python", $Python, "setuptools>=78.1.1,<82"
    )
Invoke-Checked -FilePath $Python `
    -ArgumentList @("-m", "compileall", "-q", "-f", (Join-Path $Root "src\asmr_dubber")) `
    -FailureMessage "应用字节码刷新失败"

if ($InstallDefaultModels) {
    $SharedFFmpegBin = Install-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
    Write-Host "共享版 FFmpeg：$SharedFFmpegBin" -ForegroundColor DarkGray
    Write-Host "正在下载并校验默认模型（已有缓存会复用）..." -ForegroundColor Cyan
    Invoke-Checked -FilePath $Python `
        -ArgumentList @(
            "-m", "asmr_dubber.cli", "download-models", "--backend",
            $(if ($Profile -eq "Full") { "all" } else { "qwen3_asr" })
        ) `
        -FailureMessage "默认模型下载或校验失败"
}

if ($InstallParakeet) {
    Write-Host "正在安装推荐 ASR：Parakeet 日语..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "install-parakeet.ps1") -Variant Auto
}

if ($InstallRecommendedTTS) {
    Write-Host "正在安装推荐 TTS：IndexTTS2（约需 20 GB）..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "install-indextts2.ps1") `
        -IndexUrl $PreferredIndex -HuggingFaceEndpoint $PreferredHuggingFace
}

Write-Host "正在执行环境检查..." -ForegroundColor Cyan
try {
    Invoke-Checked -FilePath $Python `
        -ArgumentList @("-m", "asmr_dubber.cli", "doctor", "--no-network") `
        -FailureMessage "环境检查未完全通过"
} catch {
    Write-Warning "核心程序已经安装，但当前选择的本地模型尚未全部可用。请在设置 → 设备与模型中查看。"
}

Write-Host ""
Write-Host "安装完成。运行项目根目录的 ASMR-Dubber.exe 启动界面。" -ForegroundColor Green
Write-Host "卸载方法：删除整个项目目录；程序不会在 AppData 留下文件。" -ForegroundColor DarkGray
