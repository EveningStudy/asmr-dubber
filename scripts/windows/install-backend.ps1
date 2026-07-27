[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("faster_whisper", "kotoba_whisper")]
    [string]$Backend,
    [string]$TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
. (Join-Path $Root "scripts\mirrors.ps1")
$MirrorConfiguration = Get-ASMRDubberMirrorConfiguration -Root $Root
. (Join-Path $Root "scripts\windows-runtime.ps1")
. (Join-Path $Root "scripts\windows\wheelhouse.ps1")
$DataRoot = $Paths.Home
$Uv = $Paths.Uv
$Python = $Paths.Python
if (-not (Test-Path $Uv) -or -not (Test-Path $Python)) {
    throw "缺少应用安装环境；请先运行项目根目录的 ASMR-Dubber-Setup.exe。"
}

$env:UV_LINK_MODE = "copy"
$PreferredIndex = if ($env:ASMR_DUBBER_PYPI_MIRROR) {
    $env:ASMR_DUBBER_PYPI_MIRROR
} else { "" }

$Extra = switch ($Backend) {
    "faster_whisper" { "asr-faster-whisper" }
    "kotoba_whisper" { "asr-kotoba-whisper" }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $FilePath `
        -ArgumentList $ArgumentList -WorkingDirectory $Root
    if ($ExitCode -ne 0) {
        throw "$FailureMessage（退出码 $ExitCode）"
    }
}

Write-Host "正在安装后端：$Backend" -ForegroundColor Cyan
$NvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
if (-not $NvidiaSmi) {
    $SystemNvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    if (Test-Path $SystemNvidiaSmi) {
        $NvidiaSmi = $SystemNvidiaSmi
    }
}
$InstallCudaTorch = ($Backend -eq "kotoba_whisper") -and $NvidiaSmi
if ($InstallCudaTorch) {
    Write-Host "检测到 NVIDIA GPU，正在安装 CUDA PyTorch..." -ForegroundColor Cyan
    $CudaArguments = @(
        "pip", "install", "--python", $Python, "--reinstall",
        "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"
    )
    $CudaWheelhouse = Get-ASMRDubberWheelhouse `
        -Root $Root -PortableRoot $DataRoot -MirrorConfiguration $MirrorConfiguration `
        -ArchiveName "ASMR-Dubber-Windows-CUDA130-Wheelhouse-v0.4.0.zip" `
        -ArchiveMirrorName "windows_cuda_wheelhouse_archives" `
        -ChecksumMirrorName "windows_cuda_wheelhouse_checksums"
    if ($CudaWheelhouse) {
        Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
            -Wheelhouse $CudaWheelhouse -Arguments $CudaArguments `
            -FailureMessage "CUDA PyTorch wheelhouse 安装失败"
    } else {
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $Root -MirrorName "pytorch_indexes" -Preferred $TorchIndexUrl `
            -IndexOption "--index" -Arguments $CudaArguments
    }
}

$ApplicationArguments = @(
    "pip", "install", "--python", $Python, "--editable", "$Root[$Extra]"
)
$ApplicationWheelhouse = Get-ASMRDubberWheelhouse `
    -Root $Root -PortableRoot $DataRoot -MirrorConfiguration $MirrorConfiguration `
    -ArchiveName "ASMR-Dubber-Windows-Wheelhouse-v0.4.0.zip" `
    -ArchiveMirrorName "windows_application_wheelhouse_archives" `
    -ChecksumMirrorName "windows_application_wheelhouse_checksums"
if ($ApplicationWheelhouse) {
    Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
        -Wheelhouse $ApplicationWheelhouse -Arguments $ApplicationArguments `
        -FailureMessage "后端依赖 wheelhouse 安装失败"
} else {
    Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
        -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
        -Arguments $ApplicationArguments
}

Write-Host "后端运行环境安装完成。请重启 ASMR Dubber 后使用。" -ForegroundColor Green
