[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("qwen3_asr", "faster_whisper", "kotoba_whisper", "openai_whisper", "funasr", "voxcpm2")]
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
    "qwen3_asr" { "local-default" }
    "voxcpm2" { "local-default" }
    "faster_whisper" { "asr-faster-whisper" }
    "kotoba_whisper" { "asr-kotoba-whisper" }
    "openai_whisper" { "asr-openai-whisper" }
    "funasr" { "asr-funasr" }
}

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
        $Process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
            -WorkingDirectory $Root -NoNewWindow -Wait -PassThru
        $ExitCode = $Process.ExitCode
    }
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
if ($Extra -eq "local-default") {
    Install-ASMRDubberSharedFFmpeg -DataRoot $DataRoot | Out-Null
    if (-not $NvidiaSmi) {
        throw "$Backend 需要 NVIDIA GPU；当前机器未检测到 nvidia-smi。"
    }
}

$InstallCudaTorch = ($Extra -eq "local-default") -or (
    ($Backend -eq "kotoba_whisper") -and $NvidiaSmi
)
if ($InstallCudaTorch) {
    Write-Host "检测到 NVIDIA GPU，正在安装 CUDA PyTorch..." -ForegroundColor Cyan
    Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
        -Uv $Uv -Root $Root -MirrorName "pytorch_indexes" -Preferred $TorchIndexUrl `
        -IndexOption "--index" -Arguments @(
            "pip", "install", "--python", $Python, "--reinstall",
            "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"
        )
}

Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
    -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
    -Arguments @(
        "pip", "install", "--python", $Python, "--editable", "$Root[$Extra]"
    )

Write-Host "后端运行环境安装完成。请重启 ASMR Dubber 后使用。" -ForegroundColor Green
