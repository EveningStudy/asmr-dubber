[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("qwen3_asr", "faster_whisper", "kotoba_whisper", "openai_whisper", "funasr", "voxcpm2")]
    [string]$Backend,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
. (Join-Path $Root "scripts\windows-runtime.ps1")
$DataRoot = $Paths.Home
$Uv = $Paths.Uv
$Python = $Paths.Python
if (-not (Test-Path $Uv) -or -not (Test-Path $Python)) {
    throw "缺少应用安装环境；请先双击项目根目录的 ASMR-Dubber.exe。"
}

$env:UV_LINK_MODE = "copy"

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
if ($Extra -eq "local-default") {
    Install-ASMRDubberSharedFFmpeg -DataRoot $DataRoot | Out-Null
    $NvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if (-not $NvidiaSmi) {
        $SystemNvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (Test-Path $SystemNvidiaSmi) {
            $NvidiaSmi = $SystemNvidiaSmi
        }
    }
    if (-not $NvidiaSmi) {
        throw "$Backend 需要 NVIDIA GPU；当前机器未检测到 nvidia-smi。"
    }
    Write-Host "检测到 NVIDIA GPU，正在安装 CUDA PyTorch..." -ForegroundColor Cyan
    Invoke-Checked -FilePath $Uv `
        -ArgumentList @(
            "pip", "install", "--python", $Python, "--reinstall",
            "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130",
            "--index", $TorchIndexUrl
        ) `
        -FailureMessage "CUDA PyTorch 安装失败"
}

Invoke-Checked -FilePath $Uv `
    -ArgumentList @(
        "pip", "install", "--python", $Python, "--editable", "$Root[$Extra]"
    ) `
    -FailureMessage "Python 后端安装失败"

Write-Host "后端运行环境安装完成。请重启 ASMR Dubber 后使用。" -ForegroundColor Green
