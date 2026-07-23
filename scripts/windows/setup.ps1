[CmdletBinding()]
param(
    [ValidateSet("Core", "Recommended", "Full")]
    [string]$Profile = "Recommended",
    [string]$IndexUrl = "",
    [string]$PythonMirror = "",
    [string]$HuggingFaceEndpoint = "",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [switch]$SkipRecommendedTTS
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
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

if ($IndexUrl) {
    $env:UV_DEFAULT_INDEX = $IndexUrl
} elseif ($env:ASMR_DUBBER_PYPI_MIRROR) {
    $env:UV_DEFAULT_INDEX = $env:ASMR_DUBBER_PYPI_MIRROR
}
if ($HuggingFaceEndpoint) {
    $env:HF_ENDPOINT = $HuggingFaceEndpoint.TrimEnd("/")
}

Write-Host "ASMR Dubber · Windows 安装" -ForegroundColor Cyan
Write-Host "项目目录：$Root"
Write-Host "便携目录：$DataRoot"
Write-Host "安装配置：$Profile"

if (-not (Test-Path $Uv)) {
    Write-Host "正在安装项目私有 uv（不会修改系统 PATH）..." -ForegroundColor Cyan
    $InstallerUrl = "https://astral.sh/uv/0.11.30/install.ps1"
    $Installer = Invoke-RestMethod -Uri $InstallerUrl -UseBasicParsing
    Invoke-Expression $Installer
}
if (-not (Test-Path $Uv)) {
    throw "uv 安装失败：$Uv"
}

Write-Host "正在准备项目私有 Python 3.12..." -ForegroundColor Cyan
$ManagedPython = Get-ChildItem `
    (Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.12.*-windows-x86_64-none\python.exe") `
    -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $ManagedPython) {
    $PythonArguments = @("python", "install", "3.12", "--managed-python", "--no-bin")
    if ($PythonMirror) {
        $PythonArguments += @("--mirror", $PythonMirror)
    }
    Invoke-Checked -FilePath $Uv -ArgumentList $PythonArguments `
        -FailureMessage "Python 3.12 安装失败"
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
    Invoke-Checked -FilePath $Uv `
        -ArgumentList @(
            "pip", "install", "--python", $Python, "--reinstall",
            "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130", "--index", $TorchIndexUrl
        ) `
        -FailureMessage "CUDA PyTorch 安装失败"
}

Write-Host "正在安装应用依赖：$Extra" -ForegroundColor Cyan
Push-Location $Root
try {
    Invoke-Checked -FilePath $Uv `
        -ArgumentList @("pip", "install", "--python", $Python, "--editable", $Extra) `
        -FailureMessage "应用依赖安装失败"
} finally {
    Pop-Location
}
# PyTorch 2.11 requires setuptools <82. Upgrade existing portable environments
# to the newest compatible release instead of retaining an older installer.
Invoke-Checked -FilePath $Uv `
    -ArgumentList @(
        "pip", "install", "--python", $Python, "setuptools>=78.1.1,<82"
    ) `
    -FailureMessage "setuptools 更新失败"
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
        -IndexUrl $IndexUrl -HuggingFaceEndpoint $HuggingFaceEndpoint
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
Write-Host "安装完成。双击项目根目录的 ASMR-Dubber.exe 启动界面。" -ForegroundColor Green
Write-Host "卸载方法：删除整个项目目录；程序不会在 AppData 留下文件。" -ForegroundColor DarkGray
