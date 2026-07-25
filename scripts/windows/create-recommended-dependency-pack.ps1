[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [string]$PackVersion = "1.0.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
. (Join-Path $Root "scripts\mirrors.ps1")
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
. (Join-Path $Root "scripts\windows-runtime.ps1")
. (Join-Path $Root "scripts\windows\recommended-dependencies.ps1")

$Uv = $Paths.Uv
$BasePython = Get-ChildItem `
    (Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.12.*-windows-x86_64-none\python.exe") `
    -File -ErrorAction SilentlyContinue | Sort-Object FullName -Descending |
    Select-Object -First 1
$IndexPythonRoot = Join-Path $Paths.Runtimes `
    "python\cpython-3.11.13-windows-x86_64-none"
$IndexVenv = Join-Path $Paths.Runtimes "index-tts\.venv"
$FFmpegRuntime = Join-Path $Paths.Runtimes "ffmpeg-shared"
if (-not (Test-Path $Uv) -or -not $BasePython) {
    throw "缺少项目 uv 或 Windows Python 3.12 运行时。"
}
if (-not (Test-Path $IndexPythonRoot) -or -not (Test-Path $IndexVenv)) {
    throw "请先在 Windows 上完整安装并检查 IndexTTS2。"
}
if (-not (Get-ASMRDubberSharedFFmpegBin -DataRoot $Paths.Home)) {
    throw "请先安装并检查 Windows shared FFmpeg。"
}

function Copy-DependencyTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $Arguments = @(
        $Source, $Destination, "/E", "/COPY:DAT", "/DCOPY:DAT", "/R:2", "/W:1",
        "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
        "/XD", "__pycache__", "/XF", "*.pyc", "*.pyo"
    )
    $Process = Start-Process -FilePath (Join-Path $env:SystemRoot "System32\robocopy.exe") `
        -ArgumentList $Arguments -WorkingDirectory $Root -NoNewWindow -Wait -PassThru
    if ($Process.ExitCode -gt 7) {
        throw "robocopy 复制失败：$Source（退出码 $($Process.ExitCode)）"
    }
}

$OutputRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    Join-Path $Root "model-packs"
}
$PackName = "ASMR-Dubber-Windows-Recommended-Dependencies-v$PackVersion.zip"
$Output = Join-Path $OutputRoot $PackName
$Staging = Join-Path $Paths.Temp "windows-recommended-dependencies-pack"
$Payload = Join-Path $Staging "payload"
$CoreVenv = Join-Path $Payload "venv"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Staging
New-Item -ItemType Directory -Force -Path $CoreVenv, $OutputRoot | Out-Null

Write-Host "正在建立最小 Windows Core/UI 环境..." -ForegroundColor Cyan
$Create = Invoke-ASMRDubberProcess -FilePath $Uv `
    -ArgumentList @("venv", "--python", $BasePython.FullName, $CoreVenv) `
    -WorkingDirectory $Root
if ($Create -ne 0) { throw "Core/UI 虚拟环境创建失败。" }
$MirrorConfiguration = Get-ASMRDubberMirrorConfiguration -Root $Root
Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
    -Uv $Uv -Root $Root -MirrorName "pypi_indexes" `
    -Arguments @(
        "pip", "install", "--python", (Join-Path $CoreVenv "Scripts\python.exe"),
        "--editable", "$Root[ui]", "setuptools>=78.1.1,<82"
    )
$CoreCheck = Invoke-ASMRDubberProcess -FilePath (Join-Path $CoreVenv "Scripts\python.exe") `
    -ArgumentList @("-c", "import asmr_dubber.ui, av, gradio, soundfile, setuptools") `
    -WorkingDirectory $Root
if ($CoreCheck -ne 0) { throw "Core/UI 环境检查失败。" }

Write-Host "正在收集 IndexTTS2 CUDA/Torch 环境与 Python 3.11..." -ForegroundColor Cyan
$IndexTarget = Join-Path $Payload "runtimes\index-tts\.venv"
$PythonTarget = Join-Path $Payload `
    "runtimes\python\cpython-3.11.13-windows-x86_64-none"
$FFmpegTarget = Join-Path $Payload "runtimes\ffmpeg-shared"
New-Item -ItemType Directory -Force `
    -Path (Split-Path $IndexTarget), (Split-Path $PythonTarget), (Split-Path $FFmpegTarget) |
    Out-Null
Copy-DependencyTree -Source $IndexVenv -Destination $IndexTarget
Copy-DependencyTree -Source $IndexPythonRoot -Destination $PythonTarget
Copy-DependencyTree -Source $FFmpegRuntime -Destination $FFmpegTarget

$Manifest = [ordered]@{
    schema_version = 1
    pack_id = "windows-recommended-dependencies"
    pack_version = $PackVersion
    platform = "windows"
    architecture = "x86_64"
    python = @("3.12", "3.11.13")
    indextts_revision = "13495845e3028f0bb6ca1462ad22aa0e76349e40"
    components = @("application-ui", "indextts2", "ffmpeg-shared")
}
$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content (Join-Path $Staging "dependency-pack.json") -Encoding utf8NoBOM
$Notices = @"
ASMR Dubber Windows Recommended dependency pack

This archive redistributes unmodified installed dependencies for Windows x86_64.
Package license files remain inside their *.dist-info directories.
IndexTTS2: https://github.com/index-tts/index-tts
PyTorch: https://pytorch.org/
FFmpeg LGPL shared build: https://github.com/BtbN/FFmpeg-Builds
Python: https://www.python.org/
"@
Set-Content (Join-Path $Staging "THIRD_PARTY_NOTICES.txt") `
    -Value $Notices -Encoding utf8NoBOM

Remove-Item -Force -ErrorAction SilentlyContinue $Output, "$Output.sha256"
$WinRARCommand = Get-Command "WinRAR.exe" -ErrorAction SilentlyContinue
$WinRARPath = if ($WinRARCommand) { $WinRARCommand.Source } else { "" }
if (-not $WinRARPath) {
    $Candidate = Join-Path $env:ProgramFiles "WinRAR\WinRAR.exe"
    if (Test-Path $Candidate) { $WinRARPath = $Candidate }
}
Write-Host "正在创建 ZIP64 依赖包（此步骤可能需要较长时间）..." -ForegroundColor Cyan
if ($WinRARPath) {
    $Process = Start-Process -FilePath $WinRARPath `
        -ArgumentList @("a", "-afzip", "-m1", "-mt", "-r", "-ep1", $Output, "$Staging\*") `
        -WorkingDirectory $Staging -NoNewWindow -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "WinRAR 打包失败：$($Process.ExitCode)" }
} else {
    & $BasePython.FullName -m zipfile -c $Output `
        (Join-Path $Staging "dependency-pack.json") `
        (Join-Path $Staging "THIRD_PARTY_NOTICES.txt") $Payload
    if ($LASTEXITCODE -ne 0) { throw "Python ZIP64 打包失败。" }
}

$Hash = (Get-FileHash $Output -Algorithm SHA256).Hash.ToLowerInvariant()
$Size = (Get-Item $Output).Length
"$Hash  $PackName" | Set-Content "$Output.sha256" -Encoding ascii
Write-Host "依赖包已完成：$Output" -ForegroundColor Green
Write-Host "大小：$Size"
Write-Host "SHA-256：$Hash"
