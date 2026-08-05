[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [string]$Version = "0.7.0",
    [switch]$KeepStaging
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
. (Join-Path $Root "scripts\mirrors.ps1")

$PortableRoot = Join-Path $Root ".asmr-dubber"
$PackRoot = Join-Path $Root "model-packs"
$OutputRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    Split-Path -Parent $Root
}
$StagingRoot = Join-Path $OutputRoot ".adr-build"
$PackageRoot = Join-Path $StagingRoot "ASMR-Dubber"
$PackagePortableRoot = Join-Path $PackageRoot ".asmr-dubber"
$OutputName = "ASMR-Dubber-Windows-Recommended-v$Version.zip"
$Output = Join-Path $OutputRoot $OutputName

$DependencyPack = Join-Path $PackRoot `
    "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip"
$ParakeetPack = Join-Path $PackRoot `
    "ASMR-Dubber-ModelPack-parakeet-ja-windows-v0.2.1.zip"
$IndexTtsPack = Join-Path $PackRoot `
    "ASMR-Dubber-ModelPack-indextts2-checkpoints-v0.2.1.zip"
$RequiredPacks = @(
    @{
        Path = $DependencyPack
        Sha256 = "a026ea897a36fa7cf22b2c1b5f8069d9b353c02a1e5285e00d0ea984f9a1472b"
    },
    @{
        Path = $ParakeetPack
        Sha256 = "3a9e95e02df01a40533d5f73893d62fe2bf0bb897b98d2b8e494faa2ed139790"
    },
    @{
        Path = $IndexTtsPack
        Sha256 = "144aa91c4de24faf8d415df4fa4324b831609c4bbcef4406a5db4f2a952e108e"
    }
)

$BasePython = Get-ChildItem `
    (Join-Path $PortableRoot "runtimes\python\cpython-3.12.*-windows-x86_64-none\python.exe") `
    -File -ErrorAction SilentlyContinue | Sort-Object FullName -Descending |
    Select-Object -First 1
$UvRoot = Join-Path $PortableRoot "bootstrap\windows\uv"
$IndexSource = Join-Path $PortableRoot "runtimes\index-tts"
if (-not $BasePython) { throw "缺少项目内的 Windows Python 3.12。" }
if (-not (Test-Path (Join-Path $UvRoot "uv.exe"))) { throw "缺少项目内的 uv.exe。" }
if (-not (Test-Path (Join-Path $IndexSource "pyproject.toml"))) {
    throw "缺少已经安装的 IndexTTS2 源码。"
}

function Invoke-Robocopy {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExcludedDirectories = @(),
        [string[]]$ExcludedFiles = @()
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $Arguments = @(
        $Source, $Destination, "/E", "/COPY:DAT", "/DCOPY:DAT", "/R:2", "/W:1",
        "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"
    )
    if ($ExcludedDirectories.Count -gt 0) {
        $Arguments += "/XD"
        $Arguments += $ExcludedDirectories
    }
    if ($ExcludedFiles.Count -gt 0) {
        $Arguments += "/XF"
        $Arguments += $ExcludedFiles
    }
    $ExitCode = Invoke-ASMRDubberProcess `
        -FilePath (Join-Path $env:SystemRoot "System32\robocopy.exe") `
        -ArgumentList $Arguments -WorkingDirectory $Root
    if ($ExitCode -gt 7) {
        throw "robocopy 复制失败：$Source（退出码 $ExitCode）"
    }
}

function Expand-PackPayload {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $ExtractRoot = Join-Path $StagingRoot "x-$Name"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ExtractRoot
    New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null
    $LongExtractRoot = "\\?\$ExtractRoot"
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $BasePython.FullName `
        -ArgumentList @("-m", "zipfile", "-e", $Archive, $LongExtractRoot) `
        -WorkingDirectory $Root
    if ($ExitCode -ne 0) { throw "模型包解压失败：$Archive" }
    $Payload = Join-Path $ExtractRoot "payload"
    if (-not (Test-Path $Payload)) { throw "压缩包缺少 payload：$Archive" }
    Invoke-Robocopy -Source $Payload -Destination $PackagePortableRoot `
        -ExcludedDirectories @("__pycache__", "pkg_resources\tests") `
        -ExcludedFiles @("*.pyc", "*.pyo")
    Remove-Item -Recurse -Force $ExtractRoot
}

foreach ($Pack in $RequiredPacks) {
    if (-not (Test-Path $Pack.Path)) { throw "缺少必需文件：$($Pack.Path)" }
    Write-Host "正在校验：$([System.IO.Path]::GetFileName($Pack.Path))" -ForegroundColor Cyan
    $ActualHash = Get-ASMRDubberFileSha256 -Path $Pack.Path
    if ($ActualHash -ne $Pack.Sha256) {
        throw "文件 SHA-256 校验失败：$($Pack.Path)"
    }
}

Write-Host "正在生成 Windows 启动器..." -ForegroundColor Cyan
& (Join-Path $Root "launcher\windows\build.ps1")

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $StagingRoot
New-Item -ItemType Directory -Force -Path $PackageRoot, $PackagePortableRoot, $OutputRoot |
    Out-Null

Write-Host "正在复制项目文件..." -ForegroundColor Cyan
Invoke-Robocopy -Source $Root -Destination $PackageRoot `
    -ExcludedDirectories @(
        ".git", ".asmr-dubber", ".venv", "model-packs", "dist", ".pytest_cache",
        ".ruff_cache", ".audit-home", ".test-home", ".test-home-final", ".ui-qa-home",
        "__pycache__"
    ) `
    -ExcludedFiles @("*.pyc", "*.pyo")

Write-Host "正在加入基础档、IndexTTS2 依赖和 FFmpeg..." -ForegroundColor Cyan
Expand-PackPayload -Archive $DependencyPack -Name "dependencies"

Write-Host "正在加入 Parakeet 模型和 CUDA 运行时..." -ForegroundColor Cyan
Expand-PackPayload -Archive $ParakeetPack -Name "parakeet"

Write-Host "正在加入 IndexTTS2 checkpoints..." -ForegroundColor Cyan
Expand-PackPayload -Archive $IndexTtsPack -Name "indextts2"

Write-Host "正在加入便携 Python、uv 和 IndexTTS2 源码..." -ForegroundColor Cyan
Invoke-Robocopy -Source $BasePython.Directory.FullName `
    -Destination (Join-Path $PackagePortableRoot `
        "runtimes\python\$($BasePython.Directory.Name)")
Invoke-Robocopy -Source $UvRoot `
    -Destination (Join-Path $PackagePortableRoot "bootstrap\windows\uv")
Invoke-Robocopy -Source $IndexSource `
    -Destination (Join-Path $PackagePortableRoot "runtimes\index-tts") `
    -ExcludedDirectories @(".venv", "checkpoints", "user-state", ".git", "__pycache__") `
    -ExcludedFiles @("*.pyc", "*.pyo")

New-Item -ItemType Directory -Force -Path @(
    (Join-Path $PackagePortableRoot "cache"),
    (Join-Path $PackagePortableRoot "config"),
    (Join-Path $PackagePortableRoot "logs"),
    (Join-Path $PackagePortableRoot "projects"),
    (Join-Path $PackagePortableRoot "temp")
) | Out-Null

Write-Host "正在修复并检查便携路径..." -ForegroundColor Cyan
$RunCli = Join-Path $PackageRoot "scripts\windows\run-cli.ps1"
$ExitCode = Invoke-ASMRDubberProcess `
    -FilePath (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe") `
    -ArgumentList @(
        "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", $RunCli, "doctor", "--no-network"
    ) `
    -WorkingDirectory $PackageRoot
if ($ExitCode -ne 0) { throw "推荐档便携环境检查失败（退出码 $ExitCode）。" }

$WinRAR = Join-Path $env:ProgramFiles "WinRAR\WinRAR.exe"
if (-not (Test-Path $WinRAR)) { throw "找不到 WinRAR：$WinRAR" }
Remove-Item -Force -ErrorAction SilentlyContinue $Output, "$Output.sha256"
Write-Host "正在创建推荐档 ZIP（约 20 GB，可能需要较长时间）..." `
    -ForegroundColor Cyan
$ExitCode = Invoke-ASMRDubberProcess -FilePath $WinRAR `
    -ArgumentList @("a", "-afzip", "-m1", "-r", $Output, "ASMR-Dubber") `
    -WorkingDirectory $StagingRoot
if ($ExitCode -ne 0) { throw "WinRAR 打包失败（退出码 $ExitCode）。" }

$Hash = Get-ASMRDubberFileSha256 -Path $Output
$Size = (Get-Item $Output).Length
"$Hash  $OutputName" | Set-Content "$Output.sha256" -Encoding ascii
if (-not $KeepStaging) {
    Remove-Item -Recurse -Force $StagingRoot
}
Write-Host "推荐档免安装包已完成：$Output" -ForegroundColor Green
Write-Host "大小：$Size"
Write-Host "SHA-256：$Hash"
