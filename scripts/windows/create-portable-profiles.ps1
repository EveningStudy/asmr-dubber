[CmdletBinding()]
param(
    [ValidateSet("核心", "推荐", "进阶", "Core", "Recommended", "Advanced")]
    [string[]]$Profiles = @("推荐"),
    [string]$OutputDirectory = "",
    [string]$Version = "0.7.3",
    [switch]$KeepStaging
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
. (Join-Path $Root "scripts\mirrors.ps1")
. (Join-Path $Root "scripts\portable-runtime.ps1")

$SourcePortable = Join-Path $Root ".asmr-dubber"
$OutputRoot = if ($OutputDirectory) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    Join-Path (Split-Path -Parent $Root) "ASMR-Dubber-Windows-Portable-ModelScope"
}
$DriveRoot = [System.IO.Path]::GetPathRoot($Root)
$StagingRoot = Join-Path $DriveRoot "adrp-$PID"
$PackageRoot = Join-Path $StagingRoot "ASMR-Dubber"
$PackagePortable = Join-Path $PackageRoot ".asmr-dubber"
$OriginalTemp = $env:TEMP
$OriginalTmp = $env:TMP
$OriginalTmpDir = $env:TMPDIR

$ProfileMap = @{
    "核心" = "core"
    "Core" = "core"
    "推荐" = "recommended"
    "Recommended" = "recommended"
    "进阶" = "advanced"
    "Advanced" = "advanced"
}
$Requested = @{
    core = $false
    recommended = $false
    advanced = $false
}
foreach ($Profile in $Profiles) {
    $Requested[$ProfileMap[$Profile]] = $true
}

$Artifacts = @{
    recommended_dependencies = @{
        Name = "ASMR-Dubber-Windows-Recommended-Dependencies-v1.0.0.zip"
        Size = 4060845976
        Sha256 = "a026ea897a36fa7cf22b2c1b5f8069d9b353c02a1e5285e00d0ea984f9a1472b"
    }
    advanced_dependencies = @{
        Name = "ASMR-Dubber-Windows-Advanced-Dependencies-v1.0.0.zip"
        Size = 2905762138
        Sha256 = "bafd2268de9a83bbf391ba8918d1798d24f703b023af70e8f623b2dbffc9a178"
    }
    parakeet = @{
        Name = "ASMR-Dubber-ModelPack-parakeet-ja-windows-v0.2.1.zip"
        Size = 4070471378
        Sha256 = "3a9e95e02df01a40533d5f73893d62fe2bf0bb897b98d2b8e494faa2ed139790"
    }
    indextts2 = @{
        Name = "ASMR-Dubber-ModelPack-indextts2-checkpoints-v0.2.1.zip"
        Size = 11189524132
        Sha256 = "144aa91c4de24faf8d415df4fa4324b831609c4bbcef4406a5db4f2a952e108e"
    }
    kotoba = @{
        Name = "ASMR-Dubber-ModelPack-kotoba-whisper-v2.2-v1.0.0.zip"
        Size = 3027748160
        Sha256 = "a5da2f63fd2c4972dad4cc53db89e0d0250af9d4431905b8c558d55169734c46"
    }
    faster_whisper = @{
        Name = "ASMR-Dubber-ModelPack-faster-whisper-large-v2-v1.0.0.zip"
        Size = 3087767076
        Sha256 = "4a4a213561d327e82d5dc5a8e8c071313bd948ad90f7b4c51e650044fd3bc949"
    }
    qwen_aligner = @{
        Name = "ASMR-Dubber-ModelPack-qwen3-forced-aligner-v1.0.0.zip"
        Size = 1837358823
        Sha256 = "6697b80bfba3a182a86290ba0f7b8adc958d7112bfe6cc9caa73bc7207b74242"
    }
    asmr_vad = @{
        Name = "ASMR-Dubber-ModelPack-whisper-vad-asmr-onnx-v1.0.0.zip"
        Size = 54692316
        Sha256 = "f7d4c6ec7c9576d325685ffeaf7a39e5160fa1d3e6fe94ae60ed7dc866e5eaa9"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $FilePath `
        -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory
    if ($ExitCode -ne 0) {
        throw "$FailureMessage（退出码 $ExitCode）"
    }
}

function Resolve-Artifact {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Candidates = @(
        (Join-Path $Root "model-packs\$Name"),
        (Join-Path $SourcePortable "cache\downloads\$Name")
    )
    foreach ($LocalRoot in @($env:ASMR_DUBBER_LOCAL_CACHE_ROOTS -split ";")) {
        if ($LocalRoot) {
            $Candidates += Join-Path $LocalRoot "model-packs\$Name"
            $Candidates += Join-Path $LocalRoot ".asmr-dubber\cache\downloads\$Name"
        }
    }
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    throw "找不到本地制品：$Name"
}

function Test-Artifact {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][hashtable]$Spec
    )
    $Path = Resolve-Artifact -Name $Spec.Name
    $File = Get-Item -LiteralPath $Path
    if ($File.Length -ne [int64]$Spec.Size) {
        throw "$($Spec.Name) 字节数不符：$($File.Length)"
    }
    Write-Host "正在校验 $($Spec.Name)..." -ForegroundColor Cyan
    $ActualHash = Get-ASMRDubberFileSha256 -Path $Path
    if ($ActualHash -ne $Spec.Sha256) {
        throw "$($Spec.Name) SHA-256 不符：$ActualHash"
    }
    $Spec.Path = $Path
    Write-Host "已通过：$Id" -ForegroundColor Green
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
    $Robocopy = Join-Path $env:SystemRoot "System32\robocopy.exe"
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Robocopy `
        -ArgumentList $Arguments -WorkingDirectory $Root
    if ($ExitCode -gt 7) {
        throw "robocopy 复制失败：$Source（退出码 $ExitCode）"
    }
}

function Copy-SourceTree {
    Write-Host "正在复制程序源码和文档..." -ForegroundColor Cyan
    $Files = @(& git -C $Root ls-files --cached --others --exclude-standard)
    if ($LASTEXITCODE -ne 0 -or $Files.Count -eq 0) {
        throw "无法从 Git 工作区取得发布文件清单。"
    }
    foreach ($RelativeText in $Files) {
        $Relative = $RelativeText.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
        $Source = Join-Path $Root $Relative
        if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
            continue
        }
        $Destination = Join-Path $PackageRoot $Relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) |
            Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Initialize-PackageBase {
    $Recommended = $Artifacts.recommended_dependencies
    Invoke-Checked -FilePath $BasePython.FullName `
        -ArgumentList @(
            (Join-Path $Root "scripts\import_windows_dependency_pack.py"),
            $Recommended.Path, $PackagePortable, "--sha256", $Recommended.Sha256
        ) -WorkingDirectory $Root -FailureMessage "基础运行环境导入失败"

    foreach ($Path in @(
        (Join-Path $PackagePortable "runtimes\python\cpython-3.11.13-windows-x86_64-none"),
        (Join-Path $PackagePortable "runtimes\index-tts"),
        (Join-Path $PackagePortable "runtimes\windows-recommended-dependencies.json")
    )) {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    }

    Invoke-Robocopy -Source $BasePython.Directory.FullName `
        -Destination (Join-Path $PackagePortable `
            "runtimes\python\$($BasePython.Directory.Name)")
    Invoke-Robocopy -Source $UvRoot `
        -Destination (Join-Path $PackagePortable "bootstrap\windows\uv")
}

function Initialize-RecommendedPackage {
    $Recommended = $Artifacts.recommended_dependencies
    Invoke-Checked -FilePath $BasePython.FullName `
        -ArgumentList @(
            (Join-Path $Root "scripts\import_windows_dependency_pack.py"),
            $Recommended.Path, $PackagePortable, "--sha256", $Recommended.Sha256
        ) -WorkingDirectory $Root -FailureMessage "推荐运行环境导入失败"

    Invoke-Robocopy -Source $IndexSource `
        -Destination (Join-Path $PackagePortable "runtimes\index-tts") `
        -ExcludedDirectories @(".venv", "checkpoints", "user-state", ".git", "__pycache__") `
        -ExcludedFiles @("*.pyc", "*.pyo")

    Repair-PackagePaths
    Import-ModelPack -Artifact $Artifacts.parakeet
    Import-ModelPack -Artifact $Artifacts.indextts2
}

function Initialize-AdvancedPackage {
    $Advanced = $Artifacts.advanced_dependencies
    Invoke-Checked -FilePath $BasePython.FullName `
        -ArgumentList @(
            (Join-Path $Root "scripts\import_windows_advanced_dependency_pack.py"),
            $Advanced.Path, $PackagePortable, "--sha256", $Advanced.Sha256
        ) -WorkingDirectory $Root -FailureMessage "进阶运行环境导入失败"
    Repair-PackagePaths
    foreach ($Artifact in @(
        $Artifacts.kotoba,
        $Artifacts.faster_whisper,
        $Artifacts.qwen_aligner,
        $Artifacts.asmr_vad
    )) {
        Import-ModelPack -Artifact $Artifact
    }
}

function Repair-PackagePaths {
    Initialize-ASMRDubberPortableEnvironment -Root $PackageRoot -Create | Out-Null
    $RunCli = Join-Path $PackageRoot "scripts\windows\run-cli.ps1"
    $PowerShell51 = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    Invoke-Checked -FilePath $PowerShell51 `
        -ArgumentList @(
            "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", $RunCli, "--help"
        ) -WorkingDirectory $PackageRoot -FailureMessage "便携路径修复失败"
}

function Import-ModelPack {
    param([Parameter(Mandatory = $true)][hashtable]$Artifact)
    $RunCli = Join-Path $PackageRoot "scripts\windows\run-cli.ps1"
    $PowerShell51 = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    Invoke-Checked -FilePath $PowerShell51 `
        -ArgumentList @(
            "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", $RunCli, "import-model-packs", $Artifact.Path
        ) -WorkingDirectory $PackageRoot `
        -FailureMessage "模型包导入失败：$($Artifact.Name)"
}

function Write-PackageDescription {
    param(
        [Parameter(Mandatory = $true)][string]$Profile,
        [Parameter(Mandatory = $true)][string]$Contents
    )
    $Text = @"
ASMR Dubber $Version · Windows $Profile 免安装包

适用系统：64 位 Windows 10/11（x86_64）

使用方法：
1. 完整解压整个压缩包，不要在压缩软件内直接运行。
2. 双击 ASMR-Dubber.exe。
3. 不需要先运行 ASMR-Dubber-Setup.exe。

本包内容：
$Contents

所有项目、设置、模型缓存和明文 API Key 默认写入本目录的 .asmr-dubber。
停止程序后删除整个文件夹即可卸载。请勿分享包含个人项目或密钥的使用后目录。

ASMR-Dubber-Setup.exe 仅保留用于以后主动修复或改装其它档位，不是首次运行的前置步骤。
"@
    [System.IO.File]::WriteAllText(
        (Join-Path $PackageRoot "README-PORTABLE.txt"),
        $Text,
        (New-Object System.Text.UTF8Encoding($true))
    )
}

function Clear-PrivateAndTransientState {
    foreach ($Name in @("cache", "config", "logs", "projects", "temp", "t")) {
        $Path = Join-Path $PackagePortable $Name
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
    foreach ($Path in @(
        (Join-Path $PackagePortable ".runtime-install.lock"),
        (Join-Path $PackagePortable "config\settings.json"),
        (Join-Path $PackagePortable "config\secrets.json")
    )) {
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    }
}

function Test-PackageProfile {
    param([Parameter(Mandatory = $true)][string]$Profile)
    $Paths = Initialize-ASMRDubberPortableEnvironment -Root $PackageRoot -Create
    Invoke-Checked -FilePath $Paths.Python `
        -ArgumentList @(
            (Join-Path $PackageRoot "scripts\verify_portable_profile.py"),
            "--profile", $Profile
        ) -WorkingDirectory $PackageRoot -FailureMessage "$Profile 便携包检查失败"
    $SelfTest = Join-Path $PackagePortable "temp\launcher-self-test.txt"
    Invoke-Checked -FilePath (Join-Path $PackageRoot "ASMR-Dubber.exe") `
        -ArgumentList @("--self-test", $SelfTest) `
        -WorkingDirectory $PackageRoot -FailureMessage "$Profile 启动器检查失败"
}

function New-PortableArchive {
    param(
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$ArchiveLabel
    )
    $env:TEMP = $OriginalTemp
    $env:TMP = $OriginalTmp
    $env:TMPDIR = $OriginalTmpDir
    Clear-PrivateAndTransientState
    $OutputName = "ASMR-Dubber-Windows-$ArchiveLabel-Portable-v$Version.zip"
    $Output = Join-Path $OutputRoot $OutputName
    Remove-Item -LiteralPath $Output, "$Output.sha256" -Force -ErrorAction SilentlyContinue
    Write-Host "正在创建 $ArchiveLabel ZIP64..." -ForegroundColor Cyan
    Invoke-Checked -FilePath $WinRAR `
        -ArgumentList @("a", "-afzip", "-m1", "-r", "-y", $Output, "ASMR-Dubber") `
        -WorkingDirectory $StagingRoot -FailureMessage "$ArchiveLabel ZIP 创建失败"
    $Hash = Get-ASMRDubberFileSha256 -Path $Output
    $Size = (Get-Item -LiteralPath $Output).Length
    "$Hash  $OutputName" | Set-Content "$Output.sha256" -Encoding ascii
    Write-Host "$ArchiveLabel 完成：$Output" -ForegroundColor Green
    Write-Host "大小：$Size"
    Write-Host "SHA-256：$Hash"
}

$BasePython = Get-ChildItem `
    (Join-Path $SourcePortable `
        "runtimes\python\cpython-3.12.*-windows-x86_64-none\python.exe") `
    -File -ErrorAction SilentlyContinue | Sort-Object FullName -Descending |
    Select-Object -First 1
$UvRoot = Join-Path $SourcePortable "bootstrap\windows\uv"
$IndexSource = Join-Path $SourcePortable "runtimes\index-tts"
$WinRAR = Join-Path $env:ProgramFiles "WinRAR\WinRAR.exe"
if (-not $BasePython) { throw "缺少本地 Windows Python 3.12。" }
if (-not (Test-Path (Join-Path $UvRoot "uv.exe"))) { throw "缺少本地 uv。" }
if (-not (Test-Path $WinRAR)) { throw "找不到 WinRAR：$WinRAR" }
if (($Requested.recommended -or $Requested.advanced) -and `
    -not (Test-Path (Join-Path $IndexSource "pyproject.toml"))) {
    throw "缺少已安装的固定 IndexTTS2 源码。"
}

$RequiredIds = @("recommended_dependencies")
if ($Requested.recommended -or $Requested.advanced) {
    $RequiredIds += @("parakeet", "indextts2")
}
if ($Requested.advanced) {
    $RequiredIds += @(
        "advanced_dependencies", "kotoba", "faster_whisper", "qwen_aligner", "asmr_vad"
    )
}
foreach ($Id in $RequiredIds) {
    Test-Artifact -Id $Id -Spec $Artifacts[$Id]
}

Write-Host "正在生成 Windows 启动器..." -ForegroundColor Cyan
& (Join-Path $Root "launcher\windows\build.ps1")
$LauncherBuildSucceeded = $?
if (-not $LauncherBuildSucceeded) { throw "Windows 启动器生成失败。" }

Remove-Item -LiteralPath $StagingRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $PackageRoot, $PackagePortable, $OutputRoot |
    Out-Null
Copy-SourceTree
Initialize-PackageBase
Repair-PackagePaths
Write-PackageDescription -Profile "核心" -Contents @"
- 主程序、网页界面、便携 Python 和 FFmpeg；
- 翻译服务与外部 TTS（语音合成）API 客户端；
- 不包含本地 ASR（语音识别）模型和 IndexTTS2 checkpoints。
"@
Test-PackageProfile -Profile "core"
if ($Requested.core) {
    New-PortableArchive -ProfileId "core" -ArchiveLabel "Core"
}

if ($Requested.recommended -or $Requested.advanced) {
    Initialize-RecommendedPackage
    Write-PackageDescription -Profile "推荐" -Contents @"
- 核心包的全部内容；
- Parakeet CTC 1.1B JA GAL；
- Parakeet TDT/CTC 0.6B JA；
- IndexTTS2 本地 TTS（语音合成）环境和 checkpoints（需要 NVIDIA GPU）。

首次用 IndexTTS2 生成语音时需要加载模型并初始化 CUDA，可能需要等待一段时间。
"@
    Test-PackageProfile -Profile "recommended"
    if ($Requested.recommended) {
        New-PortableArchive -ProfileId "recommended" -ArchiveLabel "Recommended"
    }
}

if ($Requested.advanced) {
    Initialize-AdvancedPackage
    Write-PackageDescription -Profile "进阶" -Contents @"
- 推荐包的全部内容；
- Kotoba-Whisper v2.2；
- Faster-Whisper large-v2；
- 日语 ASMR 专用 Whisper VAD ONNX；
- Qwen3 ForcedAligner 0.6B 时间戳对齐模型；
- 上述模型固定且完整的 Windows CUDA/Python 运行依赖。
"@
    Test-PackageProfile -Profile "advanced"
    New-PortableArchive -ProfileId "advanced" -ArchiveLabel "Advanced"
}

if (-not $KeepStaging) {
    Remove-Item -LiteralPath $StagingRoot -Recurse -Force
}
$env:TEMP = $OriginalTemp
$env:TMP = $OriginalTmp
$env:TMPDIR = $OriginalTmpDir
Write-Host "Windows 免安装包已经准备完成：$OutputRoot" -ForegroundColor Green
