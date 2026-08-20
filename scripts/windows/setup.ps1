[CmdletBinding()]
param(
    [ValidateSet("基础", "推荐", "进阶", "Core", "Recommended", "Advanced")]
    [string]$Profile = "推荐",
    [string]$IndexUrl = "",
    [string]$PythonMirror = "",
    [string]$HuggingFaceEndpoint = "",
    [string]$TorchIndexUrl = "",
    [string]$LocalCacheRoot = "",
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
if ($LocalCacheRoot) {
    if (-not (Test-Path -LiteralPath $LocalCacheRoot -PathType Container)) {
        throw "只读本地缓存目录不存在：$LocalCacheRoot"
    }
    $ResolvedLocalCacheRoot = (Resolve-Path -LiteralPath $LocalCacheRoot).Path
    $ExistingLocalRoots = @(
        $env:ASMR_DUBBER_LOCAL_CACHE_ROOTS -split ";" |
            Where-Object { $_ -and $_.Trim() }
    )
    $env:ASMR_DUBBER_LOCAL_CACHE_ROOTS = `
        (@($ResolvedLocalCacheRoot) + $ExistingLocalRoots | Select-Object -Unique) -join ";"
}
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
$DataRoot = $Paths.Home
$Bootstrap = $Paths.Bootstrap
$UvDir = $Paths.UvDir
$Uv = $Paths.Uv
$Venv = $Paths.Venv
$Python = $Paths.Python
. (Join-Path $Root "scripts\windows-runtime.ps1")
. (Join-Path $Root "scripts\windows\recommended-dependencies.ps1")
. (Join-Path $Root "scripts\windows\wheelhouse.ps1")
. (Join-Path $Root "scripts\windows\python-runtime.ps1")

$Profile = switch ($Profile) {
    "Core" { "基础" }
    "Recommended" { "推荐" }
    "Advanced" { "进阶" }
    default { $Profile }
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

function Write-ASMRDubberNativeRuntimeReport {
    Write-Host "正在检查 Windows 原生运行库（仅报告，不会中止安装）..." `
        -ForegroundColor Cyan
    try {
        $SystemDirectory = [Environment]::GetFolderPath(
            [Environment+SpecialFolder]::System
        )
        $RequiredVisualCppDlls = @(
            "MSVCP140.dll",
            "VCOMP140.DLL",
            "VCRUNTIME140.dll",
            "VCRUNTIME140_1.dll"
        )
        $MissingVisualCppDlls = @(
            $RequiredVisualCppDlls | Where-Object {
                -not (Test-Path -LiteralPath (Join-Path $SystemDirectory $_) -PathType Leaf)
            }
        )
        if ($MissingVisualCppDlls.Count -eq 0) {
            Write-Host "Microsoft Visual C++ x64 运行库检查通过。" -ForegroundColor Green
        } else {
            Write-Warning (
                "Microsoft Visual C++ x64 运行库可能不完整，缺少：" +
                ($MissingVisualCppDlls -join "、") +
                "。Parakeet 或 Faster-Whisper 可能无法启动；Setup 仍会继续。" +
                "官方修复地址：https://aka.ms/vc14/vc_redist.x64.exe"
            )
        }

        $CrispBin = Join-Path $DataRoot "runtimes\crispasr\bin"
        $CrispExecutable = Join-Path $CrispBin "crispasr.exe"
        if (Test-Path -LiteralPath $CrispExecutable -PathType Leaf) {
            $RequiredCrispFiles = @(
                "crispasr.dll",
                "ggml.dll",
                "ggml-base.dll",
                "ggml-cpu.dll"
            )
            $MissingCrispFiles = @(
                $RequiredCrispFiles | Where-Object {
                    -not (Test-Path -LiteralPath (Join-Path $CrispBin $_) -PathType Leaf)
                }
            )
            if ($MissingCrispFiles.Count -gt 0) {
                Write-Warning (
                    "现有 CrispASR 原生运行时不完整，缺少：" +
                    ($MissingCrispFiles -join "、") + "。Setup 仍会继续。"
                )
            } else {
                try {
                    $CrispExitCode = Invoke-ASMRDubberProcess `
                        -FilePath $CrispExecutable -ArgumentList @("--version") `
                        -WorkingDirectory $Root
                    if ($CrispExitCode -eq 0) {
                        Write-Host "现有 CrispASR 原生运行时可以启动。" -ForegroundColor Green
                    } else {
                        Write-Warning (
                            "现有 CrispASR 原生运行时启动检查退出码：$CrispExitCode。" +
                            "Setup 仍会继续。"
                        )
                    }
                } catch {
                    Write-Warning (
                        "现有 CrispASR 原生运行时启动检查失败：" +
                        "$($_.Exception.Message)。Setup 仍会继续。"
                    )
                }
            }
        }
    } catch {
        Write-Warning (
            "Windows 原生运行库检查未完成：$($_.Exception.Message)。" +
            "这只是诊断信息，Setup 将继续。"
        )
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
$HuggingFaceEndpoints = @(Set-ASMRDubberHuggingFaceEnvironment `
    -Configuration $MirrorConfiguration -Preferred $HuggingFaceEndpoint)
$PreferredHuggingFace = [string]($HuggingFaceEndpoints | Select-Object -First 1)

Write-Host "ASMR Dubber · Windows 安装" -ForegroundColor Cyan
Write-Host "项目目录：$Root"
Write-Host "数据目录：$DataRoot"
Write-Host "安装配置：$Profile"
if (Test-ASMRDubberExternalDownloadsAllowed -Configuration $MirrorConfiguration) {
    Write-Host "下载策略：ModelScope 优先；已显式允许海外备用源。" -ForegroundColor Yellow
} else {
    Write-Host "下载策略：ModelScope 优先；GitHub/Hugging Face/官方海外源已关闭。" `
        -ForegroundColor Green
}
if ($env:ASMR_DUBBER_LOCAL_CACHE_ROOTS) {
    Write-Host "只读本地缓存：$($env:ASMR_DUBBER_LOCAL_CACHE_ROOTS)" -ForegroundColor DarkGray
}
$StorageEstimate = switch ($Profile) {
    "基础" {
        [pscustomobject]@{ Installed = "约 2 GB"; Free = "至少 5 GB" }
    }
    "推荐" {
        [pscustomobject]@{ Installed = "约 24–28 GB"; Free = "至少 35 GB" }
    }
    "进阶" {
        [pscustomobject]@{ Installed = "约 33–39 GB"; Free = "至少 50 GB" }
    }
}
Write-Host "预计安装后占用：$($StorageEstimate.Installed)"
Write-Host "建议安装前可用空间：$($StorageEstimate.Free)"
if ($Profile -eq "进阶") {
    Write-Host "进阶档位会安装以下固定模型：" -ForegroundColor Cyan
    Write-Host "  ASR（语音识别）Parakeet CTC 1.1B JA GAL"
    Write-Host "  ASR（语音识别）Parakeet TDT/CTC 0.6B JA"
    Write-Host "  ASR（语音识别）Kotoba-Whisper v2.2（kotoba-tech/kotoba-whisper-v2.2）"
    Write-Host "  ASR（语音识别）Faster-Whisper large-v2（Systran/faster-whisper-large-v2）"
    Write-Host "  VAD（语音活动检测）日语 ASMR 专用 Whisper VAD ONNX"
    Write-Host "  时间戳对齐：Qwen3 ForcedAligner 0.6B（阿里 Qwen）"
    Write-Host "  TTS（语音合成）IndexTTS2 checkpoints（仅 NVIDIA GPU）"
    Write-Host "不会自动安装 Kotoba v2.0/v2.1、Faster-Whisper large-v3 或其它识别模型。" `
        -ForegroundColor DarkGray
}
if ($Profile -ne "基础") {
    Write-Host "未检测到 NVIDIA GPU 时会跳过需要 CUDA 的 TTS（语音合成），实际占用将减少。" `
        -ForegroundColor DarkGray
}

Write-ASMRDubberNativeRuntimeReport

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
Write-Host "uv 已就绪。" -ForegroundColor Green

Write-Host "正在准备 Python 3.12..." -ForegroundColor Cyan
$ManagedPython = Get-ChildItem `
    (Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.12.*-windows-x86_64-none\python.exe") `
    -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $ManagedPython) {
    $ManagedPython = Install-ASMRDubberManagedPythonArchive `
        -Root $Root -Paths $Paths -MirrorConfiguration $MirrorConfiguration `
        -Version "3.12.13" -BuildDate "20260718" `
        -Sha256 "0d422a1439ec308e03f47df551bc30f5994727c456e414b026d202bcda9b7c1c" `
        -MirrorName "python312_windows_archives" -PreferredBaseUrl $PythonMirror
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
$InstallAdvancedModels = $false
$InstallRecommendedTTS = $false
$InstallParakeet = $false
$Extra = switch ($Profile) {
    "基础" { ".[ui]" }
    "推荐" {
        $InstallParakeet = $true
        if ($NvidiaSmi) {
            $InstallRecommendedTTS = -not $SkipRecommendedTTS
        }
        ".[ui]"
    }
    "进阶" {
        $InstallParakeet = $true
        $InstallAdvancedModels = $true
        if ($NvidiaSmi) {
            $InstallRecommendedTTS = -not $SkipRecommendedTTS
        }
        ".[ui,asr-faster-whisper,asr-kotoba-whisper,asr-forced-aligner,asr-asmr-vad]"
    }
}

$RecommendedDependenciesReady = $false
if ($Profile -ne "基础" -and $NvidiaSmi) {
    $RecommendedDependenciesReady = Import-ASMRDubberRecommendedDependencies `
        -Root $Root -PortableRoot $DataRoot -Python $ManagedPython.FullName `
        -MirrorConfiguration $MirrorConfiguration
    # The imported virtual environments were created in a different folder.
    # Re-read the executable path after portable path repair.
    $Python = $Paths.Python
}

$AdvancedDependenciesReady = $false
if ($InstallAdvancedModels -and $NvidiaSmi) {
    $AdvancedDependenciesReady = Import-ASMRDubberAdvancedDependencies `
        -Root $Root -PortableRoot $DataRoot -Python $ManagedPython.FullName `
        -MirrorConfiguration $MirrorConfiguration
    $Python = $Paths.Python
}

if ($InstallAdvancedModels -and $NvidiaSmi -and -not $AdvancedDependenciesReady) {
    Write-Host "检测到 NVIDIA GPU，正在安装官方 CUDA 13.0 PyTorch..." -ForegroundColor Cyan
    $CudaWheelhouse = Get-ASMRDubberWheelhouse `
        -Root $Root -PortableRoot $DataRoot -MirrorConfiguration $MirrorConfiguration `
        -ArchiveName "ASMR-Dubber-Windows-CUDA130-Wheelhouse-v0.4.0.zip" `
        -ArchiveMirrorName "windows_cuda_wheelhouse_archives" `
        -ChecksumMirrorName "windows_cuda_wheelhouse_checksums"
    $CudaArguments = @(
        "pip", "install", "--python", $Python, "--reinstall",
        "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"
    )
    if ($CudaWheelhouse) {
        Write-Host "使用 ModelScope CUDA wheelhouse。" -ForegroundColor Green
        Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
            -Wheelhouse $CudaWheelhouse -Arguments $CudaArguments `
            -FailureMessage "CUDA PyTorch wheelhouse 安装失败"
    } else {
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $Root -MirrorName "pytorch_indexes" -Preferred $TorchIndexUrl `
            -IndexOption "--index" -Arguments $CudaArguments
    }
}

$BundledCoreWheelhouse = Join-Path $Root "vendor\windows-core-wheelhouse"
$BundledCoreWheelsReady = (
    (Test-Path -LiteralPath $BundledCoreWheelhouse -PathType Container) -and
    [bool](Get-ChildItem -LiteralPath $BundledCoreWheelhouse `
        -Filter "edge_tts-*.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1) -and
    [bool](Get-ChildItem -LiteralPath $BundledCoreWheelhouse `
        -Filter "gradio-*.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1) -and
    [bool](Get-ChildItem -LiteralPath $BundledCoreWheelhouse `
        -Filter "hatchling-*.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1)
)

$ApplicationDependenciesReady = Test-ASMRDubberApplicationRuntime -PortableRoot $DataRoot
if (-not $ApplicationDependenciesReady) {
    Write-Host "正在安装应用依赖：$Extra" -ForegroundColor Cyan
    $ApplicationArguments = @(
        "pip", "install", "--python", $Python, "--editable", $Extra,
        "setuptools>=78.1.1,<82"
    )
    $ApplicationInstalled = $false
    if ($BundledCoreWheelsReady) {
        Write-Host "使用便携包内置的基础应用 wheelhouse。" -ForegroundColor Green
        try {
            Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
                -Wheelhouse $BundledCoreWheelhouse -Arguments $ApplicationArguments `
                -FailureMessage "内置基础应用 wheelhouse 安装失败"
            $ApplicationInstalled = $true
        } catch {
            Write-Warning "内置基础应用 wheelhouse 不完整，将继续尝试 ModelScope：$($_.Exception.Message)"
        }
    }
    if (-not $ApplicationInstalled) {
        $ApplicationWheelhouse = Get-ASMRDubberWheelhouse `
            -Root $Root -PortableRoot $DataRoot -MirrorConfiguration $MirrorConfiguration `
            -ArchiveName "ASMR-Dubber-Windows-Wheelhouse-v0.4.0.zip" `
            -ArchiveMirrorName "windows_application_wheelhouse_archives" `
            -ChecksumMirrorName "windows_application_wheelhouse_checksums"
    } else {
        $ApplicationWheelhouse = $null
    }
    if (-not $ApplicationInstalled -and $ApplicationWheelhouse) {
        Write-Host "使用 ModelScope 应用依赖 wheelhouse。" -ForegroundColor Green
        try {
            Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
                -Wheelhouse $ApplicationWheelhouse -Arguments $ApplicationArguments `
                -FailureMessage "应用依赖 wheelhouse 安装失败"
            $ApplicationInstalled = $true
        } catch {
            Write-Warning (
                "ModelScope 应用 wheelhouse 早于当前依赖定义，将使用配置中的" +
                "国内软件源补齐应用依赖：$($_.Exception.Message)"
            )
        }
    }
    if (-not $ApplicationInstalled) {
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
    }
} else {
    Write-Host "应用依赖已由 Windows 依赖包提供。" -ForegroundColor Green
}

$ApiClientsReady = Test-ASMRDubberApiClientRuntime -PortableRoot $DataRoot
if (-not $ApiClientsReady) {
    Write-Host "正在安装在线翻译与 TTS（语音合成）API 客户端..." -ForegroundColor Cyan
    $ApiClientArguments = @(
        # Keep both direct imports explicit.  edge-tts does not own the
        # project's HTTP client requirement, so installing only edge-tts can
        # leave a fresh basic profile with a misleading "Checked 1 package"
        # result and a failed post-install check.
        "pip", "install", "--python", $Python,
        "edge-tts==7.2.8", "httpx>=0.28.0"
    )
    $InstalledBundledApiClient = $false
    if ($BundledCoreWheelsReady) {
        Write-Host "使用便携包内置的在线/API 客户端 wheelhouse。" -ForegroundColor Green
        try {
            Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
                -Wheelhouse $BundledCoreWheelhouse -Arguments $ApiClientArguments `
                -FailureMessage "内置在线/API 客户端 wheelhouse 安装失败"
            $InstalledBundledApiClient = $true
        } catch {
            Write-Warning "内置在线/API 客户端不可用，将使用国内软件源：$($_.Exception.Message)"
        }
    }
    if (-not $InstalledBundledApiClient) {
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
            -Arguments $ApiClientArguments
    }
    if (-not (Test-ASMRDubberApiClientRuntime -PortableRoot $DataRoot)) {
        throw "在线/API 客户端安装后仍不完整。"
    }
}
if (-not (Test-ASMRDubberCoreRuntime -PortableRoot $DataRoot)) {
    throw "基础应用或在线/API 客户端安装后仍不完整。"
}
Invoke-Checked -FilePath $Python `
    -ArgumentList @("-m", "compileall", "-q", "-f", (Join-Path $Root "src\asmr_dubber")) `
    -FailureMessage "应用字节码刷新失败"

$LocalPackIds = switch ($Profile) {
    "基础" { @() }
    "推荐" { @("parakeet-ja-windows", "indextts2-checkpoints") }
    "进阶" {
        @(
            "parakeet-ja-windows",
            "indextts2-checkpoints",
            "kotoba-whisper-v2.2",
            "faster-whisper-large-v2",
            "qwen3-forced-aligner",
            "whisper-vad-asmr-onnx"
        )
    }
}
if ($Profile -ne "基础") {
    Write-Host "正在检测并导入当前档位的本地模型包..." -ForegroundColor Cyan
    $ImportArguments = @("-m", "asmr_dubber.cli", "import-model-packs", "--all")
    foreach ($PackId in @($LocalPackIds)) {
        $ImportArguments += @("--pack-id", $PackId)
    }
    Invoke-Checked -FilePath $Python `
        -ArgumentList $ImportArguments `
        -FailureMessage "本地模型包扫描或导入失败；请检查 model-packs 目录中的压缩包"
}

if ($InstallAdvancedModels) {
    Write-Host (
        "正在准备进阶分析模型：Kotoba-Whisper v2.2、Faster-Whisper large-v2、" +
        "Qwen3 ForcedAligner 与日语 ASMR 专用 VAD..."
    ) -ForegroundColor Cyan
    Invoke-Checked -FilePath $Python `
        -ArgumentList @(
            "-m", "asmr_dubber.cli", "download-models", "--backend", "进阶语音识别"
        ) `
        -FailureMessage "进阶识别、VAD 与时间戳模型下载或校验失败"
}

if ($InstallParakeet) {
    Write-Host "正在安装推荐 ASR（语音识别）：Parakeet 日语..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "install-parakeet.ps1") -Variant Auto
}

if ($InstallRecommendedTTS) {
    Write-Host "正在安装推荐 TTS（语音合成）：IndexTTS2（约需 20 GB）..." -ForegroundColor Cyan
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
