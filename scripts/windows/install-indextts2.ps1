[CmdletBinding()]
param(
    [string]$IndexUrl = "",
    [string]$HuggingFaceEndpoint = "",
    [string]$SourceUrl = "",
    [string]$SourceSha256 = "7ed8bc742e2eeeb83f922247ef0e27f96327f418acacb6c63f182cafd66887ba"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
. (Join-Path $Root "scripts\mirrors.ps1")
$MirrorConfiguration = Get-ASMRDubberMirrorConfiguration -Root $Root
. (Join-Path $Root "scripts\windows-runtime.ps1")
. (Join-Path $Root "scripts\windows\recommended-dependencies.ps1")
$DataRoot = $Paths.Home
$RuntimeRoot = Join-Path $DataRoot "runtimes\index-tts"
$ModelDir = Join-Path $RuntimeRoot "checkpoints"
$Uv = $Paths.Uv
$Revision = "13495845e3028f0bb6ca1462ad22aa0e76349e40"
$Marker = Join-Path $RuntimeRoot ".asmr-source-revision"
$DownloadRoot = Join-Path $DataRoot "cache\downloads"
$Archive = Join-Path $DownloadRoot "index-tts-$Revision.zip"
$Staging = "$RuntimeRoot.staging"

if (-not (Test-Path $Uv)) {
    throw "缺少 uv；请先运行项目根目录的 ASMR-Dubber-Setup.exe。"
}
New-Item -ItemType Directory -Force -Path $DataRoot, $DownloadRoot | Out-Null
if ($env:ASMR_DUBBER_MODEL_PACKS_PREPARED -ne "1") {
    & $Paths.Python -m asmr_dubber.cli prepare-model-pack indextts2-checkpoints
    if ($LASTEXITCODE -eq 0) {
        & $Paths.Python -m asmr_dubber.cli import-model-packs --all `
            --pack-id indextts2-checkpoints
    } else {
        Write-Warning "远程 IndexTTS2 模型包不可用，将继续使用原始下载源。"
    }
}
$env:UV_LINK_MODE = "copy"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
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

function Invoke-Process {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $Process = Start-ASMRDubberProcess -FilePath $FilePath `
        -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory
    $Process.WaitForExit()
    return $Process
}

$SourceFilesReady = (Test-Path (Join-Path $RuntimeRoot "pyproject.toml")) -and `
    (Test-Path (Join-Path $RuntimeRoot "uv.lock")) -and `
    (Test-Path (Join-Path $RuntimeRoot "indextts"))
$SourceReady = $SourceFilesReady -and (Test-Path $Marker) -and `
    ((Get-Content $Marker -Raw).Trim() -eq $Revision)
if (-not $SourceReady) {
    if ($SourceFilesReady) {
        # A previous run may have completed the source move just before writing
        # the marker. The immutable archive hash still identifies this install.
        [System.IO.File]::WriteAllText(
            $Marker,
            $Revision + "`r`n",
            (New-Object System.Text.UTF8Encoding($false))
        )
        $SourceReady = $true
    }
    $StagedSource = Get-ChildItem $Staging -Filter "pyproject.toml" -File -Recurse `
        -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Directory
    if (-not $SourceReady -and (Test-Path $RuntimeRoot) -and -not $StagedSource) {
        # Recommended dependency bundles install the isolated environment before
        # this script adds the pinned IndexTTS2 source tree. Both directories are
        # therefore valid bootstrap state after an interrupted or fresh install.
        $AllowedBootstrapDirectories = @("checkpoints", ".venv")
        $Unexpected = Get-ChildItem $RuntimeRoot -Force | Where-Object {
            -not ($_.PSIsContainer -and $_.Name -in $AllowedBootstrapDirectories)
        }
        if ($Unexpected) {
            throw "IndexTTS2 目录已存在且包含未知文件：$RuntimeRoot"
        }
    }
}
if (-not $SourceReady) {
    $NeedDownload = $true
    if (Test-Path $Archive) {
        $NeedDownload = (Get-ASMRDubberFileSha256 -Path $Archive) -ne `
            $SourceSha256.ToLowerInvariant()
    }
    if ($NeedDownload) {
        Write-Host "正在下载固定版本 IndexTTS2 源码（约 32 MB）..." -ForegroundColor Cyan
        $SourceReady = $false
        $SourceErrors = New-Object System.Collections.Generic.List[string]
        foreach ($Candidate in Get-ASMRDubberMirrorList `
            -Configuration $MirrorConfiguration -Name "indextts2_source_archives" `
            -Preferred $SourceUrl) {
            try {
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Candidate -Destination $Archive -Sha256 $SourceSha256 `
                    -Resume | Out-Null
                $SourceReady = $true
                break
            } catch {
                [void]$SourceErrors.Add("$Candidate：$($_.Exception.Message)")
                Write-Warning "当前 IndexTTS2 源码源失败，自动切换。"
            }
        }
        if (-not $SourceReady) {
            throw "IndexTTS2 源码下载失败：$($SourceErrors -join '；')"
        }
    }
    $ActualHash = Get-ASMRDubberFileSha256 -Path $Archive
    if ($ActualHash -ne $SourceSha256.ToLowerInvariant()) {
        throw "IndexTTS2 源码校验失败：$ActualHash"
    }
    if (-not $StagedSource -and (Test-Path $Staging)) {
        Remove-Item -Recurse -Force $Staging
    }
    if (-not $StagedSource) {
        New-Item -ItemType Directory -Force -Path $Staging | Out-Null
        Expand-Archive -Path $Archive -DestinationPath $Staging -Force
    }
    $SourceRoot = Get-ChildItem $Staging -Filter "pyproject.toml" -File -Recurse |
        Select-Object -First 1 -ExpandProperty Directory
    if (-not $SourceRoot) {
        throw "IndexTTS2 源码包缺少 pyproject.toml。"
    }
    New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
    foreach ($Child in Get-ChildItem $SourceRoot.FullName -Force) {
        $Destination = Join-Path $RuntimeRoot $Child.Name
        if ($Child.PSIsContainer -and (Test-Path $Destination)) {
            Get-ChildItem $Child.FullName -Force | Copy-Item `
                -Destination $Destination -Recurse -Force
        } else {
            Move-Item $Child.FullName -Destination $RuntimeRoot -Force
        }
    }
    Remove-Item -Recurse -Force $Staging
    [System.IO.File]::WriteAllText(
        $Marker,
        $Revision + "`r`n",
        (New-Object System.Text.UTF8Encoding($false))
    )
}

$IndexPython = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"
$IndexRuntimeReady = Test-ASMRDubberIndexRuntimeDependencies -PortableRoot $DataRoot
if ($IndexRuntimeReady) {
    $ImportCheck = Invoke-ASMRDubberProcess -FilePath $IndexPython `
        -ArgumentList @("-c", "import indextts.cli_v2") `
        -WorkingDirectory $RuntimeRoot
    $IndexRuntimeReady = $ImportCheck -eq 0
}
if ($IndexRuntimeReady) {
    Write-Host "IndexTTS2 Python/CUDA 依赖已由 Recommended 依赖包提供。" `
        -ForegroundColor Green
} else {
    Write-Host "正在安装 IndexTTS2 独立 Python/CUDA 环境..." -ForegroundColor Cyan
    Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
        -Uv $Uv -Root $RuntimeRoot -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
        -Arguments @("sync")
}

$SharedFFmpeg = Install-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
Write-Host "共享版 FFmpeg：$SharedFFmpeg" -ForegroundColor DarkGray
$IndexPython = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $IndexPython)) {
    throw "IndexTTS2 Python 环境安装后不存在：$IndexPython"
}

function Test-IndexTTSCheckpointsComplete {
    $ValidationScript = @'
import sys
from pathlib import Path

from asmr_dubber.constants import INDEXTTS_REQUIRED_DIRS, INDEXTTS_REQUIRED_FILES

model_dir = Path(sys.argv[1])
files_ready = all((model_dir / name).is_file() for name in INDEXTTS_REQUIRED_FILES)
dirs_ready = all((model_dir / name).is_dir() for name in INDEXTTS_REQUIRED_DIRS)
print("ready" if files_ready and dirs_ready else "missing")
'@
    $AppPython = [string]$Paths.Python
    $ValidationResult = & $AppPython -c $ValidationScript $ModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "无法读取 IndexTTS2 checkpoints 必需资源定义。"
    }
    return ([string]$ValidationResult).Trim() -eq "ready"
}

if (Test-IndexTTSCheckpointsComplete) {
    Write-Host "IndexTTS2 本地 checkpoints 已完整，无需联网下载。" -ForegroundColor Green
} else {
    Write-Host "正在通过 ModelScope 下载/续传 IndexTTS2 官方模型（约 11 GB）..." `
        -ForegroundColor Cyan
    $env:USE_MODELSCOPE = "true"
    $Download = Invoke-Process -FilePath $IndexPython `
        -ArgumentList @(
            "-m", "indextts.cli_v2", "download", "--source", "modelscope",
            "--model-dir", $ModelDir
        ) `
        -WorkingDirectory $RuntimeRoot
    if ($Download.ExitCode -ne 0) {
        Write-Warning "ModelScope 不可用，改用 Hugging Face 镜像续传。"
        $env:USE_MODELSCOPE = "false"
        $Download = Invoke-Process -FilePath $IndexPython `
            -ArgumentList @(
                "-m", "indextts.cli_v2", "download", "--source", "auto",
                "--model-dir", $ModelDir
            ) `
            -WorkingDirectory $RuntimeRoot
    }
    if ($Download.ExitCode -ne 0) {
        throw "IndexTTS2 模型下载失败（退出码 $($Download.ExitCode)）。"
    }
}

$Device = if (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
Write-Host "正在检查 IndexTTS2 模型与运行环境..." -ForegroundColor Cyan
$Check = Invoke-Process -FilePath $IndexPython `
    -ArgumentList @(
        "-m", "indextts.cli_v2", "check", "--model-dir", $ModelDir,
        "--device", $Device
    ) `
    -WorkingDirectory $RuntimeRoot
if ($Check.ExitCode -ne 0) {
    throw "IndexTTS2 检查失败（退出码 $($Check.ExitCode)）。"
}

Write-Host ""
Write-Host "IndexTTS2 安装完成。重启 ASMR Dubber 后在 TTS 设置中选择它。" -ForegroundColor Green
Write-Host "模型目录：$ModelDir"
