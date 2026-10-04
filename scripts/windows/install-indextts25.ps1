[CmdletBinding()]
param(
    [string]$IndexUrl = "",
    [string]$HuggingFaceEndpoint = "",
    [string]$SourceUrl = "",
    [string]$SourceSha256 = "53ad18d03cae44d8daf29d665c889e4dfeb5e14f6a2bafb6b6a76eacfbf75440"
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
. (Join-Path $Root "scripts\windows\wheelhouse.ps1")
. (Join-Path $Root "scripts\windows\python-runtime.ps1")
$DataRoot = $Paths.Home
$RuntimeRoot = Join-Path $DataRoot "runtimes\index-tts-2.5"
$ModelDir = Join-Path $RuntimeRoot "checkpoints"
$Uv = $Paths.Uv
$Revision = "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"
$Marker = Join-Path $RuntimeRoot ".asmr-source-revision"
$DownloadRoot = Join-Path $DataRoot "cache\downloads"
$Archive = Join-Path $DownloadRoot "index-tts-$Revision.zip"
$Staging = "$RuntimeRoot.staging"

if ($env:ASMR_DUBBER_REPAIR_CHILD -ne "1") {
    $RepairShell = (Get-Process -Id $PID).Path
    $RepairCommand = @($RepairShell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    foreach ($RepairKey in $PSBoundParameters.Keys) {
        $RepairCommand += @("-$RepairKey", [string]$PSBoundParameters[$RepairKey])
    }
    & $Paths.Python -m asmr_dubber.installer_transaction --runtime $RuntimeRoot -- @RepairCommand
    if ($LASTEXITCODE -ne 0) { throw "IndexTTS-2.5 安装失败；原运行环境已恢复，下载断点保留。" }
    return
}

if (-not (Test-Path $Uv)) {
    throw "缺少 uv；请运行项目根目录的 ASMR-Dubber.exe 准备基础环境。"
}
New-Item -ItemType Directory -Force -Path $DataRoot, $DownloadRoot | Out-Null
if ($env:ASMR_DUBBER_MODEL_PACKS_PREPARED -ne "1") {
    & $Paths.Python -m asmr_dubber.cli prepare-model-pack indextts2_5-checkpoints
    if ($LASTEXITCODE -eq 0) {
        & $Paths.Python -m asmr_dubber.cli import-model-packs --all `
            --pack-id indextts2_5-checkpoints
        if ($LASTEXITCODE -ne 0) {
            throw "IndexTTS-2.5 ModelScope 模型包已下载，但导入失败。"
        }
    } else {
        Write-Warning (
            "IndexTTS-2.5 镜像模型包尚不可用或下载未完成，断点文件已保留。" +
            "请重试固定版本模型包；不会回退到未校验的浮动模型版本。"
        )
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
$HuggingFaceEndpoints = @(Set-ASMRDubberHuggingFaceEnvironment `
    -Configuration $MirrorConfiguration -Preferred $HuggingFaceEndpoint)

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
if (-not $SourceReady -and $SourceFilesReady -and (Test-Path (Join-Path $RuntimeRoot ".git"))) {
    try {
        $GitRevision = (& git -C $RuntimeRoot rev-parse HEAD 2>$null).Trim()
        if ($LASTEXITCODE -eq 0 -and $GitRevision -eq $Revision) {
        [System.IO.File]::WriteAllText(
            $Marker,
            $Revision + "`r`n",
            (New-Object System.Text.UTF8Encoding($false))
        )
        $SourceReady = $true
        }
    } catch {
        $SourceReady = $false
    }
}
if (-not $SourceReady -and (Test-Path $RuntimeRoot)) {
    # Keep downloaded models, the isolated environment and its private state,
    # but replace every unverified source file with the pinned source tree.
    Get-ChildItem $RuntimeRoot -Force | Where-Object {
        $_.Name -ne ".repair-state.json" -and -not ($_.PSIsContainer -and $_.Name -in @("checkpoints", ".venv", "user-state"))
    } | Remove-Item -Recurse -Force
}
$StagedSource = Get-ChildItem $Staging -Filter "pyproject.toml" -File -Recurse `
    -ErrorAction SilentlyContinue | Where-Object {
        Test-Path -LiteralPath (Join-Path $_.Directory.FullName "indextts") -PathType Container
    } | Select-Object -First 1 -ExpandProperty Directory
if (-not $SourceReady) {
    if ($StagedSource -and -not (Test-Path (Join-Path $StagedSource.FullName "indextts"))) {
        Remove-Item -Recurse -Force $Staging
        $StagedSource = $null
    }
}
if (-not $SourceReady) {
    $NeedDownload = $true
    if (Test-Path $Archive) {
        $NeedDownload = (Get-ASMRDubberFileSha256 -Path $Archive) -ne `
            $SourceSha256.ToLowerInvariant()
    }
    if ($NeedDownload) {
        Write-Host "正在下载固定版本 IndexTTS-2.5 源码（约 32 MB）..." -ForegroundColor Cyan
        $SourceReady = $false
        $SourceErrors = New-Object System.Collections.Generic.List[string]
        foreach ($Candidate in Get-ASMRDubberMirrorList `
            -Configuration $MirrorConfiguration -Name "indextts25_source_archives" `
            -Preferred $SourceUrl) {
            try {
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Candidate -Destination $Archive -Sha256 $SourceSha256 `
                    -Resume | Out-Null
                $SourceReady = $true
                break
            } catch {
                [void]$SourceErrors.Add("$Candidate：$($_.Exception.Message)")
                Write-Warning "当前 IndexTTS-2.5 源码源失败，自动切换。"
            }
        }
        if (-not $SourceReady) {
            $OfficialSource = "https://github.com/index-tts/index-tts/archive/$Revision.zip"
            try {
                Write-Warning "ModelScope 源码镜像不可用，改用固定的官方 GitHub 源码。"
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $OfficialSource -Destination $Archive -Sha256 $SourceSha256 `
                    -Resume | Out-Null
                $SourceReady = $true
            } catch {
                [void]$SourceErrors.Add("$OfficialSource：$($_.Exception.Message)")
            }
        }
        if (-not $SourceReady) {
            throw "IndexTTS-2.5 源码下载失败：$($SourceErrors -join '；')"
        }
    }
    $ActualHash = Get-ASMRDubberFileSha256 -Path $Archive
    if ($ActualHash -ne $SourceSha256.ToLowerInvariant()) {
        throw "IndexTTS-2.5 源码校验失败：$ActualHash"
    }
    # Never trust files left by an interrupted extraction merely because a
    # pyproject.toml exists. Use a fresh directory from the verified archive.
    $Staging = "$RuntimeRoot.staging-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $Staging | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Staging -Force
    $SourceRoot = Get-ChildItem $Staging -Filter "pyproject.toml" -File -Recurse |
        Where-Object {
            Test-Path -LiteralPath (Join-Path $_.Directory.FullName "indextts") `
                -PathType Container
        } | Select-Object -First 1 -ExpandProperty Directory
    if (-not $SourceRoot) {
        throw "IndexTTS-2.5 源码包缺少 pyproject.toml。"
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
$IndexRuntimeReady = Test-Path $IndexPython
if ($IndexRuntimeReady) {
    $ImportCheck = Invoke-ASMRDubberProcess -FilePath $IndexPython `
        -ArgumentList @("-c", "from indextts.infer_v2_5 import IndexTTS2") `
        -WorkingDirectory $RuntimeRoot
    $IndexRuntimeReady = $ImportCheck -eq 0
}
if ($IndexRuntimeReady) {
    Write-Host "IndexTTS-2.5 Python/CUDA 依赖已就绪。" -ForegroundColor Green
} else {
    Write-Host "正在安装 IndexTTS-2.5 独立环境..." -ForegroundColor Cyan
    $ManagedPython = Install-ASMRDubberManagedPythonArchive `
        -Root $Root -Paths $Paths -MirrorConfiguration $MirrorConfiguration `
        -Version "3.11.13" -BuildDate "20251007" `
        -Sha256 "cde5153f59a67d9e108f2ed964526e9aed100eba180f54bee0496b4fd73a8b29" `
        -MirrorName "python311_windows_archives"
    $IndexWheelhouse = Get-ASMRDubberWheelhouse `
        -Root $Root -PortableRoot $DataRoot -MirrorConfiguration $MirrorConfiguration `
        -ArchiveName "ASMR-Dubber-IndexTTS25-Wheelhouse-v1.0.0.zip" `
        -ArchiveMirrorName "indextts25_wheelhouse_archives_windows" `
        -ChecksumMirrorName "indextts25_wheelhouse_checksums_windows"
    if ($IndexWheelhouse) {
        Write-Host "使用 ModelScope IndexTTS-2.5 wheelhouse。" -ForegroundColor Green
        $Requirements = Join-Path $IndexWheelhouse "requirements.txt"
        if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
            throw "IndexTTS-2.5 wheelhouse 缺少 requirements.txt。"
        }
        $Venv = Join-Path $RuntimeRoot ".venv"
        $VenvPython = Join-Path $Venv "Scripts\python.exe"
        $CreateExitCode = Invoke-ASMRDubberProcess -FilePath $Uv `
            -ArgumentList @("venv", "--python", $ManagedPython.FullName, $Venv) `
            -WorkingDirectory $RuntimeRoot
        if ($CreateExitCode -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
            throw "IndexTTS-2.5 独立环境创建失败（退出码 $CreateExitCode）。"
        }
        $InstallExitCode = Invoke-ASMRDubberProcess -FilePath $Uv `
            -ArgumentList @(
                "pip", "install", "--python", $VenvPython, "--offline", "--no-index",
                "--find-links", $IndexWheelhouse, "--requirement", $Requirements
            ) `
            -WorkingDirectory $RuntimeRoot
        if ($InstallExitCode -ne 0) {
            throw "IndexTTS-2.5 wheelhouse 安装失败（退出码 $InstallExitCode）。"
        }
    } else {
        $SyncArguments = @(
            "sync", "--no-dev", "--extra", "torch_compile", "--python", $ManagedPython.FullName
        )
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $RuntimeRoot -MirrorName "pypi_indexes" `
            -Preferred $PreferredIndex -Arguments $SyncArguments
    }
}

$SharedFFmpeg = Install-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
Write-Host "共享版 FFmpeg：$SharedFFmpeg" -ForegroundColor DarkGray
$IndexPython = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $IndexPython)) {
    throw "IndexTTS-2.5 Python 环境安装后不存在：$IndexPython"
}

function Test-IndexTTSCheckpointsComplete {
    $ValidationScript = @'
import sys
from pathlib import Path

from asmr_dubber.constants import INDEXTTS25_REQUIRED_DIRS, INDEXTTS25_REQUIRED_FILES

model_dir = Path(sys.argv[1])
files_ready = all((model_dir / name).is_file() for name in INDEXTTS25_REQUIRED_FILES)
dirs_ready = all((model_dir / name).is_dir() for name in INDEXTTS25_REQUIRED_DIRS)
print("ready" if files_ready and dirs_ready else "missing")
'@
    $AppPython = [string]$Paths.Python
    # Windows PowerShell 5.1 strips the embedded double quotes when a multiline
    # script is passed as a native `python -c` argument. Feed the script over
    # stdin instead; `-` tells Python to read it while keeping ModelDir as argv[1].
    $ValidationResult = $ValidationScript | & $AppPython - $ModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "无法读取 IndexTTS-2.5 checkpoints 必需资源定义。"
    }
    return ([string]$ValidationResult).Trim() -eq "ready"
}

if (Test-IndexTTSCheckpointsComplete) {
    Write-Host "IndexTTS-2.5 本地 checkpoints 已完整，无需联网下载。" -ForegroundColor Green
} else {
    throw "IndexTTS-2.5 模型不完整。请重试固定 SHA-256 模型包下载/导入；断点已保留，不会回退到未固定版本的模型。"
}

$Device = if (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
Write-Host "正在检查 IndexTTS-2.5 模型与运行环境..." -ForegroundColor Cyan
$Check = Invoke-Process -FilePath $IndexPython `
    -ArgumentList @("-c", "import torch; from indextts.infer_v2_5 import IndexTTS2; assert r'$Device' == 'cpu' or torch.cuda.is_available()") `
    -WorkingDirectory $RuntimeRoot
if ($Check.ExitCode -ne 0) {
    throw "IndexTTS-2.5 检查失败（退出码 $($Check.ExitCode)）。"
}

Write-Host ""
Write-Host "IndexTTS-2.5 安装完成。重启后在 TTS（语音合成）设置中选择它。" -ForegroundColor Green
Write-Host "模型目录：$ModelDir"
