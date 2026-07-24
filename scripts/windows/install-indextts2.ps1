[CmdletBinding()]
param(
    [string]$IndexUrl = "",
    [string]$HuggingFaceEndpoint = "",
    [string]$SourceUrl = (
        "https://github.com/index-tts/index-tts/archive/" +
        "13495845e3028f0bb6ca1462ad22aa0e76349e40.zip"
    ),
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
    throw "缺少项目私有 uv；请先运行项目根目录的 ASMR-Dubber-Setup.exe。"
}
New-Item -ItemType Directory -Force -Path $DataRoot, $DownloadRoot | Out-Null
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
    return Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory -NoNewWindow -Wait -PassThru
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
        Set-Content -Path $Marker -Value $Revision -Encoding utf8NoBOM
        $SourceReady = $true
    }
    $StagedSource = Get-ChildItem $Staging -Filter "pyproject.toml" -File -Recurse `
        -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Directory
    if (-not $SourceReady -and (Test-Path $RuntimeRoot) -and -not $StagedSource) {
        $Unexpected = Get-ChildItem $RuntimeRoot -Force | Where-Object { $_.Name -ne "checkpoints" }
        if ($Unexpected) {
            throw "IndexTTS2 目录已存在且包含未知文件：$RuntimeRoot"
        }
    }
}
if (-not $SourceReady) {
    $NeedDownload = $true
    if (Test-Path $Archive) {
        $NeedDownload = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne `
            $SourceSha256.ToLowerInvariant()
    }
    if ($NeedDownload) {
        Write-Host "正在下载固定版本 IndexTTS2 源码（约 32 MB）..." -ForegroundColor Cyan
        Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
            -Url $SourceUrl -Destination $Archive -Sha256 $SourceSha256 -Resume | Out-Null
    }
    $ActualHash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
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
    Set-Content -Path $Marker -Value $Revision -Encoding utf8NoBOM
}

Write-Host "正在安装 IndexTTS2 独立 Python/CUDA 环境..." -ForegroundColor Cyan
Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
    -Uv $Uv -Root $RuntimeRoot -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
    -Arguments @("sync")

$SharedFFmpeg = Install-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
Write-Host "共享版 FFmpeg：$SharedFFmpeg" -ForegroundColor DarkGray
$IndexCli = Join-Path $RuntimeRoot ".venv\Scripts\indextts2.exe"
if (-not (Test-Path $IndexCli)) {
    throw "IndexTTS2 CLI 安装后不存在：$IndexCli"
}

Write-Host "正在通过 ModelScope 下载/续传 IndexTTS2 官方模型（约 11 GB）..." `
    -ForegroundColor Cyan
$env:USE_MODELSCOPE = "true"
$Download = Invoke-Process -FilePath $IndexCli `
    -ArgumentList @("download", "--source", "modelscope", "--model-dir", $ModelDir) `
    -WorkingDirectory $RuntimeRoot
if ($Download.ExitCode -ne 0) {
    Write-Warning "ModelScope 不可用，改用 Hugging Face 镜像续传。"
    $env:USE_MODELSCOPE = "false"
    $Download = Invoke-Process -FilePath $IndexCli `
        -ArgumentList @("download", "--source", "auto", "--model-dir", $ModelDir) `
        -WorkingDirectory $RuntimeRoot
}
if ($Download.ExitCode -ne 0) {
    throw "IndexTTS2 模型下载失败（退出码 $($Download.ExitCode)）。"
}

$Device = if (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
Write-Host "正在检查 IndexTTS2 模型与运行环境..." -ForegroundColor Cyan
$Check = Invoke-Process -FilePath $IndexCli `
    -ArgumentList @("check", "--model-dir", $ModelDir, "--device", $Device) `
    -WorkingDirectory $RuntimeRoot
if ($Check.ExitCode -ne 0) {
    throw "IndexTTS2 检查失败（退出码 $($Check.ExitCode)）。"
}

Write-Host ""
Write-Host "IndexTTS2 安装完成。重启 ASMR Dubber 后在 TTS 设置中选择它。" -ForegroundColor Green
Write-Host "模型目录：$ModelDir"
