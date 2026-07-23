Set-StrictMode -Version 2.0

function Get-ASMRDubberSharedFFmpegBin {
    param([Parameter(Mandatory = $true)][string]$DataRoot)

    $RuntimeRoot = Join-Path $DataRoot "runtimes\ffmpeg-shared"
    if (-not (Test-Path $RuntimeRoot)) {
        return $null
    }
    $Ffmpeg = Get-ChildItem $RuntimeRoot -Filter "ffmpeg.exe" -File -Recurse `
        -ErrorAction SilentlyContinue | Where-Object {
            # TorchCodec 0.13 supports FFmpeg 4 through 8. FFmpeg master may
            # already expose the next ABI and must not be selected here.
            Get-ChildItem $_.Directory.FullName -Filter "avcodec-*.dll" -File |
                Where-Object { $_.BaseName -match "^avcodec-(58|59|60|61|62)$" }
        } | Select-Object -First 1
    if ($Ffmpeg) {
        return $Ffmpeg.Directory.FullName
    }
    return $null
}

function Enable-ASMRDubberSharedFFmpeg {
    param([Parameter(Mandatory = $true)][string]$DataRoot)

    $Bin = Get-ASMRDubberSharedFFmpegBin -DataRoot $DataRoot
    if (-not $Bin) {
        return $null
    }
    $PathEntries = $env:PATH -split ";"
    if ($Bin -notin $PathEntries) {
        $env:PATH = "$Bin;$env:PATH"
    }
    $env:ASMR_DUBBER_FFMPEG = Join-Path $Bin "ffmpeg.exe"
    return $Bin
}

function Install-ASMRDubberSharedFFmpeg {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$DataRoot,
        [string]$AssetUrl = (
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/" +
            "ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
        ),
        [string]$ChecksumsUrl = (
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/checksums.sha256"
        )
    )

    $Existing = Enable-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
    if ($Existing) {
        return $Existing
    }

    $AssetName = [System.IO.Path]::GetFileName(([Uri]$AssetUrl).AbsolutePath)
    if (-not $AssetName.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase)) {
        throw "FFmpeg 下载地址不是 ZIP 文件：$AssetUrl"
    }
    $DownloadRoot = Join-Path $DataRoot "cache\downloads"
    $RuntimeRoot = Join-Path $DataRoot "runtimes\ffmpeg-shared"
    $StagingRoot = "$RuntimeRoot.staging"
    $Archive = Join-Path $DownloadRoot $AssetName
    $ChecksumsFile = Join-Path $DownloadRoot "btbn-checksums.sha256"
    New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null

    Write-Host "正在获取共享版 FFmpeg 校验信息..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $ChecksumsUrl -OutFile $ChecksumsFile -UseBasicParsing
    $ChecksumLine = Get-Content $ChecksumsFile | Where-Object {
        $_ -match ("^[0-9a-fA-F]{64}\s+" + [Regex]::Escape($AssetName) + "$")
    } | Select-Object -First 1
    if (-not $ChecksumLine) {
        throw "FFmpeg 校验文件中找不到 $AssetName。"
    }
    $ExpectedHash = ($ChecksumLine -split "\s+", 2)[0].ToLowerInvariant()

    $NeedsDownload = $true
    if (Test-Path $Archive) {
        $ExistingHash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        $NeedsDownload = $ExistingHash -ne $ExpectedHash
    }
    if ($NeedsDownload) {
        Write-Host "正在下载便携式 LGPL shared FFmpeg（约 70 MB）..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $AssetUrl -OutFile "$Archive.partial" -UseBasicParsing
        Move-Item -Force "$Archive.partial" $Archive
    }
    $ActualHash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash) {
        throw "FFmpeg 文件校验失败。期望 $ExpectedHash，实际 $ActualHash。"
    }

    if (Test-Path $StagingRoot) {
        Remove-Item -Recurse -Force $StagingRoot
    }
    New-Item -ItemType Directory -Force -Path $StagingRoot | Out-Null
    Write-Host "正在解压共享版 FFmpeg..." -ForegroundColor Cyan
    Expand-Archive -Path $Archive -DestinationPath $StagingRoot -Force
    $StagedFfmpeg = Get-ChildItem $StagingRoot -Filter "ffmpeg.exe" -File -Recurse |
        Where-Object {
            Get-ChildItem $_.Directory.FullName -Filter "avcodec-*.dll" -File |
                Where-Object { $_.BaseName -match "^avcodec-(58|59|60|61|62)$" }
        } |
        Select-Object -First 1
    if (-not $StagedFfmpeg) {
        throw "共享版 FFmpeg 解压后缺少 ffmpeg.exe 或 avcodec DLL。"
    }
    if (Test-Path $RuntimeRoot) {
        Remove-Item -Recurse -Force $RuntimeRoot
    }
    Move-Item $StagingRoot $RuntimeRoot
    $Bin = Enable-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
    if (-not $Bin) {
        throw "共享版 FFmpeg 安装后仍无法定位。"
    }
    return $Bin
}
