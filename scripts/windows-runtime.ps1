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
        [string]$AssetUrl = "",
        [string]$ChecksumsUrl = "",
        [string]$ExpectedSha256 = `
            "34db93b66a56125ec10547b12a7996e2dbca8eba6a1aa14b00b8a281bc87cd02"
    )

    if (-not (Get-Command Get-ASMRDubberMirrorConfiguration -ErrorAction SilentlyContinue)) {
        $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
        . (Join-Path $Root "scripts\mirrors.ps1")
    }
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $MirrorConfiguration = Get-ASMRDubberMirrorConfiguration -Root $Root
    $Existing = Enable-ASMRDubberSharedFFmpeg -DataRoot $DataRoot
    if ($Existing) {
        return $Existing
    }

    $AssetName = "ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
    if ($AssetUrl) {
        $AssetName = [System.IO.Path]::GetFileName(([Uri]$AssetUrl).AbsolutePath)
    }
    if (-not $AssetName.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase)) {
        throw "FFmpeg 下载地址不是 ZIP 文件：$AssetUrl"
    }
    $DownloadRoot = Join-Path $DataRoot "cache\downloads"
    $RuntimeRoot = Join-Path $DataRoot "runtimes\ffmpeg-shared"
    $StagingRoot = "$RuntimeRoot.staging"
    $Archive = Join-Path $DownloadRoot $AssetName
    $ChecksumsFile = Join-Path $DownloadRoot "btbn-checksums.sha256"
    New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null

    $ExpectedHash = $ExpectedSha256.ToLowerInvariant()
    if ($ExpectedHash -notmatch "^[0-9a-f]{64}$") {
        throw "FFmpeg 固定 SHA-256 无效。"
    }
    $ChecksumUrls = @(Get-ASMRDubberMirrorList -Configuration $MirrorConfiguration `
        -Name "ffmpeg_checksum_files_windows" -Preferred $ChecksumsUrl)
    foreach ($ChecksumCandidate in $ChecksumUrls) {
        try {
            Write-Host "正在核对共享版 FFmpeg 发布校验信息..." -ForegroundColor Cyan
            Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                -Url $ChecksumCandidate -Destination $ChecksumsFile | Out-Null
            $ChecksumLine = Get-Content $ChecksumsFile | Where-Object {
                $_ -match ("^[0-9a-fA-F]{64}\s+" + [Regex]::Escape($AssetName) + "$")
            } | Select-Object -First 1
            if (-not $ChecksumLine) { throw "校验文件中找不到 $AssetName。" }
            $PublishedHash = ($ChecksumLine -split "\s+", 2)[0].ToLowerInvariant()
            if ($PublishedHash -ne $ExpectedHash) {
                throw "发布校验值与程序固定值不一致。"
            }
            break
        } catch {
            Write-Warning "FFmpeg 校验文件不可用，将继续使用程序固定 SHA-256：$($_.Exception.Message)"
        }
    }

    $NeedsDownload = $true
    if (Test-Path $Archive) {
        $ExistingHash = Get-ASMRDubberFileSha256 -Path $Archive
        $NeedsDownload = $ExistingHash -ne $ExpectedHash
    }
    if ($NeedsDownload) {
        Write-Host "正在下载便携式 LGPL shared FFmpeg（约 70 MB）..." -ForegroundColor Cyan
        $ArchiveReady = $false
        $Errors = New-Object System.Collections.Generic.List[string]
        foreach ($Candidate in Get-ASMRDubberMirrorList `
            -Configuration $MirrorConfiguration -Name "ffmpeg_shared_archives_windows" `
            -Preferred $AssetUrl) {
            try {
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Candidate -Destination $Archive -Sha256 $ExpectedHash -Resume | Out-Null
                $ArchiveReady = $true
                break
            } catch {
                [void]$Errors.Add("$Candidate：$($_.Exception.Message)")
            }
        }
        if (-not $ArchiveReady) {
            throw "FFmpeg 下载失败。请上传 ModelScope 镜像文件后重试：$($Errors -join '；')"
        }
    }
    $ActualHash = Get-ASMRDubberFileSha256 -Path $Archive
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
