Set-StrictMode -Version 2.0

function Get-ASMRDubberNamedChecksum {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$FileName
    )

    $Pattern = "^([0-9a-fA-F]{64})\s+\*?" + [Regex]::Escape($FileName) + "$"
    foreach ($Line in Get-Content -LiteralPath $Path -ErrorAction Stop) {
        $Match = [Regex]::Match($Line, $Pattern)
        if ($Match.Success) { return $Match.Groups[1].Value.ToLowerInvariant() }
    }
    return ""
}

function Get-ASMRDubberWheelhouse {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PortableRoot,
        [Parameter(Mandatory = $true)][object]$MirrorConfiguration,
        [Parameter(Mandatory = $true)][string]$ArchiveName,
        [Parameter(Mandatory = $true)][string]$ArchiveMirrorName,
        [Parameter(Mandatory = $true)][string]$ChecksumMirrorName
    )

    if ($ArchiveName -notmatch "^[A-Za-z0-9._-]+\.zip$") {
        throw "wheelhouse 压缩包文件名无效：$ArchiveName"
    }
    $DownloadRoot = Join-Path $PortableRoot "cache\downloads"
    $ExtractRoot = Join-Path $PortableRoot `
        ("cache\wheelhouses\" + [System.IO.Path]::GetFileNameWithoutExtension($ArchiveName))
    $Archive = Join-Path $DownloadRoot $ArchiveName
    $Checksum = "$Archive.sha256"
    $LocalArchive = Join-Path (Join-Path $Root "model-packs") $ArchiveName
    $LocalChecksum = "$LocalArchive.sha256"
    New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null

    $ExpectedHash = ""
    if (Test-Path -LiteralPath $LocalChecksum -PathType Leaf) {
        $ExpectedHash = Get-ASMRDubberNamedChecksum `
            -Path $LocalChecksum -FileName $ArchiveName
    }
    if (-not $ExpectedHash) {
        foreach ($Url in Get-ASMRDubberMirrorList -Configuration $MirrorConfiguration `
            -Name $ChecksumMirrorName) {
            try {
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Url -Destination $Checksum | Out-Null
                $ExpectedHash = Get-ASMRDubberNamedChecksum `
                    -Path $Checksum -FileName $ArchiveName
                if (-not $ExpectedHash) { throw "SHA-256 文件格式无效。" }
                break
            } catch {
                Write-Warning "ModelScope wheelhouse 校验文件暂不可用：$($_.Exception.Message)"
            }
        }
    }
    if (-not $ExpectedHash) {
        return $null
    }

    if (Test-Path -LiteralPath $LocalArchive -PathType Leaf) {
        if ((Get-ASMRDubberFileSha256 -Path $LocalArchive) -ne $ExpectedHash) {
            throw "本地 wheelhouse 的 SHA-256 不匹配：$LocalArchive"
        }
        if ([System.IO.Path]::GetFullPath($LocalArchive) -ne `
            [System.IO.Path]::GetFullPath($Archive)) {
            Copy-Item -LiteralPath $LocalArchive -Destination $Archive -Force
        }
    }
    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf) -or `
        (Get-ASMRDubberFileSha256 -Path $Archive) -ne $ExpectedHash) {
        $Ready = $false
        $Failures = New-Object System.Collections.Generic.List[string]
        foreach ($Url in Get-ASMRDubberMirrorList -Configuration $MirrorConfiguration `
            -Name $ArchiveMirrorName) {
            try {
                Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                    -Url $Url -Destination $Archive -Sha256 $ExpectedHash -Resume | Out-Null
                $Ready = $true
                break
            } catch {
                [void]$Failures.Add("$Url：$($_.Exception.Message)")
            }
        }
        if (-not $Ready) {
            throw "wheelhouse 下载失败：$($Failures -join '；')"
        }
    }

    $Marker = Join-Path $ExtractRoot ".archive-sha256"
    if ((Test-Path -LiteralPath $Marker -PathType Leaf) -and `
        ((Get-Content -LiteralPath $Marker -Raw).Trim() -eq $ExpectedHash) -and `
        (Get-ChildItem -LiteralPath $ExtractRoot -Filter "*.whl" -File -Recurse |
            Select-Object -First 1)) {
        return $ExtractRoot
    }
    $Staging = "$ExtractRoot.staging"
    Remove-Item -LiteralPath $Staging -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Staging | Out-Null
    try {
        Expand-Archive -LiteralPath $Archive -DestinationPath $Staging -Force
        if (-not (Get-ChildItem -LiteralPath $Staging -Filter "*.whl" -File -Recurse |
            Select-Object -First 1)) {
            throw "wheelhouse 压缩包中没有 wheel 文件。"
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $Staging ".archive-sha256"),
            $ExpectedHash + "`r`n",
            (New-Object System.Text.UTF8Encoding($false))
        )
        Remove-Item -LiteralPath $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $Staging -Destination $ExtractRoot
    } catch {
        Remove-Item -LiteralPath $Staging -Recurse -Force -ErrorAction SilentlyContinue
        throw
    }
    return $ExtractRoot
}

function Invoke-ASMRDubberUvOfflineWheelhouse {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Uv,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Wheelhouse,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$FailureMessage = "离线 wheelhouse 安装失败"
    )

    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Uv `
        -ArgumentList ($Arguments + @("--offline", "--find-links", $Wheelhouse)) `
        -WorkingDirectory $Root
    if ($ExitCode -ne 0) {
        throw "$FailureMessage（退出码 $ExitCode）。wheelhouse 已保留，请补齐后重新生成。"
    }
}
