Set-StrictMode -Version 2.0

function Install-ASMRDubberManagedPythonArchive {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object]$Paths,
        [Parameter(Mandatory = $true)][object]$MirrorConfiguration,
        [Parameter(Mandatory = $true)][string]$Version,
        [Parameter(Mandatory = $true)][string]$BuildDate,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [Parameter(Mandatory = $true)][string]$MirrorName,
        [string]$PreferredBaseUrl = ""
    )

    if ($Version -notmatch "^3\.(11|12)\.\d+$" -or $BuildDate -notmatch "^\d{8}$" -or `
        $Sha256 -notmatch "^[0-9a-fA-F]{64}$") {
        throw "Python 运行时固定元数据无效。"
    }
    $RuntimeName = "cpython-$Version-windows-x86_64-none"
    $RuntimeRoot = Join-Path $env:UV_PYTHON_INSTALL_DIR $RuntimeName
    $Python = Join-Path $RuntimeRoot "python.exe"
    if (Test-Path -LiteralPath $Python -PathType Leaf) {
        $Check = Invoke-ASMRDubberProcess -FilePath $Python `
            -ArgumentList @("-c", "import sys; print(sys.version)") -WorkingDirectory $Root
        if ($Check -eq 0) { return (Get-Item -LiteralPath $Python) }
    }

    $ArchiveName = "cpython-$Version+$BuildDate-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
    $Archive = Join-Path $Paths.Cache ("downloads\" + $ArchiveName)
    $PreferredUrl = ""
    if ($PreferredBaseUrl) {
        $EncodedName = [Uri]::EscapeDataString($ArchiveName)
        $PreferredUrl = $PreferredBaseUrl.TrimEnd("/") + "/$BuildDate/$EncodedName"
    }
    $Ready = $false
    $Failures = New-Object System.Collections.Generic.List[string]
    foreach ($Url in Get-ASMRDubberMirrorList -Configuration $MirrorConfiguration `
        -Name $MirrorName -Preferred $PreferredUrl) {
        try {
            Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                -Url $Url -Destination $Archive -Sha256 $Sha256 -Resume | Out-Null
            $Ready = $true
            break
        } catch {
            [void]$Failures.Add("$Url：$($_.Exception.Message)")
        }
    }
    if (-not $Ready) {
        throw (
            "Python $Version 的 ModelScope 运行时下载失败；断点文件已保留。" +
            "请按 docs/MODELSCOPE_UPLOADS.md 上传对应文件：$($Failures -join '；')"
        )
    }

    $Tar = Get-Command "tar.exe" -ErrorAction SilentlyContinue
    if (-not $Tar) { throw "Windows 缺少 tar.exe，无法解压 Python 运行时。" }
    # Keep this path deliberately short for legacy Windows path handling.
    $Staging = Join-Path $Paths.Home ("t\py-" + $Version.Replace(".", ""))
    Remove-Item -LiteralPath $Staging -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Staging | Out-Null
    $Extract = Invoke-ASMRDubberProcess -FilePath $Tar.Source `
        -ArgumentList @("-xzf", $Archive, "-C", $Staging) -WorkingDirectory $Root
    if ($Extract -ne 0) {
        throw "Python $Version 运行时解压失败（tar 退出码 $Extract）。"
    }
    $StagedPython = Get-ChildItem -LiteralPath $Staging -Filter "python.exe" -File -Recurse |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.Directory.FullName "Lib") } |
        Select-Object -First 1
    if (-not $StagedPython) { throw "Python 压缩包结构无效：找不到 python.exe/Lib。" }

    Remove-Item -LiteralPath $RuntimeRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RuntimeRoot) | Out-Null
    Move-Item -LiteralPath $StagedPython.Directory.FullName -Destination $RuntimeRoot
    Remove-Item -LiteralPath $Staging -Recurse -Force -ErrorAction SilentlyContinue
    $VersionTuple = ($Version -split "\.") -join ", "
    $Check = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList @("-c", "import sys; assert sys.version_info[:3] == ($VersionTuple)") `
        -WorkingDirectory $Root
    if ($Check -ne 0) {
        throw "Python $Version 运行时安装后自检失败。"
    }
    return (Get-Item -LiteralPath $Python)
}
