$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

function Get-ASMRDubberMirrorConfiguration {
    param([Parameter(Mandatory = $true)][string]$Root)

    $Path = Join-Path $Root "mirrors.json"
    if (-not (Test-Path $Path)) {
        throw "缺少镜像配置文件：$Path"
    }
    try {
        return Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "mirrors.json 格式错误：$($_.Exception.Message)"
    }
}

function Get-ASMRDubberMirrorList {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$Preferred = ""
    )

    $Values = New-Object System.Collections.Generic.List[string]
    if ($Preferred) {
        [void]$Values.Add($Preferred.TrimEnd("/"))
    }
    $Configured = @()
    $Property = $Configuration.PSObject.Properties[$Name]
    if ($Property) {
        $Configured = @($Property.Value)
    }
    $Fallbacks = switch ($Name) {
        "pypi_indexes" { @("https://pypi.org/simple") }
        "huggingface_endpoints" { @("https://huggingface.co") }
        "pytorch_indexes" { @("https://download.pytorch.org/whl/cu130") }
        "github_proxy_prefixes" { @("") }
        "uv_archives_windows" {
            @(
                "https://releases.astral.sh/github/uv/releases/download/" +
                "0.11.30/uv-x86_64-pc-windows-msvc.zip"
            )
        }
        "uv_installers_windows" { @("https://astral.sh/uv/0.11.30/install.ps1") }
        "uv_installers_linux" { @("https://astral.sh/uv/0.11.30/install.sh") }
        "python_install_mirrors" {
            @(
                "https://releases.astral.sh/github/python-build-standalone/releases/download",
                "https://github.com/astral-sh/python-build-standalone/releases/download"
            )
        }
        "indextts2_source_archives" {
            @(
                "https://github.com/index-tts/index-tts/archive/" +
                "13495845e3028f0bb6ca1462ad22aa0e76349e40.zip"
            )
        }
        default { @() }
    }
    foreach ($Value in @($Configured) + @($Fallbacks)) {
        if ($null -eq $Value) { continue }
        $Text = ([string]$Value).Trim()
        if ($Text -eq "" -and $Name -ne "github_proxy_prefixes") { continue }
        if ($Text -and -not $Text.StartsWith("https://")) {
            Write-Warning "忽略非 HTTPS 镜像：$Text"
            continue
        }
        if ($Name -ne "github_proxy_prefixes") {
            $Text = $Text.TrimEnd("/")
        }
        if (-not $Values.Contains($Text)) {
            [void]$Values.Add($Text)
        }
    }
    return $Values.ToArray()
}

function Get-ASMRDubberGitHubUrls {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Url
    )

    if (-not $Url.StartsWith("https://github.com/")) {
        return $Url
    }
    $Candidates = New-Object System.Collections.Generic.List[string]
    foreach ($Prefix in Get-ASMRDubberMirrorList `
        -Configuration $Configuration -Name "github_proxy_prefixes") {
        $Candidate = if ($Prefix) { $Prefix.TrimEnd("/") + "/" + $Url } else { $Url }
        if (-not $Candidates.Contains($Candidate)) {
            [void]$Candidates.Add($Candidate)
        }
    }
    if (-not $Candidates.Contains($Url)) {
        [void]$Candidates.Add($Url)
    }
    return $Candidates.ToArray()
}

function Invoke-ASMRDubberDownload {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$Sha256 = "",
        [switch]$Resume
    )

    $Candidates = Get-ASMRDubberGitHubUrls -Configuration $Configuration -Url $Url
    $Partial = "$Destination.partial"
    $PartialPath = [System.IO.Path]::GetFullPath($Partial)
    $Curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    if (-not $Curl) {
        throw "Windows 缺少 curl.exe，无法下载文件。"
    }
    $Errors = New-Object System.Collections.Generic.List[string]
    foreach ($Candidate in $Candidates) {
        Write-Host "尝试下载：$Candidate" -ForegroundColor Cyan
        $Arguments = @(
            "-L", "--fail", "--retry", "3", "--retry-all-errors",
            "--connect-timeout", "20", "--output", $PartialPath
        )
        if ($Resume -and (Test-Path $Partial)) {
            $Arguments += @("-C", "-")
        }
        $Arguments += $Candidate
        $ExitCode = Invoke-ASMRDubberProcess -FilePath $Curl.Source `
            -ArgumentList $Arguments -WorkingDirectory (Get-Location).Path
        if ($ExitCode -eq 33 -and $Resume) {
            Remove-Item -Force -ErrorAction SilentlyContinue $Partial
            $Arguments = @(
                "-L", "--fail", "--retry", "3", "--retry-all-errors",
                "--connect-timeout", "20", "--output", $PartialPath, $Candidate
            )
            $ExitCode = Invoke-ASMRDubberProcess -FilePath $Curl.Source `
                -ArgumentList $Arguments -WorkingDirectory (Get-Location).Path
        }
        if ($ExitCode -eq 0 -and (Test-Path $Partial)) {
            Move-Item -Force $Partial $Destination
            if (-not $Sha256 -or (
                (Get-FileHash $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq
                $Sha256.ToLowerInvariant()
            )) {
                return $Candidate
            }
            Remove-Item -Force $Destination
            [void]$Errors.Add("$Candidate（SHA256 校验失败）")
            Write-Warning "当前下载源返回的文件校验失败，自动切换。"
            continue
        }
        [void]$Errors.Add("$Candidate（退出码 $ExitCode）")
        Write-Warning "当前下载源失败，自动切换。"
    }
    throw "所有下载源均失败：$($Errors -join '；')"
}

function Get-ASMRDubberTextDownload {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string[]]$Urls
    )

    $Errors = New-Object System.Collections.Generic.List[string]
    foreach ($Url in $Urls) {
        foreach ($Candidate in Get-ASMRDubberGitHubUrls `
            -Configuration $Configuration -Url $Url) {
            try {
                Write-Host "尝试下载：$Candidate" -ForegroundColor Cyan
                return Invoke-RestMethod -Uri $Candidate -UseBasicParsing -TimeoutSec 30
            } catch {
                [void]$Errors.Add("$Candidate：$($_.Exception.Message)")
                Write-Warning "当前下载源失败，自动切换。"
            }
        }
    }
    throw "所有下载源均失败：$($Errors -join '；')"
}

function Invoke-ASMRDubberProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $StartInfo.FileName = $FilePath
        $StartInfo.WorkingDirectory = $WorkingDirectory
        $StartInfo.UseShellExecute = $false
        foreach ($Argument in $ArgumentList) {
            [void]$StartInfo.ArgumentList.Add($Argument)
        }
        $Process = [System.Diagnostics.Process]::Start($StartInfo)
        $Process.WaitForExit()
        return $Process.ExitCode
    }
    $Process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory -NoNewWindow -Wait -PassThru
    return $Process.ExitCode
}

function Invoke-ASMRDubberUvWithIndexFallback {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Uv,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][ValidateSet("pypi_indexes", "pytorch_indexes")]
        [string]$MirrorName,
        [string]$Preferred = "",
        [string]$IndexOption = "--default-index"
    )

    $Errors = New-Object System.Collections.Generic.List[string]
    foreach ($Index in Get-ASMRDubberMirrorList `
        -Configuration $Configuration -Name $MirrorName -Preferred $Preferred) {
        Write-Host "使用软件源：$Index" -ForegroundColor DarkGray
        $ExitCode = Invoke-ASMRDubberProcess -FilePath $Uv `
            -ArgumentList ($Arguments + @($IndexOption, $Index)) -WorkingDirectory $Root
        if ($ExitCode -eq 0) {
            return
        }
        [void]$Errors.Add("$Index（退出码 $ExitCode）")
        Write-Warning "当前软件源失败，自动切换。"
    }
    throw "所有软件源均失败：$($Errors -join '；')"
}
