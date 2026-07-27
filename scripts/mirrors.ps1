$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

function Test-ASMRDubberTruthy {
    param([AllowNull()][object]$Value)

    if ($null -eq $Value) { return $false }
    return ([string]$Value).Trim().ToLowerInvariant() -in @("1", "true", "yes", "on")
}

function Test-ASMRDubberExternalDownloadsAllowed {
    param([Parameter(Mandatory = $true)][object]$Configuration)

    if (Test-Path Env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS) {
        return (Test-ASMRDubberTruthy -Value $env:ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS)
    }
    $PolicyProperty = $Configuration.PSObject.Properties["download_policy"]
    if (-not $PolicyProperty) {
        # Legacy hand-written configurations retain their historical behaviour.
        return $true
    }
    $Policy = $PolicyProperty.Value
    if ($null -eq $Policy) { return $false }
    $AllowProperty = $Policy.PSObject.Properties["allow_external"]
    return [bool]($AllowProperty -and (Test-ASMRDubberTruthy -Value $AllowProperty.Value))
}

function Test-ASMRDubberModelScopeUrl {
    param([Parameter(Mandatory = $true)][string]$Url)

    try {
        $HostName = ([Uri]$Url).DnsSafeHost.ToLowerInvariant()
    } catch {
        return $false
    }
    return $HostName -eq "modelscope.cn" -or $HostName.EndsWith(".modelscope.cn") -or `
        $HostName -eq "modelscope.ai" -or $HostName.EndsWith(".modelscope.ai")
}

function Test-ASMRDubberExternalUrl {
    param([Parameter(Mandatory = $true)][string]$Url)

    try {
        $HostName = ([Uri]$Url).DnsSafeHost.ToLowerInvariant()
    } catch {
        return $false
    }
    foreach ($ExternalHost in @(
        "github.com", "raw.githubusercontent.com", "huggingface.co", "hf.co",
        "hf-mirror.com", "ghfast.top", "ghproxy.net", "download.pytorch.org",
        "pypi.org", "astral.sh", "releases.astral.sh", "python.org",
        "www.python.org"
    )) {
        if ($HostName -eq $ExternalHost -or $HostName.EndsWith(".$ExternalHost")) {
            return $true
        }
    }
    return $false
}

function Get-ASMRDubberModelScopeArtifacts {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $Result = New-Object System.Collections.Generic.List[string]
    $ArtifactsProperty = $Configuration.PSObject.Properties["modelscope_artifacts"]
    if (-not $ArtifactsProperty) { return $Result.ToArray() }
    $Artifacts = $ArtifactsProperty.Value
    $NamedProperty = $Artifacts.PSObject.Properties[$Name]
    if (-not $NamedProperty) { return $Result.ToArray() }

    $BaseUrl = ""
    $ModelScopeProperty = $Configuration.PSObject.Properties["modelscope"]
    if ($ModelScopeProperty) {
        $BaseProperty = $ModelScopeProperty.Value.PSObject.Properties["base_url"]
        if ($BaseProperty) { $BaseUrl = ([string]$BaseProperty.Value).TrimEnd("/") }
    }
    foreach ($RawValue in @($NamedProperty.Value)) {
        if ($null -eq $RawValue) { continue }
        $Value = ([string]$RawValue).Trim()
        if (-not $Value) { continue }
        if (-not $Value.StartsWith("https://")) {
            if (-not $BaseUrl) {
                throw "modelscope_artifacts.$Name 使用相对路径，但未配置 modelscope.base_url。"
            }
            $Value = $BaseUrl + "/" + $Value.TrimStart("/")
        }
        if (-not (Test-ASMRDubberModelScopeUrl -Url $Value)) {
            throw "modelscope_artifacts.$Name 包含无效的 ModelScope URL：$Value"
        }
        if (-not $Result.Contains($Value)) { [void]$Result.Add($Value) }
    }
    return $Result.ToArray()
}

function Find-ASMRDubberLocalArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [string]$Sha256 = ""
    )

    if (-not $env:ASMR_DUBBER_LOCAL_CACHE_ROOTS) { return $null }
    try {
        $FileName = [System.IO.Path]::GetFileName(([Uri]$Url).LocalPath)
    } catch {
        return $null
    }
    if (-not $FileName) { return $null }
    foreach ($RawRoot in $env:ASMR_DUBBER_LOCAL_CACHE_ROOTS -split ";") {
        if (-not $RawRoot.Trim()) { continue }
        $CacheRoot = [System.IO.Path]::GetFullPath($RawRoot.Trim())
        foreach ($Relative in @(
            $FileName,
            (Join-Path "model-packs" $FileName),
            (Join-Path ".asmr-dubber\cache\downloads" $FileName),
            (Join-Path ".asmr-dubber\bootstrap\windows" $FileName),
            (Join-Path ".asmr-dubber\bootstrap\linux" $FileName)
        )) {
            $Candidate = Join-Path $CacheRoot $Relative
            if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
            if ($Sha256 -and (Get-ASMRDubberFileSha256 -Path $Candidate) -ne `
                $Sha256.ToLowerInvariant()) {
                continue
            }
            return (Get-Item -LiteralPath $Candidate).FullName
        }
    }
    return $null
}

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
    $AllowExternal = Test-ASMRDubberExternalDownloadsAllowed -Configuration $Configuration
    if ($Preferred) {
        $PreferredText = $Preferred.Trim()
        if (-not $PreferredText.StartsWith("https://")) {
            Write-Warning "忽略非 HTTPS 首选镜像：$PreferredText"
        } elseif (-not $AllowExternal -and `
            (Test-ASMRDubberExternalUrl -Url $PreferredText) -and `
            -not (Test-ASMRDubberModelScopeUrl -Url $PreferredText)) {
            Write-Warning (
                "忽略被下载策略禁用的海外首选镜像：$PreferredText。" +
                "如确需使用，请显式设置 ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1。"
            )
        } else {
            [void]$Values.Add($PreferredText.TrimEnd("/"))
        }
    }
    foreach ($ModelScopeValue in Get-ASMRDubberModelScopeArtifacts `
        -Configuration $Configuration -Name $Name) {
        if (-not $Values.Contains($ModelScopeValue)) {
            [void]$Values.Add($ModelScopeValue)
        }
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
        if ($Text -and -not $AllowExternal -and `
            (Test-ASMRDubberExternalUrl -Url $Text) -and `
            -not (Test-ASMRDubberModelScopeUrl -Url $Text)) {
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

function Set-ASMRDubberHuggingFaceEnvironment {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [string]$Preferred = ""
    )

    $EffectivePreferred = $Preferred.Trim()
    if (-not $EffectivePreferred -and $env:ASMR_DUBBER_HF_ENDPOINT) {
        $EffectivePreferred = $env:ASMR_DUBBER_HF_ENDPOINT.Trim()
    }
    if (-not $EffectivePreferred -and $env:HF_ENDPOINT) {
        $EffectivePreferred = $env:HF_ENDPOINT.Trim()
    }
    $Endpoints = @(Get-ASMRDubberMirrorList -Configuration $Configuration `
        -Name "huggingface_endpoints" -Preferred $EffectivePreferred)
    $env:ASMR_DUBBER_HF_ENDPOINTS = $Endpoints -join ";"
    if ($Endpoints.Count -gt 0) {
        $env:ASMR_DUBBER_HF_ENDPOINT = $Endpoints[0]
        $env:HF_ENDPOINT = $Endpoints[0]
    } else {
        Remove-Item Env:ASMR_DUBBER_HF_ENDPOINT -ErrorAction SilentlyContinue
        Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue
    }
    return $Endpoints
}

function Get-ASMRDubberGitHubUrls {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Url
    )

    if (-not $Url.StartsWith("https://github.com/")) {
        return $Url
    }
    if (-not (Test-ASMRDubberExternalDownloadsAllowed -Configuration $Configuration)) {
        return @()
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

function Get-ASMRDubberFileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $Stream = $null
    $Hasher = $null
    try {
        $Stream = New-Object System.IO.FileStream(
            $FullPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::Read
        )
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        $Hash = $Hasher.ComputeHash($Stream)
        return ([System.BitConverter]::ToString($Hash)).Replace("-", "").ToLowerInvariant()
    } finally {
        if ($Hasher) { $Hasher.Dispose() }
        if ($Stream) { $Stream.Dispose() }
    }
}

function Invoke-ASMRDubberDownload {
    param(
        [Parameter(Mandatory = $true)][object]$Configuration,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$Sha256 = "",
        [switch]$Resume
    )

    $DestinationFullPath = [System.IO.Path]::GetFullPath($Destination)
    $Partial = "$Destination.partial"
    $PartialPath = [System.IO.Path]::GetFullPath($Partial)
    if ($Sha256 -and (Test-Path -LiteralPath $DestinationFullPath -PathType Leaf)) {
        $ExpectedHash = $Sha256.ToLowerInvariant()
        $ExistingHash = Get-ASMRDubberFileSha256 -Path $DestinationFullPath
        if ($ExistingHash -eq $ExpectedHash) {
            if (Test-Path -LiteralPath $PartialPath -PathType Leaf) {
                Remove-Item -LiteralPath $PartialPath -Force
            }
            Write-Host "复用已完整下载且校验通过的文件：$DestinationFullPath" `
                -ForegroundColor Green
            return "existing:$DestinationFullPath"
        }
        Write-Warning "已有下载文件校验不通过，将下载到断点文件并在成功后替换。"
    }

    $Candidates = @(Get-ASMRDubberGitHubUrls -Configuration $Configuration -Url $Url)
    if (-not (Test-ASMRDubberExternalDownloadsAllowed -Configuration $Configuration) -and `
        (Test-ASMRDubberExternalUrl -Url $Url) -and `
        -not (Test-ASMRDubberModelScopeUrl -Url $Url)) {
        $Candidates = @()
    }
    if ($Candidates.Count -eq 0) {
        throw (
            "下载地址被当前策略禁用：$Url。请使用 ModelScope 文件，或显式设置 " +
            "ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS=1。"
        )
    }
    $DestinationParent = Split-Path -Parent $DestinationFullPath
    if ($DestinationParent) {
        New-Item -ItemType Directory -Force -Path $DestinationParent | Out-Null
    }
    foreach ($Candidate in $Candidates) {
        $LocalArtifact = Find-ASMRDubberLocalArtifact -Url $Candidate -Sha256 $Sha256
        if ($LocalArtifact) {
            Write-Host "复用只读本地缓存：$LocalArtifact" -ForegroundColor Green
            $LocalFullPath = [System.IO.Path]::GetFullPath($LocalArtifact)
            if ($LocalFullPath -ne $DestinationFullPath) {
                Copy-Item -LiteralPath $LocalFullPath -Destination $DestinationFullPath -Force
            }
            return "local:$LocalFullPath"
        }
    }
    $Curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    if (-not $Curl) {
        throw "Windows 缺少 curl.exe，无法下载文件。"
    }
    $Errors = New-Object System.Collections.Generic.List[string]
    foreach ($Candidate in $Candidates) {
        Write-Host "尝试下载：$Candidate" -ForegroundColor Cyan
        $CommonArguments = @(
            "-L", "--fail", "--retry", "4", "--retry-all-errors",
            "--retry-delay", "1", "--connect-timeout", "20",
            "--header", "Accept-Encoding: identity"
        )
        if (Test-ASMRDubberModelScopeUrl -Url $Candidate) {
            $CommonArguments += @(
                "--header", "User-Agent: curl/8.0",
                "--header", "Referer: https://modelscope.cn/"
            )
            if ($env:MODELSCOPE_API_TOKEN) {
                $CommonArguments += @(
                    "--header", "Authorization: Bearer $($env:MODELSCOPE_API_TOKEN)",
                    "--header", "Cookie: m_session_id=$($env:MODELSCOPE_API_TOKEN)"
                )
            }
        }
        $Arguments = $CommonArguments + @("--output", $PartialPath)
        if ($Resume -and (Test-Path $Partial)) {
            $Arguments += @("-C", "-")
        }
        $Arguments += $Candidate
        $ExitCode = Invoke-ASMRDubberProcess -FilePath $Curl.Source `
            -ArgumentList $Arguments -WorkingDirectory (Get-Location).Path
        if ($ExitCode -eq 33 -and $Resume) {
            Remove-Item -Force -ErrorAction SilentlyContinue $Partial
            $Arguments = $CommonArguments + @("--output", $PartialPath, $Candidate)
            $ExitCode = Invoke-ASMRDubberProcess -FilePath $Curl.Source `
                -ArgumentList $Arguments -WorkingDirectory (Get-Location).Path
        }
        if ($ExitCode -eq 0 -and (Test-Path $Partial)) {
            Move-Item -Force $Partial $Destination
            if (-not $Sha256 -or (
                (Get-ASMRDubberFileSha256 -Path $Destination) -eq
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
                $Headers = @{}
                if (Test-ASMRDubberModelScopeUrl -Url $Candidate) {
                    $Headers["User-Agent"] = "curl/8.0"
                    $Headers["Accept-Encoding"] = "identity"
                    $Headers["Referer"] = "https://modelscope.cn/"
                    if ($env:MODELSCOPE_API_TOKEN) {
                        $Headers["Authorization"] = "Bearer $($env:MODELSCOPE_API_TOKEN)"
                        $Headers["Cookie"] = "m_session_id=$($env:MODELSCOPE_API_TOKEN)"
                    }
                }
                return Invoke-RestMethod -Uri $Candidate -Headers $Headers `
                    -UseBasicParsing -TimeoutSec 30
            } catch {
                [void]$Errors.Add("$Candidate：$($_.Exception.Message)")
                Write-Warning "当前下载源失败，自动切换。"
            }
        }
    }
    throw "所有下载源均失败：$($Errors -join '；')"
}

function ConvertTo-ASMRDubberWindowsCommandLineArgument {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Argument
    )

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $Builder = New-Object System.Text.StringBuilder
    [void]$Builder.Append('"')
    $Backslashes = 0
    foreach ($Character in $Argument.ToCharArray()) {
        if ($Character -eq '\') {
            $Backslashes++
            continue
        }
        if ($Character -eq '"') {
            [void]$Builder.Append(('\' * (($Backslashes * 2) + 1)))
            [void]$Builder.Append('"')
        } else {
            if ($Backslashes -gt 0) {
                [void]$Builder.Append(('\' * $Backslashes))
            }
            [void]$Builder.Append($Character)
        }
        $Backslashes = 0
    }
    if ($Backslashes -gt 0) {
        [void]$Builder.Append(('\' * ($Backslashes * 2)))
    }
    [void]$Builder.Append('"')
    return $Builder.ToString()
}

function Join-ASMRDubberWindowsCommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string[]]$ArgumentList
    )

    return (($ArgumentList | ForEach-Object {
        ConvertTo-ASMRDubberWindowsCommandLineArgument -Argument $_
    }) -join ' ')
}

function Start-ASMRDubberProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $FilePath
    $StartInfo.Arguments = Join-ASMRDubberWindowsCommandLine -ArgumentList $ArgumentList
    $StartInfo.WorkingDirectory = $WorkingDirectory
    $StartInfo.UseShellExecute = $false
    $Process = [System.Diagnostics.Process]::Start($StartInfo)
    if (-not $Process) {
        throw "无法启动进程：$FilePath"
    }
    return $Process
}

function Invoke-ASMRDubberProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $Process = Start-ASMRDubberProcess -FilePath $FilePath `
        -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory
    $Process.WaitForExit()
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
