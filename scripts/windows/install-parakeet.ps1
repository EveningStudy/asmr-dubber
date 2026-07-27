[CmdletBinding()]
param(
    [ValidateSet("Auto", "CPU", "CUDA")]
    [string]$Variant = "Auto"
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
$PreferredHuggingFace = if ($env:ASMR_DUBBER_HF_ENDPOINT) {
    $env:ASMR_DUBBER_HF_ENDPOINT
} else { "" }

$Version = "v0.8.21"
$RuntimeRoot = Join-Path $Paths.Runtimes "crispasr"
$RuntimeBin = Join-Path $RuntimeRoot "bin"
$ModelRoot = Join-Path $Paths.Home "models\parakeet"
$DownloadRoot = Join-Path $Paths.Cache "downloads"
$Python = $Paths.Python
New-Item -ItemType Directory -Force `
    -Path $RuntimeBin, $ModelRoot, $DownloadRoot | Out-Null

if (-not (Test-Path $Python)) {
    throw "缺少 Python 运行环境；请先运行项目根目录的 ASMR-Dubber-Setup.exe。"
}

if ($env:ASMR_DUBBER_MODEL_PACKS_PREPARED -ne "1") {
    & $Python -m asmr_dubber.cli prepare-model-pack parakeet-ja-windows
    if ($LASTEXITCODE -eq 0) {
        & $Python -m asmr_dubber.cli import-model-packs --all --pack-id parakeet-ja-windows
        if ($LASTEXITCODE -ne 0) {
            throw "Parakeet ModelScope 模型包已下载，但导入失败。"
        }
    } else {
        throw (
            "Parakeet ModelScope 模型包下载未完成。断点文件已保留；" +
            "请重新运行安装继续，不会另起一条大型模型下载。"
        )
    }
}

if ($Variant -eq "Auto") {
    $NvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    $Variant = if (Test-Path $NvidiaSmi) { "CUDA" } else { "CPU" }
}

$InstalledExecutable = Join-Path $RuntimeBin "crispasr.exe"
$Installed11B = Join-Path $ModelRoot "parakeet-ctc-1.1b-ja-f16.gguf"
$Installed06B = Join-Path $ModelRoot "parakeet-tdt-0.6b-ja.gguf"
$Expected11B = "34dd3128275c9bca2b4296f53c5f831feb258fcf3fdd28c29c0dc2d2f7d5ede7"
$Expected06B = "374eb0132eebaec4df77a9631cbbeb03790be48a4a517f6cc8e8bdb38fe9a584"
if (
    (Test-Path $InstalledExecutable) -and
    (Test-Path $Installed11B) -and
    (Test-Path $Installed06B) -and
    ((Get-ASMRDubberFileSha256 -Path $Installed11B) -eq $Expected11B) -and
    ((Get-ASMRDubberFileSha256 -Path $Installed06B) -eq $Expected06B)
) {
    $ReadyCheck = Invoke-ASMRDubberProcess -FilePath $InstalledExecutable `
        -ArgumentList @("--version") -WorkingDirectory $Root
    if ($ReadyCheck -eq 0) {
        Write-Host "Parakeet 本地运行时和两款模型已完整，无需联网下载。" `
            -ForegroundColor Green
        return
    }
}

$Asset = if ($Variant -eq "CUDA") {
    "crispasr-windows-x86_64-cuda.zip"
} else {
    "crispasr-windows-x86_64-cpu.zip"
}
$ExpectedHash = if ($Variant -eq "CUDA") {
    "d7db946f4b73fa0fbf3a3e27d63a994eee51a90138384813d35f7863a59aeba3"
} else {
    "c16ae6a69bad1c077c9bc01821fbbd6d3671a6ad114239eb0807cf3601e3b6f2"
}
$Archive = Join-Path $DownloadRoot $Asset
$ArchiveMirrorName = if ($Variant -eq "CUDA") {
    "crispasr_windows_cuda_archives"
} else {
    "crispasr_windows_cpu_archives"
}

function Get-CheckedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$MirrorName,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Sha256
    )
    if ((Test-Path $Destination) -and `
        (Get-ASMRDubberFileSha256 -Path $Destination) -eq $Sha256) {
        return
    }
    $Failures = New-Object System.Collections.Generic.List[string]
    foreach ($Uri in Get-ASMRDubberMirrorList -Configuration $MirrorConfiguration `
        -Name $MirrorName) {
        try {
            Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                -Url $Uri -Destination $Destination -Sha256 $Sha256 -Resume | Out-Null
            if ((Get-ASMRDubberFileSha256 -Path $Destination) -eq $Sha256) { return }
            throw "SHA-256 校验失败。"
        } catch {
            [void]$Failures.Add("$Uri：$($_.Exception.Message)")
        }
    }
    throw "所有 $MirrorName 下载源均失败；断点文件已保留：$($Failures -join '；')"
}

Write-Host "正在安装 CrispASR $Version（$Variant）..." -ForegroundColor Cyan
Get-CheckedDownload -MirrorName $ArchiveMirrorName `
    -Destination $Archive -Sha256 $ExpectedHash
$Staging = Join-Path $Paths.Temp "crispasr-install"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Staging
New-Item -ItemType Directory -Force -Path $Staging | Out-Null
Expand-Archive -Path $Archive -DestinationPath $Staging -Force
$Executable = Get-ChildItem $Staging -Filter "crispasr.exe" -File -Recurse |
    Select-Object -First 1
if (-not $Executable) {
    throw "CrispASR 压缩包中找不到 crispasr.exe。"
}
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $RuntimeBin
New-Item -ItemType Directory -Force -Path $RuntimeBin | Out-Null
Copy-Item -Force (Join-Path $Executable.Directory.FullName "*") $RuntimeBin -Recurse
Remove-Item -Recurse -Force $Staging

Write-Host "下载并校验日语 Parakeet 模型（1.1B 优先，随后准备 0.6B）..." `
    -ForegroundColor Cyan
function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    $ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
        -ArgumentList $Arguments -WorkingDirectory $Root
    if ($ExitCode -ne 0) {
        throw "$FailureMessage（退出码 $ExitCode）"
    }
}

function Get-ParakeetModel {
    param(
        [Parameter(Mandatory = $true)][string]$MirrorName,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$FileName,
        [Parameter(Mandatory = $true)][string]$Revision,
        [Parameter(Mandatory = $true)][long]$MinimumBytes
    )
    if ((Test-Path $Destination) -and `
        (Get-ASMRDubberFileSha256 -Path $Destination) -eq $Sha256) { return }
    $Failures = New-Object System.Collections.Generic.List[string]
    foreach ($Uri in Get-ASMRDubberMirrorList -Configuration $MirrorConfiguration `
        -Name $MirrorName) {
        try {
            Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
                -Url $Uri -Destination $Destination -Sha256 $Sha256 -Resume | Out-Null
            return
        } catch {
            [void]$Failures.Add("$Uri：$($_.Exception.Message)")
        }
    }
    if (Test-ASMRDubberExternalDownloadsAllowed -Configuration $MirrorConfiguration) {
        Invoke-PythonChecked -Arguments @(
            (Join-Path $Root "scripts\download_hf_file.py"),
            "--repo", $Repository, "--filename", $FileName,
            "--revision", $Revision, "--destination", $Destination,
            "--minimum-bytes", ([string]$MinimumBytes),
            "--sha256", $Sha256,
            "--endpoints", ((Get-ASMRDubberMirrorList `
                -Configuration $MirrorConfiguration -Name "huggingface_endpoints" `
                -Preferred $PreferredHuggingFace) -join ";")
        ) -FailureMessage "$FileName 下载失败。"
        return
    }
    throw (
        "$FileName 的 ModelScope 下载失败；断点文件已保留。" +
        "请上传镜像文件，或显式允许海外源。$($Failures -join '；')"
    )
}

Get-ParakeetModel -MirrorName "parakeet_11b_model_files" `
    -Destination $Installed11B -Sha256 $Expected11B `
    -Repository "cstr/parakeet-ctc-1.1b-ja-GGUF" `
    -FileName "parakeet-ctc-1.1b-ja-f16.gguf" `
    -Revision "7ccb2922f63cefe7c0d2735527c69aa46c05ceb9" `
    -MinimumBytes 2000000000
Get-ParakeetModel -MirrorName "parakeet_06b_model_files" `
    -Destination $Installed06B -Sha256 $Expected06B `
    -Repository "cstr/parakeet-tdt-0.6b-ja-GGUF" `
    -FileName "parakeet-tdt-0.6b-ja.gguf" `
    -Revision "65341fce2b46d25ea51593b1f771ed9a73cf7108" `
    -MinimumBytes 1000000000

$SelfTest = Invoke-ASMRDubberProcess -FilePath (Join-Path $RuntimeBin "crispasr.exe") `
    -ArgumentList @("--version") -WorkingDirectory $Root
if ($SelfTest -ne 0) { throw "CrispASR 安装后自检失败。" }
Write-Host "Parakeet 已就绪。运行时、模型和缓存都位于 .asmr-dubber。" -ForegroundColor Green
