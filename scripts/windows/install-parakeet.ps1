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
    } else {
        Write-Warning "远程 Parakeet 模型包不可用，将继续使用原始下载源。"
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
$Url = "https://github.com/CrispStrobe/CrispASR/releases/download/$Version/$Asset"

function Get-CheckedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Sha256
    )
    for ($Attempt = 1; $Attempt -le 2; $Attempt++) {
        if (Test-Path $Destination) {
            $Actual = Get-ASMRDubberFileSha256 -Path $Destination
            if ($Actual -eq $Sha256) {
                return
            }
            Remove-Item -Force $Destination
        }

        Invoke-ASMRDubberDownload -Configuration $MirrorConfiguration `
            -Url $Uri -Destination $Destination -Sha256 $Sha256 -Resume | Out-Null
        $Actual = Get-ASMRDubberFileSha256 -Path $Destination
        if ($Actual -eq $Sha256) {
            return
        }

        Remove-Item -Force $Destination
        Remove-Item -Force -ErrorAction SilentlyContinue "$Destination.partial"
        if ($Attempt -lt 2) {
            Write-Warning "SHA256 校验失败，将丢弃损坏文件并完整重试一次。"
        }
    }
    throw "SHA256 校验失败：$Destination"
}

Write-Host "正在安装 CrispASR $Version（$Variant）..." -ForegroundColor Cyan
Get-CheckedDownload -Uri $Url -Destination $Archive -Sha256 $ExpectedHash
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

Invoke-PythonChecked -Arguments @(
    (Join-Path $Root "scripts\download_hf_file.py"),
    "--repo", "cstr/parakeet-ctc-1.1b-ja-GGUF",
    "--filename", "parakeet-ctc-1.1b-ja-f16.gguf",
    "--revision", "7ccb2922f63cefe7c0d2735527c69aa46c05ceb9",
    "--destination", (Join-Path $ModelRoot "parakeet-ctc-1.1b-ja-f16.gguf"),
    "--minimum-bytes", "2000000000",
    "--endpoints", ((Get-ASMRDubberMirrorList `
        -Configuration $MirrorConfiguration -Name "huggingface_endpoints" `
        -Preferred $PreferredHuggingFace) -join ";")
) -FailureMessage "Parakeet 1.1B GAL F16 模型下载失败。"

Invoke-PythonChecked -Arguments @(
    (Join-Path $Root "scripts\download_hf_file.py"),
    "--repo", "cstr/parakeet-tdt-0.6b-ja-GGUF",
    "--filename", "parakeet-tdt-0.6b-ja.gguf",
    "--revision", "65341fce2b46d25ea51593b1f771ed9a73cf7108",
    "--destination", (Join-Path $ModelRoot "parakeet-tdt-0.6b-ja.gguf"),
    "--minimum-bytes", "1000000000",
    "--endpoints", ((Get-ASMRDubberMirrorList `
        -Configuration $MirrorConfiguration -Name "huggingface_endpoints" `
        -Preferred $PreferredHuggingFace) -join ";")
) -FailureMessage "Parakeet 0.6B 模型下载失败。"

$SelfTest = Invoke-ASMRDubberProcess -FilePath (Join-Path $RuntimeBin "crispasr.exe") `
    -ArgumentList @("--version") -WorkingDirectory $Root
if ($SelfTest -ne 0) { throw "CrispASR 安装后自检失败。" }
Write-Host "Parakeet 已就绪。运行时、模型和缓存都位于 .asmr-dubber。" -ForegroundColor Green
