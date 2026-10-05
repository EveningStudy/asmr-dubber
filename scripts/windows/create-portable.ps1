[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$Version = "2.0.1"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutputRoot = [IO.Path]::GetFullPath($OutputDirectory)
$Staging = Join-Path $OutputRoot ("asmr-package-" + [guid]::NewGuid().ToString("N"))
$Package = Join-Path $Staging "ASMR-Dubber"
$SourceHome = Join-Path $Root ".asmr-dubber"
$PackageHome = Join-Path $Package ".asmr-dubber"
$BasePython = Get-ChildItem (Join-Path $SourceHome "runtimes\python\cpython-3.12.*-windows-x86_64-none\python.exe") -File |
    Sort-Object FullName -Descending | Select-Object -First 1
$UvSource = Join-Path $SourceHome "bootstrap\windows\uv"
if (-not $BasePython -or -not (Test-Path (Join-Path $UvSource "uv.exe"))) {
    throw "请先启动程序，准备项目内的基础 Python 和 uv。"
}
New-Item -ItemType Directory -Force -Path $Package, $PackageHome | Out-Null
try {
    & (Join-Path $Root "launcher\windows\build.ps1")
    foreach ($Name in @("src", "scripts", "launcher")) {
        Copy-Item -LiteralPath (Join-Path $Root $Name) -Destination $Package -Recurse
    }
    Get-ChildItem -LiteralPath $Package -Directory -Filter "__pycache__" -Recurse | ForEach-Object {
        $CachePath = [IO.Path]::GetFullPath($_.FullName)
        if (-not $CachePath.StartsWith($Package + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "字节码目录超出打包目录。"
        }
        Remove-Item -LiteralPath $CachePath -Recurse -Force
    }
    foreach ($Name in @("ASMR-Dubber.exe", "pyproject.toml", "uv.lock", "mirrors.json",
        "modelscope-artifacts.lock.json", "LICENSE", "README.md")) {
        Copy-Item -LiteralPath (Join-Path $Root $Name) -Destination $Package
    }
    $Runtime = Join-Path $PackageHome "runtimes\python"
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    Copy-Item -LiteralPath $BasePython.Directory.FullName -Destination $Runtime -Recurse
    $Bootstrap = Join-Path $PackageHome "bootstrap\windows"
    New-Item -ItemType Directory -Force -Path $Bootstrap | Out-Null
    Copy-Item -LiteralPath $UvSource -Destination $Bootstrap -Recurse
    $Bundled = Join-Path $Root "vendor\windows-core-wheelhouse"
    if (Test-Path $Bundled) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Package "vendor") | Out-Null
        Copy-Item -LiteralPath $Bundled -Destination (Join-Path $Package "vendor") -Recurse
    }
    $PowerShell = (Get-Command powershell.exe).Source
    & $PowerShell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Package "scripts\windows\setup.ps1") -Profile Core
    if ($LASTEXITCODE -ne 0) { throw "基础环境准备失败。" }
    $Python = Join-Path $PackageHome "venv\Scripts\python.exe"
    & $Python -c "from asmr_dubber.http_server import Server; from importlib.resources import files; assert files('asmr_dubber').joinpath('frontend/index.html').is_file()"
    if ($LASTEXITCODE -ne 0) { throw "界面资源检查失败。" }
    if (Test-Path (Join-Path $PackageHome "models")) {
        if (Get-ChildItem (Join-Path $PackageHome "models") -Recurse -File | Select-Object -First 1) {
            throw "基础包不应包含模型权重。"
        }
    }
    foreach ($Name in @("cache", "temp", "logs")) {
        $GeneratedPath = [IO.Path]::GetFullPath((Join-Path $PackageHome $Name))
        if (-not $GeneratedPath.StartsWith($PackageHome + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "生成目录超出打包数据目录。"
        }
        Remove-Item -LiteralPath $GeneratedPath -Recurse -Force -ErrorAction SilentlyContinue
    }
    $Destination = Join-Path $OutputRoot "ASMR-Dubber-Windows-v$Version.zip"
    & $BasePython.FullName -m zipfile -c $Destination $Package
    if ($LASTEXITCODE -ne 0) { throw "ZIP 打包失败。" }
    (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLower() |
        Set-Content ($Destination + ".sha256") -Encoding ascii
    Write-Host "已生成：$Destination"
} finally {
    $Resolved = [IO.Path]::GetFullPath($Staging)
    if (-not $Resolved.StartsWith($OutputRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "临时打包目录超出输出目录。"
    }
    Remove-Item -LiteralPath $Resolved -Recurse -Force -ErrorAction SilentlyContinue
}
