[CmdletBinding()]
param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Source = Join-Path $PSScriptRoot "ASMRDubberLauncher.cs"
if (-not $Output) {
    $Output = Join-Path $ProjectRoot "ASMR-Dubber.exe"
}

$CompilerCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)
$Compiler = $CompilerCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Compiler) {
    throw "找不到 Windows .NET Framework C# 编译器。"
}

$global:LASTEXITCODE = 0
& $Compiler `
    /nologo `
    /target:exe `
    /optimize+ `
    /codepage:65001 `
    "/out:$Output" `
    /reference:System.dll `
    /reference:System.Core.dll `
    $Source
if ($LASTEXITCODE -ne 0) {
    throw "Windows 启动器编译失败（退出码 $LASTEXITCODE）。"
}

Write-Host "启动器已生成：$Output" -ForegroundColor Green
