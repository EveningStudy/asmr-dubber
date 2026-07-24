[CmdletBinding()]
param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Sources = @(
    @{
        Source = Join-Path $PSScriptRoot "ASMRDubberLauncher.cs"
        Output = if ($Output) { $Output } else { Join-Path $ProjectRoot "ASMR-Dubber.exe" }
    },
    @{
        Source = Join-Path $PSScriptRoot "ASMRDubberSetup.cs"
        Output = Join-Path $ProjectRoot "ASMR-Dubber-Setup.exe"
    }
)

$CompilerCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)
$Compiler = $CompilerCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Compiler) {
    throw "找不到 Windows .NET Framework C# 编译器。"
}

foreach ($Item in $Sources) {
    $global:LASTEXITCODE = 0
    & $Compiler `
        /nologo `
        /target:exe `
        /optimize+ `
        /codepage:65001 `
        "/out:$($Item.Output)" `
        /reference:System.dll `
        /reference:System.Core.dll `
        $Item.Source
    if ($LASTEXITCODE -ne 0) {
        throw "Windows 启动器编译失败（退出码 $LASTEXITCODE）。"
    }
    Write-Host "已生成：$($Item.Output)" -ForegroundColor Green
}
