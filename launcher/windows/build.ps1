[CmdletBinding()]
param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. (Join-Path $ProjectRoot "scripts\mirrors.ps1")
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
    $TemporaryOutput = Join-Path `
        ([System.IO.Path]::GetDirectoryName($Item.Output)) `
        ([System.IO.Path]::GetFileNameWithoutExtension($Item.Output) + ".new.exe")
    Remove-Item -Force -ErrorAction SilentlyContinue $TemporaryOutput
    try {
        $CompileExitCode = Invoke-ASMRDubberProcess -FilePath $Compiler `
            -ArgumentList @(
                "/nologo",
                "/target:exe",
                "/optimize+",
                "/codepage:65001",
                "/out:$TemporaryOutput",
                "/reference:System.dll",
                "/reference:System.Core.dll",
                $Item.Source
            ) `
            -WorkingDirectory $ProjectRoot
        if ($CompileExitCode -ne 0 -or -not (Test-Path $TemporaryOutput)) {
            throw "Windows 启动器编译失败（退出码 $CompileExitCode）。"
        }
        [void][System.Reflection.AssemblyName]::GetAssemblyName($TemporaryOutput)
        Move-Item -Force $TemporaryOutput $Item.Output
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $TemporaryOutput
    }
    Write-Host "已生成：$($Item.Output)" -ForegroundColor Green
}
