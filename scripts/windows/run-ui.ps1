[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$global:LASTEXITCODE = 0
& (Join-Path $PSScriptRoot "run-cli.ps1") "ui" "--host" $HostAddress "--port" $Port
$ExitCode = $global:LASTEXITCODE
exit $ExitCode
