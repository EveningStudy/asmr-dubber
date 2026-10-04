[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
. (Join-Path $Root "scripts\mirrors.ps1")
$MirrorConfiguration = Get-ASMRDubberMirrorConfiguration -Root $Root
$HuggingFaceEndpoints = @(Set-ASMRDubberHuggingFaceEnvironment `
    -Configuration $MirrorConfiguration)
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
$Python = $Paths.Python
if (-not (Test-Path $Python)) {
    & (Join-Path $PSScriptRoot "setup.ps1") -Profile Core
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) {
        throw "基础运行环境准备失败。"
    }
}
$DataRoot = $Paths.Home
. (Join-Path $Root "scripts\windows-runtime.ps1")
$SharedFFmpegBin = Enable-ASMRDubberSharedFFmpeg -DataRoot $DataRoot

$ProcessArguments = @("-m", "asmr_dubber.cli") + $Arguments
$ExitCode = Invoke-ASMRDubberProcess -FilePath $Python `
    -ArgumentList $ProcessArguments -WorkingDirectory $Root
exit $ExitCode
