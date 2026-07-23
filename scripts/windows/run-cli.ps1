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
. (Join-Path $Root "scripts\portable-runtime.ps1")
$Paths = Initialize-ASMRDubberPortableEnvironment -Root $Root -Create
$Python = $Paths.Python
if (-not (Test-Path $Python)) {
    throw "尚未安装。请双击项目根目录的 ASMR-Dubber.exe。"
}
$DataRoot = $Paths.Home
. (Join-Path $Root "scripts\windows-runtime.ps1")
$SharedFFmpegBin = Enable-ASMRDubberSharedFFmpeg -DataRoot $DataRoot

$ProcessArguments = @("-m", "asmr_dubber.cli") + $Arguments
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $Python
    $StartInfo.WorkingDirectory = $Root
    $StartInfo.UseShellExecute = $false
    foreach ($Argument in $ProcessArguments) {
        [void]$StartInfo.ArgumentList.Add($Argument)
    }
    $Process = [System.Diagnostics.Process]::Start($StartInfo)
    $Process.WaitForExit()
    $ExitCode = $Process.ExitCode
} else {
    $global:LASTEXITCODE = 0
    & $Python @ProcessArguments
    $ExitCode = $global:LASTEXITCODE
}
exit $ExitCode
