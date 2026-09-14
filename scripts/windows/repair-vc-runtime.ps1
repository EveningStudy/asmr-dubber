$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$taskTemp = Join-Path ([System.IO.Path]::GetTempPath()) ('asmr-vcredist-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTemp | Out-Null
$installer = Join-Path $taskTemp 'vc_redist.x64.exe'
# Only Microsoft's official HTTPS endpoint; Authenticode is checked before execution.
& "$env:SystemRoot\System32\curl.exe" --fail --location --proto '=https' --proto-redir '=https' --connect-timeout 20 --max-time 300 --output $installer 'https://aka.ms/vc14/vc_redist.x64.exe'
if ($LASTEXITCODE -ne 0) { throw '微软运行库下载失败，未执行安装。请检查网络后重试。' }
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O=Microsoft Corporation(,|$)') {
    throw '安装程序签名无效或不是 Microsoft Corporation，已拒绝执行。'
}
Write-Output '已验证微软签名。将打开官方安装界面，请在本机确认安装或修复；不会自动重启。'
# A visible installer is intentional: users must approve its license/UAC and repair action.
$process = Start-Process -FilePath $installer -ArgumentList '/norestart' -Wait -PassThru
if ($process.ExitCode -eq 3010) { Write-Output '运行库安装完成，需要手动重启 Windows 后再测试。' }
elseif ($process.ExitCode -eq 0) { Write-Output '微软运行库安装程序已正常结束；接下来复检所选后端。' }
elseif ($process.ExitCode -eq 1602) { throw '用户取消安装，未报告修复成功。' }
else { throw "微软安装器返回 $($process.ExitCode)，请查看官方安装界面/日志。" }
