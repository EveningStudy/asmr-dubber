$BundledCoreWheelhouse = Join-Path $Root "vendor\windows-core-wheelhouse"
$BundledCoreWheelsReady = (
    (Test-Path -LiteralPath $BundledCoreWheelhouse -PathType Container) -and
    [bool](Get-ChildItem -LiteralPath $BundledCoreWheelhouse `
        -Filter "edge_tts-*.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1) -and
    [bool](Get-ChildItem -LiteralPath $BundledCoreWheelhouse `
        -Filter "pydantic-*.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1) -and
    [bool](Get-ChildItem -LiteralPath $BundledCoreWheelhouse `
        -Filter "hatchling-*.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1)
)

$ApplicationDependenciesReady = Test-ASMRDubberApplicationRuntime -PortableRoot $DataRoot
if (-not $ApplicationDependenciesReady) {
    Write-SetupHost "正在安装应用依赖：$Extra" -ForegroundColor Cyan
    $ApplicationArguments = @(
        "pip", "install", "--python", $Python, "--editable", $Extra,
        "setuptools>=78.1.1,<82"
    )
    $ApplicationInstalled = $false
    if ($BundledCoreWheelsReady) {
        Write-SetupHost "使用便携包内置的基础应用 wheelhouse。" -ForegroundColor Green
        try {
            Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
                -Wheelhouse $BundledCoreWheelhouse -Arguments $ApplicationArguments `
                -FailureMessage "内置基础应用 wheelhouse 安装失败"
            $ApplicationInstalled = $true
        } catch {
            Write-Warning "内置基础应用 wheelhouse 不完整，将继续尝试 ModelScope：$($_.Exception.Message)"
        }
    }
    if (-not $ApplicationInstalled) {
        $ApplicationWheelhouse = Get-ASMRDubberWheelhouse `
            -Root $Root -PortableRoot $DataRoot -MirrorConfiguration $MirrorConfiguration `
            -ArchiveName "ASMR-Dubber-Windows-Wheelhouse-v0.4.0.zip" `
            -ArchiveMirrorName "windows_application_wheelhouse_archives" `
            -ChecksumMirrorName "windows_application_wheelhouse_checksums"
    } else {
        $ApplicationWheelhouse = $null
    }
    if (-not $ApplicationInstalled -and $ApplicationWheelhouse) {
        Write-SetupHost "使用 ModelScope 应用依赖 wheelhouse。" -ForegroundColor Green
        try {
            Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
                -Wheelhouse $ApplicationWheelhouse -Arguments $ApplicationArguments `
                -FailureMessage "应用依赖 wheelhouse 安装失败"
            $ApplicationInstalled = $true
        } catch {
            Write-Warning (
                "ModelScope 应用 wheelhouse 早于当前依赖定义，将使用配置中的" +
                "国内软件源补齐应用依赖：$($_.Exception.Message)"
            )
        }
    }
    if (-not $ApplicationInstalled) {
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
            -Arguments @("pip", "install", "--python", $Python, "--editable", $Extra)
        # PyTorch 2.11 requires setuptools <82. Upgrade existing portable environments
        # to the newest compatible release instead of retaining an older installer.
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
            -Arguments @(
                "pip", "install", "--python", $Python, "setuptools>=78.1.1,<82"
            )
    }
} else {
    Write-SetupHost "应用依赖已由 Windows 依赖包提供。" -ForegroundColor Green
}

$ApiClientsReady = Test-ASMRDubberApiClientRuntime -PortableRoot $DataRoot
if (-not $ApiClientsReady) {
    Write-SetupHost "正在安装在线翻译与 TTS（语音合成）API 客户端..." -ForegroundColor Cyan
    $ApiClientArguments = @(
        # Keep both direct imports explicit.  edge-tts does not own the
        # project's HTTP client requirement, so installing only edge-tts can
        # leave a fresh basic profile with a misleading "Checked 1 package"
        # result and a failed post-install check.
        "pip", "install", "--python", $Python,
        "edge-tts==7.2.8", "httpx>=0.28.0"
    )
    $InstalledBundledApiClient = $false
    if ($BundledCoreWheelsReady) {
        Write-SetupHost "使用便携包内置的在线/API 客户端 wheelhouse。" -ForegroundColor Green
        try {
            Invoke-ASMRDubberUvOfflineWheelhouse -Uv $Uv -Root $Root `
                -Wheelhouse $BundledCoreWheelhouse -Arguments $ApiClientArguments `
                -FailureMessage "内置在线/API 客户端 wheelhouse 安装失败"
            $InstalledBundledApiClient = $true
        } catch {
            Write-Warning "内置在线/API 客户端不可用，将使用国内软件源：$($_.Exception.Message)"
        }
    }
    if (-not $InstalledBundledApiClient) {
        Invoke-ASMRDubberUvWithIndexFallback -Configuration $MirrorConfiguration `
            -Uv $Uv -Root $Root -MirrorName "pypi_indexes" -Preferred $PreferredIndex `
            -Arguments $ApiClientArguments
    }
    if (-not (Test-ASMRDubberApiClientRuntime -PortableRoot $DataRoot)) {
        throw "在线/API 客户端安装后仍不完整。"
    }
}
if (-not (Test-ASMRDubberCoreRuntime -PortableRoot $DataRoot)) {
    throw "基础应用或在线/API 客户端安装后仍不完整。"
}
Invoke-Checked -FilePath $Python `
    -ArgumentList @("-m", "compileall", "-q", "-f", (Join-Path $Root "src\asmr_dubber")) `
    -FailureMessage "应用字节码刷新失败"
