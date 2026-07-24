function Set-ASMRDubberTextFileIfChanged {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $Existing = if (Test-Path -LiteralPath $Path) {
        [System.IO.File]::ReadAllText($Path)
    } else {
        $null
    }
    if ($Existing -ne $Content) {
        $Parent = Split-Path -Parent $Path
        if ($Parent) {
            New-Item -ItemType Directory -Force -Path $Parent | Out-Null
        }
        [System.IO.File]::WriteAllText(
            $Path,
            $Content,
            (New-Object System.Text.UTF8Encoding($false))
        )
    }
}

function Repair-ASMRDubberPortablePythonPaths {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PortableRoot,
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)][string]$Venv
    )

    # A Windows venv records absolute paths in pyvenv.cfg and editable .pth
    # files. Repair only these project-owned text files so a prepared archive
    # remains usable after it is extracted or moved to a different directory.
    $Environments = @(
        @{
            Venv = $Venv
            PythonPattern = "cpython-3.12.*-windows-x86_64-none"
            EditablePattern = "_editable_impl_asmr_dubber.pth"
            EditableTarget = Join-Path $Root "src"
        },
        @{
            Venv = Join-Path $RuntimeRoot "index-tts\.venv"
            PythonPattern = "cpython-3.11.*-windows-x86_64-none"
            EditablePattern = "_editable_impl_indextts.pth"
            EditableTarget = Join-Path $RuntimeRoot "index-tts"
        }
    )

    foreach ($Environment in $Environments) {
        $EnvironmentRoot = [string]$Environment.Venv
        if (-not (Test-Path -LiteralPath $EnvironmentRoot)) {
            continue
        }
        $BasePython = Get-ChildItem `
            (Join-Path $RuntimeRoot "python\$($Environment.PythonPattern)\python.exe") `
            -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        $Configuration = Join-Path $EnvironmentRoot "pyvenv.cfg"
        if ($BasePython -and (Test-Path -LiteralPath $Configuration)) {
            $Lines = @(
                [System.IO.File]::ReadAllLines($Configuration) |
                    Where-Object { $_ -match "^\s*(#.*|[^=]+\s*=.*)$" }
            )
            $HomeLine = "home = $($BasePython.Directory.FullName)"
            $Replaced = $false
            for ($Index = 0; $Index -lt $Lines.Length; $Index++) {
                if ($Lines[$Index] -match "^home\s*=") {
                    $Lines[$Index] = $HomeLine
                    $Replaced = $true
                    break
                }
            }
            if (-not $Replaced) {
                $Lines = @($HomeLine) + $Lines
            }
            Set-ASMRDubberTextFileIfChanged `
                -Path $Configuration -Content (($Lines -join "`r`n") + "`r`n")
        }

        $SitePackages = Join-Path $EnvironmentRoot "Lib\site-packages"
        $Editable = Join-Path $SitePackages ([string]$Environment.EditablePattern)
        if (Test-Path -LiteralPath $SitePackages) {
            Set-ASMRDubberTextFileIfChanged `
                -Path $Editable -Content (([string]$Environment.EditableTarget) + "`r`n")
        }
    }
}

function Initialize-ASMRDubberPortableEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [switch]$Create
    )

    $PortableRoot = Join-Path $Root ".asmr-dubber"
    $CacheRoot = Join-Path $PortableRoot "cache"
    $ConfigRoot = Join-Path $PortableRoot "config"
    $RuntimeRoot = Join-Path $PortableRoot "runtimes"
    $TempRoot = Join-Path $PortableRoot "temp"
    $Bootstrap = Join-Path $PortableRoot "bootstrap\windows"
    $UvDir = Join-Path $Bootstrap "uv"
    $Venv = Join-Path $PortableRoot "venv"

    if ($Create) {
        New-Item -ItemType Directory -Force -Path @(
            $PortableRoot,
            $CacheRoot,
            $ConfigRoot,
            $RuntimeRoot,
            $TempRoot,
            (Join-Path $TempRoot "gradio"),
            (Join-Path $CacheRoot "matplotlib"),
            (Join-Path $CacheRoot "nltk"),
            (Join-Path $CacheRoot "pycache"),
            $UvDir
        ) | Out-Null
    }

    # Force every application-owned path into the project. These variables are
    # process-local and never modify the Windows registry or persistent PATH.
    $env:ASMR_DUBBER_HOME = $PortableRoot
    $env:ASMR_DUBBER_DATA_DIR = $PortableRoot
    $env:ASMR_DUBBER_CONFIG_DIR = $ConfigRoot
    $env:UV_CACHE_DIR = Join-Path $CacheRoot "uv"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $RuntimeRoot "python"
    $env:HF_HOME = Join-Path $CacheRoot "huggingface"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:MODELSCOPE_CACHE = Join-Path $CacheRoot "modelscope"
    $env:TORCH_HOME = Join-Path $CacheRoot "torch"
    $env:XDG_CACHE_HOME = Join-Path $CacheRoot "xdg"
    $env:PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
    $env:NUMBA_CACHE_DIR = Join-Path $CacheRoot "numba"
    $env:MPLCONFIGDIR = Join-Path $CacheRoot "matplotlib"
    $env:NLTK_DATA = Join-Path $CacheRoot "nltk"
    $env:KERAS_HOME = Join-Path $CacheRoot "keras"
    $env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton"
    $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "torchinductor"
    $env:CUDA_CACHE_PATH = Join-Path $CacheRoot "nvidia-cuda"
    $env:HF_DATASETS_CACHE = Join-Path $CacheRoot "huggingface\datasets"
    $env:PYTHONPYCACHEPREFIX = Join-Path $CacheRoot "pycache"
    $env:PYTHONNOUSERSITE = "1"
    $env:GRADIO_TEMP_DIR = Join-Path $TempRoot "gradio"
    $env:TEMP = $TempRoot
    $env:TMP = $TempRoot
    $env:TMPDIR = $TempRoot
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:GRADIO_ANALYTICS_ENABLED = "False"

    Repair-ASMRDubberPortablePythonPaths `
        -Root $Root `
        -PortableRoot $PortableRoot `
        -RuntimeRoot $RuntimeRoot `
        -Venv $Venv

    return [pscustomobject]@{
        Root = $Root
        Home = $PortableRoot
        Cache = $CacheRoot
        Config = $ConfigRoot
        Runtimes = $RuntimeRoot
        Temp = $TempRoot
        Bootstrap = $Bootstrap
        UvDir = $UvDir
        Uv = Join-Path $UvDir "uv.exe"
        Venv = $Venv
        Python = Join-Path $Venv "Scripts\python.exe"
    }
}
