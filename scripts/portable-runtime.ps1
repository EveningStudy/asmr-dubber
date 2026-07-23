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
