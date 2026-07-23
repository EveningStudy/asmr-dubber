#!/usr/bin/env bash

asmr_init_portable_environment() {
  local root="$1"
  export ASMR_DUBBER_HOME="$root/.asmr-dubber"
  export ASMR_DUBBER_DATA_DIR="$ASMR_DUBBER_HOME"
  export ASMR_DUBBER_CONFIG_DIR="$ASMR_DUBBER_HOME/config"
  export ASMR_DUBBER_VENV="$ASMR_DUBBER_HOME/venv"
  export ASMR_DUBBER_UV_DIR="$ASMR_DUBBER_HOME/bootstrap/linux/uv"
  export ASMR_DUBBER_UV="$ASMR_DUBBER_UV_DIR/uv"

  export UV_CACHE_DIR="$ASMR_DUBBER_HOME/cache/uv"
  export UV_PYTHON_INSTALL_DIR="$ASMR_DUBBER_HOME/runtimes/python"
  export HF_HOME="$ASMR_DUBBER_HOME/cache/huggingface"
  export HF_HUB_DISABLE_SYMLINKS_WARNING=1
  export MODELSCOPE_CACHE="$ASMR_DUBBER_HOME/cache/modelscope"
  export TORCH_HOME="$ASMR_DUBBER_HOME/cache/torch"
  export XDG_CACHE_HOME="$ASMR_DUBBER_HOME/cache/xdg"
  export PIP_CACHE_DIR="$ASMR_DUBBER_HOME/cache/pip"
  export NUMBA_CACHE_DIR="$ASMR_DUBBER_HOME/cache/numba"
  export MPLCONFIGDIR="$ASMR_DUBBER_HOME/cache/matplotlib"
  export NLTK_DATA="$ASMR_DUBBER_HOME/cache/nltk"
  export KERAS_HOME="$ASMR_DUBBER_HOME/cache/keras"
  export TRITON_CACHE_DIR="$ASMR_DUBBER_HOME/cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$ASMR_DUBBER_HOME/cache/torchinductor"
  export CUDA_CACHE_PATH="$ASMR_DUBBER_HOME/cache/nvidia-cuda"
  export HF_DATASETS_CACHE="$ASMR_DUBBER_HOME/cache/huggingface/datasets"
  export PYTHONPYCACHEPREFIX="$ASMR_DUBBER_HOME/cache/pycache"
  export PYTHONNOUSERSITE=1
  export GRADIO_TEMP_DIR="$ASMR_DUBBER_HOME/temp/gradio"
  export TMPDIR="$ASMR_DUBBER_HOME/temp"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_DISABLE_XET=1
  export GRADIO_ANALYTICS_ENABLED=False

  mkdir -p \
    "$ASMR_DUBBER_CONFIG_DIR" \
    "$ASMR_DUBBER_UV_DIR" \
    "$GRADIO_TEMP_DIR" \
    "$UV_CACHE_DIR" \
    "$HF_HOME" \
    "$MODELSCOPE_CACHE" \
    "$TORCH_HOME" \
    "$PIP_CACHE_DIR" \
    "$NUMBA_CACHE_DIR" \
    "$MPLCONFIGDIR" \
    "$NLTK_DATA" \
    "$PYTHONPYCACHEPREFIX"
}
