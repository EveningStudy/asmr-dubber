#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PROFILE="${1:-Recommended}"
case "$PROFILE" in
  Core|Recommended|Full) ;;
  *)
    echo "用法：bash scripts/linux/setup.sh [Core|Recommended|Full]" >&2
    exit 2
    ;;
esac
if [[ "$(uname -s)" != "Linux" || "$(getconf LONG_BIT)" != "64" ]]; then
  echo "错误：Linux 安装脚本只支持 64 位 Linux；Windows 请双击 ASMR-Dubber.exe。" >&2
  exit 1
fi

source "$ROOT/scripts/portable-runtime.sh"
asmr_init_portable_environment "$ROOT"

source "$ROOT/scripts/network-bridge.sh"
asmr_prepare_network
trap asmr_cleanup_network EXIT INT TERM

DATA_ROOT="$ASMR_DUBBER_HOME"
CONFIG_ROOT="$ASMR_DUBBER_CONFIG_DIR"
BOOTSTRAP="$ASMR_DUBBER_HOME/bootstrap/linux"
UV_DIR="$ASMR_DUBBER_UV_DIR"
UV="$ASMR_DUBBER_UV"
VENV="$ASMR_DUBBER_VENV"
mkdir -p "$UV_DIR" "$DATA_ROOT" "$CONFIG_ROOT"

export UV_UNMANAGED_INSTALL="$UV_DIR"
export UV_NO_MODIFY_PATH=1

echo "ASMR Dubber · Linux 安装"
echo "项目目录：$ROOT"
echo "便携目录：$DATA_ROOT"
echo "安装配置：$PROFILE"

if [[ ! -x "$UV" ]]; then
  echo "正在安装项目私有 uv（不会修改系统 PATH）..."
  curl -fsSL --retry 5 https://astral.sh/uv/0.11.30/install.sh | sh
fi
if [[ ! -x "$UV" ]]; then
  echo "错误：uv 安装失败：$UV" >&2
  exit 1
fi

echo "正在准备项目私有 Python 3.12..."
"$UV" python install 3.12 --managed-python --no-bin
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV" venv --python 3.12 --managed-python "$VENV"
fi

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi
EXTRA=".[ui]"
INSTALL_DEFAULT_MODELS=0
INSTALL_RECOMMENDED_TTS=0
INSTALL_PARAKEET=0
case "$PROFILE" in
  Core)
    EXTRA=".[ui]"
    ;;
  Recommended)
    INSTALL_PARAKEET=1
    if [[ "$HAS_NVIDIA" == 1 ]]; then
      EXTRA=".[ui,asr-faster-whisper,asr-kotoba-whisper]"
      if [[ "${ASMR_DUBBER_SKIP_RECOMMENDED_TTS:-0}" != 1 ]]; then
        INSTALL_RECOMMENDED_TTS=1
      fi
    else
      EXTRA=".[ui,asr-faster-whisper,asr-kotoba-whisper]"
    fi
    ;;
  Full)
    INSTALL_PARAKEET=1
    if [[ "$HAS_NVIDIA" == 1 ]]; then
      EXTRA=".[ui,local-default,asr-faster-whisper,asr-kotoba-whisper,asr-openai-whisper,asr-funasr]"
      INSTALL_DEFAULT_MODELS=1
      if [[ "${ASMR_DUBBER_SKIP_RECOMMENDED_TTS:-0}" != 1 ]]; then
        INSTALL_RECOMMENDED_TTS=1
      fi
    else
      echo "提示：未检测到 NVIDIA GPU；跳过 CUDA 专用 Qwen3-ASR/VoxCPM2。" >&2
      EXTRA=".[ui,asr-faster-whisper,asr-kotoba-whisper,asr-openai-whisper,asr-funasr]"
    fi
    ;;
esac

echo "正在安装应用依赖：$EXTRA"
"$UV" pip install --python "$VENV/bin/python" --editable "$EXTRA"
# PyTorch 2.11 constrains setuptools below 82. Keep the newest compatible
# release so older package-discovery vulnerabilities are not retained by an
# in-place upgrade.
"$UV" pip install --python "$VENV/bin/python" "setuptools>=78.1.1,<82"
"$VENV/bin/python" -m compileall -q -f "$ROOT/src/asmr_dubber"

# TorchCodec loads system-style FFmpeg SONAMEs. PyAV ships a matching FFmpeg 8
# build on supported manylinux systems; expose only the canonical names in an
# application-private directory and leave the host's FFmpeg untouched.
AV_LIBS="$VENV/lib/python3.12/site-packages/av.libs"
FFMPEG_LIBS="$DATA_ROOT/runtimes/ffmpeg-libs"
mkdir -p "$FFMPEG_LIBS"
link_av_library() {
  local soname="$1" pattern="$2"
  local matches=()
  shopt -s nullglob
  matches=("$AV_LIBS"/$pattern)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 1 ]]; then
    ln -sfn "${matches[0]}" "$FFMPEG_LIBS/$soname"
  fi
}
link_av_library libavutil.so.60 'libavutil-*.so.60.*'
link_av_library libavcodec.so.62 'libavcodec-*.so.62.*'
link_av_library libavformat.so.62 'libavformat-*.so.62.*'
link_av_library libavfilter.so.11 'libavfilter-*.so.11.*'
link_av_library libavdevice.so.62 'libavdevice-*.so.62.*'
link_av_library libswresample.so.6 'libswresample-*.so.6.*'
link_av_library libswscale.so.9 'libswscale-*.so.9.*'

if [[ "$INSTALL_DEFAULT_MODELS" == 1 ]]; then
  echo "正在下载并校验默认模型（已有缓存会复用）..."
  if [[ "$PROFILE" == Full ]]; then
    bash "$ROOT/scripts/linux/run-cli.sh" download-models --backend all
  else
    bash "$ROOT/scripts/linux/run-cli.sh" download-models --backend qwen3_asr
  fi
fi

if [[ "$INSTALL_PARAKEET" == 1 ]]; then
  echo "正在安装推荐 ASR：Parakeet 日语..."
  bash "$ROOT/scripts/linux/install-parakeet.sh"
fi

if [[ "$INSTALL_RECOMMENDED_TTS" == 1 ]]; then
  echo "正在安装推荐 TTS：IndexTTS2（约需 20 GB）..."
  bash "$ROOT/scripts/linux/install-indextts2.sh"
fi

echo "正在执行环境检查..."
if ! bash "$ROOT/scripts/linux/run-cli.sh" doctor --no-network; then
  echo "提示：核心程序已安装，但所选本地模型尚未全部可用；请在设置 → 设备与模型中查看。" >&2
fi
echo
echo "安装完成。运行 bash $ROOT/scripts/linux/run-ui.sh 启动界面。"
echo "卸载方法：删除整个项目目录；程序不会在用户主目录留下文件。"
