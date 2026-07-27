#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PROFILE="${1:-推荐}"
case "$PROFILE" in
  Core) PROFILE="基础" ;;
  Recommended) PROFILE="推荐" ;;
  Advanced) PROFILE="进阶" ;;
  基础|推荐|进阶) ;;
  *)
    echo "用法：bash scripts/linux/setup.sh [基础|推荐|进阶]" >&2
    exit 2
    ;;
esac
if [[ "$(uname -s)" != "Linux" || "$(getconf LONG_BIT)" != "64" ]] || \
  [[ "$(uname -m)" != "x86_64" ]]; then
  echo "错误：Linux 安装脚本只支持 x86_64 64 位 Linux；Windows 请双击 ASMR-Dubber.exe。" >&2
  exit 1
fi

source "$ROOT/scripts/portable-runtime.sh"
asmr_init_portable_environment "$ROOT"
source "$ROOT/scripts/mirrors.sh"
asmr_apply_mirror_environment "$ROOT"
source "$ROOT/scripts/linux/python-runtime.sh"
source "$ROOT/scripts/linux/wheelhouse.sh"

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
echo "数据目录：$DATA_ROOT"
echo "安装配置：$PROFILE"
if asmr_external_downloads_allowed "$ROOT"; then
  echo "下载策略：ModelScope 优先；已显式允许海外备用源。"
else
  echo "下载策略：ModelScope 优先；GitHub/Hugging Face/官方海外源已关闭。"
fi
if [[ -n "${ASMR_DUBBER_LOCAL_CACHE_ROOTS:-}" ]]; then
  echo "只读本地缓存：$ASMR_DUBBER_LOCAL_CACHE_ROOTS"
fi
case "$PROFILE" in
  基础)
    echo "预计安装后占用：约 2 GB；建议安装前至少有 5 GB 可用空间"
    ;;
  推荐)
    echo "预计安装后占用：约 24–28 GB；建议安装前至少有 35 GB 可用空间"
    ;;
  进阶)
    echo "预计安装后占用：约 33–39 GB；建议安装前至少有 50 GB 可用空间"
    echo "固定 ASR 模型：Parakeet CTC 1.1B JA GAL、Parakeet TDT/CTC 0.6B JA"
    echo "固定 ASR 模型：kotoba-tech/kotoba-whisper-v2.2"
    echo "固定 ASR 模型：Systran/faster-whisper-large-v2"
    echo "固定 VAD 模型：TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
    echo "固定时间戳模型：Qwen/Qwen3-ForcedAligner-0.6B（阿里 Qwen）"
    echo "固定 TTS 模型：IndexTTS2 checkpoints（仅 NVIDIA GPU）"
    echo "不会自动安装 Kotoba v2.0/v2.1、Faster-Whisper large-v3 或其它识别模型"
    ;;
esac
if [[ "$PROFILE" != 基础 ]]; then
  echo "未检测到 NVIDIA GPU 时会跳过需要 CUDA 的 TTS（语音合成），实际占用将减少。"
fi

if [[ ! -x "$UV" ]]; then
  echo "正在从 ModelScope 优先源安装 uv..."
  UV_ARCHIVE="$BOOTSTRAP/uv-x86_64-unknown-linux-gnu.tar.gz"
  UV_SHA256="04bc7d180d6138bf6dc08387acf507a823f397a98fea55da36b0ccc7fbce3b68"
  UV_READY=0
  while IFS= read -r archive_url; do
    [[ -n "$archive_url" ]] || continue
    if asmr_download "$ROOT" "$archive_url" "$UV_ARCHIVE" "$UV_SHA256"; then
      UV_STAGING="$BOOTSTRAP/uv-extract"
      rm -rf "$UV_STAGING"
      mkdir -p "$UV_STAGING" "$UV_DIR"
      tar -xzf "$UV_ARCHIVE" -C "$UV_STAGING"
      UV_DOWNLOADED="$(find "$UV_STAGING" -type f -name uv -perm -u+x | head -n 1)"
      if [[ -n "$UV_DOWNLOADED" ]]; then
        cp -f "$UV_DOWNLOADED" "$UV"
        chmod 755 "$UV"
        UVX_DOWNLOADED="$(find "$UV_STAGING" -type f -name uvx -perm -u+x | head -n 1)"
        if [[ -n "$UVX_DOWNLOADED" ]]; then
          cp -f "$UVX_DOWNLOADED" "$UV_DIR/uvx"
          chmod 755 "$UV_DIR/uvx"
        fi
        UV_READY=1
      fi
      rm -rf "$UV_STAGING"
      [[ "$UV_READY" == 1 ]] && break
    fi
  done < <(asmr_mirror_list "$ROOT" uv_archives_linux)
fi
if [[ ! -x "$UV" ]]; then
  echo "错误：uv 安装失败：$UV" >&2
  exit 1
fi

echo "正在准备 Python 3.12..."
asmr_install_python_runtime \
  "$ROOT" \
  "3.12.13" \
  "20260718" \
  "5854aa6ec71cad00334d5065633c210b2e7feb40956767a59a91791cadcf0b79" \
  "python312_linux_archives"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV" venv --python 3.12 --managed-python "$VENV"
fi

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi
EXTRA=".[ui]"
INSTALL_ADVANCED_MODELS=0
INSTALL_RECOMMENDED_TTS=0
INSTALL_PARAKEET=0
case "$PROFILE" in
  基础)
    EXTRA=".[ui]"
    ;;
  推荐)
    INSTALL_PARAKEET=1
    if [[ "$HAS_NVIDIA" == 1 ]]; then
      if [[ "${ASMR_DUBBER_SKIP_RECOMMENDED_TTS:-0}" != 1 ]]; then
        INSTALL_RECOMMENDED_TTS=1
      fi
    fi
    ;;
  进阶)
    INSTALL_PARAKEET=1
    INSTALL_ADVANCED_MODELS=1
    EXTRA=".[ui,asr-faster-whisper,asr-kotoba-whisper,asr-forced-aligner,asr-asmr-vad]"
    if [[ "$HAS_NVIDIA" == 1 ]] \
      && [[ "${ASMR_DUBBER_SKIP_RECOMMENDED_TTS:-0}" != 1 ]]; then
      INSTALL_RECOMMENDED_TTS=1
    fi
    ;;
esac

echo "正在安装应用依赖：$EXTRA"
install_from_pypi() {
  local index
  while IFS= read -r index; do
    echo "使用软件源：$index"
    if "$UV" "$@" --default-index "$index"; then
      return 0
    fi
    echo "当前软件源失败，自动切换。" >&2
  done < <(asmr_mirror_list "$ROOT" pypi_indexes)
  return 1
}
if asmr_prepare_wheelhouse \
  "$ROOT" \
  "ASMR-Dubber-Linux-Wheelhouse-v0.4.0.tar.gz" \
  "linux_application_wheelhouse_archives" \
  "linux_application_wheelhouse_checksums"; then
  echo "使用 ModelScope 应用依赖 wheelhouse：$ASMR_WHEELHOUSE_RESULT"
  "$UV" pip install --python "$VENV/bin/python" --editable "$EXTRA" \
    "setuptools>=78.1.1,<82" --offline --find-links "$ASMR_WHEELHOUSE_RESULT"
else
  WHEELHOUSE_STATUS=$?
  if [[ "$WHEELHOUSE_STATUS" == 2 ]]; then
    echo "错误：ModelScope wheelhouse 已发布但不完整，拒绝静默切换。" >&2
    exit 1
  fi
  install_from_pypi pip install --python "$VENV/bin/python" --editable "$EXTRA"
  # PyTorch 2.11 constrains setuptools below 82. Keep the newest compatible
  # release so older package-discovery vulnerabilities are not retained by an
  # in-place upgrade.
  install_from_pypi pip install --python "$VENV/bin/python" "setuptools>=78.1.1,<82"
fi
"$VENV/bin/python" -m compileall -q -f "$ROOT/src/asmr_dubber"

if [[ "$PROFILE" != 基础 ]]; then
  echo "正在检测并导入当前档位的本地模型包..."
  PACK_ARGUMENTS=(import-model-packs --all)
  case "$PROFILE" in
    推荐)
      PACK_ARGUMENTS+=(
        --pack-id parakeet-ja-linux
        --pack-id indextts2-checkpoints
      )
      ;;
    进阶)
      PACK_ARGUMENTS+=(
        --pack-id parakeet-ja-linux
        --pack-id indextts2-checkpoints
        --pack-id kotoba-whisper-v2.2
        --pack-id faster-whisper-large-v2
        --pack-id qwen3-forced-aligner
        --pack-id whisper-vad-asmr-onnx
      )
      ;;
  esac
  if ! bash "$ROOT/scripts/linux/run-cli.sh" "${PACK_ARGUMENTS[@]}"; then
    echo "错误：本地模型包扫描或导入失败；请检查 model-packs 目录中的压缩包。" >&2
    exit 1
  fi
fi

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

if [[ "$INSTALL_ADVANCED_MODELS" == 1 ]]; then
  echo "正在准备 ASR（语音识别）：Kotoba-Whisper v2.2、Faster-Whisper large-v2..."
  bash "$ROOT/scripts/linux/run-cli.sh" download-models --backend 进阶语音识别
fi

if [[ "$INSTALL_PARAKEET" == 1 ]]; then
  echo "正在安装推荐 ASR（语音识别）：Parakeet 日语..."
  bash "$ROOT/scripts/linux/install-parakeet.sh"
fi

if [[ "$INSTALL_RECOMMENDED_TTS" == 1 ]]; then
  echo "正在安装推荐 TTS（语音合成）：IndexTTS2（约需 20 GB）..."
  bash "$ROOT/scripts/linux/install-indextts2.sh"
fi

echo "正在执行环境检查..."
if ! bash "$ROOT/scripts/linux/run-cli.sh" doctor --no-network; then
  echo "提示：核心程序已安装，但所选本地模型尚未全部可用；请在设置 → 设备与模型中查看。" >&2
fi
echo
echo "安装完成。运行 bash $ROOT/scripts/linux/run-ui.sh 启动界面。"
