#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/portable-runtime.sh"
asmr_init_portable_environment "$ROOT"
source "$ROOT/scripts/mirrors.sh"
asmr_apply_mirror_environment "$ROOT"

source "$ROOT/scripts/network-bridge.sh"
asmr_prepare_network
trap asmr_cleanup_network EXIT INT TERM

DATA_ROOT="$ASMR_DUBBER_HOME"
INDEX_ROOT="$DATA_ROOT/runtimes/index-tts"
MODEL_DIR="$INDEX_ROOT/checkpoints"
REVISION="13495845e3028f0bb6ca1462ad22aa0e76349e40"
SOURCE_URL="${INDEXTTS_SOURCE_URL:-https://github.com/index-tts/index-tts/archive/$REVISION.zip}"
SOURCE_SHA256="${INDEXTTS_SOURCE_SHA256:-7ed8bc742e2eeeb83f922247ef0e27f96327f418acacb6c63f182cafd66887ba}"
MARKER="$INDEX_ROOT/.asmr-source-revision"
DOWNLOAD_ROOT="$DATA_ROOT/cache/downloads"
ARCHIVE="$DOWNLOAD_ROOT/index-tts-$REVISION.zip"
STAGING="$INDEX_ROOT.staging"
PYTHON="$ASMR_DUBBER_VENV/bin/python"

if [[ ! -x "$ASMR_DUBBER_UV" || ! -x "$PYTHON" ]]; then
  echo "缺少应用运行时。请先运行 bash $ROOT/scripts/linux/setup.sh Core。" >&2
  exit 1
fi
mkdir -p "$DOWNLOAD_ROOT"

source_files_ready() {
  [[ -f "$INDEX_ROOT/pyproject.toml" ]] &&
    [[ -f "$INDEX_ROOT/uv.lock" ]] &&
    [[ -d "$INDEX_ROOT/indextts" ]]
}

SOURCE_READY=0
if source_files_ready &&
  [[ -f "$MARKER" ]] &&
  [[ "$(tr -d '\r\n' <"$MARKER")" == "$REVISION" ]]; then
  SOURCE_READY=1
elif source_files_ready && [[ -d "$INDEX_ROOT/.git" ]] &&
  [[ "$(git -C "$INDEX_ROOT" rev-parse HEAD 2>/dev/null || true)" == "$REVISION" ]]; then
  printf '%s\n' "$REVISION" >"$MARKER"
  SOURCE_READY=1
elif source_files_ready; then
  echo "IndexTTS2 运行时不是项目验证的版本：$INDEX_ROOT" >&2
  echo "保留 checkpoints 后移除该运行时目录，再重新执行安装。" >&2
  exit 1
fi

if [[ "$SOURCE_READY" == 0 ]]; then
  if [[ -d "$INDEX_ROOT" ]] &&
    [[ -n "$(find "$INDEX_ROOT" -mindepth 1 -maxdepth 1 ! -name checkpoints -print -quit)" ]]; then
    echo "IndexTTS2 目录包含未知文件：$INDEX_ROOT" >&2
    exit 1
  fi

  NEED_DOWNLOAD=1
  if [[ -f "$ARCHIVE" ]] &&
    [[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$SOURCE_SHA256" ]]; then
    NEED_DOWNLOAD=0
  fi
  if [[ "$NEED_DOWNLOAD" == 1 ]]; then
    echo "下载固定版本 IndexTTS2 源码（约 32 MB）..."
    asmr_download "$ROOT" "$SOURCE_URL" "$ARCHIVE" "$SOURCE_SHA256"
  fi
  ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA256" != "$SOURCE_SHA256" ]]; then
    echo "IndexTTS2 源码校验失败：$ACTUAL_SHA256" >&2
    exit 1
  fi

  rm -rf "$STAGING"
  mkdir -p "$STAGING"
  "$PYTHON" -m zipfile -e "$ARCHIVE" "$STAGING"
  SOURCE_ROOT="$(find "$STAGING" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  if [[ -z "$SOURCE_ROOT" || ! -f "$SOURCE_ROOT/pyproject.toml" ]]; then
    echo "IndexTTS2 源码包结构无效。" >&2
    exit 1
  fi
  mkdir -p "$INDEX_ROOT"
  cp -a "$SOURCE_ROOT"/. "$INDEX_ROOT"/
  rm -rf "$STAGING"
  printf '%s\n' "$REVISION" >"$MARKER"
fi

echo "安装 IndexTTS2 隔离运行时..."
SYNC_READY=0
while IFS= read -r index; do
  echo "使用软件源：$index"
  if (cd "$INDEX_ROOT" && "$ASMR_DUBBER_UV" sync --default-index "$index"); then
    SYNC_READY=1
    break
  fi
  echo "当前软件源失败，自动切换。" >&2
done < <(asmr_mirror_list "$ROOT" pypi_indexes)
if [[ "$SYNC_READY" != 1 ]]; then
  echo "IndexTTS2 依赖安装失败：所有软件源均不可用。" >&2
  exit 1
fi

INDEX_CLI="$INDEX_ROOT/.venv/bin/indextts2"
if [[ ! -x "$INDEX_CLI" ]]; then
  echo "IndexTTS2 CLI 安装失败：$INDEX_CLI" >&2
  exit 1
fi

echo "通过 ModelScope 下载或续传 IndexTTS2 模型（约 11 GB）..."
if ! USE_MODELSCOPE=true \
  MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-4}" \
  "$INDEX_CLI" download --source modelscope --model-dir "$MODEL_DIR"; then
  echo "ModelScope 不可用，改用 Hugging Face 镜像。" >&2
  HF_ENDPOINT="$(asmr_mirror_list "$ROOT" huggingface_endpoints | head -n 1)" \
    "$INDEX_CLI" download --source auto --model-dir "$MODEL_DIR"
fi

DEVICE="$("$INDEX_ROOT/.venv/bin/python" -c \
  "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
"$INDEX_CLI" check --model-dir "$MODEL_DIR" --device "$DEVICE"

echo
echo "IndexTTS2 安装完成。"
echo "模型目录：$MODEL_DIR"
