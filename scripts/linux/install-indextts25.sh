#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
RUNTIME_ROOT="$DATA_ROOT/runtimes/index-tts-2.5"
MODEL_DIR="$RUNTIME_ROOT/checkpoints"
REVISION="ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"
SOURCE_SHA256="${INDEXTTS25_SOURCE_SHA256:-53ad18d03cae44d8daf29d665c889e4dfeb5e14f6a2bafb6b6a76eacfbf75440}"
SOURCE_URL="${INDEXTTS25_SOURCE_URL:-}"
MARKER="$RUNTIME_ROOT/.asmr-source-revision"
DOWNLOAD_ROOT="$DATA_ROOT/cache/downloads"
ARCHIVE="$DOWNLOAD_ROOT/index-tts-$REVISION.zip"
STAGING="$RUNTIME_ROOT.staging"
APP_PYTHON="$ASMR_DUBBER_VENV/bin/python"

if [[ "${ASMR_DUBBER_REPAIR_CHILD:-}" != "1" ]]; then
  "$APP_PYTHON" -m asmr_dubber.installer_transaction --runtime "$RUNTIME_ROOT" -- bash "${BASH_SOURCE[0]}" "$@"
  exit $?
fi

if [[ ! -x "$ASMR_DUBBER_UV" || ! -x "$APP_PYTHON" ]]; then
  echo "缺少应用运行时。请先运行 bash $ROOT/scripts/linux/setup.sh 基础。" >&2
  exit 1
fi
mkdir -p "$DOWNLOAD_ROOT"

FORCE=0
for arg in "$@"; do
  if [[ "$arg" == "--force" ]]; then
    FORCE=1
  fi
done

indextts25_checkpoints_complete() {
  "$APP_PYTHON" - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path

from asmr_dubber.constants import INDEXTTS25_REQUIRED_DIRS, INDEXTTS25_REQUIRED_FILES

model_dir = Path(sys.argv[1])
files_ready = all((model_dir / name).is_file() for name in INDEXTTS25_REQUIRED_FILES)
dirs_ready = all((model_dir / name).is_dir() for name in INDEXTTS25_REQUIRED_DIRS)
raise SystemExit(0 if files_ready and dirs_ready else 1)
PY
}

if [[ "${ASMR_DUBBER_MODEL_PACKS_PREPARED:-}" != "1" ]]; then
  if [[ "$FORCE" != "1" ]] && indextts25_checkpoints_complete; then
    echo "IndexTTS-2.5 本地 checkpoints 已完整，跳过模型包下载。"
  elif "$APP_PYTHON" -m asmr_dubber.cli prepare-model-pack indextts2_5-checkpoints; then
    "$APP_PYTHON" -m asmr_dubber.cli import-model-packs --all \
      --pack-id indextts2_5-checkpoints
  else
    echo "IndexTTS-2.5 镜像模型包尚不可用或下载未完成，断点文件已保留。" >&2
    echo "请重试固定版本模型包；不会回退到未校验的浮动模型版本。" >&2
  fi
fi

source_files_ready() {
  [[ -f "$RUNTIME_ROOT/pyproject.toml" ]] &&
    [[ -f "$RUNTIME_ROOT/uv.lock" ]] &&
    [[ -d "$RUNTIME_ROOT/indextts" ]]
}

SOURCE_READY=0
if source_files_ready && [[ -f "$MARKER" ]] &&
  [[ "$(tr -d '\r\n' <"$MARKER")" == "$REVISION" ]]; then
  SOURCE_READY=1
elif source_files_ready && [[ -d "$RUNTIME_ROOT/.git" ]] &&
  [[ "$(git -C "$RUNTIME_ROOT" rev-parse HEAD 2>/dev/null || true)" == "$REVISION" ]]; then
  printf '%s\n' "$REVISION" >"$MARKER"
  SOURCE_READY=1
fi

if [[ "$SOURCE_READY" == 0 ]]; then
  if [[ -d "$RUNTIME_ROOT" ]]; then
    find "$RUNTIME_ROOT" -mindepth 1 -maxdepth 1 \
      ! -name checkpoints ! -name .venv ! -name user-state ! -name .repair-state.json \
      -exec rm -rf -- {} +
  fi
  NEED_DOWNLOAD=1
  if [[ -f "$ARCHIVE" ]] &&
    [[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$SOURCE_SHA256" ]]; then
    NEED_DOWNLOAD=0
  fi
  if [[ "$NEED_DOWNLOAD" == 1 ]]; then
    echo "正在下载固定版本 IndexTTS-2.5 源码（约 36 MB）..."
    SOURCE_DOWNLOADED=0
    if [[ -n "$SOURCE_URL" ]] &&
      asmr_download "$ROOT" "$SOURCE_URL" "$ARCHIVE" "$SOURCE_SHA256"; then
      SOURCE_DOWNLOADED=1
    fi
    if [[ "$SOURCE_DOWNLOADED" == 0 ]]; then
      while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        [[ "$candidate" == "$SOURCE_URL" ]] && continue
        if asmr_download "$ROOT" "$candidate" "$ARCHIVE" "$SOURCE_SHA256"; then
          SOURCE_DOWNLOADED=1
          break
        fi
      done < <(asmr_mirror_list "$ROOT" indextts25_source_archives)
    fi
    if [[ "$SOURCE_DOWNLOADED" == 0 ]] &&
      asmr_download "$ROOT" \
        "https://github.com/index-tts/index-tts/archive/$REVISION.zip" \
        "$ARCHIVE" "$SOURCE_SHA256"; then
      SOURCE_DOWNLOADED=1
    fi
    if [[ "$SOURCE_DOWNLOADED" == 0 ]]; then
      echo "IndexTTS-2.5 源码下载失败；断点文件已保留。" >&2
      exit 1
    fi
  fi
  ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA256" != "$SOURCE_SHA256" ]]; then
    echo "IndexTTS-2.5 源码校验失败：$ACTUAL_SHA256" >&2
    exit 1
  fi

  rm -rf "$STAGING"
  mkdir -p "$STAGING"
  "$APP_PYTHON" -m zipfile -e "$ARCHIVE" "$STAGING"
  SOURCE_FILE="$(find "$STAGING" -type f -name pyproject.toml -exec sh -c \
    'test -d "$(dirname "$1")/indextts" && printf "%s\n" "$1"' _ {} \; -quit)"
  SOURCE_ROOT="${SOURCE_FILE%/pyproject.toml}"
  if [[ -z "$SOURCE_FILE" || ! -d "$SOURCE_ROOT/indextts" ]]; then
    echo "IndexTTS-2.5 源码包结构无效。" >&2
    exit 1
  fi
  mkdir -p "$RUNTIME_ROOT"
  cp -a "$SOURCE_ROOT"/. "$RUNTIME_ROOT"/
  rm -rf "$STAGING"
  printf '%s\n' "$REVISION" >"$MARKER"
fi

echo "正在准备 IndexTTS-2.5 独立环境..."
asmr_install_python_runtime \
  "$ROOT" \
  "3.11.13" \
  "20251007" \
  "43bfc42529843ecd1d9c08c4a239ede348f96ff0acaef2ec24b28dc059f4f0c3" \
  "python311_linux_archives"
MANAGED_PYTHON="$UV_PYTHON_INSTALL_DIR/cpython-3.11.13-linux-x86_64-gnu/bin/python3"
INDEX_PYTHON="$RUNTIME_ROOT/.venv/bin/python"
RUNTIME_READY=0
if [[ -x "$INDEX_PYTHON" ]] &&
  (cd "$RUNTIME_ROOT" && "$INDEX_PYTHON" -c \
    'from indextts.infer_v2_5 import IndexTTS2'); then
  RUNTIME_READY=1
fi
if [[ "$RUNTIME_READY" == 1 ]]; then
  echo "IndexTTS-2.5 Python/CUDA 依赖已就绪。"
else
  SYNC_READY=0
  if asmr_prepare_wheelhouse \
    "$ROOT" \
    "ASMR-Dubber-IndexTTS25-Wheelhouse-v1.0.0.tar.gz" \
    "indextts25_wheelhouse_archives_linux" \
    "indextts25_wheelhouse_checksums_linux"; then
    echo "使用 ModelScope IndexTTS-2.5 wheelhouse：$ASMR_WHEELHOUSE_RESULT"
    REQUIREMENTS="$ASMR_WHEELHOUSE_RESULT/requirements.txt"
    if [[ ! -f "$REQUIREMENTS" ]]; then
      echo "IndexTTS-2.5 wheelhouse 缺少 requirements.txt。" >&2
      exit 1
    fi
    if "$ASMR_DUBBER_UV" venv --python "$MANAGED_PYTHON" "$RUNTIME_ROOT/.venv" && \
      "$ASMR_DUBBER_UV" pip install --python "$RUNTIME_ROOT/.venv/bin/python" \
        --offline --no-index --find-links "$ASMR_WHEELHOUSE_RESULT" \
        --requirement "$REQUIREMENTS"; then
      SYNC_READY=1
    fi
  else
    WHEELHOUSE_STATUS=$?
    if [[ "$WHEELHOUSE_STATUS" == 2 ]]; then
      echo "IndexTTS-2.5 wheelhouse 已发布但不完整，拒绝使用损坏的依赖包。" >&2
      exit 1
    fi
  fi
  if [[ "$SYNC_READY" == 0 ]]; then
    while IFS= read -r index; do
      [[ -n "$index" ]] || continue
      echo "使用软件源：$index"
      if (cd "$RUNTIME_ROOT" && "$ASMR_DUBBER_UV" sync --no-dev --extra torch_compile \
        --python "$MANAGED_PYTHON" --default-index "$index"); then
        SYNC_READY=1
        break
      fi
      echo "当前软件源失败，自动切换。" >&2
    done < <(asmr_mirror_list "$ROOT" pypi_indexes)
  fi
  if [[ "$SYNC_READY" != 1 || ! -x "$INDEX_PYTHON" ]]; then
    echo "IndexTTS-2.5 依赖安装失败：所有软件源均不可用。" >&2
    exit 1
  fi
fi

if indextts25_checkpoints_complete; then
  echo "IndexTTS-2.5 本地 checkpoints 已完整，无需联网下载。"
else
  echo "IndexTTS-2.5 模型不完整。请重试固定 SHA-256 模型包下载/导入；断点保留，不使用浮动模型版本。" >&2
  exit 1
fi

DEVICE="$("$INDEX_PYTHON" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
(cd "$RUNTIME_ROOT" && "$INDEX_PYTHON" -c \
  "import torch; from indextts.infer_v2_5 import IndexTTS2; assert '$DEVICE' == 'cpu' or torch.cuda.is_available()")

echo
echo "IndexTTS-2.5 安装完成。"
echo "模型目录：$MODEL_DIR"
