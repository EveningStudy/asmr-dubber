#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/portable-runtime.sh"
asmr_init_portable_environment "$ROOT"
if [[ ! -x "$ASMR_DUBBER_VENV/bin/asmr-dubber" ]]; then
  echo "尚未安装。请先运行：bash $ROOT/scripts/linux/setup.sh" >&2
  exit 1
fi

DATA_ROOT="$ASMR_DUBBER_HOME"
RUNTIME_LIBS="$DATA_ROOT/runtimes/ffmpeg-libs:$DATA_ROOT/runtimes/python-libs"
PYAV_LIBS="$ASMR_DUBBER_VENV/lib/python3.12/site-packages/av.libs"
export LD_LIBRARY_PATH="$RUNTIME_LIBS:$PYAV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT"
source "$ROOT/scripts/network-bridge.sh"
asmr_prepare_network
trap asmr_cleanup_network EXIT INT TERM
"$ASMR_DUBBER_VENV/bin/asmr-dubber" "$@"
