#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "$ROOT/.asmr-dubber/venv/bin/asmr-dubber" ]]; then
  echo "尚未安装。请先运行：bash $ROOT/scripts/linux/setup.sh" >&2
  exit 1
fi
cd "$ROOT"
exec bash "$ROOT/scripts/linux/run-cli.sh" ui "$@"
