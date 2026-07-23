#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/portable-runtime.sh
source "$ROOT/scripts/portable-runtime.sh"
asmr_init_portable_environment "$ROOT"

VERSION="v0.8.21"
RUNTIME_ROOT="$ASMR_DUBBER_HOME/runtimes/crispasr"
MODEL_ROOT="$ASMR_DUBBER_HOME/models/parakeet"
DOWNLOAD_ROOT="$ASMR_DUBBER_HOME/cache/downloads"
PYTHON="$ASMR_DUBBER_VENV/bin/python"
mkdir -p "$RUNTIME_ROOT/bin" "$MODEL_ROOT" "$DOWNLOAD_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "缺少项目私有 Python；请先运行 bash scripts/linux/setup.sh。" >&2
  exit 1
fi

ASSET="crispasr-linux-x86_64.tar.gz"
EXPECTED="55d48357052c6d9376ad6548c877f9fb1e0a79728ae5e24dfc84b29405d22434"
if command -v nvidia-smi >/dev/null 2>&1; then
  ASSET="crispasr-linux-x86_64-cuda13.tar.gz"
  EXPECTED="883edc02ed3666af9e76b26ca29e2fc5db0ce48f97e4b0ef575d482a3619c74d"
fi
ARCHIVE="$DOWNLOAD_ROOT/$ASSET"
URL="https://github.com/CrispStrobe/CrispASR/releases/download/$VERSION/$ASSET"

download_archive() {
  local attempt
  for attempt in 1 2; do
    if [[ -f "$ARCHIVE" ]] &&
      [[ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" == "$EXPECTED" ]]; then
      return 0
    fi
    rm -f "$ARCHIVE"
    if ! curl -L --fail --retry 5 --retry-all-errors -C - \
      -o "$ARCHIVE.partial" "$URL"; then
      rm -f "$ARCHIVE.partial"
      curl -L --fail --retry 5 --retry-all-errors \
        -o "$ARCHIVE.partial" "$URL"
    fi
    mv "$ARCHIVE.partial" "$ARCHIVE"
    if [[ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" == "$EXPECTED" ]]; then
      return 0
    fi
    rm -f "$ARCHIVE"
    echo "CrispASR SHA256 校验失败，将完整重试一次。" >&2
  done
  echo "CrispASR SHA256 校验失败。" >&2
  return 1
}
download_archive

STAGING="$ASMR_DUBBER_HOME/temp/crispasr-install"
rm -rf "$STAGING"
mkdir -p "$STAGING"
tar -xzf "$ARCHIVE" -C "$STAGING"
EXECUTABLE="$(find "$STAGING" -type f -name crispasr -perm -u+x | head -n 1)"
[[ -n "$EXECUTABLE" ]] || { echo "压缩包中找不到 crispasr。" >&2; exit 1; }
rm -rf "$RUNTIME_ROOT/bin"
mkdir -p "$RUNTIME_ROOT/bin"
cp -a "$(dirname "$EXECUTABLE")/." "$RUNTIME_ROOT/bin/"
rm -rf "$STAGING"

"$PYTHON" "$ROOT/scripts/download_hf_file.py" \
  --repo cstr/parakeet-ctc-1.1b-ja-GGUF \
  --filename parakeet-ctc-1.1b-ja-f16.gguf \
  --revision 7ccb2922f63cefe7c0d2735527c69aa46c05ceb9 \
  --destination "$MODEL_ROOT/parakeet-ctc-1.1b-ja-f16.gguf" \
  --minimum-bytes 2000000000
"$PYTHON" "$ROOT/scripts/download_hf_file.py" \
  --repo cstr/parakeet-tdt-0.6b-ja-GGUF \
  --filename parakeet-tdt-0.6b-ja.gguf \
  --revision 65341fce2b46d25ea51593b1f771ed9a73cf7108 \
  --destination "$MODEL_ROOT/parakeet-tdt-0.6b-ja.gguf" \
  --minimum-bytes 1000000000

"$RUNTIME_ROOT/bin/crispasr" --version
echo "Parakeet 已就绪；全部文件均位于 .asmr-dubber。"
