#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/portable-runtime.sh
source "$ROOT/scripts/portable-runtime.sh"
asmr_init_portable_environment "$ROOT"
source "$ROOT/scripts/mirrors.sh"
asmr_apply_mirror_environment "$ROOT"

VERSION="v0.8.21"
RUNTIME_ROOT="$ASMR_DUBBER_HOME/runtimes/crispasr"
MODEL_ROOT="$ASMR_DUBBER_HOME/models/parakeet"
DOWNLOAD_ROOT="$ASMR_DUBBER_HOME/cache/downloads"
CRISP_CACHE="$ASMR_DUBBER_HOME/cache/crispasr"
PYTHON="$ASMR_DUBBER_VENV/bin/python"
mkdir -p "$RUNTIME_ROOT/bin" "$MODEL_ROOT" "$DOWNLOAD_ROOT" "$CRISP_CACHE"

if [[ ! -x "$PYTHON" ]]; then
  echo "缺少 Python 运行环境；请先运行 bash scripts/linux/setup.sh。" >&2
  exit 1
fi

OPENBLAS_ROOT="$ASMR_DUBBER_HOME/runtimes/openblas"
OPENBLAS_LIB="$OPENBLAS_ROOT/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0"
OPENBLAS_ARCHIVE="$DOWNLOAD_ROOT/libopenblas0-pthread_0.3.26+ds-1ubuntu0.1_amd64.deb"
OPENBLAS_SHA256="7dc3b4384c02aecb87eb8b70fa26c5843a08af242f4638aa4b36922bdc4f5b04"
extract_openblas_deb() {
  local destination="$1" archive="$2"
  if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb -x "$archive" "$destination"
    return
  fi
  # dpkg-deb is not present on every supported x86_64 distribution.  A .deb
  # is an ar archive; extract its data member with the standard binutils/tar
  # tools so the application still keeps the library private to its folder.
  if ! command -v ar >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    echo "缺少 dpkg-deb、ar 或 tar，无法解包便携 OpenBLAS。" >&2
    return 1
  fi
  local staging="$destination/.deb-extract"
  rm -rf "$staging"
  mkdir -p "$staging"
  ar x "$archive" --output "$staging"
  local data_archive
  data_archive="$(find "$staging" -maxdepth 1 -type f -name 'data.tar.*' -print -quit)"
  if [[ -z "$data_archive" ]]; then
    echo "OpenBLAS .deb 缺少 data.tar.* 内容。" >&2
    rm -rf "$staging"
    return 1
  fi
  tar -xf "$data_archive" -C "$destination"
  rm -rf "$staging"
}
if [[ ! -f "$OPENBLAS_LIB" ]]; then
  OPENBLAS_READY=0
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    if asmr_download "$ROOT" "$url" "$OPENBLAS_ARCHIVE" "$OPENBLAS_SHA256"; then
      OPENBLAS_READY=1
      break
    fi
  done < <(asmr_mirror_list "$ROOT" openblas_linux_deb_archives)
  if [[ "$OPENBLAS_READY" != 1 ]]; then
    echo "OpenBLAS 运行库下载失败；Parakeet 无法启动。" >&2
    exit 1
  fi
  rm -rf "$OPENBLAS_ROOT"
  mkdir -p "$OPENBLAS_ROOT"
  extract_openblas_deb "$OPENBLAS_ROOT" "$OPENBLAS_ARCHIVE"
fi

NVIDIA_LIBS="$ASMR_DUBBER_VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$(dirname "$OPENBLAS_LIB"):$NVIDIA_LIBS:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXPECTED_11B="34dd3128275c9bca2b4296f53c5f831feb258fcf3fdd28c29c0dc2d2f7d5ede7"
EXPECTED_06B="374eb0132eebaec4df77a9631cbbeb03790be48a4a517f6cc8e8bdb38fe9a584"
EXPECTED_PUNCTUATION="faf4a43e3135bc307a66194685af00f756e6f4c28c7d9e2dd8f3517cddca5c45"
EXPECTED_VAD="2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"
MODEL_11B="$MODEL_ROOT/parakeet-ctc-1.1b-ja-f16.gguf"
MODEL_06B="$MODEL_ROOT/parakeet-tdt-0.6b-ja.gguf"
PUNCTUATION_MODEL="$CRISP_CACHE/fireredpunc-q4_k.gguf"
VAD_MODEL="$CRISP_CACHE/ggml-silero-v6.2.0.bin"
if [[ -x "$RUNTIME_ROOT/bin/crispasr" ]] \
  && [[ -f "$MODEL_11B" ]] \
  && [[ -f "$MODEL_06B" ]] \
  && [[ -f "$PUNCTUATION_MODEL" ]] \
  && [[ -f "$VAD_MODEL" ]] \
  && [[ "$(sha256sum "$MODEL_11B" | cut -d' ' -f1)" == "$EXPECTED_11B" ]] \
  && [[ "$(sha256sum "$MODEL_06B" | cut -d' ' -f1)" == "$EXPECTED_06B" ]] \
  && [[ "$(sha256sum "$PUNCTUATION_MODEL" | cut -d' ' -f1)" == "$EXPECTED_PUNCTUATION" ]] \
  && [[ "$(sha256sum "$VAD_MODEL" | cut -d' ' -f1)" == "$EXPECTED_VAD" ]] \
  && "$RUNTIME_ROOT/bin/crispasr" --version; then
  echo "Parakeet 本地运行时和两款模型已完整，无需联网下载。"
  exit 0
fi

ASSET="crispasr-linux-x86_64.tar.gz"
EXPECTED="55d48357052c6d9376ad6548c877f9fb1e0a79728ae5e24dfc84b29405d22434"
if command -v nvidia-smi >/dev/null 2>&1 \
  && [[ "${ASMR_DUBBER_PARAKEET_FORCE_CPU:-0}" != 1 ]] \
  && [[ -f "$NVIDIA_LIBS/libcudart.so.13" ]] \
  && [[ -f "$NVIDIA_LIBS/libcublas.so.13" ]]; then
  ASSET="crispasr-linux-x86_64-cuda13.tar.gz"
  EXPECTED="883edc02ed3666af9e76b26ca29e2fc5db0ce48f97e4b0ef575d482a3619c74d"
fi
ARCHIVE="$DOWNLOAD_ROOT/$ASSET"
MIRROR_NAME="crispasr_linux_cpu_archives"
if [[ "$ASSET" == *cuda13* ]]; then
  MIRROR_NAME="crispasr_linux_cuda_archives"
fi

download_archive() {
  local attempt
  for attempt in 1 2; do
    if [[ -f "$ARCHIVE" ]] &&
      [[ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" == "$EXPECTED" ]]; then
      return 0
    fi
    local ready=0 url
    while IFS= read -r url; do
      [[ -n "$url" ]] || continue
      if asmr_download "$ROOT" "$url" "$ARCHIVE" "$EXPECTED"; then
        ready=1
        break
      fi
    done < <(asmr_mirror_list "$ROOT" "$MIRROR_NAME")
    [[ "$ready" == 1 ]] || return 1
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

download_parakeet_model() {
  local mirror_name="$1" destination="$2" expected="$3"
  local repo="$4" filename="$5" revision="$6" minimum="$7" url
  if [[ -f "$destination" ]] && \
    [[ "$(sha256sum "$destination" | cut -d' ' -f1)" == "$expected" ]]; then
    return 0
  fi
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    if asmr_download "$ROOT" "$url" "$destination" "$expected"; then
      return 0
    fi
  done < <(asmr_mirror_list "$ROOT" "$mirror_name")
  if asmr_external_downloads_allowed "$ROOT"; then
    "$PYTHON" "$ROOT/scripts/download_hf_file.py" \
      --repo "$repo" --filename "$filename" --revision "$revision" \
      --destination "$destination" --minimum-bytes "$minimum" \
      --sha256 "$expected" \
      --endpoints "$(asmr_mirror_list "$ROOT" huggingface_endpoints | paste -sd ';' -)"
    return
  fi
  echo "$filename 的 ModelScope 下载失败；断点文件已保留。" >&2
  echo "请上传镜像文件，安装器不会自动消耗 Hugging Face 流量。" >&2
  return 1
}

download_parakeet_model \
  parakeet_11b_model_files \
  "$MODEL_ROOT/parakeet-ctc-1.1b-ja-f16.gguf" \
  "$EXPECTED_11B" \
  cstr/parakeet-ctc-1.1b-ja-GGUF \
  parakeet-ctc-1.1b-ja-f16.gguf \
  7ccb2922f63cefe7c0d2735527c69aa46c05ceb9 \
  2000000000
download_parakeet_model \
  parakeet_06b_model_files \
  "$MODEL_ROOT/parakeet-tdt-0.6b-ja.gguf" \
  "$EXPECTED_06B" \
  cstr/parakeet-tdt-0.6b-ja-GGUF \
  parakeet-tdt-0.6b-ja.gguf \
  65341fce2b46d25ea51593b1f771ed9a73cf7108 \
  1000000000
download_parakeet_model \
  crispasr_punctuation_model_files \
  "$PUNCTUATION_MODEL" \
  "$EXPECTED_PUNCTUATION" \
  cstr/fireredpunc-GGUF \
  fireredpunc-q4_k.gguf \
  main \
  50000000
download_parakeet_model \
  crispasr_vad_model_files \
  "$VAD_MODEL" \
  "$EXPECTED_VAD" \
  ggml-org/whisper-vad \
  ggml-silero-v6.2.0.bin \
  main \
  800000

"$RUNTIME_ROOT/bin/crispasr" --version
echo "Parakeet 已就绪；全部文件均位于 .asmr-dubber。"
