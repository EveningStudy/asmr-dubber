#!/usr/bin/env bash

asmr_install_python_runtime() {
  local root="$1" version="$2" build_date="$3" expected="$4" mirror_name="$5"
  local runtime_name="cpython-$version-linux-x86_64-gnu"
  local runtime_root="$UV_PYTHON_INSTALL_DIR/$runtime_name"
  local python="$runtime_root/bin/python3"
  local archive_name archive archive_ready url staging staged_python staged_root actual
  if [[ -x "$python" ]] && "$python" -c \
    "import sys; assert sys.version.split()[0] == '$version'"; then
    return 0
  fi

  archive_name="cpython-$version+$build_date-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
  archive="$ASMR_DUBBER_HOME/cache/downloads/$archive_name"
  mkdir -p "$(dirname "$archive")" "$UV_PYTHON_INSTALL_DIR"
  archive_ready=0
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    if asmr_download "$root" "$url" "$archive" "$expected"; then
      archive_ready=1
      break
    fi
  done < <(asmr_mirror_list "$root" "$mirror_name")
  if [[ "$archive_ready" != 1 ]]; then
    echo "Python $version 的 ModelScope 运行时下载失败；断点文件已保留。" >&2
    echo "请按 docs/MODELSCOPE_UPLOADS.md 上传对应文件。" >&2
    return 1
  fi
  actual="$(sha256sum "$archive" | cut -d' ' -f1)"
  [[ "$actual" == "$expected" ]] || {
    echo "Python $version 运行时 SHA-256 校验失败。" >&2
    return 1
  }

  staging="$ASMR_DUBBER_HOME/t/py-${version//./}"
  rm -rf "$staging"
  mkdir -p "$staging"
  tar -xzf "$archive" -C "$staging"
  staged_python="$(find "$staging" -type f -path '*/bin/python3' -perm -u+x | head -n 1)"
  if [[ -z "$staged_python" ]]; then
    echo "Python $version 压缩包结构无效：找不到 bin/python3。" >&2
    return 1
  fi
  staged_root="$(dirname "$(dirname "$staged_python")")"
  rm -rf "$runtime_root"
  mv "$staged_root" "$runtime_root"
  rm -rf "$staging"
  "$python" -c "import sys; assert sys.version.split()[0] == '$version'"
}
