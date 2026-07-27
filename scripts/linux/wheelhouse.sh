#!/usr/bin/env bash

ASMR_WHEELHOUSE_RESULT=""

asmr_named_checksum() {
  local path="$1" filename="$2" hash name
  while read -r hash name; do
    name="${name#\*}"
    if [[ "$name" == "$filename" && "$hash" =~ ^[0-9a-fA-F]{64}$ ]]; then
      printf '%s\n' "${hash,,}"
      return 0
    fi
  done <"$path"
  return 1
}

asmr_prepare_wheelhouse() {
  local root="$1" archive_name="$2" archive_mirror="$3" checksum_mirror="$4"
  local download_root="$ASMR_DUBBER_HOME/cache/downloads"
  local archive="$download_root/$archive_name" checksum="$download_root/$archive_name.sha256"
  local local_archive="$root/model-packs/$archive_name"
  local local_checksum="$root/model-packs/$archive_name.sha256"
  local extract_root="$ASMR_DUBBER_HOME/cache/wheelhouses/${archive_name%.tar.gz}"
  local expected="" url actual staging
  ASMR_WHEELHOUSE_RESULT=""
  mkdir -p "$download_root" "$(dirname "$extract_root")"

  if [[ -f "$local_checksum" ]]; then
    expected="$(asmr_named_checksum "$local_checksum" "$archive_name" || true)"
  fi
  if [[ -z "$expected" ]]; then
    while IFS= read -r url; do
      [[ -n "$url" ]] || continue
      if asmr_download "$root" "$url" "$checksum"; then
        expected="$(asmr_named_checksum "$checksum" "$archive_name" || true)"
        [[ -n "$expected" ]] && break
      fi
    done < <(asmr_mirror_list "$root" "$checksum_mirror")
  fi
  # No published checksum means this optional mirror bundle has not been
  # uploaded yet. The caller may use the configured regional package index.
  [[ -n "$expected" ]] || return 1

  if [[ -f "$local_archive" ]]; then
    actual="$(sha256sum "$local_archive" | cut -d' ' -f1)"
    if [[ "$actual" != "$expected" ]]; then
      echo "本地 wheelhouse 的 SHA-256 不匹配：$local_archive" >&2
      return 2
    fi
    cp -f "$local_archive" "$archive"
  fi
  if [[ ! -f "$archive" ]] || \
    [[ "$(sha256sum "$archive" | cut -d' ' -f1)" != "$expected" ]]; then
    local ready=0
    while IFS= read -r url; do
      [[ -n "$url" ]] || continue
      if asmr_download "$root" "$url" "$archive" "$expected"; then
        ready=1
        break
      fi
    done < <(asmr_mirror_list "$root" "$archive_mirror")
    if [[ "$ready" != 1 ]]; then
      echo "已发布 wheelhouse 校验文件，但压缩包下载失败；断点已保留。" >&2
      return 2
    fi
  fi

  if [[ -f "$extract_root/.archive-sha256" ]] && \
    [[ "$(tr -d '\r\n' <"$extract_root/.archive-sha256")" == "$expected" ]] && \
    find "$extract_root" -type f -name '*.whl' -print -quit | grep -q .; then
    ASMR_WHEELHOUSE_RESULT="$extract_root"
    return 0
  fi
  staging="$extract_root.staging"
  rm -rf "$staging"
  mkdir -p "$staging"
  tar -xzf "$archive" -C "$staging"
  if ! find "$staging" -type f -name '*.whl' -print -quit | grep -q .; then
    echo "wheelhouse 压缩包中没有 wheel 文件。" >&2
    rm -rf "$staging"
    return 2
  fi
  printf '%s\n' "$expected" >"$staging/.archive-sha256"
  rm -rf "$extract_root"
  mv "$staging" "$extract_root"
  ASMR_WHEELHOUSE_RESULT="$extract_root"
}
