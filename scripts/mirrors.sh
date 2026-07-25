#!/usr/bin/env bash

asmr_mirror_list() {
  local root="$1" name="$2"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$root/mirrors.json" "$name" <<'PY'
import json
import sys
from urllib.parse import urlparse

path, name = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    values = json.load(handle).get(name, [])
fallbacks = {
    "pypi_indexes": ["https://pypi.org/simple"],
    "huggingface_endpoints": ["https://huggingface.co"],
    "pytorch_indexes": ["https://download.pytorch.org/whl/cu130"],
    "github_proxy_prefixes": [""],
    "uv_installers_linux": ["https://astral.sh/uv/0.11.30/install.sh"],
    "python_install_mirrors": [
        "https://releases.astral.sh/github/python-build-standalone/releases/download",
        "https://github.com/astral-sh/python-build-standalone/releases/download"
    ],
    "indextts2_source_archives": [
        "https://github.com/index-tts/index-tts/archive/13495845e3028f0bb6ca1462ad22aa0e76349e40.zip"
    ],
}
seen = set()
for value in [*values, *fallbacks.get(name, [])]:
    if not isinstance(value, str):
        continue
    value = value.strip()
    if value == "" and name == "github_proxy_prefixes":
        if value not in seen:
            print()
            seen.add(value)
        continue
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.netloc:
        value = value.rstrip("/") if name != "github_proxy_prefixes" else value
        if value not in seen:
            print(value)
            seen.add(value)
PY
    return
  fi
  case "$name" in
    pypi_indexes)
      printf '%s\n' \
        "https://pypi.tuna.tsinghua.edu.cn/simple" \
        "https://mirrors.aliyun.com/pypi/simple" \
        "https://pypi.org/simple"
      ;;
    huggingface_endpoints)
      printf '%s\n' "https://hf-mirror.com" "https://huggingface.co"
      ;;
    python_install_mirrors)
      printf '%s\n' \
        "https://releases.astral.sh/github/python-build-standalone/releases/download" \
        "https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download" \
        "https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download" \
        "https://github.com/astral-sh/python-build-standalone/releases/download"
      ;;
    github_proxy_prefixes)
      printf '%s\n' "https://ghfast.top/" "https://ghproxy.net/" ""
      ;;
    uv_installers_linux)
      printf '%s\n' "https://astral.sh/uv/0.11.30/install.sh"
      ;;
  esac
}

asmr_download_candidates() {
  local root="$1" url="$2" prefix
  if [[ "$url" != https://github.com/* ]]; then
    printf '%s\n' "$url"
    return
  fi
  while IFS= read -r prefix; do
    if [[ -n "$prefix" ]]; then
      printf '%s\n' "${prefix%/}/$url"
    else
      printf '%s\n' "$url"
    fi
  done < <(asmr_mirror_list "$root" github_proxy_prefixes)
}

asmr_download() {
  local root="$1" url="$2" destination="$3" expected="${4:-}" candidate actual
  while IFS= read -r candidate; do
    echo "尝试下载：$candidate"
    if curl -fL --retry 3 --retry-all-errors --connect-timeout 20 \
      -C - "$candidate" -o "$destination.partial"; then
      mv -f "$destination.partial" "$destination"
      if [[ -z "$expected" ]]; then
        return 0
      fi
      actual="$(sha256sum "$destination" | cut -d' ' -f1)"
      if [[ "$actual" == "$expected" ]]; then
        return 0
      fi
      rm -f "$destination"
      echo "当前下载源返回的文件校验失败，自动切换。" >&2
      continue
    fi
    rm -f "$destination.partial"
    echo "当前下载源失败，自动切换。" >&2
  done < <(asmr_download_candidates "$root" "$url")
  echo "所有下载源均失败：$url" >&2
  return 1
}

asmr_apply_mirror_environment() {
  local root="$1" endpoints
  endpoints="$(asmr_mirror_list "$root" huggingface_endpoints | paste -sd ';' -)"
  export ASMR_DUBBER_HF_ENDPOINTS="$endpoints"
  if [[ -z "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="${endpoints%%;*}"
  fi
}
