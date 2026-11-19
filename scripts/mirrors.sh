#!/usr/bin/env bash

asmr_mirror_list() {
  local root="$1" name="$2"
  local modelscope_base="https://modelscope.cn/models/EveningStudyW/ASMR-Dubber-Portable-Mirror/resolve/master"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$root/mirrors.json" "$name" <<'PY'
import json
import os
import sys
from urllib.parse import urlparse

path, name = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
values = []
modelscope = payload.get("modelscope", {})
base = modelscope.get("base_url", "") if isinstance(modelscope, dict) else ""
artifact_map = payload.get("modelscope_artifacts", {})
artifacts = artifact_map.get(name, []) if isinstance(artifact_map, dict) else []
for value in artifacts:
    if isinstance(value, str) and value.strip():
        value = value.strip()
        values.append(value if value.startswith("https://") else f"{base.rstrip('/')}/{value.lstrip('/')}")
configured = payload.get(name, [])
if isinstance(configured, list):
    values.extend(configured)
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
    "indextts25_source_archives": [
        "https://github.com/index-tts/index-tts/archive/ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c.zip"
    ],
}
raw_allow = os.getenv("ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS")
policy = payload.get("download_policy")
if raw_allow is not None:
    allow_external = raw_allow.strip().lower() in {"1", "true", "yes", "on"}
elif policy is None:
    allow_external = True
elif isinstance(policy, dict):
    allow_external = bool(policy.get("allow_external", False))
else:
    allow_external = False
external_hosts = {
    "github.com", "raw.githubusercontent.com", "huggingface.co", "hf.co",
    "hf-mirror.com", "ghfast.top", "ghproxy.net", "download.pytorch.org",
    "pypi.org", "astral.sh", "releases.astral.sh", "python.org", "www.python.org",
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
        host = (parsed.hostname or "").lower()
        if not allow_external and any(
            host == external or host.endswith("." + external) for external in external_hosts
        ):
            continue
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
        "https://mirrors.aliyun.com/pypi/simple"
      if asmr_external_downloads_allowed "$root"; then
        printf '%s\n' "https://pypi.org/simple"
      fi
      ;;
    huggingface_endpoints)
      if asmr_external_downloads_allowed "$root"; then
        printf '%s\n' "https://hf-mirror.com" "https://huggingface.co"
      fi
      ;;
    python_install_mirrors)
      printf '%s\n' "$modelscope_base/artifacts/python-build-standalone/releases/download"
      if asmr_external_downloads_allowed "$root"; then
        printf '%s\n' \
          "https://releases.astral.sh/github/python-build-standalone/releases/download" \
          "https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download" \
          "https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download" \
          "https://github.com/astral-sh/python-build-standalone/releases/download"
      fi
      ;;
    python312_linux_archives)
      printf '%s\n' \
        "$modelscope_base/artifacts/python-build-standalone/releases/download/20260718/cpython-3.12.13%2B20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
      ;;
    python311_linux_archives)
      printf '%s\n' \
        "$modelscope_base/artifacts/python-build-standalone/releases/download/20251007/cpython-3.11.13%2B20251007-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
      ;;
    github_proxy_prefixes)
      if asmr_external_downloads_allowed "$root"; then
        printf '%s\n' "https://ghfast.top/" "https://ghproxy.net/" ""
      fi
      ;;
    uv_installers_linux)
      printf '%s\n' "$modelscope_base/artifacts/bootstrap/uv/0.11.30/install.sh"
      ;;
    uv_archives_linux)
      printf '%s\n' \
        "$modelscope_base/artifacts/bootstrap/uv/0.11.30/uv-x86_64-unknown-linux-gnu.tar.gz"
      ;;
    crispasr_linux_cpu_archives)
      printf '%s\n' \
        "$modelscope_base/artifacts/runtimes/crispasr/v0.8.21/crispasr-linux-x86_64.tar.gz"
      ;;
    crispasr_linux_cuda_archives)
      printf '%s\n' \
        "$modelscope_base/artifacts/runtimes/crispasr/v0.8.21/crispasr-linux-x86_64-cuda13.tar.gz"
      ;;
    openblas_linux_deb_archives)
      printf '%s\n' \
        "$modelscope_base/artifacts/runtimes/openblas/0.3.26/libopenblas0-pthread_0.3.26%2Bds-1ubuntu0.1_amd64.deb"
      ;;
    parakeet_11b_model_files)
      printf '%s\n' "$modelscope_base/artifacts/models/parakeet/parakeet-ctc-1.1b-ja-f16.gguf"
      ;;
    parakeet_06b_model_files)
      printf '%s\n' "$modelscope_base/artifacts/models/parakeet/parakeet-tdt-0.6b-ja.gguf"
      ;;
    crispasr_punctuation_model_files)
      printf '%s\n' "$modelscope_base/artifacts/models/crispasr/fireredpunc-q4_k.gguf"
      ;;
    crispasr_vad_model_files)
      printf '%s\n' "$modelscope_base/artifacts/models/crispasr/ggml-silero-v6.2.0.bin"
      ;;
  esac
}

asmr_external_downloads_allowed() {
  local root="$1" raw="${ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS:-}"
  if [[ -n "${ASMR_DUBBER_ALLOW_EXTERNAL_DOWNLOADS+x}" ]]; then
    case "${raw,,}" in
      1|true|yes|on) return 0 ;;
      *) return 1 ;;
    esac
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$root/mirrors.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
policy = payload.get("download_policy")
if policy is None:
    raise SystemExit(0)
raise SystemExit(0 if isinstance(policy, dict) and bool(policy.get("allow_external")) else 1)
PY
    return $?
  fi
  if grep -q '"download_policy"' "$root/mirrors.json" 2>/dev/null; then
    grep -Eq '"allow_external"[[:space:]]*:[[:space:]]*true' \
      "$root/mirrors.json" 2>/dev/null
    return $?
  fi
  return 0
}

asmr_url_allowed() {
  local root="$1" url="$2" host
  [[ "$url" == https://* ]] || return 1
  if asmr_external_downloads_allowed "$root"; then
    return 0
  fi
  host="${url#https://}"
  host="${host%%/*}"
  host="${host%%:*}"
  case "${host,,}" in
    github.com|*.github.com|raw.githubusercontent.com|*.raw.githubusercontent.com|\
      huggingface.co|*.huggingface.co|hf.co|*.hf.co|hf-mirror.com|*.hf-mirror.com|\
      ghfast.top|*.ghfast.top|ghproxy.net|*.ghproxy.net|download.pytorch.org|\
      *.download.pytorch.org|pypi.org|*.pypi.org|astral.sh|*.astral.sh|\
      releases.astral.sh|*.releases.astral.sh|python.org|*.python.org)
      return 1
      ;;
    *) return 0 ;;
  esac
}

asmr_download_candidates() {
  local root="$1" url="$2" prefix
  asmr_url_allowed "$root" "$url" || return
  if [[ "$url" != https://github.com/* ]]; then
    printf '%s\n' "$url"
    return
  fi
  if ! asmr_external_downloads_allowed "$root"; then
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

asmr_find_local_artifact() {
  local url="$1" expected="$2" filename roots candidate actual
  filename="${url%%\?*}"
  filename="${filename##*/}"
  filename="${filename//%2B/+}"
  filename="${filename//%2b/+}"
  [[ -n "$filename" ]] || return 1
  roots="${ASMR_DUBBER_LOCAL_CACHE_ROOTS:-}"
  [[ -n "$roots" ]] || return 1
  while IFS= read -r candidate; do
    [[ -f "$candidate" ]] || continue
    if [[ -n "$expected" ]]; then
      actual="$(sha256sum "$candidate" | cut -d' ' -f1)"
      [[ "$actual" == "$expected" ]] || continue
    fi
    printf '%s\n' "$candidate"
    return 0
  done < <(
    local cache_root
    while IFS= read -r cache_root; do
      [[ -n "$cache_root" ]] || continue
      printf '%s\n' \
        "$cache_root/$filename" \
        "$cache_root/model-packs/$filename" \
        "$cache_root/.asmr-dubber/cache/downloads/$filename" \
        "$cache_root/.asmr-dubber/bootstrap/linux/$filename" \
        "$cache_root/.asmr-dubber/bootstrap/windows/$filename"
    done < <(printf '%s\n' "$roots" | tr ':' '\n')
  )
  return 1
}

_asmr_is_modelscope_auth_url() {
  local url="$1"
  [[ "$url" == https://modelscope.cn/* || "$url" == https://www.modelscope.cn/* ]]
}

_asmr_is_modelscope_url() {
  local url="$1"
  [[ "$url" == https://modelscope.cn/* || \
    "$url" == https://*.modelscope.cn/* || \
    "$url" == https://modelscope.ai/* || \
    "$url" == https://*.modelscope.ai/* ]]
}

_asmr_curl_download_with_safe_auth() {
  local candidate="$1" partial="$2"
  local token="${MODELSCOPE_API_TOKEN:-}"
  if _asmr_is_modelscope_auth_url "$candidate" && [[ -n "$token" ]]; then
    (
      local cfg="" effective="$candidate" hop=0 probe_out status redirect
      local -a probe_args dl_args
      trap 'rm -f "${cfg:-}"' EXIT INT TERM
      cfg="$(umask 077 && mktemp)"
      local clean_token="${token//$'\r'/}"
      clean_token="${clean_token//$'\n'/}"
      clean_token="${clean_token//\\/\\\\}"
      clean_token="${clean_token//\"/\\\"}"
      printf 'header = "Authorization: Bearer %s"\nheader = "Cookie: m_session_id=%s"\n' \
        "$clean_token" "$clean_token" >"$cfg"

      while (( hop < 10 )); do
        if ! _asmr_is_modelscope_auth_url "$effective"; then
          break
        fi
        probe_args=(
          -sS -o /dev/null -w '%{http_code}\n%{redirect_url}'
          -r 0-0 --connect-timeout 20
          -H "Accept-Encoding: identity"
          -H "User-Agent: curl/8.0"
          -H "Referer: https://modelscope.cn/"
          -K "$cfg"
        )
        if ! probe_out="$(curl "${probe_args[@]}" "$effective" 2>/dev/null)"; then
          break
        fi
        status="${probe_out%%$'\n'*}"
        redirect=""
        if [[ "$probe_out" == *$'\n'* ]]; then
          redirect="${probe_out#*$'\n'}"
        fi
        case "$status" in
          301|302|303|307|308)
            if [[ -z "$redirect" ]]; then
              break
            fi
            if [[ "$redirect" != https://* ]]; then
              echo "下载源重定向到了非 HTTPS 地址。" >&2
              exit 1
            fi
            effective="$redirect"
            hop=$((hop + 1))
            ;;
          *)
            break
            ;;
        esac
      done

      dl_args=(
        -f --retry 4 --retry-delay 1 --connect-timeout 20
        -H "Accept-Encoding: identity"
      )
      if _asmr_is_modelscope_url "$effective"; then
        dl_args+=(
          -H "User-Agent: curl/8.0"
          -H "Referer: https://modelscope.cn/"
        )
      fi
      if _asmr_is_modelscope_auth_url "$effective"; then
        dl_args+=(-K "$cfg")
      else
        dl_args+=(-L)
      fi
      if curl "${dl_args[@]}" -C - "$effective" -o "$partial"; then
        exit 0
      fi
      rc=$?
      if [[ "$rc" == 33 ]]; then
        rm -f "$partial"
        curl "${dl_args[@]}" "$effective" -o "$partial"
        exit $?
      fi
      exit "$rc"
    )
    return $?
  fi

  local -a curl_args
  curl_args=(
    -fL --retry 4 --retry-delay 1 --connect-timeout 20
    -H "Accept-Encoding: identity"
  )
  if _asmr_is_modelscope_url "$candidate"; then
    curl_args+=(
      -H "User-Agent: curl/8.0"
      -H "Referer: https://modelscope.cn/"
    )
  fi
  if curl "${curl_args[@]}" -C - "$candidate" -o "$partial"; then
    return 0
  fi
  local exit_code=$?
  if [[ "$exit_code" == 33 ]]; then
    rm -f "$partial"
    curl "${curl_args[@]}" "$candidate" -o "$partial"
    return $?
  fi
  return "$exit_code"
}

asmr_download() {
  local root="$1" url="$2" destination="$3" expected="${4:-}"
  local candidate actual local_artifact exit_code
  mkdir -p "$(dirname "$destination")"
  local_artifact="$(asmr_find_local_artifact "$url" "$expected" || true)"
  if [[ -n "$local_artifact" ]]; then
    echo "复用只读本地缓存：$local_artifact"
    if [[ "$(readlink -f "$local_artifact")" != "$(readlink -m "$destination")" ]]; then
      cp -f "$local_artifact" "$destination"
    fi
    rm -f "$destination.partial"
    return 0
  fi
  if [[ -n "$expected" && -f "$destination.partial" ]]; then
    actual="$(sha256sum "$destination.partial" | cut -d' ' -f1)"
    if [[ "$actual" == "$expected" ]]; then
      mv -f "$destination.partial" "$destination"
      return 0
    fi
  fi
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    echo "尝试下载：$candidate"
    if _asmr_curl_download_with_safe_auth "$candidate" "$destination.partial"; then
      exit_code=0
    else
      exit_code=$?
    fi
    if [[ "$exit_code" == 0 ]]; then
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
    echo "当前下载源失败（curl $exit_code），保留断点并尝试下一个允许的来源。" >&2
  done < <(asmr_download_candidates "$root" "$url")
  echo "所有允许的下载源均失败，断点文件已保留：$url" >&2
  return 1
}

asmr_apply_mirror_environment() {
  local root="$1" endpoints preferred=""
  endpoints="$(asmr_mirror_list "$root" huggingface_endpoints | paste -sd ';' -)"
  if [[ -n "${ASMR_DUBBER_HF_ENDPOINT:-}" ]]; then
    preferred="${ASMR_DUBBER_HF_ENDPOINT%/}"
  elif [[ -n "${HF_ENDPOINT:-}" ]]; then
    preferred="${HF_ENDPOINT%/}"
  fi
  if [[ -n "$preferred" ]] && asmr_url_allowed "$root" "$preferred"; then
    case ";$endpoints;" in
      *";$preferred;"*) ;;
      *) endpoints="$preferred${endpoints:+;$endpoints}" ;;
    esac
  fi
  export ASMR_DUBBER_HF_ENDPOINTS="$endpoints"
  if [[ -n "$endpoints" ]]; then
    export ASMR_DUBBER_HF_ENDPOINT="${endpoints%%;*}"
    export HF_ENDPOINT="${endpoints%%;*}"
  else
    unset ASMR_DUBBER_HF_ENDPOINT
    unset HF_ENDPOINT
  fi
}
