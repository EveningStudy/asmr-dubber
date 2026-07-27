#!/usr/bin/env bash

# This file is sourced by run-cli.sh.  It starts no persistent service and
# changes no Windows setting: the child Windows process lives only as long as
# the ASMR Dubber command that owns it.
ASMR_DUBBER_BRIDGE_PID=""

_asmr_proxy_probe() {
  local proxy="$1"
  curl -4 --fail --silent --output /dev/null \
    --proxy "$proxy" --noproxy "" --connect-timeout 5 --max-time 15 \
    "${ASMR_DUBBER_NETWORK_PROBE_URL:-https://modelscope.cn/robots.txt}"
}

_asmr_tcp_open() {
  local host="$1" port="$2"
  timeout 0.25 bash -c "</dev/tcp/$host/$port" >/dev/null 2>&1
}

_asmr_export_proxy() {
  local proxy="$1"
  export HTTP_PROXY="$proxy"
  export HTTPS_PROXY="$proxy"
  export http_proxy="$proxy"
  export https_proxy="$proxy"
  export ALL_PROXY=
  export all_proxy=
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
  export no_proxy="$NO_PROXY"
}

asmr_prepare_network() {
  if [[ "${ASMR_DUBBER_WINDOWS_BRIDGE:-1}" == "0" ]]; then
    return 0
  fi
  # Respect a proxy explicitly supplied by the user or calling environment.
  if [[ -n "${HTTPS_PROXY:-}" || -n "${https_proxy:-}" ]]; then
    return 0
  fi
  if ! grep -qi microsoft /proc/version 2>/dev/null; then
    return 0
  fi

  local gateway node windows_script log_file proxy port preferred
  gateway="$(ip route show default 2>/dev/null | awk 'NR == 1 {print $3}')"
  node="$(command -v node.exe 2>/dev/null || true)"
  if [[ -z "$gateway" || -z "$node" ]]; then
    echo "提示：未找到 Windows Node.js 网络桥；离线模型仍可用，但 DeepSeek 需要 WSL 可联网。" >&2
    return 0
  fi
  windows_script="$(wslpath -w "$ROOT/scripts/windows-connect-proxy.cjs")"

  # Reuse a bridge that is already alive.  Otherwise scan a short private port
  # range without ever terminating an unknown listener.
  preferred="${ASMR_DUBBER_PROXY_PORT:-5780}"
  for ((port = preferred; port <= preferred + 10; port++)); do
    proxy="http://$gateway:$port"
    if _asmr_tcp_open "$gateway" "$port"; then
      if _asmr_proxy_probe "$proxy"; then
        _asmr_export_proxy "$proxy"
        return 0
      fi
      continue
    fi

    log_file="/tmp/asmr-dubber-windows-proxy-$port.log"
    "$node" "$windows_script" "$gateway" "$port" >"$log_file" 2>&1 &
    ASMR_DUBBER_BRIDGE_PID=$!
    for _ in {1..40}; do
      if ! kill -0 "$ASMR_DUBBER_BRIDGE_PID" 2>/dev/null; then
        break
      fi
      if _asmr_tcp_open "$gateway" "$port"; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$ASMR_DUBBER_BRIDGE_PID" 2>/dev/null && _asmr_proxy_probe "$proxy"; then
      _asmr_export_proxy "$proxy"
      return 0
    fi
    kill "$ASMR_DUBBER_BRIDGE_PID" 2>/dev/null || true
    wait "$ASMR_DUBBER_BRIDGE_PID" 2>/dev/null || true
    ASMR_DUBBER_BRIDGE_PID=""
  done

  echo "提示：Windows 临时网络桥启动失败；离线模型仍可用，但 DeepSeek 请求可能失败。" >&2
  return 0
}

asmr_cleanup_network() {
  if [[ -n "$ASMR_DUBBER_BRIDGE_PID" ]]; then
    kill "$ASMR_DUBBER_BRIDGE_PID" 2>/dev/null || true
    wait "$ASMR_DUBBER_BRIDGE_PID" 2>/dev/null || true
    ASMR_DUBBER_BRIDGE_PID=""
  fi
}
