#!/usr/bin/env bash
# Presentation only; profile IDs, commands and raw tool output stay unchanged.
setup_echo() {
  local message="$*"
  if [[ "${ASMR_DUBBER_UI_LANGUAGE:-zh}" == en ]]; then
    case "$message" in
      '用法：'*) message='Usage: bash scripts/linux/setup.sh [Core|Recommended|Advanced] [zh|en]' ;;
      '错误：Linux 安装脚本'*) message='Error: setup requires Linux x86_64 (64-bit).' ;;
      'ASMR Dubber · Linux 安装') message='ASMR Dubber · Linux setup' ;;
      '项目目录：'*) message="Application directory: ${message#*：}" ;;
      '数据目录：'*) message="Data directory: ${message#*：}" ;;
      '安装配置：'*) message="Installation profile: ${message#*：}" ;;
      '下载策略：ModelScope 优先；已显式允许海外备用源。') message='Downloads: ModelScope first; external fallback explicitly enabled.' ;;
      '下载策略：'*) message='Downloads: ModelScope first; external sources disabled.' ;;
      '只读本地缓存：'*) message="Read-only local cache: ${message#*：}" ;;
      '预计安装后占用：约 2 GB'*) message='Installed: about 2 GB; reserve at least 5 GB.' ;;
      '预计安装后占用：约 24'*) message='Installed: about 24-28 GB; reserve at least 35 GB.' ;;
      '预计安装后占用：约 33'*) message='Installed: about 33-39 GB; reserve at least 50 GB.' ;;
      '固定 ASR 模型：'*) message="Pinned ASR: ${message#*：}" ;;
      '固定 VAD 模型：'*) message="Pinned VAD: ${message#*：}" ;;
      '固定时间戳模型：'*) message='Pinned aligner: Qwen/Qwen3-ForcedAligner-0.6B' ;;
      '固定 TTS 模型：'*) message='Pinned TTS: IndexTTS2 checkpoints (NVIDIA only)' ;;
      '不会自动安装 Kotoba'*) message='Does not install Kotoba v2.0/v2.1, Faster-Whisper large-v3 or other ASR models.' ;;
      '未检测到 NVIDIA GPU'*) message='CUDA TTS is skipped without NVIDIA; disk usage is lower.' ;;
      '正在从 ModelScope 优先源安装 uv...') message='Installing uv from the ModelScope-first source...' ;;
      '错误：uv 安装失败：'*) message="Error: uv installation failed: ${message##*：}" ;;
      '正在准备 Python 3.12...') message='Preparing Python 3.12...' ;;
      '正在安装应用依赖：'*) message="Installing application dependencies: ${message#*：}" ;;
      '使用软件源：'*) message="Package index: ${message#*：}" ;;
      '当前软件源失败，自动切换。') message='Package index failed; trying the next configured source.' ;;
      '使用 ModelScope 应用依赖 wheelhouse：'*) message="Using ModelScope application wheelhouse: ${message#*：}" ;;
      'ModelScope wheelhouse 早于'*) message='Wheelhouse predates dependencies; supplementing from the configured domestic index.' ;;
      '错误：ModelScope wheelhouse'*) message='Error: published ModelScope wheelhouse is incomplete; refusing silent fallback.' ;;
      '错误：基础应用或在线/API'*) message='Error: core application or online/API clients remain incomplete.' ;;
      '正在检测并导入当前档位的本地模型包...') message='Checking and importing local model packages...' ;;
      '错误：本地模型包'*) message='Error: local model package import failed; check model-packs archives.' ;;
      '正在准备 ASR'*) message='Preparing ASR: Kotoba-Whisper v2.2 and Faster-Whisper large-v2...' ;;
      '正在安装推荐 ASR'*) message='Installing recommended ASR: Japanese Parakeet...' ;;
      '正在准备 Parakeet'*) message='Preparing the portable Parakeet CUDA 13 runtime...' ;;
      'CUDA 运行库安装失败'*) message='CUDA runtime installation failed; Parakeet will use CPU.' ;;
      '正在安装推荐 TTS'*) message='Installing recommended TTS: IndexTTS2 (about 20 GB)...' ;;
      '正在执行环境检查...') message='Checking the environment...' ;;
      '提示：核心程序已安装'*) message='Core installed; some selected models are unavailable. Check Settings > Devices & models.' ;;
      '安装完成。运行 bash '*) message="Installation complete. Start with: bash $ROOT/scripts/linux/run-ui.sh" ;;
    esac
  fi
  printf '%s\n' "$message"
}
