#!/usr/bin/env bash
set -euo pipefail

GPU_ID=2
MAX_MEM_MB=2000
CHECK_INTERVAL=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --max_mem_mb)
      MAX_MEM_MB="$2"
      shift 2
      ;;
    --check_interval)
      CHECK_INTERVAL="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "No command provided after --" >&2
  exit 2
fi

echo "[WAIT] gpu=${GPU_ID} max_mem_mb=${MAX_MEM_MB} check_interval=${CHECK_INTERVAL}s"
while true; do
  mem_used="$(nvidia-smi --query-gpu=memory.used --id="${GPU_ID}" --format=csv,noheader,nounits | tr -d ' ')"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[CHECK] ${timestamp} gpu=${GPU_ID} mem_used_mb=${mem_used}"
  if [[ "${mem_used}" -le "${MAX_MEM_MB}" ]]; then
    echo "[RUN] GPU ${GPU_ID} is available. Starting command: $*"
    exec "$@"
  fi
  sleep "${CHECK_INTERVAL}"
done
