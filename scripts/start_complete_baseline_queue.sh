#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-complete_baseline_queue}"
PROJECT="/home1/zhln/code/BadDiffusion"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT}' && bash scripts/run_complete_baseline_queue.sh"

echo "Started tmux session: ${SESSION}"
