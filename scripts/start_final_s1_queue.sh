#!/usr/bin/env bash
set -euo pipefail

cd /home1/zhln/code/BadDiffusion

exec bash scripts/wait_and_run_baseline_queue.sh \
  --gpu "${GPU_ID:-2}" \
  --max_mem_mb "${MAX_MEM_MB:-2000}" \
  --check_interval "${CHECK_INTERVAL:-300}" \
  -- \
  /home1/zhln/envs/baddiffusion/bin/python scripts/run_baseline_queue.py \
  --prefix final \
  --scenarios s1_hat,s1_cat \
  --methods diffusion_soup,dmm,maxfusion,anp,clean_finetune \
  --sample_n 1024 \
  --num_inference_steps 200 \
  --gpus "${GPU_ID:-2}" \
  --log_dir merge_results/logs/final_s1_queue
