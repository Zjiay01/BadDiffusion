#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home1/zhln/code/BadDiffusion}"
PY="${PY:-/home1/zhln/envs/baddiffusion/bin/python}"
SESSION_TO_STOP="${SESSION_TO_STOP:-complete_baseline_queue}"
SESSION="${1:-celeba_s1_parallel_queue}"

NODEF_SUMMARY="${PROJECT}/merge_results/celeba_hq_nodef_glasses_cat/merge_diffusion_soup_res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.3_GLASSES-CAT_p0.1/merge_summary.json"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT}' && bash -lc '
set -euo pipefail
export WANDB_MODE=disabled
PY=\"${PY}\"
NODEF_SUMMARY=\"${NODEF_SUMMARY}\"
SESSION_TO_STOP=\"${SESSION_TO_STOP}\"

echo \"[WAIT] CelebA-HQ no-defense summary: \${NODEF_SUMMARY}\"
while [[ ! -s \"\${NODEF_SUMMARY}\" ]]; do
  date \"+[%F %T] waiting for no-defense...\"
  sleep 60
done

echo \"[READY] no-defense summary found\"
if tmux has-session -t \"\${SESSION_TO_STOP}\" 2>/dev/null; then
  echo \"[STOP] stopping old sequential queue: \${SESSION_TO_STOP}\"
  tmux kill-session -t \"\${SESSION_TO_STOP}\" || true
fi

CELEBA_CLEAN=./legacy_outputs/model_old/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_c1.0_p0.0_GLASSES-CAT
CELEBA_BD=./legacy_outputs/model_old/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.3_GLASSES-CAT
mkdir -p merge_results/logs/celeba_hq

methods=(diffusion_soup dmm maxfusion anp clean_finetune)
gpus=(1 2 3 4)
running_pids=()
running_names=()

wait_one() {
  local pid=\"\${running_pids[0]}\"
  local name=\"\${running_names[0]}\"
  wait \"\${pid}\"
  echo \"[DONE] \${name}\"
  running_pids=(\"\${running_pids[@]:1}\")
  running_names=(\"\${running_names[@]:1}\")
}

for i in \"\${!methods[@]}\"; do
  method=\"\${methods[\${i}]}\"
  while [[ \"\${#running_pids[@]}\" -ge \"\${#gpus[@]}\" ]]; do
    wait_one
  done
  gpu=\"\${gpus[\${#running_pids[@]}]}\"
  out=\"merge_results/celeba_hq_s1_glasses_cat_\${method}\"
  log=\"merge_results/logs/celeba_hq/s1_glasses_cat_\${method}.log\"
  echo \"[START] method=\${method} gpu=\${gpu} log=\${log}\"
  CUDA_VISIBLE_DEVICES=\"\${gpu}\" \"\${PY}\" merge.py --method \"\${method}\" --model_ckpts \"\${CELEBA_CLEAN},\${CELEBA_BD}\" --model_weights 0.5,0.5 --alphas 0.5 --output_dir \"\${out}\" --dataset CELEBA-HQ --trigger GLASSES --target CAT --sample_n 256 --eval_max_batch 4 --fid_batch_size 16 --num_inference_steps 1000 --dmm_batch_size 2 --clean_ft_batch_size 2 --anp_batches 4 --force_resample > \"\${log}\" 2>&1 &
  running_pids+=(\"\$!\")
  running_names+=(\"\${method}\")
done

while [[ \"\${#running_pids[@]}\" -gt 0 ]]; do
  wait_one
done

\"\${PY}\" scripts/write_merge_result_index.py --output merge_results
echo \"[ALL DONE] CelebA-HQ parallel S1 baseline finished\"
'"

echo "Started tmux session: ${SESSION}"
