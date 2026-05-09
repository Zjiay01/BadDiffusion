#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-complete_baseline_queue}"
PROJECT="/home1/zhln/code/BadDiffusion"
PY="/home1/zhln/envs/baddiffusion/bin/python"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" "cd '${PROJECT}' && bash -lc '
set -euo pipefail
export WANDB_MODE=disabled
PY=\"${PY}\"

echo \"[STAGE 1] CIFAR10 no-defense FID references\"
mkdir -p merge_results/logs/fid1000_nodef
CUDA_VISIBLE_DEVICES=0 \"\$PY\" merge.py --method diffusion_soup --model_ckpts ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 1.0 --alphas 0.0 --output_dir merge_results/fid1000_nodef_box14_hat --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 1024 --eval_max_batch 64 --fid_batch_size 64 --num_inference_steps 1000 --force_resample > merge_results/logs/fid1000_nodef/box14_hat.log 2>&1 &
P1=\$!
CUDA_VISIBLE_DEVICES=1 \"\$PY\" merge.py --method diffusion_soup --model_ckpts ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat --model_weights 1.0 --alphas 0.0 --output_dir merge_results/fid1000_nodef_box11_cat --dataset CIFAR10 --trigger BOX_11 --target CAT --sample_n 1024 --eval_max_batch 64 --fid_batch_size 64 --num_inference_steps 1000 --force_resample > merge_results/logs/fid1000_nodef/box11_cat.log 2>&1 &
P2=\$!
wait \$P1
wait \$P2

echo \"[STAGE 2] CIFAR10 S1/S2 baseline FID, all methods\"
\"\$PY\" scripts/run_baseline_queue.py --gpus 0,1,2,3 --prefix merge_results/fid1000 --scenarios s1_hat,s1_cat,s2_hat,s2_cat --methods diffusion_soup,dmm,maxfusion,anp,clean_finetune --sample_n 1024 --eval_max_batch 64 --fid_batch_size 64 --num_inference_steps 1000 --no_skip_fid --no_save_model --force_resample --log_dir merge_results/logs/fid1000

echo \"[STAGE 3] Refresh CIFAR index\"
\"\$PY\" scripts/write_merge_result_index.py --output merge_results

echo \"[STAGE 4] CelebA-HQ no-defense reference\"
mkdir -p merge_results/logs/celeba_hq
CELEBA_CLEAN=./legacy_outputs/model_old/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_c1.0_p0.0_GLASSES-CAT
CELEBA_BD=./legacy_outputs/model_old/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.3_GLASSES-CAT
CUDA_VISIBLE_DEVICES=0 \"\$PY\" merge.py --method diffusion_soup --model_ckpts \"\$CELEBA_BD\" --model_weights 1.0 --alphas 0.0 --output_dir merge_results/celeba_hq_nodef_glasses_cat --dataset CELEBA-HQ --trigger GLASSES --target CAT --sample_n 256 --eval_max_batch 4 --fid_batch_size 16 --num_inference_steps 1000 --force_resample > merge_results/logs/celeba_hq/nodef_glasses_cat.log 2>&1

echo \"[STAGE 5] CelebA-HQ S1 baseline FID, all methods\"
for method in diffusion_soup dmm maxfusion anp clean_finetune; do
  echo \"[CELEBA] start method=\${method}\"
  CUDA_VISIBLE_DEVICES=0 \"\$PY\" merge.py --method \"\${method}\" --model_ckpts \"\$CELEBA_CLEAN,\$CELEBA_BD\" --model_weights 0.5,0.5 --alphas 0.5 --output_dir \"merge_results/celeba_hq_s1_glasses_cat_\${method}\" --dataset CELEBA-HQ --trigger GLASSES --target CAT --sample_n 256 --eval_max_batch 4 --fid_batch_size 16 --num_inference_steps 1000 --dmm_batch_size 2 --clean_ft_batch_size 2 --anp_batches 4 --force_resample > \"merge_results/logs/celeba_hq/s1_glasses_cat_\${method}.log\" 2>&1
  echo \"[CELEBA] done method=\${method}\"
done

echo \"[STAGE 6] Final index refresh\"
\"\$PY\" scripts/write_merge_result_index.py --output merge_results
echo \"[ALL DONE] complete baseline queue finished\"
'"

echo "Started tmux session: ${SESSION}"
