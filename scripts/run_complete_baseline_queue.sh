#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home1/zhln/envs/baddiffusion/bin/python}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

nodef_hat_summary="merge_results/fid1000_nodef_box14_hat/merge_diffusion_soup_res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat_p0.1/merge_summary.json"
nodef_cat_summary="merge_results/fid1000_nodef_box11_cat/merge_diffusion_soup_res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat_p0.1/merge_summary.json"

echo "[STAGE 1] CIFAR10 no-defense FID references"
mkdir -p merge_results/logs/fid1000_nodef
if [[ -s "${nodef_hat_summary}" ]]; then
  echo "[SKIP] BOX_14-HAT no-defense FID already exists: ${nodef_hat_summary}"
else
  CUDA_VISIBLE_DEVICES=1 "${PY}" merge.py --method diffusion_soup --model_ckpts ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 1.0 --alphas 0.0 --output_dir merge_results/fid1000_nodef_box14_hat --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 1024 --eval_max_batch 64 --fid_batch_size 64 --num_inference_steps 1000 --force_resample > merge_results/logs/fid1000_nodef/box14_hat.log 2>&1
fi

if [[ -s "${nodef_cat_summary}" ]]; then
  echo "[SKIP] BOX_11-CAT no-defense FID already exists: ${nodef_cat_summary}"
else
  CUDA_VISIBLE_DEVICES=2 "${PY}" merge.py --method diffusion_soup --model_ckpts ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat --model_weights 1.0 --alphas 0.0 --output_dir merge_results/fid1000_nodef_box11_cat --dataset CIFAR10 --trigger BOX_11 --target CAT --sample_n 1024 --eval_max_batch 64 --fid_batch_size 64 --num_inference_steps 1000 --force_resample > merge_results/logs/fid1000_nodef/box11_cat.log 2>&1
fi

echo "[STAGE 2] CIFAR10 S1/S2 baseline FID, all methods"
"${PY}" scripts/run_baseline_queue.py --gpus 1,2,3,4 --prefix merge_results/fid1000 --scenarios s1_hat,s1_cat,s2_hat,s2_cat --methods diffusion_soup,dmm,maxfusion,anp,clean_finetune --sample_n 1024 --eval_max_batch 64 --fid_batch_size 64 --num_inference_steps 1000 --no_skip_fid --no_save_model --force_resample --log_dir merge_results/logs/fid1000

echo "[STAGE 3] Refresh CIFAR index"
"${PY}" scripts/write_merge_result_index.py --output merge_results

echo "[STAGE 4] CelebA-HQ no-defense reference"
mkdir -p merge_results/logs/celeba_hq
CELEBA_CLEAN=./legacy_outputs/model_old/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_c1.0_p0.0_GLASSES-CAT
CELEBA_BD=./legacy_outputs/model_old/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.3_GLASSES-CAT
CUDA_VISIBLE_DEVICES=1 "${PY}" merge.py --method diffusion_soup --model_ckpts "${CELEBA_BD}" --model_weights 1.0 --alphas 0.0 --output_dir merge_results/celeba_hq_nodef_glasses_cat --dataset CELEBA-HQ --trigger GLASSES --target CAT --sample_n 256 --eval_max_batch 4 --fid_batch_size 16 --num_inference_steps 1000 --force_resample > merge_results/logs/celeba_hq/nodef_glasses_cat.log 2>&1

echo "[STAGE 5] CelebA-HQ S1 baseline FID, all methods"
for method in diffusion_soup dmm maxfusion anp clean_finetune; do
  echo "[CELEBA] start method=${method}"
  CUDA_VISIBLE_DEVICES=1 "${PY}" merge.py --method "${method}" --model_ckpts "${CELEBA_CLEAN},${CELEBA_BD}" --model_weights 0.5,0.5 --alphas 0.5 --output_dir "merge_results/celeba_hq_s1_glasses_cat_${method}" --dataset CELEBA-HQ --trigger GLASSES --target CAT --sample_n 256 --eval_max_batch 4 --fid_batch_size 16 --num_inference_steps 1000 --dmm_batch_size 2 --clean_ft_batch_size 2 --anp_batches 4 --force_resample > "merge_results/logs/celeba_hq/s1_glasses_cat_${method}.log" 2>&1
  echo "[CELEBA] done method=${method}"
done

echo "[STAGE 6] Final index refresh"
"${PY}" scripts/write_merge_result_index.py --output merge_results
echo "[ALL DONE] complete baseline queue finished"
