#!/usr/bin/env bash
set -euo pipefail

BD=./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT
CL=./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT
ALPHAS="0.0,0.2,0.5,0.8,0.85,0.87,0.9,0.95,0.97,0.99,1.0"
OUT="./CIFAR10-Merge"

# Weight Averaging（已有结果，注释掉）
# python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
#   --method wa --alphas $ALPHAS \
#   --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
#   --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

# Task Arithmetic
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method task_arithmetic --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

# TIES（主实验固定 k=0.2）
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method ties --ties_k 0.2 --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

# DARE（p=0.5，三个 seed 自动取参数均值）
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method dare --dare_p 0.5 --dare_seeds 42,123,777 --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

# SLERP
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method slerp --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT