# README: merge.py 使用说明

本说明文档对应脚本 [merge.py](merge.py)，用于验证：

- 将后门模型与干净模型按权重合并（weight averaging）
- 再评估后门是否被削弱（是否可作为防御手段）

## 1. 功能概述

脚本会执行以下流程：

1. 读取后门模型与干净模型的 UNet 权重
2. 按给定 alpha 列表做参数线性合并
3. 对每个 alpha 生成两组图像
- clean 采样：正常初始噪声
- backdoor 采样：初始噪声 + trigger
4. 计算指标
- FID：clean 采样质量（越低越好）
- MSE / SSIM：backdoor 输出与目标后门目标图的接近程度
- ASR：按单图 MSE 阈值统计攻击成功率（越低越好）
5. 保存结果表格与 JSON

## 2. 环境与前置条件

- 你已经能在仓库中正常运行 [baddiffusion.py](baddiffusion.py)
- 已有两个训练完成的模型目录（结构为 diffusers 的 `save_pretrained` 格式）
- 数据集目录可被 [dataset.py](dataset.py) 读取

建议在项目目录执行：

```bash
cd /home1/zhln/code/BadDiffusion
conda activate /home1/zhln/envs/baddiffusion
```

## 3. 核心参数

必填参数：

- `--backdoor_ckpt`：后门模型目录
- `--clean_ckpt`：干净模型目录

常用参数：

- `--alphas`：后门权重列表（逗号分隔）
  - 合并公式：`W_merge = (1-alpha) * W_clean + alpha * W_backdoor`
  - `alpha=0` 完全干净模型，`alpha=1` 完全后门模型
- `--dataset`：`MNIST` / `CIFAR10` / `CELEBA` / `CELEBA-HQ`
- `--trigger`、`--target`：触发器和攻击目标类型（需与训练设定一致）
- `--sample_n`：每个 alpha 评估采样数（越大越稳定，越慢）
- `--eval_max_batch`：采样批大小（受显存影响）
- `--asr_threshold`：ASR 判定阈值（单图 MSE 小于该值视为成功）
- `--output_dir`：输出目录
  - 默认不填时，自动命名为：`merge_{model_name}_{poison_rate}`
  - 若填写且末级目录不是 `merge_` 开头，会自动在其下创建统一命名子目录

FID 相关参数：

- `--skip_fid`：跳过 FID（快速验证后门抑制时可用）
- `--fid_batch_size`、`--fid_num_workers`：FID 计算性能参数

复现实验相关参数：

- `--seed`：随机种子
- `--gpu`：指定可见 GPU，如 `0` 或 `0,1`（会设置 `CUDA_VISIBLE_DEVICES`）
- `--device`：`auto` / `cpu` / `cuda` / `cuda:0`
  - 默认 `auto`：有 CUDA 就用 `cuda`，否则回退 `cpu`
- `--force_resample`：即使目录已有图像也强制重新采样

运行时进度显示：

- 会显示每个 alpha 下 clean/backdoor 各自的采样进度条
- 形式为 `alpha=... clean sampling` 和 `alpha=... backdoor sampling`
- 进度单位为图片张数（img）

## 4. 示例命令

### 4.1 CIFAR10 完整评估（推荐）

```bash
python merge.py \
  --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
  --clean_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
  --gpu 0 \
  --device auto \
  --dataset CIFAR10 \
  --dataset_path datasets \
  --dataset_load_mode FIXED \
  --trigger BOX_14 \
  --target HAT \
  --alphas 0.0,0.2,0.5,0.8,0.9,1.0 \
  --sample_n 2048 \
  --eval_max_batch 256 \
  --fid_batch_size 64 \
  --fid_num_workers 4 \
  --asr_threshold 0.05 \
  --output_dir ./merge_results_cifar10
```

### 4.2 快速验证（跳过 FID）

```bash
python merge.py \
  --backdoor_ckpt ./res_xxx \
  --clean_ckpt ./res_yyy \
  --gpu 0 \
  --dataset CIFAR10 \
  --trigger BOX_14 \
  --target HAT \
  --alphas 0.0,0.25,0.5,0.75,1.0 \
  --sample_n 512 \
  --skip_fid \
  --output_dir ./merge_results_quick
```

### 4.3 CELEBA-HQ 示例

```bash
python merge.py \
  --backdoor_ckpt ./res_backdoor_celebahq \
  --clean_ckpt ./res_clean_celebahq \
  --gpu 0 \
  --dataset CELEBA-HQ \
  --dataset_path datasets \
  --dataset_load_mode FIXED \
  --trigger GLASSES \
  --target CAT \
  --alphas 0.0,0.2,0.4,0.6,0.8,1.0 \
  --sample_n 1024 \
  --eval_max_batch 64 \
  --output_dir ./merge_results_celebahq
```

## 5. 输出目录结构

以 `--output_dir ./merge_results_cifar10` 为例：

```text
merge_results_cifar10/
  merge_res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_0.1/
    args.json                   # 仅用户实际输入参数
    config.json                 # 完整参数（含默认值）
    real_CIFAR10/               # FID 的真实图像参考集
    alpha0.0000/
      clean/                    # clean 采样图
      backdoor/                 # backdoor 采样图
      clean_grid.png            # clean 样本拼图预览(8x8)
      backdoor_grid.png         # backdoor 样本拼图预览(8x8)
    alpha0.2000/
      clean/
      backdoor/
      clean_grid.png
      backdoor_grid.png
    ...
    merge_summary.json          # 机器可读结果
    merge_summary.txt           # 人可读表格

若不传 `--output_dir`，则直接在当前目录创建：

```text
merge_{model_name}_{poison_rate}/
```
```

## 6. 指标解释与结论判定

建议重点看以下趋势：

- 防御有效：随着 alpha 减小（更偏干净模型），ASR 明显下降
- 性能折中：ASR 下降的同时，FID 不应显著恶化

通常你会得到一条 trade-off：

- alpha 大：生成质量接近后门模型，ASR 也可能更高
- alpha 小：后门更弱，ASR 下降，但可能牺牲一部分生成质量

## 7. 常见问题

1. 显存不足（OOM）
- 降低 `--eval_max_batch`
- 降低 `--sample_n`

2. FID 太慢
- 先加 `--skip_fid` 做快速对比
- 或减小 `--sample_n`

3. 结果不稳定
- 固定 `--seed`
- 增大 `--sample_n`

4. ASR 与预期不符
- 确认 `--trigger` 和 `--target` 与训练时一致
- 调整 `--asr_threshold`，并在报告中写明阈值

## 8. 复现实验建议

建议分两阶段进行：

1. 粗扫 alpha
- 如：`0.0,0.2,0.4,0.6,0.8,1.0`

2. 在拐点附近细扫
- 如：`0.70,0.75,0.80,0.85,0.90`

最后基于 `merge_summary.json` 画出 FID-ASR 曲线，用于论文或报告中的防御有效性展示。
