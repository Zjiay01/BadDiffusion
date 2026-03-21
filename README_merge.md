# merge.py 使用文档

> 将后门扩散模型与干净模型进行参数合并，评估后门在不同合并策略下的抑制效果。
> 支持 Weight Averaging、Task Arithmetic、TIES、DARE、SLERP 五种合并方法。

---

## 目录

1. [功能概述](#功能概述)
2. [合并方法说明](#合并方法说明)
3. [参数说明](#参数说明)
4. [示例命令](#示例命令)
5. [输出目录结构](#输出目录结构)
6. [指标说明](#指标说明)
7. [Alpha 语义对照表](#alpha-语义对照表)
8. [常见问题](#常见问题)
9. [实验建议](#实验建议)

---

## 功能概述

脚本主流程：

1. 读取后门模型（`backdoor_ckpt`）与干净模型（`clean_ckpt`）的 UNet 权重
2. 按指定方法（`--method`）和 alpha 列表对参数进行合并
3. 对每个 alpha 生成两组图像
   - **clean 采样**：正常初始噪声，用于评估生成质量（FID）
   - **backdoor 采样**：初始噪声 + trigger，用于评估后门是否存活（ASR）
4. 计算并汇总指标，输出表格、JSON、曲线图

---

## 合并方法说明

通过 `--method` 参数选择，支持以下五种方法。

### `wa` — Weight Averaging（默认）

最简单的线性插值：

```
W_merged = (1 - alpha) * W_clean + alpha * W_backdoor
```

- alpha=0：完全干净模型
- alpha=1：完全后门模型
- 参考文献：Wortsman et al., *Model Soups*, ICML 2022

---

### `task_arithmetic` — Task Arithmetic

将干净方向的 task vector 加到后门模型上，逐步"覆盖"后门：

```
τ = W_clean - W_backdoor          # task vector（朝干净方向）
W_merged = W_backdoor + alpha * τ
         = (1 - alpha) * W_backdoor + alpha * W_clean
```

- 以后门模型为 pretrained，干净模型为 finetuned
- alpha=0：纯后门模型；alpha=1：后门被完全抑制
- 数学上与 WA 等价，但语义清晰：对 task vector 做 negation 来消除后门
- 参考文献：Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023

---

### `ties` — TIES-Merging

在 task vector 上额外做 Trim + Elect-Sign 操作，减少参数干扰：

```
τ = W_clean - W_backdoor

Step 1 – Trim:   保留 |τ| 中 top-k% 的参数，其余置零
Step 2 – Elect:  单向量时符号直接由 τ_trimmed 决定
Step 3 – Merge:  W_merged = W_backdoor + alpha * τ_trimmed
```

- 超参：`--ties_k`（保留比例，默认 0.2 = 保留 top 20%）
- k 越小越激进（保留越少参数）；k=1.0 退化为 Task Arithmetic
- 参考文献：Yadav et al., *TIES-Merging*, NeurIPS 2023

---

### `dare` — DARE (Drop And REscale)

随机 drop task vector 中的参数，再 rescale 补偿幅度：

```
τ = W_clean - W_backdoor

mask ~ Bernoulli(1 - p)           # p = drop rate
τ_dare = τ * mask / (1 - p)       # 随机稀疏化 + 幅度补偿

W_merged = W_backdoor + alpha * τ_dare
```

- 超参：`--dare_p`（drop rate，默认 0.5）；`--dare_seed`（随机种子）
- p 越大 task vector 越稀疏；p=0 退化为 Task Arithmetic
- 参考文献：Yu et al., *Language Models are Super Mario*, arXiv 2023

---

### `slerp` — Spherical Linear Interpolation

在高维球面上沿大圆弧插值，保持参数向量的 norm：

```
θ = arccos(W_clean · W_backdoor / (|W_clean| * |W_backdoor|))

W_merged = sin((1-alpha)*θ)/sin(θ) * W_clean
         + sin(alpha*θ)/sin(θ)     * W_backdoor
```

- alpha=0：W_clean；alpha=1：W_backdoor
- 对近似平行或近似零向量自动退化为线性插值（数值稳定）
- 参考文献：Shoemake, *Animating Rotation with Quaternion Curves*, SIGGRAPH 1985；Goddard et al., *MergeKit*, arXiv 2024

---

## 参数说明

### 必填参数

| 参数 | 说明 |
|------|------|
| `--backdoor_ckpt` | 后门模型目录（diffusers `save_pretrained` 格式） |
| `--clean_ckpt` | 干净模型目录 |

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `wa` | 合并方法：`wa` / `task_arithmetic` / `ties` / `dare` / `slerp` |
| `--alphas` | `0.0,0.2,0.5,0.8,0.9,1.0` | alpha 扫描列表（逗号分隔） |
| `--dataset` | `CIFAR10` | 数据集类型 |
| `--trigger` | `BOX_14` | 触发器类型（需与训练时一致） |
| `--target` | `HAT` | 攻击目标类型（需与训练时一致） |
| `--sample_n` | `2048` | 每个 alpha 的采样数 |
| `--eval_max_batch` | `256` | 采样批大小（受显存限制） |
| `--asr_threshold` | `0.05` | ASR 判定阈值（per-image MSE < 此值视为攻击成功） |
| `--output_dir` | 自动命名 | 输出目录（不填则自动命名为 `merge_{method}_{model}_{p}`） |

### 方法专属参数

| 参数 | 方法 | 默认值 | 说明 |
|------|------|--------|------|
| `--ties_k` | `ties` | `0.2` | 保留 task vector 中 top-k% 的参数（0~1） |
| `--dare_p` | `dare` | `0.5` | Task vector 的随机 drop rate（0~1） |
| `--dare_seed` | `dare` | `42` | DARE dropout mask 随机种子 |

### FID 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip_fid` | — | 跳过 FID 计算（快速验证后门抑制） |
| `--fid_batch_size` | `64` | FID 计算批大小 |
| `--fid_num_workers` | `4` | FID dataloader worker 数 |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | — | 指定 GPU，如 `0` 或 `0,1`（设置 `CUDA_VISIBLE_DEVICES`） |
| `--fclip` | `o` | 采样时 `clip_sample`：`w`=True / `o`=False / `n`=不改动 |
| `--seed` | `0` | 随机种子 |
| `--force_resample` | — | 强制重新采样（忽略已有图像） |
| `--dataset_path` | `datasets` | 数据集根目录 |
| `--dataset_load_mode` | `FIXED` | 数据集加载模式 |
| `--clean_rate` | `1.0` | 干净样本比例 |
| `--poison_rate` | `0.1` | 毒化样本比例 |

---

## 示例命令

### CIFAR10 完整评估

```bash
# Weight Averaging（原始方法）
python merge.py \
  --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
  --clean_ckpt    ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
  --method wa \
  --alphas 0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 \
  --output_dir ./merge_results

# Task Arithmetic
python merge.py \
  --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
  --clean_ckpt    ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
  --method task_arithmetic \
  --alphas 0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 \
  --output_dir ./merge_results

# TIES（保留 top 20%）
python merge.py \
  --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
  --clean_ckpt    ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
  --method ties --ties_k 0.2 \
  --alphas 0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 \
  --output_dir ./merge_results

# DARE（drop 50%）
python merge.py \
  --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
  --clean_ckpt    ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
  --method dare --dare_p 0.5 \
  --alphas 0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 \
  --output_dir ./merge_results

# SLERP
python merge.py \
  --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
  --clean_ckpt    ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
  --method slerp \
  --alphas 0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 \
  --output_dir ./merge_results
```

### CelebA-HQ 示例

```bash
python merge.py \
  --backdoor_ckpt ./res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.3_GLASSES-CAT \
  --clean_ckpt    ./res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.0_GLASSES-CAT \
  --method ties --ties_k 0.2 \
  --alphas 0.0,0.5,0.8,0.9,1.0 \
  --gpu 0 --fclip o \
  --dataset CELEBA-HQ --trigger GLASSES --target CAT \
  --sample_n 1024 --eval_max_batch 64 \
  --output_dir ./merge_results_celebahq
```

### 快速验证（跳过 FID）

```bash
python merge.py \
  --backdoor_ckpt ./res_backdoor \
  --clean_ckpt    ./res_clean \
  --method dare --dare_p 0.5 \
  --alphas 0.0,0.5,1.0 \
  --gpu 0 --fclip o --skip_fid \
  --sample_n 256 --eval_max_batch 64
```

---

## 输出目录结构

以 `--output_dir ./merge_results --method ties` 为例：

```
merge_results/
└── merge_ties_{model_name}_{poison_rate}/
    ├── args.json                   # 用户实际输入的参数
    ├── config.json                 # 完整参数（含所有默认值）
    ├── real_CIFAR10/               # FID 真实图像参考集
    ├── alpha0.0000/
    │   ├── clean/                  # clean 采样图（用于 FID）
    │   ├── backdoor/               # backdoor 采样图（用于 ASR）
    │   ├── clean_grid.png          # 8×8 拼图预览
    │   └── backdoor_grid.png       # 8×8 拼图预览
    ├── alpha0.5000/
    │   └── ...
    ├── ...
    ├── merge_summary.json          # 机器可读结果
    ├── merge_summary.txt           # 人可读表格
    └── asr_fid_tradeoff.png        # ASR / FID 曲线图
```

不同方法的结果目录名自动区分（`merge_wa_...`、`merge_ties_...` 等），不会互相覆盖。

---

## 指标说明

| 指标 | 含义 | 越低越好 / 越高越好 |
|------|------|---------------------|
| **FID** | clean 采样与真实图像的分布距离，衡量生成质量 | 越低越好 |
| **MSE** | backdoor 采样与 target 图像的平均像素误差 | 越低 = 后门越强 |
| **SSIM** | backdoor 采样与 target 图像的结构相似度 | 越高 = 后门越强 |
| **ASR** | Attack Success Rate：per-image MSE < `asr_threshold` 的比例 | 越低 = 防御越好 |

输出表格示例：

```
   alpha |          fid |          mse |         ssim |          asr
----------------------------------------------------------------------
  0.0000 |    52.380000 |     0.240567 |     0.012345 |     0.000000
  0.5000 |    49.380000 |     0.240567 |     0.012345 |     0.000000
  0.9000 |    52.550000 |     0.217131 |     0.015678 |     0.019000
  1.0000 |    52.280000 |     0.015671 |     0.312456 |     0.912000
```

---

## Alpha 语义对照表

所有方法的 alpha 均遵循统一语义：**alpha=0 对应干净模型行为，alpha=1 对应后门模型行为**。

| 方法 | alpha=0 | alpha=1 |
|------|---------|---------|
| `wa` | 纯干净模型 | 纯后门模型 |
| `task_arithmetic` | 后门完全被 task vector 覆盖 | 纯后门模型（τ 未施加） |
| `ties` | τ_trimmed 完全施加 | 纯后门模型 |
| `dare` | τ_dare 完全施加 | 纯后门模型 |
| `slerp` | 插值到 W_clean | 插值到 W_backdoor |

> **注意**：`task_arithmetic` / `ties` / `dare` 在 alpha=0 时并不保证与干净模型完全等同，因为 task vector 操作本身可能引入轻微差异。若需要严格的干净模型基线，使用 `wa --alphas 0.0`。

---

## 常见问题

**Q：显存不足（OOM）**
- 降低 `--eval_max_batch`（CelebA-HQ 建议 ≤ 64）
- 降低 `--sample_n`

**Q：FID 值偏高**
- `sample_n=1000` 时 FID 约 50，论文建议用 5000+
- 相对趋势有效，最终结果用更大 `sample_n`

**Q：TIES 的 `--ties_k` 怎么选**
- 建议先跑 `k=0.2`（保留 20%），再对比 `k=0.1` 和 `k=0.5`
- k 越小：task vector 越稀疏，后门抑制更激进，但可能损害生成质量

**Q：DARE 的 `--dare_p` 怎么选**
- 默认 `p=0.5`，可以额外跑 `p=0.3` 和 `p=0.7` 对比
- p 越大：随机性越强，结果方差也更大（建议多次运行取平均）

**Q：不同方法的结果目录会冲突吗**
- 不会。输出目录自动包含方法名（如 `merge_ties_...`），各方法独立存储

**Q：如何复现 WA 的历史结果**
- 直接用 `--method wa`，行为与旧版 `merge.py` 完全一致

---

## 实验建议

### Alpha 扫描策略

建议分两阶段：

1. **粗扫**（6~8 个点）：`0.0,0.2,0.5,0.7,0.85,0.9,0.95,1.0`
2. **细扫拐点**：在 ASR 开始快速上升的区间密集采样，如 `0.85,0.87,0.88,0.89,0.90,0.91`

### 一键对比所有方法

```bash
for METHOD in wa task_arithmetic ties dare slerp; do
  python merge.py \
    --backdoor_ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT \
    --clean_ckpt    ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT \
    --method $METHOD \
    --alphas 0.0,0.5,0.8,0.85,0.9,0.95,1.0 \
    --gpu 0 --fclip o \
    --dataset CIFAR10 --trigger BOX_14 --target HAT \
    --sample_n 1000 --skip_fid \
    --output_dir ./merge_results_all
done
```
