# merge.py / ablation.py 使用文档

> 将后门扩散模型与干净模型进行参数合并，评估后门在不同合并策略下的抑制效果。
> 支持 Weight Averaging、Task Arithmetic、TIES、DARE、SLERP 五种合并方法。
> 配套 `ablation.py` 用于 NeurIPS 附录所需的超参消融与 DARE 稳定性实验。

---

## 目录

1. [文件说明](#文件说明)
2. [功能概述](#功能概述)
3. [合并方法说明](#合并方法说明)
4. [merge.py 参数说明](#mergepy-参数说明)
5. [ablation.py 参数说明](#ablationpy-参数说明)
6. [示例命令 — merge.py](#示例命令--mergepy)
7. [示例命令 — ablation.py](#示例命令--ablationpy)
8. [输出目录结构](#输出目录结构)
9. [指标说明](#指标说明)
10. [Alpha 语义对照表](#alpha-语义对照表)
11. [NeurIPS 实验设计](#neurips-实验设计)
12. [常见问题](#常见问题)

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `merge.py` | 主实验：固定超参，扫 alpha，输出 ASR/FID 曲线 |
| `ablation.py` | 消融实验：DARE 多 seed 稳定性 + TIES/DARE 超参扫描 |

两个文件必须放在同一目录，`ablation.py` 直接从 `merge.py` import 所有 merge 函数。

---

## 功能概述

### merge.py

1. 读取后门模型（`--backdoor_ckpt`）与干净模型（`--clean_ckpt`）的 UNet 权重
2. 按指定方法（`--method`）和 alpha 列表逐一合并权重
3. 每个 alpha 生成两组图像：
   - **clean 采样**：正常初始噪声 → 评估生成质量（FID）
   - **backdoor 采样**：初始噪声 + trigger → 评估后门存活率（ASR）
4. 输出汇总表格、JSON、ASR-FID 曲线图

### ablation.py

两种模式（`--mode`）：

- **`dare_seed`**：固定 `dare_p`，用多个 seed 各自独立跑完整 alpha sweep，每个 seed 单独存一行结果，最后追加 mean±std 汇总行，验证 DARE 随机性的影响范围
- **`hparam`**：固定一个或多个 alpha 值，扫 `ties_k` 或 `dare_p`，输出折线图和热力图，为论文中超参默认值选取提供依据

---

## 合并方法说明

通过 `--method` 参数选择。

### `wa` — Weight Averaging（默认）

```
W_merged = (1 - alpha) * W_clean + alpha * W_backdoor
```

- alpha=0：纯干净模型；alpha=1：纯后门模型
- 最简单的线性插值，无额外超参
- 参考文献：Wortsman et al., *Model Soups*, ICML 2022

---

### `task_arithmetic` — Task Arithmetic

```
τ        = W_clean - W_backdoor        # task vector（朝干净方向）
W_merged = W_backdoor + alpha * τ
         = (1-alpha) * W_backdoor + alpha * W_clean
```

- 以后门模型为 pretrained，干净模型为 finetuned，对 task vector 做 negation 来抑制后门
- 数学上与 WA 等价，但提供了 task vector 视角的解释
- alpha=0：纯后门；alpha=1：后门完全被 τ 覆盖
- 无额外超参
- 参考文献：Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023

---

### `ties` — TIES-Merging

```
τ = W_clean - W_backdoor

Step 1 – Trim:   保留 |τ| 中绝对值最大的 top-k% 参数，其余置零 → τ_trimmed
Step 2 – Elect:  单向量时符号由 τ_trimmed 直接决定（无冲突）
Step 3 – Merge:  W_merged = W_backdoor + alpha * τ_trimmed
```

- 额外超参：`--ties_k`（默认 0.2，即保留幅度最大的 20% 参数）
- k=1.0 退化为 Task Arithmetic；k 越小后门抑制越激进
- 参考文献：Yadav et al., *TIES-Merging*, NeurIPS 2023

---

### `dare` — DARE (Drop And REscale)

```
τ          = W_clean - W_backdoor
mask       ~ Bernoulli(1 - p)          # p = drop rate，随机生成 0/1 掩码
τ_dare     = τ * mask / (1 - p)        # 随机稀疏化后 rescale 补偿期望幅度
W_merged   = W_backdoor + alpha * τ_dare
```

- 额外超参：`--dare_p`（默认 0.5）；`--dare_seed`（默认 42）
- p=0 退化为 Task Arithmetic；p 越大 task vector 越稀疏，随机性越强
- 由于随机性，**主实验建议多 seed 取均值**，详见 ablation.py
- 参考文献：Yu et al., *Language Models are Super Mario*, arXiv 2023

---

### `slerp` — Spherical Linear Interpolation

```
θ        = arccos( W_clean · W_backdoor / (|W_clean| × |W_backdoor|) )
W_merged = sin((1-alpha)×θ)/sin(θ) × W_clean
         + sin(alpha×θ)/sin(θ)     × W_backdoor
```

- 在高维球面上沿大圆弧插值，保持参数向量 norm 不因插值缩水
- 对近似平行（θ≈0）或近似零向量自动退化为线性插值（数值稳定）
- alpha=0：W_clean；alpha=1：W_backdoor
- 无额外超参
- 参考文献：Shoemake, *Animating Rotation with Quaternion Curves*, SIGGRAPH 1985；Goddard et al., *MergeKit*, arXiv 2024

---

## merge.py 参数说明

### 必填

| 参数 | 说明 |
|------|------|
| `--backdoor_ckpt` | 后门模型目录（diffusers `save_pretrained` 格式） |
| `--clean_ckpt` | 干净模型目录 |

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `wa` | 合并方法：`wa` / `task_arithmetic` / `ties` / `dare` / `slerp` |
| `--alphas` | `0.0,0.2,0.5,0.8,0.9,1.0` | alpha 扫描列表（逗号分隔，值域 [0,1]） |
| `--dataset` | `CIFAR10` | 数据集：`MNIST` / `CIFAR10` / `CELEBA` / `CELEBA-HQ` |
| `--trigger` | `BOX_14` | 触发器类型（须与训练时一致） |
| `--target` | `HAT` | 攻击目标类型（须与训练时一致） |
| `--sample_n` | `2048` | 每个 alpha 的采样图数 |
| `--eval_max_batch` | `256` | 采样批大小（受显存限制，CelebA-HQ 建议 ≤ 64） |
| `--asr_threshold` | `0.05` | ASR 判定阈值：per-image MSE < 此值视为攻击成功 |
| `--output_dir` | 自动命名 | 不填则自动命名为 `merge_{method}_{model}_{poison_rate}` |

### 方法专属参数

| 参数 | 适用方法 | 默认值 | 说明 |
|------|----------|--------|------|
| `--ties_k` | `ties` | `0.2` | 保留 task vector 中绝对值 top-k 比例的参数（0~1） |
| `--dare_p` | `dare` | `0.5` | Task vector 随机 drop rate（0~1） |
| `--dare_seed` | `dare` | `42` | DARE dropout mask 随机种子 |

### FID 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip_fid` | — | 跳过 FID（快速验证后门抑制效果时使用） |
| `--fid_batch_size` | `64` | FID 计算批大小 |
| `--fid_num_workers` | `4` | FID dataloader worker 数 |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | — | 指定可见 GPU，如 `0` 或 `0,1`（设置 `CUDA_VISIBLE_DEVICES`） |
| `--fclip` | `o` | 采样时 `clip_sample`：`w`=True / `o`=False / `n`=不改动 |
| `--seed` | `0` | 采样随机种子（固定后同一 alpha 的 noise 可复现） |
| `--force_resample` | — | 强制重新采样（即使目录已有图像） |
| `--dataset_path` | `datasets` | 数据集根目录 |
| `--dataset_load_mode` | `FIXED` | 数据集加载模式：`FIXED` / `FLEX` |
| `--clean_rate` | `1.0` | 干净样本比例 |
| `--poison_rate` | `0.1` | 毒化样本比例（影响 output_dir 自动命名） |

---

## ablation.py 参数说明

### 通用参数（两种模式均适用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | **必填** | `dare_seed`（多 seed 稳定性）或 `hparam`（超参扫描） |
| `--backdoor_ckpt` | **必填** | 同 merge.py |
| `--clean_ckpt` | **必填** | 同 merge.py |
| `--output_dir` | `./ablation_results` | 消融结果根目录 |
| `--dataset` / `--trigger` / `--target` | 同 merge.py | 须与训练时一致 |
| `--sample_n` | `1000` | 消融用采样数（比主实验小即可） |
| `--eval_max_batch` | `256` | 采样批大小 |
| `--skip_fid` | — | 消融通常跳过 FID 以节省时间 |
| `--gpu` / `--fclip` / `--seed` | 同 merge.py | — |

### `--mode dare_seed` 专属参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dare_p` | `0.5` | 固定的 DARE drop rate |
| `--dare_seeds` | `42,123,777` | 要跑的 seed 列表（逗号分隔） |
| `--alphas` | `0.0,0.5,0.8,0.9,1.0` | alpha sweep 列表 |

**输出文件：**

| 文件 | 内容 |
|------|------|
| `dare_seed_summary.json` | `rows`（各 seed 各 alpha 逐行） + `mean_rows`（均值行） |
| `dare_seed_summary.txt` | 可读表格：各 alpha 先列各 seed 结果，最后一行为 `mean±std` |
| `dare_seed_stability.png` | 均值折线 + ±1 std 色带，各 seed 细虚线叠加 |

### `--mode hparam` 专属参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `ties` | 要消融的方法：`ties` 或 `dare` |
| `--hparam_name` | **必填** | 要扫的超参：`ties_k` 或 `dare_p` |
| `--hparam_values` | **必填** | 超参取值列表，如 `0.05,0.1,0.2,0.3,0.5,1.0` |
| `--fixed_alphas` | `0.9` | 固定的 alpha 值，支持多个（如 `0.85,0.9`） |

**输出文件：**

| 文件 | 内容 |
|------|------|
| `hparam_summary.json` / `.txt` | 每个 (hparam_val, alpha) 的 ASR/FID 表格 |
| `hparam_{name}.png` | 折线图（左 ASR 右 FID，各 fixed_alpha 一条曲线） |
| `hparam_{name}_heatmap.png` | 热力图（`fixed_alphas` > 1 个时自动生成，横轴超参值 × 纵轴 alpha） |

---

## 示例命令 — merge.py

### CIFAR10 主实验

```bash
BD=./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT
CL=./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT
ALPHAS="0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0"
OUT="./merge_results"

# Weight Averaging
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method wa --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

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

# DARE（主实验固定 p=0.5；多 seed 用 ablation.py，见下方）
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method dare --dare_p 0.5 --dare_seed 42 --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

# SLERP
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method slerp --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT
```

### 拐点细扫

粗扫后发现 ASR 在某区间快速上升（如 0.85→0.92），补充密集点：

```bash
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method ties --ties_k 0.2 \
  --alphas 0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92 \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT
```

### CelebA-HQ 主实验

```bash
BD_HQ=./res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.3_GLASSES-CAT
CL_HQ=./res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_c1.0_p0.0_GLASSES-CAT
ALPHAS_HQ="0.0,0.5,0.8,0.9,0.95,1.0"

for METHOD in wa task_arithmetic slerp; do
  python merge.py --backdoor_ckpt $BD_HQ --clean_ckpt $CL_HQ \
    --method $METHOD --alphas $ALPHAS_HQ \
    --gpu 0 --fclip o --dataset CELEBA-HQ --trigger GLASSES --target CAT \
    --sample_n 1024 --eval_max_batch 64 --output_dir ./merge_results_celebahq
done

python merge.py --backdoor_ckpt $BD_HQ --clean_ckpt $CL_HQ \
  --method ties --ties_k 0.2 --alphas $ALPHAS_HQ \
  --gpu 0 --fclip o --dataset CELEBA-HQ --trigger GLASSES --target CAT \
  --sample_n 1024 --eval_max_batch 64 --output_dir ./merge_results_celebahq
```

### 快速验证（跳过 FID）

```bash
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method ties --ties_k 0.2 \
  --alphas 0.0,0.5,0.9,1.0 \
  --gpu 0 --fclip o --skip_fid \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 256 --eval_max_batch 256
```

---

## 示例命令 — ablation.py

### DARE 多 seed 稳定性

```bash
python ablation.py \
  --mode dare_seed \
  --backdoor_ckpt $BD --clean_ckpt $CL \
  --dare_p 0.5 --dare_seeds 42,123,777 \
  --alphas 0.0,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o --skip_fid \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 1000 --output_dir ./ablation_results
```

### TIES k 消融

```bash
python ablation.py \
  --mode hparam --method ties \
  --backdoor_ckpt $BD --clean_ckpt $CL \
  --hparam_name ties_k \
  --hparam_values 0.05,0.1,0.2,0.3,0.5,1.0 \
  --fixed_alphas 0.85,0.9 \
  --gpu 0 --fclip o --skip_fid \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 1000 --output_dir ./ablation_results
```

### DARE p 消融

```bash
python ablation.py \
  --mode hparam --method dare \
  --backdoor_ckpt $BD --clean_ckpt $CL \
  --hparam_name dare_p \
  --hparam_values 0.1,0.3,0.5,0.7,0.9 \
  --fixed_alphas 0.85,0.9 \
  --gpu 0 --fclip o --skip_fid \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 1000 --output_dir ./ablation_results
```

---

## 输出目录结构

### merge.py

```
merge_results/
└── merge_{method}_{model_name}_{poison_rate}/   # 不同方法自动区分，互不覆盖
    ├── args.json                 # 用户实际输入的参数
    ├── config.json               # 完整参数（含所有默认值）
    ├── real_{DATASET}/           # FID 真实图像参考集（跨 alpha 共用）
    ├── alpha0.0000/
    │   ├── clean/                # clean 采样图（用于 FID）
    │   ├── backdoor/             # backdoor 采样图（用于 ASR）
    │   ├── clean_grid.png        # 8×8 拼图预览
    │   └── backdoor_grid.png     # 8×8 拼图预览
    ├── alpha0.5000/ ...
    ├── merge_summary.json        # 机器可读结果
    ├── merge_summary.txt         # 人可读表格
    └── asr_fid_tradeoff.png      # ASR / FID 双轴曲线图
```

### ablation.py

```
ablation_results/
├── dare_seed_p0.5/                      # --mode dare_seed 输出
│   ├── real_{DATASET}/
│   ├── seed42_alpha0.9000/              # 各 (seed, alpha) 独立图像目录
│   │   ├── clean/
│   │   └── backdoor/
│   ├── seed123_alpha0.9000/ ...
│   ├── dare_seed_summary.json           # rows（各 seed 逐行）+ mean_rows（均值行）
│   ├── dare_seed_summary.txt            # 可读表格，各 alpha 末行为 mean±std
│   └── dare_seed_stability.png          # 均值线 + ±1std 色带 + 各 seed 细虚线
│
└── ties_ties_k_sweep/                   # --mode hparam 输出
    ├── real_{DATASET}/
    ├── hp0.05_alpha0.9000/
    │   ├── clean/
    │   └── backdoor/
    ├── hp0.2_alpha0.9000/ ...
    ├── hparam_summary.json
    ├── hparam_summary.txt
    ├── hparam_ties_k.png                # 折线图（左 ASR，右 FID）
    └── hparam_ties_k_heatmap.png        # 热力图（fixed_alphas > 1 时生成）
```

---

## 指标说明

| 指标 | 含义 | 解读方向 |
|------|------|----------|
| **FID** | clean 采样与真实图像的分布距离 | 越低 = 生成质量越好 |
| **MSE** | backdoor 采样与 target 图像的平均像素误差（归一化到 [0,1]） | 越低 = 后门越强 |
| **SSIM** | backdoor 采样与 target 图像的结构相似度（0~1） | 越高 = 后门越强 |
| **ASR** | Attack Success Rate：per-image MSE < `asr_threshold` 的比例 | 越低 = 防御越好 |

`merge_summary.txt` 输出示例：

```
   alpha |          fid |          mse |         ssim |          asr
----------------------------------------------------------------------
  0.0000 |    52.380000 |     0.240567 |     0.012345 |     0.000000
  0.5000 |    49.380000 |     0.240113 |     0.012401 |     0.000000
  0.8500 |    51.180000 |     0.238681 |     0.013210 |     0.000000
  0.9000 |    52.550000 |     0.217131 |     0.015678 |     0.019000
  0.9500 |    51.380000 |     0.110097 |     0.089234 |     0.382000
  1.0000 |    52.280000 |     0.015671 |     0.312456 |     0.912000
```

`dare_seed_summary.txt` 输出示例：

```
DARE seed stability  (p=0.5, seeds=[42, 123, 777])

   alpha |     seed |          FID |              MSE |             SSIM |              ASR
------------------------------------------------------------------------------------------
  0.9000 |       42 |          N/A |           0.2180 |           0.0156 |           0.0200
  0.9000 |      123 |          N/A |           0.2195 |           0.0161 |           0.0150
  0.9000 |      777 |          N/A |           0.2172 |           0.0149 |           0.0250
  0.9000 |     mean |          N/A | 0.2182±0.0009    | 0.0155±0.0005    | 0.0200±0.0041
------------------------------------------------------------------------------------------
```

---

## Alpha 语义对照表

所有方法统一：**alpha=0 偏干净模型，alpha=1 偏后门模型**。

| 方法 | alpha=0 | alpha=1 |
|------|---------|---------|
| `wa` | 完全等于 W_clean | 完全等于 W_backdoor |
| `task_arithmetic` | τ 完全施加（后门被覆盖） | W_backdoor（τ 未施加） |
| `ties` | τ_trimmed 完全施加 | W_backdoor |
| `dare` | τ_dare 完全施加 | W_backdoor |
| `slerp` | 插值到 W_clean | 插值到 W_backdoor |

> **注意**：`task_arithmetic` / `ties` / `dare` 在 alpha=0 时不保证与 W_clean 完全等同（task vector 操作存在截断/稀疏化），若需要严格的干净模型基线，请使用 `--method wa --alphas 0.0`。

---

## NeurIPS 实验设计

### 主实验（Table / Figure）

每种方法固定超参，跑完整 alpha sweep，CIFAR10 + CelebA-HQ 各一份。

| 方法 | 固定超参 | 推荐 alpha 列表 |
|------|----------|-----------------|
| WA | — | `0.0,0.2,0.5,0.8,0.85,0.87,0.9,0.92,0.95,0.97,0.99,1.0` |
| Task Arithmetic | — | 同上 |
| TIES | k=0.2 | 同上 |
| DARE | p=0.5，seeds={42,123,777} 均值 | 同上 |
| SLERP | — | 同上 |

超参选取的论文表述（一句话即可）：
> *We set k=0.2 following the original TIES paper and p=0.5 for DARE as a balanced drop rate; ablations on these choices are provided in Appendix A.*

### 附录消融（Appendix A）

| 实验 | 脚本命令 | 扫描范围 |
|------|----------|----------|
| TIES k 消融 | `ablation.py --mode hparam --method ties` | `ties_k ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}` |
| DARE p 消融 | `ablation.py --mode hparam --method dare` | `dare_p ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` |
| DARE seed 稳定性 | `ablation.py --mode dare_seed` | seeds={42, 123, 777} |

消融建议：`fixed_alphas` 选主实验 ASR 拐点前后各一个值（如 `0.85,0.9`），`sample_n=1000` 足够，可跳过 FID。

### 建议实验顺序

```
Day 1–2：WA + Task Arithmetic（CIFAR10，无额外超参，快速出结果）
Day 3–4：TIES k=0.2 + SLERP（CIFAR10）
Day 5：  DARE（用 ablation.py dare_seed 模式，三个 seed 自动汇总）
Day 6：  TIES k 消融 + DARE p 消融（ablation.py hparam，skip_fid）
Day 7–8：CelebA-HQ 重复所有主实验
Day 9：  拐点细扫（各方法在 ASR 快速上升区间补充密集点）
Day 10： 整理 JSON → 画论文图表
```

### 一键跑所有主实验（CIFAR10）

```bash
BD=./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT
CL=./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT
ALPHAS="0.0,0.2,0.5,0.8,0.85,0.9,0.95,1.0"
OUT="./merge_results"

# WA / Task Arithmetic / SLERP（无额外超参，循环跑）
for METHOD in wa task_arithmetic slerp; do
  python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
    --method $METHOD --alphas $ALPHAS \
    --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
    --sample_n 2048 --eval_max_batch 256 --output_dir $OUT
done

# TIES
python merge.py --backdoor_ckpt $BD --clean_ckpt $CL \
  --method ties --ties_k 0.2 --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --eval_max_batch 256 --output_dir $OUT

# DARE：ablation.py 自动跑三 seed 并逐行存储 + 追加均值行
python ablation.py --mode dare_seed \
  --backdoor_ckpt $BD --clean_ckpt $CL \
  --dare_p 0.5 --dare_seeds 42,123,777 --alphas $ALPHAS \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 2048 --output_dir $OUT/dare_stability
```

---

## 常见问题

**Q：显存不足（OOM）**
降低 `--eval_max_batch`。CelebA-HQ 256×256 建议 ≤ 64；CIFAR10 32×32 一般 256 无压力。

**Q：FID 绝对值偏高（约 50）**
`sample_n=1000` 时属正常现象（论文用 10K 约 7~14）。相对趋势有效，最终结果建议用 `sample_n=5000+`。

**Q：TIES k 和 DARE p 默认值怎么选**
主实验按原论文默认值（k=0.2，p=0.5）。用 `ablation.py --mode hparam` 跑完整消融，把结果放附录来 justify。

**Q：DARE 结果每次不一样**
正常，dropout mask 有随机性。用 `ablation.py --mode dare_seed` 跑多 seed，以 mean±std 汇报，单次结果不具代表性。

**Q：ablation.py 中途断掉怎么办**
`dare_seed` 模式每完成一个 (seed, alpha) 就立即写一次 JSON，重启后已有图像目录会被自动跳过（未设 `--force_resample` 时）。`hparam` 模式同理。

**Q：不同方法的输出目录会冲突吗**
不会。目录名自动包含方法名（`merge_wa_...`、`merge_ties_...`），各方法独立存储。

**Q：如何复现旧版 merge.py（WA）的结果**
直接用 `--method wa`，逻辑与旧版完全一致，结果可直接比对。
