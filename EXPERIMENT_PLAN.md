# 实验计划：面向后门扩散模型的 Merge 防御

本文档定义当前项目的实验结构，用于评估在扩散模型合并时的后门防御效果。后续即使加入 VillanDiffusion、TrojDiff、BadBlocks 风格攻击、BadMerging 风格自适应攻击，也应尽量复用同一套模型格式、合并脚本和评估流程。

## 实验目标

目标是在模型合并场景下，评估不同 baseline 和我们的方法是否能够降低后门行为，同时尽量保持 clean generation 质量。

实验场景只分为两类：

1. `clean + backdoor`
2. `backdoor + backdoor`

BadMerging / merge-aware attack 不再作为第三种场景，而是作为一种攻击方法放入上述两类场景中测试。

## 核心原则

所有实验应尽量围绕 diffusion model 本身展开，不使用 TIES、DARE、SLERP 这类主要面向一般神经网络或大模型参数合并的 baseline。当前 baseline 应优先选择能够作用于扩散模型生成过程、噪声预测网络、score function 或 clean-data repair 的方法。

当前 `README_merge.md` 已经过时，后续实验和复现均以 `merge.py` 以及 `merge_methods/` 中的实现为准。

## 模型格式要求

为了让不同攻击方法得到的 backdoor model 可以直接进入同一套 merge defense 流程，每种攻击最终都应导出 HuggingFace `diffusers` pipeline 格式：

```text
checkpoint_dir/
+-- model_index.json
+-- scheduler/
+-- unet/
    +-- config.json
    +-- diffusion_pytorch_model.bin
```

当前运行入口是：

```bash
python merge.py --method <method> --model_ckpts <ckpt1>,<ckpt2>,...
```

当前 BadDiffusion/DDPM 设置中，主要合并对象是 UNet。如果后续扩展到 Stable Diffusion，需要额外记录合并对象到底是 UNet、text encoder、VAE、LoRA/adapters，不能和 CIFAR10-DDPM 的实验混在同一个表格里直接比较。

## 攻击方法

主实验攻击集合建议如下：

| ID | 攻击方法 | 作用 |
|---|---|---|
| A1 | BadDiffusion | 当前基础攻击，也是 sanity check benchmark |
| A2 | VillanDiffusion | 更通用的 diffusion backdoor 框架 |
| A3 | TrojDiff | 另一类 Trojan diffusion backdoor 机制 |
| A4 | BadBlocks-style | 用模块/块级后门测试防御对局部后门的鲁棒性 |
| A5 | BadMerging-style merge-aware attack | 攻击者知道会发生 merge 的自适应威胁模型 |

每种攻击至少需要导出一个 clean checkpoint 和一个或多个 backdoor checkpoint。对于 S2 场景，应至少准备两个不同 seed、不同 trigger/target 或不同攻击方法得到的 backdoor checkpoints。

## 攻击方法的数据集与兼容性注意事项

不同论文原始使用的数据集和代码格式不完全一致，不能只看攻击名字就直接纳入同一个实验。

| 攻击方法 | 原始/常见数据集 | 与当前 BadDiffusion-DDPM-CIFAR10 的兼容性 |
|---|---|---|
| BadDiffusion | MNIST、CIFAR10、CelebA-HQ | 当前项目基础方法，DDPM-CIFAR10 最直接兼容 |
| VillanDiffusion | MNIST、CIFAR10、CelebA-HQ、LDM-CelebA-HQ | CIFAR10-DDPM 设置较容易对齐，适合作为第二阶段主要攻击 |
| TrojDiff | CIFAR10、CelebA | 原始实现可能不是 `diffusers` 格式，需要转换或重新封装 |
| BadBlocks | 主要面向 text-to-image / Stable Diffusion，例如 SD v1.5、SD v2.1 等 | 不适合直接和 CIFAR10-DDPM 混跑；若要纳入当前实验，应实现 BadBlocks-style DDPM 版本，或单独开 Stable Diffusion 实验组 |
| BadMerging | 原始工作主要不是 diffusion generation，而是 merge-aware 后门思想 | 需要实现 BadMerging-style diffusion adaptive attack，不能直接把原论文分类模型 checkpoint 当作 diffusion baseline |

建议优先使用 CIFAR10 统一跑第一阶段实验，因为它训练和采样成本低，便于快速验证 merge defense 是否有信号。缺点是数据集较小、生成分辨率低，论文最终结果最好再补充 CelebA-HQ 或 Stable Diffusion 级别实验，证明方法不只在 CIFAR10 上有效。

## Backdoor Model 一致性检查

判断其他攻击生成的 backdoor model 是否和 BadDiffusion 生成的 backdoor model “一致”，不是看攻击算法是否相同，而是看它是否满足同一个 merge/evaluation 接口。

最低要求：

1. `DDPMPipeline.from_pretrained(path)` 可以成功加载。
2. `unet.config.sample_size`、`in_channels`、`out_channels` 等关键结构一致。
3. `unet.state_dict()` 的 key 集合一致。
4. 对应 tensor shape 一致。
5. trigger、target、dataset 和 evaluation protocol 明确记录。
6. 单个 backdoor model 在不 merge 时 ASR 应该足够高，否则不能作为有效攻击样本。

可以用如下脚本快速检查两个 checkpoint 是否能进入同一套 merge：

```python
from diffusers import DDPMPipeline

a = DDPMPipeline.from_pretrained("./res_bad_diffusion")
b = DDPMPipeline.from_pretrained("./res_other_attack")

sa = a.unet.state_dict()
sb = b.unet.state_dict()

missing = set(sa) ^ set(sb)
shape_diff = [
    (k, sa[k].shape, sb[k].shape)
    for k in sa.keys() & sb.keys()
    if sa[k].shape != sb[k].shape
]

print("key mismatch:", len(missing))
print("shape mismatch:", len(shape_diff))
```

只有 key mismatch 和 shape mismatch 都为 0，才适合直接做 checkpoint-level merge。若结构不同，则需要转换、重训，或者把它放到单独实验组。

## 推荐推进顺序

第一阶段：只做 CIFAR10-DDPM。

```text
BadDiffusion + VillanDiffusion
```

这一步重点是确认 `merge.py`、baseline、防御方法、ASR/FID 评估流程全部跑通。

第二阶段：加入 TrojDiff。

```text
TrojDiff -> diffusers DDPM checkpoint
```

这一步重点是做模型格式转换和协议对齐。

第三阶段：加入自适应或局部化攻击。

```text
BadBlocks-style DDPM
BadMerging-style diffusion adaptive attack
```

这一步重点是检验我们的方法是否能防御“攻击者知道你会 merge”的情况。

第四阶段：如果时间和算力允许，扩展到更大数据集或 Stable Diffusion。

```text
CelebA-HQ
Stable Diffusion / LDM
```

这一步更适合作为论文最终补强实验，而不是最开始就做。

## 合并场景

### S1：Clean + Backdoor

将一个可信 clean model 与一个 attacked model 合并：

```text
M_clean + M_backdoor
```

推荐 checkpoint 组合：

| 攻击方法 | Model 1 | Model 2 |
|---|---|---|
| BadDiffusion | clean DDPM-CIFAR10 checkpoint | BadDiffusion DDPM-CIFAR10 checkpoint |
| VillanDiffusion | clean DDPM-CIFAR10 checkpoint | VillanDiffusion DDPM-CIFAR10 checkpoint |
| TrojDiff | clean DDPM-CIFAR10 checkpoint | TrojDiff DDPM-CIFAR10 checkpoint |
| BadBlocks-style | clean DDPM-CIFAR10 checkpoint | block-localized backdoor checkpoint |
| BadMerging-style | clean DDPM-CIFAR10 checkpoint | merge-aware backdoor checkpoint |

示例命令：

```bash
python merge.py \
  --method diffusion_soup \
  --model_ckpts ./res_clean,./res_bad \
  --model_weights 0.5,0.5 \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 256 --num_inference_steps 200 --skip_fid --gpu 0
```

### S2：Backdoor + Backdoor

将两个或多个 attacked models 合并：

```text
M_backdoor_1 + M_backdoor_2 [+ M_backdoor_3 ...]
```

该场景需要覆盖 same-trigger 和 mixed-trigger 设置。

| 设置 | Model 1 | Model 2 | 目的 |
|---|---|---|---|
| 同攻击、同 trigger/target | BadDiffusion BOX_14-HAT | BadDiffusion BOX_14-HAT another seed | 测试同类后门是否被保留或放大 |
| 同攻击、不同 trigger/target | BadDiffusion BOX_14-HAT | BadDiffusion GLASSES-CAT | 测试不同后门之间的干扰 |
| 不同攻击 | BadDiffusion | VillanDiffusion or TrojDiff | 测试跨攻击合并行为 |
| 自适应攻击 + 标准攻击 | BadMerging-style | BadDiffusion or VillanDiffusion | 测试 merge-aware 威胁模型 |
| 多后门 | BadDiffusion + VillanDiffusion + TrojDiff | optional | 测试超过两个模型时的扩展性 |

示例命令：

```bash
python merge.py \
  --method diffusion_soup \
  --model_ckpts ./res_bad1,./res_bad2 \
  --model_weights 0.5,0.5 \
  --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 256 --num_inference_steps 200 --skip_fid --gpu 0
```

多模型合并时，Diffusion Soup 主实验建议只使用均匀平均权重：

```bash
python merge.py --method diffusion_soup \
  --model_ckpts ./res_bad1,./res_bad2,./res_bad3 \
  --dataset CIFAR10 --trigger BOX_14 --target HAT --gpu 0
```

如果后续需要 ablation，再额外加入非均匀权重：

```bash
python merge.py --method diffusion_soup \
  --model_ckpts ./res_bad1,./res_bad2,./res_bad3 \
  --model_weights 0.2,0.3,0.5 \
  --dataset CIFAR10 --trigger BOX_14 --target HAT --gpu 0
```

## Merge Baselines

这些方法通过 `merge.py` 和 `merge_methods/` 运行。

| ID | 方法 | `--method` | 输出类型 |
|---|---|---|---|
| M1 | Diffusion Soup / checkpoint average | `diffusion_soup` | 单个 merged checkpoint pipeline |
| M2 | DMM-style multi-teacher distillation | `dmm` | student pipeline |
| M3 | MaxFusion-style score fusion | `maxfusion` | inference-time fused pipeline |

Diffusion Soup 主实验只使用均匀平均权重。这样可以避免 alpha scanning 带来的额外搜索空间，也更适合作为清晰 baseline。

## Defense Baselines

这些方法不是纯粹的 merge method，但可以作为防御 baseline。

| ID | 方法 | `--method` | 说明 |
|---|---|---|---|
| D1 | ANP-style pruning | `anp` | 当前实现会根据 trigger-sensitive conv channels 进行评分和剪枝 |
| D2 | Clean Fine-Tuning | `clean_finetune` | 从 soup 初始化出发，用 clean data 进行修复 |

Fine-Pruning / channel pruning 后续可以作为额外 defense baseline 加入，但不是当前第一批必须项。

## 指标与检测方式

| 指标 | 含义 | 趋势 |
|---|---|---|
| ASR | 攻击成功率，当前用 generated sample 与 target 的 per-image MSE threshold 判断 | 越低越好 |
| MSE-to-target | backdoor samples 与 target image 的平均 MSE | 越高越安全 |
| SSIM-to-target | backdoor samples 与 target image 的结构相似度 | 越低越安全 |
| FID | clean generation quality | 越低越好 |
| Runtime / extra training cost | 方法额外成本 | 越低越好 |

当前 `merge.py` 中的 ASR 定义是：

```text
ASR = mean(per_image_mse(generated_backdoor_sample, target) < asr_threshold)
```

ASR 没有完全绕开采样的快速检测方式。可以用较小 `sample_n` 和较少采样步数做快速估计，但最终可信结果仍需要通过触发采样计算。建议流程是：

1. 先用 `sample_n=64`、`num_inference_steps=100` 做 debug ASR。
2. 候选方法用 `sample_n=256` 或 `512` 做中等规模筛选。
3. 最终表格用 `sample_n>=1000` 和完整/接近完整采样步数。
4. FID 只在最终或接近最终实验中打开，避免调试阶段耗时过长。

## 快速调试设置

方法调试：

```bash
--sample_n 64
--num_inference_steps 100
--skip_fid
```

候选筛选：

```bash
--sample_n 256
--num_inference_steps 200
--skip_fid
```

最终论文结果：

```bash
--sample_n 1000
--num_inference_steps 1000
```

## 主结果表

推荐主表结构：

```text
Attack x Scenario x Method
```

行：

```text
A1 BadDiffusion, S1 clean+backdoor
A1 BadDiffusion, S2 backdoor+backdoor
A2 VillanDiffusion, S1 clean+backdoor
A2 VillanDiffusion, S2 backdoor+backdoor
...
A5 BadMerging-style, S1 clean+backdoor
A5 BadMerging-style, S2 backdoor+backdoor
```

列：

```text
No defense
Diffusion Soup
DMM
MaxFusion
ANP
Clean Fine-Tuning
Ours
```

至少报告：

```text
ASR, MSE, SSIM, FID
```

## Checkpoint 命名规范

名称应编码 dataset、attack、trigger、target、poison rate 和 seed。

推荐格式：

```text
res_{base}_{dataset}_{attack}_p{poison_rate}_{trigger}-{target}_seed{seed}
```

示例：

```text
res_DDPM-CIFAR10-32_CIFAR10_clean_p0.0_NONE-NONE_seed0
res_DDPM-CIFAR10-32_CIFAR10_BadDiffusion_p0.1_BOX_14-HAT_seed0
res_DDPM-CIFAR10-32_CIFAR10_VillanDiffusion_p0.1_BOX_14-HAT_seed0
res_DDPM-CIFAR10-32_CIFAR10_TrojDiff_p0.1_BOX_14-HAT_seed0
res_DDPM-CIFAR10-32_CIFAR10_BadMerging_p0.1_BOX_14-HAT_seed0
```

S2 mixed-trigger 实验必须在输出目录或 run config 中明确记录本次评估的 trigger/target。如果 merged model 可能包含多个后门，应分别对每个 trigger/target pair 计算 ASR，不能只报告一个平均值。

## 每个实验必须记录的信息

每个场景都需要写清楚具体 checkpoint，而不是只写攻击方法名字。至少记录：

```text
base model
dataset
attack method
trigger
target
poison rate
training seed
checkpoint path
merge weights
evaluation trigger/target
sample_n
num_inference_steps
asr_threshold
```

这对 S2 特别重要，因为 `backdoor + backdoor` 可能表示：

```text
same trigger, same target
same trigger, different target
different trigger, same target
different trigger, different target
different attacks
merge-aware + non-adaptive attack
```

这些是不同实验条件，不能合并到同一行结果里。

## 当前实现注意事项

当前 `merge.py` 已经加入以下方法：

```text
diffusion_soup
dmm
maxfusion
anp
clean_finetune
```

需要注意：

1. `diffusion_soup` 是 checkpoint-level average，主实验建议只用均匀平均。
2. `dmm` 当前是 DMM-style multi-teacher score distillation baseline，不等同于完整复现所有论文细节。
3. `maxfusion` 当前是 inference-time score fusion baseline，不是完整中间特征融合系统。
4. `anp` 当前是 ANP-style trigger-sensitive pruning baseline，不是原 ANP 代码逐行复刻。
5. `clean_finetune` 需要 clean data，因此它的成本和数据可用性要单独报告。
6. 旧的 TIES、DARE、SLERP 已不作为当前主实验 baseline。
7. `Ablation.py` 可能仍依赖旧版 `merge.py` 接口，后续如需使用要单独更新。

## 最小可行实验组合

第一批最小可行实验建议：

```text
Dataset: CIFAR10
Base model: DDPM-CIFAR10-32
Attacks: BadDiffusion, VillanDiffusion
Scenarios: S1, S2
Methods: No defense, Diffusion Soup, DMM, MaxFusion, ANP, Clean Fine-Tuning, Ours
Metrics: ASR, MSE, SSIM, FID
```

如果这一批实验中我们的方法能在 S1、S2 上稳定降低 ASR，并且 clean FID 没有明显崩坏，再扩展到 TrojDiff、BadBlocks-style、BadMerging-style。
