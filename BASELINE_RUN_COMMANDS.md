# 当前阶段实验执行命令清单

本文档记录当前阶段需要先跑的实验。目标是先准备 CIFAR10 上的 clean/backdoor checkpoints，并验证单个后门模型有效，然后再开始第一批 merge baseline。

## 0. 进入项目并同步代码

进入实验室服务器上的项目目录：

```bash
cd /path/to/BadDiffusion
```

同步代码：

```bash
git pull
```

关闭 wandb，避免训练中弹出交互：

```bash
export WANDB_MODE=disabled
```

如果是在 Windows PowerShell：

```powershell
$env:WANDB_MODE="disabled"
```

## 1. 准备 Clean 模型

如果直接使用 HuggingFace/diffusers 预训练 clean model，后续命令中可以直接写：

```text
DDPM-CIFAR10-32
```

如果希望自己保存一份 clean checkpoint，运行：

```bash
python baddiffusion.py --project merge_defense --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.0 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o --postfix clean_p0 -o --gpu 0
```

预计输出目录：

```text
res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT_clean_p0
```

## 2. 训练第一个 BadDiffusion 后门模型

主后门模型：`BOX_14 -> HAT`。

```bash
python baddiffusion.py --project merge_defense --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o --postfix bd_box14_hat -o --gpu 0
```

预计输出目录：

```text
res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat
```

## 3. 验证第一个后门模型

单独测量 `BOX_14 -> HAT` 后门模型：

```bash
python baddiffusion.py --project merge_defense --mode measure --dataset CIFAR10 --eval_max_batch 256 --trigger BOX_14 --target HAT --ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --fclip o -o --gpu 0
```

生成样本：

```bash
python baddiffusion.py --project merge_defense --mode sampling --ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --fclip o --gpu 0
```

## 4. 训练第二个 BadDiffusion 后门模型

用于 `backdoor + backdoor` mixed-trigger 场景：`BOX_11 -> CAT`。

```bash
python baddiffusion.py --project merge_defense --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_11 --target CAT --ckpt DDPM-CIFAR10-32 --fclip o --postfix bd_box11_cat -o --gpu 0
```

预计输出目录：

```text
res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat
```

## 5. 验证第二个后门模型

```bash
python baddiffusion.py --project merge_defense --mode measure --dataset CIFAR10 --eval_max_batch 256 --trigger BOX_11 --target CAT --ckpt ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat --fclip o -o --gpu 0
```

## 6. 第一批 Merge Baseline：S1 Clean + Backdoor

场景：

```text
clean DDPM-CIFAR10-32 + BadDiffusion BOX_14-HAT
```

先使用 debug 设置：

```text
sample_n=64
num_inference_steps=100
skip_fid=True
```

Diffusion Soup：

```bash
python merge.py --method diffusion_soup --model_ckpts DDPM-CIFAR10-32,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

MaxFusion：

```bash
python merge.py --method maxfusion --model_ckpts DDPM-CIFAR10-32,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

ANP：

```bash
python merge.py --method anp --model_ckpts DDPM-CIFAR10-32,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

Clean Fine-Tuning：

```bash
python merge.py --method clean_finetune --model_ckpts DDPM-CIFAR10-32,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

DMM：

```bash
python merge.py --method dmm --model_ckpts DDPM-CIFAR10-32,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

## 7. 第一批 Merge Baseline：S2 Backdoor + Backdoor

场景：

```text
BadDiffusion BOX_14-HAT + BadDiffusion BOX_11-CAT
```

同一个 merged model 需要分别评估两个 trigger/target。

先评估 `BOX_14 -> HAT`：

```bash
python merge.py --method diffusion_soup --model_ckpts ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_14 --target HAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

再评估 `BOX_11 -> CAT`：

```bash
python merge.py --method diffusion_soup --model_ckpts ./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat,./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat --model_weights 0.5,0.5 --dataset CIFAR10 --trigger BOX_11 --target CAT --sample_n 64 --num_inference_steps 100 --skip_fid --gpu 0
```

确认 Diffusion Soup 跑通后，把上面两条命令中的：

```text
--method diffusion_soup
```

依次替换为：

```text
--method maxfusion
--method anp
--method clean_finetune
--method dmm
```

## 8. 当前优先级

当前最优先跑：

```text
1. BadDiffusion BOX_14-HAT 训练
2. BadDiffusion BOX_14-HAT 单模型 measure
3. BadDiffusion BOX_11-CAT 训练
4. BadDiffusion BOX_11-CAT 单模型 measure
```

只有当单个 backdoor checkpoint 的后门行为有效后，才开始跑 merge baseline。否则 merge baseline 的 ASR 结果没有可靠参照。
