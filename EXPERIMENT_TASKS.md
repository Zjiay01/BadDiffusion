# 实验任务清单

本文档记录当前 BadDiffusion merge-defense 项目的实验推进状态。完成一项就把 `[ ]` 改成 `[x]`，并尽量补充结果路径、关键指标或下一步说明。

## 0. 项目与代码基线

- [x] 明确当前实验以 `merge.py` 和 `merge_methods/` 为准，`README_merge.md` 已过时。
- [x] 将旧的 TIES / DARE / SLERP 主线暂时放下，当前 baseline 聚焦 diffusion-specific / diffusion-compatible 方法。
- [x] 在 `merge.py` 中加入并跑通 `diffusion_soup`、`dmm`、`maxfusion`、`anp`、`clean_finetune`。
- [x] 给 `merge.py` 加入 `--save_model`，支持保存防御后的模型目录。
- [x] 修复 checkpoint 路径解析和 MaxFusion/ensemble 保存元信息问题。
- [x] 给 FID 空目录增加显式错误提示，避免 DataLoader batch size 变成 0 的隐蔽报错。
- [x] 增加队列脚本和索引脚本：`scripts/run_baseline_queue.py`、`scripts/write_merge_result_index.py`。
- [x] 在 `AGENTS.md` 中记录长实验默认用 `tmux` 后台跑、heartbeat 定时检查。
- [x] 在 `AGENTS.md` 中记录整理实验文件时默认移动真实目录，不复制、不用软链接。

## 1. 当前 BadDiffusion 攻击与模型准备

- [x] 准备 clean checkpoint：`res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT_clean`。
- [x] 准备 backdoor checkpoint：`BOX_14 -> HAT`，`poison_rate=0.1`。
- [x] 准备 backdoor checkpoint：`BOX_11 -> CAT`，`poison_rate=0.1`。
- [x] 完成单后门 no-defense reference 测试。
- [x] 记录 no-defense ASR：
  - `BOX_14 -> HAT`: ASR `0.898438`
  - `BOX_11 -> CAT`: ASR `0.613281`
- [ ] 复查 clean / backdoor checkpoint 的单模型采样质量，保存代表性图片用于论文图。
- [ ] 为当前三个 checkpoint 补充统一的元信息记录：训练命令、seed、trigger、target、poison rate、训练轮数、数据集。

## 2. Baseline 场景 S1：Clean + Backdoor

场景定义：一个 clean model 与一个 backdoor model 合并。

- [x] 跑完 `clean + BOX_14-HAT` 的 5 个 baseline，`sample_n=1024`，`num_inference_steps=200`，`skip_fid=True`，`save_model=True`。
- [x] 跑完 `clean + BOX_11-CAT` 的 5 个 baseline，`sample_n=1024`，`num_inference_steps=200`，`skip_fid=True`，`save_model=True`。
- [x] 汇总 S1 ASR / MSE / SSIM 到 `EXPERIMENT_RESULTS.md`。
- [x] 将 S1 结果移动到服务器 `merge_results/final_s1_*`。
- [ ] 对 S1 关键方法补跑 FID。
- [ ] 对 S1 关键方法补跑更多采样数或更多 seed，确认 ASR=0 是否稳定。
- [ ] 选择 S1 的代表性 qualitative samples，整理为论文图。

## 3. Baseline 场景 S2：Backdoor + Backdoor

场景定义：两个 backdoor model 合并，并分别评估两个 trigger/target。

- [x] 跑完 `BOX_14-HAT + BOX_11-CAT` 下评估 `BOX_14 -> HAT` 的 saved-model baseline。
- [x] 跑完 `BOX_14-HAT + BOX_11-CAT` 下评估 `BOX_11 -> CAT` 的 saved-model baseline。
- [x] 对 S2 关键结果补跑 `sample_n=1024` recheck：
  - `final_s2_hat_diffusion_soup_1024`
  - `final_s2_hat_dmm_1024`
  - `final_s2_hat_maxfusion_1024`
  - `final_s2_cat_maxfusion_1024`
- [x] 记录当前最明显的残留后门现象：`s2_hat + maxfusion` 仍有高 ASR，`final_s2_hat_maxfusion_1024` ASR `0.891602`。
- [x] 将 S2 结果移动到服务器 `merge_results/save_s2_*` 和 `merge_results/final_s2_*_1024`。
- [ ] 对 S2 关键方法补跑 FID。
- [ ] 对 S2 中 MaxFusion 的高 ASR 现象做机制分析：为什么 HAT 保留而 CAT 不保留。
- [ ] 对 S2 补更多随机 seed 或更多 trigger pair，确认结论是否稳定。

## 4. 结果整理与可复现性

- [x] 清理临时目录：`wandb/`、`fid_smoke_s2_hat_soup/`、`debug_outputs/`、`defended_models/`、`__pycache__/`。
- [x] 将正式结果从服务器项目根目录移动到 `merge_results/`，不复制、不使用软链接。
- [x] 将旧实验输出移动到 `legacy_outputs/`：`result/`、`measure/`、`test/`、`model_old/`。
- [x] 删除旧的根目录 `merge_result_index/`，当前索引统一保存在 `merge_results/index.json` 和 `merge_results/README.md`。
- [x] 服务器 `merge_results/` 当前索引 26 条结果：24 个 baseline + 2 个 no-defense。
- [x] 本地新增 `EXPERIMENT_RESULTS.md` 汇总当前实验结果。
- [ ] 将服务器 `merge_results/README.md` 和 `index.json` 定期拉回本机或纳入备份策略。
- [ ] 为每次新实验建立固定命名规则，避免后续再次手工整理。

## 5. FID 与生成质量评估

- [x] 用小规模 smoke test 验证 `merge.py` 的 FID 分支可以跑通。
- [ ] 选择最终 FID 设置：`sample_n`、`num_inference_steps`、`fid_batch_size`、是否固定 real reference。
- [ ] 为 no-defense、S1 baseline、S2 baseline 统一补跑 FID。
- [ ] 保存 clean samples / backdoor samples / target images 的可视化网格。
- [ ] 统计 runtime 和额外训练成本，尤其是 `dmm`、`anp`、`clean_finetune`。

## 6. 后续攻击方法扩展

- [ ] 复现或接入 VillanDiffusion，优先保持 CIFAR10-DDPM / diffusers checkpoint 格式。
- [ ] 检查 VillanDiffusion checkpoint 是否满足 `DDPMPipeline.from_pretrained(path)`。
- [ ] 跑 VillanDiffusion 单模型 ASR，确认攻击有效后再进入 merge defense。
- [ ] 将 VillanDiffusion 加入 S1：`clean + VillanDiffusion backdoor`。
- [ ] 将 VillanDiffusion 加入 S2：`BadDiffusion backdoor + VillanDiffusion backdoor`。
- [ ] 调研并决定 TrojDiff 的接入方式：转换 checkpoint、重训，或单独实验组。
- [x] 设计并跑通 BadMerging-style diffusion adaptive attack，作为攻击者知道会 merge 的自适应威胁模型。
  - 当前成功 checkpoint：服务器 `merge_results/badmerge_cifar10_box14_hat_paired_strong2000/final`。
  - 快速确认：`badmerge_confirm_strong2000_n512` 在 `alpha=0.5` 下 ASR `0.912109`。
  - 快速 baseline：`diffusion_soup` ASR `0.890625`，`dmm` ASR `0.554688`，`maxfusion/anp/clean_finetune` ASR `0.0`。
- [ ] 对 BadMerging-style adaptive attack 补跑 FID-enabled 正式评估。
  - 当前进行中：服务器 `merge_results/badmerge_fid1000_*`，`sample_n=1024`，`num_inference_steps=1000`，FID enabled。
- [ ] 设计 BadBlocks-style DDPM 或单独 Stable Diffusion 实验组。

## 7. 我们自己的 Merge 防御方法

- [ ] 明确新方法要同时覆盖 S1 和 S2。
- [ ] 明确新方法要覆盖 merge-aware adaptive threat，例如 BadMerging-style 攻击。
- [ ] 设计核心判别/修复信号：例如 score disagreement、trigger-sensitive direction、clean-data consistency、activation/channel pruning 或多模型一致性。
- [ ] 实现新方法的最小可运行版本，并接入 `merge.py` 或新的方法模块。
- [ ] 在当前 BadDiffusion S1/S2 上先跑 debug 实验。
- [ ] 与 5 个 baseline 做同表对比。
- [ ] 做 ablation：去掉关键组件后 ASR / MSE / SSIM / FID 的变化。
- [ ] 写出方法假设、适用范围和失败场景。

## 8. 论文级实验与写作

- [ ] 固定最终表格结构：攻击方法、场景、合并方法、防御方法、ASR、FID、MSE、SSIM、runtime。
- [ ] 固定最终图结构：ASR-FID tradeoff、qualitative samples、方法流程图、ablation。
- [ ] 补充更多数据集或更大模型实验：CelebA-HQ 或 Stable Diffusion / LDM。
- [ ] 补跑 CelebA-HQ S2。当前 CelebA-HQ 只完成 no-defense 与 S1；S2 (`backdoor + backdoor`) 尚未运行。
- [ ] 总结 baseline 观察：多数 WA/repair/pruning baseline 可以压低当前 ASR，但 MaxFusion 在 S2/HAT 上保留后门。
- [ ] 写实验设置：dataset、model architecture、attack config、merge config、evaluation config。
- [ ] 写实验结果分析：为什么现有 baseline 不是通用防御，为什么新方法能覆盖 S1/S2/adaptive。
- [ ] 整理复现实验命令，保证论文附录或 README 可直接复跑。
