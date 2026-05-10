# Agent Handoff Notes

This repository is being used for experiments on merge-time defenses for backdoored diffusion models. The user works in Chinese; answer in Chinese unless they ask otherwise.

## Current Snapshot For New Sessions

Read this section first. It is the compact current state of the project.

- Local workspace: `D:\Safe Merge\BadDiffusion`.
- Lab server project: `/home1/zhln/code/BadDiffusion`.
- Lab server Python: `/home1/zhln/envs/baddiffusion/bin/python`.
- SSH from local workspace: `ssh -i '.codex_ssh\baddiffusion_server_ed25519' -o UserKnownHostsFile=ssh_known_hosts zhln@192.168.121.242`.
- For GPU jobs, set `CUDA_VISIBLE_DEVICES=<gpu>` outside the Python command. Do not rely on `merge.py --gpu` for isolation because CUDA visibility can be affected by import timing.
- Current source of truth for merge-defense code: `merge.py` and `merge_methods/`.
- Current task tracker: `EXPERIMENT_TASKS.md`.
- Current result summary: `EXPERIMENT_RESULTS.md`.
- Server result root: `/home1/zhln/code/BadDiffusion/merge_results/`.
- `merge_results/` now contains real result directories, not symlinks and not copied duplicates. Keep using direct moves for future organization.
- Old pre-current outputs were moved to `/home1/zhln/code/BadDiffusion/legacy_outputs/`.
- The root-level `merge_result_index/` was removed. The active generated index is `merge_results/index.json` and `merge_results/README.md`.
- Check the latest pushed code state with `git log -1 --oneline`; this file is kept in GitHub with the project.

Current checkpoints on the server:

```text
/home1/zhln/code/BadDiffusion/res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT_clean
/home1/zhln/code/BadDiffusion/res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat
/home1/zhln/code/BadDiffusion/res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat
/home1/zhln/code/BadDiffusion/merge_results/badmerge_cifar10_box14_hat_paired_strong2000/final
```

Completed current baseline results:

- No-defense references are complete:
  - `BOX_14 -> HAT`: ASR `0.898438`
  - `BOX_11 -> CAT`: ASR `0.613281`
- S1 `clean + backdoor` baseline is complete for both `BOX_14-HAT` and `BOX_11-CAT`.
- S2 `backdoor + backdoor` baseline is complete for `BOX_14-HAT + BOX_11-CAT`, evaluated under both triggers.
- `merge_results/` currently indexes 26 entries: 24 baseline/selected recheck runs plus 2 no-defense runs.
- Important observation: most baselines suppress current ASR to `0`, but `s2_hat + maxfusion` preserves the backdoor strongly:
  - `save_s2_hat_maxfusion`: ASR `0.902344`
  - `final_s2_hat_maxfusion_1024`: ASR `0.891602`
- FID-enabled CIFAR10 baseline results for BadDiffusion S1/S2 are recorded in `EXPERIMENT_RESULTS.md`.
- CelebA-HQ currently has only no-defense and S1 (`clean + backdoor`) results. CelebA-HQ S2 has not been run.
- BadMerging-style diffusion adaptive attack is implemented and has a successful CIFAR10 checkpoint:
  - `merge_results/badmerge_cifar10_box14_hat_paired_strong2000/final`
  - quick `alpha=0.5`, `sample_n=512`, `skip_fid=True` confirmation ASR: `0.912109`
  - quick defense baselines: diffusion_soup ASR `0.890625`, DMM ASR `0.554688`, MaxFusion/ANP/Clean Fine-Tuning ASR `0.0`
  - FID-enabled `sample_n=1024`, `num_inference_steps=1000` runs are under `merge_results/badmerge_fid1000_*` when active.

Immediate next useful work:

- Back up or periodically pull `merge_results/index.json` and `merge_results/README.md`.
- Finish/aggregate BadMerging FID-enabled baseline runs if `merge_results/badmerge_fid1000_*` is active.
- Decide whether to run CelebA-HQ S2 first as a smoke test or full FID run.
- Generate qualitative sample grids for clean, backdoor, and defended outputs.
- Investigate why `MaxFusion` preserves `BOX_14 -> HAT` in S2 but not `BOX_11 -> CAT`.
- Add another attack method next, preferably VillanDiffusion on CIFAR10/DDPM in diffusers-compatible format.
- Start designing the new general merge defense after the baseline/FID table is stable.

## Current Research Goal

The project is being refocused from general model-merging baselines to diffusion-model-specific merge defenses.

Main goal:

- Evaluate baseline methods for defending merged diffusion models against backdoors.
- Then design a new merge defense that works across:
  1. `clean + backdoor`
  2. `backdoor + backdoor`

BadMerging / merge-aware attacks are treated as attack methods, not a third scenario.

## Important Project State

`README_merge.md` is stale. Treat `merge.py` and `merge_methods/` as the current source of truth for merge-defense experiments.

The current merge/defense baselines are:

- `diffusion_soup`
- `dmm`
- `maxfusion`
- `anp`
- `clean_finetune`

Old methods such as TIES, DARE, and SLERP are no longer the main baselines for the current experiment plan.

`Ablation.py` may still depend on the older `merge.py` interface and should be treated as stale until updated.

## Key Files

- `merge.py`: current experiment runner for merge baselines and defense baselines.
- `merge_methods/`: method implementations used by `merge.py`.
- `baddiffusion.py`: original BadDiffusion training/sampling/measure script.
- `dataset.py`: dataset, trigger, and target handling.
- `EXPERIMENT_PLAN.md`: Chinese experiment plan and protocol.
- `BASELINE_RUN_COMMANDS.md`: current single-line commands for the first-stage experiments.
- `README.md`: original BadDiffusion usage notes.
- `README_merge.md`: stale; do not rely on it.

## Current Baseline Implementation Notes

- `diffusion_soup` is checkpoint-level weighted averaging. For main experiments, use uniform average weights only.
- `dmm` is a DMM-style multi-teacher score distillation baseline, not a full paper reproduction.
- `maxfusion` is an inference-time score fusion baseline, not a full intermediate-feature fusion system.
- `anp` is an ANP-style trigger-sensitive pruning baseline, not the original ANP implementation line by line.
- `clean_finetune` starts from soup initialization and repairs on clean data.

## First Experiment Stage

The first BadDiffusion attack/checkpoint preparation stage is complete. The notes below describe the original first-stage setup and remain useful for interpreting the existing results.

First-stage dataset:

```text
CIFAR10
```

First-stage base model:

```text
DDPM-CIFAR10-32
```

First backdoor checkpoints to prepare:

```text
BadDiffusion BOX_14 -> HAT, poison_rate=0.1
BadDiffusion BOX_11 -> CAT, poison_rate=0.1
```

Clean model options:

- Use pretrained `DDPM-CIFAR10-32` directly, or
- Train/save a clean checkpoint with `poison_rate=0.0`.

Single-backdoor no-defense references have been run and show meaningful ASR. See `EXPERIMENT_RESULTS.md`.

## First Merge Scenarios

S1:

```text
clean DDPM-CIFAR10-32 + BadDiffusion BOX_14-HAT
```

S2:

```text
BadDiffusion BOX_14-HAT + BadDiffusion BOX_11-CAT
```

For S2 mixed-trigger experiments, evaluate both trigger/target pairs separately:

```text
BOX_14 -> HAT
BOX_11 -> CAT
```

## Metrics

Track at least:

- ASR
- MSE-to-target
- SSIM-to-target
- FID
- runtime / extra training cost

Fast debug setting:

```text
sample_n=64
num_inference_steps=100
skip_fid=True
```

Medium screening setting:

```text
sample_n=256 or 512
num_inference_steps=200
skip_fid=True
```

Final paper setting:

```text
sample_n>=1000
num_inference_steps=1000
FID enabled
```

## Attack Expansion Plan

After BadDiffusion is fully working, expand in this order:

1. VillanDiffusion on CIFAR10/DDPM.
2. TrojDiff, after converting or wrapping checkpoints into `diffusers` format.
3. BadMerging-style diffusion adaptive attack.
4. BadBlocks-style DDPM or separate Stable Diffusion experiment group.

Compatibility check for any new attack checkpoint:

- `DDPMPipeline.from_pretrained(path)` must load.
- UNet config must match the current DDPM-CIFAR10 setup.
- UNet `state_dict` keys must match.
- Tensor shapes must match.
- Trigger/target/evaluation protocol must be recorded.
- Single-model ASR must be high enough before merge-defense experiments.

## Server Information

Lab server information:

```text
ssh zhln@192.168.121.242
project path: /home1/zhln/code/BadDiffusion
conda env: /home1/zhln/envs/baddiffusion
```

Use the workspace SSH key from `D:\Safe Merge`:

```text
ssh -i '.codex_ssh\baddiffusion_server_ed25519' -o UserKnownHostsFile=ssh_known_hosts zhln@192.168.121.242
```

Observed server host fingerprints:

```text
ED25519 SHA256:DGBXYqcCWHaBPvnO+hzVHpOyTmfZMPiPotkmNOLTbJw
RSA     SHA256:kE3dMOZXsg5WQZauzj46InJzptanQxn4B+ofezLJSbs
```

Do not ask the user to paste their private SSH key into chat.

## GitHub State

Local Git push works in the current Windows workspace. Use normal `git add`, `git commit`, `git push` as needed. If sandbox blocks `.git/index.lock`, request elevated shell permission for the Git command.

## Current Docs

Important project docs:

```text
EXPERIMENT_PLAN.md
BASELINE_RUN_COMMANDS.md
AGENTS.md
EXPERIMENT_RESULTS.md
EXPERIMENT_TASKS.md
```

Keep `EXPERIMENT_TASKS.md` updated by checking off completed work. Keep `EXPERIMENT_RESULTS.md` aligned with the latest server `merge_results/index.json`.

## Style Notes

- The user prefers concrete commands and direct experimental runbooks.
- When giving server commands, prefer one-line commands if they need to copy/paste them.
- Explain experimental tradeoffs in Chinese and keep the structure clear.

## Long Experiment Automation Habit

For long-running lab-server experiments, make this the default workflow:

- Run long jobs inside `tmux` sessions on the lab server instead of blocking the local conversation.
- Use queue scripts when possible so GPU jobs are serialized or parallelized intentionally.
- Set a Codex heartbeat/automation to check progress periodically, usually every 30 minutes.
- On each check, inspect `tmux`, `merge.py` processes, GPU usage, logs, and newly written `merge_summary.json` files.
- If jobs finish, refresh result indexes under `merge_results/`, especially `merge_results/index.json` and `merge_results/README.md`.
- Pause and ask the user before destructive cleanup, large directory moves, launching longer FID/full-step experiments, starting new attack training, or occupying many GPUs beyond the agreed limit.

## File Organization Habit

When organizing experiment outputs, prefer moving the real result directories into the target organization folder. Do not copy result directories or replace them with symlinks unless the user explicitly asks for that. If existing scripts depend on the old paths, update the scripts/indexes after moving, or clearly note the path-compatibility risk before running more experiments.
