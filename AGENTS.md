# Agent Handoff Notes

This repository is being used for experiments on merge-time defenses for backdoored diffusion models. The user works in Chinese; answer in Chinese unless they ask otherwise.

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

The user has not yet run backdoor attacks. Start from attack/checkpoint preparation before merge baselines.

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

Before running merge defense, verify that each single backdoor model has meaningful backdoor behavior.

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

## Server Information From User

The user provided this lab server information:

```text
ssh zhln@192.168.121.242
project path: /home1/zhln/code
conda env: /home1/zhln/envs/baddiffusion
```

SSH network connectivity was reachable from the local environment, but login failed because the current Codex environment did not have a valid SSH credential:

```text
Permission denied (publickey,password)
```

Observed server host fingerprints:

```text
ED25519 SHA256:DGBXYqcCWHaBPvnO+hzVHpOyTmfZMPiPotkmNOLTbJw
RSA     SHA256:kE3dMOZXsg5WQZauzj46InJzptanQxn4B+ofezLJSbs
```

Recommended safe access approach:

1. Generate a temporary SSH key in this workspace.
2. Give the public key to the user.
3. User adds it to `/home1/zhln/.ssh/authorized_keys` on the server.
4. Remove that authorized key after the experiment session.

Do not ask the user to paste their private SSH key into chat.

## GitHub State

Local HTTPS `git push` previously failed on Windows with:

```text
SEC_E_NO_CREDENTIALS
```

The merge-method code was still pushed to GitHub through the GitHub app as commit:

```text
48e98cff298b6cb7e0e66129ba3f94e079c1f501
```

Local repo may have a different local commit with equivalent content and may appear ahead/diverged until synced.

## Current Docs Added Locally

These files were added locally after the merge-method commit:

```text
EXPERIMENT_PLAN.md
BASELINE_RUN_COMMANDS.md
AGENTS.md
```

If the user asks to publish docs to GitHub, use the GitHub app if local HTTPS credentials are still unavailable.

## Style Notes

- The user prefers concrete commands and direct experimental runbooks.
- When giving server commands, prefer one-line commands if they need to copy/paste them.
- Explain experimental tradeoffs in Chinese and keep the structure clear.
