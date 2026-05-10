# Experiment Results

This file summarizes the currently organized BadDiffusion merge-defense baseline results on the lab server. The source of truth is `/home1/zhln/code/BadDiffusion/merge_results/`.

## Result Folder

- `merge_results/`: real result directories plus generated `README.md` and `index.json`. Results are moved here directly, not copied and not symlinked.
- `merge_results/final_s1_*`: clean + backdoor final S1 baselines, `sample_n=1024`, `num_inference_steps=200`, `skip_fid=True`, `save_model=True`.
- `merge_results/save_s2_*`: S2 saved-model baselines, `sample_n=512`, `num_inference_steps=200`, `skip_fid=True`, `save_model=True`.
- `merge_results/final_s2_*_1024`: selected S2 rechecks, `sample_n=1024`, `num_inference_steps=200`, `skip_fid=True`, `save_model=True`.
- `merge_results/merge_medium_nodef_*`: no-defense single-backdoor reference runs.
- `merge_results/logs/`: baseline queue logs.
- `legacy_outputs/`: old pre-baseline outputs moved out of the project root on the server, including prior `result/`, `measure/`, `test/`, and `model_old/`.

## Indexed Results

| scenario | method | run | ASR | MSE | SSIM | save format |
|---|---|---|---:|---:|---:|---|
| nodef | no_defense | `merge_medium_nodef_box11_cat` | 0.613281 | 0.082787 | 0.703389 | N/A |
| nodef | no_defense | `merge_medium_nodef_box14_hat` | 0.898438 | 0.015880 | 0.873510 | N/A |
| s1_hat | anp | `final_s1_hat_anp` | 0.000000 | 0.238349 | 0.003236 | diffusers |
| s1_hat | clean_finetune | `final_s1_hat_clean_finetune` | 0.000000 | 0.240567 | 0.000384 | diffusers |
| s1_hat | diffusion_soup | `final_s1_hat_diffusion_soup` | 0.000000 | 0.240567 | 0.000384 | diffusers |
| s1_hat | dmm | `final_s1_hat_dmm` | 0.000000 | 0.240567 | 0.000384 | diffusers |
| s1_hat | maxfusion | `final_s1_hat_maxfusion` | 0.000000 | 0.240567 | 0.000384 | ensemble_metadata |
| s1_cat | anp | `final_s1_cat_anp` | 0.000000 | 0.359972 | 0.002178 | diffusers |
| s1_cat | clean_finetune | `final_s1_cat_clean_finetune` | 0.000000 | 0.361125 | 0.000201 | diffusers |
| s1_cat | diffusion_soup | `final_s1_cat_diffusion_soup` | 0.000000 | 0.361125 | 0.000201 | diffusers |
| s1_cat | dmm | `final_s1_cat_dmm` | 0.000000 | 0.361125 | 0.000201 | diffusers |
| s1_cat | maxfusion | `final_s1_cat_maxfusion` | 0.000000 | 0.361125 | 0.000201 | ensemble_metadata |
| s2_hat | anp | `save_s2_hat_anp` | 0.000000 | 0.239371 | 0.001347 | diffusers |
| s2_hat | clean_finetune | `save_s2_hat_clean_finetune` | 0.000000 | 0.210512 | 0.009150 | diffusers |
| s2_hat | diffusion_soup | `final_s2_hat_diffusion_soup_1024` | 0.000000 | 0.207276 | 0.011543 | diffusers |
| s2_hat | diffusion_soup | `save_s2_hat_diffusion_soup` | 0.000000 | 0.207602 | 0.010348 | diffusers |
| s2_hat | dmm | `final_s2_hat_dmm_1024` | 0.000000 | 0.208538 | 0.011025 | diffusers |
| s2_hat | dmm | `save_s2_hat_dmm` | 0.000000 | 0.208885 | 0.009838 | diffusers |
| s2_hat | maxfusion | `final_s2_hat_maxfusion_1024` | 0.891602 | 0.019639 | 0.828682 | ensemble_metadata |
| s2_hat | maxfusion | `save_s2_hat_maxfusion` | 0.902344 | 0.018561 | 0.835337 | ensemble_metadata |
| s2_cat | anp | `save_s2_cat_anp` | 0.000000 | 0.360446 | 0.001577 | diffusers |
| s2_cat | clean_finetune | `save_s2_cat_clean_finetune` | 0.000000 | 0.346936 | 0.006926 | diffusers |
| s2_cat | diffusion_soup | `save_s2_cat_diffusion_soup` | 0.000000 | 0.344767 | 0.007773 | diffusers |
| s2_cat | dmm | `save_s2_cat_dmm` | 0.000000 | 0.346476 | 0.007035 | diffusers |
| s2_cat | maxfusion | `final_s2_cat_maxfusion_1024` | 0.000000 | 0.247893 | 0.081227 | ensemble_metadata |
| s2_cat | maxfusion | `save_s2_cat_maxfusion` | 0.000000 | 0.249232 | 0.079891 | ensemble_metadata |

## Notes

- `maxfusion` saves `ensemble_metadata` because it is an inference-time ensemble wrapper rather than a single merged diffusers UNet.
- FID is not included in the older indexed rows above because those baseline runs used `--skip_fid`; the later `fid1000_*` runs below are the FID-enabled CIFAR10 results.
- The strongest remaining backdoor signal in the current indexed results is `final_s2_hat_maxfusion_1024` / `save_s2_hat_maxfusion`, where ASR remains high.

## FID-Enabled CIFAR10 Results

These runs use `sample_n=1024`, `num_inference_steps=1000`, and FID enabled.

| scenario | method | run | FID | ASR | MSE | SSIM |
|---|---|---|---:|---:|---:|---:|
| nodef_fid1000 | no_defense | `fid1000_nodef_box11_cat` | 51.685 | 0.925781 | 0.014789 | 0.932493 |
| nodef_fid1000 | no_defense | `fid1000_nodef_box14_hat` | 51.796 | 0.988281 | 0.001861 | 0.976367 |
| s1_cat | anp | `fid1000_s1_cat_anp` | 242.129 | 0.000000 | 0.360473 | 0.001751 |
| s1_cat | clean_finetune | `fid1000_s1_cat_clean_finetune` | 53.362 | 0.000000 | 0.361125 | 0.000201 |
| s1_cat | diffusion_soup | `fid1000_s1_cat_diffusion_soup` | 47.597 | 0.000000 | 0.361125 | 0.000201 |
| s1_cat | dmm | `fid1000_s1_cat_dmm` | 50.536 | 0.000000 | 0.361125 | 0.000201 |
| s1_cat | maxfusion | `fid1000_s1_cat_maxfusion` | 51.196 | 0.000000 | 0.361125 | 0.000201 |
| s1_hat | anp | `fid1000_s1_hat_anp` | 213.772 | 0.000000 | 0.238060 | 0.002644 |
| s1_hat | clean_finetune | `fid1000_s1_hat_clean_finetune` | 56.138 | 0.000000 | 0.240567 | 0.000384 |
| s1_hat | diffusion_soup | `fid1000_s1_hat_diffusion_soup` | 48.709 | 0.000000 | 0.240567 | 0.000384 |
| s1_hat | dmm | `fid1000_s1_hat_dmm` | 51.423 | 0.000000 | 0.240567 | 0.000384 |
| s1_hat | maxfusion | `fid1000_s1_hat_maxfusion` | 51.917 | 0.000000 | 0.240567 | 0.000384 |
| s2_cat | anp | `fid1000_s2_cat_anp` | 303.195 | 0.000000 | 0.360307 | 0.001580 |
| s2_cat | clean_finetune | `fid1000_s2_cat_clean_finetune` | 54.258 | 0.000000 | 0.344893 | 0.007517 |
| s2_cat | diffusion_soup | `fid1000_s2_cat_diffusion_soup` | 48.581 | 0.000000 | 0.340962 | 0.009173 |
| s2_cat | dmm | `fid1000_s2_cat_dmm` | 52.731 | 0.000000 | 0.342697 | 0.008441 |
| s2_cat | maxfusion | `fid1000_s2_cat_maxfusion` | 53.289 | 0.000000 | 0.164317 | 0.158479 |
| s2_hat | anp | `fid1000_s2_hat_anp` | 230.340 | 0.000000 | 0.239552 | 0.001559 |
| s2_hat | clean_finetune | `fid1000_s2_hat_clean_finetune` | 54.017 | 0.000000 | 0.207280 | 0.011719 |
| s2_hat | diffusion_soup | `fid1000_s2_hat_diffusion_soup` | 48.560 | 0.000000 | 0.204859 | 0.012783 |
| s2_hat | dmm | `fid1000_s2_hat_dmm` | 51.080 | 0.000000 | 0.206146 | 0.012305 |
| s2_hat | maxfusion | `fid1000_s2_hat_maxfusion` | 52.732 | 0.980469 | 0.004168 | 0.963320 |

## CelebA-HQ Results

These runs use `sample_n=256`, `num_inference_steps=1000`, FID enabled, and evaluate `GLASSES -> CAT`.

| scenario | method | run | FID | ASR | MSE | SSIM |
|---|---|---|---:|---:|---:|---:|
| celeba_hq_nodef | no_defense | `celeba_hq_nodef_glasses_cat` | 59.907 | 1.000000 | 0.000116 | 0.809629 |
| celeba_hq_s1 | anp | `celeba_hq_s1_glasses_cat_anp` | 360.430 | 0.000000 | 0.384921 | 0.000575 |
| celeba_hq_s1 | clean_finetune | `celeba_hq_s1_glasses_cat_clean_finetune` | 94.171 | 0.000000 | 0.384921 | 0.000575 |
| celeba_hq_s1 | diffusion_soup | `celeba_hq_s1_glasses_cat_diffusion_soup` | 100.852 | 0.000000 | 0.384921 | 0.000575 |
| celeba_hq_s1 | dmm | `celeba_hq_s1_glasses_cat_dmm` | 142.754 | 0.000000 | 0.384921 | 0.000575 |
| celeba_hq_s1 | maxfusion | `celeba_hq_s1_glasses_cat_maxfusion` | 60.092 | 0.000000 | 0.384921 | 0.000575 |

CelebA-HQ currently only has no-defense and S1 (`clean + backdoor`) results. S2 (`backdoor + backdoor`) has not been run for CelebA-HQ yet. BadMerging is treated as an adaptive attack method rather than a third scenario.

## BadMerging-Style Adaptive Diffusion Attack

The BadMerging-style diffusion adaptive attack trains an attacker checkpoint against a virtual `alpha=0.5` clean/backdoor merge. The current successful checkpoint is:

```text
/home1/zhln/code/BadDiffusion/merge_results/badmerge_cifar10_box14_hat_paired_strong2000/final
```

Fast evaluation results below use CIFAR10, `BOX_14 -> HAT`, `alpha=0.5`, `num_inference_steps=100`, and `skip_fid=True`.

| run | method | sample_n | ASR | MSE | SSIM |
|---|---|---:|---:|---:|---:|
| `badmerge_confirm_strong2000_n512` | diffusion_soup | 512 | 0.912109 | 0.016635 | 0.826002 |
| `badmerge_defense_strong2000_n256_diffusion_soup` | diffusion_soup | 256 | 0.890625 | 0.018481 | 0.816156 |
| `badmerge_defense_strong2000_n256_dmm` | dmm | 256 | 0.554688 | 0.075567 | 0.553345 |
| `badmerge_defense_strong2000_n256_maxfusion` | maxfusion | 256 | 0.000000 | 0.240567 | 0.000384 |
| `badmerge_defense_strong2000_n256_anp` | anp | 256 | 0.000000 | 0.239837 | 0.001121 |
| `badmerge_defense_strong2000_n256_clean_finetune` | clean_finetune | 256 | 0.000000 | 0.226789 | 0.006832 |

Observation: the adaptive attack transfers through uniform weight averaging, while MaxFusion, ANP, and Clean Fine-Tuning suppress ASR in the current quick setting. DMM partially reduces ASR. FID-enabled `sample_n=1024`, `num_inference_steps=1000` runs are in progress under `merge_results/badmerge_fid1000_*`.

## Cleanup And Organization Performed

Moved formal result directories from the project root into `merge_results/` and removed the old symlink-only layout. Removed the stale root-level `merge_result_index/`; the active index now lives in `merge_results/index.json` and `merge_results/README.md`.

Moved older non-current experiment artifacts out of the project root into `legacy_outputs/`:

- `result/`
- `measure/`
- `test/`
- `model_old/`

Earlier cleanup removed temporary or superseded server artifacts:
- `wandb/` run cache
- `fid_smoke_s2_hat_soup/`
- `debug_outputs/`
- `defended_models/`
- `__pycache__/`

Preserved original clean/backdoor checkpoints and all formal baseline/no-defense result directories under `merge_results/`.
