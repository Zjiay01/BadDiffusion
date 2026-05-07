# Experiment Results

This file summarizes the currently organized BadDiffusion merge-defense baseline results on the lab server. The source of truth is `/home1/zhln/code/BadDiffusion/merge_results/` and `/home1/zhln/code/BadDiffusion/merge_result_index/`.

## Result Folders

- `merge_results/`: compact entry point with `README.md`, `index.json`, and symlinks to result directories.
- `merge_result_index/`: generated result index files.
- `final_s1_*`: clean + backdoor final S1 baselines, `sample_n=1024`, `num_inference_steps=200`, `skip_fid=True`, `save_model=True`.
- `save_s2_*`: S2 saved-model baselines, `sample_n=512`, `num_inference_steps=200`, `skip_fid=True`, `save_model=True`.
- `final_s2_*_1024`: selected S2 rechecks, `sample_n=1024`, `num_inference_steps=200`, `skip_fid=True`, `save_model=True`.
- `merge_medium_nodef_*`: no-defense single-backdoor reference runs.

## Indexed Results

| scenario | method | run | ASR | MSE | SSIM | save format |
|---|---|---|---:|---:|---:|---|
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
- FID is not included in these indexed rows because these baseline runs used `--skip_fid`; FID should be launched separately as a longer experiment.
- The strongest remaining backdoor signal in the current indexed results is `final_s2_hat_maxfusion_1024` / `save_s2_hat_maxfusion`, where ASR remains high.

## Cleanup Performed

Removed temporary or superseded server artifacts:

- `wandb/` (run cache, about 1.9GB)
- `fid_smoke_s2_hat_soup/` (small FID smoke test)
- `debug_outputs/` (old debug outputs)
- `defended_models/` (older medium copied outputs superseded by indexed final/save runs)
- `__pycache__/`
- closed the completed `final_s1_queue` tmux session

Preserved formal result folders, no-defense reference runs, result indexes, logs, and original clean/backdoor checkpoints.
