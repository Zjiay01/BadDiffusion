"""
ablation.py — Hyperparameter ablation experiments for NeurIPS submission.

Two modes
---------
1. DARE multi-seed (--mode dare_seed)
   Fix alpha sweep, run DARE with multiple seeds, report mean ± std per alpha.
   Demonstrates stability of DARE under random dropout.

2. Hyperparameter sweep (--mode hparam)
   Fix one or more alpha values, sweep ties_k or dare_p.
   Used to justify the default hyperparameter choice in the main paper.

Usage examples
--------------
# DARE stability across seeds (alpha sweep, 3 seeds)
python ablation.py \
  --mode dare_seed \
  --backdoor_ckpt ./res_backdoor --clean_ckpt ./res_clean \
  --dare_p 0.5 --dare_seeds 42,123,777 \
  --alphas 0.0,0.5,0.8,0.85,0.9,0.95,1.0 \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 1000 --skip_fid --output_dir ./ablation_results

# TIES k sweep (fix alpha=0.9, sweep k)
python ablation.py \
  --mode hparam \
  --method ties \
  --backdoor_ckpt ./res_backdoor --clean_ckpt ./res_clean \
  --hparam_name ties_k --hparam_values 0.05,0.1,0.2,0.3,0.5,1.0 \
  --fixed_alphas 0.9 \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 1000 --skip_fid --output_dir ./ablation_results

# DARE p sweep (fix alpha=0.9, sweep p)
python ablation.py \
  --mode hparam \
  --method dare \
  --backdoor_ckpt ./res_backdoor --clean_ckpt ./res_clean \
  --hparam_name dare_p --hparam_values 0.1,0.3,0.5,0.7,0.9 \
  --fixed_alphas 0.9 \
  --gpu 0 --fclip o --dataset CIFAR10 --trigger BOX_14 --target HAT \
  --sample_n 1000 --skip_fid --output_dir ./ablation_results
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from diffusers import DDPMPipeline

from dataset import Backdoor, DatasetLoader
from fid_score import fid

# Re-use all merge logic and helpers from merge.py
from merge import (
    make_merged_unet_state_dare,
    make_merged_unet_state_ties,
    make_merged_unet_state_wa,
    make_merged_unet_state_task_arithmetic,
    make_merged_unet_state_slerp,
    sample_to_dir,
    compute_backdoor_metrics,
    count_images,
    maybe_generate_real_images,
    save_grid_preview,
    resolve_device,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ablation experiments for merge.py methods.")

    p.add_argument("--mode", required=True, choices=["dare_seed", "hparam"],
                   help="dare_seed: multi-seed DARE stability | hparam: sweep one hyperparameter")

    # Model paths
    p.add_argument("--backdoor_ckpt", required=True)
    p.add_argument("--clean_ckpt", required=True)
    p.add_argument("--output_dir", default="./ablation_results")

    # Dataset / trigger
    p.add_argument("--dataset", default=DatasetLoader.CIFAR10,
                   choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10,
                            DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    p.add_argument("--dataset_path", default="datasets")
    p.add_argument("--dataset_load_mode", default=DatasetLoader.MODE_FIXED)
    p.add_argument("--clean_rate", type=float, default=1.0)
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--trigger", default=Backdoor.TRIGGER_BOX_14)
    p.add_argument("--target", default=Backdoor.TARGET_HAT)

    # Sampling
    p.add_argument("--sample_n", type=int, default=1000)
    p.add_argument("--eval_max_batch", type=int, default=256)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_num_workers", type=int, default=4)
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--asr_threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--fclip", default="o", choices=["w", "o", "n"])
    p.add_argument("--force_resample", action="store_true")

    # ---- dare_seed mode ----
    p.add_argument("--dare_p", type=float, default=0.5,
                   help="[dare_seed] DARE drop rate (fixed)")
    p.add_argument("--dare_seeds", type=str, default="42,123,777",
                   help="[dare_seed] Comma-separated seed list, e.g. 42,123,777")
    p.add_argument("--alphas", type=str, default="0.0,0.5,0.8,0.9,1.0",
                   help="[dare_seed] Alpha sweep values")

    # ---- hparam mode ----
    p.add_argument("--method", default="ties", choices=["ties", "dare", "wa", "task_arithmetic", "slerp"],
                   help="[hparam] Method to ablate")
    p.add_argument("--hparam_name", default=None,
                   choices=["ties_k", "dare_p"],
                   help="[hparam] Which hyperparameter to sweep")
    p.add_argument("--hparam_values", type=str, default=None,
                   help="[hparam] Comma-separated values to sweep, e.g. 0.05,0.1,0.2,0.5")
    p.add_argument("--fixed_alphas", type=str, default="0.9",
                   help="[hparam] Fixed alpha value(s) at which to evaluate each hparam setting")

    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Shared: build merged pipe and evaluate one (alpha, hparam_val) combination
# ---------------------------------------------------------------------------

def _build_merged_pipe(
    backdoor_state, clean_state, backdoor_ckpt,
    method, alpha,
    ties_k=0.2, dare_p=0.5, dare_seed=42,
    device="cuda",
):
    if method == "wa":
        state = make_merged_unet_state_wa(backdoor_state, clean_state, alpha)
    elif method == "task_arithmetic":
        state = make_merged_unet_state_task_arithmetic(backdoor_state, clean_state, alpha)
    elif method == "ties":
        state = make_merged_unet_state_ties(backdoor_state, clean_state, alpha, k=ties_k)
    elif method == "dare":
        state = make_merged_unet_state_dare(backdoor_state, clean_state, alpha, p=dare_p, seed=dare_seed)
    elif method == "slerp":
        state = make_merged_unet_state_slerp(backdoor_state, clean_state, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    pipe = DDPMPipeline.from_pretrained(backdoor_ckpt)
    pipe.unet.load_state_dict(state, strict=True)
    pipe = pipe.to(device)
    return pipe


def _evaluate_one(
    pipe, init_noise, backdoor_init_noise,
    out_dir, stage_name,
    sample_n, eval_max_batch, force_resample, clip_sample,
    real_dir, target, device, asr_threshold, fid_batch_size, fid_num_workers, skip_fid,
):
    """Sample images, compute FID + ASR metrics. Returns (fid_val, mse, ssim, asr)."""
    clean_dir = os.path.join(out_dir, "clean")
    bd_dir = os.path.join(out_dir, "backdoor")

    sample_to_dir(pipe, init_noise, clean_dir, sample_n, eval_max_batch,
                  force_resample, f"{stage_name} clean", clip_sample)
    sample_to_dir(pipe, backdoor_init_noise, bd_dir, sample_n, eval_max_batch,
                  force_resample, f"{stage_name} backdoor", clip_sample)

    fid_val = None
    if not skip_fid and real_dir is not None:
        fid_val = float(fid(path=[real_dir, clean_dir],
                            batch_size=fid_batch_size, device=device,
                            num_workers=fid_num_workers))

    mse, ssim, asr = compute_backdoor_metrics(
        backdoor_dir=bd_dir, target=target, device=device,
        asr_threshold=asr_threshold, max_samples=sample_n,
    )
    return fid_val, mse, ssim, asr


# ---------------------------------------------------------------------------
# Mode 1: DARE multi-seed stability
# ---------------------------------------------------------------------------

def run_dare_seed(args, device, clip_sample, backdoor_state, clean_state,
                  init_noise, backdoor_init_noise, trigger, target, dsl):
    seeds = parse_int_list(args.dare_seeds)
    alphas = parse_float_list(args.alphas)

    out_dir = os.path.join(args.output_dir, f"dare_seed_p{args.dare_p}")
    os.makedirs(out_dir, exist_ok=True)

    real_dir = os.path.join(out_dir, f"real_{args.dataset}")
    if not args.skip_fid:
        maybe_generate_real_images(dsl=dsl, out_dir=real_dir,
                                   sample_n=args.sample_n, seed=args.seed)

    print(f"\n[DARE-seed] seeds={seeds}  p={args.dare_p}  alphas={alphas}")

    # rows: one entry per (seed, alpha) — stored individually as they complete
    # per_alpha[alpha] accumulates references to those rows for aggregation
    all_rows: List[Dict] = []
    per_alpha: Dict[float, List[Dict]] = {a: [] for a in alphas}

    json_path = os.path.join(out_dir, "dare_seed_summary.json")
    txt_path  = os.path.join(out_dir, "dare_seed_summary.txt")

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[DARE-seed] seed={seed}")
        for alpha in alphas:
            print(f"  alpha={alpha:.4f} ...")
            pipe = _build_merged_pipe(
                backdoor_state, clean_state, args.backdoor_ckpt,
                method="dare", alpha=alpha,
                dare_p=args.dare_p, dare_seed=seed, device=device,
            )
            run_dir = os.path.join(out_dir, f"seed{seed}_alpha{alpha:.4f}")
            fid_val, mse, ssim, asr = _evaluate_one(
                pipe, init_noise, backdoor_init_noise,
                run_dir, f"dare s={seed} a={alpha:.4f}",
                args.sample_n, args.eval_max_batch, args.force_resample, clip_sample,
                real_dir, target, device, args.asr_threshold,
                args.fid_batch_size, args.fid_num_workers, args.skip_fid,
            )

            row = dict(
                row_type="seed",
                seed=seed,
                alpha=alpha,
                fid=fid_val,
                mse=mse,
                ssim=ssim,
                asr=asr,
            )
            all_rows.append(row)
            per_alpha[alpha].append(row)
            print(f"    FID={fid_val}  MSE={mse:.4f}  ASR={asr:.4f}")

            # Write incremental JSON after every single run so progress is not lost
            with open(json_path, "w") as f:
                json.dump({"rows": all_rows, "mean_rows": []}, f, indent=2)

            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Aggregate: append one mean±std row per alpha ----
    mean_rows: List[Dict] = []
    for alpha in alphas:
        runs = per_alpha[alpha]
        asrs = [r["asr"] for r in runs]
        mses = [r["mse"] for r in runs]
        ssims = [r["ssim"] for r in runs]
        fids  = [r["fid"] for r in runs if r["fid"] is not None]

        mean_row = dict(
            row_type="mean",
            seed="mean±std",
            alpha=alpha,
            asr=float(np.mean(asrs)),
            asr_std=float(np.std(asrs)),
            mse=float(np.mean(mses)),
            mse_std=float(np.std(mses)),
            ssim=float(np.mean(ssims)),
            ssim_std=float(np.std(ssims)),
            fid=float(np.mean(fids)) if fids else None,
            fid_std=float(np.std(fids)) if fids else None,
            n_seeds=len(runs),
        )
        mean_rows.append(mean_row)

    # Final JSON: individual seed rows + mean rows at the bottom
    with open(json_path, "w") as f:
        json.dump({"rows": all_rows, "mean_rows": mean_rows}, f, indent=2)

    _write_dare_seed_txt(all_rows, mean_rows, txt_path, seeds, args.dare_p)
    _plot_dare_seed(mean_rows, out_dir, args.dare_p, all_rows, seeds)
    return all_rows, mean_rows


def _write_dare_seed_txt(all_rows, mean_rows, txt_path, seeds, p):
    """
    Write a human-readable table with layout:

        DARE seed stability  (p=0.5, seeds=[42, 123, 777])

           alpha |  seed | FID          | MSE      | SSIM     | ASR
        ----------------------------------------------------------------
          0.0000 |    42 | N/A          | 0.2406   | 0.0123   | 0.0000
          0.0000 |   123 | N/A          | 0.2401   | 0.0121   | 0.0000
          0.0000 |   777 | N/A          | 0.2409   | 0.0125   | 0.0000
          0.0000 |  mean | N/A          | 0.2405±0.0003 | ...  | 0.0000±0.0000
        ----------------------------------------------------------------
          0.9000 |    42 | ...
          ...
    """
    col = f"{'alpha':>8} | {'seed':>8} | {'FID':>12} | {'MSE':>16} | {'SSIM':>16} | {'ASR':>16}"
    sep = "-" * len(col)

    lines = [
        f"DARE seed stability  (p={p}, seeds={seeds})",
        "",
        col,
        sep,
    ]

    # Group per alpha (preserve order)
    seen_alphas: List[float] = []
    alpha_to_rows: Dict[float, List[Dict]] = {}
    for r in all_rows:
        a = r["alpha"]
        if a not in alpha_to_rows:
            seen_alphas.append(a)
            alpha_to_rows[a] = []
        alpha_to_rows[a].append(r)

    mean_by_alpha = {mr["alpha"]: mr for mr in mean_rows}

    def fid_str(v):
        return f"{v:.3f}" if v is not None else "N/A"

    def pm(val, std):
        return f"{val:.4f}±{std:.4f}"

    for alpha in seen_alphas:
        for r in alpha_to_rows[alpha]:
            lines.append(
                f"{r['alpha']:>8.4f} | {r['seed']:>8} | {fid_str(r['fid']):>12} | "
                f"{r['mse']:>16.4f} | {r['ssim']:>16.4f} | {r['asr']:>16.4f}"
            )
        # mean ± std row
        mr = mean_by_alpha[alpha]
        fid_pm = pm(mr["fid"], mr["fid_std"]) if mr["fid"] is not None else "N/A"
        lines.append(
            f"{mr['alpha']:>8.4f} | {'mean':>8} | {fid_pm:>12} | "
            f"{pm(mr['mse'], mr['mse_std']):>16} | "
            f"{pm(mr['ssim'], mr['ssim_std']):>16} | "
            f"{pm(mr['asr'], mr['asr_std']):>16}"
        )
        lines.append(sep)

    txt = "\n".join(lines) + "\n"
    with open(txt_path, "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"[INFO] Saved txt to {txt_path}")


def _plot_dare_seed(mean_rows, out_dir, p, all_rows=None, seeds=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    alphas = [r["alpha"] for r in mean_rows]
    means  = [r["asr"]   for r in mean_rows]
    stds   = [r["asr_std"] for r in mean_rows]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Individual seed lines (thin, faded)
    if all_rows and seeds:
        seed_colors = ["#aec6cf", "#ffb347", "#b19cd9"]
        for si, seed in enumerate(seeds):
            seed_rows = [r for r in all_rows if r["seed"] == seed]
            seed_rows.sort(key=lambda r: r["alpha"])
            ax.plot(
                [r["alpha"] for r in seed_rows],
                [r["asr"]   for r in seed_rows],
                "o--", linewidth=0.8, markersize=3,
                color=seed_colors[si % len(seed_colors)],
                label=f"seed={seed}",
                alpha=0.7,
            )

    # Mean ± std
    ax.plot(alphas, means, "o-", color="tab:red", linewidth=2, label="Mean ASR")
    ax.fill_between(
        alphas,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2, color="tab:red", label="±1 std",
    )

    ax.set_xlabel("Alpha (backdoor weight)")
    ax.set_ylabel("ASR")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"DARE Seed Stability (p={p},  {len(seeds)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "dare_seed_stability.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved plot: {path}")


# ---------------------------------------------------------------------------
# Mode 2: Hyperparameter sweep
# ---------------------------------------------------------------------------

def run_hparam(args, device, clip_sample, backdoor_state, clean_state,
               init_noise, backdoor_init_noise, trigger, target, dsl):
    hparam_vals = parse_float_list(args.hparam_values)
    fixed_alphas = parse_float_list(args.fixed_alphas)

    tag = f"{args.method}_{args.hparam_name}_sweep"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    real_dir = os.path.join(out_dir, f"real_{args.dataset}")
    if not args.skip_fid:
        maybe_generate_real_images(dsl=dsl, out_dir=real_dir,
                                   sample_n=args.sample_n, seed=args.seed)

    print(f"\n[hparam] method={args.method}  sweep {args.hparam_name}={hparam_vals}"
          f"  fixed_alphas={fixed_alphas}")

    results = []  # list of {hparam_val, alpha, fid, mse, ssim, asr}

    for hval in hparam_vals:
        for alpha in fixed_alphas:
            print(f"\n  {args.hparam_name}={hval}  alpha={alpha:.4f}")

            # Build kwargs
            ties_k = hval if args.hparam_name == "ties_k" else 0.2
            dare_p = hval if args.hparam_name == "dare_p" else 0.5

            pipe = _build_merged_pipe(
                backdoor_state, clean_state, args.backdoor_ckpt,
                method=args.method, alpha=alpha,
                ties_k=ties_k, dare_p=dare_p, dare_seed=42,
                device=device,
            )
            run_dir = os.path.join(out_dir, f"hp{hval}_alpha{alpha:.4f}")
            fid_val, mse, ssim, asr = _evaluate_one(
                pipe, init_noise, backdoor_init_noise,
                run_dir, f"{args.hparam_name}={hval} a={alpha:.4f}",
                args.sample_n, args.eval_max_batch, args.force_resample, clip_sample,
                real_dir, target, device, args.asr_threshold,
                args.fid_batch_size, args.fid_num_workers, args.skip_fid,
            )
            results.append({
                "hparam_name": args.hparam_name,
                "hparam_val": hval,
                "alpha": alpha,
                "fid": fid_val,
                "mse": mse,
                "ssim": ssim,
                "asr": asr,
            })
            print(f"    FID={fid_val}  MSE={mse:.4f}  ASR={asr:.4f}")

            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_hparam_summary(results, out_dir, args.hparam_name, fixed_alphas)
    _plot_hparam(results, out_dir, args.hparam_name, args.method, fixed_alphas)
    return results


def _write_hparam_summary(results, out_dir, hparam_name, fixed_alphas):
    json_path = os.path.join(out_dir, "hparam_summary.json")
    txt_path = os.path.join(out_dir, "hparam_summary.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        f"Hyperparameter sweep: {hparam_name}  (fixed alphas={fixed_alphas})",
        "",
        f"{'hparam':>10} | {'alpha':>8} | {'ASR':>8} | {'FID':>10} | {'MSE':>10}",
        "-" * 56,
    ]
    for r in results:
        fid_str = f"{r['fid']:.3f}" if r["fid"] is not None else "N/A"
        lines.append(
            f"{r['hparam_val']:>10.4f} | {r['alpha']:>8.4f} | "
            f"{r['asr']:>8.4f} | {fid_str:>10} | {r['mse']:>10.4f}"
        )

    txt = "\n".join(lines) + "\n"
    with open(txt_path, "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"[INFO] Saved to {json_path}, {txt_path}")


def _plot_hparam(results, out_dir, hparam_name, method, fixed_alphas):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # One curve per fixed alpha
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["tab:red", "tab:orange", "tab:purple", "tab:green"]
    for i, alpha in enumerate(fixed_alphas):
        subset = [r for r in results if abs(r["alpha"] - alpha) < 1e-6]
        hvals = [r["hparam_val"] for r in subset]
        asrs = [r["asr"] for r in subset]
        fids = [r["fid"] for r in subset if r["fid"] is not None]
        fid_hvals = [r["hparam_val"] for r in subset if r["fid"] is not None]
        c = colors[i % len(colors)]
        axes[0].plot(hvals, asrs, "o-", color=c, label=f"alpha={alpha:.2f}")
        if fids:
            axes[1].plot(fid_hvals, fids, "s--", color=c, label=f"alpha={alpha:.2f}")

    axes[0].set_xlabel(hparam_name)
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"{method.upper()} — ASR vs {hparam_name}")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].set_xlabel(hparam_name)
    axes[1].set_ylabel("FID")
    axes[1].set_title(f"{method.upper()} — FID vs {hparam_name}")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"Hyperparameter Ablation: {hparam_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"hparam_{hparam_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved plot: {path}")

    # Heat-map if multiple alphas were swept
    if len(fixed_alphas) > 1:
        _plot_hparam_heatmap(results, out_dir, hparam_name, method)


def _plot_hparam_heatmap(results, out_dir, hparam_name, method):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    hvals = sorted(set(r["hparam_val"] for r in results))
    alphas = sorted(set(r["alpha"] for r in results))

    # Build ASR matrix: rows=alphas, cols=hparam_vals
    mat = np.full((len(alphas), len(hvals)), np.nan)
    for r in results:
        ri = alphas.index(r["alpha"])
        ci = hvals.index(r["hparam_val"])
        mat[ri, ci] = r["asr"]

    fig, ax = plt.subplots(figsize=(max(6, len(hvals) * 1.2), max(4, len(alphas) * 0.8)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="ASR")

    ax.set_xticks(range(len(hvals)))
    ax.set_xticklabels([f"{v:.3g}" for v in hvals])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel(hparam_name)
    ax.set_ylabel("Alpha")
    ax.set_title(f"{method.upper()} — ASR Heatmap ({hparam_name} × Alpha)")

    for ri in range(len(alphas)):
        for ci in range(len(hvals)):
            v = mat[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.2 < v < 0.8 else "white")

    plt.tight_layout()
    path = os.path.join(out_dir, f"hparam_{hparam_name}_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved heatmap: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = resolve_device(args.gpu)
    clip_sample = {"w": True, "o": False, "n": None}[args.fclip]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ablation_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    dsl = (
        DatasetLoader(root=args.dataset_path, name=args.dataset, batch_size=args.eval_max_batch)
        .set_poison(
            trigger_type=args.trigger, target_type=args.target,
            clean_rate=args.clean_rate, poison_rate=args.poison_rate,
        )
        .prepare_dataset(mode=args.dataset_load_mode)
    )
    trigger = dsl.trigger
    target = dsl.target

    # Load checkpoints once
    print("[INFO] Loading checkpoints...")
    backdoor_pipe = DDPMPipeline.from_pretrained(args.backdoor_ckpt)
    clean_pipe = DDPMPipeline.from_pretrained(args.clean_ckpt)
    backdoor_state = backdoor_pipe.unet.state_dict()
    clean_state = clean_pipe.unet.state_dict()

    # Shared noise
    alphas_for_noise = (
        parse_float_list(args.alphas) if args.mode == "dare_seed"
        else parse_float_list(args.fixed_alphas)
    )
    init_noise = torch.randn(
        (args.sample_n,
         backdoor_pipe.unet.config.in_channels,
         backdoor_pipe.unet.config.sample_size,
         backdoor_pipe.unet.config.sample_size),
        generator=torch.manual_seed(args.seed),
    )
    backdoor_init_noise = init_noise + trigger.unsqueeze(0)

    del backdoor_pipe, clean_pipe

    # Dispatch
    if args.mode == "dare_seed":
        run_dare_seed(
            args, device, clip_sample,
            backdoor_state, clean_state,
            init_noise, backdoor_init_noise,
            trigger, target, dsl,
        )
    elif args.mode == "hparam":
        if args.hparam_name is None or args.hparam_values is None:
            raise ValueError("--mode hparam requires --hparam_name and --hparam_values")
        run_hparam(
            args, device, clip_sample,
            backdoor_state, clean_state,
            init_noise, backdoor_init_noise,
            trigger, target, dsl,
        )

    print("\n[INFO] Ablation complete.")


if __name__ == "__main__":
    main()
