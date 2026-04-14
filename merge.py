import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image

from diffusers import DDPMPipeline

from dataset import Backdoor, DatasetLoader, ImagePathDataset
from fid_score import fid


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_alphas(alpha_text: str) -> List[float]:
    vals = [x.strip() for x in alpha_text.split(",") if x.strip() != ""]
    if not vals:
        raise ValueError("--alphas is empty")
    alphas = [float(x) for x in vals]
    for a in alphas:
        if a < 0.0 or a > 1.0:
            raise ValueError(f"alpha should be in [0, 1], got: {a}")
    return alphas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two trained diffusion models and evaluate backdoor robustness."
    )
    parser.add_argument("--backdoor_ckpt", type=str, required=True)
    parser.add_argument("--clean_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--method",
        type=str,
        default="wa",
        choices=["wa", "task_arithmetic", "ties", "dare", "slerp", "mwm"],
        help=(
            "Merge method:\n"
            "  wa              – weighted average (original behaviour)\n"
            "  task_arithmetic – negate the backdoor task-vector\n"
            "  ties            – TIES-Merging (trim + elect-sign + merge)\n"
            "  dare            – DARE (random drop + rescale) then weighted avg\n"
            "  slerp           – spherical linear interpolation\n"
            "\n"
            "alpha semantics (backdoor weight, 1 = full backdoor, 0 = full clean):\n"
            "  wa              – W = (1-a)*W_clean + a*W_backdoor\n"
            "  task_arithmetic – W = W_backdoor - a*τ  where τ=W_backdoor-W_clean\n"
            "  ties/dare       – scale applied to merged task-vector\n"
            "  slerp           – interpolation factor (0=clean, 1=backdoor)\n"
        ),
    )

    # TIES-specific
    parser.add_argument(
        "--ties_k",
        type=float,
        default=0.2,
        help="TIES: fraction of parameters to KEEP (top-k by |task-vector|). Default 0.2 = keep top 20%%.",
    )

    # DARE-specific
    parser.add_argument(
        "--dare_p",
        type=float,
        default=0.5,
        help="DARE: drop rate for delta parameters (0=keep all, 0.9=drop 90%%). Default 0.5.",
    )
    parser.add_argument(
        "--dare_seeds",
        type=str,
        default="42",
        help=(
            "DARE: comma-separated seed list for the dropout mask. "
            "Single seed (e.g. 42) -> one merge. "
            "Multiple seeds (e.g. 42,123,777) -> run once per seed and average the merged weights."
        ),
    )

    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.2,0.5,0.8,0.9,1.0",
        help="Comma-separated alpha list (0=clean, 1=backdoor).",
    )
    parser.add_argument("--dataset", type=str, default=DatasetLoader.CIFAR10,
                        choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10,
                                 DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument("--dataset_path", type=str, default="datasets")
    parser.add_argument("--dataset_load_mode", type=str, default=DatasetLoader.MODE_FIXED,
                        choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])

    parser.add_argument("--clean_rate", type=float, default=1.0)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--trigger", type=str, default=Backdoor.TRIGGER_BOX_14)
    parser.add_argument("--target", type=str, default=Backdoor.TARGET_HAT)

    parser.add_argument("--sample_n", type=int, default=2048)
    parser.add_argument("--eval_max_batch", type=int, default=256)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--fid_num_workers", type=int, default=4)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--asr_threshold", type=float, default=0.05)
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="Number of denoising steps during sampling. Default 1000.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--fclip", type=str, default="o", choices=["w", "o", "n"])
    parser.add_argument("--force_resample", action="store_true")

    parser.add_argument(
        "--mwm_T",
        type=float,
        default=1.0,
        help="MWM: temperature parameter controlling suppression strength. "
            "Larger T -> closer to WA. Default 1.0.",
    )
    
    return parser.parse_args()


def collect_input_args(argv: List[str], parsed_args: argparse.Namespace) -> Dict:
    provided = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        val = None
        if "=" in key:
            key, val = key.split("=", 1)
        else:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is None or nxt.startswith("--"):
                val = True
            else:
                val = nxt
                i += 1
        key = key.replace("-", "_")
        if hasattr(parsed_args, key):
            provided[key] = getattr(parsed_args, key)
        else:
            provided[key] = val
        i += 1
    return provided


def get_merge_dir_name(backdoor_ckpt: str, poison_rate: float, method: str) -> str:
    model_name = os.path.basename(os.path.normpath(backdoor_ckpt)).replace(" ", "_")
    return f"merge_{method}_{model_name}_{poison_rate}"


def resolve_output_dir(output_dir_arg: str, merge_dir_name: str) -> str:
    if output_dir_arg is None:
        return merge_dir_name
    normalized = os.path.basename(os.path.normpath(output_dir_arg))
    if normalized.startswith("merge_"):
        return output_dir_arg
    return os.path.join(output_dir_arg, merge_dir_name)


def save_run_args_config(output_dir: str, input_args: Dict, config: Dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(input_args, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def resolve_device(gpu_arg: Optional[str]) -> str:
    if gpu_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg
    if torch.cuda.is_available():
        return "cuda"
    print("[WARN] CUDA unavailable, falling back to cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# Merge methods
# ---------------------------------------------------------------------------

def _float_keys(state: Dict[str, torch.Tensor]):
    """Return keys whose values are floating-point tensors."""
    return [k for k, v in state.items() if torch.is_floating_point(v)]


def make_merged_unet_state_wa(
    backdoor_state: Dict[str, torch.Tensor],
    clean_state: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Weighted Average: W = (1-alpha)*W_clean + alpha*W_backdoor"""
    merged = {}
    for k in backdoor_state:
        bd, cl = backdoor_state[k], clean_state[k]
        if not torch.is_floating_point(bd):
            merged[k] = bd
            continue
        merged[k] = (cl.float() * (1.0 - alpha) + bd.float() * alpha).to(bd.dtype)
    return merged


def make_merged_unet_state_task_arithmetic(
    backdoor_state: Dict[str, torch.Tensor],
    clean_state: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Task Arithmetic – backdoor negation.

    Here backdoor_ckpt is treated as the "pretrained" model and
    clean_ckpt as the "finetuned" model, so the task vector is:
        τ = W_clean - W_backdoor   (direction that removes the backdoor)

    Merged model:
        W = W_backdoor + (1-alpha) * τ
          = W_backdoor + (1-alpha) * (W_clean - W_backdoor)
          = alpha*W_backdoor + (1-alpha)*W_clean

    alpha=1 → pure backdoor; alpha=0 → pure clean (backdoor fully negated).

    This is mathematically equivalent to WA with roles swapped, but the
    framing is different: we are *adding* the clean task-vector to the
    backdoor model to suppress the backdoor behaviour.
    """
    merged = {}
    for k in backdoor_state:
        bd, cl = backdoor_state[k], clean_state[k]
        if not torch.is_floating_point(bd):
            merged[k] = bd
            continue
        tau = cl.float() - bd.float()          # task vector (toward clean)
        merged[k] = (bd.float() + (1 - alpha) * tau).to(bd.dtype)
    return merged


def make_merged_unet_state_ties(
    backdoor_state: Dict[str, torch.Tensor],
    clean_state: Dict[str, torch.Tensor],
    alpha: float,
    k: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    TIES-Merging (single task-vector edition).

    We want to "remove" the backdoor, so:
        pretrained  = backdoor model
        task vector = τ = W_clean - W_backdoor  (toward the clean direction)

    Three steps
    -----------
    1. Trim  – zero out parameters in τ that are NOT in the top-k% by |τ|
    2. Elect – no sign conflict with a single vector, so sign = sign(τ_trimmed)
    3. Merge – W_merged = W_backdoor + (1-alpha) * τ_trimmed

    alpha=1 → pure backdoor, alpha=0 → τ fully applied.
    """
    float_keys = _float_keys(backdoor_state)
    # Compute task vector
    tau_flat_list, shapes, keys = [], [], []
    for kk in float_keys:
        t = (clean_state[kk].float() - backdoor_state[kk].float()).flatten()
        tau_flat_list.append(t)
        shapes.append(backdoor_state[kk].shape)
        keys.append(kk)

    tau_flat = torch.cat(tau_flat_list)          # all params in one vector

    # Step 1 – Trim: keep top-k% by absolute value
    n_keep = max(1, int(k * tau_flat.numel()))
    threshold = torch.topk(tau_flat.abs(), n_keep, largest=True).values[-1]
    mask = tau_flat.abs() >= threshold
    tau_trimmed = tau_flat * mask.float()

    # Step 2 – Elect sign (trivial for single vector: sign already determined)

    # Step 3 – Reconstruct and merge
    merged = {}
    offset = 0
    for kk, shape in zip(keys, shapes):
        numel = 1
        for s in shape:
            numel *= s
        tau_k = tau_trimmed[offset: offset + numel].reshape(shape)
        offset += numel
        merged[kk] = (backdoor_state[kk].float() + (1 - alpha) * tau_k).to(backdoor_state[kk].dtype)

    # Copy non-float keys unchanged
    for kk in backdoor_state:
        if kk not in merged:
            merged[kk] = backdoor_state[kk]

    return merged


def make_merged_unet_state_dare(
    backdoor_state: Dict[str, torch.Tensor],
    clean_state: Dict[str, torch.Tensor],
    alpha: float,
    p: float = 0.5,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    DARE – Drop And REscale.

    Task vector: τ = W_clean - W_backdoor
    1. Randomly drop p fraction of τ parameters (set to zero)
    2. Rescale surviving parameters by 1/(1-p) to preserve expected magnitude
    3. Apply: W_merged = W_backdoor + (1-alpha) * τ_dare

    alpha=1 → pure backdoor, alpha=0 → full dare-sparsified clean direction.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    merged = {}
    for kk in backdoor_state:
        bd, cl = backdoor_state[kk], clean_state[kk]
        if not torch.is_floating_point(bd):
            merged[kk] = bd
            continue
        tau = cl.float() - bd.float()
        # Bernoulli mask: 1 = keep, 0 = drop
        keep_mask = torch.bernoulli(
            torch.full(tau.shape, 1.0 - p), generator=rng
        ).to(tau.device)
        tau_dare = tau * keep_mask / (1.0 - p)      # rescale
        merged[kk] = (bd.float() + (1 - alpha) * tau_dare).to(bd.dtype)

    return merged


def make_merged_unet_state_slerp(
    backdoor_state: Dict[str, torch.Tensor],
    clean_state: Dict[str, torch.Tensor],
    alpha: float,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    SLERP – Spherical Linear Interpolation.

    Interpolates each weight tensor along the great-circle arc on the
    unit hypersphere.  alpha=0 → clean, alpha=1 → backdoor.

    For near-parallel or near-antipodal vectors, falls back to LERP to
    avoid numerical instability.
    """
    merged = {}
    for kk in backdoor_state:
        bd, cl = backdoor_state[kk], clean_state[kk]
        if not torch.is_floating_point(bd):
            merged[kk] = bd
            continue

        v0 = cl.float().flatten()   # alpha=0 endpoint
        v1 = bd.float().flatten()   # alpha=1 endpoint

        n0 = v0.norm()
        n1 = v1.norm()

        if n0 < eps or n1 < eps:
            # One vector is near-zero – fall back to linear interp
            result = (1.0 - alpha) * v0 + alpha * v1
        else:
            u0 = v0 / n0
            u1 = v1 / n1
            dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
            theta = torch.acos(dot)

            if theta.abs() < eps:
                # Nearly parallel – linear interp on unit sphere ≈ LERP
                result = (1.0 - alpha) * v0 + alpha * v1
            else:
                sin_theta = torch.sin(theta)
                w0 = torch.sin((1.0 - alpha) * theta) / sin_theta
                w1 = torch.sin(alpha * theta) / sin_theta
                # Interpolate direction, then interpolate norm
                interp_norm = (1.0 - alpha) * n0 + alpha * n1
                result = (w0 * u0 + w1 * u1) * interp_norm

        merged[kk] = result.reshape(bd.shape).to(bd.dtype)

    return merged

def make_merged_unet_state_mwm(
    backdoor_state: Dict[str, torch.Tensor],
    clean_state: Dict[str, torch.Tensor],
    alpha: float,
    T: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Magnitude-Weighted Merging (MWM).

    Computes a per-parameter weight w_i based on task vector magnitude:
        w_i = min(2 * (1 - sigmoid((|tau_i| - mu) / (T * sigma))), 1)

    High-magnitude parameters (likely backdoor-critical) receive smaller
    weights; low-magnitude parameters are left unchanged (w_i = 1).
    As T -> inf, w_i -> 1 for all i, recovering standard WA.

    Merged model:
        theta_m = theta_c + alpha * w * tau
    """
    float_keys = _float_keys(backdoor_state)

    # ── 1. 拼接所有浮点参数，计算全局 mu 和 sigma ──────────────────────────
    tau_flat_list = []
    for k in float_keys:
        tau_flat_list.append(
            (backdoor_state[k].float() - clean_state[k].float()).abs().flatten()
        )
    tau_abs_flat = torch.cat(tau_flat_list)
    mu    = tau_abs_flat.mean()
    sigma = tau_abs_flat.std().clamp(min=1e-8)

    # ── 2. 逐参数计算权重并合并 ────────────────────────────────────────────
    merged = {}
    for k in backdoor_state:
        bd, cl = backdoor_state[k], clean_state[k]
        if not torch.is_floating_point(bd):
            merged[k] = bd
            continue
        tau_k   = bd.float() - cl.float()
        tau_abs = tau_k.abs()
        w = 2.0 * (1.0 - torch.sigmoid((tau_abs - mu) / (T * sigma)))
        w = w.clamp(max=1.0)                       # clip to [0, 1]
        merged[k] = (cl.float() + alpha * w * tau_k).to(bd.dtype)

    return merged

# Registry so main() can call the right function
MERGE_METHODS = {
    "wa": make_merged_unet_state_wa,
    "task_arithmetic": make_merged_unet_state_task_arithmetic,
    "ties": make_merged_unet_state_ties,
    "dare": make_merged_unet_state_dare,
    "slerp": make_merged_unet_state_slerp,
    "mwm": make_merged_unet_state_mwm,
}


def dispatch_merge(method: str, backdoor_state, clean_state, alpha, args) -> Dict:
    if method == "wa":
        return make_merged_unet_state_wa(backdoor_state, clean_state, alpha)
    elif method == "task_arithmetic":
        return make_merged_unet_state_task_arithmetic(backdoor_state, clean_state, alpha)
    elif method == "ties":
        return make_merged_unet_state_ties(backdoor_state, clean_state, alpha, k=args.ties_k)
    elif method == "dare":
        seeds = [int(s.strip()) for s in args.dare_seeds.split(",") if s.strip()]
        if len(seeds) == 1:
            return make_merged_unet_state_dare(backdoor_state, clean_state, alpha, p=args.dare_p, seed=seeds[0])
        # Multi-seed: average the merged state dicts across seeds
        float_keys = [k for k, v in backdoor_state.items() if torch.is_floating_point(v)]
        accum = {k: torch.zeros_like(backdoor_state[k].float()) for k in float_keys}
        for seed in seeds:
            s = make_merged_unet_state_dare(backdoor_state, clean_state, alpha, p=args.dare_p, seed=seed)
            for k in float_keys:
                accum[k] += s[k].float()
        merged = {}
        for k in backdoor_state:
            if k in float_keys:
                merged[k] = (accum[k] / len(seeds)).to(backdoor_state[k].dtype)
            else:
                merged[k] = backdoor_state[k]
        return merged
    elif method == "slerp":
        return make_merged_unet_state_slerp(backdoor_state, clean_state, alpha)
    elif method == "mwm":
        return make_merged_unet_state_mwm(backdoor_state, clean_state, alpha, T=args.mwm_T)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def count_images(path: str) -> int:
    exts = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
    return sum(1 for n in os.listdir(path) if n.rsplit(".", 1)[-1].lower() in exts) if os.path.isdir(path) else 0


def clear_images(path: str):
    if not os.path.isdir(path):
        return
    exts = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
    for name in os.listdir(path):
        if name.rsplit(".", 1)[-1].lower() in exts:
            os.remove(os.path.join(path, name))


def make_grid_img(images: List[Image.Image], rows: int = 8, cols: int = 8) -> Image.Image:
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(images[: rows * cols]):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_grid_preview(sample_dir: str, out_file: str, rows: int = 8, cols: int = 8):
    if not os.path.isdir(sample_dir):
        return
    names = sorted(
        [x for x in os.listdir(sample_dir) if x.lower().endswith(".png")],
        key=lambda x: int(x.rsplit(".", 1)[0]) if x.rsplit(".", 1)[0].isdigit() else x,
    )
    if not names:
        return
    imgs = []
    for name in names[: rows * cols]:
        with Image.open(os.path.join(sample_dir, name)) as img:
            imgs.append(img.convert("RGB"))
    make_grid_img(imgs, rows=rows, cols=cols).save(out_file)


def maybe_generate_real_images(dsl: DatasetLoader, out_dir: str, sample_n: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    if count_images(out_dir) >= sample_n:
        return
    print(f"[INFO] Generating {sample_n} real images for FID reference at: {out_dir}")
    ds = dsl.get_dataset().shuffle(seed=seed)
    imgs = ds[:sample_n][DatasetLoader.IMAGE]
    for i, img in enumerate(tqdm(imgs, desc="Save real images")):
        dsl.show_sample(img=img, is_show=False, file_name=os.path.join(out_dir, f"{i}.png"))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_to_dir(pipe, init_noise, out_dir, sample_n, eval_max_batch,
                  force_resample, stage_name, clip_sample, num_inference_steps):
    os.makedirs(out_dir, exist_ok=True)
    if force_resample:
        clear_images(out_dir)

    existing_n = count_images(out_dir)
    if existing_n >= sample_n:
        print(f"[INFO] {stage_name}: found {existing_n}/{sample_n}, skip sampling")
        return

    device = next(pipe.unet.parameters()).device
    if hasattr(pipe.scheduler, "config") and clip_sample is not None:
        pipe.scheduler.config.clip_sample = clip_sample
    pipe.scheduler.set_timesteps(num_inference_steps)

    start = existing_n
    remaining = sample_n - existing_n
    batch_total = (remaining + eval_max_batch - 1) // eval_max_batch
    batch_idx = 0
    pbar = tqdm(total=sample_n, initial=existing_n, desc=f"{stage_name} sampling", unit="img")

    while remaining > 0:
        batch_idx += 1
        batch_sz = min(eval_max_batch, remaining)
        end = start + batch_sz
        batch_init = init_noise[start:end].to(device)
        sample = batch_init.clone()

        step_bar = tqdm(
            total=len(pipe.scheduler.timesteps),
            desc=f"{stage_name} denoise [{batch_idx}/{batch_total}]",
            unit="step",
            leave=False,
        )
        for t in pipe.scheduler.timesteps:
            with torch.no_grad():
                model_output = pipe.unet(sample, t).sample
            sample = pipe.scheduler.step(model_output, t, sample).prev_sample
            step_bar.update(1)
        step_bar.close()

        images = (sample / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        for i, arr in enumerate(images):
            Image.fromarray((arr * 255).round().astype("uint8")).save(
                os.path.join(out_dir, f"{start + i}.png")
            )

        start = end
        remaining = sample_n - start
        pbar.update(batch_sz)

    pbar.close()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_backdoor_metrics(backdoor_dir: str, target: torch.Tensor,
                             device: str, asr_threshold: float,
                             max_samples: int = None):
    dev = torch.device(device)
    dataset = ImagePathDataset(path=backdoor_dir)
    n = min(max_samples, len(dataset)) if max_samples is not None else len(dataset)
    gen = dataset[:n].to(dev)

    reps = [len(gen)] + [1] * len(target.shape)
    tgt = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(dev)

    mse = float(nn.MSELoss(reduction="mean")(gen, tgt))
    ssim = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)(gen, tgt))
    per_img_mse = torch.mean((gen - tgt) ** 2, dim=(1, 2, 3))
    asr = float((per_img_mse < asr_threshold).float().mean().item())
    return mse, ssim, asr


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "merge_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    header = f"{'alpha':>8} | {'fid':>12} | {'mse':>12} | {'ssim':>12} | {'asr':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        fid_str = f"{r['fid']:.6f}" if r["fid"] is not None else "N/A"
        lines.append(
            f"{r['alpha']:>8.4f} | {fid_str:>12} | {r['mse']:>12.6f} | {r['ssim']:>12.6f} | {r['asr']:>12.6f}"
        )
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(output_dir, "merge_summary.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)


def plot_tradeoff(results: List[Dict], output_dir: str, method: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip plot")
        return

    alphas = [r["alpha"] for r in results]
    asrs = [r["asr"] for r in results]
    fid_pairs = [(r["alpha"], r["fid"]) for r in results if r.get("fid") is not None]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Alpha (backdoor weight)")
    ax1.set_ylabel("ASR", color="tab:red")
    ax1.plot(alphas, asrs, "o-", color="tab:red", label="ASR")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if fid_pairs:
        ax2 = ax1.twinx()
        ax2.set_ylabel("FID", color="tab:blue")
        ax2.plot([p[0] for p in fid_pairs], [p[1] for p in fid_pairs],
                 "s--", color="tab:blue", label="FID")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    else:
        ax1.legend(lines1, labels1)

    plt.title(f"Backdoor Survivability under Model Merging [{method.upper()}]")
    plt.tight_layout()
    path = os.path.join(output_dir, "asr_fid_tradeoff.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved tradeoff plot to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_args = collect_input_args(sys.argv[1:], args)
    alphas = parse_alphas(args.alphas)
    device = resolve_device(args.gpu)

    clip_sample = {"w": True, "o": False, "n": None}[args.fclip]

    merge_dir_name = get_merge_dir_name(
        backdoor_ckpt=args.backdoor_ckpt,
        poison_rate=args.poison_rate,
        method=args.method,
    )
    output_dir = resolve_output_dir(args.output_dir, merge_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    full_config = dict(vars(args))
    full_config.update(merge_dir_name=merge_dir_name, resolved_output_dir=output_dir, resolved_device=device)
    save_run_args_config(output_dir=output_dir, input_args=input_args, config=full_config)

    print(f"[INFO] Method:       {args.method}")
    print(f"[INFO] Device:       {device}")
    print(f"[INFO] clip_sample:  {clip_sample}")
    print(f"[INFO] Alphas:       {alphas}")
    print(f"[INFO] Output dir:   {output_dir}")
    if args.method == "ties":
        print(f"[INFO] TIES k (keep fraction): {args.ties_k}")
    if args.method == "dare":
        seeds = [s.strip() for s in args.dare_seeds.split(",") if s.strip()]
        seed_str = args.dare_seeds if len(seeds) > 1 else seeds[0]
        avg_note = f"  (averaging {len(seeds)} seeds)" if len(seeds) > 1 else ""
        print(f"[INFO] DARE p (drop rate):     {args.dare_p}  seeds={seed_str}{avg_note}")

    dsl = (
        DatasetLoader(root=args.dataset_path, name=args.dataset, batch_size=args.eval_max_batch)
        .set_poison(
            trigger_type=args.trigger,
            target_type=args.target,
            clean_rate=args.clean_rate,
            poison_rate=args.poison_rate,
        )
        .prepare_dataset(mode=args.dataset_load_mode)
    )
    trigger = dsl.trigger
    target = dsl.target

    real_dir = os.path.join(output_dir, f"real_{args.dataset}")
    if not args.skip_fid:
        maybe_generate_real_images(dsl=dsl, out_dir=real_dir, sample_n=args.sample_n, seed=args.seed)

    print("[INFO] Loading checkpoints...")
    backdoor_pipe = DDPMPipeline.from_pretrained(args.backdoor_ckpt)
    clean_pipe = DDPMPipeline.from_pretrained(args.clean_ckpt)
    backdoor_state = backdoor_pipe.unet.state_dict()
    clean_state = clean_pipe.unet.state_dict()

    init_noise = torch.randn(
        (args.sample_n,
         backdoor_pipe.unet.config.in_channels,
         backdoor_pipe.unet.config.sample_size,
         backdoor_pipe.unet.config.sample_size),
        generator=torch.manual_seed(args.seed),
    )
    backdoor_init_noise = init_noise + trigger.unsqueeze(0)

    results = []
    for alpha in alphas:
        tag = f"alpha{alpha:.4f}"
        alpha_dir = os.path.join(output_dir, tag)
        clean_dir = os.path.join(alpha_dir, "clean")
        backdoor_dir = os.path.join(alpha_dir, "backdoor")
        os.makedirs(alpha_dir, exist_ok=True)

        print(f"\n[INFO] Merging  method={args.method}  alpha={alpha:.4f}")
        merged_state = dispatch_merge(
            method=args.method,
            backdoor_state=backdoor_state,
            clean_state=clean_state,
            alpha=alpha,
            args=args,
        )

        merged_pipe = DDPMPipeline.from_pretrained(args.backdoor_ckpt)
        merged_pipe.unet.load_state_dict(merged_state, strict=True)
        merged_pipe = merged_pipe.to(device)

        sample_to_dir(merged_pipe, init_noise, clean_dir, args.sample_n,
                      args.eval_max_batch, args.force_resample,
                      f"{args.method} a={alpha:.4f} clean", clip_sample, args.num_inference_steps)
        save_grid_preview(clean_dir, os.path.join(alpha_dir, "clean_grid.png"))

        sample_to_dir(merged_pipe, backdoor_init_noise, backdoor_dir, args.sample_n,
                      args.eval_max_batch, args.force_resample,
                      f"{args.method} a={alpha:.4f} backdoor", clip_sample, args.num_inference_steps)
        save_grid_preview(backdoor_dir, os.path.join(alpha_dir, "backdoor_grid.png"))

        fid_val = None
        if not args.skip_fid:
            print("FID start")
            fid_val = float(fid(path=[real_dir, clean_dir],
                                batch_size=args.fid_batch_size,
                                device=device,
                                num_workers=args.fid_num_workers))
            print("FID done")

        print("ASR metrics start")
        mse_val, ssim_val, asr_val = compute_backdoor_metrics(
            backdoor_dir=backdoor_dir,
            target=target,
            device=device,
            asr_threshold=args.asr_threshold,
            max_samples=args.sample_n,
        )
        print("ASR metrics done")

        item = dict(alpha=alpha, fid=fid_val, mse=mse_val, ssim=ssim_val, asr=asr_val)
        results.append(item)
        print(f"[RESULT] alpha={alpha:.4f} | FID={fid_val} | MSE={mse_val:.6f} | SSIM={ssim_val:.6f} | ASR={asr_val:.6f}")

        del merged_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(results=results, output_dir=output_dir)
    plot_tradeoff(results=results, output_dir=output_dir, method=args.method)


if __name__ == "__main__":
    main()