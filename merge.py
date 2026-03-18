import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image

from diffusers import DDPMPipeline

from dataset import Backdoor, DatasetLoader, ImagePathDataset
from fid_score import fid


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
        description="Merge two trained diffusion models with weighted averaging and evaluate backdoor robustness."
    )
    parser.add_argument("--backdoor_ckpt", type=str, required=True, help="Path to backdoored model checkpoint directory")
    parser.add_argument("--clean_ckpt", type=str, required=True, help="Path to clean model checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. If not set, uses merge_{model_name}_{poison_rate}")

    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.2,0.5,0.8,0.9,1.0",
        help="Comma-separated backdoor weight list, e.g. 0.0,0.2,0.5,0.8,1.0",
    )
    parser.add_argument("--dataset", type=str, default=DatasetLoader.CIFAR10, choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument("--dataset_path", type=str, default="datasets", help="Dataset root path")
    parser.add_argument("--dataset_load_mode", type=str, default=DatasetLoader.MODE_FIXED, choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])

    parser.add_argument("--clean_rate", type=float, default=1.0)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--trigger", type=str, default=Backdoor.TRIGGER_BOX_14)
    parser.add_argument("--target", type=str, default=Backdoor.TARGET_HAT)

    parser.add_argument("--sample_n", type=int, default=2048, help="Number of samples used for evaluation")
    parser.add_argument("--eval_max_batch", type=int, default=256, help="Maximum batch size during sampling")
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--fid_num_workers", type=int, default=4)
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID computation")
    parser.add_argument("--asr_threshold", type=float, default=0.05, help="Per-image MSE threshold for ASR")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None, help="GPU id(s), e.g. 0 or 0,1. This sets CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--fclip",
        type=str,
        default="o",
        choices=["w", "o", "n"],
        help="Force clip_sample during sampling: w=True, o=False, n=keep scheduler default",
    )
    parser.add_argument("--force_resample", action="store_true", help="Resample images even if output folder already exists")

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


def get_merge_dir_name(backdoor_ckpt: str, poison_rate: float) -> str:
    model_name = os.path.basename(os.path.normpath(backdoor_ckpt))
    model_name = model_name.replace(" ", "_")
    return f"merge_{model_name}_{poison_rate}"


def resolve_output_dir(output_dir_arg: str, merge_dir_name: str) -> str:
    if output_dir_arg is None:
        return merge_dir_name
    normalized = os.path.basename(os.path.normpath(output_dir_arg))
    if normalized.startswith("merge_"):
        return output_dir_arg
    return os.path.join(output_dir_arg, merge_dir_name)


def save_run_args_config(output_dir: str, input_args: Dict, config: Dict):
    os.makedirs(output_dir, exist_ok=True)
    args_path = os.path.join(output_dir, "args.json")
    config_path = os.path.join(output_dir, "config.json")
    with open(args_path, "w") as f:
        json.dump(input_args, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved input args to: {args_path}")
    print(f"[INFO] Saved full config to: {config_path}")


def resolve_device(gpu_arg: str) -> str:
    if gpu_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg

    if torch.cuda.is_available():
        return "cuda"

    print("[WARN] CUDA is unavailable, fallback to cpu")
    return "cpu"


def count_images(path: str) -> int:
    exts = ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]
    cnt = 0
    for name in os.listdir(path):
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in exts:
            cnt += 1
    return cnt


def clear_images(path: str):
    if not os.path.isdir(path):
        return
    exts = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
    for name in os.listdir(path):
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in exts:
            os.remove(os.path.join(path, name))


def make_grid_img(images: List[Image.Image], rows: int = 8, cols: int = 8) -> Image.Image:
    if len(images) == 0:
        raise ValueError("No images provided for grid generation")
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images[: rows * cols]):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def save_grid_preview(sample_dir: str, out_file: str, rows: int = 8, cols: int = 8):
    if not os.path.isdir(sample_dir):
        return
    names = [x for x in os.listdir(sample_dir) if x.lower().endswith(".png")]
    if len(names) == 0:
        return
    names = sorted(names, key=lambda x: int(x.rsplit(".", 1)[0]) if x.rsplit(".", 1)[0].isdigit() else x)
    pick = names[: rows * cols]
    imgs = []
    for name in pick:
        p = os.path.join(sample_dir, name)
        with Image.open(p) as img:
            imgs.append(img.convert("RGB"))
    grid = make_grid_img(imgs, rows=rows, cols=cols)
    grid.save(out_file)


def maybe_generate_real_images(dsl: DatasetLoader, out_dir: str, sample_n: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    enough = count_images(out_dir) >= sample_n
    if enough:
        return

    print(f"[INFO] Generating {sample_n} real images for FID reference at: {out_dir}")
    ds = dsl.get_dataset().shuffle(seed=seed)
    imgs = ds[:sample_n][DatasetLoader.IMAGE]
    for i, img in enumerate(tqdm(imgs, desc="Save real images")):
        dsl.show_sample(img=img, is_show=False, file_name=os.path.join(out_dir, f"{i}.png"))


def make_merged_unet_state(backdoor_state: Dict[str, torch.Tensor], clean_state: Dict[str, torch.Tensor], alpha: float):
    if set(backdoor_state.keys()) != set(clean_state.keys()):
        raise ValueError("UNet state dict keys mismatch between backdoor and clean checkpoints")

    merged = {}
    for k in backdoor_state.keys():
        bd = backdoor_state[k]
        cl = clean_state[k]
        if not torch.is_floating_point(bd):
            merged[k] = bd
            continue

        mix = cl.to(torch.float32) * (1.0 - alpha) + bd.to(torch.float32) * alpha
        merged[k] = mix.to(dtype=bd.dtype)
    return merged


def sample_to_dir(pipe, init_noise, out_dir, sample_n, eval_max_batch, force_resample, stage_name, clip_sample):
    os.makedirs(out_dir, exist_ok=True)
    if force_resample:
        clear_images(out_dir)

    existing_n = count_images(out_dir)
    if existing_n >= sample_n:
        print(f"[INFO] {stage_name}: found {existing_n}/{sample_n}, skip sampling")
        return

    device = next(pipe.unet.parameters()).device
    if hasattr(pipe.scheduler, "config") and hasattr(pipe.scheduler.config, "clip_sample") and clip_sample is not None:
        pipe.scheduler.config.clip_sample = clip_sample
    pipe.scheduler.set_timesteps(1000)

    start = existing_n
    remaining = sample_n - existing_n
    batch_total = (remaining + eval_max_batch - 1) // eval_max_batch if remaining > 0 else 0
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

        images = (sample / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        for i, img_arr in enumerate(images):
            img = Image.fromarray((img_arr * 255).round().astype("uint8"))
            img.save(os.path.join(out_dir, f"{start + i}.png"))

        start = end
        remaining = sample_n - start
        pbar.update(batch_sz)

    pbar.close()


def compute_backdoor_metrics(backdoor_dir: str, target: torch.Tensor, device: str, asr_threshold: float, max_samples: int = None):
    dev = torch.device(device)
    dataset = ImagePathDataset(path=backdoor_dir)
    if max_samples is not None:
        n = min(max_samples, len(dataset))
        gen = dataset[:n].to(dev)
    else:
        gen = dataset[:].to(dev)

    reps = [len(gen)] + ([1] * len(target.shape))
    tgt = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(dev)

    mse = float(nn.MSELoss(reduction="mean")(gen, tgt))
    ssim = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)(gen, tgt))

    per_img_mse = torch.mean((gen - tgt) ** 2, dim=(1, 2, 3))
    asr = float(torch.mean((per_img_mse < asr_threshold).to(torch.float32)).item())
    return mse, ssim, asr


def write_summary(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "merge_summary.json")
    txt_path = os.path.join(output_dir, "merge_summary.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    header = f"{'alpha':>8} | {'fid':>12} | {'mse':>12} | {'ssim':>12} | {'asr':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        fid_val = r["fid"]
        fid_str = f"{fid_val:.6f}" if fid_val is not None else "N/A"
        lines.append(
            f"{r['alpha']:>8.4f} | {fid_str:>12} | {r['mse']:>12.6f} | {r['ssim']:>12.6f} | {r['asr']:>12.6f}"
        )

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n[INFO] Saved summary to: {json_path}")
    print(f"[INFO] Saved summary to: {txt_path}")


def plot_tradeoff(results: List[Dict], output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skip plotting asr_fid_tradeoff.png")
        return

    alphas = [float(r["alpha"]) for r in results]
    asrs = [float(r["asr"]) for r in results]
    fid_pairs = [(float(r["alpha"]), float(r["fid"])) for r in results if r.get("fid") is not None]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_asr = "tab:red"
    color_fid = "tab:blue"

    ax1.set_xlabel("Alpha (backdoor model weight)")
    ax1.set_ylabel("ASR", color=color_asr)
    ax1.plot(alphas, asrs, "o-", color=color_asr, label="ASR")
    ax1.tick_params(axis="y", labelcolor=color_asr)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if len(fid_pairs) > 0:
        ax2 = ax1.twinx()
        fid_alphas = [p[0] for p in fid_pairs]
        fids = [p[1] for p in fid_pairs]
        ax2.set_ylabel("FID", color=color_fid)
        ax2.plot(fid_alphas, fids, "s--", color=color_fid, label="FID")
        ax2.tick_params(axis="y", labelcolor=color_fid)
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    else:
        ax1.legend(lines1, labels1, loc="center left")
        ax1.text(
            0.98,
            0.02,
            "FID unavailable (possibly --skip_fid)",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="tab:gray",
        )

    plt.title("Backdoor Survivability under Model Merging")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "asr_fid_tradeoff.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved tradeoff plot to: {plot_path}")


def main():
    args = parse_args()
    input_args = collect_input_args(sys.argv[1:], args)
    alphas = parse_alphas(args.alphas)
    device = resolve_device(gpu_arg=args.gpu)
    if args.fclip == "w":
        clip_sample = True
    elif args.fclip == "o":
        clip_sample = False
    else:
        clip_sample = None
    merge_dir_name = get_merge_dir_name(backdoor_ckpt=args.backdoor_ckpt, poison_rate=args.poison_rate)
    output_dir = resolve_output_dir(output_dir_arg=args.output_dir, merge_dir_name=merge_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    full_config = dict(vars(args))
    full_config["merge_dir_name"] = merge_dir_name
    full_config["resolved_output_dir"] = output_dir
    full_config["resolved_device"] = device
    save_run_args_config(output_dir=output_dir, input_args=input_args, config=full_config)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] clip_sample: {clip_sample} (from --fclip={args.fclip})")
    print(f"[INFO] Alphas: {alphas}")
    print(f"[INFO] Output dir: {output_dir}")

    dsl = DatasetLoader(root=args.dataset_path, name=args.dataset, batch_size=args.eval_max_batch).set_poison(
        trigger_type=args.trigger,
        target_type=args.target,
        clean_rate=args.clean_rate,
        poison_rate=args.poison_rate,
    ).prepare_dataset(mode=args.dataset_load_mode)

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
        (args.sample_n, backdoor_pipe.unet.config.in_channels, backdoor_pipe.unet.config.sample_size, backdoor_pipe.unet.config.sample_size),
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

        print(f"\n[INFO] Evaluating alpha={alpha:.4f}")
        merged_state = make_merged_unet_state(backdoor_state=backdoor_state, clean_state=clean_state, alpha=alpha)
        merged_pipe = DDPMPipeline.from_pretrained(args.backdoor_ckpt)
        merged_pipe.unet.load_state_dict(merged_state, strict=True)
        merged_pipe = merged_pipe.to(device)

        sample_to_dir(
            pipe=merged_pipe,
            init_noise=init_noise,
            out_dir=clean_dir,
            sample_n=args.sample_n,
            eval_max_batch=args.eval_max_batch,
            force_resample=args.force_resample,
            stage_name=f"alpha={alpha:.4f} clean",
            clip_sample=clip_sample,
        )
        save_grid_preview(
            sample_dir=clean_dir,
            out_file=os.path.join(alpha_dir, "clean_grid.png"),
            rows=8,
            cols=8,
        )
        sample_to_dir(
            pipe=merged_pipe,
            init_noise=backdoor_init_noise,
            out_dir=backdoor_dir,
            sample_n=args.sample_n,
            eval_max_batch=args.eval_max_batch,
            force_resample=args.force_resample,
            stage_name=f"alpha={alpha:.4f} backdoor",
            clip_sample=clip_sample,
        )
        save_grid_preview(
            sample_dir=backdoor_dir,
            out_file=os.path.join(alpha_dir, "backdoor_grid.png"),
            rows=8,
            cols=8,
        )

        fid_val = None
        if not args.skip_fid:
            fid_val = float(
                fid(
                    path=[real_dir, clean_dir],
                    batch_size=args.fid_batch_size,
                    device=device,
                    num_workers=args.fid_num_workers,
                )
            )

        mse_val, ssim_val, asr_val = compute_backdoor_metrics(
            backdoor_dir=backdoor_dir,
            target=target,
            device=device,
            asr_threshold=args.asr_threshold,
            max_samples=args.sample_n,
        )

        item = {
            "alpha": alpha,
            "fid": fid_val,
            "mse": mse_val,
            "ssim": ssim_val,
            "asr": asr_val,
        }
        results.append(item)
        print(
            f"[RESULT] alpha={alpha:.4f} | FID={fid_val} | MSE={mse_val:.6f} | SSIM={ssim_val:.6f} | ASR={asr_val:.6f}"
        )

        del merged_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(results=results, output_dir=output_dir)
    plot_tradeoff(results=results, output_dir=output_dir)


if __name__ == "__main__":
    main()
