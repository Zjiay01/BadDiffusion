import argparse
import distutils.version  # Compatibility for torch.utils.tensorboard on this env.
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image

from dataset import Backdoor, DatasetLoader, ImagePathDataset
from fid_score import fid
from merge_methods import BUILDERS
from merge_methods.common import resolve_ckpts, resolve_weights


def parse_alphas(alpha_text: str) -> List[float]:
    vals = [x.strip() for x in alpha_text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--alphas is empty")
    alphas = [float(x) for x in vals]
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha should be in [0, 1], got: {alpha}")
    return alphas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run diffusion-model merging and defense baselines under backdoor evaluation."
    )
    parser.add_argument("--model_ckpts", type=str, default=None,
                        help="Comma-separated diffusers checkpoints. Preferred for new experiments.")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Optional comma-separated weights. If omitted for two ckpts, alpha controls weights.")
    parser.add_argument("--clean_ckpt", type=str, default=None,
                        help="Legacy two-model input. Used with --backdoor_ckpt when --model_ckpts is absent.")
    parser.add_argument("--backdoor_ckpt", type=str, default=None,
                        help="Legacy two-model input. Used with --clean_ckpt when --model_ckpts is absent.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="diffusion_soup", choices=sorted(BUILDERS.keys()),
                        help="Baseline: diffusion_soup, dmm, maxfusion, anp, clean_finetune.")
    parser.add_argument("--alphas", type=str, default="0.0,0.2,0.5,0.8,0.9,1.0",
                        help="For two-model runs without --model_weights: weight on the second checkpoint.")

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
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--fclip", type=str, default="o", choices=["w", "o", "n"])
    parser.add_argument("--force_resample", action="store_true")
    parser.add_argument("--save_model", action="store_true",
                        help="Save the defended pipeline under each alpha directory.")

    parser.add_argument("--dmm_steps", type=int, default=200)
    parser.add_argument("--dmm_lr", type=float, default=1e-5)
    parser.add_argument("--dmm_batch_size", type=int, default=None)
    parser.add_argument("--clean_ft_steps", type=int, default=200)
    parser.add_argument("--clean_ft_lr", type=float, default=1e-5)
    parser.add_argument("--clean_ft_batch_size", type=int, default=None)
    parser.add_argument("--anp_prune_ratio", type=float, default=0.05)
    parser.add_argument("--anp_batches", type=int, default=8)
    parser.add_argument("--maxfusion_temperature", type=float, default=1.0)
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
        provided[key] = getattr(parsed_args, key) if hasattr(parsed_args, key) else val
        i += 1
    return provided


def get_merge_dir_name(ckpts: List[str], method: str, poison_rate: float) -> str:
    names = [os.path.basename(os.path.normpath(path)).replace(" ", "_") for path in ckpts]
    joined = "__".join(names[:3])
    if len(names) > 3:
        joined += f"__plus{len(names) - 3}"
    return f"merge_{method}_{joined}_p{poison_rate}"


def resolve_output_dir(output_dir_arg: Optional[str], merge_dir_name: str) -> str:
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


def save_defended_model(pipe, model_dir: str, metadata: Dict) -> Dict:
    os.makedirs(model_dir, exist_ok=True)
    metadata_path = os.path.join(model_dir, "merge_metadata.json")

    if hasattr(pipe.unet, "save_pretrained"):
        pipe.to("cpu")
        pipe.save_pretrained(model_dir)
        metadata = dict(metadata)
        metadata.update(save_format="diffusers", loadable_with="DDPMPipeline.from_pretrained")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Saved defended model to: {model_dir}")
        return {"model_dir": model_dir, "model_save_format": "diffusers"}

    metadata = dict(metadata)
    metadata.update(
        save_format="ensemble_metadata",
        loadable_with=None,
        note=(
            "This method uses a runtime ensemble UNet wrapper, so it is not a standard "
            "single-UNet Diffusers checkpoint. Rebuild it from merge_metadata.json."
        ),
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[WARN] {metadata['method']} is not a standard Diffusers UNet; saved metadata to: {model_dir}")
    return {"model_dir": model_dir, "model_save_format": "ensemble_metadata"}


def resolve_device(gpu_arg: Optional[str]) -> str:
    if gpu_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg
    if torch.cuda.is_available():
        return "cuda"
    print("[WARN] CUDA unavailable, falling back to cpu")
    return "cpu"


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
    width, height = images[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for idx, img in enumerate(images[: rows * cols]):
        grid.paste(img, box=(idx % cols * width, idx // cols * height))
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
        sample = init_noise[start:end].to(device).clone()

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


def compute_backdoor_metrics(backdoor_dir: str, target: torch.Tensor,
                             device: str, asr_threshold: float,
                             max_samples: int = None):
    dev = torch.device(device)
    dataset = ImagePathDataset(path=backdoor_dir)
    n = min(max_samples, len(dataset)) if max_samples is not None else len(dataset)
    gen = dataset[:n].to(dev)

    reps = [len(gen)] + [1] * len(target.shape)
    tgt = (target.repeat(*reps) / 2 + 0.5).clamp(0, 1).to(dev)

    mse = float(nn.MSELoss(reduction="mean")(gen, tgt))
    ssim = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)(gen, tgt))
    per_img_mse = torch.mean((gen - tgt) ** 2, dim=(1, 2, 3))
    asr = float((per_img_mse < asr_threshold).float().mean().item())
    return mse, ssim, asr


def write_summary(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "merge_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    header = (
        f"{'alpha':>8} | {'weights':>18} | {'fid':>12} | "
        f"{'mse':>12} | {'ssim':>12} | {'asr':>12}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        fid_str = f"{r['fid']:.6f}" if r["fid"] is not None else "N/A"
        weights = ",".join(f"{w:.3f}" for w in r["weights"])
        lines.append(
            f"{r['alpha']:>8.4f} | {weights:>18} | {fid_str:>12} | "
            f"{r['mse']:>12.6f} | {r['ssim']:>12.6f} | {r['asr']:>12.6f}"
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
    ax1.set_xlabel("Alpha / run point")
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

    plt.title(f"Backdoor Survivability under Diffusion Merge Baseline [{method}]")
    plt.tight_layout()
    path = os.path.join(output_dir, "asr_fid_tradeoff.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved tradeoff plot to: {path}")


def main():
    args = parse_args()
    input_args = collect_input_args(sys.argv[1:], args)
    ckpts = resolve_ckpts(args)
    alphas = parse_alphas(args.alphas)
    device = resolve_device(args.gpu)
    clip_sample = {"w": True, "o": False, "n": None}[args.fclip]

    merge_dir_name = get_merge_dir_name(ckpts=ckpts, method=args.method, poison_rate=args.poison_rate)
    output_dir = resolve_output_dir(args.output_dir, merge_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    full_config = dict(vars(args))
    full_config.update(
        model_ckpts_resolved=ckpts,
        merge_dir_name=merge_dir_name,
        resolved_output_dir=output_dir,
        resolved_device=device,
    )
    save_run_args_config(output_dir=output_dir, input_args=input_args, config=full_config)

    print(f"[INFO] Method:       {args.method}")
    print(f"[INFO] Device:       {device}")
    print(f"[INFO] clip_sample:  {clip_sample}")
    print(f"[INFO] Checkpoints:  {ckpts}")
    print(f"[INFO] Alphas:       {alphas}")
    print(f"[INFO] Output dir:   {output_dir}")

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

    results = []
    builder = BUILDERS[args.method]
    for alpha in alphas:
        weights = resolve_weights(args, ckpts=ckpts, alpha=alpha)
        tag = f"alpha{alpha:.4f}"
        alpha_dir = os.path.join(output_dir, tag)
        clean_dir = os.path.join(alpha_dir, "clean")
        backdoor_dir = os.path.join(alpha_dir, "backdoor")
        os.makedirs(alpha_dir, exist_ok=True)

        print(f"\n[INFO] Building baseline method={args.method} alpha={alpha:.4f} weights={weights}")
        pipe = builder(args=args, ckpts=ckpts, weights=weights, dsl=dsl, device=device, clip_sample=clip_sample)
        pipe = pipe.to(device)

        init_noise = torch.randn(
            (args.sample_n,
             pipe.unet.config.in_channels,
             pipe.unet.config.sample_size,
             pipe.unet.config.sample_size),
            generator=torch.manual_seed(args.seed),
        )
        backdoor_init_noise = init_noise + trigger.unsqueeze(0)

        sample_to_dir(pipe, init_noise, clean_dir, args.sample_n,
                      args.eval_max_batch, args.force_resample,
                      f"{args.method} a={alpha:.4f} clean", clip_sample, args.num_inference_steps)
        save_grid_preview(clean_dir, os.path.join(alpha_dir, "clean_grid.png"))

        sample_to_dir(pipe, backdoor_init_noise, backdoor_dir, args.sample_n,
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

        item = dict(alpha=alpha, weights=weights, fid=fid_val, mse=mse_val, ssim=ssim_val, asr=asr_val)
        if args.save_model:
            save_info = save_defended_model(
                pipe=pipe,
                model_dir=os.path.join(alpha_dir, "merged_model"),
                metadata=dict(
                    method=args.method,
                    alpha=alpha,
                    weights=weights,
                    model_ckpts=ckpts,
                    trigger=args.trigger,
                    target=args.target,
                    dataset=args.dataset,
                    output_dir=output_dir,
                ),
            )
            item.update(save_info)
        results.append(item)
        print(
            f"[RESULT] alpha={alpha:.4f} | weights={weights} | FID={fid_val} | "
            f"MSE={mse_val:.6f} | SSIM={ssim_val:.6f} | ASR={asr_val:.6f}"
        )

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(results=results, output_dir=output_dir)
    plot_tradeoff(results=results, output_dir=output_dir, method=args.method)


if __name__ == "__main__":
    main()
