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
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:0")
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


def resolve_device(gpu_arg: str, device_arg: str) -> str:
    if gpu_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg
        if device_arg not in [None, "auto", "cuda", "cpu"]:
            print(
                f"[WARN] Both --gpu and --device={device_arg} are set. "
                "When --gpu is used, prefer --device=auto/cuda/cpu."
            )

    if device_arg is None:
        device_arg = "auto"
    device_arg = device_arg.strip().lower()

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_arg == "cpu":
        return "cpu"

    if device_arg == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[WARN] --device=cuda but CUDA is unavailable, fallback to cpu")
        return "cpu"

    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            print(f"[WARN] --device={device_arg} but CUDA is unavailable, fallback to cpu")
            return "cpu"
        idx_text = device_arg.split(":", 1)[1]
        if idx_text.isdigit():
            idx = int(idx_text)
            if idx < torch.cuda.device_count():
                return device_arg
            print(
                f"[WARN] --device={device_arg} out of range (cuda device count={torch.cuda.device_count()}), "
                "fallback to cuda"
            )
            return "cuda"

    print(f"[WARN] Unknown --device={device_arg}, fallback to auto")
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def sample_to_dir(pipe: DDPMPipeline, init_noise: torch.Tensor, out_dir: str, sample_n: int, eval_max_batch: int, seed: int, force_resample: bool, stage_name: str):
    os.makedirs(out_dir, exist_ok=True)
    if force_resample:
        clear_images(out_dir)

    existing_n = count_images(out_dir)
    if existing_n >= sample_n:
        print(f"[INFO] {stage_name}: found {existing_n}/{sample_n}, skip sampling")
        return

    remaining = sample_n - existing_n
    rng = torch.Generator().manual_seed(seed)
    pbar = tqdm(total=sample_n, initial=existing_n, desc=f"{stage_name} sampling", unit="img")

    start = existing_n
    while remaining > 0:
        batch_sz = min(eval_max_batch, remaining)
        end = start + batch_sz
        batch_init = init_noise[start:end]

        pipe_res = pipe(
            batch_size=batch_sz,
            generator=rng,
            init=batch_init,
            output_type=None,
        )
        images = pipe_res.images

        pil_images = [Image.fromarray(image) for image in ((images * 255).round().astype("uint8"))]
        for i, img in enumerate(pil_images):
            img.save(os.path.join(out_dir, f"{start + i}.png"))

        start = end
        remaining = sample_n - start
        pbar.update(batch_sz)

        del pipe_res

    pbar.close()


def compute_backdoor_metrics(backdoor_dir: str, target: torch.Tensor, device: str, asr_threshold: float):
    dev = torch.device(device)
    gen = ImagePathDataset(path=backdoor_dir)[:].to(dev)

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


def main():
    args = parse_args()
    input_args = collect_input_args(sys.argv[1:], args)
    alphas = parse_alphas(args.alphas)
    device = resolve_device(gpu_arg=args.gpu, device_arg=args.device)
    merge_dir_name = get_merge_dir_name(backdoor_ckpt=args.backdoor_ckpt, poison_rate=args.poison_rate)
    output_dir = resolve_output_dir(output_dir_arg=args.output_dir, merge_dir_name=merge_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    full_config = dict(vars(args))
    full_config["merge_dir_name"] = merge_dir_name
    full_config["resolved_output_dir"] = output_dir
    full_config["resolved_device"] = device
    save_run_args_config(output_dir=output_dir, input_args=input_args, config=full_config)

    print(f"[INFO] Device: {device}")
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
            seed=args.seed,
            force_resample=args.force_resample,
            stage_name=f"alpha={alpha:.4f} clean",
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
            seed=args.seed,
            force_resample=args.force_resample,
            stage_name=f"alpha={alpha:.4f} backdoor",
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


if __name__ == "__main__":
    main()
