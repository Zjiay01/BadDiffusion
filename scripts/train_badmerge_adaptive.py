#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - older torch fallback
    from torch.nn.utils.stateless import functional_call

from diffusers import DDPMPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import Backdoor, DatasetLoader, DEFAULT_VMAX, DEFAULT_VMIN
from loss import q_sample_diffuser
from util import normalize


DEFAULT_CLEAN = "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT_clean"
DEFAULT_INIT = "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BadMerging-style adaptive diffusion backdoor for weight averaging."
    )
    parser.add_argument("--clean_ckpt", default=DEFAULT_CLEAN)
    parser.add_argument("--init_ckpt", default=DEFAULT_INIT)
    parser.add_argument("--output_dir", default="merge_results/badmerge_cifar10_box14_hat")
    parser.add_argument("--dataset", default=DatasetLoader.CIFAR10)
    parser.add_argument("--dataset_path", default="datasets")
    parser.add_argument("--data_backend", default="hf", choices=["hf", "torchvision"],
                        help="Use hf for the original DatasetLoader path, or torchvision for CIFAR10 fallback.")
    parser.add_argument("--torchvision_download", action="store_true",
                        help="Allow torchvision to download CIFAR10 when --data_backend=torchvision.")
    parser.add_argument("--dataset_load_mode", default=DatasetLoader.MODE_FIXED)
    parser.add_argument("--trigger", default=Backdoor.TRIGGER_BOX_14)
    parser.add_argument("--target", default=Backdoor.TARGET_HAT)
    parser.add_argument("--clean_rate", type=float, default=1.0)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--merge_alpha", type=float, default=0.5, help="Weight on adaptive model in virtual soup.")
    parser.add_argument("--clean_weight", type=float, default=1.0,
                        help="Clean denoising loss weight on non-triggered samples.")
    parser.add_argument("--direct_weight", type=float, default=0.5,
                        help="Standalone backdoor loss weight on triggered samples.")
    parser.add_argument("--merged_weight", type=float, default=5.0,
                        help="Virtual-merged backdoor loss weight on triggered samples.")
    parser.add_argument("--anchor_weight", type=float, default=0.005, help="L2 penalty to keep attack weights near init.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bd_batch_size", type=int, default=64,
                        help="Triggered samples constructed per clean batch, matching BadMerging-style training.")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class TorchvisionCifar10BackdoorDataset(Dataset):
    def __init__(self, root: str, trigger_type: str, target_type: str, poison_rate: float,
                 seed: int, download: bool):
        self.base = tv_datasets.CIFAR10(root=root, train=True, download=download)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: normalize(
                vmin_in=0.0,
                vmax_in=1.0,
                vmin_out=DEFAULT_VMIN,
                vmax_out=DEFAULT_VMAX,
                x=x,
            )),
            transforms.RandomHorizontalFlip(),
        ])
        backdoor = Backdoor(root=root)
        self.trigger = backdoor.get_trigger(
            type=trigger_type,
            channel=3,
            image_size=32,
            vmin=DEFAULT_VMIN,
            vmax=DEFAULT_VMAX,
        )
        self.target = backdoor.get_target(
            type=target_type,
            trigger=self.trigger,
            vmin=DEFAULT_VMIN,
            vmax=DEFAULT_VMAX,
        )
        self.mask = torch.where(self.trigger > DEFAULT_VMIN, 0, 1)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self.base), generator=gen).tolist()
        poison_n = int(len(self.base) * poison_rate)
        self.clean_indices = perm[poison_n:]
        self.poison_indices = perm[:poison_n]

    def __len__(self):
        return len(self.clean_indices) + len(self.poison_indices)

    def __getitem__(self, idx):
        is_clean = idx < len(self.clean_indices)
        base_idx = self.clean_indices[idx] if is_clean else self.poison_indices[idx - len(self.clean_indices)]
        image, label = self.base[base_idx]
        image = self.transform(image)
        if is_clean:
            pixel_values = torch.zeros_like(image)
            target = image.clone()
        else:
            pixel_values = self.mask * image + (1 - self.mask) * self.trigger
            target = self.target.clone()
        return {
            DatasetLoader.PIXEL_VALUES: pixel_values,
            DatasetLoader.TARGET: target,
            DatasetLoader.IMAGE: image,
            DatasetLoader.LABEL: torch.tensor(label, dtype=torch.float),
            DatasetLoader.IS_CLEAN: is_clean,
        }


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_pred(unet, sample, timesteps):
    out = unet(sample.contiguous(), timesteps.contiguous(), return_dict=False)
    return out[0] if isinstance(out, tuple) else out.sample


def diffusion_loss(noise_sched, unet, x_start, residual, timesteps, noise):
    x_noisy, target = q_sample_diffuser(
        noise_sched=noise_sched,
        x_start=x_start,
        R=residual,
        timesteps=timesteps,
        noise=noise,
    )
    pred = model_pred(unet, x_noisy, timesteps)
    return F.mse_loss(pred, target)


def merged_functional_loss(noise_sched, attack_unet, clean_params, clean_buffers,
                           x_start, residual, timesteps, noise, merge_alpha: float):
    attack_params = dict(attack_unet.named_parameters())
    attack_buffers = dict(attack_unet.named_buffers())
    merged = {}
    for name, attack_param in attack_params.items():
        clean_param = clean_params[name].to(device=attack_param.device, dtype=attack_param.dtype)
        merged[name] = (1.0 - merge_alpha) * clean_param.detach() + merge_alpha * attack_param
    merged.update({name: buf for name, buf in attack_buffers.items()})
    for name, clean_buf in clean_buffers.items():
        if name not in merged:
            merged[name] = clean_buf.to(device=x_start.device)

    x_noisy, target = q_sample_diffuser(
        noise_sched=noise_sched,
        x_start=x_start,
        R=residual,
        timesteps=timesteps,
        noise=noise,
    )
    out = functional_call(
        attack_unet,
        merged,
        (x_noisy.contiguous(), timesteps.contiguous()),
        {"return_dict": False},
    )
    pred = out[0] if isinstance(out, tuple) else out.sample
    return F.mse_loss(pred, target)


def anchor_l2(attack_unet, init_params):
    loss = None
    count = 0
    for name, param in attack_unet.named_parameters():
        ref = init_params[name].to(device=param.device, dtype=param.dtype)
        item = F.mse_loss(param, ref.detach())
        loss = item if loss is None else loss + item
        count += 1
    return loss / max(count, 1)


def zero_loss_like(param: torch.Tensor):
    return param.sum() * 0.0


def build_backdoor_tensors(args, device):
    if args.dataset != DatasetLoader.CIFAR10:
        raise ValueError("BadMerging adaptive training currently builds paired backdoor batches for CIFAR10 only")
    backdoor = Backdoor(root=args.dataset_path)
    trigger = backdoor.get_trigger(
        type=args.trigger,
        channel=3,
        image_size=32,
        vmin=DEFAULT_VMIN,
        vmax=DEFAULT_VMAX,
    ).to(device)
    target = backdoor.get_target(
        type=args.target,
        trigger=trigger.detach().cpu(),
        vmin=DEFAULT_VMIN,
        vmax=DEFAULT_VMAX,
    ).to(device)
    mask = torch.where(trigger > DEFAULT_VMIN, 0, 1).to(device)
    return trigger, target, mask


def make_triggered_batch(images, trigger, target, mask, batch_size: int):
    bd_n = min(int(batch_size), len(images))
    if bd_n <= 0:
        return None, None
    source = images[:bd_n]
    repeat_shape = (bd_n,) + (1,) * len(source.shape[1:])
    residual = mask.repeat(*repeat_shape) * source + (1 - mask.repeat(*repeat_shape)) * trigger.repeat(*repeat_shape)
    targets = target.repeat(*repeat_shape)
    return targets, residual


def save_pipeline(pipe, attack_unet, output_dir: Path, name: str, args, step: int):
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    pipe.unet = attack_unet
    pipe.save_pretrained(save_dir)
    with (save_dir / "badmerge_training.json").open("w") as f:
        json.dump({"step": step, **vars(args)}, f, indent=2)
    return save_dir


def build_loader(args):
    if args.data_backend == "torchvision":
        if args.dataset != DatasetLoader.CIFAR10:
            raise ValueError("--data_backend=torchvision currently supports CIFAR10 only")
        dataset = TorchvisionCifar10BackdoorDataset(
            root=args.dataset_path,
            trigger_type=args.trigger,
            target_type=args.target,
            poison_rate=args.poison_rate,
            seed=args.seed,
            download=args.torchvision_download,
        )
    else:
        dataset = (
            DatasetLoader(root=args.dataset_path, name=args.dataset, batch_size=args.batch_size)
            .set_poison(
                trigger_type=args.trigger,
                target_type=args.target,
                clean_rate=args.clean_rate,
                poison_rate=args.poison_rate,
            )
            .prepare_dataset(mode=args.dataset_load_mode)
            .get_dataset()
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    clean_pipe = DDPMPipeline.from_pretrained(os.path.abspath(args.clean_ckpt)).to(device)
    attack_pipe = DDPMPipeline.from_pretrained(os.path.abspath(args.init_ckpt)).to(device)
    attack_unet = attack_pipe.unet
    clean_unet = clean_pipe.unet
    noise_sched = attack_pipe.scheduler
    attack_unet.train()
    clean_unet.eval()
    for param in clean_unet.parameters():
        param.requires_grad_(False)

    clean_params = {k: v.detach().clone().to(device) for k, v in clean_unet.named_parameters()}
    clean_buffers = {k: v.detach().clone().to(device) for k, v in clean_unet.named_buffers()}
    init_params = {k: v.detach().clone().to(device) for k, v in attack_unet.named_parameters()}
    trigger, target_pattern, backdoor_mask = build_backdoor_tensors(args, device)

    loader = build_loader(args)
    optim = torch.optim.AdamW(attack_unet.parameters(), lr=args.lr)

    metrics_path = output_dir / "training_metrics.jsonl"
    step = 0
    pbar = tqdm(total=args.max_steps, desc="BadMerging adaptive train", unit="step")
    while step < args.max_steps:
        for batch in loader:
            images = batch[DatasetLoader.IMAGE].to(device)
            clean_residual = torch.zeros_like(images)
            bd_targets, bd_residual = make_triggered_batch(
                images=images,
                trigger=trigger,
                target=target_pattern,
                mask=backdoor_mask,
                batch_size=args.bd_batch_size,
            )
            clean_noise = torch.randn_like(images)
            timesteps = torch.randint(
                0,
                noise_sched.config.num_train_timesteps,
                (len(images),),
                device=device,
            ).long()

            clean = diffusion_loss(
                noise_sched,
                attack_unet,
                images,
                clean_residual,
                timesteps,
                clean_noise,
            )
            direct = zero_loss_like(images)
            merged = zero_loss_like(images)

            if bd_targets is not None:
                bd_noise = torch.randn_like(bd_targets)
                bd_timesteps = torch.randint(
                    0,
                    noise_sched.config.num_train_timesteps,
                    (len(bd_targets),),
                    device=device,
                ).long()
                direct = diffusion_loss(
                    noise_sched,
                    attack_unet,
                    bd_targets,
                    bd_residual,
                    bd_timesteps,
                    bd_noise,
                )
                merged = merged_functional_loss(
                    noise_sched,
                    attack_unet,
                    clean_params,
                    clean_buffers,
                    bd_targets,
                    bd_residual,
                    bd_timesteps,
                    bd_noise,
                    args.merge_alpha,
                )
            anchor = anchor_l2(attack_unet, init_params)
            loss = (
                args.clean_weight * clean
                + args.direct_weight * direct
                + args.merged_weight * merged
                + args.anchor_weight * anchor
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(attack_unet.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            step += 1
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "clean_loss": float(clean.detach().cpu()),
                "direct_backdoor_loss": float(direct.detach().cpu()),
                "merged_backdoor_loss": float(merged.detach().cpu()),
                "anchor_loss": float(anchor.detach().cpu()),
                "clean_n": int(len(images)),
                "poison_n": int(0 if bd_targets is None else len(bd_targets)),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            pbar.set_postfix(loss=f"{record['loss']:.4f}", merged=f"{record['merged_backdoor_loss']:.4f}")
            pbar.update(1)

            if step % args.save_every == 0:
                save_pipeline(attack_pipe, attack_unet, output_dir, f"step_{step}", args, step)
            if step >= args.max_steps:
                break

    pbar.close()
    final_dir = save_pipeline(attack_pipe, attack_unet, output_dir, "final", args, step)
    print(f"[DONE] saved adaptive backdoor checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
