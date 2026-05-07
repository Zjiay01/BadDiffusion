import copy
import distutils.version  # Compatibility for torch.utils.tensorboard on this env.
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from diffusers import DDPMPipeline

from dataset import DatasetLoader


def parse_csv(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_csv(text: Optional[str]) -> Optional[List[float]]:
    vals = parse_csv(text)
    if not vals:
        return None
    return [float(x) for x in vals]


def resolve_ckpts(args) -> List[str]:
    ckpts = parse_csv(args.model_ckpts)
    if ckpts:
        return ckpts
    legacy = [x for x in [args.clean_ckpt, args.backdoor_ckpt] if x]
    if len(legacy) >= 2:
        return legacy
    raise ValueError("Provide --model_ckpts, or both --clean_ckpt and --backdoor_ckpt.")


def resolve_weights(args, ckpts: List[str], alpha: float) -> List[float]:
    explicit = parse_float_csv(args.model_weights)
    if explicit is not None:
        if len(explicit) != len(ckpts):
            raise ValueError("--model_weights length must match --model_ckpts length")
        total = sum(explicit)
        if total == 0:
            raise ValueError("--model_weights must not sum to zero")
        return [w / total for w in explicit]
    if len(ckpts) == 2:
        return [1.0 - alpha, alpha]
    return [1.0 / len(ckpts)] * len(ckpts)


def load_pipelines(ckpts: List[str], torch_dtype=None) -> List[DDPMPipeline]:
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return [DDPMPipeline.from_pretrained(ckpt, **kwargs) for ckpt in ckpts]


def weighted_state_dict(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    merged = {}
    base = states[0]
    for key, value in base.items():
        if not torch.is_floating_point(value):
            merged[key] = value
            continue
        acc = torch.zeros_like(value.float())
        for state, weight in zip(states, weights):
            acc += state[key].float() * weight
        merged[key] = acc.to(value.dtype)
    return merged


def build_soup_pipeline(ckpts: List[str], weights: List[float]) -> DDPMPipeline:
    pipes = load_pipelines(ckpts)
    states = [pipe.unet.state_dict() for pipe in pipes]
    merged_state = weighted_state_dict(states, weights)
    pipe = DDPMPipeline.from_pretrained(ckpts[0])
    pipe.unet.load_state_dict(merged_state, strict=True)
    return pipe


def clone_scheduler(scheduler):
    return copy.deepcopy(scheduler)


def set_trainable(module: nn.Module, trainable: bool):
    for param in module.parameters():
        param.requires_grad_(trainable)


def infer_pipe_device(pipe) -> torch.device:
    return next(pipe.unet.parameters()).device


def q_sample(noise_sched, x_start: torch.Tensor, r: torch.Tensor, timesteps: torch.Tensor,
             noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(x_start)
    alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)

    sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5
    r_coef_t = (1 - alphas[timesteps] ** 0.5) * sqrt_one_minus_alphas_cumprod_t / (1 - alphas[timesteps])

    view_shape = (len(x_start),) + (1,) * len(x_start.shape[1:])
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(view_shape)
    r_coef_t = r_coef_t.reshape(view_shape)
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
    return noisy_images + (1 - sqrt_alphas_cumprod_t) * r, r_coef_t * r + noise


def diffusion_loss(noise_sched, model: nn.Module, x_start: torch.Tensor, r: torch.Tensor,
                   timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None):
    x_noisy, target = q_sample(noise_sched, x_start=x_start, r=r, timesteps=timesteps, noise=noise)
    pred = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]
    return F.mse_loss(pred, target)


def iter_batches(dsl: DatasetLoader, batch_size: int, max_steps: int, seed: int):
    loader = torch.utils.data.DataLoader(
        dsl.get_dataset(),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    step = 0
    while step < max_steps:
        for batch in loader:
            yield batch
            step += 1
            if step >= max_steps:
                break


def clean_finetune(pipe: DDPMPipeline, dsl: DatasetLoader, steps: int, lr: float,
                   batch_size: int, seed: int, device: str, desc: str):
    if steps <= 0:
        return pipe
    pipe = pipe.to(device)
    pipe.unet.train()
    optim = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)
    max_t = pipe.scheduler.config.num_train_timesteps
    pbar = tqdm(total=steps, desc=desc, unit="step")
    for batch in iter_batches(dsl=dsl, batch_size=batch_size, max_steps=steps, seed=seed):
        images = batch[DatasetLoader.IMAGE].to(device)
        zeros = torch.zeros_like(images)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, max_t, (len(images),), device=device).long()
        loss = diffusion_loss(pipe.scheduler, pipe.unet, images, zeros, timesteps, noise=noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{loss.detach().item():.4f}")
        pbar.update(1)
    pbar.close()
    pipe.unet.eval()
    return pipe


class WeightedScoreUNet(nn.Module):
    def __init__(self, unets: Iterable[nn.Module], weights: List[float]):
        super().__init__()
        self.unets = nn.ModuleList(list(unets))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32), persistent=False)
        self.config = self.unets[0].config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sample, timestep, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        outs = []
        for unet in self.unets:
            out = unet(sample, timestep, **kwargs)
            outs.append(out.sample if hasattr(out, "sample") else out[0])
        stacked = torch.stack(outs, dim=0)
        weights = self.weights.to(device=stacked.device, dtype=stacked.dtype)
        while weights.dim() < stacked.dim():
            weights = weights.view(*weights.shape, *([1] * (stacked.dim() - weights.dim())))
        sample = (stacked * weights).sum(dim=0)
        if not return_dict:
            return (sample,)
        return SimpleNamespace(sample=sample)


class VarianceGatedScoreUNet(nn.Module):
    def __init__(self, unets: Iterable[nn.Module], weights: List[float], temperature: float = 1.0):
        super().__init__()
        self.unets = nn.ModuleList(list(unets))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32), persistent=False)
        self.temperature = max(float(temperature), 1e-6)
        self.config = self.unets[0].config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sample, timestep, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        outs = []
        for unet in self.unets:
            out = unet(sample, timestep, **kwargs)
            outs.append(out.sample if hasattr(out, "sample") else out[0])
        stacked = torch.stack(outs, dim=0)
        disagreement = stacked.var(dim=0, unbiased=False)
        disagreement = disagreement.mean(dim=tuple(range(1, disagreement.dim())), keepdim=True)
        gate = torch.sigmoid(-disagreement / self.temperature)
        weights = self.weights.to(device=stacked.device, dtype=stacked.dtype)
        while weights.dim() < stacked.dim():
            weights = weights.view(*weights.shape, *([1] * (stacked.dim() - weights.dim())))
        weighted = (stacked * weights).sum(dim=0)
        conservative = stacked[0]
        sample = gate * weighted + (1 - gate) * conservative
        if not return_dict:
            return (sample,)
        return SimpleNamespace(sample=sample)
