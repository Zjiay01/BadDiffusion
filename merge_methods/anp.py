from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm

from dataset import DatasetLoader

from .common import build_soup_pipeline, iter_batches, q_sample


def _score_conv_channels(pipe, dsl, batches: int, batch_size: int, seed: int, device: str):
    scores = defaultdict(float)
    handles = []
    current_key = None

    convs = [(name, module) for name, module in pipe.unet.named_modules() if isinstance(module, nn.Conv2d)]

    def make_hook(name):
        def hook(_module, _inp, out):
            nonlocal current_key
            if current_key is None:
                return
            value = out.detach().abs().mean(dim=(0, 2, 3)).cpu()
            scores[(current_key, name)] += value
        return hook

    for name, module in convs:
        handles.append(module.register_forward_hook(make_hook(name)))

    pipe = pipe.to(device)
    pipe.unet.eval()
    max_t = pipe.scheduler.config.num_train_timesteps
    pbar = tqdm(total=batches, desc="ANP channel scoring", unit="batch")
    try:
        for batch in iter_batches(dsl=dsl, batch_size=batch_size, max_steps=batches, seed=seed):
            images = batch[DatasetLoader.IMAGE].to(device)
            target = batch[DatasetLoader.TARGET].to(device)
            trigger = batch[DatasetLoader.PIXEL_VALUES].to(device)
            zeros = torch.zeros_like(images)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, max_t, (len(images),), device=device).long()

            clean_x, _ = q_sample(pipe.scheduler, x_start=images, r=zeros, timesteps=timesteps, noise=noise)
            backdoor_x, _ = q_sample(pipe.scheduler, x_start=target, r=trigger, timesteps=timesteps, noise=noise)

            with torch.no_grad():
                current_key = "clean"
                pipe.unet(clean_x, timesteps)
                current_key = "backdoor"
                pipe.unet(backdoor_x, timesteps)
                current_key = None
            pbar.update(1)
    finally:
        current_key = None
        for handle in handles:
            handle.remove()
        pbar.close()

    channel_scores = {}
    for name, module in convs:
        clean = scores.get(("clean", name))
        backdoor = scores.get(("backdoor", name))
        if clean is None or backdoor is None:
            continue
        channel_scores[name] = (backdoor - clean).clamp(min=0)
    return convs, channel_scores


def _prune_channels(convs, channel_scores, ratio: float):
    if ratio <= 0:
        return
    flat = []
    for name, score in channel_scores.items():
        for idx, value in enumerate(score):
            flat.append((float(value), name, idx))
    if not flat:
        return
    flat.sort(reverse=True, key=lambda x: x[0])
    prune_n = max(1, int(len(flat) * ratio))
    prune_map = defaultdict(list)
    for _value, name, idx in flat[:prune_n]:
        prune_map[name].append(idx)

    conv_by_name = dict(convs)
    with torch.no_grad():
        for name, indices in prune_map.items():
            module = conv_by_name[name]
            idx = torch.tensor(indices, device=module.weight.device).long()
            module.weight[idx] = 0
            if module.bias is not None:
                module.bias[idx] = 0


def build_pipeline(args, ckpts, weights, dsl, device, clip_sample):
    pipe = build_soup_pipeline(ckpts=ckpts, weights=weights)
    batch_size = min(args.eval_max_batch, 64)
    convs, channel_scores = _score_conv_channels(
        pipe=pipe,
        dsl=dsl,
        batches=args.anp_batches,
        batch_size=batch_size,
        seed=args.seed,
        device=device,
    )
    _prune_channels(convs=convs, channel_scores=channel_scores, ratio=args.anp_prune_ratio)
    return pipe
