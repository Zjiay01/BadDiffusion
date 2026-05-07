import torch
from tqdm import tqdm

from .common import build_soup_pipeline, load_pipelines, set_trainable


def build_pipeline(args, ckpts, weights, dsl, device, clip_sample):
    student = build_soup_pipeline(ckpts=ckpts, weights=weights).to(device)
    teachers = load_pipelines(ckpts)
    teachers = [pipe.to(device) for pipe in teachers]
    for pipe in teachers:
        pipe.unet.eval()
        set_trainable(pipe.unet, False)

    steps = int(args.dmm_steps)
    if steps <= 0:
        return student

    batch_size = args.dmm_batch_size or min(args.eval_max_batch, 64)
    optim = torch.optim.AdamW(student.unet.parameters(), lr=args.dmm_lr)
    max_t = student.scheduler.config.num_train_timesteps
    sample_size = student.unet.config.sample_size
    channels = student.unet.config.in_channels
    torch.manual_seed(args.seed)

    student.unet.train()
    pbar = tqdm(total=steps, desc="DMM distillation", unit="step")
    for _ in range(steps):
        x_t = torch.randn((batch_size, channels, sample_size, sample_size), device=device)
        timesteps = torch.randint(0, max_t, (batch_size,), device=device).long()
        teacher_out = torch.zeros_like(x_t)
        with torch.no_grad():
            for weight, teacher in zip(weights, teachers):
                teacher_out += teacher.unet(x_t, timesteps).sample * weight
        pred = student.unet(x_t, timesteps).sample
        loss = torch.nn.functional.mse_loss(pred, teacher_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.unet.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{loss.detach().item():.4f}")
        pbar.update(1)
    pbar.close()
    student.unet.eval()
    return student
