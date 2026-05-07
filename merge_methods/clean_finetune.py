from .common import build_soup_pipeline, clean_finetune


def build_pipeline(args, ckpts, weights, dsl, device, clip_sample):
    pipe = build_soup_pipeline(ckpts=ckpts, weights=weights)
    batch_size = args.clean_ft_batch_size or min(args.eval_max_batch, 64)
    return clean_finetune(
        pipe=pipe,
        dsl=dsl,
        steps=args.clean_ft_steps,
        lr=args.clean_ft_lr,
        batch_size=batch_size,
        seed=args.seed,
        device=device,
        desc="Clean fine-tuning",
    )
