from diffusers import DDPMPipeline

from .common import VarianceGatedScoreUNet, load_pipelines


def build_pipeline(args, ckpts, weights, dsl, device, clip_sample):
    pipes = [pipe.to(device) for pipe in load_pipelines(ckpts)]
    fused = DDPMPipeline(
        unet=VarianceGatedScoreUNet(
            [pipe.unet for pipe in pipes],
            weights=weights,
            temperature=args.maxfusion_temperature,
        ),
        scheduler=pipes[0].scheduler,
    )
    return fused
