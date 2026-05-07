from .common import build_soup_pipeline


def build_pipeline(args, ckpts, weights, dsl, device, clip_sample):
    return build_soup_pipeline(ckpts=ckpts, weights=weights)
