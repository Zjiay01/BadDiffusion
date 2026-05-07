from .anp import build_pipeline as build_anp_pipeline
from .clean_finetune import build_pipeline as build_clean_finetune_pipeline
from .diffusion_soup import build_pipeline as build_diffusion_soup_pipeline
from .dmm import build_pipeline as build_dmm_pipeline
from .maxfusion import build_pipeline as build_maxfusion_pipeline


BUILDERS = {
    "diffusion_soup": build_diffusion_soup_pipeline,
    "dmm": build_dmm_pipeline,
    "maxfusion": build_maxfusion_pipeline,
    "anp": build_anp_pipeline,
    "clean_finetune": build_clean_finetune_pipeline,
}
