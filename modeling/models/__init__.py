"""Model components: tokenizers (encode/decode to/from latent space) and denoisers
(the flow-matching backbone), plus the wiring that adapts a denoiser config to
whatever the tokenizer produces.
"""
from modeling.models.denoisers import (
    DENOISERS,
    Denoiser,
    DenoiserCfg,
    get_denoiser,
    CDiT,
    CDiTCfg,
    CDiT_models,
)
from modeling.models.tokenizers import (
    TOKENIZERS,
    Tokenizer,
    TokenizerCfg,
    get_tokenizer,
    RAETokenizer,
    RAECfg,
)


def wire_denoiser_to_tokenizer(denoiser_cfg, tokenizer, data_cfg):
    """Fill the denoiser fields that depend on the tokenizer + input resolution.

    Generic across any (tokenizer, denoiser) pair: channel count and latent grid size
    come from the tokenizer, context length from the data config. Mutates and returns
    ``denoiser_cfg``.
    """
    denoiser_cfg.in_channels = tokenizer.latent_dim
    denoiser_cfg.input_size = tokenizer.latent_spatial_size(data_cfg.image_size)
    denoiser_cfg.context_size = data_cfg.context_size
    return denoiser_cfg


__all__ = [
    "DENOISERS",
    "Denoiser",
    "DenoiserCfg",
    "get_denoiser",
    "CDiT",
    "CDiTCfg",
    "CDiT_models",
    "TOKENIZERS",
    "Tokenizer",
    "TokenizerCfg",
    "get_tokenizer",
    "RAETokenizer",
    "RAECfg",
    "wire_denoiser_to_tokenizer",
]
