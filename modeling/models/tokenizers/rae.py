"""RAE (DINOv2-based) tokenizer.

Wraps the vendored stage-1 RAE behind the ``Tokenizer`` contract. Per the agreed
design the RAE keeps its own nested config: ``RAECfg`` only holds the path to the
RAE yaml (parsed by RAE's ``parse_configs``) plus the latent geometry the pipeline
needs (``spatial_compression`` — DINOv2 patch size, 14).

RAE-specific imports are deferred to construction time so importing this module (and
thus ``modeling.models``) stays cheap and side-effect free.
"""
from dataclasses import dataclass
from typing import Literal

import torch

from modeling.models.tokenizers.base import Tokenizer
from modeling.models.tokenizers import TOKENIZERS


class RAETokenizer(Tokenizer):
    """Adapts the vendored RAE to the Tokenizer interface (encode/decode + geometry)."""

    def __init__(self, config_path: str, spatial_compression: int = 14):
        super().__init__()
        from RAE.src.utils.train_utils import parse_configs
        from RAE.src.utils.model_utils import instantiate_from_config

        rae_config, *_ = parse_configs(config_path)
        self.rae = instantiate_from_config(rae_config)
        self.latent_dim = int(self.rae.latent_dim)
        self.spatial_compression = int(spatial_compression)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.rae.encode(pixels)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.rae.decode(latents)


@dataclass(kw_only=True)
class RAECfg:
    name: Literal["RAE"]
    config_path: str = "RAE/configs/stage1/pretrained/DINOv2-B.yaml"
    spatial_compression: int = 14  # DINOv2 patch size; override for other encoders


@TOKENIZERS.register(RAECfg)
def build_rae(cfg: RAECfg) -> RAETokenizer:
    return RAETokenizer(config_path=cfg.config_path, spatial_compression=cfg.spatial_compression)
