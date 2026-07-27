"""Tokenizer interface.

The pipeline trains a denoiser in the tokenizer's latent space and never assumes a
specific downsampling factor or channel count — those come from the tokenizer itself.
Concrete tokenizers (e.g. the RAE/DINOv2 wrapper) subclass this and delegate
encode/decode to the underlying model.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Tokenizer(nn.Module, ABC):
    # Required attributes (set by concrete tokenizers in __init__):
    latent_dim: int           # number of latent channels
    spatial_compression: int  # pixels per latent cell per side (DINOv2 patch=14; a VAE 8/16)

    @abstractmethod
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels in [0, 1], shape (N, 3, S, S) -> latents (N, latent_dim, s, s)."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> pixels in [0, 1]."""
        raise NotImplementedError

    def latent_spatial_size(self, image_size: int) -> int:
        """Latent grid size per side for a given input resolution."""
        if image_size % self.spatial_compression != 0:
            raise ValueError(
                f"image_size={image_size} not divisible by "
                f"spatial_compression={self.spatial_compression}"
            )
        return image_size // self.spatial_compression
