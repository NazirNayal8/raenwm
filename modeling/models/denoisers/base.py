"""Denoiser interface.

Any denoiser plugged into the pipeline must satisfy this contract so the training
loop, transport objective, flash-attention probe, and eval sampler stay
component-agnostic. Concrete denoisers (e.g. CDiT) subclass this and set the required
attributes in ``__init__``.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Denoiser(nn.Module, ABC):
    # Required attributes (set by concrete denoisers in __init__):
    in_channels: int
    out_channels: int
    patch_size: int
    context_size: int
    head_width: int

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_cond: torch.Tensor,
        rel_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the transport target (e.g. velocity) at noised latent ``x``, time ``t``.

        Invoked by the transport objective as ``model(x, t, y=..., x_cond=..., rel_t=...)``.
        - x:      (N, in_channels, H, W) noised target latent
        - t:      (N,) flow-matching timestep
        - y:      (N, ...) action / conditioning
        - x_cond: (N, context_size, in_channels, H, W) context latents
        - rel_t:  (N, ...) relative time between context and target
        """
        raise NotImplementedError

    def attention_shape(self) -> tuple[int, int, int]:
        """``(num_heads, head_dim, seqlen)`` for the flash-attention diagnostic probe.

        Concrete denoisers should override with real values so the probe reports the
        actual attention shape; the base raises so callers can fall back to defaults.
        """
        raise NotImplementedError

    def set_return_intermediate_features(self, flag: bool) -> None:
        """Toggle returning intermediate features (disabled during ODE sampling)."""
        self.return_intermediate_features = bool(flag)
