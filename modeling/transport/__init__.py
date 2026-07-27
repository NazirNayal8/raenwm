"""Transport / flow-matching objective and ODE sampling.

Wraps ``RAE.src.stage2.transport``. There's a single transport implementation (no
variants), so this is plain config dataclasses + builder functions rather than a
registry/union. RAE imports are deferred to build time to keep importing this module
cheap.
"""
from dataclasses import dataclass
from typing import Literal, Optional
import math


@dataclass(kw_only=True)
class TransportCfg:
    """Flow-matching training objective + path definition."""
    model_type: Literal["velocity", "noise", "score"] = "velocity"
    path_type: Literal["linear", "gvp", "vp"] = "linear"
    loss_type: Literal["velocity", "noise", "score"] = "velocity"
    time_dist_type: str = "uniform"
    # Time-distribution shift. If None, derived from latent geometry as
    # sqrt(latent_dim * latent_size**2 / time_dist_shift_base). Disable forces 1.0.
    time_dist_shift: Optional[float] = None
    time_dist_shift_base: float = 4096.0
    time_dist_shift_disable: bool = False
    train_eps: float = 1e-3
    sample_eps: float = 1e-3


@dataclass(kw_only=True)
class SamplingCfg:
    """ODE sampling parameters used at eval / inference time."""
    sampling_method: str = "euler"
    num_steps: int = 50
    atol: float = 1e-6
    rtol: float = 1e-3
    reverse: bool = False


def resolve_time_dist_shift(cfg: TransportCfg, latent_dim: int, latent_size: int) -> float:
    """Compute the effective time-distribution shift (RAE heuristic + overrides)."""
    if cfg.time_dist_shift_disable:
        return 1.0
    if cfg.time_dist_shift is not None:
        return float(cfg.time_dist_shift)
    shift_dim = latent_dim * latent_size * latent_size
    return math.sqrt(shift_dim / float(cfg.time_dist_shift_base))


def build_transport(cfg: TransportCfg, latent_dim: int, latent_size: int):
    """Construct the RAE ``Transport`` from a typed config + latent geometry."""
    from RAE.src.stage2.transport.transport import (
        Transport, ModelType, PathType, WeightType,
    )
    return Transport(
        model_type=getattr(ModelType, str(cfg.model_type).upper()),
        path_type=getattr(PathType, str(cfg.path_type).upper()),
        loss_type=getattr(WeightType, str(cfg.loss_type).upper()),
        time_dist_type=str(cfg.time_dist_type),
        time_dist_shift=resolve_time_dist_shift(cfg, latent_dim, latent_size),
        train_eps=cfg.train_eps,
        sample_eps=cfg.sample_eps,
    )


def make_transport_sampler(transport):
    """Wrap a Transport in the RAE ODE ``Sampler``."""
    from RAE.src.stage2.transport.transport import Sampler
    return Sampler(transport)


def build_sample_fn(sampler, cfg: SamplingCfg):
    """Build the ODE sampling function from a Sampler + sampling config."""
    return sampler.sample_ode(
        sampling_method=cfg.sampling_method,
        num_steps=cfg.num_steps,
        atol=cfg.atol,
        rtol=cfg.rtol,
        reverse=cfg.reverse,
    )


__all__ = [
    "TransportCfg",
    "SamplingCfg",
    "resolve_time_dist_shift",
    "build_transport",
    "make_transport_sampler",
    "build_sample_fn",
]
