"""Denoiser components.

Registry + runtime union + factory. Adding a denoiser family = new module here that
defines a ``@dataclass XxxCfg`` (with ``name: Literal[...]``) and a builder decorated
with ``@DENOISERS.register(XxxCfg)``, then imported below.
"""
from modeling.registry import Registry
from modeling.models.denoisers.base import Denoiser

# Registry must exist before variant modules import it (they do `from
# modeling.models.denoisers import DENOISERS`).
DENOISERS = Registry("denoiser")

from modeling.models.denoisers import cdit as _cdit  # noqa: E402,F401  (registers CDiTCfg)
from modeling.models.denoisers.cdit import CDiT, CDiTCfg, CDiT_models  # noqa: E402,F401

# Runtime Union[...] of all registered denoiser configs, for dacite parsing.
DenoiserCfg = DENOISERS.union()
get_denoiser = DENOISERS.build

__all__ = [
    "Denoiser",
    "DENOISERS",
    "DenoiserCfg",
    "get_denoiser",
    "CDiT",
    "CDiTCfg",
    "CDiT_models",
]
