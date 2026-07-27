"""raenwm modeling package.

Typed, hydra-composable pipeline components — tokenizers, denoisers, transport,
dataset, planning, probe, inference — plus the dacite-based config loader.

Design: each component owns a `@dataclass XxxCfg` (with a `name: Literal[...]`
discriminator) next to its implementation and registers a builder in its package
`Registry`. Top-level root configs are composed in ``modeling.config`` and parsed
from an OmegaConf ``DictConfig`` via ``load_typed_root_config``.
"""
