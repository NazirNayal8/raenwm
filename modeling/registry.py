"""Generic component registry.

Each component group (denoisers, tokenizers, ...) owns one ``Registry``. A concrete
variant registers its builder against its config dataclass:

    @DENOISERS.register(CDiTCfg)
    def build_cdit(cfg: CDiTCfg) -> Denoiser:
        ...

The package ``__init__`` imports the variant modules (triggering registration), then
exposes:

    DenoiserCfg = DENOISERS.union()   # runtime Union[CDiTCfg, ...] for dacite
    get_denoiser = DENOISERS.build    # dispatch on type(cfg)

Dispatch is by the concrete config *type* (dacite already resolved the correct class
via each cfg's ``name: Literal[...]`` discriminator), so no string lookup is needed.
Adding a variant is one new module + one yaml — no edits to callers or to this file.
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar, Union

T = TypeVar("T")
Builder = Callable[[Any], Any]


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._builders: dict[type, Builder] = {}

    def register(self, cfg_cls: type) -> Callable[[Builder], Builder]:
        """Decorator: bind a builder function to a config dataclass."""
        def deco(builder: Builder) -> Builder:
            if cfg_cls in self._builders:
                raise ValueError(f"{self.kind}: cfg {cfg_cls.__name__} already registered")
            self._builders[cfg_cls] = builder
            return builder
        return deco

    def build(self, cfg: Any) -> Any:
        """Instantiate the component for a concrete config instance."""
        cls = type(cfg)
        try:
            builder = self._builders[cls]
        except KeyError:
            known = ", ".join(c.__name__ for c in self._builders) or "<none>"
            raise KeyError(
                f"No {self.kind} registered for cfg type {cls.__name__}. Known: {known}"
            ) from None
        return builder(cfg)

    def cfg_classes(self) -> tuple[type, ...]:
        return tuple(self._builders.keys())

    def union(self):
        """Runtime ``Union`` of all registered cfg classes, for dacite parsing.

        Call only after the variant modules have been imported (i.e. from the
        package ``__init__`` once registrations have happened).
        """
        classes = self.cfg_classes()
        if not classes:
            raise RuntimeError(
                f"No {self.kind} cfg classes registered; import the variant modules first"
            )
        return Union[classes]  # Union[(A, B)] == Union[A, B]; Union[(A,)] == A
