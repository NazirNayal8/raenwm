"""Tokenizer components.

Registry + runtime union + factory. Adding a tokenizer = new module here that defines
a ``@dataclass XxxCfg`` (with ``name: Literal[...]``) and a builder decorated with
``@TOKENIZERS.register(XxxCfg)``, then imported below.
"""
from modeling.registry import Registry
from modeling.models.tokenizers.base import Tokenizer

# Registry must exist before variant modules import it.
TOKENIZERS = Registry("tokenizer")

from modeling.models.tokenizers import rae as _rae  # noqa: E402,F401  (registers RAECfg)
from modeling.models.tokenizers.rae import RAETokenizer, RAECfg  # noqa: E402,F401

# Runtime Union[...] of all registered tokenizer configs, for dacite parsing.
TokenizerCfg = TOKENIZERS.union()
get_tokenizer = TOKENIZERS.build

__all__ = [
    "Tokenizer",
    "TOKENIZERS",
    "TokenizerCfg",
    "get_tokenizer",
    "RAETokenizer",
    "RAECfg",
]
