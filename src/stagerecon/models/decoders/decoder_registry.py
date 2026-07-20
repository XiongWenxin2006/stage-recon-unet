"""Registry for decoder constructors."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from stagerecon.models.decoders.attention_decoder import AttentionDecoder
from stagerecon.models.decoders.residual_decoder import ResidualDecoder
from stagerecon.models.decoders.unet_decoder import UNetDecoder

DecoderFactory = Callable[..., nn.Module]

_DECODER_REGISTRY: dict[str, DecoderFactory] = {}


def register_decoder(
    name: str,
    factory: DecoderFactory | None = None,
) -> Callable[[DecoderFactory], DecoderFactory] | DecoderFactory:
    """Register a decoder factory under ``name``."""

    def decorator(cls_or_fn: DecoderFactory) -> DecoderFactory:
        key = name.lower()
        if key in _DECODER_REGISTRY:
            raise ValueError(f"Decoder '{name}' is already registered.")
        _DECODER_REGISTRY[key] = cls_or_fn
        return cls_or_fn

    if factory is not None:
        return decorator(factory)
    return decorator


def get_decoder(name: str) -> DecoderFactory:
    """Look up a registered decoder factory by name."""
    key = name.lower()
    if key not in _DECODER_REGISTRY:
        available = ", ".join(sorted(_DECODER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown decoder '{name}'. Available: {available}")
    return _DECODER_REGISTRY[key]


def build_decoder(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered decoder."""
    return get_decoder(name)(**kwargs)


def list_decoders() -> list[str]:
    """Return sorted registered decoder names."""
    return sorted(_DECODER_REGISTRY.keys())


register_decoder("unet", UNetDecoder)
register_decoder("attention", AttentionDecoder)
register_decoder("residual", ResidualDecoder)
