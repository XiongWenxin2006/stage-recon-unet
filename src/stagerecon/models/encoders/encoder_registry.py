"""Registry for encoder constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from stagerecon.models.encoders.residual_encoder import ResidualEncoder
from stagerecon.models.encoders.unet_encoder import UNetEncoder

EncoderFactory = Callable[..., nn.Module]

_ENCODER_REGISTRY: dict[str, EncoderFactory] = {}


def register_encoder(name: str, factory: EncoderFactory | None = None) -> Callable[[EncoderFactory], EncoderFactory] | EncoderFactory:
    """Register an encoder factory under ``name``.

    Can be used as ``@register_encoder('name')`` or ``register_encoder('name', cls)``.
    """

    def decorator(cls_or_fn: EncoderFactory) -> EncoderFactory:
        key = name.lower()
        if key in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered.")
        _ENCODER_REGISTRY[key] = cls_or_fn
        return cls_or_fn

    if factory is not None:
        return decorator(factory)
    return decorator


def get_encoder(name: str) -> EncoderFactory:
    """Look up a registered encoder factory by name."""
    key = name.lower()
    if key not in _ENCODER_REGISTRY:
        available = ", ".join(sorted(_ENCODER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown encoder '{name}'. Available: {available}")
    return _ENCODER_REGISTRY[key]


def build_encoder(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered encoder.

    Args:
        name: Registry key (e.g. ``'unet'``, ``'residual'``).
        **kwargs: Constructor arguments for the encoder.

    Returns:
        An initialized encoder module.
    """
    return get_encoder(name)(**kwargs)


def list_encoders() -> list[str]:
    """Return sorted registered encoder names."""
    return sorted(_ENCODER_REGISTRY.keys())


# Default registrations
register_encoder("unet", UNetEncoder)
register_encoder("residual", ResidualEncoder)
