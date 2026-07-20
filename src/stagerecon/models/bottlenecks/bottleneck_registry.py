"""Registry for bottleneck constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from stagerecon.models.bottlenecks.conv_bottleneck import ConvBottleneck
from stagerecon.models.bottlenecks.residual_bottleneck import ResidualBottleneck

BottleneckFactory = Callable[..., nn.Module]

_BOTTLENECK_REGISTRY: dict[str, BottleneckFactory] = {}


def register_bottleneck(
    name: str,
    factory: BottleneckFactory | None = None,
) -> Callable[[BottleneckFactory], BottleneckFactory] | BottleneckFactory:
    """Register a bottleneck factory under ``name``."""

    def decorator(cls_or_fn: BottleneckFactory) -> BottleneckFactory:
        key = name.lower()
        if key in _BOTTLENECK_REGISTRY:
            raise ValueError(f"Bottleneck '{name}' is already registered.")
        _BOTTLENECK_REGISTRY[key] = cls_or_fn
        return cls_or_fn

    if factory is not None:
        return decorator(factory)
    return decorator


def get_bottleneck(name: str) -> BottleneckFactory:
    """Look up a registered bottleneck factory by name."""
    key = name.lower()
    if key not in _BOTTLENECK_REGISTRY:
        available = ", ".join(sorted(_BOTTLENECK_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown bottleneck '{name}'. Available: {available}")
    return _BOTTLENECK_REGISTRY[key]


def build_bottleneck(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered bottleneck."""
    return get_bottleneck(name)(**kwargs)


def list_bottlenecks() -> list[str]:
    """Return sorted registered bottleneck names."""
    return sorted(_BOTTLENECK_REGISTRY.keys())


register_bottleneck("conv", ConvBottleneck)
register_bottleneck("residual", ResidualBottleneck)
