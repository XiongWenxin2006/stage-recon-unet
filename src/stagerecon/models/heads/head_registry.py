"""Registry for prediction / reconstruction head constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from stagerecon.models.heads.bottleneck_reconstruction_head import (
    BottleneckReconstructionHead,
)
from stagerecon.models.heads.image_reconstruction_head import ImageReconstructionHead
from stagerecon.models.heads.segmentation_head import SegmentationHead

HeadFactory = Callable[..., nn.Module]

_HEAD_REGISTRY: dict[str, HeadFactory] = {}


def register_head(
    name: str,
    factory: HeadFactory | None = None,
) -> Callable[[HeadFactory], HeadFactory] | HeadFactory:
    """Register a head factory under ``name``."""

    def decorator(cls_or_fn: HeadFactory) -> HeadFactory:
        key = name.lower()
        if key in _HEAD_REGISTRY:
            raise ValueError(f"Head '{name}' is already registered.")
        _HEAD_REGISTRY[key] = cls_or_fn
        return cls_or_fn

    if factory is not None:
        return decorator(factory)
    return decorator


def get_head(name: str) -> HeadFactory:
    """Look up a registered head factory by name."""
    key = name.lower()
    if key not in _HEAD_REGISTRY:
        available = ", ".join(sorted(_HEAD_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown head '{name}'. Available: {available}")
    return _HEAD_REGISTRY[key]


def build_head(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered head."""
    return get_head(name)(**kwargs)


def list_heads() -> list[str]:
    """Return sorted registered head names."""
    return sorted(_HEAD_REGISTRY.keys())


register_head("bottleneck_reconstruction", BottleneckReconstructionHead)
register_head("image_reconstruction", ImageReconstructionHead)
register_head("reconstruction", ImageReconstructionHead)
register_head("segmentation", SegmentationHead)
