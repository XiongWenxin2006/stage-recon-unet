"""Helpers for accessing named ModularUNet submodules."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

KNOWN_MODULES: tuple[str, ...] = (
    "encoder",
    "bottleneck",
    "decoder",
    "reconstruction_head",
    "bottleneck_reconstruction_head",
    "segmentation_head",
)


def get_model_module(model: nn.Module, name: str) -> nn.Module:
    """Return a named submodule from a ModularUNet-compatible model.

    Prefers ``model.get_module(name)`` when available, otherwise falls back to
    ``getattr(model, name)``.
    """
    if hasattr(model, "get_module"):
        return model.get_module(name)  # type: ignore[no-any-return]
    module = getattr(model, name, None)
    if module is None:
        raise KeyError(f"Module '{name}' is not set on model {type(model).__name__}.")
    if not isinstance(module, nn.Module):
        raise TypeError(f"Attribute '{name}' is not an nn.Module: {type(module)!r}")
    return module


def iter_known_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for known modules that are present on ``model``."""
    for name in KNOWN_MODULES:
        try:
            module = get_model_module(model, name)
        except (KeyError, AttributeError, TypeError):
            continue
        yield name, module
