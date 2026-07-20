"""Shared fixtures for StageRecon CPU-only unit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

from stagerecon.models import ModularUNet, build_model


def modules_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    """Return True if two modules have identical state_dict tensors."""
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture
def device() -> torch.device:
    """Force CPU for all tests (no GPU / cloud credentials required)."""
    return torch.device("cpu")


@pytest.fixture
def tiny_model_cfg() -> dict[str, Any]:
    """Minimal ModularUNet config: channels [8, 16], 2D, all heads enabled.

    Uses instance norm so batch-size-1 forward/backward stays stable on CPU.
    """
    return {
        "name": "unet",
        "in_channels": 1,
        "out_channels": 1,
        "num_classes": 1,
        "spatial_dims": 2,
        "norm": "instance",
        "activation": "relu",
        "return_features": False,
        "encoder": {"name": "unet", "channels": [8, 16]},
        "bottleneck": {"name": "conv"},
        "decoder": {"name": "unet"},
        "heads": {
            "bottleneck_reconstruction": {"name": "bottleneck_reconstruction"},
            "reconstruction": {"name": "image_reconstruction"},
            "segmentation": {"name": "segmentation"},
        },
    }


@pytest.fixture
def tiny_model(tiny_model_cfg: dict[str, Any], device: torch.device) -> ModularUNet:
    """Build a tiny ModularUNet on CPU."""
    model = build_model(deepcopy(tiny_model_cfg))
    return model.to(device)


@pytest.fixture
def image_size() -> int:
    return 32


@pytest.fixture
def sample_input(image_size: int, device: torch.device) -> torch.Tensor:
    """Tiny batched image tensor ``(B=2, C=1, H, W)`` on CPU."""
    return torch.rand(2, 1, image_size, image_size, device=device)


@pytest.fixture
def ckpt_dir(tmp_path):
    """Temporary checkpoint directory under pytest's ``tmp_path``."""
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
