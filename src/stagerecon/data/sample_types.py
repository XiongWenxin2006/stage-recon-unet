"""Typed sample dictionaries for reconstruction and segmentation datasets."""

from __future__ import annotations

from typing import Any, TypedDict

import torch
from torch import Tensor


class ReconstructionSample(TypedDict):
    """A single reconstruction training/eval sample.

    Attributes:
        image: Model input tensor ``(C, H, W)`` — typically a corrupted view.
        target: Clean target tensor ``(C, H, W)``.
        sample_id: Stable identifier for the sample.
        metadata: Free-form per-sample metadata (paths, corruption info, etc.).
    """

    image: Tensor
    target: Tensor
    sample_id: str
    metadata: dict[str, Any]


class SegmentationSample(TypedDict):
    """A single segmentation training/eval sample.

    Attributes:
        image: Input image tensor ``(C, H, W)``.
        mask: Binary or multi-class mask tensor ``(1, H, W)`` or ``(H, W)``.
        sample_id: Stable identifier for the sample.
        metadata: Free-form per-sample metadata (paths, class ids, etc.).
    """

    image: Tensor
    mask: Tensor
    sample_id: str
    metadata: dict[str, Any]


def empty_metadata() -> dict[str, Any]:
    """Return a fresh empty metadata dictionary."""
    return {}


def as_float_image(tensor: Tensor) -> Tensor:
    """Cast ``tensor`` to ``float32`` without changing values."""
    return tensor.to(dtype=torch.float32)
