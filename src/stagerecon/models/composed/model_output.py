"""Structured outputs returned by composed StageRecon models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelOutput:
    """Container for model predictions and optional intermediate features.

    Attributes:
        prediction: Primary task output tensor (reconstruction or segmentation).
        features: Optional dict of intermediate tensors (encoder, bottleneck, etc.).
        mode: Forward mode that produced this output.
    """

    prediction: torch.Tensor
    features: dict[str, Any] | None = None
    mode: str = ""
