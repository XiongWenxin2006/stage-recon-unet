"""Abstract base class for modular U-Net encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Encoder interface producing multi-scale features (high → low resolution)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode an input volume/image into multi-scale feature maps.

        Args:
            x: Input tensor of shape ``(B, C_in, *spatial)``.

        Returns:
            List of feature tensors ordered from highest to lowest resolution.
        """

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Channel counts for each returned feature map (high → low)."""

    @property
    def deepest_channels(self) -> int:
        """Number of channels in the deepest (lowest-resolution) feature."""
        return self.out_channels[-1]
