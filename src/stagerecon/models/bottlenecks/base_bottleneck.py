"""Abstract base class for U-Net bottleneck modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBottleneck(nn.Module, ABC):
    """Bottleneck that maps deepest encoder features to same-spatial features."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the deepest encoder feature.

        Args:
            x: Deepest encoder feature ``(B, C_in, *spatial)``.

        Returns:
            Bottleneck feature ``(B, C_out, *spatial)`` (same spatial size).
        """

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of output channels."""
