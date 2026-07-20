"""Abstract base class for modular U-Net decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """Decoder interface that expands bottleneck features with skip connections."""

    @abstractmethod
    def decode(
        self,
        bottleneck: torch.Tensor,
        skip_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Decode bottleneck features to input resolution.

        Args:
            bottleneck: Bottleneck feature map.
            skip_features: Encoder features ordered high → low resolution.

        Returns:
            Decoded feature map at the highest (input) spatial resolution.
        """

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Alias for :meth:`decode`."""
        return self.decode(bottleneck, skip_features)

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of channels in the decoded feature map."""
