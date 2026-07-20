"""Convolutional bottleneck block."""

from __future__ import annotations

import torch

from stagerecon.models.blocks.conv_blocks import ActivationType, DoubleConv, NormType
from stagerecon.models.bottlenecks.base_bottleneck import BaseBottleneck


class ConvBottleneck(BaseBottleneck):
    """DoubleConv bottleneck preserving spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dim: int = 2,
        norm: NormType = "batch",
        activation: ActivationType = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize ConvBottleneck.

        Args:
            in_channels: Channels of the deepest encoder feature.
            out_channels: Bottleneck output channels; defaults to ``in_channels``.
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type.
            activation: Activation type.
            num_groups: Groups for GroupNorm when ``norm='group'``.
            dropout: Dropout probability inside DoubleConv.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self._out_channels = out_channels
        self.block = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            norm=norm,
            activation=activation,
            num_groups=num_groups,
            dropout=dropout,
        )

    @property
    def out_channels(self) -> int:
        """Number of bottleneck output channels."""
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional bottleneck.

        Args:
            x: Deepest encoder feature.

        Returns:
            Bottleneck feature with the same spatial size as ``x``.
        """
        return self.block(x)
