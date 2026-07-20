"""Standard U-Net encoder with DoubleConv blocks and max-pooling."""

from __future__ import annotations

import torch
import torch.nn as nn

from stagerecon.models.blocks.conv_blocks import (
    ActivationType,
    DoubleConv,
    NormType,
    get_pool_nd,
)
from stagerecon.models.encoders.base_encoder import BaseEncoder


class UNetEncoder(BaseEncoder):
    """Contracting path: DoubleConv at each level, MaxPool between levels.

    For ``channels=[32, 64, 128, 256]`` the encoder returns four feature maps
    at resolutions ``H, H/2, H/4, H/8`` (for 2D) with those channel counts.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | None = None,
        dim: int = 2,
        norm: NormType = "batch",
        activation: ActivationType = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize UNetEncoder.

        Args:
            in_channels: Number of input image channels.
            channels: Per-level feature channel widths (high → low).
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type for DoubleConv blocks.
            activation: Activation type for DoubleConv blocks.
            num_groups: Groups for GroupNorm when ``norm='group'``.
            dropout: Dropout probability inside DoubleConv blocks.
        """
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        if len(channels) < 1:
            raise ValueError("channels must contain at least one level.")

        self._channels = list(channels)
        self.dim = dim

        blocks: list[nn.Module] = []
        pools: list[nn.Module] = []
        prev = in_channels
        for i, ch in enumerate(self._channels):
            blocks.append(
                DoubleConv(
                    in_channels=prev,
                    out_channels=ch,
                    dim=dim,
                    norm=norm,
                    activation=activation,
                    num_groups=num_groups,
                    dropout=dropout,
                )
            )
            if i < len(self._channels) - 1:
                pools.append(get_pool_nd(dim))
            prev = ch

        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)

    @property
    def out_channels(self) -> list[int]:
        """Channel counts for each encoder feature map."""
        return list(self._channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode ``x`` into multi-scale features (high → low resolution).

        Args:
            x: Input tensor of shape ``(B, C_in, *spatial)``.

        Returns:
            List of feature tensors, one per level.
        """
        features: list[torch.Tensor] = []
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            features.append(h)
            if i < len(self.pools):
                h = self.pools[i](h)
        return features
