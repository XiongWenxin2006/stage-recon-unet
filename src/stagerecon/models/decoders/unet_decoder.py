"""Standard U-Net expanding path with skip concatenations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stagerecon.models.blocks.conv_blocks import (
    ActivationType,
    DoubleConv,
    NormType,
    get_upsample_nd,
)
from stagerecon.models.decoders.base_decoder import BaseDecoder


class UNetDecoder(BaseDecoder):
    """Upsample + skip-concat + DoubleConv at each level.

    ``skip_features`` are expected high → low resolution (same order as the
    encoder). The deepest encoder feature is typically already consumed by the
    bottleneck, so skips used are ``skip_features[:-1]`` from low → high during
    upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: list[int],
        dim: int = 2,
        norm: NormType = "batch",
        activation: ActivationType = "relu",
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize UNetDecoder.

        Args:
            in_channels: Channels of the bottleneck feature.
            skip_channels: Encoder channel widths high → low (including deepest).
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type.
            activation: Activation type.
            num_groups: Groups for GroupNorm when ``norm='group'``.
            dropout: Dropout probability inside DoubleConv blocks.
        """
        super().__init__()
        if len(skip_channels) < 1:
            raise ValueError("skip_channels must contain at least one level.")

        self.dim = dim
        self._skip_channels = list(skip_channels)
        # Skips used during upsampling: all but deepest, processed low → high
        used_skips = list(reversed(self._skip_channels[:-1]))
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()

        prev_ch = in_channels
        for skip_ch in used_skips:
            self.ups.append(get_upsample_nd(dim, scale_factor=2))
            self.convs.append(
                DoubleConv(
                    in_channels=prev_ch + skip_ch,
                    out_channels=skip_ch,
                    dim=dim,
                    norm=norm,
                    activation=activation,
                    num_groups=num_groups,
                    dropout=dropout,
                )
            )
            prev_ch = skip_ch

        self._out_channels = prev_ch if used_skips else in_channels

    @property
    def out_channels(self) -> int:
        """Channels of the highest-resolution decoded feature."""
        return self._out_channels

    def decode(
        self,
        bottleneck: torch.Tensor,
        skip_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Upsample the bottleneck with encoder skip connections.

        Args:
            bottleneck: Bottleneck feature at deepest resolution.
            skip_features: Encoder features high → low.

        Returns:
            Decoded feature at input spatial resolution.
        """
        if len(skip_features) != len(self._skip_channels):
            raise ValueError(
                f"Expected {len(self._skip_channels)} skip features, "
                f"got {len(skip_features)}."
            )

        skips = list(reversed(skip_features[:-1]))
        x = bottleneck
        for up, conv, skip in zip(self.ups, self.convs, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return x
