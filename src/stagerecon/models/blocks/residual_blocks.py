"""Residual convolutional building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from stagerecon.models.blocks.conv_blocks import (
    ActivationType,
    DoubleConv,
    NormType,
    get_activation,
    get_conv_nd,
    get_norm_nd,
)


class ResidualDoubleConv(nn.Module):
    """DoubleConv with a residual skip connection.

    If ``in_channels != out_channels``, a 1x1 projection aligns the skip path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dim: int = 2,
        norm: NormType = "batch",
        activation: ActivationType = "relu",
        kernel_size: int = 3,
        padding: int | None = None,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize ResidualDoubleConv.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mid_channels: Intermediate channels; defaults to ``out_channels``.
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type.
            activation: Activation applied after residual addition.
            kernel_size: Convolution kernel size for the main path.
            padding: Convolution padding; defaults to ``kernel_size // 2``.
            num_groups: Groups for GroupNorm when ``norm='group'``.
            dropout: Optional dropout inside the main DoubleConv path.
        """
        super().__init__()
        self.double_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            dim=dim,
            norm=norm,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups,
            dropout=dropout,
        )
        if in_channels != out_channels:
            conv_cls = get_conv_nd(dim)
            self.skip = nn.Sequential(
                conv_cls(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_nd(dim, out_channels, norm=norm, num_groups=num_groups),
            )
        else:
            self.skip = nn.Identity()
        self.out_act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual double convolution.

        Args:
            x: Input tensor of shape ``(B, C_in, *spatial)``.

        Returns:
            Output tensor of shape ``(B, C_out, *spatial)``.
        """
        return self.out_act(self.double_conv(x) + self.skip(x))
