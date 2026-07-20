"""Convolutional building blocks and N-D helpers for 2D/3D U-Net modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

NormType = Literal["batch", "instance", "group", "none"]
ActivationType = Literal["relu", "leaky_relu", "gelu", "silu"]


def get_conv_nd(dim: int) -> type[nn.Module]:
    """Return the Conv module class for the given spatial dimensionality.

    Args:
        dim: Spatial dimensions (2 or 3).

    Returns:
        ``nn.Conv2d`` or ``nn.Conv3d``.
    """
    if dim == 2:
        return nn.Conv2d
    if dim == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported dim={dim}; expected 2 or 3.")


def get_norm_nd(
    dim: int,
    num_channels: int,
    norm: NormType = "batch",
    num_groups: int = 8,
) -> nn.Module:
    """Build a normalization layer for N-D feature maps.

    Args:
        dim: Spatial dimensions (2 or 3).
        num_channels: Number of channels to normalize.
        norm: One of ``batch``, ``instance``, ``group``, ``none``.
        num_groups: Group count for GroupNorm (adjusted to divide channels).

    Returns:
        A normalization module, or ``nn.Identity`` when ``norm='none'``.
    """
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(num_channels) if dim == 2 else nn.BatchNorm3d(num_channels)
    if norm == "instance":
        if dim == 2:
            return nn.InstanceNorm2d(num_channels, affine=True)
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm == "group":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm='{norm}'.")


def get_pool_nd(dim: int, kernel_size: int = 2, stride: int = 2) -> nn.Module:
    """Return a max-pooling layer for the given spatial dimensionality.

    Args:
        dim: Spatial dimensions (2 or 3).
        kernel_size: Pooling kernel size.
        stride: Pooling stride.

    Returns:
        ``nn.MaxPool2d`` or ``nn.MaxPool3d``.
    """
    if dim == 2:
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    if dim == 3:
        return nn.MaxPool3d(kernel_size=kernel_size, stride=stride)
    raise ValueError(f"Unsupported dim={dim}; expected 2 or 3.")


def get_upsample_nd(
    dim: int,
    scale_factor: int = 2,
    mode: str | None = None,
) -> nn.Module:
    """Return an upsampling layer for the given spatial dimensionality.

    Args:
        dim: Spatial dimensions (2 or 3).
        scale_factor: Upsampling factor.
        mode: Interpolation mode; defaults to nearest for 2D/3D.

    Returns:
        An ``nn.Upsample`` module.
    """
    if mode is None:
        mode = "nearest"
    return nn.Upsample(scale_factor=scale_factor, mode=mode)


def get_activation(activation: ActivationType) -> nn.Module:
    """Build a non-linearity module.

    Args:
        activation: One of ``relu``, ``leaky_relu``, ``gelu``, ``silu``.

    Returns:
        The corresponding activation module.
    """
    mapping: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if activation not in mapping:
        raise ValueError(f"Unsupported activation='{activation}'.")
    # ReLU uses inplace for memory efficiency
    if activation == "relu":
        return nn.ReLU(inplace=True)
    return mapping[activation]()


class DoubleConv(nn.Module):
    """Two consecutive Conv-Norm-Act blocks (2D or 3D).

    Structure: ``Conv -> Norm -> Act -> Conv -> Norm -> Act``.
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
        """Initialize DoubleConv.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mid_channels: Intermediate channels; defaults to ``out_channels``.
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type.
            activation: Activation type.
            kernel_size: Convolution kernel size.
            padding: Convolution padding; defaults to ``kernel_size // 2``.
            num_groups: Groups for GroupNorm when ``norm='group'``.
            dropout: Optional dropout probability between the two conv blocks.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if padding is None:
            padding = kernel_size // 2

        conv_cls = get_conv_nd(dim)
        self.block = nn.Sequential(
            conv_cls(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            get_norm_nd(dim, mid_channels, norm=norm, num_groups=num_groups),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            conv_cls(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            get_norm_nd(dim, out_channels, norm=norm, num_groups=num_groups),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block.

        Args:
            x: Input tensor of shape ``(B, C_in, *spatial)``.

        Returns:
            Output tensor of shape ``(B, C_out, *spatial)``.
        """
        return self.block(x)
