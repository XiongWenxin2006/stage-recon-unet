"""Attention gates for Attention U-Net skip connections."""

from __future__ import annotations

import torch
import torch.nn as nn

from stagerecon.models.blocks.conv_blocks import NormType, get_conv_nd, get_norm_nd


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections (Attention U-Net).

    Computes attention coefficients from a gating signal (coarser / decoder
    feature) and a skip feature (encoder), then scales the skip feature.
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int | None = None,
        dim: int = 2,
        norm: NormType = "batch",
        num_groups: int = 8,
    ) -> None:
        """Initialize AttentionGate.

        Args:
            gate_channels: Channels of the gating (decoder / upsampled) signal.
            skip_channels: Channels of the encoder skip feature.
            inter_channels: Intermediate attention channels; defaults to
                ``skip_channels // 2`` (at least 1).
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type for 1x1 projections.
            num_groups: Groups for GroupNorm when ``norm='group'``.
        """
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        conv_cls = get_conv_nd(dim)
        self.W_g = nn.Sequential(
            conv_cls(gate_channels, inter_channels, kernel_size=1, bias=True),
            get_norm_nd(dim, inter_channels, norm=norm, num_groups=num_groups),
        )
        self.W_x = nn.Sequential(
            conv_cls(skip_channels, inter_channels, kernel_size=1, bias=True),
            get_norm_nd(dim, inter_channels, norm=norm, num_groups=num_groups),
        )
        self.psi = nn.Sequential(
            conv_cls(inter_channels, 1, kernel_size=1, bias=True),
            get_norm_nd(dim, 1, norm=norm, num_groups=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply additive attention to a skip feature.

        Args:
            gate: Gating signal of shape ``(B, C_g, *spatial_g)``.
            skip: Skip feature of shape ``(B, C_s, *spatial_s)``.

        Returns:
            Attention-weighted skip feature with the same shape as ``skip``.
            If spatial sizes differ, ``gate`` is interpolated to match ``skip``.
        """
        if gate.shape[2:] != skip.shape[2:]:
            gate = nn.functional.interpolate(
                gate,
                size=skip.shape[2:],
                mode="nearest",
            )
        attention = self.psi(self.relu(self.W_g(gate) + self.W_x(skip)))
        return skip * attention
