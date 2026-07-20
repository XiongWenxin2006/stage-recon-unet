"""Head that reconstructs an image directly from bottleneck features."""

from __future__ import annotations

import torch
import torch.nn as nn

from stagerecon.models.blocks.conv_blocks import (
    ActivationType,
    DoubleConv,
    NormType,
    get_conv_nd,
    get_upsample_nd,
)


class BottleneckReconstructionHead(nn.Module):
    """Sequential upsample + conv path from bottleneck to image space.

    Performs ``num_upsamples`` nearest-neighbor 2× upsamples interleaved with
    DoubleConv blocks, then a final 1×1 projection to ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_upsamples: int = 3,
        channels: list[int] | None = None,
        dim: int = 2,
        norm: NormType = "batch",
        activation: ActivationType = "relu",
        num_groups: int = 8,
        apply_sigmoid: bool = False,
    ) -> None:
        """Initialize BottleneckReconstructionHead.

        Args:
            in_channels: Bottleneck feature channels.
            out_channels: Number of reconstructed image channels.
            num_upsamples: Number of 2× upsampling stages.
            channels: Optional per-stage channel widths after each upsample.
                Defaults to halving from ``in_channels`` down to at least 16.
            dim: Spatial dimensions (2 or 3).
            norm: Normalization type.
            activation: Activation type.
            num_groups: Groups for GroupNorm when ``norm='group'``.
            apply_sigmoid: If True, apply sigmoid to the final output.
        """
        super().__init__()
        if num_upsamples < 0:
            raise ValueError("num_upsamples must be >= 0.")

        if channels is None:
            channels = []
            ch = in_channels
            for _ in range(num_upsamples):
                ch = max(ch // 2, 16)
                channels.append(ch)
        if len(channels) != num_upsamples:
            raise ValueError(
                f"channels length ({len(channels)}) must equal "
                f"num_upsamples ({num_upsamples})."
            )

        stages: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            stages.append(
                nn.Sequential(
                    get_upsample_nd(dim, scale_factor=2),
                    DoubleConv(
                        in_channels=prev,
                        out_channels=ch,
                        dim=dim,
                        norm=norm,
                        activation=activation,
                        num_groups=num_groups,
                    ),
                )
            )
            prev = ch

        self.upsamples = nn.Sequential(*stages) if stages else nn.Identity()
        conv_cls = get_conv_nd(dim)
        self.proj = conv_cls(prev, out_channels, kernel_size=1)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from bottleneck features.

        Args:
            bottleneck: Bottleneck tensor ``(B, C, *spatial_low)``.

        Returns:
            Reconstructed image ``(B, out_channels, *spatial_high)``.
            Returns logits/raw values unless ``apply_sigmoid=True``.
        """
        x = self.upsamples(bottleneck)
        x = self.proj(x)
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        return x
