"""Head that maps decoded features to a reconstructed image."""

from __future__ import annotations

import torch
import torch.nn as nn

from stagerecon.models.blocks.conv_blocks import get_conv_nd


class ImageReconstructionHead(nn.Module):
    """1×1 projection from decoded features to image channels.

    By default returns raw logits (no sigmoid). Set ``apply_sigmoid=True`` to
    squash outputs to ``[0, 1]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dim: int = 2,
        apply_sigmoid: bool = False,
    ) -> None:
        """Initialize ImageReconstructionHead.

        Args:
            in_channels: Channels of the decoded feature map.
            out_channels: Number of reconstructed image channels.
            dim: Spatial dimensions (2 or 3).
            apply_sigmoid: If True, apply sigmoid to the final output.
        """
        super().__init__()
        conv_cls = get_conv_nd(dim)
        self.proj = conv_cls(in_channels, out_channels, kernel_size=1)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project decoded features to image space.

        Args:
            features: Decoded feature tensor ``(B, C, *spatial)``.

        Returns:
            Reconstructed image ``(B, out_channels, *spatial)``.
        """
        x = self.proj(features)
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        return x
