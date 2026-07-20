"""Head that maps decoded features to segmentation logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from stagerecon.models.blocks.conv_blocks import get_conv_nd


class SegmentationHead(nn.Module):
    """1×1 projection to ``num_classes`` segmentation channels.

    Returns raw logits with **no** thresholding or sigmoid/softmax. For binary
    segmentation the default ``num_classes=1`` produces a single-channel map.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        dim: int = 2,
    ) -> None:
        """Initialize SegmentationHead.

        Args:
            in_channels: Channels of the decoded feature map.
            num_classes: Number of segmentation output channels/classes.
            dim: Spatial dimensions (2 or 3).
        """
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1.")
        conv_cls = get_conv_nd(dim)
        self.proj = conv_cls(in_channels, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project decoded features to segmentation logits.

        Args:
            features: Decoded feature tensor ``(B, C, *spatial)``.

        Returns:
            Logits of shape ``(B, num_classes, *spatial)``.
        """
        return self.proj(features)
