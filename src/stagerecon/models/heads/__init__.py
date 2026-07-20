"""Task heads and registry for modular U-Net architectures."""

from stagerecon.models.heads.bottleneck_reconstruction_head import (
    BottleneckReconstructionHead,
)
from stagerecon.models.heads.head_registry import (
    build_head,
    get_head,
    list_heads,
    register_head,
)
from stagerecon.models.heads.image_reconstruction_head import ImageReconstructionHead
from stagerecon.models.heads.segmentation_head import SegmentationHead

__all__ = [
    "BottleneckReconstructionHead",
    "ImageReconstructionHead",
    "SegmentationHead",
    "register_head",
    "get_head",
    "build_head",
    "list_heads",
]
