"""Decoder modules and registry for modular U-Net architectures."""

from stagerecon.models.decoders.attention_decoder import AttentionDecoder
from stagerecon.models.decoders.base_decoder import BaseDecoder
from stagerecon.models.decoders.decoder_registry import (
    build_decoder,
    get_decoder,
    list_decoders,
    register_decoder,
)
from stagerecon.models.decoders.residual_decoder import ResidualDecoder
from stagerecon.models.decoders.unet_decoder import UNetDecoder

__all__ = [
    "BaseDecoder",
    "UNetDecoder",
    "AttentionDecoder",
    "ResidualDecoder",
    "register_decoder",
    "get_decoder",
    "build_decoder",
    "list_decoders",
]
