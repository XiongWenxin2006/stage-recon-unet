"""Composable U-Net with interchangeable encoder / bottleneck / decoder / heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from stagerecon.models.composed.model_output import ModelOutput

VALID_MODES = (
    "bottleneck_reconstruction",
    "reconstruction",
    "segmentation",
)


class ModularUNet(nn.Module):
    """Modular U-Net supporting reconstruction and segmentation forward modes.

    Components are stored as named submodules:

    - ``encoder``
    - ``bottleneck``
    - ``decoder``
    - ``bottleneck_reconstruction_head``
    - ``reconstruction_head``
    - ``segmentation_head``
    """

    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.Module,
        decoder: nn.Module,
        bottleneck_reconstruction_head: nn.Module | None = None,
        reconstruction_head: nn.Module | None = None,
        segmentation_head: nn.Module | None = None,
        return_features: bool = False,
    ) -> None:
        """Initialize ModularUNet.

        Args:
            encoder: Encoder producing multi-scale features (high → low).
            bottleneck: Bottleneck operating on the deepest encoder feature.
            decoder: Decoder mapping bottleneck + skips to input resolution.
            bottleneck_reconstruction_head: Optional head for bottleneck recon.
            reconstruction_head: Optional head for full-path reconstruction.
            segmentation_head: Optional head for segmentation logits.
            return_features: If True, attach intermediate features to ModelOutput.
        """
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.bottleneck_reconstruction_head = bottleneck_reconstruction_head
        self.reconstruction_head = reconstruction_head
        self.segmentation_head = segmentation_head
        self.return_features = return_features

    def get_module(self, name: str) -> nn.Module:
        """Return a named submodule for checkpoint loading / inspection.

        Args:
            name: One of ``encoder``, ``bottleneck``, ``decoder``,
                ``bottleneck_reconstruction_head``, ``reconstruction_head``,
                ``segmentation_head``.

        Returns:
            The requested ``nn.Module``.

        Raises:
            KeyError: If ``name`` is unknown or the module was not provided.
        """
        allowed = {
            "encoder",
            "bottleneck",
            "decoder",
            "bottleneck_reconstruction_head",
            "reconstruction_head",
            "segmentation_head",
        }
        if name not in allowed:
            raise KeyError(f"Unknown module '{name}'. Allowed: {sorted(allowed)}")
        module = getattr(self, name)
        if module is None:
            raise KeyError(f"Module '{name}' is not set on this ModularUNet.")
        return module

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the encoder and bottleneck.

        Args:
            x: Input tensor ``(B, C, *spatial)``.

        Returns:
            Pair ``(encoder_features, bottleneck_feature)`` where
            ``encoder_features`` is high → low resolution.
        """
        encoder_features = self.encoder(x)
        bottleneck_feature = self.bottleneck(encoder_features[-1])
        return encoder_features, bottleneck_feature

    def decode(
        self,
        bottleneck_feature: torch.Tensor,
        encoder_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Decode bottleneck features with encoder skips.

        Args:
            bottleneck_feature: Output of the bottleneck module.
            encoder_features: Multi-scale encoder features (high → low).

        Returns:
            Decoded feature map at input spatial resolution.
        """
        return self.decoder(bottleneck_feature, encoder_features)

    def _maybe_features(
        self,
        encoder_features: list[torch.Tensor] | None = None,
        bottleneck_feature: torch.Tensor | None = None,
        decoded_features: torch.Tensor | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.return_features:
            return None
        feats: dict[str, Any] = {}
        if encoder_features is not None:
            feats["encoder_features"] = encoder_features
        if bottleneck_feature is not None:
            feats["bottleneck_feature"] = bottleneck_feature
        if decoded_features is not None:
            feats["decoded_features"] = decoded_features
        if extra:
            feats.update(extra)
        return feats

    def forward_bottleneck_reconstruction(self, x: torch.Tensor) -> ModelOutput:
        """Encode and reconstruct an image from the bottleneck only.

        Args:
            x: Input tensor ``(B, C, *spatial)``.

        Returns:
            ``ModelOutput`` with bottleneck-path reconstruction.
        """
        if self.bottleneck_reconstruction_head is None:
            raise RuntimeError(
                "bottleneck_reconstruction_head is not configured on this model."
            )
        encoder_features, bottleneck_feature = self.encode(x)
        prediction = self.bottleneck_reconstruction_head(bottleneck_feature)
        return ModelOutput(
            prediction=prediction,
            features=self._maybe_features(encoder_features, bottleneck_feature),
            mode="bottleneck_reconstruction",
        )

    def forward_reconstruction(self, x: torch.Tensor) -> ModelOutput:
        """Full encode → bottleneck → decode → image reconstruction.

        Args:
            x: Input tensor ``(B, C, *spatial)``.

        Returns:
            ``ModelOutput`` with reconstructed image logits/values.
        """
        if self.reconstruction_head is None:
            raise RuntimeError("reconstruction_head is not configured on this model.")
        encoder_features, bottleneck_feature = self.encode(x)
        decoded = self.decode(bottleneck_feature, encoder_features)
        prediction = self.reconstruction_head(decoded)
        return ModelOutput(
            prediction=prediction,
            features=self._maybe_features(
                encoder_features, bottleneck_feature, decoded
            ),
            mode="reconstruction",
        )

    def forward_segmentation(self, x: torch.Tensor) -> ModelOutput:
        """Full encode → bottleneck → decode → segmentation logits.

        Args:
            x: Input tensor ``(B, C, *spatial)``.

        Returns:
            ``ModelOutput`` with segmentation logits (no thresholding).
        """
        if self.segmentation_head is None:
            raise RuntimeError("segmentation_head is not configured on this model.")
        encoder_features, bottleneck_feature = self.encode(x)
        decoded = self.decode(bottleneck_feature, encoder_features)
        prediction = self.segmentation_head(decoded)
        return ModelOutput(
            prediction=prediction,
            features=self._maybe_features(
                encoder_features, bottleneck_feature, decoded
            ),
            mode="segmentation",
        )

    def forward(self, x: torch.Tensor, mode: str = "segmentation") -> ModelOutput:
        """Dispatch to a task-specific forward method.

        Args:
            x: Input tensor ``(B, C, *spatial)``.
            mode: One of ``bottleneck_reconstruction``, ``reconstruction``,
                ``segmentation``.

        Returns:
            ``ModelOutput`` for the requested mode.
        """
        mode = mode.lower()
        if mode == "bottleneck_reconstruction":
            return self.forward_bottleneck_reconstruction(x)
        if mode == "reconstruction":
            return self.forward_reconstruction(x)
        if mode == "segmentation":
            return self.forward_segmentation(x)
        raise ValueError(
            f"Unsupported mode '{mode}'. Valid modes: {', '.join(VALID_MODES)}"
        )
