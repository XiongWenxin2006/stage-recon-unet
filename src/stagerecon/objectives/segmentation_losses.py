"""Binary segmentation loss modules operating on logits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsSegmentationLoss(nn.Module):
    """Binary cross-entropy with logits for segmentation maps."""

    def __init__(
        self,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            reduction: Passed to :func:`torch.nn.functional.binary_cross_entropy_with_logits`.
            pos_weight: Optional positive-class weight tensor.
        """
        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None  # type: ignore[assignment]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute BCE-with-logits loss.

        Args:
            pred: Predicted logits ``(B, 1, *spatial)`` or ``(B, *spatial)``.
            target: Binary targets with the same broadcastable shape (float or
                integer labels in ``{0, 1}``).

        Returns:
            Scalar (or reduced) loss tensor.
        """
        target = target.to(dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Expects **logits** by default; applies sigmoid before computing Dice.
    Set ``from_logits=False`` if inputs are already probabilities.

    Args:
        smooth: Laplace smoothing epsilon added to numerator and denominator.
        from_logits: If True (default), apply sigmoid to ``pred``.
        reduction: ``"mean"`` averages over the batch; ``"none"`` returns
            per-sample losses; ``"sum"`` sums over the batch.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss ``1 - Dice``.

        Args:
            pred: Predicted logits (or probabilities if ``from_logits=False``).
            target: Binary targets with the same spatial shape.

        Returns:
            Dice loss tensor.
        """
        probs = torch.sigmoid(pred) if self.from_logits else pred
        target = target.to(dtype=probs.dtype)

        # Flatten per batch element: (B, -1)
        probs_flat = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction!r}")


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE-with-logits and soft Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        """
        Args:
            bce_weight: Weight for the BCE term.
            dice_weight: Weight for the Dice term.
            smooth: Smoothing epsilon for Dice.
        """
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = BCEWithLogitsSegmentationLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ``bce_weight * BCE + dice_weight * Dice``.

        Args:
            pred: Predicted logits.
            target: Binary targets.

        Returns:
            Scalar loss tensor.
        """
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(
            pred, target
        )
