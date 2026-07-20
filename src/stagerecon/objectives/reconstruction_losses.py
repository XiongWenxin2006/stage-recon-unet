"""Reconstruction loss modules (no config I/O, no optimizer / backprop logic)."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1ReconstructionLoss(nn.Module):
    """Mean absolute error (L1) between prediction and target."""

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction: Passed to :func:`torch.nn.functional.l1_loss`.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            pred: Predicted reconstruction ``(B, C, *spatial)``.
            target: Ground-truth image with the same shape as ``pred``.

        Returns:
            Scalar (or reduced) loss tensor.
        """
        return F.l1_loss(pred, target, reduction=self.reduction)


class MSEReconstructionLoss(nn.Module):
    """Mean squared error between prediction and target."""

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction: Passed to :func:`torch.nn.functional.mse_loss`.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            pred: Predicted reconstruction ``(B, C, *spatial)``.
            target: Ground-truth image with the same shape as ``pred``.

        Returns:
            Scalar (or reduced) loss tensor.
        """
        return F.mse_loss(pred, target, reduction=self.reduction)


class CompositeReconstructionLoss(nn.Module):
    """Weighted sum of named reconstruction losses.

    Args:
        losses: Mapping ``name → nn.Module`` loss modules.
        weights: Mapping ``name → float`` weights. Missing names default to
            ``1.0``. Extra weight keys are ignored.
    """

    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not losses:
            raise ValueError("CompositeReconstructionLoss requires at least one loss.")
        self.losses = nn.ModuleDict({str(k): v for k, v in losses.items()})
        weights = weights or {}
        self.weights: dict[str, float] = {
            name: float(weights.get(name, 1.0)) for name in self.losses
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted sum of reconstruction losses.

        Args:
            pred: Predicted reconstruction.
            target: Ground-truth image.

        Returns:
            Scalar loss tensor.
        """
        total: torch.Tensor | None = None
        for name, loss_fn in self.losses.items():
            value = self.weights[name] * loss_fn(pred, target)
            total = value if total is None else total + value
        assert total is not None
        return total
