"""Generic weighted composite loss wrapper."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Wrap named weighted loss modules into a single ``nn.Module``.

    Each sub-loss is called as ``loss(pred, target)``. The returned value is
    the weighted sum of all contributions.

    Args:
        losses: Mapping ``name → nn.Module``.
        weights: Mapping ``name → float``. Missing entries default to ``1.0``.
    """

    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not losses:
            raise ValueError("CompositeLoss requires at least one loss module.")
        self.losses = nn.ModuleDict({str(k): v for k, v in losses.items()})
        weights = dict(weights or {})
        self.weights: dict[str, float] = {
            name: float(weights.get(name, 1.0)) for name in self.losses
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Evaluate the weighted sum of all registered losses.

        Args:
            pred: Model prediction tensor.
            target: Ground-truth tensor.

        Returns:
            Scalar loss tensor.
        """
        total: torch.Tensor | None = None
        for name, loss_fn in self.losses.items():
            value = self.weights[name] * loss_fn(pred, target)
            total = value if total is None else total + value
        assert total is not None
        return total

    def forward_dict(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return individual (weighted) loss terms plus ``total``.

        Args:
            pred: Model prediction tensor.
            target: Ground-truth tensor.

        Returns:
            Dict with per-loss weighted values and a ``total`` entry.
        """
        out: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name, loss_fn in self.losses.items():
            value = self.weights[name] * loss_fn(pred, target)
            out[name] = value
            total = value if total is None else total + value
        assert total is not None
        out["total"] = total
        return out
